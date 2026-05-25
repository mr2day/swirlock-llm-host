import { Logger } from '@nestjs/common';
import { Ollama, type ChatResponse, type Message, type Options } from 'ollama';
import { upstreamUnavailable, validationFailed } from '../api-error';
import {
  computeTargetNumCtx,
  extractArchitectureFromModelInfo,
  type ContextWindowComputation,
  type HardwareSpec,
} from '../context-window';
import {
  formatKeepAlive,
  getRequiredBooleanEnv,
  getRequiredFractionEnv,
  getRequiredPositiveIntegerEnv,
  getRequiredPositiveNumberEnv,
  getRequiredStringEnv,
  getRequiredStringListEnv,
  parseKeepAlive,
} from '../runtime';
import { createApiMeta } from '../response-meta';
import type {
  HealthResponse,
  ImageInputPart,
  InferRequest,
  InferenceOptions,
  InputPart,
  ModelCapabilities,
  ModelLifecycleRequest,
  ModelLifecycleResponse,
  ModelStatusResponse,
  RequestContext,
} from '../types';
import type {
  Backend,
  BackendInfo,
  QueueWaitInfo,
  StreamEvent,
} from './backend.interface';

const MODEL_SLOTS = 1;
const RECENT_DURATION_SAMPLE_SIZE = 20;
const LOWEST_PRIORITY = Number.NEGATIVE_INFINITY;

interface NormalizedInput {
  messages: Message[];
}

interface AppliedOptions {
  responseFormat: 'text' | 'json';
  thinking: boolean;
  publicOptions: InferenceOptions;
  ollamaOptions: Partial<Options>;
}

interface RuntimeState {
  ollamaReachable: boolean;
  modelAvailable: boolean;
  loaded: boolean;
  version?: string;
  error?: string;
}

interface QueueEntry {
  sortPriority: number;
  requestedPriority?: number;
  sequence: number;
  resolve: (slot: AcquiredModelSlot) => void;
  reject: (error: Error) => void;
  signal?: AbortSignal;
  notifyQueued?: (info: QueueWaitInfo) => void;
}

interface NormalizedPriority {
  sortPriority: number;
  requestedPriority?: number;
}

interface AcquiredModelSlot {
  release: () => void;
}

/**
 * Ollama-backed implementation. Behaviour preserved byte-for-byte
 * from the pre-multi-backend `LlmService` so that existing Ollama
 * deployments (the friends-testing instance on main) keep working
 * with no observable change after the refactor.
 */
export class OllamaBackend implements Backend {
  readonly name = 'ollama' as const;
  private readonly logger = new Logger(OllamaBackend.name);
  private readonly availableModels = getRequiredStringListEnv('OLLAMA_MODELS');
  private readonly modelId = selectConfiguredModel(
    getRequiredStringEnv('OLLAMA_MODEL'),
    this.availableModels,
  );
  private readonly ollamaHost = getRequiredStringEnv('OLLAMA_HOST');
  private readonly keepAlive = parseKeepAlive(getRequiredStringEnv('OLLAMA_KEEP_ALIVE'));
  private readonly preloadModel = getRequiredBooleanEnv('PRELOAD_MODEL');
  private readonly imageInputEnabled = getRequiredBooleanEnv('MODEL_IMAGE_INPUT');
  private readonly thinkingEnabled = getRequiredBooleanEnv('MODEL_THINKING');
  private readonly hardware: HardwareSpec = {
    totalVramBytes: getRequiredPositiveIntegerEnv('HARDWARE_TOTAL_VRAM_BYTES'),
    overheadReserveBytes: getRequiredPositiveIntegerEnv(
      'HARDWARE_OVERHEAD_RESERVE_BYTES',
    ),
    kvCacheElementBytes: getRequiredPositiveNumberEnv(
      'HARDWARE_KV_CACHE_ELEMENT_BYTES',
    ),
  };
  private readonly promptBudgetFraction = getRequiredFractionEnv(
    'PROMPT_BUDGET_FRACTION',
  );
  private readonly defaultNumCtx = getRequiredPositiveIntegerEnv(
    'DEFAULT_NUM_CTX',
  );
  private contextWindow: ContextWindowComputation | null = null;
  private activeRequests = 0;
  private queueSequence = 0;
  private readonly waitQueue: QueueEntry[] = [];
  private readonly recentRequestDurationsMs: number[] = [];

  private readonly ollama = new Ollama({
    host: this.ollamaHost,
  });

  get info(): BackendInfo {
    return {
      name: this.name,
      displayName: `Ollama — ${this.modelId} (local)`,
      modelId: this.modelId,
      location: 'local',
    };
  }

  async init(): Promise<void> {
    await this.tryResolveContextWindowAndPreload(1);
  }

  private async tryResolveContextWindowAndPreload(
    attempt: number,
  ): Promise<void> {
    let cw: ContextWindowComputation;
    try {
      cw = await this.computeContextWindow();
    } catch (error) {
      this.logger.warn(
        `Context-window computation crashed on attempt ${attempt}; using default num_ctx=${this.defaultNumCtx}. ${getErrorMessage(error)}`,
      );
      cw = {
        targetNumCtx: this.defaultNumCtx,
        rawMaxNumCtx: 0,
        availableForKvBytes: 0,
        kvPerTokenBytes: 0,
        fellBackToDefault: true,
        fallbackReason: getErrorMessage(error),
      };
    }

    this.contextWindow = cw;
    this.logger.log(
      `Context window resolved (attempt ${attempt}): num_ctx=${cw.targetNumCtx}, ` +
        `prompt budget=${this.promptBudgetTokens}, ` +
        `kvPerToken=${Math.round(cw.kvPerTokenBytes)} B, ` +
        `availableForKv=${formatBytes(cw.availableForKvBytes)}` +
        (cw.fellBackToDefault
          ? ` (fell back to default: ${cw.fallbackReason})`
          : ''),
    );

    if (this.preloadModel) {
      try {
        await this.preloadHostedModel();
        this.logger.log(
          `Preloaded ${this.modelId} with keep_alive=${this.keepAliveText}, num_ctx=${this.numCtx} (attempt ${attempt})`,
        );
      } catch (error) {
        this.logger.warn(
          `Could not preload ${this.modelId} on attempt ${attempt}. Requests will fail until Ollama can load it. ${getErrorMessage(error)}`,
        );
      }
    }

    if (cw.fellBackToDefault && isTransientFallback(cw.fallbackReason)) {
      const backoffMs = Math.min(2000 * 2 ** Math.min(attempt - 1, 5), 60_000);
      this.logger.warn(
        `Context-window fell back to default because ollama was unreachable; ` +
          `retrying in ${backoffMs}ms (next attempt ${attempt + 1}).`,
      );
      setTimeout(() => {
        void this.tryResolveContextWindowAndPreload(attempt + 1);
      }, backoffMs);
    }
  }

  private async computeContextWindow(): Promise<ContextWindowComputation> {
    let modelInfo: Record<string, unknown> | undefined;
    let modelSizeBytes = 0;
    try {
      const show = (await this.ollama.show({
        model: this.modelId,
      })) as unknown as {
        model_info?: Map<string, unknown> | Record<string, unknown>;
      };
      modelInfo = toPlainObject(show.model_info);
    } catch (error) {
      this.logger.warn(
        `ollama.show failed; using default num_ctx=${this.defaultNumCtx}. ${getErrorMessage(error)}`,
      );
      return {
        targetNumCtx: this.defaultNumCtx,
        rawMaxNumCtx: 0,
        availableForKvBytes: 0,
        kvPerTokenBytes: 0,
        fellBackToDefault: true,
        fallbackReason: `ollama.show failed: ${getErrorMessage(error)}`,
      };
    }

    try {
      const list = (await this.ollama.list()) as unknown as {
        models?: Array<{
          name?: string;
          model?: string;
          size?: number;
        }>;
      };
      const match = (list.models ?? []).find(
        (m) => m.name === this.modelId || m.model === this.modelId,
      );
      if (match && typeof match.size === 'number' && match.size > 0) {
        modelSizeBytes = match.size;
      }
    } catch (error) {
      this.logger.warn(
        `ollama.list failed when resolving model size; using default num_ctx=${this.defaultNumCtx}. ${getErrorMessage(error)}`,
      );
      return {
        targetNumCtx: this.defaultNumCtx,
        rawMaxNumCtx: 0,
        availableForKvBytes: 0,
        kvPerTokenBytes: 0,
        fellBackToDefault: true,
        fallbackReason: `ollama.list failed: ${getErrorMessage(error)}`,
      };
    }

    if (modelSizeBytes <= 0) {
      return {
        targetNumCtx: this.defaultNumCtx,
        rawMaxNumCtx: 0,
        availableForKvBytes: 0,
        kvPerTokenBytes: 0,
        fellBackToDefault: true,
        fallbackReason: `ollama.list did not include "${this.modelId}" or reported size 0; cannot compute KV memory budget.`,
      };
    }

    const architecture = extractArchitectureFromModelInfo(modelInfo);
    if (!architecture) {
      return {
        targetNumCtx: this.defaultNumCtx,
        rawMaxNumCtx: 0,
        availableForKvBytes:
          this.hardware.totalVramBytes -
          modelSizeBytes -
          this.hardware.overheadReserveBytes,
        kvPerTokenBytes: 0,
        fellBackToDefault: true,
        fallbackReason:
          'Could not extract architecture (block_count/embedding_length/head_count) from ollama.show model_info.',
      };
    }

    return computeTargetNumCtx({
      hardware: this.hardware,
      modelSizeBytes,
      architecture,
      defaultNumCtx: this.defaultNumCtx,
    });
  }

  private get numCtx(): number {
    return this.contextWindow?.targetNumCtx ?? this.defaultNumCtx;
  }

  private get promptBudgetTokens(): number {
    return Math.floor(this.numCtx * this.promptBudgetFraction);
  }

  private get responseReserveTokens(): number {
    return this.numCtx - this.promptBudgetTokens;
  }

  async streamInfer(
    correlationId: string,
    request: InferRequest,
    emit: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    this.assertRequestContext(request?.requestContext);

    const meta = createApiMeta(correlationId);
    emit({ type: 'accepted', meta });

    const slot = await this.acquireModelSlot(request.requestContext.priority, signal, (waitInfo) =>
      emit({ type: 'queued', meta, data: waitInfo }),
    );

    try {
      const input = await this.normalizeInput(request);
      const appliedOptions = this.normalizeOptions(request.options);

      emit({ type: 'started', meta });

      console.log(
        '===== OLLAMA CALL =====\n' +
          'model: ' + this.modelId + '\n' +
          'think: ' + appliedOptions.thinking + '\n' +
          'options: ' + JSON.stringify(appliedOptions.ollamaOptions) + '\n' +
          'format: ' + (appliedOptions.responseFormat === 'json' ? 'json' : '(none)') + '\n' +
          '=======================',
      );

      const stream = await this.ollama.chat({
        model: this.modelId,
        messages: input.messages,
        stream: true,
        keep_alive: this.keepAlive,
        think: appliedOptions.thinking,
        options: { num_ctx: this.numCtx, ...appliedOptions.ollamaOptions },
        ...(appliedOptions.responseFormat === 'json' ? { format: 'json' } : {}),
      });

      const abortStream = () => stream.abort();
      signal?.addEventListener('abort', abortStream, { once: true });

      let finalChunk: ChatResponse | undefined;

      try {
        for await (const chunk of stream) {
          if (signal?.aborted) {
            break;
          }

          finalChunk = chunk;

          if (chunk.message?.thinking) {
            emit({ type: 'thinking', meta, data: { text: chunk.message.thinking } });
          }

          if (chunk.message?.content) {
            emit({ type: 'chunk', meta, data: { text: chunk.message.content } });
          }
        }
      } finally {
        signal?.removeEventListener('abort', abortStream);
      }

      emit({
        type: 'done',
        meta,
        data: {
          finishReason: mapFinishReason(finalChunk?.done_reason),
          appliedOptions: appliedOptions.publicOptions as Record<string, unknown>,
        },
      });
    } catch (error) {
      throw this.normalizeUpstreamError(error);
    } finally {
      slot.release();
    }
  }

  async health(correlationId: string): Promise<HealthResponse> {
    const state = await this.getRuntimeState();

    return {
      meta: createApiMeta(correlationId),
      data: {
        status: state.modelAvailable ? 'ok' : state.ollamaReachable ? 'degraded' : 'unavailable',
        ready: state.modelAvailable,
      },
    };
  }

  async modelStatus(correlationId: string): Promise<ModelStatusResponse> {
    const state = await this.getRuntimeState();
    const cw = this.contextWindow;

    return {
      meta: createApiMeta(correlationId),
      data: {
        modelId: this.modelId,
        availableModels: this.availableModels,
        ready: state.modelAvailable,
        loaded: state.loaded,
        keepAlive: this.keepAliveText,
        capabilities: this.capabilities,
        capacity: {
          activeRequests: this.activeRequests,
          modelSlots: MODEL_SLOTS,
          queueDepth: this.waitQueue.length,
          averageRequestDurationMs: this.averageRequestDurationMs,
        },
        contextWindow: {
          numCtx: this.numCtx,
          promptBudgetTokens: this.promptBudgetTokens,
          responseReserveTokens: this.responseReserveTokens,
          promptBudgetFraction: this.promptBudgetFraction,
          ...(cw
            ? {
                rawMaxNumCtx: cw.rawMaxNumCtx,
                kvPerTokenBytes: cw.kvPerTokenBytes,
                availableForKvBytes: cw.availableForKvBytes,
                fellBackToDefault: cw.fellBackToDefault,
                ...(cw.fallbackReason
                  ? { fallbackReason: cw.fallbackReason }
                  : {}),
              }
            : {}),
        },
        runtime: {
          backend: 'ollama',
          ollamaHost: this.ollamaHost,
          ollamaReachable: state.ollamaReachable,
          version: state.version,
          thinkingEnabled: this.thinkingEnabled,
          error: state.error,
        },
      },
    };
  }

  async preload(
    correlationId: string,
    request: ModelLifecycleRequest,
  ): Promise<ModelLifecycleResponse> {
    this.assertRequestContext(request?.requestContext);

    try {
      await this.preloadHostedModel();
      return {
        meta: createApiMeta(correlationId),
        data: {
          accepted: true,
          modelId: this.modelId,
          status: 'loaded',
        },
      };
    } catch (error) {
      throw this.normalizeUpstreamError(error);
    }
  }

  async unload(
    correlationId: string,
    request: ModelLifecycleRequest,
  ): Promise<ModelLifecycleResponse> {
    this.assertRequestContext(request?.requestContext);

    try {
      await this.ollama.generate({
        model: this.modelId,
        prompt: '',
        stream: false,
        keep_alive: 0,
      });

      return {
        meta: createApiMeta(correlationId),
        data: {
          accepted: true,
          modelId: this.modelId,
          status: 'unloaded',
        },
      };
    } catch (error) {
      throw this.normalizeUpstreamError(error);
    }
  }

  private async preloadHostedModel(): Promise<void> {
    await this.ollama.generate({
      model: this.modelId,
      prompt: '',
      stream: false,
      keep_alive: this.keepAlive,
      options: { num_ctx: this.numCtx },
    });
  }

  private async normalizeInput(request: InferRequest): Promise<NormalizedInput> {
    if (!isRecord(request?.input)) {
      throw validationFailed('input must be an object.');
    }

    const parts = Array.isArray(request.input.parts) ? request.input.parts : undefined;
    const rawMessages = Array.isArray(request.input.messages) ? request.input.messages : undefined;
    if (!parts && !rawMessages) {
      throw validationFailed('input.parts or input.messages must be a non-empty array.');
    }

    const messages: Message[] = rawMessages ? this.normalizeMessages(rawMessages) : [];
    const textParts: string[] = [];
    const images: string[] = [];

    if (parts) {
      if (parts.length === 0 && !rawMessages) {
        throw validationFailed('input.parts must contain at least one part.');
      }

      for (const [index, part] of parts.entries()) {
        if (!isRecord(part)) {
          throw validationFailed(`input.parts[${index}] must be an object.`);
        }

        if (part.type === 'text') {
          textParts.push(this.normalizeTextPart(part as unknown as InputPart, index));
          continue;
        }

        if (part.type === 'image') {
          images.push(await this.normalizeImagePart(part as unknown as ImageInputPart, index));
          continue;
        }

        throw validationFailed(`input.parts[${index}].type must be text or image.`);
      }
    }

    const text = textParts.join('\n\n').trim();

    if (rawMessages) {
      if (text) {
        messages.push({ role: 'user', content: text });
      }
      if (images.length > 0) {
        this.attachImagesToLastUserMessage(messages, images);
      }
      if (messages.length === 0) {
        throw validationFailed('input.messages must contain at least one message.');
      }
      return { messages };
    }

    if (!text && images.length === 0) {
      throw validationFailed('Inference input must include text, images, or both.');
    }

    return {
      messages: [
        {
          role: 'user',
          content: text,
          ...(images.length > 0 ? { images } : {}),
        },
      ],
    };
  }

  private normalizeMessages(value: unknown): Message[] {
    if (!Array.isArray(value)) {
      throw validationFailed('input.messages must be an array.');
    }

    if (value.length === 0) return [];

    return value.map((message, index) => {
      if (!isRecord(message)) {
        throw validationFailed(`input.messages[${index}] must be an object.`);
      }

      const role = message.role;
      if (role !== 'system' && role !== 'user' && role !== 'assistant') {
        throw validationFailed(`input.messages[${index}].role must be system, user, or assistant.`);
      }

      if (typeof message.content !== 'string' || !message.content.trim()) {
        throw validationFailed(`input.messages[${index}].content must be a non-empty string.`);
      }

      return { role, content: message.content };
    });
  }

  private attachImagesToLastUserMessage(messages: Message[], images: string[]): void {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.role !== 'user') continue;
      (message as Message & { images?: string[] }).images = images;
      return;
    }

    throw validationFailed('Image parts require at least one user message.');
  }

  private normalizeTextPart(part: InputPart, index: number): string {
    if (part.type !== 'text' || typeof part.text !== 'string') {
      throw validationFailed(`input.parts[${index}].text must be a string.`);
    }

    return part.text;
  }

  private async normalizeImagePart(part: ImageInputPart, index: number): Promise<string> {
    if (!this.imageInputEnabled) {
      throw validationFailed('This model host does not accept image input.');
    }

    const hasBase64 = typeof part.imageBase64 === 'string' && part.imageBase64.trim().length > 0;
    const hasUrl = typeof part.imageUrl === 'string' && part.imageUrl.trim().length > 0;

    if (hasBase64 === hasUrl) {
      throw validationFailed(
        `input.parts[${index}] must include exactly one of imageBase64 or imageUrl.`,
      );
    }

    return hasBase64
      ? this.normalizeImageBase64(part.imageBase64 as string, index)
      : this.fetchImageUrl(part.imageUrl as string, index);
  }

  private normalizeImageBase64(value: string, index: number): string {
    const normalized = stripDataUrlPrefix(value).replace(/\s+/g, '');

    if (!normalized) {
      throw validationFailed(`input.parts[${index}].imageBase64 is empty.`);
    }

    if (!/^[A-Za-z0-9+/]+={0,2}$/.test(normalized)) {
      throw validationFailed(`input.parts[${index}].imageBase64 is not valid base64.`);
    }

    Buffer.from(normalized, 'base64');

    return normalized;
  }

  private async fetchImageUrl(value: string, index: number): Promise<string> {
    let url: URL;
    try {
      url = new URL(value);
    } catch {
      throw validationFailed(`input.parts[${index}].imageUrl must be a valid URL.`);
    }

    if (!['http:', 'https:'].includes(url.protocol)) {
      throw validationFailed(`input.parts[${index}].imageUrl must use http or https.`);
    }

    const response = await fetch(url, {
      headers: { Accept: 'image/*' },
    });

    if (!response.ok) {
      throw upstreamUnavailable('Could not fetch imageUrl.', {
        partIndex: index,
        status: response.status,
      });
    }

    const contentType = response.headers.get('content-type');
    if (contentType && !contentType.toLowerCase().startsWith('image/')) {
      throw validationFailed(`input.parts[${index}].imageUrl did not return an image.`);
    }

    const buffer = Buffer.from(await response.arrayBuffer());
    return buffer.toString('base64');
  }

  private normalizeOptions(options: InferRequest['options']): AppliedOptions {
    if (options !== undefined && !isRecord(options)) {
      throw validationFailed('options must be an object.');
    }

    const responseFormat = normalizeResponseFormat(options?.responseFormat);
    const thinkingRequested =
      typeof options?.thinking === 'boolean' ? options.thinking : this.thinkingEnabled;
    const thinking = this.thinkingEnabled && thinkingRequested;
    const ollamaOptions = normalizeOllamaOptions(options?.ollama);

    const publicOptions: InferenceOptions = {
      responseFormat,
      thinking,
      ...(ollamaOptions !== undefined ? { ollama: ollamaOptions } : {}),
    };

    return {
      responseFormat,
      thinking,
      publicOptions,
      ollamaOptions: (ollamaOptions ?? {}) as Partial<Options>,
    };
  }

  private acquireModelSlot(
    priority: RequestContext['priority'],
    signal?: AbortSignal,
    onQueued?: (waitInfo: QueueWaitInfo) => void,
  ): Promise<AcquiredModelSlot> {
    if (this.activeRequests < MODEL_SLOTS) {
      return Promise.resolve(this.startModelSlot());
    }

    const normalizedPriority = normalizePriority(priority);

    return new Promise((resolve, reject) => {
      const entry: QueueEntry = {
        sortPriority: normalizedPriority.sortPriority,
        requestedPriority: normalizedPriority.requestedPriority,
        sequence: this.queueSequence++,
        resolve,
        reject,
        signal,
        notifyQueued: onQueued,
      };

      const abortQueued = () => {
        const index = this.waitQueue.indexOf(entry);
        if (index >= 0) {
          this.waitQueue.splice(index, 1);
          this.emitQueueUpdates();
          reject(validationFailed('Queued model request was aborted before it started.'));
        }
      };

      signal?.addEventListener('abort', abortQueued, { once: true });

      this.waitQueue.push(entry);
      this.emitQueueUpdates();
    });
  }

  private startModelSlot(): AcquiredModelSlot {
    const startedAt = Date.now();
    this.activeRequests += 1;

    return {
      release: () => this.releaseModelSlot(startedAt),
    };
  }

  private releaseModelSlot(startedAt: number): void {
    this.recordRequestDuration(Date.now() - startedAt);
    this.activeRequests = Math.max(0, this.activeRequests - 1);
    this.startNextQueuedRequest();
  }

  private startNextQueuedRequest(): void {
    if (this.activeRequests >= MODEL_SLOTS || this.waitQueue.length === 0) {
      return;
    }

    const nextIndex = this.nextQueueIndex();
    const [entry] = this.waitQueue.splice(nextIndex, 1);

    if (entry.signal?.aborted) {
      entry.reject(validationFailed('Queued model request was aborted before it started.'));
      this.startNextQueuedRequest();
      return;
    }

    const slot = this.startModelSlot();
    this.emitQueueUpdates();
    entry.resolve(slot);
  }

  private nextQueueIndex(): number {
    let bestIndex = 0;

    for (let index = 1; index < this.waitQueue.length; index += 1) {
      const candidate = this.waitQueue[index];
      const best = this.waitQueue[bestIndex];
      if (compareQueueEntries(candidate, best) < 0) {
        bestIndex = index;
      }
    }

    return bestIndex;
  }

  private emitQueueUpdates(): void {
    for (const entry of this.waitQueue) {
      entry.notifyQueued?.(this.queueWaitInfo(entry));
    }
  }

  private queueWaitInfo(entry: QueueEntry): QueueWaitInfo {
    const position = this.queuePosition(entry);
    const requestsAhead = this.activeRequests + position - 1;
    const averageRequestDurationMs = this.averageRequestDurationMs;
    const estimatedWaitMs =
      averageRequestDurationMs !== undefined ? requestsAhead * averageRequestDurationMs : undefined;
    const estimatedStartAt =
      estimatedWaitMs !== undefined
        ? new Date(Date.now() + estimatedWaitMs).toISOString()
        : undefined;

    return {
      position,
      requestsAhead,
      queueDepth: this.waitQueue.length,
      defaultPriority: entry.requestedPriority === undefined,
      ...(entry.requestedPriority !== undefined ? { priority: entry.requestedPriority } : {}),
      averageRequestDurationMs,
      estimatedWaitMs,
      estimatedStartAt,
    };
  }

  private queuePosition(entry: QueueEntry): number {
    return this.waitQueue.slice().sort(compareQueueEntries).indexOf(entry) + 1;
  }

  private recordRequestDuration(durationMs: number): void {
    this.recentRequestDurationsMs.push(durationMs);

    if (this.recentRequestDurationsMs.length > RECENT_DURATION_SAMPLE_SIZE) {
      this.recentRequestDurationsMs.shift();
    }
  }

  private get averageRequestDurationMs(): number | undefined {
    if (this.recentRequestDurationsMs.length === 0) {
      return undefined;
    }

    const total = this.recentRequestDurationsMs.reduce((sum, value) => sum + value, 0);
    return Math.round(total / this.recentRequestDurationsMs.length);
  }

  private assertRequestContext(context: RequestContext | undefined): void {
    if (!isRecord(context)) {
      throw validationFailed('requestContext is required.');
    }

    if (typeof context.callerService !== 'string' || !context.callerService.trim()) {
      throw validationFailed('requestContext.callerService is required.');
    }

    if (
      context.priority !== undefined &&
      (typeof context.priority !== 'number' || !Number.isFinite(context.priority))
    ) {
      throw validationFailed('requestContext.priority must be a finite number when provided.');
    }

    if (
      typeof context.requestedAt !== 'string' ||
      !context.requestedAt.endsWith('Z') ||
      Number.isNaN(Date.parse(context.requestedAt))
    ) {
      throw validationFailed('requestContext.requestedAt must be an ISO 8601 UTC timestamp.');
    }
  }

  private async getRuntimeState(): Promise<RuntimeState> {
    const [versionResult, showResult, psResult] = await Promise.allSettled([
      this.ollama.version(),
      this.ollama.show({ model: this.modelId }),
      this.ollama.ps(),
    ]);

    const loaded =
      psResult.status === 'fulfilled' &&
      psResult.value.models.some(
        (model) => model.name === this.modelId || model.model === this.modelId,
      );

    return {
      ollamaReachable: versionResult.status === 'fulfilled',
      modelAvailable: showResult.status === 'fulfilled',
      loaded,
      version: versionResult.status === 'fulfilled' ? versionResult.value.version : undefined,
      error:
        versionResult.status === 'rejected'
          ? getErrorMessage(versionResult.reason)
          : showResult.status === 'rejected'
            ? getErrorMessage(showResult.reason)
            : undefined,
    };
  }

  private normalizeUpstreamError(error: unknown): Error {
    if (error instanceof Error && error.name === 'AbortError') {
      return upstreamUnavailable('Ollama request was aborted.', {
        modelId: this.modelId,
      });
    }

    if (isApiErrorException(error)) {
      return error;
    }

    return upstreamUnavailable('Ollama request failed.', {
      modelId: this.modelId,
      detail: getErrorMessage(error),
    });
  }

  private get capabilities(): ModelCapabilities {
    return {
      textInput: true,
      imageInput: this.imageInputEnabled,
      textOutput: true,
      imageOutput: false,
    };
  }

  private get keepAliveText(): string {
    return formatKeepAlive(this.keepAlive);
  }
}

function normalizeResponseFormat(value: unknown): 'text' | 'json' {
  if (value === undefined) {
    return 'text';
  }

  if (value === 'text' || value === 'json') {
    return value;
  }

  throw validationFailed('options.responseFormat must be text or json.');
}

function selectConfiguredModel(modelId: string, availableModels: string[]): string {
  if (availableModels.includes(modelId)) {
    return modelId;
  }

  throw new Error(
    `OLLAMA_MODEL must be one of OLLAMA_MODELS. Received "${modelId}". Allowed: ${availableModels.join(', ')}`,
  );
}

function normalizePriority(value: RequestContext['priority']): NormalizedPriority {
  if (value === undefined) {
    return { sortPriority: LOWEST_PRIORITY };
  }

  return { sortPriority: value, requestedPriority: value };
}

function compareQueueEntries(a: QueueEntry, b: QueueEntry): number {
  if (a.sortPriority > b.sortPriority) {
    return -1;
  }

  if (a.sortPriority < b.sortPriority) {
    return 1;
  }

  return a.sequence - b.sequence;
}

function normalizeOllamaOptions(value: unknown): Record<string, unknown> | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (!isRecord(value)) {
    throw validationFailed('options.ollama must be an object.');
  }

  return value;
}

function mapFinishReason(value: string | undefined): 'stop' | 'length' | 'error' {
  if (value === 'length') {
    return 'length';
  }
  if (value === 'error') {
    return 'error';
  }
  return 'stop';
}

function stripDataUrlPrefix(value: string): string {
  const trimmed = value.trim();
  const dataUrlMatch = /^data:image\/[-+.a-zA-Z0-9]+;base64,(?<data>.*)$/s.exec(trimmed);
  return dataUrlMatch?.groups?.data ?? trimmed;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isApiErrorException(value: unknown): value is Error {
  return (
    value instanceof Error &&
    typeof (value as { getStatus?: unknown }).getStatus === 'function' &&
    typeof (value as { getResponse?: unknown }).getResponse === 'function'
  );
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

function isTransientFallback(reason: string | undefined): boolean {
  if (typeof reason !== 'string') return false;
  return (
    reason.startsWith('ollama.show failed') ||
    reason.startsWith('ollama.list failed') ||
    reason.includes('did not include')
  );
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return String(bytes);
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) return `${mb.toFixed(2)} MB`;
  return `${bytes} B`;
}

function toPlainObject(
  value: Map<string, unknown> | Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  if (!value) return undefined;
  if (value instanceof Map) return Object.fromEntries(value);
  return value;
}
