import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { Ollama, type ChatResponse, type Message, type Options } from 'ollama';
import { upstreamUnavailable, validationFailed } from './api-error';
import { formatKeepAlive, getBooleanEnv, getStringEnv, parseKeepAlive } from './runtime';
import { createApiMeta } from './response-meta';
import type {
  HealthResponse,
  ImageInputPart,
  InferRequest,
  InferResponse,
  InferenceOptions,
  InputPart,
  ModelCapabilities,
  ModelLifecycleRequest,
  ModelLifecycleResponse,
  ModelStatusResponse,
  RequestContext,
} from './types';

const DEFAULT_MODEL = 'qwen3.5:9b';
const DEFAULT_OLLAMA_HOST = 'http://127.0.0.1:11434';
const MODEL_SLOTS = 1;
const RECENT_DURATION_SAMPLE_SIZE = 20;

interface NormalizedInput {
  text: string;
  images: string[];
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
  registrationNumber: string;
  priority: RequestContext['priority'];
  sequence: number;
  resolve: (slot: AcquiredModelSlot) => void;
  reject: (error: Error) => void;
  signal?: AbortSignal;
}

interface AcquiredModelSlot {
  registrationNumber: string;
  release: () => void;
}

interface QueueEstimate {
  registrationNumber: string;
  position: number;
  estimatedWaitMs?: number;
  estimatedStartAt?: string;
  tryAgainAt?: string;
}

export type StreamEvent =
  | {
      type: 'accepted';
      meta: ReturnType<typeof createApiMeta>;
      data: { registrationNumber: string };
    }
  | { type: 'queued'; meta: ReturnType<typeof createApiMeta>; data: QueueEstimate }
  | {
      type: 'started';
      meta: ReturnType<typeof createApiMeta>;
      data: { registrationNumber: string };
    }
  | { type: 'thinking'; meta: ReturnType<typeof createApiMeta>; data: { text: string } }
  | { type: 'chunk'; meta: ReturnType<typeof createApiMeta>; data: { text: string } }
  | {
      type: 'done';
      meta: ReturnType<typeof createApiMeta>;
      data: {
        registrationNumber: string;
        finishReason: 'stop' | 'length' | 'error';
        appliedOptions: InferenceOptions;
      };
    };

@Injectable()
export class LlmService implements OnModuleInit {
  private readonly logger = new Logger(LlmService.name);
  private readonly modelId = getStringEnv('OLLAMA_MODEL', DEFAULT_MODEL);
  private readonly ollamaHost = getStringEnv('OLLAMA_HOST', DEFAULT_OLLAMA_HOST);
  private readonly keepAlive = parseKeepAlive(getStringEnv('OLLAMA_KEEP_ALIVE', '-1'));
  private readonly preloadModel = getBooleanEnv('PRELOAD_MODEL', true);
  private readonly imageInputEnabled = getBooleanEnv('MODEL_IMAGE_INPUT', true);
  private readonly thinkingEnabled = getBooleanEnv('MODEL_THINKING', false);
  private activeRequests = 0;
  private queueSequence = 0;
  private registrationSequence = 0;
  private currentRegistrationNumber: string | undefined;
  private readonly waitQueue: QueueEntry[] = [];
  private readonly recentRequestDurationsMs: number[] = [];

  private readonly ollama = new Ollama({
    host: this.ollamaHost,
  });

  async onModuleInit(): Promise<void> {
    if (!this.preloadModel) {
      return;
    }

    try {
      await this.preloadHostedModel();
      this.logger.log(`Preloaded ${this.modelId} with keep_alive=${this.keepAliveText}`);
    } catch (error) {
      this.logger.warn(
        `Could not preload ${this.modelId}. Requests will fail until Ollama can load it. ${getErrorMessage(error)}`,
      );
    }
  }

  async infer(correlationId: string, request: InferRequest): Promise<InferResponse> {
    this.assertRequestContext(request?.requestContext);

    const slot = await this.acquireModelSlot(request.requestContext.priority);

    try {
      const input = await this.normalizeInput(request);
      const appliedOptions = this.normalizeOptions(request.options);
      const messages: Message[] = [
        {
          role: 'user',
          content: input.text,
          ...(input.images.length > 0 ? { images: input.images } : {}),
        },
      ];

      const response = await this.ollama.chat({
        model: this.modelId,
        messages,
        stream: false,
        keep_alive: this.keepAlive,
        think: appliedOptions.thinking,
        options: appliedOptions.ollamaOptions,
        ...(appliedOptions.responseFormat === 'json' ? { format: 'json' } : {}),
      });

      return {
        meta: createApiMeta(correlationId),
        data: {
          modelId: response.model,
          registrationNumber: slot.registrationNumber,
          output: {
            text: response.message?.content ?? '',
          },
          finishReason: mapFinishReason(response.done_reason),
          generatedAt: new Date().toISOString(),
          appliedOptions: appliedOptions.publicOptions,
        },
      };
    } catch (error) {
      throw this.normalizeUpstreamError(error);
    } finally {
      slot.release();
    }
  }

  async streamInfer(
    correlationId: string,
    request: InferRequest,
    emit: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    this.assertRequestContext(request?.requestContext);

    const meta = createApiMeta(correlationId);

    const slot = await this.acquireModelSlot(request.requestContext.priority, signal, (estimate) =>
      emit({ type: 'queued', meta, data: estimate }),
    );

    emit({
      type: 'accepted',
      meta,
      data: { registrationNumber: slot.registrationNumber },
    });

    try {
      const input = await this.normalizeInput(request);
      const appliedOptions = this.normalizeOptions(request.options);
      const messages: Message[] = [
        {
          role: 'user',
          content: input.text,
          ...(input.images.length > 0 ? { images: input.images } : {}),
        },
      ];

      emit({ type: 'started', meta, data: { registrationNumber: slot.registrationNumber } });

      const stream = await this.ollama.chat({
        model: this.modelId,
        messages,
        stream: true,
        keep_alive: this.keepAlive,
        think: appliedOptions.thinking,
        options: appliedOptions.ollamaOptions,
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
          registrationNumber: slot.registrationNumber,
          finishReason: mapFinishReason(finalChunk?.done_reason),
          appliedOptions: appliedOptions.publicOptions,
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

    return {
      meta: createApiMeta(correlationId),
      data: {
        modelId: this.modelId,
        ready: state.modelAvailable,
        loaded: state.loaded,
        keepAlive: this.keepAliveText,
        capabilities: this.capabilities,
        capacity: {
          activeRequests: this.activeRequests,
          modelSlots: MODEL_SLOTS,
          queueDepth: this.waitQueue.length,
          currentRegistrationNumber: this.currentRegistrationNumber,
          averageRequestDurationMs: this.averageRequestDurationMs,
        },
        runtime: {
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
    });
  }

  private async normalizeInput(request: InferRequest): Promise<NormalizedInput> {
    if (!isRecord(request?.input) || !Array.isArray(request.input.parts)) {
      throw validationFailed('input.parts must be a non-empty array.');
    }

    if (request.input.parts.length === 0) {
      throw validationFailed('input.parts must contain at least one part.');
    }

    const textParts: string[] = [];
    const images: string[] = [];

    for (const [index, part] of request.input.parts.entries()) {
      if (!isRecord(part)) {
        throw validationFailed(`input.parts[${index}] must be an object.`);
      }

      if (part.type === 'text') {
        textParts.push(this.normalizeTextPart(part as InputPart, index));
        continue;
      }

      if (part.type === 'image') {
        images.push(await this.normalizeImagePart(part as ImageInputPart, index));
        continue;
      }

      throw validationFailed(`input.parts[${index}].type must be text or image.`);
    }

    const text = textParts.join('\n\n').trim();

    if (!text && images.length === 0) {
      throw validationFailed('Inference input must include text, images, or both.');
    }

    return { text, images };
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
    const thinking =
      typeof options?.thinking === 'boolean' ? options.thinking : this.thinkingEnabled;
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
    onQueued?: (estimate: QueueEstimate) => void,
  ): Promise<AcquiredModelSlot> {
    const registrationNumber = this.nextRegistrationNumber();

    if (this.activeRequests < MODEL_SLOTS) {
      return Promise.resolve(this.startModelSlot(registrationNumber));
    }

    return new Promise((resolve, reject) => {
      const entry: QueueEntry = {
        registrationNumber,
        priority,
        sequence: this.queueSequence++,
        resolve,
        reject,
        signal,
      };

      const abortQueued = () => {
        const index = this.waitQueue.indexOf(entry);
        if (index >= 0) {
          this.waitQueue.splice(index, 1);
          reject(validationFailed('Queued model request was aborted before it started.'));
        }
      };

      signal?.addEventListener('abort', abortQueued, { once: true });

      this.waitQueue.push(entry);
      onQueued?.(this.queueEstimate(entry));
    });
  }

  private startModelSlot(registrationNumber: string): AcquiredModelSlot {
    const startedAt = Date.now();
    this.activeRequests += 1;
    this.currentRegistrationNumber = registrationNumber;

    return {
      registrationNumber,
      release: () => this.releaseModelSlot(registrationNumber, startedAt),
    };
  }

  private releaseModelSlot(registrationNumber: string, startedAt: number): void {
    this.recordRequestDuration(Date.now() - startedAt);

    if (this.currentRegistrationNumber === registrationNumber) {
      this.currentRegistrationNumber = undefined;
    }

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

    entry.resolve(this.startModelSlot(entry.registrationNumber));
  }

  private nextQueueIndex(): number {
    let bestIndex = 0;

    for (let index = 1; index < this.waitQueue.length; index += 1) {
      const candidate = this.waitQueue[index];
      const best = this.waitQueue[bestIndex];
      if (priorityRank(candidate.priority) < priorityRank(best.priority)) {
        bestIndex = index;
      }
    }

    return bestIndex;
  }

  private queueEstimate(entry: QueueEntry): QueueEstimate {
    const position = this.queuePosition(entry);
    const averageRequestDurationMs = this.averageRequestDurationMs;
    const estimatedWaitMs =
      averageRequestDurationMs !== undefined ? position * averageRequestDurationMs : undefined;
    const estimatedStartAt =
      estimatedWaitMs !== undefined
        ? new Date(Date.now() + estimatedWaitMs).toISOString()
        : undefined;

    return {
      registrationNumber: entry.registrationNumber,
      position,
      estimatedWaitMs,
      estimatedStartAt,
      tryAgainAt: estimatedStartAt,
    };
  }

  private queuePosition(entry: QueueEntry): number {
    return (
      this.waitQueue
        .slice()
        .sort(
          (a, b) => priorityRank(a.priority) - priorityRank(b.priority) || a.sequence - b.sequence,
        )
        .indexOf(entry) + 1
    );
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

  private nextRegistrationNumber(): string {
    this.registrationSequence += 1;
    return `U-${String(this.registrationSequence).padStart(6, '0')}`;
  }

  private assertRequestContext(context: RequestContext | undefined): void {
    if (!isRecord(context)) {
      throw validationFailed('requestContext is required.');
    }

    if (typeof context.callerService !== 'string' || !context.callerService.trim()) {
      throw validationFailed('requestContext.callerService is required.');
    }

    if (!['interactive', 'background', 'maintenance'].includes(String(context.priority))) {
      throw validationFailed(
        'requestContext.priority must be interactive, background, or maintenance.',
      );
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

function priorityRank(priority: RequestContext['priority']): number {
  if (priority === 'interactive') {
    return 0;
  }
  if (priority === 'background') {
    return 1;
  }
  return 2;
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
