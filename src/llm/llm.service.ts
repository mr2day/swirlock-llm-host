import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { Ollama, type ChatResponse, type Message, type Options } from 'ollama';
import { limitExceeded, modelOverloaded, upstreamUnavailable, validationFailed } from './api-error';
import {
  formatKeepAlive,
  getBooleanEnv,
  getIntegerEnv,
  getStringEnv,
  parseKeepAlive,
} from './runtime';
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
  ModelLimits,
  ModelStatusResponse,
  RequestContext,
} from './types';

const DEFAULT_MODEL = 'qwen3.5:9b';
const DEFAULT_OLLAMA_HOST = 'http://127.0.0.1:11434';

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
  priority: RequestContext['priority'];
  sequence: number;
  resolve: (release: () => void) => void;
  reject: (error: Error) => void;
  signal?: AbortSignal;
}

export type StreamEvent =
  | { type: 'accepted'; meta: ReturnType<typeof createApiMeta> }
  | { type: 'queued'; meta: ReturnType<typeof createApiMeta>; data: { position: number } }
  | { type: 'started'; meta: ReturnType<typeof createApiMeta> }
  | { type: 'thinking'; meta: ReturnType<typeof createApiMeta>; data: { text: string } }
  | { type: 'chunk'; meta: ReturnType<typeof createApiMeta>; data: { text: string } }
  | {
      type: 'done';
      meta: ReturnType<typeof createApiMeta>;
      data: {
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
  private readonly maxTextChars = getIntegerEnv('MAX_TEXT_CHARS', 20000);
  private readonly maxImages = getIntegerEnv('MAX_IMAGES', getIntegerEnv('MAX_IMAGE_FILES', 8));
  private readonly maxImageBytes = getIntegerEnv('MAX_IMAGE_BYTES', 20 * 1024 * 1024);
  private readonly maxConcurrentRequests = getIntegerEnv('MAX_CONCURRENT_REQUESTS', 1);
  private readonly maxQueueSize = getIntegerEnv('MAX_QUEUE_SIZE', 8);
  private activeRequests = 0;
  private queueSequence = 0;
  private readonly waitQueue: QueueEntry[] = [];

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

    const release = await this.acquireModelSlot(request.requestContext.priority);

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
      release();
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
    emit({ type: 'accepted', meta });

    const release = await this.acquireModelSlot(
      request.requestContext.priority,
      signal,
      (position) => emit({ type: 'queued', meta, data: { position } }),
    );

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

      emit({ type: 'started', meta });

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
          finishReason: mapFinishReason(finalChunk?.done_reason),
          appliedOptions: appliedOptions.publicOptions,
        },
      });
    } catch (error) {
      throw this.normalizeUpstreamError(error);
    } finally {
      release();
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
        limits: this.limits,
        capacity: {
          activeRequests: this.activeRequests,
          maxConcurrentRequests: this.maxConcurrentRequests,
          queueDepth: this.waitQueue.length,
          maxQueueSize: this.maxQueueSize,
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

    if (images.length > this.maxImages) {
      throw limitExceeded('Image count exceeds host limit.', {
        maxImages: this.maxImages,
        receivedImages: images.length,
      });
    }

    const text = textParts.join('\n\n').trim();

    if (!text && images.length === 0) {
      throw validationFailed('Inference input must include text, images, or both.');
    }

    if (text.length > this.maxTextChars) {
      throw limitExceeded('Text input exceeds host limit.', {
        maxTextChars: this.maxTextChars,
        receivedTextChars: text.length,
      });
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

    if (part.mimeType !== undefined && !part.mimeType.startsWith('image/')) {
      throw validationFailed(`input.parts[${index}].mimeType must start with image/.`);
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

    const imageBytes = Buffer.from(normalized, 'base64').byteLength;

    if (imageBytes > this.maxImageBytes) {
      throw limitExceeded('Image payload exceeds host limit.', {
        maxImageBytes: this.maxImageBytes,
        receivedImageBytes: imageBytes,
        partIndex: index,
      });
    }

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

    const contentLength = response.headers.get('content-length');
    if (contentLength && Number(contentLength) > this.maxImageBytes) {
      throw limitExceeded('Image URL content exceeds host limit.', {
        maxImageBytes: this.maxImageBytes,
        receivedImageBytes: Number(contentLength),
        partIndex: index,
      });
    }

    const buffer = Buffer.from(await response.arrayBuffer());
    if (buffer.byteLength > this.maxImageBytes) {
      throw limitExceeded('Image URL content exceeds host limit.', {
        maxImageBytes: this.maxImageBytes,
        receivedImageBytes: buffer.byteLength,
        partIndex: index,
      });
    }

    return buffer.toString('base64');
  }

  private normalizeOptions(options: InferRequest['options']): AppliedOptions {
    if (options !== undefined && !isRecord(options)) {
      throw validationFailed('options must be an object.');
    }

    const responseFormat = normalizeResponseFormat(options?.responseFormat);
    const thinking =
      typeof options?.thinking === 'boolean' ? options.thinking : this.thinkingEnabled;

    const publicOptions: InferenceOptions = {
      responseFormat,
      thinking,
    };

    const ollamaOptions: Partial<Options> = {};

    if (options?.temperature !== undefined) {
      publicOptions.temperature = clampNumber(options.temperature, 0, 2, 'options.temperature');
      ollamaOptions.temperature = publicOptions.temperature;
    }

    if (options?.topP !== undefined) {
      publicOptions.topP = clampNumber(options.topP, 0, 1, 'options.topP');
      ollamaOptions.top_p = publicOptions.topP;
    }

    if (options?.stopSequences !== undefined) {
      if (
        !Array.isArray(options.stopSequences) ||
        !options.stopSequences.every((item) => typeof item === 'string')
      ) {
        throw validationFailed('options.stopSequences must be an array of strings.');
      }
      publicOptions.stopSequences = options.stopSequences.slice(0, 16);
      ollamaOptions.stop = publicOptions.stopSequences;
    }

    if (options?.seed !== undefined) {
      publicOptions.seed = clampInteger(options.seed, -2147483648, 2147483647, 'options.seed');
      ollamaOptions.seed = publicOptions.seed;
    }

    return {
      responseFormat,
      thinking,
      publicOptions,
      ollamaOptions,
    };
  }

  private acquireModelSlot(
    priority: RequestContext['priority'],
    signal?: AbortSignal,
    onQueued?: (position: number) => void,
  ): Promise<() => void> {
    if (this.activeRequests < this.maxConcurrentRequests) {
      this.activeRequests += 1;
      return Promise.resolve(() => this.releaseModelSlot());
    }

    if (this.waitQueue.length >= this.maxQueueSize || priority === 'maintenance') {
      throw modelOverloaded('Model host queue is full or unavailable for this priority.', {
        activeRequests: this.activeRequests,
        maxConcurrentRequests: this.maxConcurrentRequests,
        queueDepth: this.waitQueue.length,
        maxQueueSize: this.maxQueueSize,
      });
    }

    return new Promise((resolve, reject) => {
      const entry: QueueEntry = {
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
      onQueued?.(this.queuePosition(entry));
    });
  }

  private releaseModelSlot(): void {
    this.activeRequests = Math.max(0, this.activeRequests - 1);
    this.startNextQueuedRequest();
  }

  private startNextQueuedRequest(): void {
    if (this.activeRequests >= this.maxConcurrentRequests || this.waitQueue.length === 0) {
      return;
    }

    const nextIndex = this.nextQueueIndex();
    const [entry] = this.waitQueue.splice(nextIndex, 1);

    if (entry.signal?.aborted) {
      entry.reject(validationFailed('Queued model request was aborted before it started.'));
      this.startNextQueuedRequest();
      return;
    }

    this.activeRequests += 1;
    entry.resolve(() => this.releaseModelSlot());
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

  private get limits(): ModelLimits {
    return {
      maxTextChars: this.maxTextChars,
      maxImages: this.imageInputEnabled ? this.maxImages : 0,
      maxImageBytes: this.imageInputEnabled ? this.maxImageBytes : 0,
      maxConcurrentRequests: this.maxConcurrentRequests,
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

function clampInteger(value: unknown, min: number, max: number, fieldName: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw validationFailed(`${fieldName} must be an integer.`);
  }

  return Math.min(Math.max(value, min), max);
}

function clampNumber(value: unknown, min: number, max: number, fieldName: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw validationFailed(`${fieldName} must be a number.`);
  }

  return Math.min(Math.max(value, min), max);
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
