import { Logger } from '@nestjs/common';
import Anthropic from '@anthropic-ai/sdk';
import type {
  MessageParam,
  ContentBlockParam,
  ImageBlockParam,
  TextBlockParam,
  RawMessageStreamEvent,
  StopReason,
} from '@anthropic-ai/sdk/resources/messages';
import { upstreamUnavailable, validationFailed } from '../api-error';
import {
  getRequiredPositiveIntegerEnv,
  getRequiredStringEnv,
  getRequiredStringListEnv,
} from '../runtime';
import { createApiMeta } from '../response-meta';
import type {
  HealthResponse,
  ImageInputPart,
  InferRequest,
  InferenceMessage,
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
  StreamEvent,
} from './backend.interface';

/**
 * Per-model context windows for the Anthropic models this host is
 * allowed to serve. Used by `model.status` to report a usable
 * `numCtx` figure to the orchestrator's prompt-budget calculator.
 */
const ANTHROPIC_MODEL_CONTEXTS: Record<string, number> = {
  'claude-haiku-4-5-20251001': 200_000,
  'claude-haiku-4-5': 200_000,
  'claude-sonnet-4-6': 200_000,
  'claude-opus-4-7': 200_000,
};

const ANTHROPIC_DEFAULT_CONTEXT = 200_000;
const ANTHROPIC_DEFAULT_PROMPT_BUDGET_FRACTION = 0.8;
const ANTHROPIC_THINKING_MIN_BUDGET_TOKENS = 1024;

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
  'claude-haiku-4-5': 'Claude Haiku 4.5',
  'claude-sonnet-4-6': 'Claude Sonnet 4.6',
  'claude-opus-4-7': 'Claude Opus 4.7',
};

interface NormalizedAnthropicInput {
  system?: string;
  messages: MessageParam[];
}

interface AppliedOptions {
  thinking: boolean;
  publicOptions: InferenceOptions;
}

interface RuntimeState {
  reachable: boolean;
  error?: string;
}

/**
 * Hosted-API backend for Anthropic Claude models. Implements the same
 * `Backend` contract as `OllamaBackend`, so the WebSocket layer and
 * `LlmService` shell remain unchanged.
 */
export class AnthropicBackend implements Backend {
  readonly name = 'anthropic' as const;
  private readonly logger = new Logger(AnthropicBackend.name);
  private readonly apiKey = getRequiredStringEnv('ANTHROPIC_API_KEY');
  private readonly baseUrl = getRequiredStringEnv('ANTHROPIC_BASE_URL');
  private readonly availableModels = getRequiredStringListEnv(
    'ANTHROPIC_MODELS',
  );
  private readonly modelId = selectConfiguredModel(
    getRequiredStringEnv('ANTHROPIC_MODEL'),
    this.availableModels,
  );
  private readonly maxOutputTokens = getRequiredPositiveIntegerEnv(
    'ANTHROPIC_MAX_OUTPUT_TOKENS',
  );
  private readonly imageInputEnabled = true;
  private readonly thinkingEnabled = parseBooleanEnv('MODEL_THINKING');

  private readonly client = new Anthropic({
    apiKey: this.apiKey,
    baseURL: this.baseUrl,
  });

  get info(): BackendInfo {
    const friendly = MODEL_DISPLAY_NAMES[this.modelId] ?? this.modelId;
    return {
      name: this.name,
      displayName: `Anthropic — ${friendly} (cloud)`,
      modelId: this.modelId,
      location: 'cloud',
    };
  }

  async init(): Promise<void> {
    this.logger.log(
      `AnthropicBackend ready: model=${this.modelId}, baseURL=${this.baseUrl}, maxOutputTokens=${this.maxOutputTokens}, thinkingEnabled=${this.thinkingEnabled}`,
    );
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

    const input = await this.normalizeInput(request);
    const appliedOptions = this.normalizeOptions(request.options);

    emit({ type: 'started', meta });

    console.log(
      '===== ANTHROPIC CALL =====\n' +
        'model: ' + this.modelId + '\n' +
        'thinking: ' + appliedOptions.thinking + '\n' +
        'max_tokens: ' + this.maxOutputTokens + '\n' +
        'system: ' + (input.system ? input.system.slice(0, 80) + '...' : '(none)') + '\n' +
        'messages: ' + input.messages.length + '\n' +
        '=======================',
    );

    let finishReason: 'stop' | 'length' | 'error' = 'stop';

    try {
      const stream = this.client.messages.stream(
        {
          model: this.modelId,
          max_tokens: this.maxOutputTokens,
          ...(input.system ? { system: input.system } : {}),
          messages: input.messages,
          ...(appliedOptions.thinking
            ? {
                thinking: {
                  type: 'enabled' as const,
                  budget_tokens: Math.max(
                    ANTHROPIC_THINKING_MIN_BUDGET_TOKENS,
                    Math.floor(this.maxOutputTokens / 2),
                  ),
                },
              }
            : {}),
        },
        signal ? { signal } : undefined,
      );

      try {
        for await (const event of stream as AsyncIterable<RawMessageStreamEvent>) {
          if (signal?.aborted) {
            break;
          }
          this.handleStreamEvent(event, meta, emit);
        }

        const final = await stream.finalMessage();
        finishReason = mapStopReason(final.stop_reason);
      } catch (error) {
        if (signal?.aborted) {
          finishReason = 'error';
        } else {
          throw error;
        }
      }

      emit({
        type: 'done',
        meta,
        data: {
          finishReason,
          appliedOptions: appliedOptions.publicOptions as Record<string, unknown>,
        },
      });
    } catch (error) {
      throw this.normalizeUpstreamError(error);
    }
  }

  private handleStreamEvent(
    event: RawMessageStreamEvent,
    meta: ReturnType<typeof createApiMeta>,
    emit: (event: StreamEvent) => void,
  ): void {
    if (event.type !== 'content_block_delta') {
      return;
    }

    const delta = event.delta as { type?: string; text?: string; thinking?: string };

    if (delta.type === 'text_delta' && typeof delta.text === 'string') {
      emit({ type: 'chunk', meta, data: { text: delta.text } });
      return;
    }

    if (delta.type === 'thinking_delta' && typeof delta.thinking === 'string') {
      emit({ type: 'thinking', meta, data: { text: delta.thinking } });
      return;
    }
  }

  async health(correlationId: string): Promise<HealthResponse> {
    const state = await this.getRuntimeState();

    return {
      meta: createApiMeta(correlationId),
      data: {
        status: state.reachable ? 'ok' : 'unavailable',
        ready: state.reachable,
      },
    };
  }

  async modelStatus(correlationId: string): Promise<ModelStatusResponse> {
    const state = await this.getRuntimeState();
    const numCtx =
      ANTHROPIC_MODEL_CONTEXTS[this.modelId] ?? ANTHROPIC_DEFAULT_CONTEXT;
    const promptBudgetTokens = Math.floor(
      numCtx * ANTHROPIC_DEFAULT_PROMPT_BUDGET_FRACTION,
    );

    return {
      meta: createApiMeta(correlationId),
      data: {
        modelId: this.modelId,
        availableModels: this.availableModels,
        ready: state.reachable,
        loaded: state.reachable,
        keepAlive: 'n/a',
        capabilities: this.capabilities,
        capacity: {
          activeRequests: 0,
          modelSlots: Number.POSITIVE_INFINITY,
          queueDepth: 0,
        },
        contextWindow: {
          numCtx,
          promptBudgetTokens,
          responseReserveTokens: numCtx - promptBudgetTokens,
          promptBudgetFraction: ANTHROPIC_DEFAULT_PROMPT_BUDGET_FRACTION,
        },
        runtime: {
          backend: 'anthropic',
          baseUrl: this.baseUrl,
          reachable: state.reachable,
          thinkingEnabled: this.thinkingEnabled,
          maxOutputTokens: this.maxOutputTokens,
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
    return {
      meta: createApiMeta(correlationId),
      data: {
        accepted: true,
        modelId: this.modelId,
        status: 'unsupported',
      },
    };
  }

  async unload(
    correlationId: string,
    request: ModelLifecycleRequest,
  ): Promise<ModelLifecycleResponse> {
    this.assertRequestContext(request?.requestContext);
    return {
      meta: createApiMeta(correlationId),
      data: {
        accepted: true,
        modelId: this.modelId,
        status: 'unsupported',
      },
    };
  }

  private async normalizeInput(
    request: InferRequest,
  ): Promise<NormalizedAnthropicInput> {
    if (!isRecord(request?.input)) {
      throw validationFailed('input must be an object.');
    }

    const parts = Array.isArray(request.input.parts)
      ? request.input.parts
      : undefined;
    const rawMessages = Array.isArray(request.input.messages)
      ? request.input.messages
      : undefined;

    if (!parts && !rawMessages) {
      throw validationFailed(
        'input.parts or input.messages must be a non-empty array.',
      );
    }

    let system: string | undefined;
    const messages: MessageParam[] = [];

    if (rawMessages) {
      const normalized = this.normalizeMessages(rawMessages);
      const systemTexts: string[] = [];
      for (const message of normalized) {
        if (message.role === 'system') {
          systemTexts.push(message.content);
        } else {
          messages.push({ role: message.role, content: message.content });
        }
      }
      if (systemTexts.length > 0) {
        system = systemTexts.join('\n\n');
      }
    }

    const textParts: string[] = [];
    const imageBlocks: ImageBlockParam[] = [];

    if (parts) {
      if (parts.length === 0 && !rawMessages) {
        throw validationFailed('input.parts must contain at least one part.');
      }

      for (const [index, part] of parts.entries()) {
        if (!isRecord(part)) {
          throw validationFailed(`input.parts[${index}] must be an object.`);
        }

        if (part.type === 'text') {
          textParts.push(
            this.normalizeTextPart(part as unknown as InputPart, index),
          );
          continue;
        }

        if (part.type === 'image') {
          imageBlocks.push(
            await this.normalizeImagePart(
              part as unknown as ImageInputPart,
              index,
            ),
          );
          continue;
        }

        throw validationFailed(
          `input.parts[${index}].type must be text or image.`,
        );
      }
    }

    const text = textParts.join('\n\n').trim();

    if (rawMessages) {
      if (text || imageBlocks.length > 0) {
        const content = this.buildContentBlocks(text, imageBlocks);
        messages.push({ role: 'user', content });
      }
      if (messages.length === 0) {
        throw validationFailed(
          'input.messages must contain at least one non-system message.',
        );
      }
      return { system, messages };
    }

    if (!text && imageBlocks.length === 0) {
      throw validationFailed(
        'Inference input must include text, images, or both.',
      );
    }

    messages.push({
      role: 'user',
      content: this.buildContentBlocks(text, imageBlocks),
    });

    return { system, messages };
  }

  private buildContentBlocks(
    text: string,
    images: ImageBlockParam[],
  ): string | ContentBlockParam[] {
    if (images.length === 0) {
      return text;
    }

    const blocks: ContentBlockParam[] = [...images];
    if (text) {
      const textBlock: TextBlockParam = { type: 'text', text };
      blocks.push(textBlock);
    }
    return blocks;
  }

  private normalizeMessages(value: unknown): InferenceMessage[] {
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
        throw validationFailed(
          `input.messages[${index}].role must be system, user, or assistant.`,
        );
      }

      if (typeof message.content !== 'string' || !message.content.trim()) {
        throw validationFailed(
          `input.messages[${index}].content must be a non-empty string.`,
        );
      }

      return { role, content: message.content };
    });
  }

  private normalizeTextPart(part: InputPart, index: number): string {
    if (part.type !== 'text' || typeof part.text !== 'string') {
      throw validationFailed(`input.parts[${index}].text must be a string.`);
    }

    return part.text;
  }

  private async normalizeImagePart(
    part: ImageInputPart,
    index: number,
  ): Promise<ImageBlockParam> {
    if (!this.imageInputEnabled) {
      throw validationFailed('This model host does not accept image input.');
    }

    const hasBase64 =
      typeof part.imageBase64 === 'string' &&
      part.imageBase64.trim().length > 0;
    const hasUrl =
      typeof part.imageUrl === 'string' && part.imageUrl.trim().length > 0;

    if (hasBase64 === hasUrl) {
      throw validationFailed(
        `input.parts[${index}] must include exactly one of imageBase64 or imageUrl.`,
      );
    }

    if (hasUrl) {
      return {
        type: 'image',
        source: {
          type: 'url',
          url: part.imageUrl as string,
        },
      };
    }

    const data = this.normalizeImageBase64(part.imageBase64 as string, index);
    const mediaType = normalizeMediaType(part.mimeType);

    return {
      type: 'image',
      source: {
        type: 'base64',
        media_type: mediaType,
        data,
      },
    };
  }

  private normalizeImageBase64(value: string, index: number): string {
    const normalized = stripDataUrlPrefix(value).replace(/\s+/g, '');

    if (!normalized) {
      throw validationFailed(`input.parts[${index}].imageBase64 is empty.`);
    }

    if (!/^[A-Za-z0-9+/]+={0,2}$/.test(normalized)) {
      throw validationFailed(
        `input.parts[${index}].imageBase64 is not valid base64.`,
      );
    }

    Buffer.from(normalized, 'base64');

    return normalized;
  }

  private normalizeOptions(
    options: InferRequest['options'],
  ): AppliedOptions {
    if (options !== undefined && !isRecord(options)) {
      throw validationFailed('options must be an object.');
    }

    const thinkingRequested =
      typeof options?.thinking === 'boolean'
        ? options.thinking
        : this.thinkingEnabled;
    const thinking = this.thinkingEnabled && thinkingRequested;

    const publicOptions: InferenceOptions = {
      responseFormat: 'text',
      thinking,
    };

    return { thinking, publicOptions };
  }

  private assertRequestContext(context: RequestContext | undefined): void {
    if (!isRecord(context)) {
      throw validationFailed('requestContext is required.');
    }

    if (
      typeof context.callerService !== 'string' ||
      !context.callerService.trim()
    ) {
      throw validationFailed('requestContext.callerService is required.');
    }

    if (
      context.priority !== undefined &&
      (typeof context.priority !== 'number' ||
        !Number.isFinite(context.priority))
    ) {
      throw validationFailed(
        'requestContext.priority must be a finite number when provided.',
      );
    }

    if (
      typeof context.requestedAt !== 'string' ||
      !context.requestedAt.endsWith('Z') ||
      Number.isNaN(Date.parse(context.requestedAt))
    ) {
      throw validationFailed(
        'requestContext.requestedAt must be an ISO 8601 UTC timestamp.',
      );
    }
  }

  private async getRuntimeState(): Promise<RuntimeState> {
    return {
      reachable: this.apiKey.length > 0,
    };
  }

  private normalizeUpstreamError(error: unknown): Error {
    if (error instanceof Error && error.name === 'AbortError') {
      return upstreamUnavailable('Anthropic request was aborted.', {
        modelId: this.modelId,
      });
    }

    if (isApiErrorException(error)) {
      return error;
    }

    if (error instanceof Anthropic.APIError) {
      return upstreamUnavailable('Anthropic API call failed.', {
        modelId: this.modelId,
        status: error.status,
        detail: error.message,
      });
    }

    return upstreamUnavailable('Anthropic request failed.', {
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
}

function selectConfiguredModel(
  modelId: string,
  availableModels: string[],
): string {
  if (availableModels.includes(modelId)) {
    return modelId;
  }

  throw new Error(
    `ANTHROPIC_MODEL must be one of ANTHROPIC_MODELS. Received "${modelId}". Allowed: ${availableModels.join(', ')}`,
  );
}

function parseBooleanEnv(name: string): boolean {
  const raw = process.env[name];
  if (raw === undefined) return false;
  const normalized = raw.trim().toLowerCase();
  return ['1', 'true', 'yes', 'on'].includes(normalized);
}

function normalizeMediaType(
  value: string | undefined,
): 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp' {
  if (!value) return 'image/jpeg';
  const normalized = value.trim().toLowerCase();
  if (
    normalized === 'image/jpeg' ||
    normalized === 'image/png' ||
    normalized === 'image/gif' ||
    normalized === 'image/webp'
  ) {
    return normalized;
  }
  if (normalized === 'image/jpg') return 'image/jpeg';
  throw validationFailed(
    `Unsupported image mimeType "${value}". Anthropic accepts image/jpeg, image/png, image/gif, image/webp.`,
  );
}

function mapStopReason(value: StopReason | null): 'stop' | 'length' | 'error' {
  if (value === 'max_tokens') return 'length';
  if (value === null) return 'error';
  return 'stop';
}

function stripDataUrlPrefix(value: string): string {
  const trimmed = value.trim();
  const dataUrlMatch =
    /^data:image\/[-+.a-zA-Z0-9]+;base64,(?<data>.*)$/s.exec(trimmed);
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
