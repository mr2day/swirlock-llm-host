export type FinishReason = 'stop' | 'length' | 'error';
export type ResponseFormat = 'text' | 'json';
export type ModelLifecycleStatus = 'loading' | 'loaded' | 'unloading' | 'unloaded' | 'unsupported';

export interface ApiMeta {
  requestId: string;
  correlationId: string;
  apiVersion: 'v4';
  servedAt: string;
}

export interface ApiEnvelope<TData> {
  meta: ApiMeta;
  data: TData;
}

export interface RequestContext {
  callerService: string;
  priority?: number;
  requestedAt: string;
  debug?: boolean;
}

export interface TextInputPart {
  type: 'text';
  text: string;
}

export interface ImageInputPart {
  type: 'image';
  imageBase64?: string;
  imageUrl?: string;
  mimeType?: string;
}

export type InputPart = TextInputPart | ImageInputPart;

export interface InferenceMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface InferenceInput {
  /**
   * Legacy flat input. When `messages` is omitted, text parts are joined into
   * one user message and image parts are attached to it.
   */
  parts?: InputPart[];
  /**
   * Chat-shaped input for callers that need to preserve system/context/user
   * boundaries. This maps directly to Ollama chat messages.
   */
  messages?: InferenceMessage[];
}

export interface InferenceOptions {
  responseFormat?: ResponseFormat;
  thinking?: boolean;
  ollama?: Record<string, unknown>;
}

export interface InferRequest {
  requestContext: RequestContext;
  input: InferenceInput;
  options?: InferenceOptions;
  /**
   * Optional backend selector. When omitted, the host routes to its
   * configured default (the `BACKEND` env var). When present, must
   * match a backend the host has been configured to serve. Backends
   * not configured for this host instance reject with
   * `validation_failed`.
   *
   * Recognised values:
   *   'ollama'    — local Ollama
   *   'anthropic' — Anthropic API
   */
  backend?: 'ollama' | 'anthropic';
}

export interface ModelCapabilities {
  textInput: boolean;
  imageInput: boolean;
  textOutput: boolean;
  imageOutput: boolean;
}

export interface ModelCapacity {
  activeRequests: number;
  modelSlots: number;
  queueDepth: number;
  averageRequestDurationMs?: number;
}

export interface HealthData {
  status: 'ok' | 'degraded' | 'unavailable';
  ready: boolean;
}

export type HealthResponse = ApiEnvelope<HealthData>;

export interface ModelContextWindow {
  /** The num_ctx Ollama is asked to load the model with. */
  numCtx: number;
  /** Tokens the orchestrator can fill in the prompt. */
  promptBudgetTokens: number;
  /** Tokens reserved for the model's response. */
  responseReserveTokens: number;
  /** Fraction of numCtx allocated to the prompt (the rest is the response). */
  promptBudgetFraction: number;
  /** Pre-rounding output of the equation, for diagnostics. */
  rawMaxNumCtx?: number;
  /** KV cache cost per token in bytes, for diagnostics. */
  kvPerTokenBytes?: number;
  /** Bytes available for the KV cache after model + overhead. */
  availableForKvBytes?: number;
  /** True if any input was missing and we fell back to DEFAULT_NUM_CTX. */
  fellBackToDefault?: boolean;
  /** Human-readable fallback reason, present only when fellBackToDefault. */
  fallbackReason?: string;
}

export interface ModelStatusData {
  modelId: string;
  availableModels: string[];
  ready: boolean;
  loaded: boolean;
  keepAlive: string;
  capabilities: ModelCapabilities;
  capacity: ModelCapacity;
  contextWindow: ModelContextWindow;
  runtime?: Record<string, unknown>;
}

export type ModelStatusResponse = ApiEnvelope<ModelStatusData>;

export interface ModelLifecycleRequest {
  requestContext: RequestContext;
}

export interface ModelLifecycleData {
  accepted: boolean;
  modelId: string;
  status?: ModelLifecycleStatus;
}

export type ModelLifecycleResponse = ApiEnvelope<ModelLifecycleData>;
