export type Priority = 'interactive' | 'background' | 'maintenance';
export type FinishReason = 'stop' | 'length' | 'error';
export type ResponseFormat = 'text' | 'json';
export type ModelLifecycleStatus = 'loading' | 'loaded' | 'unloading' | 'unloaded' | 'unsupported';

export interface ApiMeta {
  requestId: string;
  correlationId: string;
  apiVersion: 'v2';
  servedAt: string;
}

export interface ApiEnvelope<TData> {
  meta: ApiMeta;
  data: TData;
}

export interface RequestContext {
  callerService: string;
  priority: Priority;
  requestedAt: string;
  timeoutMs?: number;
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

export interface InferenceInput {
  parts: InputPart[];
}

export interface InferenceOptions {
  temperature?: number;
  topP?: number;
  maxOutputTokens?: number;
  stopSequences?: string[];
  seed?: number;
  responseFormat?: ResponseFormat;
  thinking?: boolean;
}

export interface InferRequest {
  requestContext: RequestContext;
  input: InferenceInput;
  options?: InferenceOptions;
}

export interface Usage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

export interface InferData {
  modelId: string;
  output: {
    text: string;
  };
  finishReason: FinishReason;
  generatedAt: string;
  usage?: Usage;
  appliedOptions?: InferenceOptions;
}

export type InferResponse = ApiEnvelope<InferData>;

export interface ModelCapabilities {
  textInput: boolean;
  imageInput: boolean;
  textOutput: boolean;
  imageOutput: boolean;
}

export interface ModelLimits {
  maxTextChars: number;
  maxImages: number;
  maxImageBytes: number;
  maxOutputTokens: number;
  maxContextTokens?: number;
  maxConcurrentRequests: number;
  requestTimeoutMs: number;
}

export interface ModelCapacity {
  activeRequests: number;
  maxConcurrentRequests: number;
  queueDepth?: number;
  maxQueueSize?: number;
}

export interface HealthData {
  status: 'ok' | 'degraded' | 'unavailable';
  ready: boolean;
}

export type HealthResponse = ApiEnvelope<HealthData>;

export interface ModelStatusData {
  modelId: string;
  ready: boolean;
  loaded: boolean;
  keepAlive: string;
  capabilities: ModelCapabilities;
  limits: ModelLimits;
  capacity: ModelCapacity;
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
