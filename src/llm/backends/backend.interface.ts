import type {
  HealthResponse,
  InferRequest,
  ModelLifecycleRequest,
  ModelLifecycleResponse,
  ModelStatusResponse,
} from '../types';
import type { createApiMeta } from '../response-meta';

/**
 * Stream events emitted by `Backend.streamInfer` during a single
 * inference request. The shape is wire-stable across backends — the
 * WebSocket layer maps these directly to envelope frames without
 * knowing or caring which backend is serving the request.
 */
export type StreamEvent =
  | { type: 'accepted'; meta: ReturnType<typeof createApiMeta> }
  | {
      type: 'queued';
      meta: ReturnType<typeof createApiMeta>;
      data: QueueWaitInfo;
    }
  | { type: 'started'; meta: ReturnType<typeof createApiMeta> }
  | {
      type: 'thinking';
      meta: ReturnType<typeof createApiMeta>;
      data: { text: string };
    }
  | {
      type: 'chunk';
      meta: ReturnType<typeof createApiMeta>;
      data: { text: string };
    }
  | {
      type: 'done';
      meta: ReturnType<typeof createApiMeta>;
      data: {
        finishReason: 'stop' | 'length' | 'error';
        appliedOptions: Record<string, unknown>;
      };
    };

export interface QueueWaitInfo {
  position: number;
  requestsAhead: number;
  queueDepth: number;
  defaultPriority: boolean;
  priority?: number;
  averageRequestDurationMs?: number;
  estimatedWaitMs?: number;
  estimatedStartAt?: string;
}

/**
 * Static (per-host) info about a configured backend, returned in
 * `backends.list` so the UI can render its model picker without
 * paying for a full `model.status` round-trip per backend.
 */
export interface BackendInfo {
  name: string;            // 'ollama' | 'anthropic'
  displayName: string;     // 'Ollama — ministral-3:14b (local)'
  modelId: string;         // current model the backend serves
  location: 'local' | 'cloud';
}

/**
 * The contract every backend implementation must satisfy. `LlmService`
 * instantiates every configured backend at construction time and
 * forwards each request to whichever one the caller picks (or to the
 * default).
 */
export interface Backend {
  /** Stable backend identifier used in the `backend` request field. */
  readonly name: 'ollama' | 'anthropic';

  /** Static info for the UI's model picker. */
  readonly info: BackendInfo;

  /**
   * Optional async initialization invoked from
   * `LlmService.onModuleInit`. Use this to compute hardware-derived
   * settings, preload models, ping the upstream API, etc. Backends
   * that need no setup may omit this method.
   */
  init?(): Promise<void>;

  streamInfer(
    correlationId: string,
    request: InferRequest,
    emit: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void>;

  health(correlationId: string): Promise<HealthResponse>;

  modelStatus(correlationId: string): Promise<ModelStatusResponse>;

  preload(
    correlationId: string,
    request: ModelLifecycleRequest,
  ): Promise<ModelLifecycleResponse>;

  unload(
    correlationId: string,
    request: ModelLifecycleRequest,
  ): Promise<ModelLifecycleResponse>;
}
