import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { validationFailed } from './api-error';
import { AnthropicBackend } from './backends/anthropic-backend';
import type {
  Backend,
  BackendInfo,
  StreamEvent,
} from './backends/backend.interface';
import { OllamaBackend } from './backends/ollama-backend';
import { createApiMeta } from './response-meta';
import type {
  HealthResponse,
  InferRequest,
  ModelLifecycleRequest,
  ModelLifecycleResponse,
  ModelStatusResponse,
} from './types';

export type { StreamEvent } from './backends/backend.interface';

type BackendName = 'ollama' | 'anthropic';

export interface BackendsListResponse {
  meta: ReturnType<typeof createApiMeta>;
  data: {
    defaultBackend: BackendName;
    backends: BackendInfo[];
  };
}

/**
 * The public surface of the LLM host. Instantiates every configured
 * backend at construction time and dispatches each request to the
 * one the caller picks (or to the env-configured default).
 *
 * The default backend is always whatever the `BACKEND` env var says
 * (`ollama` if unset). Other backends are instantiated only when
 * their required env vars are present — Ollama always (its env vars
 * are required), Anthropic only when `ANTHROPIC_API_KEY` is set.
 * That keeps friends-on-Ollama deployments untouched after the
 * refactor: no key, no Anthropic backend, no behaviour change.
 */
@Injectable()
export class LlmService implements OnModuleInit {
  private readonly logger = new Logger(LlmService.name);
  private readonly defaultBackendName: BackendName;
  private readonly backends: Partial<Record<BackendName, Backend>> = {};

  constructor() {
    this.defaultBackendName = resolveBackendName(
      (process.env.BACKEND ?? 'ollama').trim(),
    );

    // Ollama is the historical default; its env vars are required by
    // host.config.cjs, so it always instantiates cleanly on main.
    try {
      this.backends['ollama'] = new OllamaBackend();
    } catch (error) {
      this.logger.warn(
        `OllamaBackend unavailable: ${getErrorMessage(error)}`,
      );
    }

    // Anthropic instantiates only when ANTHROPIC_API_KEY is set, so
    // hosts without a key keep their pre-refactor behaviour byte-for-
    // byte (Ollama only, exposed to UI as the sole backend choice).
    if ((process.env.ANTHROPIC_API_KEY ?? '').trim().length > 0) {
      try {
        this.backends['anthropic'] = new AnthropicBackend();
      } catch (error) {
        this.logger.warn(
          `AnthropicBackend unavailable: ${getErrorMessage(error)}`,
        );
      }
    }

    if (!this.backends[this.defaultBackendName]) {
      throw new Error(
        `Default BACKEND='${this.defaultBackendName}' is not available. ` +
          `Configured backends: ${Object.keys(this.backends).join(', ') || '(none)'}`,
      );
    }

    this.logger.log(
      `LLM host backends configured: ${Object.keys(this.backends).join(', ')} ` +
        `(default: ${this.defaultBackendName})`,
    );
  }

  async onModuleInit(): Promise<void> {
    for (const backend of Object.values(this.backends)) {
      if (backend?.init) {
        try {
          await backend.init();
        } catch (error) {
          this.logger.warn(
            `Backend ${backend.name} init failed: ${getErrorMessage(error)}`,
          );
        }
      }
    }
  }

  streamInfer(
    correlationId: string,
    request: InferRequest,
    emit: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const backend = this.resolveBackend(request?.backend);
    return backend.streamInfer(correlationId, request, emit, signal);
  }

  health(correlationId: string, backendName?: string): Promise<HealthResponse> {
    const backend = this.resolveBackend(backendName);
    return backend.health(correlationId);
  }

  modelStatus(
    correlationId: string,
    backendName?: string,
  ): Promise<ModelStatusResponse> {
    const backend = this.resolveBackend(backendName);
    return backend.modelStatus(correlationId);
  }

  preload(
    correlationId: string,
    request: ModelLifecycleRequest,
    backendName?: string,
  ): Promise<ModelLifecycleResponse> {
    const backend = this.resolveBackend(backendName);
    return backend.preload(correlationId, request);
  }

  unload(
    correlationId: string,
    request: ModelLifecycleRequest,
    backendName?: string,
  ): Promise<ModelLifecycleResponse> {
    const backend = this.resolveBackend(backendName);
    return backend.unload(correlationId, request);
  }

  /**
   * Lists every backend this host has been configured to serve. Used
   * by the UI's model picker to render its dropdown. Includes the
   * default backend name so the UI can highlight it as the initial
   * selection when the user hasn't picked yet.
   */
  listBackends(correlationId: string): BackendsListResponse {
    const backends: BackendInfo[] = [];
    // Stable iteration order: default first, then the rest.
    const defaultBackend = this.backends[this.defaultBackendName];
    if (defaultBackend) backends.push(defaultBackend.info);
    for (const [name, backend] of Object.entries(this.backends) as Array<
      [BackendName, Backend | undefined]
    >) {
      if (!backend) continue;
      if (name === this.defaultBackendName) continue;
      backends.push(backend.info);
    }

    return {
      meta: createApiMeta(correlationId),
      data: {
        defaultBackend: this.defaultBackendName,
        backends,
      },
    };
  }

  private resolveBackend(backendName: string | undefined): Backend {
    if (backendName === undefined || backendName === null || backendName === '') {
      return this.backends[this.defaultBackendName] as Backend;
    }

    const normalized = backendName.trim().toLowerCase();
    if (normalized !== 'ollama' && normalized !== 'anthropic') {
      throw validationFailed(
        `Unknown backend "${backendName}". Configured: ${Object.keys(this.backends).join(', ')}.`,
      );
    }

    const backend = this.backends[normalized];
    if (!backend) {
      throw validationFailed(
        `Backend "${normalized}" is not configured on this host. Configured: ${Object.keys(this.backends).join(', ')}.`,
      );
    }
    return backend;
  }
}

function resolveBackendName(raw: string): BackendName {
  const normalized = raw.trim().toLowerCase();
  if (normalized === 'ollama' || normalized === 'anthropic') {
    return normalized;
  }
  throw new Error(
    `BACKEND must be one of: ollama, anthropic. Received "${raw}".`,
  );
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
