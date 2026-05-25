import { HttpException, HttpStatus } from '@nestjs/common';
import type { Server } from 'node:http';
import { WebSocket, WebSocketServer, type RawData } from 'ws';
import { MODEL_WS_PATH } from '../config';
import type { ApiErrorBody } from './api-error';
import { LlmService, type StreamEvent } from './llm.service';
import type { InferRequest, ModelLifecycleRequest } from './types';

interface V4Envelope {
  type: string;
  correlationId: string;
  payload?: unknown;
  error?: ApiErrorBody;
}

type ActiveRequestMap = Map<string, AbortController>;

export function attachLlmWebSocketServer(server: Server, llmService: LlmService): WebSocketServer {
  const webSocketServer = new WebSocketServer({
    server,
    path: MODEL_WS_PATH,
  });

  webSocketServer.on('connection', (socket) => {
    const activeRequests: ActiveRequestMap = new Map();

    socket.on('close', () => abortAll(activeRequests));
    socket.on('error', () => abortAll(activeRequests));
    socket.on('message', (data) => {
      void handleMessage(socket, data, llmService, activeRequests);
    });
  });

  return webSocketServer;
}

async function handleMessage(
  socket: WebSocket,
  data: RawData,
  llmService: LlmService,
  activeRequests: ActiveRequestMap,
): Promise<void> {
  let correlationId = 'missing-correlation-id';

  try {
    const envelope = parseEnvelope(data);
    correlationId = envelope.correlationId;

    if (envelope.type === 'heartbeat') {
      sendEnvelope(socket, {
        type: 'heartbeat',
        correlationId,
        payload: { receivedAt: new Date().toISOString() },
      });
      return;
    }

    if (envelope.type === 'cancel') {
      const active = activeRequests.get(correlationId);
      active?.abort();
      activeRequests.delete(correlationId);
      return;
    }

    if (envelope.type === 'health.get') {
      const result = await llmService.health(
        correlationId,
        optionalBackend(envelope),
      );
      sendEnvelope(socket, {
        type: 'health',
        correlationId,
        payload: result.data,
      });
      return;
    }

    if (envelope.type === 'model.status') {
      const result = await llmService.modelStatus(
        correlationId,
        optionalBackend(envelope),
      );
      sendEnvelope(socket, {
        type: 'model.status',
        correlationId,
        payload: result.data,
      });
      return;
    }

    if (envelope.type === 'backends.list') {
      const result = llmService.listBackends(correlationId);
      sendEnvelope(socket, {
        type: 'backends.list',
        correlationId,
        payload: result.data,
      });
      return;
    }

    if (envelope.type === 'model.preload') {
      const result = await llmService.preload(
        correlationId,
        requireRequest<ModelLifecycleRequest>(envelope),
        optionalBackend(envelope),
      );
      sendEnvelope(socket, {
        type: 'model.preload',
        correlationId,
        payload: result.data,
      });
      return;
    }

    if (envelope.type === 'model.unload') {
      const result = await llmService.unload(
        correlationId,
        requireRequest<ModelLifecycleRequest>(envelope),
        optionalBackend(envelope),
      );
      sendEnvelope(socket, {
        type: 'model.unload',
        correlationId,
        payload: result.data,
      });
      return;
    }

    if (envelope.type !== 'infer') {
      throw validationException(
        'WebSocket message type must be infer, health.get, model.status, backends.list, model.preload, model.unload, cancel, or heartbeat.',
      );
    }

    if (activeRequests.has(correlationId)) {
      sendError(
        socket,
        correlationId,
        validationError(`Inference request ${correlationId} is already active.`),
      );
      return;
    }

    const abortController = new AbortController();
    activeRequests.set(correlationId, abortController);

    try {
      await llmService.streamInfer(
        correlationId,
        requireRequest<InferRequest>(envelope),
        (event) => sendEnvelope(socket, toEnvelope(event, correlationId)),
        abortController.signal,
      );
    } finally {
      activeRequests.delete(correlationId);
    }
  } catch (error) {
    sendError(socket, correlationId, normalizeError(error));
  }
}

function parseEnvelope(data: RawData): V4Envelope {
  let parsed: unknown;

  try {
    parsed = JSON.parse(data.toString('utf8'));
  } catch {
    throw validationException('WebSocket message must be valid JSON.');
  }

  if (!isRecord(parsed)) {
    throw validationException('WebSocket message must be an object.');
  }

  if (typeof parsed.type !== 'string' || !parsed.type.trim()) {
    throw validationException('type is required.');
  }

  if (typeof parsed.correlationId !== 'string' || !parsed.correlationId.trim()) {
    throw validationException('correlationId is required.');
  }

  return {
    type: parsed.type.trim(),
    correlationId: parsed.correlationId.trim(),
    payload: isRecord(parsed.payload) ? parsed.payload : undefined,
  };
}

function requireRequest<T>(envelope: V4Envelope): T {
  if (!isRecord(envelope.payload) || !isRecord(envelope.payload.request)) {
    throw validationException('payload.request is required.');
  }

  return envelope.payload.request as T;
}

/**
 * Optional `backend` selector from the envelope payload (e.g. for
 * `model.status` to ask for a non-default backend's status). Returns
 * undefined when not present, in which case LlmService falls back to
 * the default backend.
 */
function optionalBackend(envelope: V4Envelope): string | undefined {
  if (!isRecord(envelope.payload)) return undefined;
  const value = envelope.payload.backend;
  return typeof value === 'string' && value.trim().length > 0 ? value : undefined;
}

function toEnvelope(event: StreamEvent, correlationId: string): V4Envelope {
  if ('data' in event && isRecord(event.data)) {
    return { type: event.type, correlationId, payload: event.data };
  }

  return { type: event.type, correlationId, payload: {} };
}

function sendEnvelope(socket: WebSocket, envelope: V4Envelope): void {
  if (socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify(envelope));
}

function sendError(socket: WebSocket, correlationId: string, error: ApiErrorBody): void {
  sendEnvelope(socket, { type: 'error', correlationId, error });
}

function abortAll(activeRequests: ActiveRequestMap): void {
  for (const controller of activeRequests.values()) {
    controller.abort();
  }
  activeRequests.clear();
}

function validationException(message: string): HttpException {
  return new HttpException(
    {
      error: validationError(message),
    },
    HttpStatus.BAD_REQUEST,
  );
}

function normalizeError(error: unknown): ApiErrorBody {
  if (error instanceof HttpException) {
    const response = error.getResponse();

    if (isRecord(response) && isRecord(response.error)) {
      return normalizeApiErrorBody(response.error, error.getStatus());
    }

    return {
      code: fallbackCode(error.getStatus()),
      message: error.message || fallbackMessage(error.getStatus()),
      retryable: error.getStatus() >= 500,
    };
  }

  return {
    code: 'internal_error',
    message: error instanceof Error ? error.message : 'WebSocket inference failed.',
    retryable: true,
  };
}

function normalizeApiErrorBody(error: Record<string, unknown>, status: number): ApiErrorBody {
  return {
    code:
      typeof error.code === 'string' ? (error.code as ApiErrorBody['code']) : fallbackCode(status),
    message: typeof error.message === 'string' ? error.message : fallbackMessage(status),
    retryable: typeof error.retryable === 'boolean' ? error.retryable : status >= 500,
    details: isRecord(error.details) ? error.details : undefined,
  };
}

function validationError(message: string): ApiErrorBody {
  return {
    code: 'validation_failed',
    message,
    retryable: false,
  };
}

function fallbackCode(status: number): ApiErrorBody['code'] {
  if (status === HttpStatus.BAD_REQUEST) return 'bad_request';
  if (status === HttpStatus.PAYLOAD_TOO_LARGE) return 'limit_exceeded';
  if (status >= 500) return 'internal_error';
  return 'validation_failed';
}

function fallbackMessage(status: number): string {
  return status >= 500 ? 'Internal server error' : 'Request failed';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
