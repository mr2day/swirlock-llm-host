import { ArgumentsHost, Catch, ExceptionFilter, HttpException, HttpStatus } from '@nestjs/common';
import type { Request, Response } from 'express';
import { createApiMeta } from './llm/response-meta';
import type { ApiErrorBody } from './llm/api-error';

@Catch()
export class HttpExceptionFilter implements ExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const request = ctx.getRequest<Request>();
    const response = ctx.getResponse<Response>();
    const status = exception instanceof HttpException ? exception.getStatus() : 500;
    const body = exception instanceof HttpException ? exception.getResponse() : undefined;
    const error = normalizeError(status, body, exception);

    response.status(status).json({
      meta: createApiMeta(request.header('x-correlation-id') ?? 'missing-correlation-id'),
      error,
    });
  }
}

function normalizeError(status: number, body: unknown, exception: unknown): ApiErrorBody {
  if (isRecord(body) && isRecord(body.error)) {
    const error = body.error;
    return {
      code:
        typeof error.code === 'string'
          ? (error.code as ApiErrorBody['code'])
          : fallbackCode(status),
      message: typeof error.message === 'string' ? error.message : fallbackMessage(status),
      retryable: typeof error.retryable === 'boolean' ? error.retryable : status >= 500,
      details: isRecord(error.details) ? error.details : undefined,
    };
  }

  if (isRecord(body)) {
    const message = body.message;
    return {
      code: fallbackCode(status),
      message: Array.isArray(message)
        ? message.join('; ')
        : typeof message === 'string'
          ? message
          : fallbackMessage(status),
      retryable: status >= 500,
    };
  }

  return {
    code: fallbackCode(status),
    message: exception instanceof Error ? exception.message : fallbackMessage(status),
    retryable: status >= 500,
  };
}

function fallbackCode(status: number): ApiErrorBody['code'] {
  if (status === HttpStatus.BAD_REQUEST) {
    return 'bad_request';
  }
  if (status === HttpStatus.PAYLOAD_TOO_LARGE) {
    return 'limit_exceeded';
  }
  if (status === HttpStatus.TOO_MANY_REQUESTS) {
    return 'model_overloaded';
  }
  if (status >= 500) {
    return 'internal_error';
  }
  return 'validation_failed';
}

function fallbackMessage(status: number): string {
  return status >= 500 ? 'Internal server error' : 'Request failed';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
