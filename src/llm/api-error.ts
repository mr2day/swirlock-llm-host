import { HttpException, HttpStatus } from '@nestjs/common';

export type ApiErrorCode =
  | 'bad_request'
  | 'validation_failed'
  | 'upstream_unavailable'
  | 'model_unavailable'
  | 'model_overloaded'
  | 'limit_exceeded'
  | 'internal_error';

export interface ApiErrorBody {
  code: ApiErrorCode;
  message: string;
  retryable: boolean;
  details?: Record<string, unknown>;
}

export class ApiErrorException extends HttpException {
  constructor(status: HttpStatus, body: ApiErrorBody) {
    super({ error: body }, status);
  }
}

export function validationFailed(message: string, details?: Record<string, unknown>) {
  return new ApiErrorException(HttpStatus.BAD_REQUEST, {
    code: 'validation_failed',
    message,
    retryable: false,
    details,
  });
}

export function limitExceeded(message: string, details?: Record<string, unknown>) {
  return new ApiErrorException(HttpStatus.PAYLOAD_TOO_LARGE, {
    code: 'limit_exceeded',
    message,
    retryable: false,
    details,
  });
}

export function modelOverloaded(message: string, details?: Record<string, unknown>) {
  return new ApiErrorException(HttpStatus.TOO_MANY_REQUESTS, {
    code: 'model_overloaded',
    message,
    retryable: true,
    details,
  });
}

export function upstreamUnavailable(message: string, details?: Record<string, unknown>) {
  return new ApiErrorException(HttpStatus.BAD_GATEWAY, {
    code: 'upstream_unavailable',
    message,
    retryable: true,
    details,
  });
}
