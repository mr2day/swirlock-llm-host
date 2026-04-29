import { randomUUID } from 'node:crypto';
import type { ApiMeta } from './types';

export function createApiMeta(correlationId: string): ApiMeta {
  return {
    requestId: randomUUID(),
    correlationId,
    apiVersion: 'v2',
    servedAt: new Date().toISOString(),
  };
}
