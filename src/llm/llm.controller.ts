import { Body, Controller, Get, Headers, HttpCode, HttpStatus, Post } from '@nestjs/common';
import { validationFailed } from './api-error';
import { LlmService } from './llm.service';
import type {
  HealthResponse,
  InferRequest,
  InferResponse,
  ModelLifecycleRequest,
  ModelLifecycleResponse,
  ModelStatusResponse,
} from './types';

@Controller('v2')
export class LlmController {
  constructor(private readonly llmService: LlmService) {}

  @Post('infer')
  @HttpCode(HttpStatus.OK)
  async infer(
    @Headers('x-correlation-id') correlationId: string | undefined,
    @Body() body: InferRequest,
  ): Promise<InferResponse> {
    return this.llmService.infer(requireCorrelationId(correlationId), body);
  }

  @Get('health')
  async health(
    @Headers('x-correlation-id') correlationId: string | undefined,
  ): Promise<HealthResponse> {
    return this.llmService.health(requireCorrelationId(correlationId));
  }

  @Get('model/status')
  async modelStatus(
    @Headers('x-correlation-id') correlationId: string | undefined,
  ): Promise<ModelStatusResponse> {
    return this.llmService.modelStatus(requireCorrelationId(correlationId));
  }

  @Post('model/preload')
  @HttpCode(HttpStatus.ACCEPTED)
  async preloadModel(
    @Headers('x-correlation-id') correlationId: string | undefined,
    @Body() body: ModelLifecycleRequest,
  ): Promise<ModelLifecycleResponse> {
    return this.llmService.preload(requireCorrelationId(correlationId), body);
  }

  @Post('model/unload')
  @HttpCode(HttpStatus.ACCEPTED)
  async unloadModel(
    @Headers('x-correlation-id') correlationId: string | undefined,
    @Body() body: ModelLifecycleRequest,
  ): Promise<ModelLifecycleResponse> {
    return this.llmService.unload(requireCorrelationId(correlationId), body);
  }
}

function requireCorrelationId(value: string | undefined): string {
  if (!value?.trim()) {
    throw validationFailed('x-correlation-id header is required.');
  }

  return value.trim();
}
