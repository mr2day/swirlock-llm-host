import 'reflect-metadata';
import './env';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { LlmService } from './llm/llm.service';
import { attachLlmWebSocketServer } from './llm/llm.websocket';
import { getRequiredStringEnv } from './llm/runtime';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  attachLlmWebSocketServer(app.getHttpServer(), app.get(LlmService));

  const port = parsePort(getRequiredStringEnv('PORT'));
  const host = getRequiredStringEnv('HOST');

  await app.listen(port, host);
  console.log(`Swirlock LLM Host listening for WS on ws://${host}:${port}/v4/model`);
}

void bootstrap();

function parsePort(value: string): number {
  const port = Number(value);
  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new Error('PORT must be an integer between 1 and 65535 in host.config.cjs.');
  }

  return port;
}
