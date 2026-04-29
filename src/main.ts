import 'reflect-metadata';
import './env';
import { join } from 'node:path';
import { json, static as serveStatic, urlencoded } from 'express';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { HttpExceptionFilter } from './http-exception.filter';
import { LlmService } from './llm/llm.service';
import { attachLlmWebSocketServer } from './llm/llm.websocket';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { bodyParser: false });
  const jsonLimit = process.env.JSON_BODY_LIMIT ?? '50mb';

  app.useGlobalFilters(new HttpExceptionFilter());
  app.enableCors({ origin: true });
  app.use('/test', serveStatic(join(process.cwd(), 'public', 'test'), { index: 'index.html' }));
  app.use(json({ limit: jsonLimit }));
  app.use(urlencoded({ extended: true, limit: jsonLimit }));
  attachLlmWebSocketServer(app.getHttpServer(), app.get(LlmService));

  const port = Number(process.env.PORT ?? 3000);
  const host = process.env.HOST ?? '0.0.0.0';

  await app.listen(port, host);
  console.log(`Utility LLM Model Host listening on http://${host}:${port}`);
}

void bootstrap();
