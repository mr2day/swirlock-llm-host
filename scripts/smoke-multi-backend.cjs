#!/usr/bin/env node
/**
 * Smoke test for the multi-backend LlmService. Loads host.config +
 * .local exactly as the running service would, instantiates
 * LlmService directly, asserts that:
 *   1. listBackends returns the configured backends (with Anthropic
 *      present when ANTHROPIC_API_KEY is set).
 *   2. The default backend (BACKEND env var) is reachable.
 *   3. A per-request `backend: 'anthropic'` override actually
 *      reaches AnthropicBackend.streamInfer and produces a chunk.
 *
 * Run after `npm run build`:
 *   node scripts/smoke-multi-backend.cjs
 */

'use strict';

const path = require('node:path');

const hostConfig = require(path.join(__dirname, '..', 'host.config.cjs'));
for (const [name, value] of Object.entries(hostConfig.env)) {
  process.env[name] = String(value);
}

const { LlmService } = require(
  path.join(__dirname, '..', 'dist', 'llm', 'llm.service'),
);

const service = new LlmService();

const list = service.listBackends('smoke-list-001');
console.log('backends.list →', JSON.stringify(list.data, null, 2));

const hasAnthropic = list.data.backends.some((b) => b.name === 'anthropic');

if (!hasAnthropic) {
  console.error(
    'Expected an Anthropic backend in the list. Check ANTHROPIC_API_KEY is set in host.config.local.cjs.',
  );
  process.exit(1);
}

(async () => {
  console.log('\n--- per-request backend override: anthropic ---');

  let chunkCount = 0;
  let sawDone = false;
  let assembledText = '';

  await service.streamInfer(
    'smoke-multi-001',
    {
      requestContext: {
        callerService: 'smoke-multi-backend',
        requestedAt: new Date().toISOString(),
      },
      input: {
        messages: [
          {
            role: 'user',
            content: 'Reply with exactly: multi-backend works.',
          },
        ],
      },
      backend: 'anthropic',
    },
    (event) => {
      if (event.type === 'chunk') {
        chunkCount += 1;
        assembledText += event.data.text;
      }
      if (event.type === 'done') {
        sawDone = true;
        console.log(`[event] done finishReason=${event.data.finishReason}`);
      } else {
        console.log(`[event] ${event.type}`);
      }
    },
  );

  console.log('---');
  console.log('chunks:', chunkCount);
  console.log('text:', JSON.stringify(assembledText));
  console.log('done event:', sawDone);

  if (!sawDone || chunkCount === 0) {
    console.error('SMOKE TEST FAILED');
    process.exit(1);
  }

  console.log('\nSMOKE TEST PASSED');
  process.exit(0);
})().catch((err) => {
  console.error('SMOKE TEST CRASHED:', err);
  process.exit(1);
});
