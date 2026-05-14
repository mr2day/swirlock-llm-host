// Shared, machine-agnostic defaults for this model host. Machine-specific
// values (which model is hosted, which Ollama URL, etc.) belong in
// `host.config.local.cjs`, which is gitignored. Local values override the
// defaults below. See `host.config.local.cjs.example` for the local template.

const fs = require('node:fs');
const path = require('node:path');

const defaults = {
  NODE_ENV: 'production',
  PORT: '3213',
  HOST: '0.0.0.0',
  OLLAMA_HOST: 'http://127.0.0.1:11434',
  OLLAMA_KEEP_ALIVE: '-1',
  PRELOAD_MODEL: 'true',
  MODEL_IMAGE_INPUT: 'true',
  MODEL_THINKING: 'false',
  JSON_BODY_LIMIT: '256mb',
  // KV cache element size in bytes. The default fp16 is 2 bytes per
  // K/V element. Set to 1 if you run Ollama with
  // OLLAMA_KV_CACHE_TYPE=q8_0 (halves the KV memory cost) or 0.5 for
  // q4_0 (quartered, with measurable quality loss). Used by the
  // context-window equations to compute how big a num_ctx fits in
  // this machine's VRAM after the model and CUDA overhead.
  HARDWARE_KV_CACHE_ELEMENT_BYTES: '2',
  // CUDA context, framebuffer, and headroom that must be left free
  // after the model and KV cache. ~1 GB is a safe starting value;
  // raise it on machines that also drive a display.
  HARDWARE_OVERHEAD_RESERVE_BYTES: String(1 * 1024 * 1024 * 1024),
  // Fraction of the loaded num_ctx that the orchestrator will fill
  // with prompt tokens. The remainder is reserved for the model's
  // response. 0.80 = 80% prompt / 20% response.
  PROMPT_BUDGET_FRACTION: '0.80',
  // Hard floor for num_ctx if the equations can't be evaluated
  // (architecture info missing from Ollama's /api/show, or this
  // host's HARDWARE_TOTAL_VRAM_BYTES not set). 8 K is a safe value
  // any decent GPU running a quantised mid-size model can serve.
  DEFAULT_NUM_CTX: '8192',
};

const localPath = path.join(__dirname, 'host.config.local.cjs');
const localOverrides = fs.existsSync(localPath)
  ? require(localPath).env || {}
  : {};

module.exports = { env: { ...defaults, ...localOverrides } };
