// Single source of truth for this model host's runtime settings.
// Change the hosted model, port, Ollama URL, and model behavior here.

const env = {
  NODE_ENV: 'production',
  PORT: '3000',
  HOST: '0.0.0.0',
  OLLAMA_HOST: 'http://127.0.0.1:11434',
  OLLAMA_MODELS: 'qwen3.5:9b,gemma4:e4b,qwen2.5-coder:14b',
  OLLAMA_MODEL: 'qwen3.5:9b',
  OLLAMA_KEEP_ALIVE: '-1',
  PRELOAD_MODEL: 'true',
  MODEL_IMAGE_INPUT: 'true',
  MODEL_THINKING: 'false',
  JSON_BODY_LIMIT: '256mb',
};

module.exports = { env };
