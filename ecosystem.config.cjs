module.exports = {
  apps: [
    {
      name: 'swirlock-utility-llm-api',
      script: 'dist/main.js',
      instances: 1,
      autorestart: true,
      watch: false,
      env: {
        NODE_ENV: 'production',
        HOST: '0.0.0.0',
        PORT: 3000,
        OLLAMA_HOST: 'http://127.0.0.1:11434',
        OLLAMA_MODEL: 'qwen3.5:9b',
        OLLAMA_KEEP_ALIVE: '-1',
        PRELOAD_MODEL: 'true',
        MODEL_IMAGE_INPUT: 'true',
        MODEL_THINKING: 'false',
        JSON_BODY_LIMIT: '256mb',
      },
    },
  ],
};
