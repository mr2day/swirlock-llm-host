# Utility LLM Model Host

Small NestJS model-host server for the local Ollama model `qwen3.5:9b`.

This service is intentionally agnostic. It does not know whether callers are doing chat, RAG support, memory work, classification, or image understanding. Its job is to expose the model safely and protect the model host with size, timeout, token, and concurrency limits.

It accepts text plus optional image input and returns text only. It does not support image generation.

## Setup

```powershell
npm install
copy .env.example .env
npm run build
npm run start:prod
```

The default server binds to `0.0.0.0:3000`, so another computer on the same Wi-Fi can call:

```text
http://<this-computer-lan-ip>:3000/v2/infer
```

Make sure Windows Firewall allows inbound TCP traffic on port `3000`.

## Endpoints

- `POST /v2/infer`
- `GET /v2/health`
- `GET /v2/model/status`
- `POST /v2/model/preload`
- `POST /v2/model/unload`

All requests require:

```text
x-correlation-id: <stable request or turn id>
```

## Inference

Text-only request:

```json
{
  "requestContext": {
    "callerService": "rag-engine",
    "priority": "interactive",
    "requestedAt": "2026-04-29T12:00:00Z"
  },
  "input": {
    "parts": [
      {
        "type": "text",
        "text": "Summarize this paragraph."
      }
    ]
  },
  "options": {
    "temperature": 0.2,
    "maxOutputTokens": 256,
    "responseFormat": "text"
  }
}
```

Text plus image request:

```json
{
  "requestContext": {
    "callerService": "context-fragmenter",
    "priority": "background",
    "requestedAt": "2026-04-29T12:00:00Z"
  },
  "input": {
    "parts": [
      {
        "type": "text",
        "text": "Describe only visible facts in this image."
      },
      {
        "type": "image",
        "imageBase64": "<base64 image or data:image/png;base64,...>",
        "mimeType": "image/png"
      }
    ]
  }
}
```

Images can also be supplied with `imageUrl` instead of `imageBase64`.

Response:

```json
{
  "meta": {
    "requestId": "9bd77b0c-a99e-4ed2-a316-9126930ecf57",
    "correlationId": "example-correlation-id",
    "apiVersion": "v2",
    "servedAt": "2026-04-29T12:00:01.000Z"
  },
  "data": {
    "modelId": "qwen3.5:9b",
    "output": {
      "text": "The image shows ..."
    },
    "finishReason": "stop",
    "generatedAt": "2026-04-29T12:00:01.000Z",
    "usage": {
      "inputTokens": 10,
      "outputTokens": 20,
      "totalTokens": 30
    },
    "appliedOptions": {
      "maxOutputTokens": 256,
      "responseFormat": "text",
      "temperature": 0.2
    }
  }
}
```

## Model Protection

The host enforces:

- `MAX_TEXT_CHARS`
- `MAX_IMAGES`
- `MAX_IMAGE_BYTES`
- `MAX_OUTPUT_TOKENS`
- `MAX_CONTEXT_TOKENS`
- `MAX_CONCURRENT_REQUESTS`
- `REQUEST_TIMEOUT_MS`
- `IMAGE_FETCH_TIMEOUT_MS`

Caller options are treated as hints. The host only accepts a small safe option set and clamps output tokens to `MAX_OUTPUT_TOKENS`.

`MODEL_THINKING=false` is the default so thinking-capable models return usable response text through this API instead of spending the output budget on internal reasoning fields.

## Keeping The Model Loaded

The app preloads the model on startup with `OLLAMA_KEEP_ALIVE=-1`. Every inference request also sends the same keep-alive value.

This is the best Ollama-native way to keep the model resident in RAM/VRAM during normal use. It cannot be absolute: if another app consumes enough VRAM, the OS, driver, or Ollama may evict or fail to load the model.

For constant service on Windows, build the app and run it under PM2:

```powershell
npm install -g pm2
npm run build
pm2 start ecosystem.config.cjs
pm2 save
```

## Notes

- Ollama must be running locally on this machine.
- `qwen3.5:9b` is configured by default and can be changed with `OLLAMA_MODEL`.
- This host is the Utility LLM profile of the Swirlock v2 Model Host API.
