# Swirlock LLM Host

Small NestJS model-host server for one local Ollama model.

This service is intentionally agnostic. It does not know whether callers are doing chat, RAG support, memory work, classification, or image understanding. Its job is to expose the model safely, keep it loaded, and serialize access to the local model runner.

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

## Model Selection

The server hosts exactly one model per running process. Choose it at startup with `OLLAMA_MODEL`.
The model must be present in `OLLAMA_MODELS`, a comma-separated allow-list:

```env
OLLAMA_MODELS=qwen3.5:9b,gemma4:e4b
OLLAMA_MODEL=qwen3.5:9b
```

For the main LLM computer, use:

```env
OLLAMA_MODEL=gemma4:e4b
```

For the utility LLM computer, keep:

```env
OLLAMA_MODEL=qwen3.5:9b
```

If you add another Ollama model later, add it to `OLLAMA_MODELS` first. The server refuses to start
if `OLLAMA_MODEL` is not in that list, so a typo cannot silently launch the wrong model.

Image input is controlled separately with `MODEL_IMAGE_INPUT`. Keep it `true` for vision-capable
models. Set it to `false` for a text-only model.

## Browser Test Page

Open this URL on the host machine:

```text
http://127.0.0.1:3000/test
```

From another computer on the same Wi-Fi, use:

```text
http://<this-computer-lan-ip>:3000/test
```

The test page uses the streaming WebSocket endpoint. It has a single composer for text plus
image attachments, removable image thumbnails, a thinking toggle, and live streamed model output.

## Endpoints

- `POST /v2/infer`
- `WS /v2/infer/stream`
- `GET /v2/health`
- `GET /v2/model/status`
- `POST /v2/model/preload`
- `POST /v2/model/unload`

All HTTP requests require:

```text
x-correlation-id: <stable request or turn id>
```

## Inference

Text-only request:

```json
{
  "requestContext": {
    "callerService": "rag-engine",
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
    "thinking": false,
    "responseFormat": "text",
    "ollama": {
      "temperature": 0.2
    }
  }
}
```

Text plus image request:

```json
{
  "requestContext": {
    "callerService": "context-fragmenter",
    "priority": 10,
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
    "appliedOptions": {
      "responseFormat": "text",
      "thinking": false,
      "ollama": {
        "temperature": 0.2
      }
    }
  }
}
```

`GET /v2/model/status` reports the active model and the configured model list:

```json
{
  "data": {
    "modelId": "qwen3.5:9b",
    "availableModels": ["qwen3.5:9b", "gemma4:e4b"]
  }
}
```

## Streaming Inference

Connect to:

```text
ws://<host>:3000/v2/infer/stream
```

Send one JSON message per WebSocket connection:

```json
{
  "type": "infer",
  "correlationId": "example-correlation-id",
  "request": {
    "requestContext": {
      "callerService": "browser-test",
      "requestedAt": "2026-04-29T12:00:00Z"
    },
    "input": {
      "parts": [{ "type": "text", "text": "Describe what you see." }]
    },
    "options": {
      "responseFormat": "text",
      "thinking": false,
      "ollama": {
        "temperature": 0.2
      }
    }
  }
}
```

The server streams JSON events:

- `accepted`
- `queued`
- `started`
- `thinking`
- `chunk`
- `done`
- `error`

Open multiple WebSocket connections for multiple callers. The host runs exactly one model request
at a time and queues the rest.

`requestContext.priority` is optional and numeric. Higher numbers run first. If a caller omits
`priority`, the request is treated as lower priority than every request that provides a finite
priority number. Requests with the same priority, including omitted-priority requests, run in
arrival order.

Queued events are only a wait-decision snapshot for the already accepted queued request. They
include:

- `position`: current 1-based position inside the waiting queue
- `requestsAhead`: active request plus queued requests ahead
- `queueDepth`: current total queued requests
- `defaultPriority`: `true` when the caller omitted `requestContext.priority`
- `priority`: the request priority number, only present when the caller provided one
- `averageRequestDurationMs`, when the server has recent request duration samples
- `estimatedWaitMs`, when the server has recent request duration samples
- `estimatedStartAt`, when the server has recent request duration samples

## Host Protection

The host keeps only host-level protections:

- `JSON_BODY_LIMIT`, default `256mb`
- one hardcoded active model request at a time
- an in-memory queue for additional requests

The host does not impose text length, image count, image byte, output length, elapsed time, or rate
limits. Callers can pass Ollama generation settings through `options.ollama`.

`MODEL_THINKING=false` is the default so thinking-capable models return usable response text through this API instead of spending work on internal reasoning fields.

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
- `gemma4:e4b` is included in the default `OLLAMA_MODELS` allow-list for the main LLM machine.
- This host implements the Swirlock v2 Model Host API.
