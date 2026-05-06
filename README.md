# Swirlock LLM Host

Generic Model Host implementation for Swirlock. The ecosystem API is
WebSocket-only in contracts v4.

## Endpoint

```text
WS /v4/model
```

Client message types:

- `infer`
- `health.get`
- `model.status`
- `model.preload`
- `model.unload`
- `cancel`
- `heartbeat`

Server message types:

- `accepted`
- `queued`
- `started`
- `thinking`
- `chunk`
- `done`
- `health`
- `model.status`
- `model.preload`
- `model.unload`
- `error`
- `heartbeat`

Each message is the shared v4 envelope:

```json
{
  "type": "infer",
  "correlationId": "example-correlation-id",
  "payload": {
    "request": {
      "requestContext": {
        "callerService": "chat-orchestrator",
        "requestedAt": "2026-05-06T12:00:00Z",
        "priority": 100
      },
      "input": {
        "parts": [{ "type": "text", "text": "Hello" }]
      },
      "options": {
        "responseFormat": "text",
        "thinking": false
      }
    }
  }
}
```

There are no ecosystem REST endpoints.

## Run

```powershell
npm install
npm run build
npm run start
```

Configuration lives in `host.config.cjs`.
