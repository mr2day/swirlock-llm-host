# Hardware Setup

This host computes the largest `num_ctx` it can ask Ollama to load
the model with, based on the physical GPU it's running on. If you've
just cloned the repo onto a new machine, or you're picking up an
existing deployment that pre-dates the
[`context-window` work](../src/llm/context-window.ts), this is the
file that tells you what to set and where.

## TL;DR for an agent picking this up cold

There is exactly **one machine-specific value** you must add to
`host.config.local.cjs` (gitignored, machine-local) before the host
will start:

```js
// host.config.local.cjs
module.exports = {
  env: {
    OLLAMA_MODELS: '...',
    OLLAMA_MODEL: '...',
    HARDWARE_TOTAL_VRAM_BYTES: String(N * 1024 * 1024 * 1024),  // <-- this
  },
};
```

where `N` is your GPU's total VRAM in gigabytes. If you don't add
this, the host errors at startup with:

> `HARDWARE_TOTAL_VRAM_BYTES must be a positive integer in host.config.cjs.`

Known machines in the Swirlock ecosystem:

| Machine | Role | GPU VRAM | `HARDWARE_TOTAL_VRAM_BYTES` |
| --- | --- | --- | --- |
| Vanamonde LLM Host (main) | persona LLM (ministral-3:14b) | 16 GB | `String(16 * 1024 * 1024 * 1024)` |
| Utility / Fragmenter LLM Host | fragmenter LLM (qwen3.5:9b) | 12 GB | `String(12 * 1024 * 1024 * 1024)` |

If you're on a different machine, find your card's VRAM via
`nvidia-smi --query-gpu=memory.total --format=csv,noheader` (Linux
or Windows in PowerShell) and multiply by `1024 * 1024` to get
bytes.

## Why this exists

The host now sets `num_ctx` on every Ollama call (preload + per
chat) based on what will actually fit in this machine's VRAM. The
equations live in [`src/llm/context-window.ts`](../src/llm/context-window.ts);
the inputs are:

| Input | Source of truth |
| --- | --- |
| Total VRAM | `HARDWARE_TOTAL_VRAM_BYTES` (this file) |
| CUDA + framebuffer overhead | `HARDWARE_OVERHEAD_RESERVE_BYTES` (default 1 GB) |
| KV cache element size | `HARDWARE_KV_CACHE_ELEMENT_BYTES` (default 2 = fp16) |
| Model file size | queried live from `ollama.list()` |
| Model architecture | queried live from `ollama.show()` |

The output is the `num_ctx` Ollama is told to load with — the
largest power of two that fits available VRAM after subtracting the
model and overhead. The host then reports this back to its callers
(orchestrator, fragmenter) via the `contextWindow` block on
`model.status`, so they can size their prompt budgets without doing
any math themselves.

If `HARDWARE_TOTAL_VRAM_BYTES` is wrong (too high), Ollama will fail
to load the model at the computed `num_ctx`. If it's too low, the
host will load the model with a smaller `num_ctx` than the GPU can
actually support — degraded but functional. Always lean conservative
if you don't know — `8 * 1024 ** 3` (8 GB) is a safe default for
most consumer cards.

## Verification — what to look for after restart

After updating `host.config.local.cjs` and running
`pm2 reload swirlock-llm-host --update-env` (or
`pm2 startOrReload ecosystem.config.cjs --update-env` if you
changed an env var rather than just rebuilt), check the startup
log:

```text
[LlmService] Context window resolved: num_ctx=32768, prompt budget=26214, kvPerToken=204800 B, availableForKv=6.54 GB
[LlmService] Preloaded ministral-3:14b with keep_alive=-1, num_ctx=32768
```

You're looking for:

- `Context window resolved:` line present (not absent).
- A `num_ctx` value > the Ollama default of `4096`. If you still see
  `num_ctx=8192` and a `(fell back to default: …)` suffix, the
  equation didn't run cleanly — read the suffix message; it names
  what went wrong (usually `ollama.list` didn't return your model,
  or `HARDWARE_TOTAL_VRAM_BYTES` was missing).
- A `Preloaded ... num_ctx=...` line that matches.

## Refinements

Two values you may want to override later in `host.config.local.cjs`:

- `HARDWARE_OVERHEAD_RESERVE_BYTES` — raise this on machines that
  also drive a display (Windows desktops often need 2 GB rather
  than 1).
- `HARDWARE_KV_CACHE_ELEMENT_BYTES` — set to `1` if you set
  `OLLAMA_KV_CACHE_TYPE=q8_0` (halves KV memory, near-zero quality
  cost; ~2× num_ctx for the same VRAM). Set to `0.5` for q4_0
  (~4× num_ctx, measurable quality loss).

Both are documented inline in `host.config.cjs` defaults.
