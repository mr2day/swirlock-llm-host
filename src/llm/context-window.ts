// Equations for computing the maximum num_ctx this machine can actually
// load, based on hardware specs (sources of truth in host.config.cjs),
// model size (queried from Ollama), and model architecture params
// (also queried from Ollama, from the GGUF metadata).
//
// The formulas live here so they are testable in isolation and so the
// LlmService stays a thin orchestrator that consumes ready numbers.

export interface HardwareSpec {
  /** Total VRAM available on the GPU this host runs on, in bytes. */
  totalVramBytes: number;
  /** CUDA context + framebuffer + headroom reserve, in bytes. */
  overheadReserveBytes: number;
  /** Bytes per KV-cache element (2 for fp16, 1 for q8_0, 0.5 for q4_0). */
  kvCacheElementBytes: number;
}

export interface ModelArchitecture {
  /** Number of transformer blocks (layers). */
  numLayers: number;
  /** Embedding length (hidden dimension). */
  embeddingLength: number;
  /** Total attention heads. */
  numHeads: number;
  /** KV heads under GQA — equals numHeads for non-GQA architectures. */
  kvHeads: number;
}

export interface ContextWindowComputation {
  /** Number actually picked: largest power of 2 ≤ rawMaxNumCtx, capped. */
  targetNumCtx: number;
  /** Floor of the equation output before rounding. */
  rawMaxNumCtx: number;
  /** Memory left for KV cache after subtracting model + overhead. */
  availableForKvBytes: number;
  /** Cost per token in the KV cache. */
  kvPerTokenBytes: number;
  /** Did we fall back to the default because inputs were incomplete? */
  fellBackToDefault: boolean;
  /** Free-text reason if we fell back; useful for diagnostics. */
  fallbackReason?: string;
}

/**
 * Largest power of 2 not exceeding `n`. Returns 0 for n < 1.
 *
 * Used to snap the raw computed num_ctx to a stable boundary
 * (1024, 2048, 4096, 8192, ...). Power-of-2 num_ctx values are
 * universally supported by Ollama and avoid odd-shaped KV cache
 * allocations.
 */
export function largestPowerOfTwoNotExceeding(n: number): number {
  if (!Number.isFinite(n) || n < 1) return 0;
  return 2 ** Math.floor(Math.log2(n));
}

/**
 * Core equation: from hardware + model size + architecture, compute
 * how big a num_ctx can be loaded.
 *
 *   availableForKv = totalVram - modelSize - overheadReserve
 *   kvPerToken     = 2 * numLayers * kvHeads * headDim * kvCacheElementBytes
 *   headDim        = embeddingLength / numHeads
 *   rawMax         = floor(availableForKv / kvPerToken)
 *   target         = largestPowerOfTwoNotExceeding(rawMax)
 *
 * Returns the rounded target plus the intermediate values for logging.
 */
export function computeTargetNumCtx(args: {
  hardware: HardwareSpec;
  modelSizeBytes: number;
  architecture: ModelArchitecture;
  defaultNumCtx: number;
}): ContextWindowComputation {
  const { hardware, modelSizeBytes, architecture, defaultNumCtx } = args;

  const availableForKvBytes =
    hardware.totalVramBytes -
    modelSizeBytes -
    hardware.overheadReserveBytes;

  if (availableForKvBytes <= 0) {
    return {
      targetNumCtx: defaultNumCtx,
      rawMaxNumCtx: 0,
      availableForKvBytes,
      kvPerTokenBytes: 0,
      fellBackToDefault: true,
      fallbackReason:
        'No VRAM left for KV cache after model and overhead reserve. Check HARDWARE_TOTAL_VRAM_BYTES or shrink the model.',
    };
  }

  const headDim = architecture.embeddingLength / architecture.numHeads;
  const kvPerTokenBytes =
    2 *
    architecture.numLayers *
    architecture.kvHeads *
    headDim *
    hardware.kvCacheElementBytes;

  if (kvPerTokenBytes <= 0) {
    return {
      targetNumCtx: defaultNumCtx,
      rawMaxNumCtx: 0,
      availableForKvBytes,
      kvPerTokenBytes,
      fellBackToDefault: true,
      fallbackReason:
        'Invalid architecture params (kvPerTokenBytes ≤ 0). Check Ollama /api/show response.',
    };
  }

  const rawMaxNumCtx = Math.floor(availableForKvBytes / kvPerTokenBytes);
  const targetNumCtx = largestPowerOfTwoNotExceeding(rawMaxNumCtx);

  if (targetNumCtx === 0) {
    return {
      targetNumCtx: defaultNumCtx,
      rawMaxNumCtx,
      availableForKvBytes,
      kvPerTokenBytes,
      fellBackToDefault: true,
      fallbackReason:
        'Computed target rounded to 0 (insufficient VRAM for any meaningful num_ctx).',
    };
  }

  return {
    targetNumCtx,
    rawMaxNumCtx,
    availableForKvBytes,
    kvPerTokenBytes,
    fellBackToDefault: false,
  };
}

/**
 * Extracts the four architecture numbers we need from Ollama's
 * `/api/show` response `model_info` blob.
 *
 * Ollama's keys are namespaced by architecture family: `qwen3.*`,
 * `llama.*`, `mistral.*`, etc. We don't know the family up front, so
 * we scan for the suffixes (`block_count`, `embedding_length`,
 * `attention.head_count`, `attention.head_count_kv`) regardless of
 * prefix.
 *
 * Returns null when any required field is missing — caller falls back
 * to the configured default.
 */
export function extractArchitectureFromModelInfo(
  modelInfo: Record<string, unknown> | undefined,
): ModelArchitecture | null {
  if (!modelInfo) return null;

  const find = (suffix: string): number | null => {
    for (const [key, value] of Object.entries(modelInfo)) {
      if (key.endsWith(suffix) && typeof value === 'number' && value > 0) {
        return value;
      }
    }
    return null;
  };

  const numLayers = find('.block_count');
  const embeddingLength = find('.embedding_length');
  const numHeads = find('.attention.head_count');
  const kvHeadsRaw = find('.attention.head_count_kv');

  if (numLayers === null || embeddingLength === null || numHeads === null) {
    return null;
  }

  // For architectures without GQA, kv_head_count may be missing or
  // equal to head_count.
  const kvHeads = kvHeadsRaw ?? numHeads;

  return { numLayers, embeddingLength, numHeads, kvHeads };
}
