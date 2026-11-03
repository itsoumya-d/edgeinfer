/**
 * Host-side driver for the paged-attention decode kernel.
 *
 * Zero readback by construction: the K and V caches are uploaded once into
 * STORAGE buffers that carry no COPY_SRC usage and are never mapped. The only
 * buffer copied back to the CPU is the small per-head output (plus a 4-byte
 * status word). There is no `mapAsync`/`getMappedRange` on KV buffers on this
 * path.
 *
 * Not wired into `EdgeInfer.generate()` yet. This is an opt-in building block
 * behind explicit calls; the ORT decoder remains the default path. Integration
 * waits on the acceptance gates in `docs/wgsl-paged-attention.md`.
 */
import { PAGED_ATTENTION_WGSL } from './paged-attention-wgsl';

export const PAGED_ATTENTION_BLOCK_SIZE = 16;
export const PAGED_ATTENTION_MAX_HEAD_DIM = 128;
export const PAGED_ATTENTION_WORKGROUP_SIZE = 64;

export type PagedAttentionErrorCode =
  | 'INVALID_PARAMS'
  | 'WEBGPU_UNAVAILABLE'
  | 'NON_FINITE_SCORE';

export class PagedAttentionError extends Error {
  readonly code: PagedAttentionErrorCode;
  constructor(code: PagedAttentionErrorCode, message: string) {
    super(message);
    this.name = 'PagedAttentionError';
    this.code = code;
  }
}

export interface PagedAttentionParams {
  device: GPUDevice;
  /** Single layer slice; the cache layout is [block][slot][kv_head][head_dim]. */
  layer: number;
  pastLen: number;
  numQueryHeads: number;
  numKvHeads: number;
  headDim: number;
  maxSeqLen: number;
  /** Must be 16. Present for forward compatibility and explicitness. */
  blockSize?: number;
  /** Defaults to 1 / sqrt(headDim). */
  scale?: number;
  /** Flat [numQueryHeads * headDim], row-major. */
  query: Float32Array;
  /** Flat [numBlocks * blockSize * numKvHeads * headDim] for ONE layer. */
  keyCache: Float32Array;
  /** Same shape as keyCache. */
  valueCache: Float32Array;
  /** Logical -> physical block ids, a permutation of [0, numBlocks). */
  blockTable: Uint32Array;
}

export interface PagedAttentionResult {
  /** Flat [numQueryHeads * headDim], row-major. */
  output: Float32Array;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

async function getPipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  let cached = pipelineCache.get(device);
  if (!cached) {
    cached = (async () => {
      const module = device.createShaderModule({ code: PAGED_ATTENTION_WGSL });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length > 0) {
        throw new PagedAttentionError(
          'WEBGPU_UNAVAILABLE',
          `paged-attention shader failed to compile: ${errors.map((e) => e.message).join('; ')}`,
        );
      }
      return device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    })();
    pipelineCache.set(device, cached);
  }
  return cached;
}

function fail(message: string): never {
  throw new PagedAttentionError('INVALID_PARAMS', message);
}

// Float32 -> IEEE 754 binary16 bits. Handles finite values and saturates
// overflow to infinity; the current fixture stays well inside ±1.
function float32ToFloat16Bits(values: Float32Array): Uint16Array {
  const out = new Uint16Array(values.length);
  const floatView = new Float32Array(1);
  const intView = new Int32Array(floatView.buffer);
  for (let index = 0; index < values.length; index += 1) {
    floatView[0] = values[index];
    const x = intView[0];
    let bits = (x >> 16) & 0x8000;
    let mantissa = (x >> 12) & 0x07ff;
    const exponent = (x >> 23) & 0xff;
    if (exponent < 103) {
      out[index] = bits;
      continue;
    }
    if (exponent > 142) {
      bits |= 0x7c00;
      bits |= (exponent === 255 ? 0x200 : 0) | ((x >> 13) & 0x3ff);
    } else if (exponent < 113) {
      mantissa |= 0x0800;
      bits |= (mantissa >> (114 - exponent)) + ((mantissa >> (113 - exponent)) & 1);
    } else {
      bits |= ((exponent - 112) << 10) | (mantissa >> 1);
      bits += mantissa & 1;
    }
    out[index] = bits;
  }
  return out;
}

export async function runPagedAttention(params: PagedAttentionParams): Promise<PagedAttentionResult> {
  const {
    device,
    layer,
    pastLen,
    numQueryHeads,
    numKvHeads,
    headDim,
    maxSeqLen,
    blockSize = PAGED_ATTENTION_BLOCK_SIZE,
    query,
    keyCache,
    valueCache,
    blockTable,
  } = params;

  if (!device) fail('device is required');
  if (blockSize !== PAGED_ATTENTION_BLOCK_SIZE) fail(`blockSize must be ${PAGED_ATTENTION_BLOCK_SIZE}`);
  if (!Number.isInteger(headDim) || headDim < 1 || headDim > PAGED_ATTENTION_MAX_HEAD_DIM) {
    fail(`headDim must be an integer in [1, ${PAGED_ATTENTION_MAX_HEAD_DIM}]`);
  }
  if (!Number.isInteger(numQueryHeads) || numQueryHeads < 1) fail('numQueryHeads must be >= 1');
  if (!Number.isInteger(numKvHeads) || numKvHeads < 1) fail('numKvHeads must be >= 1');
  if (numQueryHeads % numKvHeads !== 0) fail('numQueryHeads must be a multiple of numKvHeads');
  if (!Number.isInteger(maxSeqLen) || maxSeqLen < 1) fail('maxSeqLen must be >= 1');
  if (!Number.isInteger(pastLen) || pastLen < 0 || pastLen > maxSeqLen) {
    fail('pastLen must be an integer in [0, maxSeqLen]');
  }
  if (!Number.isInteger(layer) || layer < 0) fail('layer must be >= 0');

  const numBlocks = Math.ceil(maxSeqLen / blockSize);
  const scale = params.scale ?? 1 / Math.sqrt(headDim);
  if (!Number.isFinite(scale)) fail('scale must be finite');

  const expectedQuery = numQueryHeads * headDim;
  if (query.length !== expectedQuery) fail(`query must have ${expectedQuery} elements`);
  const expectedLayer = numBlocks * blockSize * numKvHeads * headDim;
  if (keyCache.length !== expectedLayer) fail(`keyCache must have ${expectedLayer} elements`);
  if (valueCache.length !== expectedLayer) fail(`valueCache must have ${expectedLayer} elements`);
  if (blockTable.length !== numBlocks) fail(`blockTable must have ${numBlocks} entries`);
  const seen = new Set(blockTable);
  if (seen.size !== numBlocks || [...seen].some((id) => id < 0 || id >= numBlocks)) {
    fail('blockTable must be a permutation of physical block ids');
  }

  const outSize = expectedQuery * 4;
  if (pastLen === 0) {
    return { output: new Float32Array(expectedQuery) };
  }
  if (!device.features?.has?.('shader-f16')) {
    throw new PagedAttentionError(
      'WEBGPU_UNAVAILABLE',
      'shader-f16 is required for the paged-attention f16 KV cache',
    );
  }

  const pipeline = await getPipeline(device);
  const keyBits = float32ToFloat16Bits(keyCache);
  const valueBits = float32ToFloat16Bits(valueCache);

  const qBuffer = device.createBuffer({ size: expectedQuery * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  // NOTE: no COPY_SRC on the KV buffers — they can never be read back.
  const kBuffer = device.createBuffer({ size: expectedLayer * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const vBuffer = device.createBuffer({ size: expectedLayer * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const tableBuffer = device.createBuffer({ size: numBlocks * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBuffer = device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const statusBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const uniformBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const outRead = device.createBuffer({ size: outSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const statusRead = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  device.queue.writeBuffer(qBuffer, 0, query.buffer as ArrayBuffer);
  device.queue.writeBuffer(kBuffer, 0, keyBits.buffer as ArrayBuffer);
  device.queue.writeBuffer(vBuffer, 0, valueBits.buffer as ArrayBuffer);
  device.queue.writeBuffer(tableBuffer, 0, blockTable.buffer as ArrayBuffer);
  // The kernel only sets this to 1 on a non-finite score. Reset it once on the
  // host so one workgroup cannot overwrite another workgroup's failure.
  device.queue.writeBuffer(statusBuffer, 0, new Uint32Array([0]).buffer as ArrayBuffer);

  const uniforms = new ArrayBuffer(48);
  const u32 = new Uint32Array(uniforms);
  const f32 = new Float32Array(uniforms);
  u32[0] = pastLen;
  u32[1] = numQueryHeads;
  u32[2] = numKvHeads;
  u32[3] = headDim;
  u32[4] = maxSeqLen;
  u32[5] = blockSize;
  u32[6] = layer;
  u32[7] = numBlocks;
  f32[8] = scale;
  // u32[9..11] / f32[9..11] remain zero padding (16-byte struct alignment).
  device.queue.writeBuffer(uniformBuffer, 0, uniforms);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: qBuffer } },
      { binding: 2, resource: { buffer: kBuffer } },
      { binding: 3, resource: { buffer: vBuffer } },
      { binding: 4, resource: { buffer: tableBuffer } },
      { binding: 5, resource: { buffer: outBuffer } },
      { binding: 6, resource: { buffer: statusBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numQueryHeads);
  pass.end();
  encoder.copyBufferToBuffer(outBuffer, 0, outRead, 0, outSize);
  encoder.copyBufferToBuffer(statusBuffer, 0, statusRead, 0, 4);
  device.queue.submit([encoder.finish()]);

  try {
    await outRead.mapAsync(GPUMapMode.READ);
    await statusRead.mapAsync(GPUMapMode.READ);
    const status = new Uint32Array(statusRead.getMappedRange().slice(0))[0];
    const output = new Float32Array(outRead.getMappedRange().slice(0));
    outRead.unmap();
    statusRead.unmap();
    if (status !== 0) {
      throw new PagedAttentionError('NON_FINITE_SCORE', 'attention score was non-finite at some position');
    }
    return { output };
  } finally {
    for (const buffer of [qBuffer, kBuffer, vBuffer, tableBuffer, outBuffer, statusBuffer, uniformBuffer, outRead, statusRead]) {
      buffer.destroy();
    }
  }
}
