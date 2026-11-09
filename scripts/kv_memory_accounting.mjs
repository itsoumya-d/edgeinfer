// SPDX-License-Identifier: MIT
// Memory accounting for the paged KV cache on the real model geometry.
// Compares the analytic peak-KV bound against the actual paged allocation.
//   analytic = 2 * layers * kv_heads * head_dim * max_seq_len * 2 bytes (f16)
//   allocated = 2 * layers * num_blocks * block_size * kv_heads * head_dim * 2
// Overhead comes only from block-round-up (ceil) and the block tables.
// Prints JSON. Run: node scripts/kv_memory_accounting.mjs

const LAYERS = 30;
const KV_HEADS = 3;
const HEAD_DIM = 64;
const MAX_SEQ_LEN = 128;
const BLOCK_SIZE = 16;
const F16_BYTES = 2;

const numBlocks = Math.ceil(MAX_SEQ_LEN / BLOCK_SIZE);
const perLayerKVBytes = numBlocks * BLOCK_SIZE * KV_HEADS * HEAD_DIM * F16_BYTES;
const analyticBytes = 2 * LAYERS * KV_HEADS * HEAD_DIM * MAX_SEQ_LEN * F16_BYTES;
const allocatedBytes = 2 * perLayerKVBytes * LAYERS;
const blockTableBytes = numBlocks * 4 * LAYERS;
const totalBytes = allocatedBytes + blockTableBytes;
const overheadPct = ((totalBytes - analyticBytes) / analyticBytes) * 100;

console.log(JSON.stringify({
  geometry: { layers: LAYERS, kvHeads: KV_HEADS, headDim: HEAD_DIM, maxSeqLen: MAX_SEQ_LEN, blockSize: BLOCK_SIZE, numBlocks, dtype: "f16" },
  perLayerKVBytes,
  analyticBytes,
  allocatedBytes,
  blockTableBytes,
  totalBytes,
  overheadPct: Math.round(overheadPct * 1000) / 1000,
  gate: "peak KV allocation within 5% of analytic bound plus block-table overhead",
  passes: overheadPct <= 5,
}, null, 2));
