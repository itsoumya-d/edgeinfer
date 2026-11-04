// SPDX-License-Identifier: MIT
// CPU oracle + memory accounting for the real-geometry paged-attention fixture.
// Two modes:
//   node scripts/smollm2_attention_oracle.mjs
//     Computes the f32 online-softmax reference for every case in
//     smollm2-realgeometry-12.json and writes
//     scripts/fixtures/smollm2-realgeometry-reference.json
//     ({ name, expected } rows). This is the CPU oracle the browser harness
//     compares the WGSL output against (no browser required).
//   node scripts/smollm2_attention_oracle.mjs --memory
//     Prints the analytic peak-KV memory + allocation audit for the real
//     model geometry (9 q heads, 3 KV heads, head_dim 64, 30 layers,
//     max_seq_len 128, f16). This is the memory accounting gate.

import { readFile, writeFile } from "node:fs/promises";

function flatten(nested) {
  const out = [];
  for (const b of nested) for (const s of b) for (const h of s) for (const x of h) out.push(x);
  return out;
}

// JS port of scripts/paged_attention_reference.py online softmax.
function pagedAttention(q, K, V, pastLen, kvHeads, headDim, blockTable, blockSize, scale) {
  const qHeads = q.length;
  const group = qHeads / kvHeads;
  const out = [];
  for (let head = 0; head < qHeads; head += 1) {
    const kvh = Math.floor(head / group);
    let m = -Infinity;
    let l = 0;
    const acc = new Array(headDim).fill(0);
    for (let p = 0; p < pastLen; p += 1) {
      const logical = Math.floor(p / blockSize);
      const slot = p % blockSize;
      const phys = blockTable[logical];
      const base = ((phys * blockSize + slot) * kvHeads + kvh) * headDim;
      let s = 0;
      for (let d = 0; d < headDim; d += 1) s += q[head][d] * K[base + d];
      s *= scale;
      const nm = Math.max(m, s);
      const alpha = m === -Infinity ? 0 : Math.exp(m - nm);
      const prob = Math.exp(s - nm);
      l = l * alpha + prob;
      for (let d = 0; d < headDim; d += 1) acc[d] = acc[d] * alpha + prob * V[base + d];
      m = nm;
    }
    out.push(acc.map((x) => x / l));
  }
  return out;
}

if (process.argv.includes("--memory")) {
  const LAYERS = 30;
  const KV_HEADS = 3;
  const HEAD_DIM = 64;
  const MAX_SEQ_LEN = 128;
  const BLOCK_SIZE = 16;
  const numBlocks = Math.ceil(MAX_SEQ_LEN / BLOCK_SIZE);
  const perLayerBytes = numBlocks * BLOCK_SIZE * KV_HEADS * HEAD_DIM * 2; // f16 K or V
  const analytic = 2 * LAYERS * KV_HEADS * HEAD_DIM * MAX_SEQ_LEN * 2;
  const allocated = 2 * perLayerBytes * LAYERS;
  const blockTableBytes = 2 * LAYERS * numBlocks * 4 * 2; // K and V tables per layer? single table per layer
  const tablePerLayer = numBlocks * 4;
  console.log(JSON.stringify({
    geometry: { layers: LAYERS, kvHeads: KV_HEADS, headDim: HEAD_DIM, maxSeqLen: MAX_SEQ_LEN, blockSize: BLOCK_SIZE, numBlocks, dtype: "f16" },
    perLayerKVBytes: 2 * perLayerBytes * 2,
    analyticTotalBytes: analytic,
    allocatedTotalBytes: allocated,
    blockTableBytes: tablePerLayer * LAYERS,
    overheadPct: ((allocated - analytic) / analytic) * 100,
  }, null, 2));
  process.exit(0);
}

const fixture = JSON.parse(await readFile("scripts/fixtures/smollm2-realgeometry-12.json", "utf8"));
const rows = [];
for (const c of fixture.cases) {
  const K = flatten(c.cache.layers[0].k);
  const V = flatten(c.cache.layers[0].v);
  const scale = 1 / Math.sqrt(c.cache.head_dim);
  const expected = pagedAttention(c.q, K, V, c.past_len, c.cache.num_kv_heads, c.cache.head_dim, c.cache.block_table, c.cache.block_size, scale);
  rows.push({ name: c.name, expected });
}
const payload = { schema_version: 1, kind: "smollm2_realgeometry_reference", cases: rows };
await writeFile("scripts/fixtures/smollm2-realgeometry-reference.json", `${JSON.stringify(payload)}\n`, "utf8");
console.log(`wrote ${rows.length} reference rows`);
