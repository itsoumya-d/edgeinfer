// SPDX-License-Identifier: MIT
// Dump REAL SmolLM2 K/V tensors for paged-attention parity.
// Runs the model via ORT (CPU) prefill on real prompts, reads present.*,
// decodes f16 bits to f32, and emits a real-geometry paged-attention fixture
// shaped exactly like the synthetic one (same case schema), so the browser
// harness can run the WGSL kernel against true model-distribution K/V.
// This is NOT a decode-parity gate; it only captures representative tensors.
//
// Usage:
//   node scripts/smollm2_dump_kv.mjs \
//     --model models/smollm2-135m-instruct/onnx/model_fp16.onnx \
//     --prompts scripts/fixtures/smollm2-135m-20-prompts.json \
//     --out scripts/fixtures/smollm2-realgeometry-12.json

import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import * as ort from "onnxruntime-web";

function parseArgs(argv) {
  const args = {
    model: "models/smollm2-135m-instruct/onnx/model_fp16.onnx",
    prompts: "scripts/fixtures/smollm2-135m-20-prompts.json",
    out: "scripts/fixtures/smollm2-realgeometry-12.json",
  };
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (flag === "--model") args.model = value;
    else if (flag === "--prompts") args.prompts = value;
    else if (flag === "--out") args.out = value;
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
}

function int64Tensor(values, dims) {
  return new ort.Tensor(
    "int64",
    BigInt64Array.from(values, (value) => BigInt(value)),
    dims,
  );
}

// Exact f16 -> f32 decode (values are exactly representable).
function f16ToF32(bits) {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i += 1) {
    const b = bits[i];
    const sign = (b >> 15) & 1;
    const exp = (b >> 10) & 0x1f;
    const mant = b & 0x3ff;
    let value;
    if (exp === 0) {
      value = sign ? -1 : 1 * Math.pow(2, -14) * (mant / 1024);
      value = (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
    } else if (exp === 31) {
      value = mant ? NaN : sign ? -Infinity : Infinity;
    } else {
      value = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
    }
    out[i] = value;
  }
  return out;
}

// Build paged-layout K/V [block][slot][kv_head][head_dim] from ORT [B,KV,S,D] flat.
function toPagedLayout(flat, kvHeads, seq, headDim, blockSize, maxSeqLen, blockTable) {
  const numBlocks = Math.ceil(maxSeqLen / blockSize);
  const out = new Float32Array(numBlocks * blockSize * kvHeads * headDim);
  for (let pos = 0; pos < seq; pos += 1) {
    const slot = pos % blockSize;
    const phys = blockTable[Math.floor(pos / blockSize)];
    const dstBase = (phys * blockSize + slot) * kvHeads * headDim;
    const srcBase = pos * kvHeads * headDim;
    out.set(flat.subarray(srcBase, srcBase + kvHeads * headDim), dstBase);
  }
  return Array.from(out);
}

// Convert flat -> nested [block][slot][kv][dim] matching the synthetic fixture schema.
function toNested(paged, numBlocks, blockSize, kvHeads, headDim) {
  const block = [];
  for (let b = 0; b < numBlocks; b += 1) {
    const slots = [];
    for (let s = 0; s < blockSize; s += 1) {
      const heads = [];
      for (let h = 0; h < kvHeads; h += 1) {
        const base = ((b * blockSize + s) * kvHeads + h) * headDim;
        heads.push(paged.slice(base, base + headDim));
      }
      slots.push(heads);
    }
    block.push(slots);
  }
  return block;
}

const args = parseArgs(process.argv.slice(2));
ort.env.wasm.wasmPaths = pathToFileURL(
  `${join(process.cwd(), "node_modules", "onnxruntime-web", "dist")}/`,
).href;
const session = await ort.InferenceSession.create(args.model, {
  executionProviders: ["cpu"],
});

const fixture = JSON.parse(await readFile(args.prompts, "utf8"));
const prompts = fixture.prompts;
const layerCount = session.inputNames.filter((name) => /^past_key_values\.\d+\.key$/.test(name)).length;
const BLOCK_SIZE = 16;
const QUERY_HEADS = 9;
const KV_HEADS = 3;
const HEAD_DIM = 64;
const MAX_SEQ_LEN = 128;

function round6(x) {
  return Math.round(x * 1e6) / 1e6;
}

const cases = [];
for (const entry of prompts) {
  const ids = entry.input_ids;
  const feeds = {
    input_ids: int64Tensor(ids, [1, ids.length]),
    attention_mask: int64Tensor(new Array(ids.length).fill(1), [1, ids.length]),
    position_ids: int64Tensor(ids.map((_, i) => i), [1, ids.length]),
  };
  for (let layer = 0; layer < layerCount; layer += 1) {
    feeds[`past_key_values.${layer}.key`] = new ort.Tensor("float16", new Uint16Array(0), [1, KV_HEADS, 0, HEAD_DIM]);
    feeds[`past_key_values.${layer}.value`] = new ort.Tensor("float16", new Uint16Array(0), [1, KV_HEADS, 0, HEAD_DIM]);
  }
  const output = await session.run(feeds);
  const seq = ids.length;

  // Take layer 0 for the fixture (any layer is equally valid for geometry).
  const pk = output["present.0.key"];
  const pv = output["present.0.value"];
  const kFlat = f16ToF32(pk.data);
  const vFlat = f16ToF32(pv.data);
  void kFlat; void vFlat;

  const numBlocks = Math.ceil(MAX_SEQ_LEN / BLOCK_SIZE);
  const blockTable = Array.from({ length: numBlocks }, (_, i) => i);
  // Deterministic permutation (reverse) to exercise the block table.
  blockTable.reverse();

  const kPaged = (() => {
    const out = new Float32Array(numBlocks * BLOCK_SIZE * KV_HEADS * HEAD_DIM);
    for (let pos = 0; pos < seq; pos += 1) {
      const slot = pos % BLOCK_SIZE;
      const phys = blockTable[Math.floor(pos / BLOCK_SIZE)];
      const dst = (phys * BLOCK_SIZE + slot) * KV_HEADS * HEAD_DIM;
      const src = pos * KV_HEADS * HEAD_DIM;
      out.set(kFlat.subarray(src, src + KV_HEADS * HEAD_DIM), dst);
    }
    return out;
  })();
  const vPaged = (() => {
    const out = new Float32Array(numBlocks * BLOCK_SIZE * KV_HEADS * HEAD_DIM);
    for (let pos = 0; pos < seq; pos += 1) {
      const slot = pos % BLOCK_SIZE;
      const phys = blockTable[Math.floor(pos / BLOCK_SIZE)];
      const dst = (phys * BLOCK_SIZE + slot) * KV_HEADS * HEAD_DIM;
      const src = pos * KV_HEADS * HEAD_DIM;
      out.set(vFlat.subarray(src, src + KV_HEADS * HEAD_DIM), dst);
    }
    return out;
  })();

  const q = Array.from({ length: QUERY_HEADS }, (_, h) =>
    Array.from({ length: HEAD_DIM }, (_, d) => round6(Math.sin((h + 1) * 0.3 + d * 0.137) * 0.8)),
  );

  cases.push({
    name: entry.id,
    sequence_length: seq,
    layer: 0,
    past_len: seq,
    cache: {
      num_layers: 1,
      num_kv_heads: KV_HEADS,
      head_dim: HEAD_DIM,
      max_seq_len: MAX_SEQ_LEN,
      block_size: BLOCK_SIZE,
      block_table: blockTable,
      layers: [
        {
          k: toNested(Array.from(kPaged), numBlocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM),
          v: toNested(Array.from(vPaged), numBlocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM),
        },
      ],
    },
    q,
    expected: null, // browser-computed reference not included; harness compares to Python oracle separately
  });
  console.log(`${entry.id}: seq=${seq}`);
}

const payload = {
  schema_version: 1,
  kind: "smollm2_realgeometry_attention",
  note: "Real K/V from SmolLM2-135M ORT present.* tensors. q is the deterministic sin(seed) vector documented in smollm2_dump_kv.mjs. Harness compares WGSL output against a CPU reference computed in Node at runtime.",
  atol: 2e-3,
  rtol: 2e-2,
  cases,
};
await writeFile(args.out, `${JSON.stringify(payload)}\n`, "utf8");
console.log(`wrote ${cases.length} cases to ${args.out}`);
