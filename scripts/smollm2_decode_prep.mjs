// SPDX-License-Identifier: MIT
// Decode-parity preparation harness (P06).
// This is NOT decode parity. It preps the CPU side of the evidence chain
// for the eventual greedy-decode = ORT-baseline claim:
//
//   1. Loads the 20-prompt ORT baseline fixture
//      (`scripts/fixtures/smollm2-135m-20-baseline.json`): per-prompt token
//      ids, chosen logits, and top-5 logits.
//   2. Loads the real-geometry fixture + oracle reference: per-case
//      (prompt-N) attention output expectations.
//   3. Recomputes the CPU oracle from the real-geometry fixture and checks
//      it matches the committed reference within the fixture tolerances.
//   4. Writes a prep manifest (`scripts/fixtures/smollm2-decode-prep.json`)
//      that maps each baseline prompt to its oracle case name, so the later
//      decode-parity browser harness has a single source of truth.
//
// What it does NOT do: run the WGSL kernel, chain attention into a full
// decoder, or compare tokens/logits. The kernel is still an opt-in building
// block and the ORT path remains the default.
//
// Usage:
//   node scripts/smollm2_decode_prep.mjs [--write] [--stdout]
//   --write   persist scripts/fixtures/smollm2-decode-prep.json
//   --stdout  print the manifest JSON (default when --write is absent)
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const ROOT = "scripts/fixtures";
const BASELINE_PATH = join(ROOT, "smollm2-135m-20-baseline.json");
const GEOMETRY_PATH = join(ROOT, "smollm2-realgeometry-12.json");
const REFERENCE_PATH = join(ROOT, "smollm2-realgeometry-reference.json");
const OUT_PATH = join(ROOT, "smollm2-decode-prep.json");

function flatten(nested) {
  const out = [];
  for (const b of nested) for (const s of b) for (const h of s) for (const x of h) out.push(x);
  return out;
}

// JS port of scripts/paged_attention_reference.py online softmax
// (identical recurrence to scripts/smollm2_attention_oracle.mjs).
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

function compare(expected, actual, atol, rtol) {
  let maxAbs = 0;
  let failures = 0;
  for (let r = 0; r < expected.length; r += 1) {
    for (let c = 0; c < expected[r].length; c += 1) {
      const want = expected[r][c];
      const got = actual[r][c];
      const absolute = Math.abs(want - got);
      const relative = absolute / Math.max(Math.abs(want), 1e-12);
      maxAbs = Math.max(maxAbs, absolute);
      if (absolute > atol && relative > rtol) failures += 1;
    }
  }
  return { ok: failures === 0, maxAbs, failures };
}

const baseline = JSON.parse(await readFile(BASELINE_PATH, "utf8"));
const geometry = JSON.parse(await readFile(GEOMETRY_PATH, "utf8"));
const reference = JSON.parse(await readFile(REFERENCE_PATH, "utf8"));

const atol = geometry.atol ?? 0.002;
const rtol = geometry.rtol ?? 0.02;

// --- 1. Baseline shape -------------------------------------------------
const baselinePrompts = baseline.prompts ?? baseline.baseline ?? [];
const baselineCases = baselinePrompts.map((p, i) => ({
  index: i,
  name: p.id ?? p.name ?? `prompt-${String(i + 1).padStart(2, "0")}`,
  tokens: (p.generated_tokens ?? p.token_ids ?? p.tokens ?? []).length,
  top5: (p.steps ?? []).length,
}));
const totalTokens = baselineCases.reduce((n, c) => n + c.tokens, 0);

// --- 2-3. Recompute oracle + compare ----------------------------------
const refByName = new Map(reference.cases.map((r) => [r.name, r.expected]));
const oracleRows = [];
let oracleFailures = 0;
for (const c of geometry.cases) {
  const K = flatten(c.cache.layers[0].k);
  const V = flatten(c.cache.layers[0].v);
  const scale = 1 / Math.sqrt(c.cache.head_dim);
  const recomputed = pagedAttention(c.q, K, V, c.past_len, c.cache.num_kv_heads, c.cache.head_dim, c.cache.block_table, c.cache.block_size, scale);
  const committed = refByName.get(c.name);
  if (!committed) {
    oracleFailures += 1;
    oracleRows.push({ name: c.name, ok: false, error: "missing from committed reference" });
    continue;
  }
  const verdict = compare(committed, recomputed, atol, rtol);
  if (!verdict.ok) oracleFailures += 1;
  oracleRows.push({ name: c.name, ...verdict });
}

// --- 4. Manifest -------------------------------------------------------
const promptToCase = [];
for (const c of geometry.cases) {
  const base = baselineCases.find((b) => b.name === c.name);
  promptToCase.push({
    case: c.name,
    baselineTokens: base ? base.tokens : null,
    pastLen: c.past_len,
    numQueryHeads: c.q.length,
    numKvHeads: c.cache.num_kv_heads,
    headDim: c.cache.head_dim,
    blockTable: c.cache.block_table,
  });
}

const manifest = {
  schema_version: 1,
  kind: "smollm2_decode_prep",
  date: "2026-11-04",
  note: "Prep only: CPU oracle re-verified against the committed real-geometry reference and each baseline prompt mapped to its oracle case. This is not token/logit decode parity.",
  baseline: {
    path: BASELINE_PATH,
    cases: baselineCases.length,
    totalTokens,
  },
  geometry: {
    path: GEOMETRY_PATH,
    cases: geometry.cases.length,
    atol,
    rtol,
  },
  reference: {
    path: REFERENCE_PATH,
    cases: reference.cases.length,
  },
  oracleCheck: {
    ok: oracleFailures === 0,
    failures: oracleFailures,
    maxAbs: oracleRows.filter((r) => typeof r.maxAbs === "number").reduce((m, r) => Math.max(m, r.maxAbs), 0),
  },
  promptToCase,
};

if (process.argv.includes("--write")) {
  await writeFile(OUT_PATH, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`wrote ${OUT_PATH}`);
} else {
  console.log(JSON.stringify(manifest, null, 2));
}

if (oracleFailures > 0) {
  console.error(`oracle check failed on ${oracleFailures} case(s)`);
  process.exitCode = 1;
}
