// SPDX-License-Identifier: MIT
// Synthetic throughput + Recall@10 benchmark for the Matryoshka index.
// This is not an MTEB run; it measures the local index mechanics.
import { performance } from 'node:perf_hooks';
import { writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const { MatryoshkaIndex } = await import(join(here, '..', 'dist', 'index.mjs'));

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

const random = mulberry32(1234);
const randomVector = (dimensions) => {
  const vector = new Float32Array(dimensions);
  for (let i = 0; i < dimensions; i++) vector[i] = random() * 2 - 1;
  return vector;
};

const DOCS = 100_000;
const DIMENSIONS = 768;
const CLUSTERS = 128;
const centers = Array.from({ length: CLUSTERS }, () => randomVector(DIMENSIONS));
const records = new Array(DOCS);
for (let i = 0; i < DOCS; i++) {
  const center = centers[i % CLUSTERS];
  const embedding = new Float32Array(DIMENSIONS);
  for (let d = 0; d < DIMENSIONS; d++) embedding[d] = center[d] + (random() - 0.5) * 0.6;
  records[i] = { id: `doc-${i}`, text: `synthetic chunk ${i}`, embedding };
}

const buildStart = performance.now();
const index = new MatryoshkaIndex({
  dimensions: DIMENSIONS,
  coarseDimension: 64,
  refineTopK: 50,
});
await index.build(records);
const buildMs = performance.now() - buildStart;

const queryIndexes = Array.from({ length: 20 }, (_, i) => (i * 4999 + 1234) % DOCS);
const queries = queryIndexes.map((docIndex) => {
  const base = records[docIndex].embedding;
  const query = new Float32Array(DIMENSIONS);
  for (let d = 0; d < DIMENSIONS; d++) query[d] = base[d] + (random() - 0.5) * 0.03;
  return query;
});

const latencies = [];
let hits = 0;
for (let i = 0; i < queries.length; i++) {
  const start = performance.now();
  const results = index.search(queries[i], { topK: 10 });
  latencies.push(performance.now() - start);
  if (results.some((result) => result.id === `doc-${queryIndexes[i]}`)) hits++;
}
const sorted = [...latencies].sort((a, b) => a - b);
const percentile = (p) => sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)];

const recallRecords = records.slice(0, 2000);
const recallIndex = new MatryoshkaIndex({
  dimensions: DIMENSIONS,
  coarseDimension: 64,
});
await recallIndex.build(recallRecords);
const recallQueries = queries.slice(0, 8);
const groundTruth = recallQueries.map((query) =>
  recallIndex
    .exactSearchAtDimension(query, DIMENSIONS, 10)
    .map((candidate) => recallIndex.idList[candidate.index])
);
const dimensions = [64, 128, 256, 512, 768];
const recall = Object.fromEntries(
  dimensions.map((dimension) => [
    String(dimension),
    recallIndex.recallAtK(recallQueries, groundTruth, dimension, 10),
  ])
);

const report = {
  corpus: { documents: DOCS, dimensions: DIMENSIONS, clusters: CLUSTERS },
  build_ms: Number(buildMs.toFixed(2)),
  coarse_scan: {
    queries: queries.length,
    p50_ms: Number(percentile(0.5).toFixed(2)),
    p95_ms: Number(percentile(0.95).toFixed(2)),
    queries_per_second: Number(
      (queries.length / (latencies.reduce((a, b) => a + b, 0) / 1000)).toFixed(1)
    ),
    top10_hit_rate: Number((hits / queries.length).toFixed(3)),
  },
  recall_at_10_by_dimension: recall,
  note: 'Synthetic vectors and synthetic queries; not MTEB.',
};
console.log(JSON.stringify(report, null, 2));
writeFileSync(
  join(here, '..', 'bench-matryoshka.json'),
  JSON.stringify(report, null, 2) + '\n'
);
