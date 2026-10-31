// SPDX-License-Identifier: MIT
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import 'fake-indexeddb/auto';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const {
  MatryoshkaIndex,
  MemoryVectorStore,
  IndexedDbVectorStore,
  createMatryoshkaEmbedder,
  dequantizeInt8,
  quantizeInt8,
  truncateAndNormalize,
} = await import(join(__dirname, '../dist/index.mjs'));

function seeded(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function makeRecords() {
  const random = seeded(7);
  const dimensions = 8;
  const centers = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
  ];
  const records = [];
  for (let cluster = 0; cluster < centers.length; cluster++) {
    for (let index = 0; index < 10; index++) {
      const embedding = new Float32Array(dimensions);
      for (let d = 0; d < dimensions; d++) {
        embedding[d] = centers[cluster][d] + (random() - 0.5) * 0.15;
      }
      records.push({ id: `c${cluster}-${index}`, text: `doc ${cluster}/${index}`, embedding });
    }
  }
  return records;
}

function queryFor(cluster = 2) {
  const embedding = new Float32Array(8);
  embedding[cluster] = 1;
  return embedding;
}

describe('Matryoshka primitives', () => {
  test('truncateAndNormalize returns a unit vector at the requested dimension', () => {
    const normalized = truncateAndNormalize(Float32Array.from([3, 4, 99]), 2);
    assert.equal(normalized.length, 2);
    assert.ok(Math.abs(normalized[0] - 0.6) < 1e-6);
    assert.ok(Math.abs(normalized[1] - 0.8) < 1e-6);
    assert.ok(Math.abs(Math.hypot(normalized[0], normalized[1]) - 1) < 1e-6);
  });

  test('INT8 quantization round-trips within one scale', () => {
    const vector = Float32Array.from([-1, 0.5, 0]);
    const quantized = quantizeInt8(vector, 3);
    assert.equal(quantized.data[0], -127);
    const restored = dequantizeInt8(quantized.data, quantized.scale);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(restored[i] - vector[i]) <= quantized.scale + 1e-6);
    }
  });

  test('embedMatryoshka factory truncates full embeddings', async () => {
    const embed = async () => Float32Array.from([0, 3, 4, 8]);
    const matryoshka = createMatryoshkaEmbedder(embed);
    const vector = await matryoshka('ignored', 2);
    assert.equal(vector.length, 2);
    assert.ok(Math.abs(vector[1] - 1) < 1e-6);
  });
});

describe('two-stage Matryoshka search', () => {
  test('coarse scan + full re-rank finds the right cluster', async () => {
    const index = new MatryoshkaIndex({ dimensions: 8, coarseDimension: 4, refineTopK: 20 });
    await index.build(makeRecords());
    const results = index.search(queryFor(2), { topK: 5 });
    assert.equal(results.length, 5);
    assert.ok(results[0].id.startsWith('c2-'));
    assert.ok(results[0].score > 0.9);
  });

  test('refine=false keeps coarse scores', async () => {
    const index = new MatryoshkaIndex({ dimensions: 8, coarseDimension: 4 });
    await index.build(makeRecords());
    const results = index.search(queryFor(1), { topK: 3, refine: false });
    assert.ok(results[0].id.startsWith('c1-'));
    assert.equal(results[0].score, results[0].coarseScore);
  });

  test('two-stage top-5 overlaps exact full search', async () => {
    const index = new MatryoshkaIndex({ dimensions: 8, coarseDimension: 4, refineTopK: 20 });
    await index.build(makeRecords());
    const approximate = index.search(queryFor(3), { topK: 5 }).map((result) => result.id);
    const exact = index
      .exactSearchAtDimension(queryFor(3), 8, 5)
      .map((candidate) => index.idList[candidate.index]);
    const overlap = approximate.filter((id) => exact.includes(id)).length;
    assert.ok(overlap >= 4, `expected overlap >= 4, got ${overlap}`);
  });

  test('recallAtK is high at 4 and 8 dimensions', async () => {
    const index = new MatryoshkaIndex({ dimensions: 8, coarseDimension: 4 });
    await index.build(makeRecords());
    const queries = [queryFor(0), queryFor(1), queryFor(2), queryFor(3)];
    const groundTruth = queries.map((query) =>
      index.exactSearchAtDimension(query, 8, 5).map((candidate) => index.idList[candidate.index])
    );
    const recall4 = index.recallAtK(queries, groundTruth, 4, 5);
    const recall8 = index.recallAtK(queries, groundTruth, 8, 5);
    assert.ok(recall4 > 0.6, `recall@4=${recall4}`);
    assert.ok(recall8 > 0.99, `recall@8=${recall8}`);
  });

  test('rejects a coarse stride larger than the indexed stride', async () => {
    const index = new MatryoshkaIndex({ dimensions: 8, coarseDimension: 4 });
    await index.build(makeRecords());
    assert.throws(() => index.search(queryFor(0), { coarseDimension: 8 }));
  });
});

describe('IndexedDB persistence', () => {
  test('build, persist, reload, and search', async () => {
    const store = new IndexedDbVectorStore(`matryoshka-test-${Date.now()}`, 'vectors');
    const records = makeRecords();
    const index = new MatryoshkaIndex({
      dimensions: 8,
      coarseDimension: 4,
      store,
    });
    await index.build(records);
    assert.equal(await store.count(), records.length);

    const reloaded = await MatryoshkaIndex.load(store, {
      dimensions: 8,
      coarseDimension: 4,
    });
    const results = reloaded.search(queryFor(2), { topK: 3 });
    assert.ok(results[0].id.startsWith('c2-'));
    await store.clear();
    assert.equal(await store.count(), 0);
    store.close();
  });
});
