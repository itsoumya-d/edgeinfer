// SPDX-License-Identifier: MIT
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const {
  PAGED_ATTENTION_WGSL,
  PagedAttentionError,
  runPagedAttention,
} = await import(pathToFileURL(join(__dirname, '../dist/index.mjs')).href);

const wgslPath = join(__dirname, '../src/webgpu/kernels/paged_attention.wgsl');
const wgsl = readFileSync(wgslPath, 'utf8').replace(/\r\n/g, '\n');

const valid = {
  device: {},
  layer: 0,
  pastLen: 0,
  numQueryHeads: 2,
  numKvHeads: 1,
  headDim: 3,
  maxSeqLen: 32,
  query: new Float32Array(6),
  keyCache: new Float32Array(2 * 16 * 1 * 3),
  valueCache: new Float32Array(2 * 16 * 1 * 3),
  blockTable: new Uint32Array([0, 1]),
};

function invalid(overrides) {
  return { ...valid, ...overrides };
}

test('generated WGSL string matches the source file', () => {
  assert.equal(PAGED_ATTENTION_WGSL, wgsl);
});

test('zero-length past returns zeros without touching the device', async () => {
  const result = await runPagedAttention(valid);
  assert.deepEqual(Array.from(result.output), [0, 0, 0, 0, 0, 0]);
});

test('rejects a non-16 block size', async () => {
  await assert.rejects(
    () => runPagedAttention(invalid({ blockSize: 8 })),
    (error) => error instanceof PagedAttentionError && error.code === 'INVALID_PARAMS',
  );
});

test('rejects an invalid head dimension', async () => {
  await assert.rejects(
    () => runPagedAttention(invalid({ headDim: 0 })),
    (error) => error.code === 'INVALID_PARAMS',
  );
});

test('rejects query heads that are not a multiple of KV heads', async () => {
  await assert.rejects(
    () => runPagedAttention(invalid({ numQueryHeads: 3, numKvHeads: 2 })),
    (error) => error.code === 'INVALID_PARAMS',
  );
});

test('rejects a query with the wrong length', async () => {
  await assert.rejects(
    () => runPagedAttention(invalid({ query: new Float32Array(5) })),
    (error) => error.code === 'INVALID_PARAMS',
  );
});

test('rejects a block table that is not a permutation', async () => {
  await assert.rejects(
    () => runPagedAttention(invalid({ blockTable: new Uint32Array([0, 0]) })),
    (error) => error.code === 'INVALID_PARAMS',
  );
});
