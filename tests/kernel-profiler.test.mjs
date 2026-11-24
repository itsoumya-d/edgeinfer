// SPDX-License-Identifier: MIT
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const {
  TENSOR_REDUCTION_WGSL,
  calculateKernelThroughput,
  cpuReferenceReduction,
} = await import(pathToFileURL(join(__dirname, '../dist/index.mjs')).href);

test('TENSOR_REDUCTION_WGSL contains valid compute entry point and workgroup barrier', () => {
  assert.ok(typeof TENSOR_REDUCTION_WGSL === 'string');
  assert.ok(TENSOR_REDUCTION_WGSL.includes('@compute @workgroup_size(64, 1, 1)'));
  assert.ok(TENSOR_REDUCTION_WGSL.includes('workgroupBarrier()'));
  assert.ok(TENSOR_REDUCTION_WGSL.includes('output_data[group_id.x] = sdata[0]'));
});

test('calculateKernelThroughput computes GFLOPS and bandwidth accurately', () => {
  const elements = 10_000_000;
  const durationMs = 10.0; // 0.01 sec
  const result = calculateKernelThroughput(elements, durationMs, 4, 2);

  // 20M FLOPs in 0.01s = 2 GFLOPS
  assert.strictEqual(result.gflops, 2.0);
  // 40MB in 0.01s = 4 GB/s
  assert.strictEqual(result.bandwidthGBs, 4.0);
});

test('calculateKernelThroughput handles zero or negative duration safely', () => {
  const rZero = calculateKernelThroughput(1000, 0);
  assert.strictEqual(rZero.gflops, 0);
  assert.strictEqual(rZero.bandwidthGBs, 0);

  const rNeg = calculateKernelThroughput(1000, -5);
  assert.strictEqual(rNeg.gflops, 0);
  assert.strictEqual(rNeg.bandwidthGBs, 0);
});

test('cpuReferenceReduction calculates exact sum of Float32Array', () => {
  const data = new Float32Array([1.5, 2.5, 3.0, 4.0, 5.0]);
  const sum = cpuReferenceReduction(data);
  assert.strictEqual(sum, 16.0);
});
