// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath

export interface GpuKernelConfig {
  name: string;
  wgsl: string;
  workgroupSize: [number, number, number];
  elementsCount: number;
  bytesPerElement?: number;
}

export interface GpuBenchmarkResult {
  kernelName: string;
  durationMs: number;
  elementsProcessed: number;
  gflops: number;
  bandwidthGBs: number;
  timestampSupported: boolean;
}

export const TENSOR_REDUCTION_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

var<workgroup> sdata: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>
) {
  let tid = local_id.x;
  let idx = global_id.x;
  sdata[tid] = input_data[idx];
  workgroupBarrier();

  var s: u32 = 32u;
  while (s > 0u) {
    if (tid < s) {
      sdata[tid] = sdata[tid] + sdata[tid + s];
    }
    workgroupBarrier();
    s = s >> 1u;
  }

  if (tid == 0u) {
    output_data[group_id.x] = sdata[0];
  }
}
`;

export function calculateKernelThroughput(
  elements: number,
  durationMs: number,
  bytesPerElement: number = 4,
  flopsPerElement: number = 2
): { gflops: number; bandwidthGBs: number } {
  if (durationMs <= 0 || elements <= 0) {
    return { gflops: 0, bandwidthGBs: 0 };
  }
  const sec = durationMs / 1000.0;
  const totalFlops = elements * flopsPerElement;
  const totalBytes = elements * bytesPerElement;

  const gflops = totalFlops / (sec * 1e9);
  const bandwidthGBs = totalBytes / (sec * 1e9);
  return {
    gflops: Number(gflops.toFixed(3)),
    bandwidthGBs: Number(bandwidthGBs.toFixed(3))
  };
}

export function cpuReferenceReduction(data: Float32Array): number {
  let sum = 0.0;
  for (let i = 0; i < data.length; i++) {
    sum += data[i];
  }
  return sum;
}
