// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath

export * from './types';
export * from './events';
export * from './model-cache';
export * from './runtime-manager';
export * from './tokenizer';
export * from './image-processor';
export * from './edgeinfer';

export type { DownloadOptions } from './download';
export * from './matryoshka';
export * from './speculative';
export * from './onnx-lm';

// Opt-in paged-attention building blocks (not wired into EdgeInfer.generate).
export {
  PAGED_ATTENTION_BLOCK_SIZE,
  PAGED_ATTENTION_MAX_HEAD_DIM,
  PAGED_ATTENTION_WORKGROUP_SIZE,
  PagedAttentionError,
  runPagedAttention,
} from './webgpu/paged-attention';
export { PAGED_ATTENTION_WGSL } from './webgpu/paged-attention-wgsl';
export type {
  PagedAttentionErrorCode,
  PagedAttentionParams,
  PagedAttentionResult,
} from './webgpu/paged-attention';
