// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath
//
// ONNX Runtime Web adapter for causal language models. This adapter recomputes
// the full token sequence on each call, so it is correctness-first and works
// with standard exports that accept input_ids/attention_mask and return logits.
// Cache-aware adapters can implement LanguageModel.truncate and store KV tensors
// in ModelState.cache without changing the speculative decoder.

import * as ort from 'onnxruntime-web';
import { LanguageModel, ModelState, DecodeResult } from './speculative';

export interface OnnxCausalOptions {
  inputIdsName?: string;
  attentionMaskName?: string;
  positionIdsName?: string;
  logitsName?: string;
}

export class OnnxCausalLM implements LanguageModel {
  readonly vocabSize: number;
  private readonly inputIdsName: string;
  private readonly attentionMaskName: string;
  private readonly positionIdsName?: string;
  private readonly logitsName: string;

  constructor(
    private readonly session: ort.InferenceSession,
    vocabSize: number,
    options: OnnxCausalOptions = {}
  ) {
    if (!Number.isInteger(vocabSize) || vocabSize < 1) {
      throw new Error('vocabSize must be a positive integer');
    }
    this.vocabSize = vocabSize;
    this.inputIdsName = options.inputIdsName ?? 'input_ids';
    this.attentionMaskName = options.attentionMaskName ?? 'attention_mask';
    this.positionIdsName = options.positionIdsName;
    this.logitsName = options.logitsName ?? 'logits';
  }

  async prefill(tokens: number[]): Promise<ModelState> {
    if (tokens.length === 0) throw new Error('prefill requires a non-empty prompt');
    return { tokens: tokens.slice() };
  }

  async decode(state: ModelState, tokens: number[]): Promise<DecodeResult> {
    const all = [...state.tokens, ...tokens];
    if (all.length === 0) throw new Error('decode requires at least one token');
    if (state.tokens.length === 0) throw new Error('decode requires a prefilled state');

    const feeds: Record<string, ort.Tensor> = {
      [this.inputIdsName]: new ort.Tensor(
        'int64',
        BigInt64Array.from(all, (token) => BigInt(token)),
        [1, all.length]
      ),
      [this.attentionMaskName]: new ort.Tensor(
        'int64',
        BigInt64Array.from({ length: all.length }, () => 1n),
        [1, all.length]
      ),
    };
    if (this.positionIdsName) {
      feeds[this.positionIdsName] = new ort.Tensor(
        'int64',
        BigInt64Array.from(all, (_, index) => BigInt(index)),
        [1, all.length]
      );
    }

    const outputs = await this.session.run(feeds);
    const tensor = outputs[this.logitsName];
    if (!tensor) {
      throw new Error(`ONNX model did not return "${this.logitsName}"`);
    }
    const dims = tensor.dims;
    const vocab = dims[dims.length - 1];
    if (vocab !== this.vocabSize) {
      throw new Error(
        `ONNX logits vocabulary ${vocab} does not match configured ${this.vocabSize}`
      );
    }
    const data = tensor.data as Float32Array;
    const seq = all.length;
    const expected = seq * vocab;
    if (data.length < expected) {
      throw new Error(`ONNX logits tensor is too small (${data.length} < ${expected})`);
    }

    const start = state.tokens.length - 1;
    const logits: Float32Array[] = [];
    for (let i = 0; i <= tokens.length; i++) {
      const position = start + i;
      const offset = dims.length === 3 ? position * vocab : position * vocab;
      logits.push(new Float32Array(data.subarray(offset, offset + vocab)));
    }
    return { state: { tokens: all }, logits };
  }

  truncate(state: ModelState, length: number): ModelState {
    return { tokens: state.tokens.slice(0, length) };
  }
}
