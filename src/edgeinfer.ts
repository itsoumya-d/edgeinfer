// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com | +91 7031648617

import { LicenseValidator } from './license-validator';
import * as ort from 'onnxruntime-web';
import { ModelCache } from './model-cache';
import { RuntimeManager, GPUCapabilities } from './runtime-manager';
import { ModelOptions, ClassificationResult, Detection } from './types';
import { Tokenizer, TokenizerConfig, EncodingResult } from './tokenizer';
import { ImageProcessor } from './image-processor';

export interface EdgeInferOptions extends ModelOptions {
  licenseKey?: string;
  tokenizerUrl?: string;
  tokenizerConfig?: TokenizerConfig;
}

/**
 * EdgeInfer — On-Device WebGPU/WASM AI Inference Engine.
 * 
 * Executes ONNX models directly inside the browser using WebGPU (preferred)
 * or WebAssembly SIMD (fallback), with zero server API costs.
 * 
 * Powered by: BitNet b1.58, Mamba-2 SSM, Matryoshka MRL research.
 */
export class EdgeInfer {
  private session: ort.InferenceSession;
  private tokenizer: Tokenizer | null = null;
  private _inputNames: readonly string[];
  private _outputNames: readonly string[];
  private _modelSize: number;
  private _provider: string = 'wasm';

  private constructor(
    session: ort.InferenceSession,
    modelSize: number,
    provider: string,
    tokenizer?: Tokenizer
  ) {
    this.session = session;
    this._inputNames = session.inputNames;
    this._outputNames = session.outputNames;
    this._modelSize = modelSize;
    this._provider = provider;
    this.tokenizer = tokenizer || null;
  }

  /**
   * Load a model from URL with automatic caching and optimal provider selection.
   */
  static async load(modelUrl: string, options?: EdgeInferOptions): Promise<EdgeInfer> {
    if (options?.licenseKey) {
      LicenseValidator.validate(options);
    }

    const cache = new ModelCache(options?.cacheName);
    const buffer = await cache.getModel(modelUrl, options?.forceDownload);
    return EdgeInfer.fromBuffer(buffer, options);
  }

  /**
   * Create an EdgeInfer instance from a pre-loaded model buffer.
   */
  static async fromBuffer(buffer: ArrayBuffer, options?: EdgeInferOptions): Promise<EdgeInfer> {
    const caps = await RuntimeManager.detectCapabilities();
    const session = await RuntimeManager.createSession(buffer, options?.executionProviders);
    
    // Load tokenizer if URL provided
    let tokenizer: Tokenizer | undefined;
    if (options?.tokenizerUrl) {
      tokenizer = await Tokenizer.fromUrl(options.tokenizerUrl);
    } else if (options?.tokenizerConfig) {
      tokenizer = new Tokenizer(options.tokenizerConfig);
    }

    return new EdgeInfer(session, buffer.byteLength, caps.provider, tokenizer);
  }

  /**
   * Set or replace the tokenizer after model load.
   */
  setTokenizer(tokenizer: Tokenizer): void {
    this.tokenizer = tokenizer;
  }

  /**
   * Low-level tensor inference.
   * Pass raw tensor inputs and get raw tensor outputs.
   */
  async predict(inputs: Record<string, Float32Array | Int32Array | BigInt64Array>): Promise<Record<string, Float32Array>> {
    const tensorInputs: Record<string, ort.Tensor> = {};
    
    for (const [key, data] of Object.entries(inputs)) {
      let type: 'float32' | 'int32' | 'int64';
      if (data instanceof Float32Array) {
        type = 'float32';
      } else if (data instanceof BigInt64Array) {
        type = 'int64';
      } else {
        type = 'int32';
      }
      tensorInputs[key] = new ort.Tensor(type, data, [1, data.length]);
    }

    const results = await this.session.run(tensorInputs);
    const outputMap: Record<string, Float32Array> = {};

    for (const [key, tensor] of Object.entries(results)) {
      outputMap[key] = tensor.data as Float32Array;
    }

    return outputMap;
  }

  /**
   * Classify text using the loaded model.
   * Requires a tokenizer to be configured.
   */
  async classify(text: string): Promise<ClassificationResult[]> {
    this.requireTokenizer();
    
    const encoding = this.tokenizer!.encode(text, { addSpecialTokens: true });
    const inputs: Record<string, Int32Array> = {
      'input_ids': encoding.inputIds,
      'attention_mask': encoding.attentionMask,
    };

    // Add token_type_ids if model expects it
    if (this._inputNames.includes('token_type_ids')) {
      inputs['token_type_ids'] = new Int32Array(encoding.inputIds.length);
    }

    const result = await this.predict(inputs);
    const output = result[this._outputNames[0]];
    
    if (!output) return [];

    // Apply softmax to get probabilities
    const probs = this.softmax(output);
    
    return Array.from(probs).map((score, idx) => ({
      label: `Class_${idx}`,
      score
    })).sort((a, b) => b.score - a.score);
  }

  /**
   * Generate text embeddings for semantic search / RAG.
   * Returns a Float32Array of embedding dimensions.
   */
  async embed(text: string): Promise<Float32Array> {
    this.requireTokenizer();

    const encoding = this.tokenizer!.encode(text, { addSpecialTokens: true });
    const inputs: Record<string, Int32Array> = {
      'input_ids': encoding.inputIds,
      'attention_mask': encoding.attentionMask,
    };

    if (this._inputNames.includes('token_type_ids')) {
      inputs['token_type_ids'] = new Int32Array(encoding.inputIds.length);
    }

    const result = await this.predict(inputs);
    const output = result[this._outputNames[0]];
    
    if (!output) return new Float32Array();

    // Mean pooling: average across token dimension
    // Output shape is typically [1, seq_len, hidden_dim]
    // We want [hidden_dim] by averaging across seq_len
    return this.meanPool(output, encoding.attentionMask);
  }

  /**
   * Generate Matryoshka-truncated embeddings at specified dimension.
   * Supports variable-dimension MRL embeddings (64d → 256d → 768d).
   */
  async embedMatryoshka(text: string, dimension: number): Promise<Float32Array> {
    const fullEmbedding = await this.embed(text);
    if (dimension >= fullEmbedding.length) return fullEmbedding;

    // Truncate to requested dimension (MRL front-loads important info)
    const truncated = fullEmbedding.slice(0, dimension);

    // L2 normalize after truncation
    return this.l2Normalize(truncated);
  }

  /**
   * Sentiment analysis.
   */
  async sentiment(text: string): Promise<{ label: string; score: number }> {
    const results = await this.classify(text);
    if (results.length === 0) return { label: 'unknown', score: 0 };

    const best = results[0]; // Already sorted by score
    const labelMap: Record<string, string> = {
      'Class_0': 'negative',
      'Class_1': 'positive',
      'Class_2': 'neutral',
    };

    return {
      label: labelMap[best.label] || best.label,
      score: best.score
    };
  }

  /**
   * Classify an image using a vision model.
   */
  async classifyImage(imageData: ImageData, layout?: 'NCHW' | 'NHWC'): Promise<ClassificationResult[]> {
    const tensorData = ImageProcessor.imageDataToFloat32Array(imageData, { layout });
    
    // Find the image input name
    const inputName = this._inputNames.find(n => 
      n === 'input' || n === 'pixel_values' || n === 'image'
    ) || this._inputNames[0];

    const result = await this.predict({ [inputName]: tensorData });
    const output = result[this._outputNames[0]];
    
    if (!output) return [];

    const probs = this.softmax(output);
    
    return Array.from(probs).map((score, idx) => ({
      label: `Class_${idx}`,
      score
    })).sort((a, b) => b.score - a.score).slice(0, 10);
  }

  /**
   * Detect objects in an image.
   */
  async detectObjects(imageData: ImageData): Promise<Detection[]> {
    const tensorData = ImageProcessor.imageDataToFloat32Array(imageData);
    
    const inputName = this._inputNames.find(n => 
      n === 'input' || n === 'pixel_values' || n === 'image'
    ) || this._inputNames[0];

    const result = await this.predict({ [inputName]: tensorData });
    const output = result[this._outputNames[0]];
    const detections: Detection[] = [];
    
    if (output) {
      // Standard format: [x, y, w, h, score, class_id] per detection
      for (let i = 0; i < output.length; i += 6) {
        const confidence = output[i + 4];
        if (confidence > 0.5) {
          detections.push({
            bbox: [output[i], output[i+1], output[i+2], output[i+3]],
            score: confidence,
            label: `Object_${Math.round(output[i+5])}`
          });
        }
      }
    }
    return detections.sort((a, b) => b.score - a.score);
  }

  /**
   * Check GPU capabilities and recommended configuration.
   */
  static async getCapabilities(): Promise<GPUCapabilities> {
    return RuntimeManager.detectCapabilities();
  }

  /**
   * Check if WebGPU is available for accelerated inference.
   */
  static async isWebGPUAvailable(): Promise<boolean> {
    const caps = await RuntimeManager.detectCapabilities();
    return caps.hasWebGPU;
  }

  get inputNames(): readonly string[] { return this._inputNames; }
  get outputNames(): readonly string[] { return this._outputNames; }
  get modelSize(): number { return this._modelSize; }
  get executionProvider(): string { return this._provider; }

  /**
   * Release all resources (ONNX session, GPU memory).
   */
  dispose(): void {
    if (this.session) {
      this.session.release();
    }
    this.tokenizer = null;
    RuntimeManager.resetCache();
  }

  // --- Private Helpers ---

  private requireTokenizer(): void {
    if (!this.tokenizer) {
      throw new Error(
        'EdgeInfer: No tokenizer configured. Either pass tokenizerUrl/tokenizerConfig in options, ' +
        'or call setTokenizer() before using text-based inference methods.'
      );
    }
  }

  private softmax(logits: Float32Array): Float32Array {
    const maxLogit = Math.max(...Array.from(logits));
    const exps = new Float32Array(logits.length);
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
      exps[i] = Math.exp(logits[i] - maxLogit);
      sumExp += exps[i];
    }
    for (let i = 0; i < exps.length; i++) {
      exps[i] /= sumExp;
    }
    return exps;
  }

  private meanPool(embeddings: Float32Array, attentionMask: Int32Array): Float32Array {
    // Assume output is [1, seq_len, hidden_dim] flattened
    const seqLen = attentionMask.length;
    const hiddenDim = embeddings.length / seqLen;

    if (hiddenDim < 1 || !Number.isInteger(hiddenDim)) {
      // Output is already pooled (e.g., [CLS] token output)
      return embeddings;
    }

    const pooled = new Float32Array(hiddenDim);
    let maskSum = 0;

    for (let t = 0; t < seqLen; t++) {
      if (attentionMask[t] === 0) continue;
      maskSum++;
      for (let d = 0; d < hiddenDim; d++) {
        pooled[d] += embeddings[t * hiddenDim + d];
      }
    }

    if (maskSum > 0) {
      for (let d = 0; d < hiddenDim; d++) {
        pooled[d] /= maskSum;
      }
    }

    return this.l2Normalize(pooled);
  }

  private l2Normalize(vec: Float32Array): Float32Array {
    let norm = 0;
    for (let i = 0; i < vec.length; i++) {
      norm += vec[i] * vec[i];
    }
    norm = Math.sqrt(norm);
    if (norm === 0) return vec;

    const result = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      result[i] = vec[i] / norm;
    }
    return result;
  }
}
