// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath

import * as ort from 'onnxruntime-web';
import { ModelCache } from './model-cache';
import { DownloadOptions, checkDownloadAbort } from './download';
import { RuntimeManager, GPUCapabilities } from './runtime-manager';
import { ModelOptions, ClassificationResult, Detection } from './types';
import { Tokenizer, TokenizerConfig, EncodingResult } from './tokenizer';
import { ImageProcessor } from './image-processor';

export interface EdgeInferOptions extends ModelOptions, DownloadOptions {
  tokenizerUrl?: string;
  tokenizerConfig?: TokenizerConfig;
  /**
   * Force int64 `input_ids`/`attention_mask` tensors. By default the engine
   * inspects ONNX input metadata and falls back to int32.
   */
  int64Inputs?: boolean;
}

export interface GenerateOptions {
  /** Maximum number of new tokens to emit. Default: 64. */
  maxNewTokens?: number;
  /** 0 (default) is greedy decoding; >0 enables temperature sampling. */
  temperature?: number;
  /** Optional top-k truncation for sampling. 0 disables it. */
  topK?: number;
  /** Stop token. Defaults to the tokenizer's `</s>` when present. */
  eosTokenId?: number;
}

export interface GenerationResult {
  text: string;
  tokenIds: number[];
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
  private _int64Inputs: boolean;

  private constructor(
    session: ort.InferenceSession,
    modelSize: number,
    provider: string,
    int64Inputs: boolean,
    tokenizer?: Tokenizer
  ) {
    this.session = session;
    this._inputNames = session.inputNames;
    this._outputNames = session.outputNames;
    this._modelSize = modelSize;
    this._provider = provider;
    this._int64Inputs = int64Inputs;
    this.tokenizer = tokenizer || null;
  }

  /**
   * Load a model from URL with automatic caching and optimal provider selection.
   */
  static async load(modelUrl: string, options?: EdgeInferOptions): Promise<EdgeInfer> {
    const cache = new ModelCache(options?.cacheName);
    const buffer = await cache.getModel(modelUrl, options?.forceDownload, options);
    return EdgeInfer.fromBuffer(buffer, options);
  }

  /**
   * Create an EdgeInfer instance from a pre-loaded model buffer.
   */
  static async fromBuffer(buffer: ArrayBuffer, options?: EdgeInferOptions): Promise<EdgeInfer> {
    // `provider` is the provider that actually created the session, which is
    // not necessarily the detected preference — see RuntimeManager.createSession.
    checkDownloadAbort(options?.signal);
    const { session, provider } = await RuntimeManager.createSession(
      buffer,
      options?.executionProviders
    );

    try {
      checkDownloadAbort(options?.signal);
      // Load tokenizer if URL provided
      let tokenizer: Tokenizer | undefined;
      if (options?.tokenizerUrl) {
        tokenizer = await Tokenizer.fromUrl(options.tokenizerUrl, options);
      } else if (options?.tokenizerConfig) {
        tokenizer = new Tokenizer(options.tokenizerConfig);
      }

      checkDownloadAbort(options?.signal);
      return new EdgeInfer(session, buffer.byteLength, provider, options?.int64Inputs ?? false, tokenizer);
    } catch (error) {
      try { await session.release(); } catch { /* Preserve the original load failure. */ }
      throw error;
    }
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
   *
   * `shapes` optionally gives the ONNX tensor shape per input name. When an
   * input has no explicit shape it defaults to `[1, data.length]`, which is
   * correct for 2-D sequence inputs (`input_ids`, `attention_mask`) but wrong
   * for anything of higher rank — a vision model declaring
   * `pixel_values: [1,3,224,224]` is rank 4 and rejects a rank-2 tensor with
   * "Invalid rank for input". Pass `shapes` for such models.
   */
  async predict(
    inputs: Record<string, Float32Array | Int32Array | BigInt64Array>,
    shapes?: Record<string, readonly number[]>
  ): Promise<Record<string, Float32Array>> {
    const results = await this.runInference(inputs, shapes);
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
    const inputs = this.textInputs(encoding);

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
    const inputs = this.textInputs(encoding);

    const result = await this.runInference(inputs);
    const output = result[this._outputNames[0]];
    
    if (!output) return new Float32Array();

    if (output.type !== 'float32') {
      throw new Error('EdgeInfer: Embedding output must be a float32 tensor.');
    }

    return this.poolEmbedding(output.data as Float32Array, output.dims, encoding.attentionMask);
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
   * Greedy or temperature-sampled autoregressive generation.
   *
   * Decoder models that expose `present.*` outputs and `past_key_values.*`
   * inputs reuse those tensors across steps (KV-cache). Models without cache
   * inputs are re-run over the full sequence each step.
   */
  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerationResult> {
    this.requireTokenizer();

    const maxNewTokens = Math.max(0, options.maxNewTokens ?? 64);
    const temperature = options.temperature ?? 0;
    const topK = options.topK ?? 0;
    const tokenizer = this.tokenizer!;
    const eosTokenId = options.eosTokenId
      ?? tokenizer.getSpecialTokenId('</s>')
      ?? tokenizer.getSpecialTokenId('<|endoftext|>');

    const encoding = tokenizer.encode(prompt, { addSpecialTokens: true });
    const promptIds = Array.from(encoding.inputIds);
    const int64 = this.useInt64Inputs();

    const makeIds = (values: number[]): Int32Array | BigInt64Array =>
      int64
        ? BigInt64Array.from(values, v => BigInt(v))
        : Int32Array.from(values);

    const dtype = int64 ? 'int64' : 'int32';
    let current = promptIds;
    let past: Record<string, ort.Tensor> = {};
    const generated: number[] = [];

    for (let step = 0; step < maxNewTokens; step++) {
      const totalLength = promptIds.length + generated.length;
      const inputs: Record<string, ort.Tensor> = {
        input_ids: new ort.Tensor(dtype, makeIds(current), [1, current.length]),
        attention_mask: new ort.Tensor(dtype, makeIds(new Array(totalLength).fill(1)), [1, totalLength]),
      };

      if (this._inputNames.includes('position_ids')) {
        const offset = totalLength - current.length;
        inputs.position_ids = new ort.Tensor(
          dtype,
          makeIds(current.map((_, i) => offset + i)),
          [1, current.length]
        );
      }

      for (const [name, tensor] of Object.entries(past)) {
        if (this._inputNames.includes(name)) inputs[name] = tensor;
      }

      const results = await this.session.run(inputs);
      const logits = this.findLogits(results);
      const next = this.pickNextToken(logits, temperature, topK);
      generated.push(next);

      if (eosTokenId !== undefined && next === eosTokenId) break;

      const nextPast: Record<string, ort.Tensor> = {};
      for (const outputName of this._outputNames) {
        const inputName = EdgeInfer.pastInputName(outputName);
        if (inputName && this._inputNames.includes(inputName) && results[outputName]) {
          nextPast[inputName] = results[outputName];
        }
      }

      if (Object.keys(nextPast).length > 0) {
        past = nextPast;
        current = [next];
      } else {
        past = {};
        current = [...promptIds, ...generated];
      }
    }

    return {
      text: tokenizer.decode(generated),
      tokenIds: generated,
    };
  }

  /** Pick the last-token logits from the first output that looks like logits. */
  private findLogits(results: ort.InferenceSession.ReturnType): ort.Tensor {
    const exact = results['logits'];
    if (exact) return exact;
    for (const name of Object.keys(results)) {
      if (name.toLowerCase().includes('logits')) return results[name];
    }
    return results[this._outputNames[0]];
  }

  /** Argmax (temperature 0) or temperature/top-k sampling over last-token logits. */
  private pickNextToken(logits: ort.Tensor, temperature: number, topK: number): number {
    const dims = logits.dims;
    const vocabSize = dims[dims.length - 1];
    const seqLen = dims.length >= 3 ? dims[dims.length - 2] : 1;
    const data = logits.data as Float32Array;
    const offset = Math.max(0, (seqLen - 1) * vocabSize);

    if (temperature <= 0) {
      let best = 0;
      let bestValue = -Infinity;
      for (let i = 0; i < vocabSize; i++) {
        const value = data[offset + i];
        if (value > bestValue) {
          bestValue = value;
          best = i;
        }
      }
      return best;
    }

    const scaled = new Float32Array(vocabSize);
    let maxLogit = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      scaled[i] = data[offset + i] / temperature;
      if (scaled[i] > maxLogit) maxLogit = scaled[i];
    }

    const candidates: Array<{ id: number; prob: number }> = [];
    let sum = 0;
    for (let i = 0; i < vocabSize; i++) {
      const prob = Math.exp(scaled[i] - maxLogit);
      candidates.push({ id: i, prob });
      sum += prob;
    }

    if (topK > 0 && topK < candidates.length) {
      candidates.sort((a, b) => b.prob - a.prob);
      candidates.length = topK;
      sum = candidates.reduce((acc, c) => acc + c.prob, 0);
    }

    let sample = Math.random() * sum;
    for (const candidate of candidates) {
      sample -= candidate.prob;
      if (sample <= 0) return candidate.id;
    }
    return candidates[candidates.length - 1].id;
  }

  private static pastInputName(outputName: string): string | null {
    if (!outputName.startsWith('present')) return null;
    return outputName.replace(/^present/, 'past_key_values');
  }

  private useInt64Inputs(): boolean {
    if (this._int64Inputs) return true;
    const metadata = (this.session as unknown as {
      inputMetadata?: Record<string, { type?: string }>;
    }).inputMetadata;
    return metadata?.['input_ids']?.type === 'int64';
  }

  /** Build input_ids/attention_mask tensors in the dtype the model declares. */
  private textInputs(encoding: EncodingResult): Record<string, Int32Array | BigInt64Array> {
    const int64 = this.useInt64Inputs();
    const inputs: Record<string, Int32Array | BigInt64Array> = {
      input_ids: int64 ? encoding.inputIdsInt64 : encoding.inputIds,
      attention_mask: int64 ? encoding.attentionMaskInt64 : encoding.attentionMask,
    };

    if (this._inputNames.includes('token_type_ids')) {
      inputs['token_type_ids'] = int64
        ? new BigInt64Array(encoding.inputIds.length)
        : new Int32Array(encoding.inputIds.length);
    }

    return inputs;
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
    const effectiveLayout = layout ?? 'NCHW';
    const tensorData = ImageProcessor.imageDataToFloat32Array(imageData, { layout: effectiveLayout });

    // Find the image input name
    const inputName = this._inputNames.find(n =>
      n === 'input' || n === 'pixel_values' || n === 'image'
    ) || this._inputNames[0];

    const result = await this.predict(
      { [inputName]: tensorData },
      { [inputName]: EdgeInfer.imageShape(imageData, effectiveLayout) }
    );
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

    const result = await this.predict(
      { [inputName]: tensorData },
      { [inputName]: EdgeInfer.imageShape(imageData, 'NCHW') }
    );
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

  /** Run without discarding the output tensor dimensions needed by embed(). */
  private async runInference(
    inputs: Record<string, Float32Array | Int32Array | BigInt64Array>,
    shapes?: Record<string, readonly number[]>
  ): Promise<ort.InferenceSession.ReturnType> {
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
      const shape = shapes?.[key] ?? [1, data.length];
      tensorInputs[key] = new ort.Tensor(type, data, shape as number[]);
    }

    return this.session.run(tensorInputs);
  }

  /** ONNX tensor shape for a 3-channel image tensor in the given layout. */
  private static imageShape(
    imageData: { width: number; height: number },
    layout: 'NCHW' | 'NHWC'
  ): number[] {
    const { width, height } = imageData;
    return layout === 'NCHW' ? [1, 3, height, width] : [1, height, width, 3];
  }

  private requireTokenizer(): void {
    if (!this.tokenizer) {
      throw new Error(
        'EdgeInfer: No tokenizer configured. Either pass tokenizerUrl/tokenizerConfig in options, ' +
        'or call setTokenizer() before using text-based inference methods.'
      );
    }
  }

  private softmax(logits: Float32Array): Float32Array {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
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

  private poolEmbedding(
    embeddings: Float32Array,
    dims: readonly number[],
    attentionMask: Int32Array
  ): Float32Array {
    // The shape, not divisibility by input length, distinguishes pooled output
    // from per-token output. Both pooled forms retain every model dimension.
    if ((dims.length === 1 || (dims.length === 2 && dims[0] === 1)) &&
        dims[dims.length - 1] > 0) {
      return this.l2Normalize(embeddings);
    }

    if (dims.length !== 3 || dims[0] !== 1 || dims[1] < 1 || dims[2] < 1) {
      throw new Error(
        `EdgeInfer: Unsupported embedding output shape [${dims.join(', ')}]. ` +
        'Expected [hidden], [1, hidden], or [1, sequence, hidden].'
      );
    }

    const [, seqLen, hiddenDim] = dims;
    if (seqLen !== attentionMask.length) {
      throw new Error(
        `EdgeInfer: Embedding sequence length ${seqLen} does not match ` +
        `attention mask length ${attentionMask.length}.`
      );
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
