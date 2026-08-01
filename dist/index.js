"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/index.ts
var index_exports = {};
__export(index_exports, {
  EdgeInfer: () => EdgeInfer,
  EventEmitter: () => EventEmitter,
  ImageProcessor: () => ImageProcessor,
  ModelCache: () => ModelCache,
  RuntimeManager: () => RuntimeManager,
  Tokenizer: () => Tokenizer
});
module.exports = __toCommonJS(index_exports);

// src/events.ts
var EventEmitter = class {
  constructor() {
    this.listeners = {};
  }
  on(event, callback) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(callback);
  }
  emit(event, ...args) {
    if (this.listeners[event]) {
      this.listeners[event].forEach((cb) => cb(...args));
    }
  }
};

// src/model-cache.ts
var ModelCache = class extends EventEmitter {
  constructor(cacheName = "edgeinfer-models") {
    super();
    this.cacheName = cacheName;
  }
  async getModel(url, forceDownload = false) {
    if (typeof caches === "undefined") {
      const response2 = await fetch(url);
      if (!response2.ok) {
        throw new Error(
          `EdgeInfer: failed to fetch model from ${url} - HTTP ${response2.status} ${response2.statusText}`
        );
      }
      return response2.arrayBuffer();
    }
    const cache = await caches.open(this.cacheName);
    const cachedResponse = await cache.match(url);
    if (cachedResponse && !forceDownload) {
      return cachedResponse.arrayBuffer();
    }
    this.emit("downloadStart", url);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `EdgeInfer: failed to fetch model from ${url} - HTTP ${response.status} ${response.statusText}`
      );
    }
    await cache.put(url, response.clone());
    this.emit("downloadComplete", url);
    return response.arrayBuffer();
  }
};

// src/runtime-manager.ts
var ort = __toESM(require("onnxruntime-web"));
var ADAPTER_TIMEOUT_MS = 5e3;
var _RuntimeManager = class _RuntimeManager {
  /**
   * Detect GPU capabilities and recommend the best execution configuration.
   */
  static async detectCapabilities() {
    if (this.cachedCapabilities) return this.cachedCapabilities;
    const capabilities = {
      provider: "wasm",
      hasWebGPU: false,
      hasWebNN: false,
      estimatedVRAM: 0,
      recommendedQuantization: "int8"
    };
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      try {
        const gpu = navigator.gpu;
        const adapter = await _RuntimeManager.withTimeout(
          Promise.resolve(gpu.requestAdapter()),
          ADAPTER_TIMEOUT_MS
        );
        if (adapter) {
          capabilities.hasWebGPU = true;
          capabilities.provider = "webgpu";
          const maxBufferSize = adapter.limits?.maxBufferSize || 0;
          const maxStorageSize = adapter.limits?.maxStorageBufferBindingSize || 0;
          capabilities.estimatedVRAM = Math.max(maxBufferSize, maxStorageSize);
          const vramGB = capabilities.estimatedVRAM / (1024 * 1024 * 1024);
          if (vramGB >= 8) {
            capabilities.recommendedQuantization = "fp16";
          } else if (vramGB >= 4) {
            capabilities.recommendedQuantization = "int8";
          } else {
            capabilities.recommendedQuantization = "int4";
          }
        }
      } catch {
      }
    }
    if (typeof navigator !== "undefined" && "ml" in navigator) {
      try {
        const ml = navigator.ml;
        if (ml) {
          capabilities.hasWebNN = true;
          if (!capabilities.hasWebGPU) {
            capabilities.provider = "webnn";
          }
        }
      } catch {
      }
    }
    this.cachedCapabilities = capabilities;
    return capabilities;
  }
  /**
   * Get the best available execution provider with smart fallback.
   * Priority: WebGPU → WebNN → WASM
   */
  static async getBestExecutionProvider() {
    const caps = await this.detectCapabilities();
    return caps.provider;
  }
  /**
   * Create an ONNX inference session with automatic provider selection.
   * Falls back gracefully if the preferred provider fails.
   *
   * Returns BOTH the session and the provider that actually served it. The
   * detected capability (`detectCapabilities().provider`) is only a preference:
   * a provider can be detected as available and still fail to create a session
   * (for example onnxruntime-web only registers the WebGPU execution provider
   * in some builds/versions). Callers must report the returned `provider`, not
   * the detected one, or `EdgeInfer.executionProvider` will lie.
   */
  static async createSession(modelBuffer, providers) {
    const requestedProviders = providers || [await this.getBestExecutionProvider()];
    for (const provider of requestedProviders) {
      try {
        const session2 = await ort.InferenceSession.create(
          modelBuffer,
          {
            executionProviders: [provider],
            graphOptimizationLevel: "all"
          }
        );
        return { session: session2, provider };
      } catch (err) {
        console.warn(`EdgeInfer: Failed to create session with provider "${provider}", trying next...`, err);
      }
    }
    console.warn("EdgeInfer: All preferred providers failed, falling back to WASM");
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });
    return { session, provider: "wasm" };
  }
  /**
   * Resolve `promise`, or resolve to `null` if it has not settled within `ms`.
   * Used to stop a hung GPU driver from blocking model loading forever.
   */
  static withTimeout(promise, ms) {
    let timer;
    return Promise.race([
      promise.then((v) => {
        clearTimeout(timer);
        return v;
      }),
      new Promise((resolve) => {
        timer = setTimeout(() => {
          console.warn(`EdgeInfer: navigator.gpu.requestAdapter() did not settle within ${ms}ms; falling back to WASM.`);
          resolve(null);
        }, ms);
        if (typeof timer?.unref === "function") timer.unref();
      })
    ]);
  }
  /**
   * Estimate if a model of given size (bytes) can fit in available VRAM.
   */
  static async canFitModel(modelSizeBytes) {
    const caps = await this.detectCapabilities();
    if (!caps.hasWebGPU) return true;
    return caps.estimatedVRAM >= modelSizeBytes * 1.5;
  }
  /**
   * Reset cached capabilities (useful for testing or re-detection).
   */
  static resetCache() {
    this.cachedCapabilities = null;
  }
};
_RuntimeManager.cachedCapabilities = null;
var RuntimeManager = _RuntimeManager;

// src/tokenizer.ts
var Tokenizer = class _Tokenizer {
  constructor(config) {
    this.vocab = new Map(Object.entries(config.vocab));
    this.inverseVocab = /* @__PURE__ */ new Map();
    for (const [token, id] of this.vocab) {
      this.inverseVocab.set(id, token);
    }
    this.merges = (config.merges || []).map((m) => {
      if (Array.isArray(m)) {
        return [m[0], m[1]];
      }
      const parts = String(m).split(" ");
      return [parts[0], parts[1]];
    });
    this.specialTokens = new Map(Object.entries(config.specialTokens || {}));
    this.maxLength = config.maxLength || 512;
    this.padTokenId = config.padTokenId ?? 0;
    this.unkTokenId = config.unkTokenId ?? 100;
    this.clsTokenId = config.clsTokenId ?? 101;
    this.sepTokenId = config.sepTokenId ?? 102;
    this.bpeCache = /* @__PURE__ */ new Map();
  }
  /**
   * Load tokenizer configuration from a JSON URL (tokenizer.json or vocab.json).
   */
  static async fromUrl(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load tokenizer from ${url}: ${response.status}`);
    const config = await response.json();
    if (config.model && config.model.vocab) {
      const specialTokens = {};
      if (config.added_tokens) {
        for (const t of config.added_tokens) {
          specialTokens[t.content] = t.id;
        }
      }
      return new _Tokenizer({
        vocab: config.model.vocab,
        merges: config.model.merges || [],
        specialTokens,
        maxLength: config.truncation?.max_length || 512,
        padTokenId: specialTokens["[PAD]"] ?? 0,
        unkTokenId: specialTokens["[UNK]"] ?? 100,
        clsTokenId: specialTokens["[CLS]"] ?? 101,
        sepTokenId: specialTokens["[SEP]"] ?? 102
      });
    }
    return new _Tokenizer({ vocab: config });
  }
  /**
   * Encode text into token IDs with attention mask.
   * Handles WordPiece tokenization for BERT-like models.
   */
  encode(text, options) {
    const addSpecial = options?.addSpecialTokens !== false;
    const maxLen = options?.maxLength || this.maxLength;
    const padding = options?.padding ?? false;
    const words = this.preTokenize(text);
    const tokenIds = [];
    if (addSpecial) {
      tokenIds.push(this.clsTokenId);
    }
    for (const word of words) {
      const subTokens = this.tokenizeWord(word);
      for (const subToken of subTokens) {
        if (tokenIds.length >= maxLen - (addSpecial ? 1 : 0)) break;
        tokenIds.push(subToken);
      }
    }
    if (addSpecial) {
      tokenIds.push(this.sepTokenId);
    }
    const truncated = tokenIds.slice(0, maxLen);
    if (truncated.length === 0) {
      const fallback = new Int32Array([this.padTokenId]);
      return {
        inputIds: fallback,
        attentionMask: new Int32Array([0])
      };
    }
    const attentionMask = new Int32Array(padding ? maxLen : truncated.length);
    attentionMask.fill(1, 0, truncated.length);
    const inputIds = new Int32Array(padding ? maxLen : truncated.length);
    inputIds.set(truncated);
    if (padding) {
      inputIds.fill(this.padTokenId, truncated.length);
    }
    return { inputIds, attentionMask };
  }
  /**
   * Decode token IDs back to text.
   */
  decode(tokens) {
    const tokenArray = tokens instanceof Int32Array ? Array.from(tokens) : tokens;
    const words = [];
    for (const id of tokenArray) {
      if (id === this.padTokenId || id === this.clsTokenId || id === this.sepTokenId) continue;
      const token = this.inverseVocab.get(id);
      if (!token) {
        words.push("[UNK]");
      } else if (token.startsWith("##")) {
        if (words.length > 0) {
          words[words.length - 1] += token.slice(2);
        } else {
          words.push(token.slice(2));
        }
      } else {
        words.push(token);
      }
    }
    return words.join(" ");
  }
  /**
   * Get vocabulary size.
   */
  get vocabSize() {
    return this.vocab.size;
  }
  /**
   * Pre-tokenize text: lowercase, split on whitespace and punctuation boundaries.
   */
  preTokenize(text) {
    const normalized = text.toLowerCase().trim();
    if (!normalized) return [];
    const tokens = [];
    let current = "";
    for (const char of normalized) {
      if (/\s/.test(char)) {
        if (current) tokens.push(current);
        current = "";
      } else if (/[.,!?;:'"()\[\]{}\-\/\\@#$%^&*~`|<>]/.test(char) || /\p{P}/u.test(char)) {
        if (current) tokens.push(current);
        tokens.push(char);
        current = "";
      } else {
        current += char;
      }
    }
    if (current) tokens.push(current);
    return tokens;
  }
  /**
   * Tokenize a single word using WordPiece algorithm.
   */
  tokenizeWord(word) {
    if (this.vocab.has(word)) {
      return [this.vocab.get(word)];
    }
    const tokens = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let found = false;
      while (start < end) {
        let substr = word.slice(start, end);
        if (start > 0) {
          substr = "##" + substr;
        }
        if (this.vocab.has(substr)) {
          tokens.push(this.vocab.get(substr));
          found = true;
          break;
        }
        end--;
      }
      if (!found) {
        tokens.push(this.unkTokenId);
        start++;
      } else {
        start = end;
      }
    }
    return tokens;
  }
  /**
   * Apply BPE merges to a word (for BPE-style tokenizers like GPT-2).
   */
  applyBPE(word) {
    const cached = this.bpeCache.get(word);
    if (cached) return cached;
    let pairs = this.getCharPairs(word.split(""));
    if (pairs.length === 0) return [word];
    let tokens = word.split("");
    for (const [first, second] of this.merges) {
      const newTokens = [];
      let i = 0;
      while (i < tokens.length) {
        const j = tokens.indexOf(first, i);
        if (j === -1) {
          newTokens.push(...tokens.slice(i));
          break;
        }
        newTokens.push(...tokens.slice(i, j));
        if (j < tokens.length - 1 && tokens[j + 1] === second) {
          newTokens.push(first + second);
          i = j + 2;
        } else {
          newTokens.push(tokens[j]);
          i = j + 1;
        }
      }
      tokens = newTokens;
    }
    this.bpeCache.set(word, tokens);
    return tokens;
  }
  getCharPairs(chars) {
    const pairs = [];
    for (let i = 0; i < chars.length - 1; i++) {
      pairs.push([chars[i], chars[i + 1]]);
    }
    return pairs;
  }
};

// src/image-processor.ts
var ImageProcessor = class {
  static imageDataToFloat32Array(imageData, options = {}) {
    const { normalize = true, layout = "NCHW" } = options;
    const { width, height, data } = imageData;
    const size = width * height;
    const float32Data = new Float32Array(size * 3);
    for (let i = 0; i < size; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const nr = normalize ? r / 255 : r;
      const ng = normalize ? g / 255 : g;
      const nb = normalize ? b / 255 : b;
      if (layout === "NCHW") {
        float32Data[i] = nr;
        float32Data[size + i] = ng;
        float32Data[2 * size + i] = nb;
      } else {
        float32Data[i * 3] = nr;
        float32Data[i * 3 + 1] = ng;
        float32Data[i * 3 + 2] = nb;
      }
    }
    return float32Data;
  }
};

// src/license-validator.ts
var LicenseValidator = class {
  static validate(options) {
    const key = options?.licenseKey || (typeof process !== "undefined" ? process.env.COMMERCIAL_LICENSE_KEY : void 0);
    const isDev = typeof window !== "undefined" ? window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" : typeof process !== "undefined" && process.env.NODE_ENV !== "production";
    if (isDev || options?.allowEval) {
      return true;
    }
    if (!key || !key.startsWith("BSL11-")) {
      console.warn(`
================================================================================
\u{1F512} COMMERCIAL USE WARNING \u2014 BUSINESS SOURCE LICENSE 1.1 REQUIRED
Product: EDGEINFER | Copyright (c) 2024-2026 Soumya Debnath

Production use of this software requires a valid paid commercial license key.
Unlicensed commercial deployment constitutes copyright infringement under DMCA \xA7 1201.

Purchase a commercial license key:
\u{1F4E7} Email: soumyadebnath1619@gmail.com
================================================================================
      `);
      return false;
    }
    return true;
  }
};
LicenseValidator.AUTHOR = "Soumya Debnath";
LicenseValidator.CONTACT = "soumyadebnath1619@gmail.com";

// src/edgeinfer.ts
var ort2 = __toESM(require("onnxruntime-web"));
var EdgeInfer = class _EdgeInfer {
  constructor(session, modelSize, provider, tokenizer) {
    this.tokenizer = null;
    this._provider = "wasm";
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
  static async load(modelUrl, options) {
    if (options?.licenseKey) {
      LicenseValidator.validate(options);
    }
    const cache = new ModelCache(options?.cacheName);
    const buffer = await cache.getModel(modelUrl, options?.forceDownload);
    return _EdgeInfer.fromBuffer(buffer, options);
  }
  /**
   * Create an EdgeInfer instance from a pre-loaded model buffer.
   */
  static async fromBuffer(buffer, options) {
    const { session, provider } = await RuntimeManager.createSession(
      buffer,
      options?.executionProviders
    );
    let tokenizer;
    if (options?.tokenizerUrl) {
      tokenizer = await Tokenizer.fromUrl(options.tokenizerUrl);
    } else if (options?.tokenizerConfig) {
      tokenizer = new Tokenizer(options.tokenizerConfig);
    }
    return new _EdgeInfer(session, buffer.byteLength, provider, tokenizer);
  }
  /**
   * Set or replace the tokenizer after model load.
   */
  setTokenizer(tokenizer) {
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
  async predict(inputs, shapes) {
    const tensorInputs = {};
    for (const [key, data] of Object.entries(inputs)) {
      let type;
      if (data instanceof Float32Array) {
        type = "float32";
      } else if (data instanceof BigInt64Array) {
        type = "int64";
      } else {
        type = "int32";
      }
      const shape = shapes?.[key] ?? [1, data.length];
      tensorInputs[key] = new ort2.Tensor(type, data, shape);
    }
    const results = await this.session.run(tensorInputs);
    const outputMap = {};
    for (const [key, tensor] of Object.entries(results)) {
      outputMap[key] = tensor.data;
    }
    return outputMap;
  }
  /**
   * Classify text using the loaded model.
   * Requires a tokenizer to be configured.
   */
  async classify(text) {
    this.requireTokenizer();
    const encoding = this.tokenizer.encode(text, { addSpecialTokens: true });
    const inputs = {
      "input_ids": encoding.inputIds,
      "attention_mask": encoding.attentionMask
    };
    if (this._inputNames.includes("token_type_ids")) {
      inputs["token_type_ids"] = new Int32Array(encoding.inputIds.length);
    }
    const result = await this.predict(inputs);
    const output = result[this._outputNames[0]];
    if (!output) return [];
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
  async embed(text) {
    this.requireTokenizer();
    const encoding = this.tokenizer.encode(text, { addSpecialTokens: true });
    const inputs = {
      "input_ids": encoding.inputIds,
      "attention_mask": encoding.attentionMask
    };
    if (this._inputNames.includes("token_type_ids")) {
      inputs["token_type_ids"] = new Int32Array(encoding.inputIds.length);
    }
    const result = await this.predict(inputs);
    const output = result[this._outputNames[0]];
    if (!output) return new Float32Array();
    return this.meanPool(output, encoding.attentionMask);
  }
  /**
   * Generate Matryoshka-truncated embeddings at specified dimension.
   * Supports variable-dimension MRL embeddings (64d → 256d → 768d).
   */
  async embedMatryoshka(text, dimension) {
    const fullEmbedding = await this.embed(text);
    if (dimension >= fullEmbedding.length) return fullEmbedding;
    const truncated = fullEmbedding.slice(0, dimension);
    return this.l2Normalize(truncated);
  }
  /**
   * Sentiment analysis.
   */
  async sentiment(text) {
    const results = await this.classify(text);
    if (results.length === 0) return { label: "unknown", score: 0 };
    const best = results[0];
    const labelMap = {
      "Class_0": "negative",
      "Class_1": "positive",
      "Class_2": "neutral"
    };
    return {
      label: labelMap[best.label] || best.label,
      score: best.score
    };
  }
  /**
   * Classify an image using a vision model.
   */
  async classifyImage(imageData, layout) {
    const effectiveLayout = layout ?? "NCHW";
    const tensorData = ImageProcessor.imageDataToFloat32Array(imageData, { layout: effectiveLayout });
    const inputName = this._inputNames.find(
      (n) => n === "input" || n === "pixel_values" || n === "image"
    ) || this._inputNames[0];
    const result = await this.predict(
      { [inputName]: tensorData },
      { [inputName]: _EdgeInfer.imageShape(imageData, effectiveLayout) }
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
  async detectObjects(imageData) {
    const tensorData = ImageProcessor.imageDataToFloat32Array(imageData);
    const inputName = this._inputNames.find(
      (n) => n === "input" || n === "pixel_values" || n === "image"
    ) || this._inputNames[0];
    const result = await this.predict(
      { [inputName]: tensorData },
      { [inputName]: _EdgeInfer.imageShape(imageData, "NCHW") }
    );
    const output = result[this._outputNames[0]];
    const detections = [];
    if (output) {
      for (let i = 0; i < output.length; i += 6) {
        const confidence = output[i + 4];
        if (confidence > 0.5) {
          detections.push({
            bbox: [output[i], output[i + 1], output[i + 2], output[i + 3]],
            score: confidence,
            label: `Object_${Math.round(output[i + 5])}`
          });
        }
      }
    }
    return detections.sort((a, b) => b.score - a.score);
  }
  /**
   * Check GPU capabilities and recommended configuration.
   */
  static async getCapabilities() {
    return RuntimeManager.detectCapabilities();
  }
  /**
   * Check if WebGPU is available for accelerated inference.
   */
  static async isWebGPUAvailable() {
    const caps = await RuntimeManager.detectCapabilities();
    return caps.hasWebGPU;
  }
  get inputNames() {
    return this._inputNames;
  }
  get outputNames() {
    return this._outputNames;
  }
  get modelSize() {
    return this._modelSize;
  }
  get executionProvider() {
    return this._provider;
  }
  /**
   * Release all resources (ONNX session, GPU memory).
   */
  dispose() {
    if (this.session) {
      this.session.release();
    }
    this.tokenizer = null;
    RuntimeManager.resetCache();
  }
  // --- Private Helpers ---
  /** ONNX tensor shape for a 3-channel image tensor in the given layout. */
  static imageShape(imageData, layout) {
    const { width, height } = imageData;
    return layout === "NCHW" ? [1, 3, height, width] : [1, height, width, 3];
  }
  requireTokenizer() {
    if (!this.tokenizer) {
      throw new Error(
        "EdgeInfer: No tokenizer configured. Either pass tokenizerUrl/tokenizerConfig in options, or call setTokenizer() before using text-based inference methods."
      );
    }
  }
  softmax(logits) {
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
  meanPool(embeddings, attentionMask) {
    const seqLen = attentionMask.length;
    const hiddenDim = embeddings.length / seqLen;
    if (hiddenDim < 1 || !Number.isInteger(hiddenDim)) {
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
  l2Normalize(vec) {
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
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  EdgeInfer,
  EventEmitter,
  ImageProcessor,
  ModelCache,
  RuntimeManager,
  Tokenizer
});
//# sourceMappingURL=index.js.map