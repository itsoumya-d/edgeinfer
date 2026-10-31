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

// src/download.ts
function checkDownloadAbort(signal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Download aborted", "AbortError");
}
async function withDownloadDeadline(operation, options = {}) {
  const timeout = options.downloadTimeoutMs ?? 12e4;
  if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 2147483647) {
    throw new RangeError("downloadTimeoutMs must be between 0 and 2147483647 milliseconds");
  }
  const controller = new AbortController();
  const abort = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) abort();
  else options.signal?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(() => controller.abort(new DOMException("Download deadline exceeded", "TimeoutError")), timeout);
  let onAbort = () => {
  };
  try {
    checkDownloadAbort(controller.signal);
    const cancelled = new Promise((_, reject) => {
      onAbort = () => reject(controller.signal.reason ?? new DOMException("Download aborted", "AbortError"));
      controller.signal.addEventListener("abort", onAbort, { once: true });
    });
    return await Promise.race([operation(controller.signal), cancelled]);
  } finally {
    clearTimeout(timer);
    controller.signal.removeEventListener("abort", onAbort);
    options.signal?.removeEventListener("abort", abort);
  }
}

// src/model-cache.ts
var ModelCache = class extends EventEmitter {
  constructor(cacheName = "edgeinfer-models") {
    super();
    this.cacheName = cacheName;
  }
  async getModel(url, forceDownload = false, options = {}) {
    return withDownloadDeadline(async (signal) => {
      const cache = typeof caches === "undefined" ? void 0 : await caches.open(this.cacheName);
      checkDownloadAbort(signal);
      const cached = !forceDownload && cache ? await cache.match(url) : void 0;
      checkDownloadAbort(signal);
      if (cached) {
        const buffer2 = await cached.arrayBuffer();
        checkDownloadAbort(signal);
        return buffer2;
      }
      this.emit("downloadStart", url);
      checkDownloadAbort(signal);
      const response = await fetch(url, { signal });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`EdgeInfer: failed to fetch model from ${url} - HTTP ${response.status} ${response.statusText}`);
      }
      if (response.status === 206 || response.headers.has("content-range")) {
        await response.body?.cancel();
        throw new Error("EdgeInfer: partial model responses are not supported");
      }
      const buffer = await response.arrayBuffer();
      checkDownloadAbort(signal);
      if (cache) await cache.put(url, new Response(buffer, { headers: response.headers }));
      checkDownloadAbort(signal);
      this.emit("downloadComplete", url);
      return buffer;
    }, options);
  }
};

// src/runtime-manager.ts
import * as ort from "onnxruntime-web";
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
    this.usesBPE = this.merges.length > 0;
    this.bpeSpaceMarker = config.bpeSpaceMarker ?? "";
    this.mergeRanks = /* @__PURE__ */ new Map();
    this.merges.forEach(([first, second], rank) => {
      this.mergeRanks.set(`${first} ${second}`, rank);
    });
    this.lowercase = config.lowercase ?? !this.usesBPE;
  }
  /**
   * Load tokenizer configuration from a JSON URL (tokenizer.json or vocab.json).
   */
  static async fromUrl(url, options = {}) {
    const config = await withDownloadDeadline(async (signal) => {
      const response = await fetch(url, { signal });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`Failed to load tokenizer from ${url}: ${response.status}`);
      }
      return response.json();
    }, options);
    if (config.model && config.model.vocab) {
      const specialTokens = {};
      if (config.added_tokens) {
        for (const t of config.added_tokens) {
          specialTokens[t.content] = t.id;
        }
      }
      return new _Tokenizer({
        vocab: config.model.vocab,
        lowercase: config.normalizer === void 0 ? void 0 : _Tokenizer.normalizerLowercase(config.normalizer),
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
  static normalizerLowercase(normalizer) {
    if (normalizer === void 0) return true;
    if (normalizer === null) return false;
    if (typeof normalizer !== "object") throw new Error("Unsupported tokenizer normalizer");
    const value = normalizer;
    if (value.type === "Lowercase") return true;
    if (value.type === "BertNormalizer" && typeof value.lowercase === "boolean" && value.strip_accents === false && value.clean_text === false && value.handle_chinese_chars === false) {
      return value.lowercase;
    }
    throw new Error("Unsupported tokenizer normalizer; only identity, Lowercase, or casing-only BertNormalizer is supported");
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
        attentionMask: new Int32Array([0]),
        inputIdsInt64: _Tokenizer.toInt64(fallback),
        attentionMaskInt64: new BigInt64Array([0n])
      };
    }
    const attentionMask = new Int32Array(padding ? maxLen : truncated.length);
    attentionMask.fill(1, 0, truncated.length);
    const inputIds = new Int32Array(padding ? maxLen : truncated.length);
    inputIds.set(truncated);
    if (padding) {
      inputIds.fill(this.padTokenId, truncated.length);
    }
    return {
      inputIds,
      attentionMask,
      inputIdsInt64: _Tokenizer.toInt64(inputIds),
      attentionMaskInt64: _Tokenizer.toInt64(attentionMask)
    };
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
      } else if (token.startsWith("\u2581") || token.startsWith("\u0120")) {
        words.push(token.slice(1));
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
   * Resolve a special token (e.g. `</s>`, `<|endoftext|>`) to its ID, whether it
   * was declared in `specialTokens` or is present in the main vocabulary.
   */
  getSpecialTokenId(token) {
    return this.specialTokens.get(token) ?? this.vocab.get(token);
  }
  /**
   * Pre-tokenize text: lowercase, split on whitespace and punctuation boundaries.
   */
  preTokenize(text) {
    const normalized = (this.lowercase ? text.toLowerCase() : text).trim();
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
   * Tokenize a single word using the tokenizer's model type: BPE when a merge
   * table is present, WordPiece otherwise.
   */
  tokenizeWord(word) {
    if (this.usesBPE) return this.tokenizeWordBPE(word);
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
   * Apply the merge table to a word, then map the resulting pieces to IDs.
   * Unknown pieces fall back to the unknown token rather than throwing, which
   * matches ONNX Runtime's text-tokenizer behaviour.
   */
  tokenizeWordBPE(word) {
    const whole = this.vocab.get(
      this.bpeSpaceMarker ? this.bpeSpaceMarker + word : word
    );
    if (whole !== void 0) return [whole];
    const pieces = this.applyBPE(word);
    const ids = [];
    for (let i = 0; i < pieces.length; i++) {
      const piece = i === 0 && this.bpeSpaceMarker ? this.bpeSpaceMarker + pieces[i] : pieces[i];
      const id = this.vocab.get(piece);
      if (id !== void 0) {
        ids.push(id);
        continue;
      }
      ids.push(this.unkTokenId);
    }
    return ids;
  }
  /**
   * Apply BPE merges to a word (for BPE-style tokenizers like GPT-2/Llama).
   * Merge rules are applied in ascending rank order until no adjacent pair is
   * present in the merge table; caching is keyed on the original word.
   */
  applyBPE(word) {
    const cached = this.bpeCache.get(word);
    if (cached) return cached;
    let tokens = word.split("");
    if (tokens.length <= 1 || this.mergeRanks.size === 0) {
      this.bpeCache.set(word, tokens);
      return tokens;
    }
    while (tokens.length > 1) {
      let bestRank = Number.POSITIVE_INFINITY;
      for (let i = 0; i < tokens.length - 1; i++) {
        const rank = this.mergeRanks.get(`${tokens[i]} ${tokens[i + 1]}`);
        if (rank !== void 0 && rank < bestRank) bestRank = rank;
      }
      if (bestRank === Number.POSITIVE_INFINITY) break;
      const merged = [];
      for (let i = 0; i < tokens.length; i++) {
        if (i < tokens.length - 1 && this.mergeRanks.get(`${tokens[i]} ${tokens[i + 1]}`) === bestRank) {
          merged.push(tokens[i] + tokens[i + 1]);
          i++;
        } else {
          merged.push(tokens[i]);
        }
      }
      if (merged.length === tokens.length) break;
      tokens = merged;
    }
    this.bpeCache.set(word, tokens);
    return tokens;
  }
  static toInt64(values) {
    const out = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = BigInt(values[i]);
    return out;
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

// src/edgeinfer.ts
import * as ort2 from "onnxruntime-web";
var EdgeInfer = class _EdgeInfer {
  constructor(session, modelSize, provider, int64Inputs, tokenizer) {
    this.tokenizer = null;
    this._provider = "wasm";
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
  static async load(modelUrl, options) {
    const cache = new ModelCache(options?.cacheName);
    const buffer = await cache.getModel(modelUrl, options?.forceDownload, options);
    return _EdgeInfer.fromBuffer(buffer, options);
  }
  /**
   * Create an EdgeInfer instance from a pre-loaded model buffer.
   */
  static async fromBuffer(buffer, options) {
    checkDownloadAbort(options?.signal);
    const { session, provider } = await RuntimeManager.createSession(
      buffer,
      options?.executionProviders
    );
    try {
      checkDownloadAbort(options?.signal);
      let tokenizer;
      if (options?.tokenizerUrl) {
        tokenizer = await Tokenizer.fromUrl(options.tokenizerUrl, options);
      } else if (options?.tokenizerConfig) {
        tokenizer = new Tokenizer(options.tokenizerConfig);
      }
      checkDownloadAbort(options?.signal);
      return new _EdgeInfer(session, buffer.byteLength, provider, options?.int64Inputs ?? false, tokenizer);
    } catch (error) {
      try {
        await session.release();
      } catch {
      }
      throw error;
    }
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
    const results = await this.runInference(inputs, shapes);
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
    const inputs = this.textInputs(encoding);
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
    const inputs = this.textInputs(encoding);
    const result = await this.runInference(inputs);
    const output = result[this._outputNames[0]];
    if (!output) return new Float32Array();
    if (output.type !== "float32") {
      throw new Error("EdgeInfer: Embedding output must be a float32 tensor.");
    }
    return this.poolEmbedding(output.data, output.dims, encoding.attentionMask);
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
   * Greedy or temperature-sampled autoregressive generation.
   *
   * Decoder models that expose `present.*` outputs and `past_key_values.*`
   * inputs reuse those tensors across steps (KV-cache). Models without cache
   * inputs are re-run over the full sequence each step.
   */
  async generate(prompt, options = {}) {
    this.requireTokenizer();
    const maxNewTokens = Math.max(0, options.maxNewTokens ?? 64);
    const temperature = options.temperature ?? 0;
    const topK = options.topK ?? 0;
    const tokenizer = this.tokenizer;
    const eosTokenId = options.eosTokenId ?? tokenizer.getSpecialTokenId("</s>") ?? tokenizer.getSpecialTokenId("<|endoftext|>");
    const encoding = tokenizer.encode(prompt, { addSpecialTokens: true });
    const promptIds = Array.from(encoding.inputIds);
    const int64 = this.useInt64Inputs();
    const makeIds = (values) => int64 ? BigInt64Array.from(values, (v) => BigInt(v)) : Int32Array.from(values);
    const dtype = int64 ? "int64" : "int32";
    let current = promptIds;
    let past = {};
    const generated = [];
    for (let step = 0; step < maxNewTokens; step++) {
      const totalLength = promptIds.length + generated.length;
      const inputs = {
        input_ids: new ort2.Tensor(dtype, makeIds(current), [1, current.length]),
        attention_mask: new ort2.Tensor(dtype, makeIds(new Array(totalLength).fill(1)), [1, totalLength])
      };
      if (this._inputNames.includes("position_ids")) {
        const offset = totalLength - current.length;
        inputs.position_ids = new ort2.Tensor(
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
      if (eosTokenId !== void 0 && next === eosTokenId) break;
      const nextPast = {};
      for (const outputName of this._outputNames) {
        const inputName = _EdgeInfer.pastInputName(outputName);
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
      tokenIds: generated
    };
  }
  /** Pick the last-token logits from the first output that looks like logits. */
  findLogits(results) {
    const exact = results["logits"];
    if (exact) return exact;
    for (const name of Object.keys(results)) {
      if (name.toLowerCase().includes("logits")) return results[name];
    }
    return results[this._outputNames[0]];
  }
  /** Argmax (temperature 0) or temperature/top-k sampling over last-token logits. */
  pickNextToken(logits, temperature, topK) {
    const dims = logits.dims;
    const vocabSize = dims[dims.length - 1];
    const seqLen = dims.length >= 3 ? dims[dims.length - 2] : 1;
    const data = logits.data;
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
    const candidates = [];
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
    let sample2 = Math.random() * sum;
    for (const candidate of candidates) {
      sample2 -= candidate.prob;
      if (sample2 <= 0) return candidate.id;
    }
    return candidates[candidates.length - 1].id;
  }
  static pastInputName(outputName) {
    if (!outputName.startsWith("present")) return null;
    return outputName.replace(/^present/, "past_key_values");
  }
  useInt64Inputs() {
    if (this._int64Inputs) return true;
    const metadata = this.session.inputMetadata;
    return metadata?.["input_ids"]?.type === "int64";
  }
  /** Build input_ids/attention_mask tensors in the dtype the model declares. */
  textInputs(encoding) {
    const int64 = this.useInt64Inputs();
    const inputs = {
      input_ids: int64 ? encoding.inputIdsInt64 : encoding.inputIds,
      attention_mask: int64 ? encoding.attentionMaskInt64 : encoding.attentionMask
    };
    if (this._inputNames.includes("token_type_ids")) {
      inputs["token_type_ids"] = int64 ? new BigInt64Array(encoding.inputIds.length) : new Int32Array(encoding.inputIds.length);
    }
    return inputs;
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
  /** Run without discarding the output tensor dimensions needed by embed(). */
  async runInference(inputs, shapes) {
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
    return this.session.run(tensorInputs);
  }
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
  poolEmbedding(embeddings, dims, attentionMask) {
    if ((dims.length === 1 || dims.length === 2 && dims[0] === 1) && dims[dims.length - 1] > 0) {
      return this.l2Normalize(embeddings);
    }
    if (dims.length !== 3 || dims[0] !== 1 || dims[1] < 1 || dims[2] < 1) {
      throw new Error(
        `EdgeInfer: Unsupported embedding output shape [${dims.join(", ")}]. Expected [hidden], [1, hidden], or [1, sequence, hidden].`
      );
    }
    const [, seqLen, hiddenDim] = dims;
    if (seqLen !== attentionMask.length) {
      throw new Error(
        `EdgeInfer: Embedding sequence length ${seqLen} does not match attention mask length ${attentionMask.length}.`
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

// src/matryoshka.ts
function l2Normalize(vector, dimension = vector.length) {
  const output = new Float32Array(dimension);
  if (dimension > vector.length) {
    throw new Error(`Cannot normalize ${dimension} dims from a ${vector.length}-dim vector`);
  }
  let norm = 0;
  for (let i = 0; i < dimension; i++) norm += vector[i] * vector[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return output;
  for (let i = 0; i < dimension; i++) output[i] = vector[i] / norm;
  return output;
}
function truncateAndNormalize(vector, dimension) {
  if (!Number.isInteger(dimension) || dimension < 1) {
    throw new Error("dimension must be a positive integer");
  }
  return l2Normalize(vector, dimension);
}
function quantizeInt8(vector, dimension = vector.length) {
  let maxAbs = 0;
  for (let i = 0; i < dimension; i++) {
    const value = Math.abs(vector[i]);
    if (value > maxAbs) maxAbs = value;
  }
  const scale = maxAbs > 0 ? maxAbs / 127 : 1;
  const data = new Int8Array(dimension);
  for (let i = 0; i < dimension; i++) {
    const value = Math.round(vector[i] / scale);
    data[i] = value > 127 ? 127 : value < -127 ? -127 : value;
  }
  return { data, scale };
}
function dequantizeInt8(data, scale) {
  const output = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) output[i] = data[i] * scale;
  return output;
}
function dot(a, b, dimension) {
  let sum = 0;
  for (let i = 0; i < dimension; i++) sum += a[i] * b[i];
  return sum;
}
var TopKHeap = class {
  constructor(k) {
    this.k = k;
    this.items = [];
    if (k < 1) throw new Error("top-k must be positive");
  }
  push(index, score) {
    if (this.items.length < this.k) {
      this.items.push({ index, score });
      if (this.items.length === this.k) this.heapify();
      return;
    }
    if (score > this.items[0].score) {
      this.items[0] = { index, score };
      this.siftDown(0);
    }
  }
  heapify() {
    for (let i = Math.floor(this.items.length / 2) - 1; i >= 0; i--) this.siftDown(i);
  }
  siftDown(start) {
    let index = start;
    const length = this.items.length;
    for (; ; ) {
      const left = index * 2 + 1;
      const right = left + 1;
      let smallest = index;
      if (left < length && this.items[left].score < this.items[smallest].score) smallest = left;
      if (right < length && this.items[right].score < this.items[smallest].score) smallest = right;
      if (smallest === index) break;
      [this.items[index], this.items[smallest]] = [this.items[smallest], this.items[index]];
      index = smallest;
    }
  }
  values() {
    return [...this.items].sort((a, b) => b.score - a.score);
  }
};
var MemoryVectorStore = class {
  constructor() {
    this.records = [];
  }
  async add(records) {
    this.records.push(...records);
  }
  async getAll() {
    return this.records.slice();
  }
  async clear() {
    this.records = [];
  }
  async count() {
    return this.records.length;
  }
};
var IndexedDbVectorStore = class {
  constructor(dbName = "edgeinfer-matryoshka", storeName = "vectors") {
    this.dbName = dbName;
    this.storeName = storeName;
    this.dbPromise = null;
  }
  open() {
    if (this.dbPromise) return this.dbPromise;
    if (typeof indexedDB === "undefined") {
      return Promise.reject(new Error("IndexedDB is unavailable in this environment"));
    }
    this.dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: "id" });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
    });
    return this.dbPromise;
  }
  async add(records) {
    const db = await this.open();
    await new Promise((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readwrite");
      const store = transaction.objectStore(this.storeName);
      for (const record of records) {
        const stored = {
          id: record.id,
          text: record.text,
          metadata: record.metadata,
          embedding: record.embedding.slice().buffer
        };
        store.put(stored);
      }
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB write failed"));
      transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB write aborted"));
    });
  }
  async getAll() {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.getAll();
      request.onsuccess = () => {
        const records = request.result.map((stored) => ({
          id: stored.id,
          text: stored.text,
          metadata: stored.metadata,
          embedding: new Float32Array(stored.embedding)
        }));
        resolve(records);
      };
      request.onerror = () => reject(request.error ?? new Error("IndexedDB read failed"));
    });
  }
  async clear() {
    const db = await this.open();
    await new Promise((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readwrite");
      transaction.objectStore(this.storeName).clear();
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB clear failed"));
    });
  }
  async count() {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(this.storeName, "readonly");
      const request = transaction.objectStore(this.storeName).count();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB count failed"));
    });
  }
  close() {
    void this.dbPromise?.then((db) => db.close());
    this.dbPromise = null;
  }
};
var MatryoshkaIndex = class _MatryoshkaIndex {
  constructor(options = {}) {
    this.ids = [];
    this.texts = [];
    this.metadata = [];
    this.full = new Float32Array(0);
    this.coarse = new Int8Array(0);
    this.scales = new Float32Array(0);
    this.size = 0;
    this.dimensions = options.dimensions ?? 768;
    this.coarseDimension = options.coarseDimension ?? 64;
    this.refineTopK = options.refineTopK ?? 50;
    this.store = options.store;
    if (this.coarseDimension > this.dimensions) {
      throw new Error("coarseDimension cannot exceed dimensions");
    }
  }
  get count() {
    return this.size;
  }
  async build(records, { persist = true } = {}) {
    if (records.length === 0) throw new Error("At least one record is required");
    for (const record of records) {
      if (record.embedding.length !== this.dimensions) {
        throw new Error(
          `Record ${record.id} has ${record.embedding.length} dims; expected ${this.dimensions}`
        );
      }
    }
    this.ids = new Array(records.length);
    this.texts = new Array(records.length);
    this.metadata = new Array(records.length);
    this.full = new Float32Array(records.length * this.dimensions);
    this.coarse = new Int8Array(records.length * this.coarseDimension);
    this.scales = new Float32Array(records.length);
    this.size = records.length;
    for (let row = 0; row < records.length; row++) {
      const record = records[row];
      const normalized = l2Normalize(record.embedding, this.dimensions);
      this.ids[row] = record.id;
      this.texts[row] = record.text;
      this.metadata[row] = record.metadata;
      this.full.set(normalized, row * this.dimensions);
      const quantized = quantizeInt8(normalized, this.coarseDimension);
      this.coarse.set(quantized.data, row * this.coarseDimension);
      this.scales[row] = quantized.scale;
    }
    if (persist && this.store) {
      await this.store.clear();
      const normalizedRecords = records.map((record, row) => ({
        ...record,
        embedding: this.full.slice(row * this.dimensions, (row + 1) * this.dimensions)
      }));
      await this.store.add(normalizedRecords);
    }
  }
  static async load(store, options = {}) {
    const index = new _MatryoshkaIndex({ ...options, store });
    const records = await store.getAll();
    await index.build(records, { persist: false });
    return index;
  }
  search(query, options = {}) {
    if (this.size === 0) throw new Error("Index is empty; call build() first");
    const topK = options.topK ?? 10;
    const refine = options.refine ?? true;
    const coarseDimension = options.coarseDimension ?? this.coarseDimension;
    if (coarseDimension > this.coarseDimension) {
      throw new Error(
        `coarseDimension ${coarseDimension} exceeds the indexed stride ${this.coarseDimension}`
      );
    }
    const queryFull = l2Normalize(query, this.dimensions);
    const queryCoarse = quantizeInt8(queryFull, coarseDimension);
    const refineTopK = Math.max(
      topK,
      Math.min(this.refineTopK, options.refineTopK ?? this.refineTopK, this.size)
    );
    const heap = new TopKHeap(refineTopK);
    for (let row = 0; row < this.size; row++) {
      const offset = row * this.coarseDimension;
      let sum = 0;
      for (let i = 0; i < coarseDimension; i++) {
        sum += queryCoarse.data[i] * this.coarse[offset + i];
      }
      heap.push(row, sum * this.scales[row]);
    }
    let candidates = heap.values().map((candidate) => ({ ...candidate, coarseScore: candidate.score }));
    if (refine) {
      candidates = candidates.map((candidate) => ({
        ...candidate,
        score: dot(
          queryFull,
          this.full.subarray(
            candidate.index * this.dimensions,
            (candidate.index + 1) * this.dimensions
          ),
          this.dimensions
        )
      })).sort((a, b) => b.score - a.score);
    }
    return candidates.slice(0, topK).map((candidate) => ({
      id: this.ids[candidate.index],
      text: this.texts[candidate.index],
      metadata: this.metadata[candidate.index],
      score: candidate.score,
      coarseScore: candidate.coarseScore
    }));
  }
  exactSearchAtDimension(query, dimension, topK = 10) {
    if (dimension > this.dimensions) throw new Error("dimension exceeds index dimensions");
    const queryNormalized = truncateAndNormalize(query, dimension);
    const scored = [];
    for (let row = 0; row < this.size; row++) {
      const vector = truncateAndNormalize(
        this.full.subarray(row * this.dimensions, (row + 1) * this.dimensions),
        dimension
      );
      scored.push({ index: row, score: dot(queryNormalized, vector, dimension) });
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
  }
  recallAtK(queries, groundTruth, dimension, topK = 10) {
    let hits = 0;
    let total = 0;
    for (let q = 0; q < queries.length; q++) {
      const expected = new Set(groundTruth[q].slice(0, topK));
      const candidates = this.exactSearchAtDimension(queries[q], dimension, topK);
      for (const candidate of candidates) {
        if (expected.has(this.ids[candidate.index])) hits++;
      }
      total += Math.min(topK, expected.size);
    }
    return total === 0 ? 0 : hits / total;
  }
  get idList() {
    return this.ids;
  }
};
async function embedMatryoshka(text, dimension, embed) {
  const full = await embed(text);
  return truncateAndNormalize(full, dimension);
}
function createMatryoshkaEmbedder(embed) {
  return (text, dimension = 768) => embedMatryoshka(text, dimension, embed);
}

// src/speculative.ts
var SpeculativeDecoder = class {
  constructor(options) {
    if (options.draft.vocabSize !== options.target.vocabSize) {
      throw new Error("Draft and target models must share a vocabulary size");
    }
    const gamma = options.gamma ?? 5;
    if (!Number.isInteger(gamma) || gamma < 1) {
      throw new Error("gamma must be a positive integer");
    }
    this.draft = options.draft;
    this.target = options.target;
    this.gamma = gamma;
    this.rng = options.rng ?? Math.random;
    this.eosTokenId = options.eosTokenId;
  }
  async generate(prompt, maxNewTokens = 64, { temperature = 1 } = {}) {
    if (prompt.length === 0) throw new Error("prompt must not be empty");
    if (maxNewTokens < 1) throw new Error("maxNewTokens must be positive");
    if (temperature <= 0) throw new Error("temperature must be positive");
    const started = performance.now();
    let draftState = await this.draft.prefill(prompt);
    let targetState = await this.target.prefill(prompt);
    const output = [];
    let draftTokens = 0;
    let acceptedTokens = 0;
    let bonusTokens = 0;
    let rejectedTokens = 0;
    let targetCalls = 0;
    let draftCalls = 0;
    while (output.length < maxNewTokens) {
      const gamma = Math.min(this.gamma, maxNewTokens - output.length);
      const draft = await this.draftBatch(draftState, gamma, temperature, draftCalls);
      draftState = draft.state;
      draftCalls = draft.calls;
      draftTokens += draft.tokens.length;
      const prefixLength = targetState.tokens.length;
      const verification = await this.target.decode(targetState, draft.tokens);
      targetCalls++;
      targetState = verification.state;
      let rejected = false;
      for (let i = 0; i < draft.tokens.length; i++) {
        const token = draft.tokens[i];
        const q = draft.probabilities[i][token];
        const p = probability(verification.logits[i], token, temperature);
        if (this.rng() < Math.min(1, p / q)) {
          output.push(token);
          acceptedTokens++;
          if (token === this.eosTokenId || output.length >= maxNewTokens) break;
          continue;
        }
        const residual = residualDistribution(
          verification.logits[i],
          draft.probabilities[i],
          temperature
        );
        const replacement = sample(residual, this.rng());
        output.push(replacement);
        rejectedTokens++;
        rejected = true;
        const keep = prefixLength + i;
        targetState = this.target.truncate(targetState, keep);
        draftState = this.draft.truncate(draftState, keep);
        const targetAdvance2 = await this.target.decode(targetState, [replacement]);
        targetCalls++;
        targetState = targetAdvance2.state;
        const draftAdvance2 = await this.draft.decode(draftState, [replacement]);
        draftCalls++;
        draftState = draftAdvance2.state;
        break;
      }
      if (rejected) continue;
      if (output.length >= maxNewTokens || output[output.length - 1] === this.eosTokenId) break;
      const bonusDistribution = softmax(
        verification.logits[draft.tokens.length],
        temperature
      );
      const bonus = sample(bonusDistribution, this.rng());
      output.push(bonus);
      bonusTokens++;
      const targetAdvance = await this.target.decode(targetState, [bonus]);
      targetCalls++;
      targetState = targetAdvance.state;
      const draftAdvance = await this.draft.decode(draftState, [bonus]);
      draftCalls++;
      draftState = draftAdvance.state;
      if (bonus === this.eosTokenId) break;
    }
    const elapsedMs = performance.now() - started;
    const stats = {
      outputTokens: output.length,
      draftTokens,
      acceptedTokens,
      bonusTokens,
      rejectedTokens,
      targetCalls,
      draftCalls,
      acceptanceRate: draftTokens === 0 ? 0 : acceptedTokens / draftTokens,
      targetCallsPerToken: output.length === 0 ? 0 : targetCalls / output.length,
      elapsedMs,
      tokensPerSecond: elapsedMs === 0 ? 0 : output.length / elapsedMs * 1e3
    };
    return { tokens: output, stats };
  }
  async draftBatch(state, gamma, temperature, calls) {
    let current = state;
    const tokens = [];
    const probabilities = [];
    const first = await this.draft.decode(current, []);
    calls++;
    let nextDistribution = first.logits[0];
    for (let i = 0; i < gamma; i++) {
      const distribution = softmax(nextDistribution, temperature);
      const token = sample(distribution, this.rng());
      tokens.push(token);
      probabilities.push(distribution);
      const advanced = await this.draft.decode(current, [token]);
      calls++;
      current = advanced.state;
      nextDistribution = advanced.logits[advanced.logits.length - 1];
    }
    return { state: current, tokens, probabilities, calls };
  }
};
function softmax(logits, temperature = 1) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const output = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp((logits[i] - max) / temperature);
    output[i] = value;
    sum += value;
  }
  if (sum === 0) return output;
  for (let i = 0; i < output.length; i++) output[i] /= sum;
  return output;
}
function probability(logits, token, temperature = 1) {
  return softmax(logits, temperature)[token] ?? 0;
}
function sample(distribution, random) {
  let cumulative = 0;
  for (let i = 0; i < distribution.length; i++) {
    cumulative += distribution[i];
    if (random <= cumulative) return i;
  }
  return distribution.length - 1;
}
function residualDistribution(targetLogits, draftProbabilities, temperature = 1) {
  const p = softmax(targetLogits, temperature);
  const q = draftProbabilities;
  const residual = new Float32Array(p.length);
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    const value = Math.max(0, p[i] - q[i]);
    residual[i] = value;
    sum += value;
  }
  if (sum === 0) return p;
  for (let i = 0; i < residual.length; i++) residual[i] /= sum;
  return residual;
}
var MockLanguageModel = class {
  constructor(policy, vocabSize, latencyMs = 0) {
    this.policy = policy;
    this.vocabSize = vocabSize;
    this.latencyMs = latencyMs;
  }
  async prefill(tokens) {
    await delay(this.latencyMs);
    return { tokens: tokens.slice() };
  }
  async decode(state, tokens) {
    await delay(this.latencyMs);
    const logits = [];
    for (let i = 0; i <= tokens.length; i++) {
      logits.push(this.policy([...state.tokens, ...tokens.slice(0, i)]));
    }
    return { state: { tokens: [...state.tokens, ...tokens] }, logits };
  }
  truncate(state, length) {
    return { ...state, tokens: state.tokens.slice(0, length), cache: void 0 };
  }
};
function delay(ms) {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// src/onnx-lm.ts
import * as ort3 from "onnxruntime-web";
var OnnxCausalLM = class {
  constructor(session, vocabSize, options = {}) {
    this.session = session;
    if (!Number.isInteger(vocabSize) || vocabSize < 1) {
      throw new Error("vocabSize must be a positive integer");
    }
    this.vocabSize = vocabSize;
    this.inputIdsName = options.inputIdsName ?? "input_ids";
    this.attentionMaskName = options.attentionMaskName ?? "attention_mask";
    this.positionIdsName = options.positionIdsName;
    this.logitsName = options.logitsName ?? "logits";
  }
  async prefill(tokens) {
    if (tokens.length === 0) throw new Error("prefill requires a non-empty prompt");
    return { tokens: tokens.slice() };
  }
  async decode(state, tokens) {
    const all = [...state.tokens, ...tokens];
    if (all.length === 0) throw new Error("decode requires at least one token");
    if (state.tokens.length === 0) throw new Error("decode requires a prefilled state");
    const feeds = {
      [this.inputIdsName]: new ort3.Tensor(
        "int64",
        BigInt64Array.from(all, (token) => BigInt(token)),
        [1, all.length]
      ),
      [this.attentionMaskName]: new ort3.Tensor(
        "int64",
        BigInt64Array.from({ length: all.length }, () => 1n),
        [1, all.length]
      )
    };
    if (this.positionIdsName) {
      feeds[this.positionIdsName] = new ort3.Tensor(
        "int64",
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
    const data = tensor.data;
    const seq = all.length;
    const expected = seq * vocab;
    if (data.length < expected) {
      throw new Error(`ONNX logits tensor is too small (${data.length} < ${expected})`);
    }
    const start = state.tokens.length - 1;
    const logits = [];
    for (let i = 0; i <= tokens.length; i++) {
      const position = start + i;
      const offset = dims.length === 3 ? position * vocab : position * vocab;
      logits.push(new Float32Array(data.subarray(offset, offset + vocab)));
    }
    return { state: { tokens: all }, logits };
  }
  truncate(state, length) {
    return { tokens: state.tokens.slice(0, length) };
  }
};
export {
  EdgeInfer,
  EventEmitter,
  ImageProcessor,
  IndexedDbVectorStore,
  MatryoshkaIndex,
  MemoryVectorStore,
  MockLanguageModel,
  ModelCache,
  OnnxCausalLM,
  RuntimeManager,
  SpeculativeDecoder,
  Tokenizer,
  createMatryoshkaEmbedder,
  dequantizeInt8,
  embedMatryoshka,
  l2Normalize,
  probability,
  quantizeInt8,
  residualDistribution,
  sample,
  softmax,
  truncateAndNormalize
};
//# sourceMappingURL=index.mjs.map