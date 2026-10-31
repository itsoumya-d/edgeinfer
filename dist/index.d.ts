import * as ort from 'onnxruntime-web';

interface ModelOptions {
    executionProviders?: string[];
    cacheName?: string;
    forceDownload?: boolean;
}
interface ClassificationResult {
    label: string;
    score: number;
}
interface Detection {
    bbox: [number, number, number, number];
    label: string;
    score: number;
}

declare class EventEmitter {
    private listeners;
    on(event: string, callback: Function): void;
    emit(event: string, ...args: any[]): void;
}

/** Per-transfer options; covers response headers and the entire response body. */
interface DownloadOptions {
    signal?: AbortSignal;
    /** Default 120 seconds per model/tokenizer transfer, not an inference timeout. */
    downloadTimeoutMs?: number;
}

declare class ModelCache extends EventEmitter {
    private cacheName;
    constructor(cacheName?: string);
    getModel(url: string, forceDownload?: boolean, options?: DownloadOptions): Promise<ArrayBuffer>;
}

interface VRAMProbeResult {
    available: boolean;
    /** Largest single storage buffer the device actually allocated. */
    maxAllocationBytes: number;
    testedAllocations: number[];
    failedAllocationBytes: number | null;
    adapterLimits: {
        maxBufferSize: number;
        maxStorageBufferBindingSize: number;
    };
    deviceCreated: boolean;
    probe: 'empirical' | 'limits' | 'unavailable';
}
interface VRAMProbeOptions {
    startBytes?: number;
    maxBytes?: number;
    stepBytes?: number;
}
interface GPUCapabilities {
    provider: string;
    hasWebGPU: boolean;
    hasWebNN: boolean;
    /**
     * NOT an amount of video memory. This is the largest single GPU storage
     * buffer allocated by the empirical probe, bounded by the adapter's
     * `maxBufferSize`. Browser baselines are commonly 256 MiB regardless of the
     * physical card, so treat this as a maximum-tensor-size budget, not as VRAM.
     */
    estimatedVRAM: number;
    /** Raw adapter limits, kept separate from the empirical probe. */
    adapterLimits: {
        maxBufferSize: number;
        maxStorageBufferBindingSize: number;
    } | null;
    /** Evidence for how `estimatedVRAM` was obtained. */
    vramProbe: VRAMProbeResult | null;
    recommendedQuantization: 'fp32' | 'fp16' | 'int8' | 'int4';
}
/**
 * RuntimeManager handles execution provider detection and ONNX session creation.
 * Implements a smart fallback chain: WebGPU → WebNN → WASM
 * with automatic quantization level recommendation based on available GPU memory.
 */
declare class RuntimeManager {
    private static cachedCapabilities;
    /**
     * Detect GPU capabilities and recommend the best execution configuration.
     */
    static detectCapabilities(): Promise<GPUCapabilities>;
    /**
     * Get the best available execution provider with smart fallback.
     * Priority: WebGPU → WebNN → WASM
     */
    static getBestExecutionProvider(): Promise<string>;
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
    static createSession(modelBuffer: ArrayBuffer, providers?: string[]): Promise<{
        session: ort.InferenceSession;
        provider: string;
    }>;
    /**
     * Empirically probe the largest single storage buffer the adapter's device
     * will allocate. Allocates and destroys one buffer at a time; never relies on
     * `maxBufferSize` as if it were physical VRAM.
     */
    static probeVRAM(adapter: any, options?: VRAMProbeOptions): Promise<VRAMProbeResult>;
    /**
     * Resolve `promise`, or resolve to `null` if it has not settled within `ms`.
     * Used to stop a hung GPU driver from blocking model loading forever.
     */
    private static withTimeout;
    /**
     * Estimate if a model of given size (bytes) can fit in available VRAM.
     */
    static canFitModel(modelSizeBytes: number): Promise<boolean>;
    /**
     * Reset cached capabilities (useful for testing or re-detection).
     */
    static resetCache(): void;
}

interface TokenizerConfig {
    vocab: Record<string, number>;
    /** Keep the legacy uncased default; set false for cased WordPiece vocabularies. */
    lowercase?: boolean;
    /**
     * BPE merge rules, either `"a b"` (older HuggingFace tokenizer.json) or
     * `["a", "b"]` (newer tokenizer.json). Accepted and normalised. When present,
     * `encode()` applies these merges and preserves case by default; otherwise it
     * uses WordPiece.
     */
    merges?: Array<string | [string, string]>;
    specialTokens?: Record<string, number>;
    maxLength?: number;
    padTokenId?: number;
    unkTokenId?: number;
    clsTokenId?: number;
    sepTokenId?: number;
    /**
     * Optional marker prepended to the first BPE piece of each word, e.g. `"▁"`
     * for SentencePiece vocabularies or `"Ġ"` for byte-level BPE vocabularies.
     * Without it, pieces are looked up exactly as produced by the merge table.
     */
    bpeSpaceMarker?: string;
}
interface EncodingResult {
    inputIds: Int32Array;
    attentionMask: Int32Array;
    tokenTypeIds?: Int32Array;
    /** int64 view for ONNX models declared as `tensor(int64)`. */
    inputIdsInt64: BigInt64Array;
    attentionMaskInt64: BigInt64Array;
    tokenTypeIdsInt64?: BigInt64Array;
}
/**
 * Production BPE/WordPiece tokenizer for ONNX model inference.
 * Supports loading vocabularies from JSON, encoding text to token IDs,
 * and decoding token IDs back to text.
 */
declare class Tokenizer {
    private lowercase;
    private vocab;
    private inverseVocab;
    private merges;
    private specialTokens;
    private maxLength;
    private padTokenId;
    private unkTokenId;
    private clsTokenId;
    private sepTokenId;
    private bpeCache;
    private mergeRanks;
    private usesBPE;
    private bpeSpaceMarker;
    constructor(config: TokenizerConfig);
    /**
     * Load tokenizer configuration from a JSON URL (tokenizer.json or vocab.json).
     */
    static fromUrl(url: string, options?: DownloadOptions): Promise<Tokenizer>;
    private static normalizerLowercase;
    /**
     * Encode text into token IDs with attention mask.
     * Handles WordPiece tokenization for BERT-like models.
     */
    encode(text: string, options?: {
        addSpecialTokens?: boolean;
        maxLength?: number;
        padding?: boolean;
    }): EncodingResult;
    /**
     * Decode token IDs back to text.
     */
    decode(tokens: Int32Array | number[]): string;
    /**
     * Get vocabulary size.
     */
    get vocabSize(): number;
    /**
     * Resolve a special token (e.g. `</s>`, `<|endoftext|>`) to its ID, whether it
     * was declared in `specialTokens` or is present in the main vocabulary.
     */
    getSpecialTokenId(token: string): number | undefined;
    /**
     * Pre-tokenize text: lowercase, split on whitespace and punctuation boundaries.
     */
    private preTokenize;
    /**
     * Tokenize a single word using the tokenizer's model type: BPE when a merge
     * table is present, WordPiece otherwise.
     */
    private tokenizeWord;
    /**
     * Apply the merge table to a word, then map the resulting pieces to IDs.
     * Unknown pieces fall back to the unknown token rather than throwing, which
     * matches ONNX Runtime's text-tokenizer behaviour.
     */
    private tokenizeWordBPE;
    /**
     * Apply BPE merges to a word (for BPE-style tokenizers like GPT-2/Llama).
     * Merge rules are applied in ascending rank order until no adjacent pair is
     * present in the merge table; caching is keyed on the original word.
     */
    private applyBPE;
    private static toInt64;
}

declare class ImageProcessor {
    static imageDataToFloat32Array(imageData: ImageData, options?: {
        normalize?: boolean;
        layout?: 'NCHW' | 'NHWC';
    }): Float32Array;
}

interface EdgeInferOptions extends ModelOptions, DownloadOptions {
    tokenizerUrl?: string;
    tokenizerConfig?: TokenizerConfig;
    /**
     * Force int64 `input_ids`/`attention_mask` tensors. By default the engine
     * inspects ONNX input metadata and falls back to int32.
     */
    int64Inputs?: boolean;
}
interface GenerateOptions {
    /** Maximum number of new tokens to emit. Default: 64. */
    maxNewTokens?: number;
    /** 0 (default) is greedy decoding; >0 enables temperature sampling. */
    temperature?: number;
    /** Optional top-k truncation for sampling. 0 disables it. */
    topK?: number;
    /** Stop token. Defaults to the tokenizer's `</s>` when present. */
    eosTokenId?: number;
    /** Abort an in-flight stream between decode steps. */
    signal?: AbortSignal;
}
interface GenerationResult {
    text: string;
    tokenIds: number[];
}
interface GenerationEvent {
    tokenId: number;
    /** Best-effort decoded delta for this token. */
    text: string;
    index: number;
}
interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
}
interface ChatOptions extends GenerateOptions {
    /** Override the default role-marker template for the target model. */
    chatTemplate?: (messages: ChatMessage[]) => string;
}
interface ChatResult extends GenerationResult {
    prompt: string;
}
/**
 * EdgeInfer — On-Device WebGPU/WASM AI Inference Engine.
 *
 * Executes ONNX models directly inside the browser using WebGPU (preferred)
 * or WebAssembly SIMD (fallback), with zero server API costs.
 *
 * Powered by: BitNet b1.58, Mamba-2 SSM, Matryoshka MRL research.
 */
declare class EdgeInfer {
    private session;
    private tokenizer;
    private _inputNames;
    private _outputNames;
    private _modelSize;
    private _provider;
    private _int64Inputs;
    private constructor();
    /**
     * Load a model from URL with automatic caching and optimal provider selection.
     */
    static load(modelUrl: string, options?: EdgeInferOptions): Promise<EdgeInfer>;
    /**
     * Create an EdgeInfer instance from a pre-loaded model buffer.
     */
    static fromBuffer(buffer: ArrayBuffer, options?: EdgeInferOptions): Promise<EdgeInfer>;
    /**
     * Set or replace the tokenizer after model load.
     */
    setTokenizer(tokenizer: Tokenizer): void;
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
    predict(inputs: Record<string, Float32Array | Int32Array | BigInt64Array>, shapes?: Record<string, readonly number[]>): Promise<Record<string, Float32Array>>;
    /**
     * Classify text using the loaded model.
     * Requires a tokenizer to be configured.
     */
    classify(text: string): Promise<ClassificationResult[]>;
    /**
     * Generate text embeddings for semantic search / RAG.
     * Returns a Float32Array of embedding dimensions.
     */
    embed(text: string): Promise<Float32Array>;
    /**
     * Generate Matryoshka-truncated embeddings at specified dimension.
     * Supports variable-dimension MRL embeddings (64d → 256d → 768d).
     */
    embedMatryoshka(text: string, dimension: number): Promise<Float32Array>;
    /**
     * Greedy or temperature-sampled autoregressive generation.
     *
     * Decoder models that expose `present.*` outputs and `past_key_values.*`
     * inputs reuse those tensors across steps (KV-cache). Models without cache
     * inputs are re-run over the full sequence each step.
     */
    generate(prompt: string, options?: GenerateOptions): Promise<GenerationResult>;
    /**
     * Stream generated tokens. The decoder loop is identical to generate(), but
     * each token is yielded as soon as it is chosen so a UI can paint before the
     * full completion finishes.
     */
    generateStream(prompt: string, options?: GenerateOptions): AsyncGenerator<GenerationEvent, void, unknown>;
    /**
     * Stream a chat completion using a role-marker template. The default template
     * is model-agnostic; pass `chatTemplate` to match the target tokenizer.
     */
    chatStream(messages: ChatMessage[], options?: ChatOptions): AsyncGenerator<GenerationEvent, void, unknown>;
    /** Collect a chat completion. */
    chat(messages: ChatMessage[], options?: ChatOptions): Promise<ChatResult>;
    /** Pick the last-token logits from the first output that looks like logits. */
    private findLogits;
    /** Argmax (temperature 0) or temperature/top-k sampling over last-token logits. */
    private pickNextToken;
    private static pastInputName;
    private useInt64Inputs;
    /** Build input_ids/attention_mask tensors in the dtype the model declares. */
    private textInputs;
    /**
     * Sentiment analysis.
     */
    sentiment(text: string): Promise<{
        label: string;
        score: number;
    }>;
    /**
     * Classify an image using a vision model.
     */
    classifyImage(imageData: ImageData, layout?: 'NCHW' | 'NHWC'): Promise<ClassificationResult[]>;
    /**
     * Detect objects in an image.
     */
    detectObjects(imageData: ImageData): Promise<Detection[]>;
    /**
     * Check GPU capabilities and recommended configuration.
     */
    static getCapabilities(): Promise<GPUCapabilities>;
    /**
     * Check if WebGPU is available for accelerated inference.
     */
    static isWebGPUAvailable(): Promise<boolean>;
    get inputNames(): readonly string[];
    get outputNames(): readonly string[];
    get modelSize(): number;
    get executionProvider(): string;
    /**
     * Release all resources (ONNX session, GPU memory).
     */
    dispose(): void;
    /** Run without discarding the output tensor dimensions needed by embed(). */
    private runInference;
    /** ONNX tensor shape for a 3-channel image tensor in the given layout. */
    private static imageShape;
    private requireTokenizer;
    private softmax;
    private poolEmbedding;
    private l2Normalize;
}

interface VectorRecord {
    id: string;
    text?: string;
    embedding: Float32Array;
    metadata?: Record<string, unknown>;
}
interface SearchResult {
    id: string;
    text?: string;
    metadata?: Record<string, unknown>;
    score: number;
    coarseScore: number;
}
interface VectorStore {
    add(records: VectorRecord[]): Promise<void>;
    getAll(): Promise<VectorRecord[]>;
    clear(): Promise<void>;
    count(): Promise<number>;
    close?(): void;
}
interface MatryoshkaOptions {
    dimensions?: number;
    coarseDimension?: number;
    refineTopK?: number;
    store?: VectorStore;
}
interface SearchOptions {
    topK?: number;
    refineTopK?: number;
    coarseDimension?: number;
    refine?: boolean;
}
declare function l2Normalize(vector: Float32Array, dimension?: number): Float32Array;
declare function truncateAndNormalize(vector: Float32Array, dimension: number): Float32Array;
declare function quantizeInt8(vector: Float32Array, dimension?: number): {
    data: Int8Array;
    scale: number;
};
declare function dequantizeInt8(data: Int8Array, scale: number): Float32Array;
declare class MemoryVectorStore implements VectorStore {
    private records;
    add(records: VectorRecord[]): Promise<void>;
    getAll(): Promise<VectorRecord[]>;
    clear(): Promise<void>;
    count(): Promise<number>;
}
declare class IndexedDbVectorStore implements VectorStore {
    private readonly dbName;
    private readonly storeName;
    private dbPromise;
    constructor(dbName?: string, storeName?: string);
    private open;
    add(records: VectorRecord[]): Promise<void>;
    getAll(): Promise<VectorRecord[]>;
    clear(): Promise<void>;
    count(): Promise<number>;
    close(): void;
}
declare class MatryoshkaIndex {
    readonly dimensions: number;
    readonly coarseDimension: number;
    readonly refineTopK: number;
    private store?;
    private ids;
    private texts;
    private metadata;
    private full;
    private coarse;
    private scales;
    private size;
    constructor(options?: MatryoshkaOptions);
    get count(): number;
    build(records: VectorRecord[], { persist }?: {
        persist?: boolean | undefined;
    }): Promise<void>;
    static load(store: VectorStore, options?: MatryoshkaOptions): Promise<MatryoshkaIndex>;
    search(query: Float32Array, options?: SearchOptions): SearchResult[];
    exactSearchAtDimension(query: Float32Array, dimension: number, topK?: number): Array<{
        index: number;
        score: number;
    }>;
    recallAtK(queries: Float32Array[], groundTruth: string[][], dimension: number, topK?: number): number;
    get idList(): readonly string[];
}
type EmbedFn = (text: string) => Promise<Float32Array> | Float32Array;
declare function embedMatryoshka(text: string, dimension: number, embed: EmbedFn): Promise<Float32Array>;
declare function createMatryoshkaEmbedder(embed: EmbedFn): (text: string, dimension?: number) => Promise<Float32Array<ArrayBufferLike>>;

interface ModelState {
    tokens: number[];
    /** Opaque KV-cache handle for cache-aware adapters. */
    cache?: unknown;
}
interface DecodeResult {
    state: ModelState;
    /**
     * One next-token distribution for each input position, plus one distribution
     * after the last input token. Length = tokens.length + 1. Calling decode()
     * with an empty token array returns the distribution after the current state.
     */
    logits: Float32Array[];
}
interface LanguageModel {
    readonly vocabSize: number;
    prefill(tokens: number[]): Promise<ModelState>;
    decode(state: ModelState, tokens: number[]): Promise<DecodeResult>;
    /**
     * Drop cache entries after `length` total tokens. Cache-aware adapters must
     * implement this; stateless models can just slice the token list.
     */
    truncate(state: ModelState, length: number): ModelState;
}
interface SpeculativeOptions {
    draft: LanguageModel;
    target: LanguageModel;
    gamma?: number;
    rng?: () => number;
    eosTokenId?: number;
}
interface SpeculativeStats {
    outputTokens: number;
    draftTokens: number;
    acceptedTokens: number;
    bonusTokens: number;
    rejectedTokens: number;
    targetCalls: number;
    draftCalls: number;
    acceptanceRate: number;
    targetCallsPerToken: number;
    elapsedMs: number;
    tokensPerSecond: number;
}
interface SpeculativeGenerationResult {
    tokens: number[];
    stats: SpeculativeStats;
}
declare class SpeculativeDecoder {
    readonly draft: LanguageModel;
    readonly target: LanguageModel;
    readonly gamma: number;
    private readonly rng;
    private readonly eosTokenId?;
    constructor(options: SpeculativeOptions);
    generate(prompt: number[], maxNewTokens?: number, { temperature }?: {
        temperature?: number;
    }): Promise<SpeculativeGenerationResult>;
    private draftBatch;
}
declare function softmax(logits: Float32Array, temperature?: number): Float32Array;
declare function probability(logits: Float32Array, token: number, temperature?: number): number;
declare function sample(distribution: Float32Array, random: number): number;
declare function residualDistribution(targetLogits: Float32Array, draftProbabilities: Float32Array, temperature?: number): Float32Array;
type LogitPolicy = (tokens: number[]) => Float32Array;
/** Deterministic mock used by tests and the throughput benchmark. */
declare class MockLanguageModel implements LanguageModel {
    private readonly policy;
    readonly vocabSize: number;
    private readonly latencyMs;
    constructor(policy: LogitPolicy, vocabSize: number, latencyMs?: number);
    prefill(tokens: number[]): Promise<ModelState>;
    decode(state: ModelState, tokens: number[]): Promise<DecodeResult>;
    truncate(state: ModelState, length: number): ModelState;
}

interface OnnxCausalOptions {
    inputIdsName?: string;
    attentionMaskName?: string;
    positionIdsName?: string;
    logitsName?: string;
}
declare class OnnxCausalLM implements LanguageModel {
    private readonly session;
    readonly vocabSize: number;
    private readonly inputIdsName;
    private readonly attentionMaskName;
    private readonly positionIdsName?;
    private readonly logitsName;
    constructor(session: ort.InferenceSession, vocabSize: number, options?: OnnxCausalOptions);
    prefill(tokens: number[]): Promise<ModelState>;
    decode(state: ModelState, tokens: number[]): Promise<DecodeResult>;
    truncate(state: ModelState, length: number): ModelState;
}

export { type ChatMessage, type ChatOptions, type ChatResult, type ClassificationResult, type DecodeResult, type Detection, type DownloadOptions, EdgeInfer, type EdgeInferOptions, type EmbedFn, type EncodingResult, EventEmitter, type GPUCapabilities, type GenerateOptions, type GenerationEvent, type GenerationResult, ImageProcessor, IndexedDbVectorStore, type LanguageModel, type LogitPolicy, MatryoshkaIndex, type MatryoshkaOptions, MemoryVectorStore, MockLanguageModel, ModelCache, type ModelOptions, type ModelState, OnnxCausalLM, type OnnxCausalOptions, RuntimeManager, type SearchOptions, type SearchResult, SpeculativeDecoder, type SpeculativeGenerationResult, type SpeculativeOptions, type SpeculativeStats, Tokenizer, type TokenizerConfig, type VRAMProbeOptions, type VRAMProbeResult, type VectorRecord, type VectorStore, createMatryoshkaEmbedder, dequantizeInt8, embedMatryoshka, l2Normalize, probability, quantizeInt8, residualDistribution, sample, softmax, truncateAndNormalize };
