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

declare class ModelCache extends EventEmitter {
    private cacheName;
    constructor(cacheName?: string);
    getModel(url: string, forceDownload?: boolean): Promise<ArrayBuffer>;
}

interface GPUCapabilities {
    provider: string;
    hasWebGPU: boolean;
    hasWebNN: boolean;
    estimatedVRAM: number;
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
     * Falls back gracefully if preferred provider fails.
     */
    static createSession(modelBuffer: ArrayBuffer, providers?: string[]): Promise<ort.InferenceSession>;
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
    merges?: string[];
    specialTokens?: Record<string, number>;
    maxLength?: number;
    padTokenId?: number;
    unkTokenId?: number;
    clsTokenId?: number;
    sepTokenId?: number;
}
interface EncodingResult {
    inputIds: Int32Array;
    attentionMask: Int32Array;
    tokenTypeIds?: Int32Array;
}
/**
 * Production BPE/WordPiece tokenizer for ONNX model inference.
 * Supports loading vocabularies from JSON, encoding text to token IDs,
 * and decoding token IDs back to text.
 */
declare class Tokenizer {
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
    constructor(config: TokenizerConfig);
    /**
     * Load tokenizer configuration from a JSON URL (tokenizer.json or vocab.json).
     */
    static fromUrl(url: string): Promise<Tokenizer>;
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
     * Pre-tokenize text: lowercase, split on whitespace and punctuation boundaries.
     */
    private preTokenize;
    /**
     * Tokenize a single word using WordPiece algorithm.
     */
    private tokenizeWord;
    /**
     * Apply BPE merges to a word (for BPE-style tokenizers like GPT-2).
     */
    private applyBPE;
    private getCharPairs;
}

declare class ImageProcessor {
    static imageDataToFloat32Array(imageData: ImageData, options?: {
        normalize?: boolean;
        layout?: 'NCHW' | 'NHWC';
    }): Float32Array;
}

interface EdgeInferOptions extends ModelOptions {
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
declare class EdgeInfer {
    private session;
    private tokenizer;
    private _inputNames;
    private _outputNames;
    private _modelSize;
    private _provider;
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
     */
    predict(inputs: Record<string, Float32Array | Int32Array | BigInt64Array>): Promise<Record<string, Float32Array>>;
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
    private requireTokenizer;
    private softmax;
    private meanPool;
    private l2Normalize;
}

export { type ClassificationResult, type Detection, EdgeInfer, type EdgeInferOptions, type EncodingResult, EventEmitter, type GPUCapabilities, ImageProcessor, ModelCache, type ModelOptions, RuntimeManager, Tokenizer, type TokenizerConfig };
