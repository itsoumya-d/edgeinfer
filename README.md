<!--
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath
-->

# EdgeInfer

<div align="center">
  <p><strong>EdgeInfer runs small ONNX models inside the browser on WebGPU or WebAssembly, so inference costs nothing per call and user data never leaves the device.</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Status](https://img.shields.io/badge/status-pre--release-orange.svg)]()
</div>

---

> **Pre-release software. Not published to npm. No production adopters yet. See [Known Limitations](#known-limitations).**

---

## What is EdgeInfer?

EdgeInfer is a TypeScript library that runs ONNX models in the browser using ONNX Runtime Web, with automatic fallback from WebGPU to WebAssembly SIMD to plain WASM. It includes a Tokenizer, ImageProcessor, and ModelCache for common ML preprocessing tasks.

Real exported symbols: `EdgeInfer`, `EventEmitter`, `ImageProcessor`, `ModelCache`, `RuntimeManager`, `Tokenizer`. Nothing is published under a separate npm package.

---

## What's new in v2

- **Two-stage Matryoshka retrieval** with an INT8 coarse scan (`d=64`) and FP32
  re-ranking (`d=768`), persisted in IndexedDB or an in-memory store.
- **Speculative decoding** with `gamma=5` draft tokens verified in one target
  forward pass, residual resampling on rejection, and cache-aware state
  truncation.
- **ONNX Runtime Web causal-LM adapter** (`OnnxCausalLM`) for standard
  `input_ids`/`attention_mask` exports.
- MIT licensed, no license key, 95 tests.

## Installation

EdgeInfer is **not published on npm**. `npm install edgeinfer` will fail. Install from source:

> **EdgeInfer requires `onnxruntime-web` at runtime.** It is a real dependency, not
> bundled: `dist/index.mjs` contains `import * as ort from "onnxruntime-web"`. Because
> that is a *bare module specifier*, a browser cannot resolve it on its own — loading
> `dist/index.mjs` straight from a CDN fails with
> `TypeError: Failed to resolve module specifier "onnxruntime-web"`.
> Use **onnxruntime-web >= 1.21.0**: in 1.17–1.20 the package's root entry point
> registers only the `cpu` and `wasm` execution providers, so requesting `webgpu`
> fails and EdgeInfer silently falls back to WASM.

**Option 1 — jsDelivr CDN (no build step). An import map is required:**
```html
<script type="importmap">
{
  "imports": {
    "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.bundle.min.mjs"
  }
}
</script>
<script type="module">
  import { EdgeInfer } from 'https://cdn.jsdelivr.net/gh/itsoumya-d/edgeinfer@main/dist/index.mjs';
</script>
```
The import map must appear **before** the module script. Without it the import throws.

**Option 2 — Clone and build (bundler resolves onnxruntime-web for you):**
```bash
git clone https://github.com/itsoumya-d/edgeinfer.git
cd edgeinfer
npm install
npm run build
```
Then import from `./dist/index.mjs` or `./dist/index.js`.

---

## Release process

The package metadata is npm-ready (`exports`, `publishConfig.provenance`, `pack:check`). Publishing is tag-driven and requires the repository secret `NPM_TOKEN` with publish rights:

```bash
git tag v2.0.0
git push origin v2.0.0
```

`.github/workflows/release.yml` verifies that the tag matches `package.json`, runs typecheck, tests, build and `npm pack --dry-run`, publishes with npm provenance, and creates a GitHub release. The package remains **pre-release and unpublished** until that workflow is run with a valid token; no npm publication has been performed yet.

## Quick Start

```typescript
import { EdgeInfer } from './dist/index.mjs';

// 1. Check WebGPU availability (optional — loads fall back to WASM automatically)
const caps = await EdgeInfer.getCapabilities();
console.log(`Running on: ${caps.provider}`); // 'webgpu' | 'wasm'

// 2. Load a model (URL must return an ONNX binary)
const model = await EdgeInfer.load('https://example.com/models/sentiment.onnx', {
  tokenizerUrl: 'https://example.com/models/tokenizer.json',
});

// 3. Run inference
const result = await model.sentiment('This is great!');
console.log(result); // { label: 'positive', score: 0.98 }

// 4. Release resources
model.dispose();
```

---

## API Reference

### `EdgeInfer` Class

#### `static async load(modelUrl: string, options?: EdgeInferOptions): Promise<EdgeInfer>`
Downloads (and caches via Cache API), then creates an inference session. `options.executionProviders` overrides auto-detection.

#### `static async fromBuffer(buffer: ArrayBuffer, options?: EdgeInferOptions): Promise<EdgeInfer>`
Creates a session from a pre-loaded model buffer. Rejects with `Error` if the buffer is not a valid ONNX model.

#### `static async isWebGPUAvailable(): Promise<boolean>`
Returns `true` if `navigator.gpu` is present and the adapter request succeeds.

#### `static async getCapabilities(): Promise<GPUCapabilities>`
Returns `{ provider, hasWebGPU, hasWebNN, estimatedVRAM, recommendedQuantization }`.

#### `async predict(inputs, shapes?): Promise<Record<string, Float32Array>>`
Low-level: pass raw typed arrays keyed by ONNX input name, receive raw output tensors.
`shapes` optionally supplies the ONNX tensor shape per input name. Inputs without an
explicit shape default to `[1, data.length]`, which is correct for 2-D sequence inputs
(`input_ids`, `attention_mask`) but **wrong for higher-rank inputs** — a vision model
declaring `pixel_values: [1,3,224,224]` will reject a rank-2 tensor with
`Invalid rank for input`. Pass `shapes` for those:
```js
await model.predict({ pixel_values: chw }, { pixel_values: [1, 3, 224, 224] });
```

Note on input dtype: `classify()`, `embed()` and `sentiment()` select **`tensor(int64)`**
automatically when the ONNX session declares `input_ids` as int64, or when
`int64Inputs: true` is passed at load time; otherwise they use `tensor(int32)`.
`predict()` still accepts either `Int32Array` or `BigInt64Array` directly.

#### `async classify(text: string): Promise<ClassificationResult[]>`
Requires a tokenizer (pass `tokenizerUrl` or `tokenizerConfig` in options, or call `setTokenizer()`). Throws `Error` with a clear message if no tokenizer is configured.

#### `async embed(text: string): Promise<Float32Array>`
Returns an L2-normalized embedding vector from the first model output. Requires tokenizer
and a `float32` output with one of these ONNX shapes:

- `[hidden]` or `[1, hidden]`: already pooled; preserves every embedding dimension.
- `[1, sequence, hidden]`: mean-pools non-masked tokens, then normalizes. The sequence
  length must match the tokenizer's attention mask. An all-masked sequence returns zeros.

Other shapes, batched outputs, non-`float32` outputs, and sequence/mask mismatches throw
clear errors rather than guessing the shape from the flattened array length. Use
`predict()` for model-specific postprocessing; its raw flattened output API is unchanged.

#### `async embedMatryoshka(text: string, dimension: number): Promise<Float32Array>`
Truncates the embedding to `dimension` and re-normalizes (Matryoshka MRL).

#### `async generate(prompt: string, options?: GenerateOptions): Promise<GenerationResult>`
Greedy (`temperature: 0`, default) or temperature-sampled autoregressive decoding.
When the session exposes `present.*` outputs and matching `past_key_values.*` inputs,
those tensors are reused across steps (KV cache). Otherwise the full sequence is
re-run each step.

```typescript
const result = await model.generate('Explain KV caching.', {
  maxNewTokens: 64,
  temperature: 0,
  topK: 0,
  eosTokenId: 2,
});
// result: { text: string, tokenIds: number[] }
```

The decoder must expose a `logits` output (or an output whose name contains
`logits`). Input dtype is selected from ONNX metadata, with `int64Inputs` as an
explicit override.

#### `async sentiment(text: string): Promise<{ label: string; score: number }>`
Returns `{ label: 'positive'|'negative'|'neutral', score }`. Requires tokenizer.

#### `async classifyImage(imageData: ImageData): Promise<ClassificationResult[]>`
Expects browser `ImageData`. Converts to `Float32Array` via `ImageProcessor`.

#### `async detectObjects(imageData: ImageData): Promise<Detection[]>`
Returns `{ bbox, label, score }[]` for detections above 0.5 confidence.

#### `dispose(): void`
Releases the ONNX session and resets the provider cache.

### `RuntimeManager`

#### `static async detectCapabilities(): Promise<GPUCapabilities>`
In Node.js (no `navigator.gpu`): returns `{ provider: 'wasm', hasWebGPU: false, ... }` without throwing.

#### `static resetCache(): void`
Clears the cached capabilities so the next call re-probes the environment.

### `Tokenizer`

Constructor takes `{ vocab, merges?, specialTokens?, maxLength?, padTokenId?, unkTokenId?, clsTokenId?, sepTokenId?, bpeSpaceMarker? }`. Note: `vocab` is the key — not `vocabulary`.

`encode(text, { addSpecialTokens, maxLength, padding })` returns
`{ inputIds: Int32Array, attentionMask: Int32Array, inputIdsInt64: BigInt64Array, attentionMaskInt64: BigInt64Array }`.
Both dtypes are always produced, so callers can feed either `tensor(int32)` or
`tensor(int64)` models without re-encoding.

When `merges` is present the tokenizer applies rank-ordered BPE merges and
**preserves case by default**; without merges it uses WordPiece and retains the
legacy uncased default. Pass `lowercase: true` to force uncased input, or
`bpeSpaceMarker: '▁' | 'Ġ'` for SentencePiece/byte-level vocabularies. Empty input
returns a single pad token rather than an empty array (which would crash the ONNX
session).

### `ImageProcessor`

`static imageDataToFloat32Array(imageData, options?)` converts an RGBA `ImageData`-like object to a CHW `Float32Array` in `[0,1]` range. Works in Node with a mock `ImageData` struct `{ data: Uint8ClampedArray, width, height }`.

---

## Browser-local retrieval and decoding

### Matryoshka hybrid RAG

`MatryoshkaIndex` stores full FP32 unit vectors and a truncated INT8 view. A
query scans every chunk at 64 dimensions, then re-scores the top 50 candidates
at 768 dimensions. Vectors can be persisted in IndexedDB:

```ts
import { MatryoshkaIndex, IndexedDbVectorStore } from 'edgeinfer';

const store = new IndexedDbVectorStore();
const index = new MatryoshkaIndex({ dimensions: 768, coarseDimension: 64, store });
await index.build(records); // records: { id, text, embedding: Float32Array(768) }[]

const results = index.search(queryEmbedding, { topK: 10 });
```

For a text query, wrap any embedding function:

```ts
import { createMatryoshkaEmbedder } from 'edgeinfer';
const embed = createMatryoshkaEmbedder((text) => engine.embed(text));
const query64 = await embed('refund policy', 64);
```

Synthetic benchmark on an Apple Silicon laptop (Node 26, 100,000 chunks at 768
dimensions): build 1.76 s; coarse scan + re-rank p50 25.1 ms / p95 59.4 ms;
top-10 hit rate 0.90; Recall@10 rises from 0.725 at `d=64` to 1.000 at `d=768`.
This is a synthetic workload, not an MTEB evaluation. Reproduce with
`npm run bench:matryoshka`.

### Speculative decoding

`SpeculativeDecoder` samples `gamma` tokens from a draft `LanguageModel`, asks
the target for one forward pass over those tokens, accepts each token with
probability `min(1, p_target/q_draft)`, resamples from the residual
`max(0, p_target - q_draft)` on rejection, and truncates both model states.
`MockLanguageModel` is included for tests and throughput analysis. `OnnxCausalLM`
adapts a standard causal-LM ONNX session; it recomputes the full sequence per
call, so a cache-aware adapter can implement `LanguageModel.truncate` for KV
caching without changing the decoder.

Mock benchmark (`gamma=5`, draft 0.2 ms/forward, target 2.0 ms/forward):
68.4% acceptance, 0.45 target calls per generated token, and a modeled 1.64x
speedup over target-only decoding. This is an analytical mock result, not a
WebGPU measurement. Reproduce with `npm run bench:speculative`.

## Performance

**Claimed benchmark (from original README): M2 MacBook Air, Chrome 120, WebGPU**
- MobileBERT: ~4ms | MiniLM: ~8ms | MobileNetV3: ~6ms | YOLOv8 Nano: ~15ms | Phi-3-mini: ~25 tok/s

**These numbers cannot be reproduced in this environment.** WebGPU is not available in Node.js. A meaningful benchmark requires a real browser with WebGPU support and a loaded ONNX model. The claims above are removed from the main README to avoid misleading users. If you have browser benchmark results to contribute, open a PR with your hardware spec, browser version, and methodology.

---

## Known Limitations

- **Pre-release, no npm package.** Use jsDelivr or clone from source.
- **WebGPU requires a real browser.** `RuntimeManager.detectCapabilities()` returns `{ provider: 'wasm', hasWebGPU: false }` in Node.js — not an error, just WASM fallback.
- **No WebGPU = WASM execution.** On browsers without WebGPU (Firefox without flag, older Chromium), all inference runs on WASM. This is slower but still functional.
- **Model loading requires fetch.** `EdgeInfer.load()` calls `fetch()`. In Node.js, you need Node 18+ (native fetch) or a polyfill. WASM file loading may also require the `wasm-unsafe-eval` Content-Security-Policy directive in the browser.
- **No text methods without a tokenizer.** `classify()`, `embed()`, `sentiment()` throw a clear `Error` if no tokenizer is configured: `EdgeInfer: No tokenizer configured. Either pass tokenizerUrl/tokenizerConfig...`. This is not a silent hang.
- **Performance claims deleted.** The original README listed latency numbers (`~4ms`, `~25 tokens/sec`) measured on an M2 MacBook with Chrome WebGPU. These cannot be verified in a CI environment with no GPU. The table has been removed.
- **Bundle size claim "< 5KB" is incorrect.** `dist/index.mjs` is ~24KB before gzip. `onnxruntime-web` is a real runtime **dependency** (not a peer dependency) and adds ~1.5MB plus the `.wasm` binaries.
- **CDN use needs an import map.** See Installation. `dist/index.mjs` imports the bare specifier `onnxruntime-web`, which browsers cannot resolve unaided.
- **`onnxruntime-web` must be >= 1.21.0** for the WebGPU execution provider to be registered by the root entry point; `package.json` declares `^1.21.0`.
- **`estimatedVRAM` is not VRAM.** It is `max(maxBufferSize, maxStorageBufferBindingSize)` from the WebGPU adapter — per-buffer limits whose spec baselines are 256 MiB / 128 MiB. Browsers commonly report near-baseline values regardless of physical VRAM, so the `>= 8GB → fp16` branch of `recommendedQuantization` is effectively unreachable, and `RuntimeManager.canFitModel()` is unreliable. WebGPU exposes no total-VRAM API.
- **Generic generation only.** `generate()` supports models that expose `present.*` outputs and `past_key_values.*` inputs plus a `logits` output. Architectures with a different cache layout (for example a single combined cache tensor) need a custom adapter. The loop is unit-tested with a deterministic mock session; no real decoder model is benchmarked in CI. There is no streaming token callback yet.
- **No quantization.** `recommendedQuantization` is a string hint; EdgeInfer never quantizes weights. Models must be pre-quantized.
- **No GGUF, no audio.** ONNX only. No Whisper/speech pipeline exists.
- **Tokenizer covers WordPiece and merge-table BPE.** Byte-level pre-tokenization (GPT-2 regex and `Ġ` handling) and full SentencePiece normalization pipelines are not reproduced; supply `bpeSpaceMarker` and a compatible vocabulary. Unsupported normalizers fail rather than silently change token IDs.
- **Downloads have per-transfer deadlines, not whole-load deadlines.** Model/tokenizer headers and bodies default to 120 seconds each and support cancellation. Runtime initialization and inference are not covered. Partial HTTP model responses are rejected; range assembly is not implemented.
- **No production adopters yet.** APIs may change without notice.

---

## Competitor Comparison

| Feature | EdgeInfer | TensorFlow.js | Transformers.js |
|---|---|---|---|
| Core Engine | ONNX Runtime Web | Custom / WebGL | ONNX Runtime Web |
| WebGPU Support | First-class | Experimental | First-class |
| npm availability | Not published | Published | Published |
| Production-ready | Pre-release | Yes | Yes |

---

## 📄 License

**[MIT](LICENSE)** — free to use, copy, modify, merge, publish, distribute, sublicense, and sell,
including in commercial and closed-source products. Keep the copyright and permission notice.

Contributions are accepted under the MIT License.


### Bounded downloads and tokenizer casing

`EdgeInfer.load(url, { signal, downloadTimeoutMs })` passes cancellation and a
per-transfer deadline to model and tokenizer downloads. `ModelCache.getModel`
accepts these options as its third argument and `Tokenizer.fromUrl` as its second.
The default is 120,000 ms per transfer, including response headers and the full
body. Use a finite positive value no larger than 2,147,483,647 ms. This is not an
inference/session-creation timeout. A tokenizer failure or cancellation after
session creation releases the session. Browser Cache API operations cannot be
physically cancelled, but callers still receive cancellation and no partial
network body is committed; an already-started complete cache write may finish.

WordPiece tokenization remains lowercase by default. Set `lowercase: false` in
`TokenizerConfig` to preserve case. Loaded HuggingFace JSON supports an explicit
null normalizer (identity), `Lowercase`, or a `BertNormalizer` with explicit
`lowercase` and all of `strip_accents`, `clean_text`, `handle_chinese_chars` false.
Other normalizers are rejected, not silently approximated. Omitted normalizers
retain the legacy lowercase default. This does not implement BPE, accent stripping,
Unicode normalization, or full HuggingFace tokenizer compatibility.
