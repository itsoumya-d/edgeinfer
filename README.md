<!--
// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Dual-licensed: AGPL-3.0-or-later (free, see LICENSE) OR a commercial licence
// (see COMMERCIAL_LICENSE.md) if you cannot meet the AGPL's source-disclosure terms.
// Contact: soumyadebnath1619@gmail.com
-->

# EdgeInfer

<div align="center">
  <p><strong>EdgeInfer runs small ONNX models inside the browser on WebGPU or WebAssembly, so inference costs nothing per call and user data never leaves the device.</strong></p>

  [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
  [![Status](https://img.shields.io/badge/status-pre--release-orange.svg)]()
</div>

---

> **Pre-release software. Not published to npm. No production adopters yet. See [Known Limitations](#known-limitations).**

---

## What is EdgeInfer?

EdgeInfer is a TypeScript library that runs ONNX models in the browser using ONNX Runtime Web, with automatic fallback from WebGPU to WebAssembly SIMD to plain WASM. It includes a Tokenizer, ImageProcessor, and ModelCache for common ML preprocessing tasks.

Real exported symbols: `EdgeInfer`, `EventEmitter`, `ImageProcessor`, `ModelCache`, `RuntimeManager`, `Tokenizer`. Nothing is published under a separate npm package.

---

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

Note on input dtype: `classify()`, `embed()` and `sentiment()` build **`Int32Array`**
inputs (`tensor(int32)`). Most ONNX models exported from HuggingFace Transformers
declare `input_ids`/`attention_mask` as **`tensor(int64)`** and will reject them with
`Unexpected input data type. Actual: (tensor(int32)), expected: (tensor(int64))`.
For those models, call `predict()` directly with `BigInt64Array` inputs.

#### `async classify(text: string): Promise<ClassificationResult[]>`
Requires a tokenizer (pass `tokenizerUrl` or `tokenizerConfig` in options, or call `setTokenizer()`). Throws `Error` with a clear message if no tokenizer is configured.

#### `async embed(text: string): Promise<Float32Array>`
Returns mean-pooled, L2-normalized embedding vector. Requires tokenizer.

#### `async embedMatryoshka(text: string, dimension: number): Promise<Float32Array>`
Truncates the embedding to `dimension` and re-normalizes (Matryoshka MRL).

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

Constructor takes `{ vocab, merges?, specialTokens?, maxLength?, padTokenId?, unkTokenId?, clsTokenId?, sepTokenId? }`. Note: `vocab` is the key — not `vocabulary`.

`encode(text, { addSpecialTokens, maxLength, padding })` returns `{ inputIds: Int32Array, attentionMask: Int32Array }`. Empty string returns a single pad token rather than an empty array (which would crash the ONNX session).

### `ImageProcessor`

`static imageDataToFloat32Array(imageData, options?)` converts an RGBA `ImageData`-like object to a CHW `Float32Array` in `[0,1]` range. Works in Node with a mock `ImageData` struct `{ data: Uint8ClampedArray, width, height }`.

---

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
- **`onnxruntime-web` must be >= 1.21.0** for the WebGPU execution provider to be registered by the root entry point, even though `package.json` currently allows `^1.17.0`.
- **`estimatedVRAM` is not VRAM.** It is `max(maxBufferSize, maxStorageBufferBindingSize)` from the WebGPU adapter — per-buffer limits whose spec baselines are 256 MiB / 128 MiB. Browsers commonly report near-baseline values regardless of physical VRAM, so the `>= 8GB → fp16` branch of `recommendedQuantization` is effectively unreachable, and `RuntimeManager.canFitModel()` is unreliable. WebGPU exposes no total-VRAM API.
- **No text generation.** There is no chat, streaming, KV cache or autoregressive decode loop. EdgeInfer cannot run Llama, Phi-3, Gemma or any generative LLM. It does classification, embeddings and vision.
- **No quantization.** `recommendedQuantization` is a string hint; EdgeInfer never quantizes weights. Models must be pre-quantized.
- **No GGUF, no audio.** ONNX only. No Whisper/speech pipeline exists.
- **Tokenizer is WordPiece-only and always lowercases.** `TokenizerConfig.merges` is accepted and normalised, but `applyBPE()` is never called, so true BPE (GPT-2 style) tokenization is not performed. `preTokenize()` unconditionally lowercases, so cased models are tokenized incorrectly.
- **`embed()` can silently mis-pool.** Mean pooling infers `hiddenDim = output.length / seqLen`. If a model's first output is already pooled (`[1, hidden]`) and `hidden` happens to be divisible by the token count, the guard does not fire and the vector is averaged into `hidden/seqLen` dimensions instead of being returned as-is.
- **Model download has no timeout.** `EdgeInfer.load()` awaits `fetch()` with no deadline; a server that accepts the connection and never responds leaves the promise pending indefinitely. Pass your own `AbortSignal`-wrapped fetch, or a watchdog, if you need bounded load time.
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

**Dual-licensed — choose either:**

1. **[AGPL-3.0-or-later](LICENSE)** — free for any purpose, including commercial and production
   use. No payment, no permission, no key required. The obligation it carries: if you modify this
   software and let users interact with it over a network, you must offer those users your modified
   source under the same licence.

2. **[Commercial licence](COMMERCIAL_LICENSE.md)** — for organisations that cannot or prefer not to
   meet the AGPL's source-disclosure obligation. This buys an exception, not access.

Contributions are accepted under AGPL-3.0-or-later. Full terms: [LICENSING.md](LICENSING.md).

