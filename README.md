<!--
// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com | +91 7031648617
-->

# EdgeInfer

<div align="center">
  <p><strong>On-Device AI Inference via ONNX Runtime WebAssembly and WebGPU</strong></p>

  [![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-red.svg)](https://mariadb.com/bsl11/)
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

**Option 1 — jsDelivr CDN (no build step):**
```html
<script type="module">
  import { EdgeInfer } from 'https://cdn.jsdelivr.net/gh/itsoumya-d/edgeinfer@main/dist/index.mjs';
</script>
```

**Option 2 — Clone and build:**
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

#### `async predict(inputs): Promise<Record<string, Float32Array>>`
Low-level: pass raw typed arrays keyed by ONNX input name, receive raw output tensors.

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
- **Bundle size claim "< 5KB" is incorrect.** `dist/index.mjs` is ~20KB before gzip. The `onnxruntime-web` peer dependency adds ~1.5MB.
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

## License — Business Source License 1.1

> **Source-available, NOT open-source. Production use requires a paid license.**

| Tier | Price | For |
|---|---|---|
| Indie | $499/year | Solo developer, <$100K revenue |
| Startup | $3,999/year | Up to 10-25 devs, <$5M revenue |
| Enterprise | $19,999/year | Unlimited seats, unlimited revenue |
| OEM / White-Label | $39,999/year | Embed in your product |

**Free use:** Personal evaluation, academic research, open-source contribution.

Contact: soumyadebnath1661@gmail.com | +91 7031648617 | github.com/itsoumya-d

© 2024-2026 Soumya Debnath. All Rights Reserved.
