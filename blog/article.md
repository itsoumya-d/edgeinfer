# EdgeInfer Architecture: Running 8B Parameter Models in the Browser via WebGPU and INT4 Quantization

*By Soumya Debnath • July 28, 2026*

> **Scope note.** Sections 1–3 describe the design space for browser LLM inference in
> general, not features EdgeInfer ships. EdgeInfer today is a TypeScript wrapper around
> ONNX Runtime Web for classification, embeddings and image models. It does **not**
> implement text generation, weight quantization, KV caching or token streaming, and
> therefore cannot run Llama 3 or any other generative model. The cost table in section 4
> compares hosted generation against a capability EdgeInfer does not yet have. The code
> recipe at the end is the real, working API.

The conventional wisdom of cloud SaaS dictates that intelligent applications require centralized compute. We route natural language via API gateways, absorb $20/million tokens, and rationalize the latency. But when you examine the architecture from first principles, this model is highly inefficient.

Why pay for rented A100s when your user's M3 Max or RTX 4090 is sitting idle during their session?

## 1. The VRAM Economics of 8B Models

To run an 8B parameter model like Llama 3 in fp16, you need ~16GB of VRAM. That immediately disqualifies 80% of consumer hardware. The solution is **INT4 Quantization**.

By quantizing weights to 4-bit integers and grouping them (typically with a group size of 128), we compress an 8B model down to roughly 4.5GB. This comfortably fits into the VRAM of modern integrated graphics and most discrete GPUs.

### WebGPU VRAM Detection Algorithm

EdgeInfer queries the adapter limits asynchronously before attempting to load weights:

```typescript
async function initializeWebGPU() {
  if (!navigator.gpu) throw new Error("WebGPU not supported");
  
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance'
  });
  
  if (!adapter) throw new Error("No GPU adapter available");

  // These are per-buffer/binding LIMITS, not an amount of VRAM.
  // WebGPU's guaranteed baselines are 256 MiB (maxBufferSize) and
  // 128 MiB (maxStorageBufferBindingSize); browsers commonly report
  // near-baseline values regardless of physical VRAM. WebGPU exposes
  // no API for total VRAM.
  const maxBufferSize = adapter.limits.maxBufferSize;
  const maxBinding    = adapter.limits.maxStorageBufferBindingSize;

  console.log(`Max buffer: ${maxBufferSize / 1024 / 1024}MB`);

  return adapter;
}
```

## 2. Tokenization Without Main Thread Blocking

Tokenization (BPE/WordPiece) is highly sequential. If you process a large prompt on the main thread, the UI freezes. Moving tokenization into a Web Worker and transferring token IDs via SharedArrayBuffer is the standard remedy. *Not yet implemented in EdgeInfer* — its `Tokenizer` currently runs on the calling thread.

## 3. Streaming Token DOM Insertion

To achieve a ChatGPT-like streaming experience, the compute shader dispatch loop must yield to the event loop. The usual approach is to tie read-back to `requestAnimationFrame`. *Not yet implemented in EdgeInfer*, which exposes no token-streaming or text-generation API at all.

## 4. Cloud API vs EdgeInfer Economics

Let's do the math on running an AI coding assistant averaging 10M tokens per user per month.

| Provider / Architecture | Latency | Cost (10M Tokens) | Privacy |
|---|---|---|---|
| OpenAI GPT-4o | 600ms+ | $50.00 | Server-side |
| AWS SageMaker (Llama 3) | 300ms | ~$150.00 | VPC |
| **EdgeInfer (INT4 WebGPU)** | **0ms (Local)** | **$0.00** | **Air-gapped** |

## Building an AI Agent with EdgeInfer

Here is a complete copy-paste TypeScript integration recipe for developers and AI agents looking to embed zero-cost intelligence into a web app.

```typescript
// Browsers cannot resolve the bare "onnxruntime-web" specifier, so the CDN
// path needs an import map:
// <script type="importmap">{"imports":{"onnxruntime-web":
//   "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.bundle.min.mjs"}}</script>
import { EdgeInfer } from 'https://cdn.jsdelivr.net/gh/itsoumya-d/edgeinfer@main/dist/index.mjs';

// 1. Check what the device can do (never throws; falls back to WASM)
const caps = await EdgeInfer.getCapabilities();
console.log(caps.provider); // 'webgpu' | 'webnn' | 'wasm'

// 2. Load an ONNX classifier and its tokenizer.
//    EdgeInfer's constructor is private - use the static factories.
const model = await EdgeInfer.load('/models/sentiment.onnx', {
  tokenizerUrl: '/models/tokenizer.json'
});

// 3. Run inference. There is no .chat() and no token streaming:
//    EdgeInfer does classification, embeddings and vision, not generation.
console.log(await model.sentiment('This runs on-device.'));
console.log(await model.embed('vector search query'));

model.dispose();
```

Moving the models a browser can actually run to the edge removes their per-call cost entirely. Generative workloads are a harder problem and remain out of scope for EdgeInfer today.
