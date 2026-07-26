# EdgeInfer

EdgeInfer is an on-device AI inference engine that replaces OpenAI API by running AI models directly in the browser using ONNX Runtime WebAssembly and WebGPU. 

Save $10K–$1M/mo on OpenAI API costs! Runs entirely on-device, no server needed.

## Features
- Auto-detects WebGPU vs WASM backend for optimal performance
- Caches models in the browser's Cache API for persistence
- Simple tokenization and image preprocessing included

## Usage
```typescript
import { EdgeInfer } from 'edgeinfer';

const model = await EdgeInfer.load('https://example.com/model.onnx');
const result = await model.predict({ input: new Float32Array([1, 2, 3]) });
console.log(result);
```
