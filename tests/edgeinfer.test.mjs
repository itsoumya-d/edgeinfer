import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const dist = require('../dist/index.js');

// ── 1. Module shape ──────────────────────────────────────────────────────────

describe('module exports', () => {
  test('exports EdgeInfer class', () => {
    assert.strictEqual(typeof dist.EdgeInfer, 'function');
  });

  test('exports EventEmitter class', () => {
    assert.strictEqual(typeof dist.EventEmitter, 'function');
  });

  test('exports ImageProcessor class', () => {
    assert.strictEqual(typeof dist.ImageProcessor, 'function');
  });

  test('exports ModelCache class', () => {
    assert.strictEqual(typeof dist.ModelCache, 'function');
  });

  test('exports RuntimeManager class', () => {
    assert.strictEqual(typeof dist.RuntimeManager, 'function');
  });

  test('exports Tokenizer class', () => {
    assert.strictEqual(typeof dist.Tokenizer, 'function');
  });
});

// ── 2. WebGPU feature detection ──────────────────────────────────────────────

describe('RuntimeManager.detectCapabilities()', () => {
  test('resolves without throwing in Node (no WebGPU)', async () => {
    const caps = await dist.RuntimeManager.detectCapabilities();
    assert.strictEqual(typeof caps, 'object');
  });

  test('returns provider = wasm when WebGPU is absent (Node)', async () => {
    dist.RuntimeManager.resetCache();
    const caps = await dist.RuntimeManager.detectCapabilities();
    assert.strictEqual(caps.provider, 'wasm');
  });

  test('hasWebGPU is false in Node', async () => {
    dist.RuntimeManager.resetCache();
    const caps = await dist.RuntimeManager.detectCapabilities();
    assert.strictEqual(caps.hasWebGPU, false);
  });

  test('returns a recommendedQuantization string', async () => {
    dist.RuntimeManager.resetCache();
    const caps = await dist.RuntimeManager.detectCapabilities();
    assert.ok(['fp32', 'fp16', 'int8', 'int4'].includes(caps.recommendedQuantization));
  });
});

describe('EdgeInfer.isWebGPUAvailable()', () => {
  test('returns false in Node (no navigator.gpu)', async () => {
    dist.RuntimeManager.resetCache();
    const result = await dist.EdgeInfer.isWebGPUAvailable();
    assert.strictEqual(result, false);
  });
});

describe('EdgeInfer.getCapabilities()', () => {
  test('returns capabilities object without throwing', async () => {
    dist.RuntimeManager.resetCache();
    const caps = await dist.EdgeInfer.getCapabilities();
    assert.ok('hasWebGPU' in caps);
    assert.ok('provider' in caps);
  });
});

// ── 3. ModelCache ────────────────────────────────────────────────────────────

describe('ModelCache', () => {
  test('can be instantiated', () => {
    assert.doesNotThrow(() => new dist.ModelCache('test-cache'));
  });

  test('getModel rejects for unreachable URL in Node (no caches API, fetch fails)', async () => {
    // In Node there is no caches API, so ModelCache falls back to fetch().
    // An unreachable URL must reject with a TypeError/Error, not hang.
    const cache = new dist.ModelCache('test-cache');
    await assert.rejects(
      () => cache.getModel('http://127.0.0.1:1/nonexistent.onnx'),
      (err) => {
        assert.ok(err instanceof Error);
        return true;
      }
    );
  });
});

// ── 4. Tokenizer ─────────────────────────────────────────────────────────────

describe('Tokenizer', () => {
  // Tokenizer constructor uses config.vocab (not config.vocabulary)
  const makeTokenizer = () => new dist.Tokenizer({
    vocab: { '[CLS]': 101, '[SEP]': 102, '[PAD]': 0, '[UNK]': 100, hello: 7592, world: 2088 },
    maxLength: 16,
    clsTokenId: 101,
    sepTokenId: 102,
    padTokenId: 0,
    unkTokenId: 100,
  });

  test('can be constructed with a config', () => {
    assert.doesNotThrow(makeTokenizer);
  });

  test('encode returns inputIds and attentionMask', () => {
    const tokenizer = makeTokenizer();
    const result = tokenizer.encode('hello world', { addSpecialTokens: true });
    assert.ok(result.inputIds instanceof Int32Array, 'inputIds must be Int32Array');
    assert.ok(result.attentionMask instanceof Int32Array, 'attentionMask must be Int32Array');
    assert.strictEqual(result.inputIds.length, result.attentionMask.length);
  });

  test('encode with addSpecialTokens prepends CLS and appends SEP', () => {
    const tokenizer = makeTokenizer();
    const result = tokenizer.encode('hello world', { addSpecialTokens: true });
    // First token should be [CLS] = 101
    assert.strictEqual(result.inputIds[0], 101);
    // Last real token should be [SEP] = 102
    assert.strictEqual(result.inputIds[result.inputIds.length - 1], 102);
  });

  test('encode with empty string does not throw', () => {
    const tokenizer = makeTokenizer();
    assert.doesNotThrow(() => tokenizer.encode('', { addSpecialTokens: true }));
  });

  test('encode empty string returns at least one token (fallback pad)', () => {
    const tokenizer = makeTokenizer();
    const result = tokenizer.encode('', { addSpecialTokens: false });
    assert.ok(result.inputIds.length >= 1, 'must have at least one token');
  });

  test('decode returns a string without throwing', () => {
    const tokenizer = makeTokenizer();
    const result = tokenizer.encode('hello', { addSpecialTokens: true });
    assert.doesNotThrow(() => tokenizer.decode(result.inputIds));
    assert.strictEqual(typeof tokenizer.decode(result.inputIds), 'string');
  });

  test('vocabSize returns correct count', () => {
    const tokenizer = makeTokenizer();
    assert.strictEqual(tokenizer.vocabSize, 6);
  });
});

// ── 5. ImageProcessor ────────────────────────────────────────────────────────

describe('ImageProcessor', () => {
  test('imageDataToFloat32Array converts ImageData-like object to Float32Array', () => {
    const width = 2;
    const height = 2;
    const data = new Uint8ClampedArray(width * height * 4).fill(128);
    const imageData = { data, width, height };
    const result = dist.ImageProcessor.imageDataToFloat32Array(imageData);
    assert.ok(result instanceof Float32Array);
    // 3 channels (RGB) x width x height
    assert.strictEqual(result.length, 3 * width * height);
  });

  test('imageDataToFloat32Array values are in [0,1]', () => {
    const data = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255]);
    const result = dist.ImageProcessor.imageDataToFloat32Array({ data, width: 2, height: 1 });
    for (let i = 0; i < result.length; i++) {
      assert.ok(result[i] >= 0 && result[i] <= 1, `value at ${i} out of [0,1]: ${result[i]}`);
    }
  });
});

// ── 6. EdgeInfer.fromBuffer rejects clearly for invalid model data ────────────

describe('EdgeInfer.fromBuffer with invalid model', () => {
  test('fromBuffer rejects with Error for invalid model data (not a valid ONNX model)', async () => {
    const badBuffer = new ArrayBuffer(16);
    await assert.rejects(
      () => dist.EdgeInfer.fromBuffer(badBuffer),
      (err) => {
        assert.ok(err instanceof Error);
        return true;
      }
    );
  });
});

// ── 7. EventEmitter ──────────────────────────────────────────────────────────

describe('EventEmitter', () => {
  test('on/emit round-trip works', () => {
    const emitter = new dist.EventEmitter();
    let received = null;
    emitter.on('data', (v) => { received = v; });
    emitter.emit('data', 42);
    assert.strictEqual(received, 42);
  });

  test('multiple listeners all receive the event', () => {
    const emitter = new dist.EventEmitter();
    let count = 0;
    emitter.on('x', () => count++);
    emitter.on('x', () => count++);
    emitter.emit('x', null);
    assert.strictEqual(count, 2);
  });

  test('emitting an unregistered event does not throw', () => {
    const emitter = new dist.EventEmitter();
    assert.doesNotThrow(() => emitter.emit('noop', {}));
  });
});
