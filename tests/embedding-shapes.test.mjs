import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';

const require = createRequire(import.meta.url);
const { EdgeInfer, Tokenizer } = require('../dist/index.js');
const fixtures = JSON.parse(readFileSync(new URL('./fixtures/embedding-models.json', import.meta.url)));
const tokenizerConfig = {
  vocab: { '[CLS]': 101, '[SEP]': 102, '[PAD]': 0, '[UNK]': 100, hello: 1, world: 2 },
};

// These tests execute the public loading/inference path with real local ONNX
// Constant models. No model download, GPU, credentials, or trained weights.
async function loadFixture(t, name) {
  const bytes = Buffer.from(fixtures[name].base64, 'base64');
  const model = await EdgeInfer.fromBuffer(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    { executionProviders: ['wasm'], tokenizerConfig }
  );
  t.after(() => model.dispose());
  return model;
}

function normalized(values) {
  const norm = Math.hypot(...values);
  return values.map(value => norm === 0 ? value : value / norm);
}

function closeTo(actual, expected) {
  assert.ok(actual instanceof Float32Array);
  assert.equal(actual.length, expected.length, 'embedding dimension must match the model output');
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) < 1e-6,
      `component ${i}: expected ${expected[i]}, got ${actual[i]}`);
  }
}

for (const fixture of ['pooled', 'pooled-vector']) {
  test(`${fixture}: retain all dimensions when token count divides the embedding size`, async (t) => {
    const model = await loadFixture(t, fixture);
    // Four tokens: [CLS], hello, world, [SEP]. Eight output elements are NOT
    // four two-dimensional token vectors when ONNX says [1, 8] or [8].
    closeTo(await model.embed('hello world'), normalized([1, 2, 3, 4, 5, 6, 7, 8]));
  });
}

test('pooled embeddings have the same dimensions and normalization across input lengths', async (t) => {
  const model = await loadFixture(t, 'pooled');
  const expected = normalized([1, 2, 3, 4, 5, 6, 7, 8]);
  closeTo(await model.embed('hello'), expected); // Three tokens, not a divisor of eight.
  closeTo(await model.embed(''), expected); // Two special tokens.
});

test('token embeddings use the declared sequence and hidden dimensions', async (t) => {
  const model = await loadFixture(t, 'tokens');
  closeTo(await model.embed('hello world'), normalized([109 / 4, 212 / 4]));
});

test('token pooling excludes masked padding', async (t) => {
  const model = await loadFixture(t, 'tokens');
  class PaddedTokenizer extends Tokenizer {
    encode(text, options) { return super.encode(text, { ...options, padding: true, maxLength: 4 }); }
  }
  model.setTokenizer(new PaddedTokenizer(tokenizerConfig));
  closeTo(await model.embed('hello'), [0.6, 0.8]); // Mean of [1,2], [3,4], [5,6].
});

test('zero token vectors remain finite zero vectors', async (t) => {
  const model = await loadFixture(t, 'zero-tokens');
  closeTo(await model.embed('hello world'), [0, 0]);
});

test('an all-masked sequence returns a zero vector with the declared hidden dimension', async (t) => {
  const model = await loadFixture(t, 'tokens');
  class MaskedTokenizer extends Tokenizer {
    encode(text, options) {
      const result = super.encode(text, options);
      result.attentionMask.fill(0);
      return result;
    }
  }
  model.setTokenizer(new MaskedTokenizer(tokenizerConfig));
  closeTo(await model.embed('hello world'), [0, 0]);
});

test('mismatched token output length rejects instead of guessing a pooled vector', async (t) => {
  const model = await loadFixture(t, 'tokens');
  await assert.rejects(() => model.embed('hello'), /sequence length.*4.*attention mask.*3/i);
});

for (const fixture of ['batched', 'rank-four']) {
  test(`${fixture}: reject unsupported embedding shapes explicitly`, async (t) => {
    const model = await loadFixture(t, fixture);
    await assert.rejects(() => model.embed('hello world'), /unsupported embedding output shape/i);
  });
}

test('integer output rejects rather than claiming it is a float embedding', async (t) => {
  const model = await loadFixture(t, 'integer');
  await assert.rejects(() => model.embed('hello world'), /embedding output.*float32/i);
});

test('Matryoshka truncation uses the complete pooled embedding before normalization', async (t) => {
  const model = await loadFixture(t, 'pooled');
  closeTo(await model.embedMatryoshka('hello world', 4), normalized([1, 2, 3, 4]));
});

test('predict still returns raw flattened output values', async (t) => {
  const model = await loadFixture(t, 'pooled');
  const result = await model.predict({
    input_ids: new Int32Array([101, 1, 2, 102]),
    attention_mask: new Int32Array([1, 1, 1, 1]),
  }, { input_ids: [1, 4], attention_mask: [1, 4] });
  assert.deepEqual(result.embedding, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
});
