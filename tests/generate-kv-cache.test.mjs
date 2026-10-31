// Exercises the autoregressive loop and KV-cache reuse with a deterministic mock
// ONNX session. No model download or GPU is required.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const { EdgeInfer, Tokenizer } = await import(join(__dirname, '../dist/index.mjs'));

test('generate reuses KV-cache inputs and stops at EOS', async () => {
  const tokenizer = new Tokenizer({ vocab: { a: 2, b: 3 }, unkTokenId: 99 });
  const seen = [];
  let calls = 0;

  const session = {
    inputNames: [
      'input_ids',
      'attention_mask',
      'past_key_values.0.key',
      'past_key_values.0.value',
    ],
    outputNames: ['logits', 'present.0.key', 'present.0.value'],
    inputMetadata: { input_ids: { type: 'int32' } },
    async run(inputs) {
      seen.push(inputs);
      const sequenceLength = inputs.input_ids.data.length;
      const vocabSize = 4;
      const data = new Float32Array(sequenceLength * vocabSize);
      const pick = calls === 0 ? 3 : 2;
      for (let t = 0; t < sequenceLength; t++) data[t * vocabSize + pick] = 10;
      calls++;

      return {
        logits: { data, dims: [1, sequenceLength, vocabSize], type: 'float32' },
        'present.0.key': {
          data: new Float32Array(1),
          dims: [1, 1, 1, 1],
          type: 'float32',
        },
        'present.0.value': {
          data: new Float32Array(1),
          dims: [1, 1, 1, 1],
          type: 'float32',
        },
      };
    },
  };

  // The constructor is private in TypeScript but callable from JavaScript; this
  // is the same shape fromBuffer() builds after a successful session load.
  const model = new EdgeInfer(session, 0, 'wasm', false, tokenizer);
  const result = await model.generate('a', { maxNewTokens: 5, eosTokenId: 2 });

  assert.deepEqual(result.tokenIds, [3, 2]);
  assert.equal(calls, 2);

  // First pass: [CLS] a [SEP].
  assert.equal(seen[0].input_ids.data.length, 3);
  // Second pass: one new token plus the cached key/value tensors.
  assert.equal(seen[1].input_ids.data.length, 1);
  assert.ok(seen[1]['past_key_values.0.key']);
  assert.ok(seen[1]['past_key_values.0.value']);
  assert.equal(seen[1].attention_mask.data.length, 4);
});
