// Streaming generation and chat templates against a deterministic mock session.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const { EdgeInfer, Tokenizer } = await import(join(__dirname, '../dist/index.mjs'));

function createSession({ tokens = [3, 2], vocabSize = 4 } = {}) {
  let calls = 0;
  const seen = [];
  return {
    seen,
    get calls() { return calls; },
    inputNames: [
      'input_ids',
      'attention_mask',
      'past_key_values.0.key',
      'past_key_values.0.value'
    ],
    outputNames: ['logits', 'present.0.key', 'present.0.value'],
    inputMetadata: { input_ids: { type: 'int32' } },
    async run(inputs) {
      seen.push(inputs);
      const sequenceLength = inputs.input_ids.data.length;
      const data = new Float32Array(sequenceLength * vocabSize);
      const pick = tokens[Math.min(calls, tokens.length - 1)];
      for (let t = 0; t < sequenceLength; t++) data[t * vocabSize + pick] = 10;
      calls++;
      return {
        logits: { data, dims: [1, sequenceLength, vocabSize], type: 'float32' },
        'present.0.key': { data: new Float32Array(1), dims: [1, 1, 1, 1], type: 'float32' },
        'present.0.value': { data: new Float32Array(1), dims: [1, 1, 1, 1], type: 'float32' }
      };
    }
  };
}

function createModel(options) {
  const tokenizer = new Tokenizer({ vocab: { a: 2, b: 3 }, unkTokenId: 99 });
  const session = createSession(options);
  const model = new EdgeInfer(session, 0, 'wasm', false, tokenizer);
  return { model, session, tokenizer };
}

describe('generateStream', () => {
  test('yields each token with an index and decoded delta', async () => {
    const { model, tokenizer } = createModel({ tokens: [3, 2] });
    const events = [];
    for await (const event of model.generateStream('a', { maxNewTokens: 5, eosTokenId: 2 })) {
      events.push(event);
    }
    assert.deepEqual(events.map((event) => event.tokenId), [3, 2]);
    assert.deepEqual(events.map((event) => event.index), [0, 1]);
    assert.equal(events[0].text, tokenizer.decode([3]));
    assert.equal(events[1].text, tokenizer.decode([2]));
  });

  test('generate() collects the same stream', async () => {
    const { model, tokenizer } = createModel({ tokens: [3, 2] });
    const result = await model.generate('a', { maxNewTokens: 5, eosTokenId: 2 });
    assert.deepEqual(result.tokenIds, [3, 2]);
    assert.equal(result.text, tokenizer.decode([3, 2]));
  });

  test('zero maxNewTokens yields nothing', async () => {
    const { model, session } = createModel({ tokens: [3] });
    const events = [];
    for await (const event of model.generateStream('a', { maxNewTokens: 0 })) {
      events.push(event);
    }
    assert.deepEqual(events, []);
    assert.equal(session.calls, 0);
  });

  test('an aborted signal stops before the first decode', async () => {
    const { model, session } = createModel({ tokens: [3] });
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(async () => {
      for await (const _ of model.generateStream('a', { signal: controller.signal })) {
        // unreachable
      }
    }, { name: 'AbortError' });
    assert.equal(session.calls, 0);
  });
});

describe('chat', () => {
  test('builds the default role-marker prompt and collects tokens', async () => {
    const { model, tokenizer } = createModel({ tokens: [3] });
    const result = await model.chat(
      [
        { role: 'system', content: 'be nice' },
        { role: 'user', content: 'hi' }
      ],
      { maxNewTokens: 2, eosTokenId: 99 }
    );
    assert.equal(
      result.prompt,
      '<|system|>be nice\n<|user|>hi\n<|assistant|>'
    );
    assert.deepEqual(result.tokenIds, [3, 3]);
    assert.equal(result.text, tokenizer.decode([3, 3]));
  });

  test('chatStream yields the same tokens', async () => {
    const { model } = createModel({ tokens: [3] });
    const events = [];
    for await (const event of model.chatStream(
      [{ role: 'user', content: 'hi' }],
      { maxNewTokens: 2, eosTokenId: 99 }
    )) {
      events.push(event.tokenId);
    }
    assert.deepEqual(events, [3, 3]);
  });

  test('rejects malformed messages', async () => {
    const { model } = createModel({ tokens: [3] });
    await assert.rejects(() => model.chat([], {}), /at least one message/);
    await assert.rejects(
      () => model.chat([{ role: 'tool', content: 'x' }], {}),
      /invalid role/
    );
    await assert.rejects(
      () => model.chat([{ role: 'user', content: 42 }], {}),
      /content must be a string/
    );
  });
});
