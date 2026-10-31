// SPDX-License-Identifier: MIT
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const {
  SpeculativeDecoder,
  MockLanguageModel,
  residualDistribution,
  softmax,
} = await import(join(__dirname, '../dist/index.mjs'));

function logDist(probabilities) {
  return Float32Array.from(probabilities.map((p) => (p > 0 ? Math.log(p) : -50)));
}

class StaticModel {
  constructor(probabilities, vocabSize = probabilities.length) {
    this.vocabSize = vocabSize;
    this.logits = logDist(probabilities);
    this.truncations = [];
  }
  async prefill(tokens) {
    return { tokens: [...tokens] };
  }
  async decode(state, tokens) {
    const logits = [];
    for (let i = 0; i <= tokens.length; i++) logits.push(new Float32Array(this.logits));
    return { state: { tokens: [...state.tokens, ...tokens] }, logits };
  }
  truncate(state, length) {
    this.truncations.push(length);
    return { tokens: state.tokens.slice(0, length) };
  }
}

describe('speculative primitives', () => {
  test('softmax normalizes logits', () => {
    const probabilities = softmax(Float32Array.from([0, 0, 0, 0]));
    assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-6);
  });

  test('residual distribution removes draft mass', () => {
    const target = logDist([0.5, 0.5, 0, 0]);
    const draft = Float32Array.from([1, 0, 0, 0]);
    const residual = residualDistribution(target, draft);
    assert.ok(Math.abs(residual[0]) < 1e-6);
    assert.ok(Math.abs(residual[1] - 1) < 1e-6);
    assert.ok(Math.abs(residual.reduce((a, b) => a + b, 0) - 1) < 1e-6);
  });
});

describe('SpeculativeDecoder', () => {
  test('all accepted drafts produce gamma plus a bonus token', async () => {
    const draft = new StaticModel([0.25, 0.25, 0.25, 0.25]);
    const target = new StaticModel([0.25, 0.25, 0.25, 0.25]);
    const decoder = new SpeculativeDecoder({
      draft,
      target,
      gamma: 2,
      rng: () => 0,
    });
    const result = await decoder.generate([9], 3);
    assert.equal(result.tokens.length, 3);
    assert.equal(result.stats.acceptedTokens, 2);
    assert.equal(result.stats.bonusTokens, 1);
    assert.equal(result.stats.rejectedTokens, 0);
    assert.equal(result.stats.acceptanceRate, 1);
    assert.ok(result.stats.targetCalls < 3);
  });

  test('a rejected draft resamples from the residual distribution', async () => {
    const draft = new StaticModel([1, 0, 0, 0]);
    const target = new StaticModel([0.5, 0.5, 0, 0]);
    const decoder = new SpeculativeDecoder({
      draft,
      target,
      gamma: 1,
      rng: () => 0.9,
    });
    const result = await decoder.generate([9], 1);
    assert.deepEqual(result.tokens, [1]);
    assert.equal(result.stats.acceptedTokens, 0);
    assert.equal(result.stats.rejectedTokens, 1);
    assert.deepEqual(draft.truncations, [1]);
  });

  test('accepts when the target has enough mass', async () => {
    const draft = new StaticModel([1, 0, 0, 0]);
    const target = new StaticModel([0.8, 0.2, 0, 0]);
    const decoder = new SpeculativeDecoder({
      draft,
      target,
      gamma: 1,
      rng: () => 0.5,
    });
    const result = await decoder.generate([9], 1);
    assert.deepEqual(result.tokens, [0]);
    assert.equal(result.stats.acceptedTokens, 1);
    assert.equal(result.stats.bonusTokens, 0);
  });

  test('stops on EOS without adding a bonus token', async () => {
    const draft = new StaticModel([1, 0, 0, 0]);
    const target = new StaticModel([1, 0, 0, 0]);
    const decoder = new SpeculativeDecoder({
      draft,
      target,
      gamma: 2,
      rng: () => 0,
      eosTokenId: 0,
    });
    const result = await decoder.generate([9], 4);
    assert.deepEqual(result.tokens, [0]);
    assert.equal(result.stats.bonusTokens, 0);
  });

  test('rejects mismatched vocabularies and invalid gamma', () => {
    assert.throws(
      () =>
        new SpeculativeDecoder({
          draft: new StaticModel([1, 0]),
          target: new StaticModel([1, 0, 0], 3),
        })
    );
    assert.throws(
      () =>
        new SpeculativeDecoder({
          draft: new StaticModel([1, 0]),
          target: new StaticModel([1, 0]),
          gamma: 0,
        })
    );
  });

  test('mock language model supports the empty-token forward call', async () => {
    const model = new MockLanguageModel(() => Float32Array.from([0, 1, 0]), 3);
    const state = await model.prefill([1, 2]);
    const result = await model.decode(state, []);
    assert.equal(result.logits.length, 1);
    assert.equal(result.state.tokens.length, 2);
  });
});
