// Regression tests for cased BPE wiring and int64 tensor outputs.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const { Tokenizer } = await import(join(__dirname, '../dist/index.mjs'));

const vocab = {
  h: 0,
  e: 1,
  l: 2,
  o: 3,
  he: 4,
  hel: 5,
  hell: 6,
  hello: 7,
  Hello: 8,
  '<unk>': 99,
};

const merges = [
  ['h', 'e'],
  ['he', 'l'],
  ['hel', 'l'],
  ['hell', 'o'],
];

describe('BPE tokenizer', () => {
  test('applies merges and preserves case when a merge table is present', () => {
    const tokenizer = new Tokenizer({ vocab, merges, unkTokenId: 99 });
    const lower = tokenizer.encode('hello', { addSpecialTokens: false });
    assert.deepEqual(Array.from(lower.inputIds), [7]);

    const cased = tokenizer.encode('Hello', { addSpecialTokens: false });
    assert.deepEqual(Array.from(cased.inputIds), [8]);
  });

  test('an explicit lowercase normalizer still lowercases BPE input', () => {
    const tokenizer = new Tokenizer({ vocab, merges, lowercase: true, unkTokenId: 99 });
    const encoding = tokenizer.encode('Hello', { addSpecialTokens: false });
    assert.deepEqual(Array.from(encoding.inputIds), [7]);
  });

  test('int64 views mirror the int32 token and attention tensors', () => {
    const tokenizer = new Tokenizer({ vocab, merges, unkTokenId: 99 });
    const encoding = tokenizer.encode('hello', { addSpecialTokens: true });

    assert.deepEqual(
      Array.from(encoding.inputIdsInt64),
      Array.from(encoding.inputIds, (value) => BigInt(value))
    );
    assert.deepEqual(
      Array.from(encoding.attentionMaskInt64),
      Array.from(encoding.attentionMask, (value) => BigInt(value))
    );
    assert.equal(encoding.inputIdsInt64.constructor, BigInt64Array);
  });
});
