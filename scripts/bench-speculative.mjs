// SPDX-License-Identifier: MIT
// Mock-model speculative decoding throughput benchmark.
import { writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const { SpeculativeDecoder, MockLanguageModel } = await import(
  join(here, '..', 'dist', 'index.mjs')
);

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

const VOCAB = 32;
function logitsFor(tokens, peakProbability) {
  const last = tokens.length ? tokens[tokens.length - 1] : 0;
  const peak = (last + 1) % VOCAB;
  const rest = (1 - peakProbability) / (VOCAB - 1);
  const logits = new Float32Array(VOCAB);
  for (let token = 0; token < VOCAB; token++) {
    logits[token] = Math.log(token === peak ? peakProbability : rest);
  }
  return logits;
}

const TARGET_LATENCY_MS = 2.0;
const DRAFT_LATENCY_MS = 0.2;
const target = new MockLanguageModel((tokens) => logitsFor(tokens, 0.8), VOCAB, 0);
const draft = new MockLanguageModel((tokens) => logitsFor(tokens, 0.65), VOCAB, 0);

const tokensToGenerate = 100;
const targetOnlyMs = tokensToGenerate * TARGET_LATENCY_MS;
const decoder = new SpeculativeDecoder({
  draft,
  target,
  gamma: 5,
  rng: mulberry32(99),
});
const result = await decoder.generate([1], tokensToGenerate);

const modeledSpeculativeMs =
  result.stats.targetCalls * TARGET_LATENCY_MS +
  result.stats.draftCalls * DRAFT_LATENCY_MS;
const report = {
  model: { draft: 'mock-135m-shape', target: 'mock-1.5b-shape', vocab: VOCAB },
  latency_ms: { draft_forward: DRAFT_LATENCY_MS, target_forward: TARGET_LATENCY_MS },
  gamma: 5,
  tokens: tokensToGenerate,
  speculative: result.stats,
  modeled_speculative_ms: Number(modeledSpeculativeMs.toFixed(2)),
  target_only_estimate_ms: targetOnlyMs,
  speedup_vs_target_only: Number((targetOnlyMs / modeledSpeculativeMs).toFixed(2)),
  note: 'Analytical speedup from call counts and configured mock forward latencies; not a WebGPU measurement.',
};
console.log(JSON.stringify(report, null, 2));
writeFileSync(
  join(here, '..', 'bench-speculative.json'),
  JSON.stringify(report, null, 2) + '\n'
);
