// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath
//
// Speculative decoding with a small draft model and a larger target model.
// The target verifies gamma draft tokens in one forward pass. Accepted tokens
// keep their draft distribution; a rejection resamples from the residual
// max(0, p_target - p_draft) and truncates both KV-cache states.

export interface ModelState {
  tokens: number[];
  /** Opaque KV-cache handle for cache-aware adapters. */
  cache?: unknown;
}

export interface DecodeResult {
  state: ModelState;
  /**
   * One next-token distribution for each input position, plus one distribution
   * after the last input token. Length = tokens.length + 1. Calling decode()
   * with an empty token array returns the distribution after the current state.
   */
  logits: Float32Array[];
}

export interface LanguageModel {
  readonly vocabSize: number;
  prefill(tokens: number[]): Promise<ModelState>;
  decode(state: ModelState, tokens: number[]): Promise<DecodeResult>;
  /**
   * Drop cache entries after `length` total tokens. Cache-aware adapters must
   * implement this; stateless models can just slice the token list.
   */
  truncate(state: ModelState, length: number): ModelState;
}

export interface SpeculativeOptions {
  draft: LanguageModel;
  target: LanguageModel;
  gamma?: number;
  rng?: () => number;
  eosTokenId?: number;
}

export interface SpeculativeStats {
  outputTokens: number;
  draftTokens: number;
  acceptedTokens: number;
  bonusTokens: number;
  rejectedTokens: number;
  targetCalls: number;
  draftCalls: number;
  acceptanceRate: number;
  targetCallsPerToken: number;
  elapsedMs: number;
  tokensPerSecond: number;
}

export interface SpeculativeGenerationResult {
  tokens: number[];
  stats: SpeculativeStats;
}

export class SpeculativeDecoder {
  readonly draft: LanguageModel;
  readonly target: LanguageModel;
  readonly gamma: number;
  private readonly rng: () => number;
  private readonly eosTokenId?: number;

  constructor(options: SpeculativeOptions) {
    if (options.draft.vocabSize !== options.target.vocabSize) {
      throw new Error("Draft and target models must share a vocabulary size");
    }
    const gamma = options.gamma ?? 5;
    if (!Number.isInteger(gamma) || gamma < 1) {
      throw new Error("gamma must be a positive integer");
    }
    this.draft = options.draft;
    this.target = options.target;
    this.gamma = gamma;
    this.rng = options.rng ?? Math.random;
    this.eosTokenId = options.eosTokenId;
  }

  async generate(
    prompt: number[],
    maxNewTokens = 64,
    { temperature = 1 }: { temperature?: number } = {}
  ): Promise<SpeculativeGenerationResult> {
    if (prompt.length === 0) throw new Error("prompt must not be empty");
    if (maxNewTokens < 1) throw new Error("maxNewTokens must be positive");
    if (temperature <= 0) throw new Error("temperature must be positive");

    const started = performance.now();
    let draftState = await this.draft.prefill(prompt);
    let targetState = await this.target.prefill(prompt);

    const output: number[] = [];
    let draftTokens = 0;
    let acceptedTokens = 0;
    let bonusTokens = 0;
    let rejectedTokens = 0;
    let targetCalls = 0;
    let draftCalls = 0;

    while (output.length < maxNewTokens) {
      const gamma = Math.min(this.gamma, maxNewTokens - output.length);
      const draft = await this.draftBatch(draftState, gamma, temperature, draftCalls);
      draftState = draft.state;
      draftCalls = draft.calls;
      draftTokens += draft.tokens.length;

      const prefixLength = targetState.tokens.length;
      const verification = await this.target.decode(targetState, draft.tokens);
      targetCalls++;
      targetState = verification.state;

      let rejected = false;
      for (let i = 0; i < draft.tokens.length; i++) {
        const token = draft.tokens[i];
        const q = draft.probabilities[i][token];
        const p = probability(verification.logits[i], token, temperature);
        if (this.rng() < Math.min(1, p / q)) {
          output.push(token);
          acceptedTokens++;
          if (token === this.eosTokenId || output.length >= maxNewTokens) break;
          continue;
        }

        const residual = residualDistribution(
          verification.logits[i],
          draft.probabilities[i],
          temperature
        );
        const replacement = sample(residual, this.rng());
        output.push(replacement);
        rejectedTokens++;
        rejected = true;

        const keep = prefixLength + i;
        targetState = this.target.truncate(targetState, keep);
        draftState = this.draft.truncate(draftState, keep);
        const targetAdvance = await this.target.decode(targetState, [replacement]);
        targetCalls++;
        targetState = targetAdvance.state;
        const draftAdvance = await this.draft.decode(draftState, [replacement]);
        draftCalls++;
        draftState = draftAdvance.state;
        break;
      }

      if (rejected) continue;
      if (output.length >= maxNewTokens || output[output.length - 1] === this.eosTokenId) break;

      const bonusDistribution = softmax(
        verification.logits[draft.tokens.length],
        temperature
      );
      const bonus = sample(bonusDistribution, this.rng());
      output.push(bonus);
      bonusTokens++;
      const targetAdvance = await this.target.decode(targetState, [bonus]);
      targetCalls++;
      targetState = targetAdvance.state;
      const draftAdvance = await this.draft.decode(draftState, [bonus]);
      draftCalls++;
      draftState = draftAdvance.state;
      if (bonus === this.eosTokenId) break;
    }

    const elapsedMs = performance.now() - started;
    const stats: SpeculativeStats = {
      outputTokens: output.length,
      draftTokens,
      acceptedTokens,
      bonusTokens,
      rejectedTokens,
      targetCalls,
      draftCalls,
      acceptanceRate: draftTokens === 0 ? 0 : acceptedTokens / draftTokens,
      targetCallsPerToken: output.length === 0 ? 0 : targetCalls / output.length,
      elapsedMs,
      tokensPerSecond: elapsedMs === 0 ? 0 : (output.length / elapsedMs) * 1000,
    };
    return { tokens: output, stats };
  }

  private async draftBatch(
    state: ModelState,
    gamma: number,
    temperature: number,
    calls: number
  ): Promise<{ state: ModelState; tokens: number[]; probabilities: Float32Array[]; calls: number }> {
    let current = state;
    const tokens: number[] = [];
    const probabilities: Float32Array[] = [];

    const first = await this.draft.decode(current, []);
    calls++;
    let nextDistribution = first.logits[0];
    for (let i = 0; i < gamma; i++) {
      const distribution = softmax(nextDistribution, temperature);
      const token = sample(distribution, this.rng());
      tokens.push(token);
      probabilities.push(distribution);
      const advanced = await this.draft.decode(current, [token]);
      calls++;
      current = advanced.state;
      nextDistribution = advanced.logits[advanced.logits.length - 1];
    }
    return { state: current, tokens, probabilities, calls };
  }
}

export function softmax(logits: Float32Array, temperature = 1): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const output = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp((logits[i] - max) / temperature);
    output[i] = value;
    sum += value;
  }
  if (sum === 0) return output;
  for (let i = 0; i < output.length; i++) output[i] /= sum;
  return output;
}

export function probability(logits: Float32Array, token: number, temperature = 1): number {
  return softmax(logits, temperature)[token] ?? 0;
}

export function sample(distribution: Float32Array, random: number): number {
  let cumulative = 0;
  for (let i = 0; i < distribution.length; i++) {
    cumulative += distribution[i];
    if (random <= cumulative) return i;
  }
  return distribution.length - 1;
}

export function residualDistribution(
  targetLogits: Float32Array,
  draftProbabilities: Float32Array,
  temperature = 1
): Float32Array {
  const p = softmax(targetLogits, temperature);
  const q = draftProbabilities;
  const residual = new Float32Array(p.length);
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    const value = Math.max(0, p[i] - q[i]);
    residual[i] = value;
    sum += value;
  }
  if (sum === 0) return p;
  for (let i = 0; i < residual.length; i++) residual[i] /= sum;
  return residual;
}

export type LogitPolicy = (tokens: number[]) => Float32Array;

/** Deterministic mock used by tests and the throughput benchmark. */
export class MockLanguageModel implements LanguageModel {
  constructor(
    private readonly policy: LogitPolicy,
    readonly vocabSize: number,
    private readonly latencyMs = 0
  ) {}

  async prefill(tokens: number[]): Promise<ModelState> {
    await delay(this.latencyMs);
    return { tokens: tokens.slice() };
  }

  async decode(state: ModelState, tokens: number[]): Promise<DecodeResult> {
    await delay(this.latencyMs);
    const logits: Float32Array[] = [];
    for (let i = 0; i <= tokens.length; i++) {
      logits.push(this.policy([...state.tokens, ...tokens.slice(0, i)]));
    }
    return { state: { tokens: [...state.tokens, ...tokens] }, logits };
  }

  truncate(state: ModelState, length: number): ModelState {
    return { ...state, tokens: state.tokens.slice(0, length), cache: undefined };
  }
}

function delay(ms: number): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, ms));
}
