# edgeinfer v2 Results

## Environment

| Item | Value |
|---|---|
| Date | 2026-10-31 |
| OS | macOS (Apple Silicon) |
| Node | 26.3.0 |
| Tests | 106/106 passing |
| License | MIT |
| Version | 2.0.0 |

## Test coverage

`npm test` covers the existing tokenizer/ONNX/KV-cache/download tests plus the
v2 additions:

- INT8 quantization and Matryoshka truncation/normalization,
- two-stage search versus exact full search,
- Recall@10 across dimensions,
- IndexedDB persistence (build → persist → reload → search),
- speculative accept/reject/bonus/EOS paths,
- residual resampling and state truncation,
- the ONNX causal-LM adapter's interface contract,
- streaming `generateStream()` token events and abort handling,
- `chat()` / `chatStream()` prompt templating and validation,
- empirical `probeVRAM()` binary search and `detectCapabilities()` integration.

## Matryoshka synthetic benchmark

Command: `npm run bench:matryoshka`

| Metric | Value |
|---|---:|
| Corpus | 100,000 chunks × 768 dims |
| Build time | 1.76 s |
| Coarse scan p50 | 25.08 ms |
| Coarse scan p95 | 59.41 ms |
| Throughput | 33.6 queries/s |
| Top-10 hit rate | 0.90 |
| Recall@10, d=64 | 0.725 |
| Recall@10, d=128 | 0.750 |
| Recall@10, d=256 | 0.7625 |
| Recall@10, d=512 | 0.8375 |
| Recall@10, d=768 | 1.000 |

The vectors and queries are synthetic clustered data. This is **not** an MTEB
Retrieval Recall@10 result and must not be presented as one. The dimension
curve demonstrates the expected truncation trade-off inside the index.

## Speculative decoding benchmark

Command: `npm run bench:speculative`

Mock 32-token vocabulary, `gamma=5`, prompt length 1, 100 generated tokens:

| Metric | Value |
|---|---:|
| Draft tokens proposed | 114 |
| Accepted draft tokens | 78 |
| Rejected draft tokens | 11 |
| Bonus tokens | 11 |
| Acceptance rate | 68.4% |
| Target calls | 45 |
| Target calls / output token | 0.45 |
| Modeled target-only time | 200 ms |
| Modeled speculative time | 121.8 ms |
| Modeled speedup | 1.64× |

The speedup is computed analytically from the call counts and the configured
mock forward latencies (draft 0.2 ms, target 2.0 ms). It is **not** a measured
WebGPU tokens/sec or TTFT result.

## WebGPU probe (unit-level)

`RuntimeManager.probeVRAM()` is tested with a mocked adapter that allocates up to
100 MiB. The binary search returns 100 MiB at 4 MiB resolution, and a real
adapter's `detectCapabilities()` records the same probe value instead of the
256 MiB / 128 MiB adapter baselines. These are deterministic mock results, not a
measurement of this machine's GPU.

## Not executed

- **MTEB retrieval** requires an embedding model and MTEB data; no HF/API
  credentials were configured.
- **Playwright WebGPU** benchmarks: Chromium in this environment exposes no
  `navigator.gpu` in either headless or headed mode with the available flags, so
  no WebGPU TTFT/tokens-per-second numbers are claimed.
- **Paged GPUBuffer-resident KV** requires a custom WebGPU attention kernel;
  the ONNX-runtime wrapper keeps cache tensors at the ORT tensor boundary.
- **Hugging Face Space / npm publication** require authentication; the repo is
  MIT and package version is 2.0.0, but it is not published.
- **KV-cache pinned buffers**: the `LanguageModel` interface and
  `truncate()` contract support cache-aware adapters; the included
  `OnnxCausalLM` is a correctness-first recompute adapter. A KV-cache ONNX
  adapter still needs model-specific input/output naming.

## Reproduce

```bash
cd repos/edgeinfer
npm run build
npm test
npm run bench:matryoshka
npm run bench:speculative
```
