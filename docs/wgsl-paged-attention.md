# WGSL paged-attention design note

Status: **f16 KV WGSL kernel implemented and browser-tested on synthetic
cases**. It passes 20/20 fixture cases on an NVIDIA Turing adapter (Chrome 156)
with max absolute error **4.79e-8**. No SmolLM2 token parity, validation-layer
trace, memory-within-5% measurement, or performance benchmark is shipped yet.
The SDK still documents that GPUBuffer-resident paged KV is not wired into
`EdgeInfer.generate()`.

## Why a separate kernel path is required

`onnxruntime-web` owns the tensors returned by `session.run()`. A JS caller can
read `present.*` / `past_key_values.*` tensors, but cannot pin them in a
`GPUBuffer` across decoding steps without a CPU round-trip. Eliminating that
readback requires owning the attention computation, not wrapping ORT's decoder.

The kernel path therefore targets pre-exported projection weights and a
Python-side reference implementation, then compares token-for-token logits
against the ORT path before any performance number is published.

## CPU reference and harness

- `scripts/paged_attention_reference.py` implements the float32 recurrence for
  a flat `KV[block_id][slot][kv_head][head_dim]` layout, with `block_size = 16`
  and a logical-to-physical block table.
- `scripts/paged_attention_parity.py` compares a candidate output against the
  reference using `atol=2e-3`, `rtol=2e-2`; it can also generate deterministic
  synthetic cases for harness testing.
- `scripts/test_paged_attention.py` covers block boundaries, block-table
  permutation, out-of-range slots, and the comparison tolerances.
- These are tensor-level checks. They do not prove SmolLM2 token parity,
  WebGPU correctness, or a speedup.

## ORT baseline

- `scripts/make_smollm2_prompts.py` tokenizes 20 fixed chat prompts with the
  local SmolLM2 tokenizer.
- `scripts/smollm2_ort_baseline.mjs` runs the public
  `HuggingFaceTB/SmolLM2-135M-Instruct` FP16 ONNX export through
  `onnxruntime-web` on the CPU provider, using `past_key_values.*` and
  `present.*` across 30 layers.
- `scripts/fixtures/smollm2-135m-20-baseline.json` records generated token IDs,
  chosen logits, and top-5 logits. It contains **542 generated tokens** across
  20 prompts (some stop at EOS before 32).
- This is an ORT token baseline, not WGSL parity. No paged-attention kernel is
  involved in producing it.

## KV cache layout

The cache is a flat `f16` storage buffer per layer:

```
KV[block_id][slot][kv_head][head_dim]
```

`block_size = 16` tokens, `slot = token_index % block_size`, and `num_blocks =
ceil(max_seq_len / block_size)`. A block table maps logical block index to
physical block id, so sequences can grow without reallocating the whole cache.
`past_len` and `block_table` live in a small uniform buffer updated once per
decode step. The host converts `Float32Array` inputs to IEEE-754 half bits; the
Python reference fixture is generated from the same f16-rounded K/V values.

## Bindings

| Binding | Type | Purpose |
|---|---|---|
| 0 | `uniform` | `past_len`, query/KV heads, `head_dim`, `scale`, `block_size`, `layer`, `num_blocks` |
| 1 | `storage` (read) | query vector, `[num_query_heads, head_dim]` |
| 2 | `storage` (read) | K cache blocks (`f16`) |
| 3 | `storage` (read) | V cache blocks (`f16`) |
| 4 | `storage` (read) | block table |
| 5 | `storage` (read/write) | output vector |
| 6 | `storage` (read/write) | status word: 0 = ok, 1 = non-finite score |

Workgroup size: `64` invocations, one workgroup per query head. Each invocation
accumulates one or two `head_dim` elements and the `q·k` dot product is reduced
through workgroup shared memory. The kernel favours correctness over occupancy;
tuning is a later, measured step.

## Online softmax

For each KV block, with running max `m` and running denominator `l`:

```
s = dot(q, k) * scale
m_new = max(m, max(s))
alpha = exp(m - m_new)
p = exp(s - m_new)
l = l * alpha + sum(p)
acc = acc * alpha + sum(p * v)
```

The final output is `acc / l`. This never materializes the `N x N` attention
matrix and keeps state in registers/shared memory. The Python reference must
implement the same recurrence in `float32`, and the WGSL path must match it
within a documented tolerance (`atol=2e-3`, `rtol=2e-2` on logits) before it is
considered correct.

## WGSL kernel and browser result

- `src/webgpu/kernels/paged_attention.wgsl` implements one workgroup per query
  head, 64 invocations, online softmax, GQA head mapping, and a status word for
  non-finite scores.
- `src/webgpu/paged-attention.ts` is the host driver. K/V buffers are created
  with `STORAGE | COPY_DST` only; they have no `COPY_SRC` or `MAP_READ` usage.
  Only the small output and status buffers are copied back.
- `scripts/build_parity_fixture.py` builds the 20-case synthetic fixture with
  float32 reference outputs.
- `scripts/webgpu-paged-attention/index.html` runs the kernel against that
  fixture; open it from a local HTTP server (not `file://`).
- `scripts/build-wgsl-harness.mjs` rebuilds the standalone browser bundle.

Measured on 2026-11-03 in Chrome 156 on the RTX 2060 (NVIDIA Turing adapter):

| Cases | Passed | Failed | Max absolute error | Zero-Readback |
|---:|---:|---:|---:|---|
| 20 | **20** | 0 | **4.79e-8** | **40/40 KV creates clean** |

The fixture includes MHA and GQA cases. `zero_readback` is a runtime
buffer-usage capability check: no KV-sized buffer is created with `COPY_SRC` or
`MAP_READ`; it is not a validation-layer `mapAsync` trace. The fixture is
synthetic attention tensors, not SmolLM2 logits.

The kernel is exported as an opt-in building block (`runPagedAttention`) and is
not wired into `EdgeInfer.generate()`; the ORT path remains the default.

## Acceptance gates

1. **Parity:** greedy decode for 32 tokens matches the ORT baseline on
   SmolLM2-135M-Instruct for at least 20 prompts.
2. **Zero readback:** a WebGPU validation layer trace shows no
   `mapAsync`/`getMappedRange` on KV buffers during decode.
3. **Memory:** peak KV allocation is within 5% of
   `2 * layers * kv_heads * head_dim * max_seq_len * 2 bytes` plus block-table
   overhead.
4. **Performance:** report p50/p95 decode latency on at least two devices, with
   the browser and driver versions. No speedup is claimed before this gate.
5. **Fallback:** if `requestAdapter()` or the shader compilation fails, the SDK
   returns to the existing ORT path and surfaces a machine-readable
   `WEBGPU_UNAVAILABLE` capability flag.

## Non-goals

- Training or fine-tuning.
- Quantized KV cache; the current kernel uses f16 storage and f32 accumulation.
- MQA and GQA-specific tuning beyond the standard head mapping; the browser
  fixture includes MHA and GQA cases.

## Integration

When the gates pass, the shader and host code live under
`src/webgpu/kernels/`, behind a feature flag. The README will link to the
benchmark artifact and the exact hardware/software matrix. Until then, this
note is the only paged-attention artifact in the repository.
