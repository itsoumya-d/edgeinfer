# WGSL paged-attention design note

Status: **design only**. No shader, benchmark, or speedup claim is shipped in
this change. The current SDK explicitly documents that GPUBuffer-resident paged
KV is not implemented.

## Why a separate kernel path is required

`onnxruntime-web` owns the tensors returned by `session.run()`. A JS caller can
read `present.*` / `past_key_values.*` tensors, but cannot pin them in a
`GPUBuffer` across decoding steps without a CPU round-trip. Eliminating that
readback requires owning the attention computation, not wrapping ORT's decoder.

The kernel path therefore targets pre-exported projection weights and a
Python-side reference implementation, then compares token-for-token logits
against the ORT path before any performance number is published.

## KV cache layout

The cache is a flat `f16` storage buffer per layer:

```
KV[block_id][slot][kv_head][head_dim]
```

- `block_size = 16` tokens
- `slot = token_index % block_size`
- `num_blocks = ceil(max_seq_len / block_size)`
- A block table maps logical block index to physical block id, so sequences can
  grow without reallocating the whole cache.
- `past_len` and `block_table` live in a small uniform buffer updated once per
  decode step.

## Bindings

| Binding | Type | Purpose |
|---|---|---|
| 0 | `uniform` | `past_len`, `num_kv_heads`, `head_dim`, `scale`, `block_size` |
| 1 | `storage` (read) | query vector, `[num_heads, head_dim]` |
| 2 | `storage` (read) | K cache blocks |
| 3 | `storage` (read) | V cache blocks |
| 4 | `storage` (read) | block table |
| 5 | `storage` (read/write) | output vector and per-head `m`, `l` state |

Workgroup size: `64` invocations, one workgroup per `(head, head_dim tile)`.
Each invocation accumulates a `head_dim` slice and reduces through workgroup
shared memory. The first implementation should favour correctness over
occupancy; tuning is a later, measured step.

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
- Quantized KV cache (the first version is `f16`).
- Multi-query/grouped-query attention beyond the layout above; GQA is a
  follow-up once parity is established.

## Integration

When the gates pass, the shader and host code live under
`src/webgpu/kernels/`, behind a feature flag. The README will link to the
benchmark artifact and the exact hardware/software matrix. Until then, this
note is the only paged-attention artifact in the repository.
