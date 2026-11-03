# WebGPU paged-attention parity harness

This is a local browser harness, not a package entry point.

## Run

From the repository root:

```bash
node scripts/build-wgsl-harness.mjs
python -m http.server 8767 --bind 127.0.0.1
```

Open `http://127.0.0.1:8767/scripts/webgpu-paged-attention/index.html` in a
WebGPU-capable browser. The page loads `paged-attention-20.json`, runs the
`runPagedAttention` kernel for each case, and compares against the Python
reference with `atol=2e-3`, `rtol=2e-2`. Results are also written to
`window.__pagedAttentionResult`.

## What the result means

- 20/20 synthetic tensor cases passed on Chrome 156, NVIDIA Turing (RTX 2060),
  max absolute error `4.79e-8`, f16 KV storage.
- The fixture includes MHA and GQA cases.
- The harness also checks that no KV-sized buffer is created with `COPY_SRC` or
  `MAP_READ`; it is a capability check, not a validation-layer trace.
- This is **not** SmolLM2 token parity, a validation-layer trace, a memory
  measurement, or a speed benchmark.
