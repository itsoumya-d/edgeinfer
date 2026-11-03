// GENERATED from src/webgpu/kernels/paged_attention.wgsl — keep in sync.
// tests/paged-attention-wgsl.test.mjs asserts this string equals the .wgsl file.

/** WGSL source for the paged-attention decode kernel (float32, single layer). */
export const PAGED_ATTENTION_WGSL: string = /* wgsl */ `// Paged attention (decode-time), float32, single layer.
// Layout: KV[block_id][slot][kv_head][head_dim], block_size = 16 (uniform).
// MHA and GQA: query head h reads KV head h / (num_query_heads / num_kv_heads).
//
// One workgroup per query head; 64 invocations. The kernel keeps the running
// max m and denominator l in registers (replicated across threads) and reduces
// q·k with workgroup shared memory. The N x N attention matrix is never
// materialized. Matches scripts/paged_attention_reference.py within
// atol=2e-3 / rtol=2e-2 on the synthetic fixture.
//
// Zero readback by construction: the K and V buffers are declared
// \`storage, read\` and are never mapped, copied, or read back by the host.

struct Params {
  past_len: u32,
  num_query_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
  max_seq_len: u32,
  block_size: u32,
  layer: u32,
  num_blocks: u32,
  scale: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;        // [num_query_heads, head_dim]
@group(0) @binding(2) var<storage, read> key_cache: array<f32>;    // [num_blocks, block_size, num_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read> value_cache: array<f32>;  // same layout as key_cache
@group(0) @binding(4) var<storage, read> block_table: array<u32>;  // [num_blocks]
@group(0) @binding(5) var<storage, read_write> output: array<f32>; // [num_query_heads, head_dim]
@group(0) @binding(6) var<storage, read_write> status: array<u32>; // [1]; 0 = ok, 1 = non-finite score

const MAX_HEAD_DIM: u32 = 128u;
const WORKGROUP_SIZE: u32 = 64u;
const NEG_INF: f32 = -3.0e38;
const F32_MAX: f32 = 3.0e38;

var<workgroup> q_shared: array<f32, MAX_HEAD_DIM>;
var<workgroup> reduce: array<f32, WORKGROUP_SIZE>;
var<workgroup> score_shared: array<f32, 1>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let head = wid.x;
  let head_dim = params.head_dim;
  let kv_head = head / (params.num_query_heads / params.num_kv_heads);

  for (var d = lid; d < head_dim; d += WORKGROUP_SIZE) {
    q_shared[d] = query[head * head_dim + d];
  }
  workgroupBarrier();

  var m: f32 = NEG_INF;
  var l: f32 = 0.0;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  let d1 = lid + WORKGROUP_SIZE;

  var p: u32 = 0u;
  loop {
    if (p >= params.past_len) {
      break;
    }
    let logical = p / params.block_size;
    let slot = p % params.block_size;
    let physical = block_table[logical];
    let base = ((physical * params.block_size + slot) * params.num_kv_heads + kv_head) * head_dim;

    var partial: f32 = 0.0;
    var d = lid;
    loop {
      if (d >= head_dim) {
        break;
      }
      partial = partial + q_shared[d] * key_cache[base + d];
      d = d + WORKGROUP_SIZE;
    }
    reduce[lid] = partial;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
      if (stride == 0u) {
        break;
      }
      if (lid < stride) {
        reduce[lid] = reduce[lid] + reduce[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }

    if (lid == 0u) {
      score_shared[0] = reduce[0] * params.scale;
    }
    workgroupBarrier();
    let s = score_shared[0];
    workgroupBarrier();

    // Reference raises on non-finite scores; the kernel clamps to a status flag.
    let is_nan = s != s;
    let is_inf = abs(s) > F32_MAX;
    let bad = is_nan || is_inf;
    if (bad && lid == 0u) {
      status[0] = 1u;
    }
    let s_safe = select(s, 0.0, bad);

    let new_m = max(m, s_safe);
    let alpha = exp(m - new_m); // underflows to 0 on the first iteration
    let probability = exp(s_safe - new_m);
    l = l * alpha + probability;

    if (lid < head_dim) {
      acc0 = acc0 * alpha + probability * value_cache[base + lid];
    }
    if (d1 < head_dim) {
      acc1 = acc1 * alpha + probability * value_cache[base + d1];
    }
    m = new_m;
    workgroupBarrier(); // everyone finished reading score_shared before next iteration
    p = p + 1u;
  }

  var out_index = head * head_dim + lid;
  if (lid < head_dim) {
    if (status[0] != 0u) {
      output[out_index] = 0.0;
    } else if (l == 0.0) {
      output[out_index] = 0.0;
    } else {
      output[out_index] = acc0 / l;
    }
  }
  out_index = head * head_dim + d1;
  if (d1 < head_dim) {
    if (status[0] != 0u) {
      output[out_index] = 0.0;
    } else if (l == 0.0) {
      output[out_index] = 0.0;
    } else {
      output[out_index] = acc1 / l;
    }
  }
}
`;
