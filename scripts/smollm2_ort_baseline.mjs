// SPDX-License-Identifier: MIT
// Generate a deterministic ORT baseline for the SmolLM2-135M-Instruct ONNX
// export. This is a baseline fixture generator; it does not implement paged
// attention and does not claim token parity with a WGSL kernel.
//
// Usage:
//   node scripts/smollm2_ort_baseline.mjs \
//     --model models/smollm2-135m-instruct/onnx/model_fp16.onnx \
//     --prompts scripts/fixtures/smollm2-135m-20-prompts.json \
//     --out scripts/fixtures/smollm2-135m-20-baseline.json \
//     --limit 1 --max-new-tokens 4

import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import * as ort from "onnxruntime-web";

function parseArgs(argv) {
  const args = {
    model: "models/smollm2-135m-instruct/onnx/model_fp16.onnx",
    prompts: "scripts/fixtures/smollm2-135m-20-prompts.json",
    out: "scripts/fixtures/smollm2-135m-20-baseline.json",
    limit: 0,
    maxNewTokens: 0
  };
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (flag === "--model") args.model = value;
    else if (flag === "--prompts") args.prompts = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--limit") args.limit = Number(value);
    else if (flag === "--max-new-tokens") args.maxNewTokens = Number(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
}

function int64Tensor(values, dims) {
  return new ort.Tensor(
    "int64",
    BigInt64Array.from(values, (value) => BigInt(value)),
    dims
  );
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}

function topK(values, k) {
  const ranked = [];
  for (let index = 0; index < values.length; index += 1) {
    if (ranked.length < k || values[index] > values[ranked[ranked.length - 1].token]) {
      ranked.push({ token: index, logit: values[index] });
      ranked.sort((left, right) => right.logit - left.logit);
      if (ranked.length > k) ranked.pop();
    }
  }
  return ranked;
}

const args = parseArgs(process.argv.slice(2));
ort.env.wasm.wasmPaths = pathToFileURL(
  `${join(process.cwd(), "node_modules", "onnxruntime-web", "dist")}/`
).href;
const session = await ort.InferenceSession.create(args.model, {
  executionProviders: ["cpu"]
});

const fixture = JSON.parse(await readFile(args.prompts, "utf8"));
const prompts = args.limit > 0 ? fixture.prompts.slice(0, args.limit) : fixture.prompts;
const maxNewTokens = args.maxNewTokens || fixture.max_new_tokens;
const layerCount = session.inputNames.filter((name) => /^past_key_values\.\d+\.key$/.test(name)).length;
const eosTokenId = 2;
const started = Date.now();
const results = [];

for (const entry of prompts) {
  let inputIds = entry.input_ids.slice();
  let past = null;
  let totalLength = inputIds.length;
  const generated = [];
  const steps = [];

  for (let step = 0; step < maxNewTokens; step += 1) {
    const feeds = {
      input_ids: int64Tensor(inputIds, [1, inputIds.length]),
      attention_mask: int64Tensor(new Array(totalLength).fill(1), [1, totalLength]),
      position_ids: int64Tensor(
        inputIds.map((_, index) => totalLength - inputIds.length + index),
        [1, inputIds.length]
      )
    };
    for (let layer = 0; layer < layerCount; layer += 1) {
      if (past) {
        feeds[`past_key_values.${layer}.key`] = past[`present.${layer}.key`];
        feeds[`past_key_values.${layer}.value`] = past[`present.${layer}.value`];
      } else {
        feeds[`past_key_values.${layer}.key`] = new ort.Tensor(
          "float16",
          new Uint16Array(0),
          [1, 3, 0, 64]
        );
        feeds[`past_key_values.${layer}.value`] = new ort.Tensor(
          "float16",
          new Uint16Array(0),
          [1, 3, 0, 64]
        );
      }
    }

    const output = await session.run(feeds);
    const logits = output.logits;
    const vocab = logits.dims[logits.dims.length - 1];
    const sequence = logits.dims[logits.dims.length - 2];
    const offset = (sequence - 1) * vocab;
    const last = logits.data.subarray(offset, offset + vocab);
    const token = argmax(last);
    generated.push(token);
    steps.push({
      token,
      logit: last[token],
      top5: topK(last, 5)
    });
    past = output;
    totalLength += 1;
    inputIds = [token];
    if (token === eosTokenId) break;
  }

  results.push({
    id: entry.id,
    prompt: entry.prompt,
    input_ids: entry.input_ids,
    generated_tokens: generated,
    steps
  });
  console.log(`${entry.id}: ${generated.length} tokens`);
}

const payload = {
  schema_version: 1,
  model: "HuggingFaceTB/SmolLM2-135M-Instruct",
  model_file: args.model,
  execution_provider: "cpu",
  max_new_tokens: maxNewTokens,
  eos_token_id: eosTokenId,
  generated_in_ms: Date.now() - started,
  prompts: results
};
await writeFile(args.out, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
console.log(`wrote ${results.length} prompts to ${args.out}`);
