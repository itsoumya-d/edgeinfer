#!/usr/bin/env python3
"""Build the compact browser parity fixture for the WGSL paged-attention kernel.

Writes scripts/fixtures/paged-attention-20.json containing the deterministic
synthetic cases from paged_attention_parity.generate plus the float32 reference
outputs, so a browser harness can run the kernel offline and compare.

The fixture is synthetic and is not a SmolLM2 benchmark.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from paged_attention_reference import PagedKVCache, compare_outputs, paged_attention
from paged_attention_parity import _load_cache

BLOCK_SIZE = 16
SEED = 42
NUM_CASES = 20
LENGTHS = [1, 15, 16, 17, 31, 32, 33, 47, 48, 63]
NUM_KV_HEADS = 2
HEAD_DIM = 3
MAX_SEQ_LEN = 64


def _nested(blocks: int, block_size: int, heads: int, head_dim: int, rng: random.Random):
    return [
        [
            [
                [round(rng.uniform(-1.0, 1.0), 8) for _ in range(head_dim)]
                for _ in range(heads)
            ]
            for _ in range(block_size)
        ]
        for _ in range(blocks)
    ]


def build() -> dict:
    rng = random.Random(SEED)
    num_blocks = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)
    cases = []
    for index in range(NUM_CASES):
        past_len = LENGTHS[index % len(LENGTHS)]
        k = _nested(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, rng)
        v = _nested(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, rng)
        block_table = list(range(num_blocks))
        rng.shuffle(block_table)
        case = {
            "name": f"synthetic-case-{index + 1:02d}",
            "sequence_length": past_len,
            "layer": 0,
            "past_len": past_len,
            "cache": {
                "num_layers": 1,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "max_seq_len": MAX_SEQ_LEN,
                "block_size": BLOCK_SIZE,
                "block_table": block_table,
                "layers": [{"k": k, "v": v}],
            },
            "q": [
                [round(rng.uniform(-1.0, 1.0), 8) for _ in range(HEAD_DIM)]
                for _ in range(4 if index % 2 else 2)
            ],
        }
        cache = _load_cache(case["cache"])
        expected = paged_attention(case["q"], cache, 0, past_len, None)
        case["expected"] = expected
        verdict = compare_outputs(expected, expected)
        assert verdict["ok"]
        cases.append(case)
    return {
        "schema_version": 1,
        "kind": "paged_attention_browser_fixture",
        "seed": SEED,
        "note": "Synthetic tensor cases with float32 reference outputs. Not SmolLM2 logits.",
        "atol": 2e-3,
        "rtol": 2e-2,
        "cases": cases,
    }


def main() -> None:
    out = Path(__file__).parent / "fixtures" / "paged-attention-20.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
