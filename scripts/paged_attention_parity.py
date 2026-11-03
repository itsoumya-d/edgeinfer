#!/usr/bin/env python3
"""Parity harness for paged attention.

This harness compares a candidate output (for example, a future WGSL kernel)
against the float32 CPU reference. It can also generate deterministic synthetic
cases for testing the harness itself.

It does **not** claim SmolLM2 parity, WebGPU correctness, or a speedup. Model
weights and a WGSL candidate are required before the P06 gate can be evaluated.

Usage:
    python scripts/paged_attention_parity.py generate --out scripts/fixtures/paged-attention-20.json
    python scripts/paged_attention_parity.py check --case case.json --candidate candidate.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from paged_attention_reference import (
    PagedKVCache,
    _validate_cache,
    compare_outputs,
    paged_attention,
)

DEFAULT_BLOCK_SIZE = 16


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def generate(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    cases = []
    lengths = [1, 15, 16, 17, 31, 32, 33, 47, 48, 63]
    for index in range(args.cases):
        past_len = lengths[index % len(lengths)]
        num_heads = 2
        head_dim = 3
        max_seq_len = 64
        num_blocks = math.ceil(max_seq_len / DEFAULT_BLOCK_SIZE)
        k = _nested(num_blocks, DEFAULT_BLOCK_SIZE, num_heads, head_dim, rng)
        v = _nested(num_blocks, DEFAULT_BLOCK_SIZE, num_heads, head_dim, rng)
        block_table = list(range(num_blocks))
        rng.shuffle(block_table)
        case = {
            "name": f"synthetic-case-{index + 1:02d}",
            "kind": "synthetic_tensor",
            "sequence_length": past_len,
            "cache": {
                "num_layers": 1,
                "num_kv_heads": num_heads,
                "head_dim": head_dim,
                "max_seq_len": max_seq_len,
                "block_size": DEFAULT_BLOCK_SIZE,
                "block_table": block_table,
                "layers": [{"k": k, "v": v}],
            },
            "layer": 0,
            "past_len": past_len,
            "q": [
                [round(rng.uniform(-1.0, 1.0), 8) for _ in range(head_dim)]
                for _ in range(num_heads)
            ],
        }
        if args.include_scale:
            case["scale"] = round(0.5 + 0.1 * (index % 3), 8)
        cases.append(case)
    _save_json(args.out, {
        "schema_version": 1,
        "kind": "synthetic_paged_attention_cases",
        "seed": args.seed,
        "note": "Synthetic tensor cases for harness testing; not SmolLM2 logits.",
        "cases": cases,
    })
    print(f"wrote {len(cases)} cases to {args.out}")
    return 0


def _load_cache(data: dict) -> PagedKVCache:
    required = {"num_layers", "num_kv_heads", "head_dim", "max_seq_len"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError("cache must contain num_layers, num_kv_heads, head_dim, and max_seq_len")
    block_size = data.get("block_size", DEFAULT_BLOCK_SIZE)
    cache = PagedKVCache(
        num_layers=int(data["num_layers"]),
        num_kv_heads=int(data["num_kv_heads"]),
        head_dim=int(data["head_dim"]),
        max_seq_len=int(data["max_seq_len"]),
        block_size=int(block_size),
    )
    block_table = data.get("block_table")
    if block_table is not None:
        cache.set_block_table(block_table)

    layers = data.get("layers")
    if not isinstance(layers, list) or len(layers) != cache.num_layers:
        raise ValueError(f"cache.layers must contain {cache.num_layers} entries")
    for layer, layer_data in enumerate(layers):
        if not isinstance(layer_data, dict):
            raise ValueError(f"cache.layers[{layer}] must be an object")
        if "k" not in layer_data or "v" not in layer_data:
            raise ValueError(f"cache.layers[{layer}] must contain k and v")
        cache.k[layer] = layer_data["k"]
        cache.v[layer] = layer_data["v"]
    _validate_cache(cache)
    return cache


def check(args: argparse.Namespace) -> int:
    case_file = Path(args.case)
    candidate_file = Path(args.candidate)
    case = json.loads(case_file.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))

    cases = case.get("cases")
    candidate_cases = candidate.get("cases")
    if not isinstance(cases, list) or not isinstance(candidate_cases, list):
        raise ValueError("both files must contain a cases array")

    candidates = {item.get("name"): item for item in candidate_cases if isinstance(item, dict)}
    failures = []
    results = []
    for item in cases:
        name = item.get("name")
        candidate_item = candidates.get(name)
        if not candidate_item or "output" not in candidate_item:
            failures.append({"case": name, "reason": "missing_candidate"})
            continue
        cache = _load_cache(item["cache"])
        output = paged_attention(
            item["q"],
            cache,
            int(item.get("layer", 0)),
            int(item["past_len"]),
            item.get("scale"),
        )
        verdict = compare_outputs(
            output,
            candidate_item["output"],
            atol=args.atol,
            rtol=args.rtol,
        )
        results.append({"case": name, **verdict})
        if not verdict["ok"]:
            failures.append({"case": name, **verdict})

    summary = {
        "ok": not failures,
        "cases": len(cases),
        "compared": len(results),
        "failures": failures,
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic synthetic cases")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--cases", type=int, default=20)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--include-scale", action="store_true")
    gen.set_defaults(func=generate)

    chk = sub.add_parser("check", help="compare a candidate output file against the reference")
    chk.add_argument("--case", required=True)
    chk.add_argument("--candidate", required=True)
    chk.add_argument("--atol", type=float, default=2e-3)
    chk.add_argument("--rtol", type=float, default=2e-2)
    chk.set_defaults(func=check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
