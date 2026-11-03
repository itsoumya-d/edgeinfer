#!/usr/bin/env python3
"""CPU reference for paged attention.

This is the Python-side reference described by `docs/wgsl-paged-attention.md`.
It implements the same online-softmax recurrence in float32 and exposes a small
JSON CLI so a future WGSL candidate can be compared against it.

Cache layout, matching the design note:

    KV[block_id][slot][kv_head][head_dim]

- `block_size` is 16 by default.
- `block_table[logical_block]` maps to a physical block id.
- Positions at or beyond `past_len` in the final block are ignored.
- The first version is multi-head attention only (`num_heads == num_kv_heads`).

No WGSL is shipped here. The comparison harness lives in
`paged_attention_parity.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Sequence

DEFAULT_BLOCK_SIZE = 16


def _shape(value, name: str) -> list[int]:
    shape = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def _check_matrix(value, rows: int, cols: int, name: str) -> None:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"{name} row {index} must have {cols} columns")


def _check_rank4(value, blocks: int, block_size: int, heads: int, head_dim: int, name: str) -> None:
    if not isinstance(value, list) or len(value) != blocks:
        raise ValueError(f"{name} must have {blocks} physical blocks")
    for physical, block in enumerate(value):
        if not isinstance(block, list) or len(block) != block_size:
            raise ValueError(f"{name}[{physical}] must have {block_size} slots")
        for slot, slot_value in enumerate(block):
            if not isinstance(slot_value, list) or len(slot_value) != heads:
                raise ValueError(f"{name}[{physical}][{slot}] must have {heads} heads")
            for head, head_value in enumerate(slot_value):
                if not isinstance(head_value, list) or len(head_value) != head_dim:
                    raise ValueError(
                        f"{name}[{physical}][{slot}][{head}] must have {head_dim} elements"
                    )


@dataclass
class PagedKVCache:
    """Physical KV cache plus a logical-to-physical block table."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    block_size: int = DEFAULT_BLOCK_SIZE

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.num_kv_heads < 1:
            raise ValueError("num_kv_heads must be at least 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be at least 1")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        if self.block_size != DEFAULT_BLOCK_SIZE:
            raise ValueError(f"block_size must be {DEFAULT_BLOCK_SIZE}")

        self.num_blocks = math.ceil(self.max_seq_len / self.block_size)
        self.k = [
            self._new_tensor() for _ in range(self.num_layers)
        ]
        self.v = [
            self._new_tensor() for _ in range(self.num_layers)
        ]
        self.block_table = list(range(self.num_blocks))

    def _new_tensor(self) -> list[list[list[list[float]]]]:
        return [
            [
                [[0.0 for _ in range(self.head_dim)] for _ in range(self.num_kv_heads)]
                for _ in range(self.block_size)
            ]
            for _ in range(self.num_blocks)
        ]

    def set_block_table(self, block_table: Sequence[int]) -> None:
        if list(block_table) == [] and self.num_blocks == 0:
            return
        if len(block_table) != self.num_blocks:
            raise ValueError(f"block_table must contain {self.num_blocks} entries")
        table = [int(item) for item in block_table]
        if sorted(table) != list(range(self.num_blocks)):
            raise ValueError("block_table must be a permutation of physical block ids")
        self.block_table = table

    def append(self, layer: int, position: int, key: Sequence[float], value: Sequence[float]) -> None:
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer must be in [0, {self.num_layers})")
        if not 0 <= position < self.max_seq_len:
            raise ValueError(f"position must be in [0, {self.max_seq_len})")
        _check_matrix(key, self.num_kv_heads, self.head_dim, "key")
        _check_matrix(value, self.num_kv_heads, self.head_dim, "value")

        logical = position // self.block_size
        slot = position % self.block_size
        physical = self.block_table[logical]
        self.k[layer][physical][slot] = [
            [float(item) for item in head] for head in key
        ]
        self.v[layer][physical][slot] = [
            [float(item) for item in head] for head in value
        ]


def _validate_cache(cache: PagedKVCache) -> None:
    if not isinstance(cache, PagedKVCache):
        raise TypeError("cache must be a PagedKVCache")
    if len(cache.k) != cache.num_layers or len(cache.v) != cache.num_layers:
        raise ValueError("cache tensor count does not match num_layers")
    if len(cache.block_table) != cache.num_blocks:
        raise ValueError("cache block_table length does not match max_seq_len")
    if sorted(cache.block_table) != list(range(cache.num_blocks)):
        raise ValueError("cache block_table must be a permutation")
    for layer in range(cache.num_layers):
        _check_rank4(cache.k[layer], cache.num_blocks, cache.block_size, cache.num_kv_heads, cache.head_dim, f"cache.k[{layer}]")
        _check_rank4(cache.v[layer], cache.num_blocks, cache.block_size, cache.num_kv_heads, cache.head_dim, f"cache.v[{layer}]")


def paged_attention(
    q: Sequence[Sequence[float]],
    cache: PagedKVCache,
    layer: int,
    past_len: int,
    scale: float | None = None,
) -> list[list[float]]:
    """Compute paged attention for one query position."""
    _validate_cache(cache)
    if not 0 <= layer < cache.num_layers:
        raise ValueError(f"layer must be in [0, {cache.num_layers})")
    if not 0 <= past_len <= cache.max_seq_len:
        raise ValueError(f"past_len must be in [0, {cache.max_seq_len}]")
    _check_matrix(q, cache.num_kv_heads, cache.head_dim, "q")

    if scale is None:
        scale = 1.0 / math.sqrt(cache.head_dim)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")

    heads = cache.num_kv_heads
    if past_len == 0:
        return [[0.0 for _ in range(cache.head_dim)] for _ in range(heads)]

    output: list[list[float]] = []
    for head in range(heads):
        m = -math.inf
        denominator = 0.0
        accumulator = [0.0 for _ in range(cache.head_dim)]

        for logical in range(math.ceil(past_len / cache.block_size)):
            physical = cache.block_table[logical]
            for slot in range(cache.block_size):
                position = logical * cache.block_size + slot
                if position >= past_len:
                    break
                key = cache.k[layer][physical][slot][head]
                value = cache.v[layer][physical][slot][head]
                score = sum(q[head][dim] * key[dim] for dim in range(cache.head_dim)) * scale
                if not math.isfinite(score):
                    raise ValueError(f"non-finite attention score at position {position}")
                new_m = max(m, score)
                alpha = math.exp(m - new_m) if m != -math.inf else 0.0
                probability = math.exp(score - new_m)
                denominator = denominator * alpha + probability
                accumulator = [
                    acc * alpha + probability * value[dim]
                    for dim, acc in enumerate(accumulator)
                ]
                m = new_m

        if denominator == 0.0:
            output.append([0.0 for _ in range(cache.head_dim)])
        else:
            output.append([item / denominator for item in accumulator])
    return output


def dense_attention(
    q: Sequence[Sequence[float]],
    keys: Sequence[Sequence[Sequence[float]]],
    values: Sequence[Sequence[Sequence[float]]],
    past_len: int,
    scale: float | None = None,
) -> list[list[float]]:
    """Reference dense softmax, used by tests and the parity harness."""
    if not isinstance(keys, list) or not isinstance(values, list) or len(keys) != len(values):
        raise ValueError("keys and values must be equally sized lists")
    _check_matrix(q, len(q), len(q[0]) if q else 0, "q")
    heads = len(q)
    head_dim = len(q[0])
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if not 0 <= past_len <= len(keys):
        raise ValueError(f"past_len must be in [0, {len(keys)}]")

    output = []
    for head in range(heads):
        scores = []
        for position in range(past_len):
            if len(keys[position]) <= head or len(values[position]) <= head:
                raise ValueError(f"position {position} is missing head {head}")
            key = keys[position][head]
            if len(key) != head_dim or len(values[position][head]) != head_dim:
                raise ValueError(f"position {position} head {head} has the wrong dimension")
            scores.append(sum(q[head][dim] * key[dim] for dim in range(head_dim)) * scale)
        if not scores:
            output.append([0.0 for _ in range(head_dim)])
            continue
        max_score = max(scores)
        probabilities = [math.exp(score - max_score) for score in scores]
        total = sum(probabilities)
        output.append(
            [
                sum(probability * values[position][head][dim] for position, probability in enumerate(probabilities))
                / total
                for dim in range(head_dim)
            ]
        )
    return output


def compare_outputs(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    atol: float = 2e-3,
    rtol: float = 2e-2,
) -> dict[str, object]:
    """Compare two nested matrices; intended for WGSL candidate outputs."""
    if not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError("atol and rtol must be finite")
    _check_matrix(reference, len(reference), len(reference[0]) if reference else 0, "reference")
    _check_matrix(candidate, len(candidate), len(candidate[0]) if candidate else 0, "candidate")
    if len(reference) != len(candidate) or len(reference[0]) != len(candidate[0]):
        return {
            "ok": False,
            "reason": "shape_mismatch",
            "reference_shape": _shape(reference, "reference"),
            "candidate_shape": _shape(candidate, "candidate"),
        }

    max_abs = 0.0
    max_rel = 0.0
    failures = 0
    for ref_row, cand_row in zip(reference, candidate):
        for ref, cand in zip(ref_row, cand_row):
            if not math.isfinite(cand):
                failures += 1
                continue
            absolute = abs(ref - cand)
            relative = absolute / max(abs(ref), 1e-12)
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            if absolute > atol and relative > rtol:
                failures += 1
    return {
        "ok": failures == 0,
        "atol": atol,
        "rtol": rtol,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "failures": failures,
        "shape": _shape(reference, "reference"),
    }
