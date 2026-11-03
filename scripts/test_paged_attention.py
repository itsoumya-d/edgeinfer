# SPDX-License-Identifier: MIT

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from paged_attention_reference import (
    PagedKVCache,
    compare_outputs,
    dense_attention,
    paged_attention,
)
import paged_attention_parity as parity


def nested_zeros(blocks: int, block_size: int, heads: int, head_dim: int):
    return [
        [[[0.0 for _ in range(head_dim)] for _ in range(heads)] for _ in range(block_size)]
        for _ in range(blocks)
    ]


def make_cache(past_len: int, num_heads: int = 2, head_dim: int = 3, seed: int = 42, block_table=None):
    random_state = random.Random(seed)
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max(32, past_len),
        block_size=16,
    )
    if block_table is not None:
        cache.set_block_table(block_table)
    expected_keys = []
    expected_values = []
    for position in range(past_len):
        key = [[random_state.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(num_heads)]
        value = [[random_state.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(num_heads)]
        cache.append(0, position, key, value)
        expected_keys.append(key)
        expected_values.append(value)
    return cache, expected_keys, expected_values


class PagedAttentionReferenceTests(unittest.TestCase):
    def test_matches_dense_softmax_for_block_boundaries(self):
        for past_len in [1, 15, 16, 17, 31, 32]:
            with self.subTest(past_len=past_len):
                cache, keys, values = make_cache(past_len, seed=past_len)
                q = [[0.2, -0.1, 0.7], [0.3, 0.4, -0.2]]
                scale = 0.5
                actual = paged_attention(q, cache, 0, past_len, scale=scale)
                expected = dense_attention(q, keys, values, past_len, scale=scale)
                self.assertEqual(len(actual), 2)
                self.assertEqual(len(actual[0]), 3)
                for actual_row, expected_row in zip(actual, expected):
                    for actual_value, expected_value in zip(actual_row, expected_row):
                        self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_block_table_controls_logical_order(self):
        cache, keys, values = make_cache(17, seed=17, block_table=[1, 0])
        q = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        actual = paged_attention(q, cache, 0, 17)
        expected = dense_attention(q, keys, values, 17)
        for actual_row, expected_row in zip(actual, expected):
            for actual_value, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_ignores_positions_beyond_past_len(self):
        cache, keys, values = make_cache(17, seed=17)
        q = [[0.2, -0.1, 0.7], [0.3, 0.4, -0.2]]
        sentinel_key = [[float("nan") for _ in range(3)] for _ in range(2)]
        sentinel_value = [[float("inf") for _ in range(3)] for _ in range(2)]
        # Positions 15 and 16 are outside the evaluated range.
        cache.k[0][0][15] = sentinel_key
        cache.k[0][1][0] = sentinel_key
        cache.v[0][0][15] = sentinel_value
        cache.v[0][1][0] = sentinel_value
        actual = paged_attention(q, cache, 0, 15)
        expected = dense_attention(q, keys, values, 15)
        for actual_row, expected_row in zip(actual, expected):
            for actual_value, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_empty_past_len_returns_zeros(self):
        cache, _, _ = make_cache(1, seed=1)
        q = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(
            paged_attention(q, cache, 0, 0),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

    def test_invalid_shapes_raise(self):
        cache, _, _ = make_cache(1, seed=1)
        with self.assertRaises(ValueError):
            cache.append(0, 0, [[1.0, 2.0]], [[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            paged_attention([[1.0]], cache, 0, 1)
        with self.assertRaises(ValueError):
            cache.set_block_table([0, 0])

    def test_compare_outputs_reports_shape_mismatch(self):
        result = compare_outputs([[1.0]], [[1.0], [2.0]])
        self.assertEqual(result["ok"], False)
        self.assertEqual(result["reason"], "shape_mismatch")

    def test_compare_outputs_tolerance(self):
        self.assertTrue(compare_outputs([[1.0, 2.0]], [[1.001, 2.01]])["ok"])
        self.assertFalse(compare_outputs([[1.0, 2.0]], [[1.1, 2.1]])["ok"])


class PagedAttentionParityTests(unittest.TestCase):
    def test_generate_and_check_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "case.json"
            candidate_path = root / "candidate.json"
            self.assertEqual(
                parity.main(["generate", "--out", str(case_path), "--cases", "20", "--seed", "42"]),
                0,
            )
            case = json.loads(case_path.read_text(encoding="utf-8"))
            candidate = {"cases": []}
            for item in case["cases"]:
                cache = parity._load_cache(item["cache"])
                candidate["cases"].append(
                    {
                        "name": item["name"],
                        "output": paged_attention(
                            item["q"],
                            cache,
                            item.get("layer", 0),
                            item["past_len"],
                            item.get("scale"),
                        ),
                    }
                )
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            self.assertEqual(
                parity.main(["check", "--case", str(case_path), "--candidate", str(candidate_path)]),
                0,
            )
            candidate["cases"][0]["output"][0][0] += 0.1
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            self.assertEqual(
                parity.main(["check", "--case", str(case_path), "--candidate", str(candidate_path)]),
                1,
            )


if __name__ == "__main__":
    unittest.main()
