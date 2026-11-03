# SPDX-License-Identifier: MIT

import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "fixtures" / "smollm2-135m-20-prompts.json"
BASELINE = ROOT / "fixtures" / "smollm2-135m-20-baseline.json"
VOCAB_SIZE = 49152


class SmolLM2BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
        cls.baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    def test_prompt_fixture_has_20_deterministic_prompts(self):
        prompts = self.prompts["prompts"]
        self.assertEqual(len(prompts), 20)
        self.assertEqual(len({item["id"] for item in prompts}), 20)
        for item in prompts:
            self.assertTrue(item["input_ids"])
            self.assertTrue(all(isinstance(token, int) and 0 <= token < VOCAB_SIZE for token in item["input_ids"]))

    def test_baseline_covers_the_same_prompts(self):
        prompt_ids = [item["id"] for item in self.prompts["prompts"]]
        baseline_ids = [item["id"] for item in self.baseline["prompts"]]
        self.assertEqual(baseline_ids, prompt_ids)
        self.assertEqual(self.baseline["execution_provider"], "cpu")
        self.assertEqual(self.baseline["model"], "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertEqual(self.baseline["max_new_tokens"], 32)

    def test_generated_tokens_and_steps_are_consistent(self):
        for item in self.baseline["prompts"]:
            tokens = item["generated_tokens"]
            steps = item["steps"]
            self.assertTrue(tokens)
            self.assertLessEqual(len(tokens), 32)
            self.assertEqual(len(steps), len(tokens))
            for token, step in zip(tokens, steps):
                self.assertEqual(step["token"], token)
                self.assertTrue(0 <= token < VOCAB_SIZE)
                self.assertTrue(math.isfinite(step["logit"]))

    def test_top5_is_sorted_and_contains_the_choice(self):
        for item in self.baseline["prompts"]:
            for step in item["steps"]:
                top5 = step["top5"]
                self.assertEqual(len(top5), 5)
                self.assertEqual(top5[0]["token"], step["token"])
                logits = [entry["logit"] for entry in top5]
                self.assertEqual(logits, sorted(logits, reverse=True))
                self.assertTrue(all(math.isfinite(value) for value in logits))
                self.assertTrue(all(0 <= entry["token"] < VOCAB_SIZE for entry in top5))


if __name__ == "__main__":
    unittest.main()
