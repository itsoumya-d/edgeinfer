#!/usr/bin/env python3
"""Generate the 20-prompt token fixture for the SmolLM2 ONNX baseline.

Requires `transformers` locally. The generated JSON is committed; this script is
not part of the default test/runtime path.

Usage:
    python scripts/make_smollm2_prompts.py --model models/smollm2-135m-instruct \
        --out scripts/fixtures/smollm2-135m-20-prompts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Name the largest planet in the solar system.",
    "What year did the Berlin Wall fall?",
    "Write a haiku about the ocean.",
    "What is 17 multiplied by 6?",
    "Define recursion in one sentence.",
    "Who wrote Pride and Prejudice?",
    "What is the boiling point of water at sea level?",
    "Give one reason to use version control.",
    "What does HTTP stand for?",
    "Name a primary color.",
    "What is the speed of light in a vacuum?",
    "Explain what a database index does.",
    "What is the difference between TCP and UDP?",
    "Name the smallest prime number.",
    "What is the capital of Japan?",
    "Write one sentence about why tests matter.",
    "What is 100 divided by 4?",
    "Name a common sorting algorithm.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for index, prompt in enumerate(PROMPTS, start=1):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        prompts.append(
            {
                "id": f"prompt-{index:02d}",
                "prompt": prompt,
                "text": text,
                "input_ids": ids,
            }
        )

    payload = {
        "schema_version": 1,
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "tokenizer": args.model,
        "chat_template": True,
        "max_new_tokens": 32,
        "prompts": prompts,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(prompts)} prompts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
