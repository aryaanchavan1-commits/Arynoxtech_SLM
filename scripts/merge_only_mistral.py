#!/usr/bin/env python3
"""Merge-only utility for the Mistral LoRA adapter.

It avoids importing `scripts.train_mistral_slm` in a way that may conflict with
threading/env issues, and instead directly calls `merge_and_save()`.

Usage:
  py -3.13 scripts/merge_only_mistral.py

Env overrides:
  MERGE_LORA_DIR (default: ./models/ministral-3-3b-trained-slm-lora)
  MERGED_OUT_DIR (default: ./models/ministral-3-3b-trained-slm)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_mistral_slm import (
    TrainingConfig,
    load_base_model_and_tokenizer,
    merge_and_save,
)


def main() -> None:
    lora_dir = os.environ.get(
        "MERGE_LORA_DIR", "./models/ministral-3-3b-trained-slm-lora"
    )
    out_dir = os.environ.get(
        "MERGED_OUT_DIR", "./models/ministral-3-3b-trained-slm"
    )

    config = TrainingConfig()
    config.lora_output_dir = lora_dir
    config.output_dir = out_dir

    print("=" * 70)
    print("MERGE ONLY (direct utility)")
    print("LoRA adapter:", config.lora_output_dir)
    print("Merged output:", config.output_dir)
    print("Base model:", config.base_model)
    print("Device:", config.device)
    print("=" * 70)

    model, tokenizer = load_base_model_and_tokenizer(config)
    merge_and_save(model, tokenizer, config)

    print("=" * 70)
    print("MERGE COMPLETE")
    print("Merged exists:", os.path.isdir(config.output_dir))
    print("=" * 70)


if __name__ == "__main__":
    main()

