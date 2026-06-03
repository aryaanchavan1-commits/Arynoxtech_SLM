#!/usr/bin/env python3
"""Verify that the trained Mistral merged model powers the end-to-end WorldModel+Agentic pipeline.

Checks:
1) merged model directory exists
2) ModelManager can load it (not mock)
3) run a small WorldModel query that triggers tools + self-eval

Usage:
  python scripts/verify_mistral_training_integration.py

Optional overrides:
  PRODUCTION_MODEL_PATH env var (default: ./models/ministral-3-3b-trained-slm)
"""

import os
import asyncio
import sys
from pathlib import Path

# Ensure project root is importable when running `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.world_model import WorldModel
from serving.model import ModelManager



PRODUCTION_MODEL_PATH = os.environ.get(
    "PRODUCTION_MODEL_PATH", "./models/ministral-3-3b-trained-slm"
)


async def main() -> None:
    wm = WorldModel(auto_adjust_depth=True, auto_adjust_steps=True)

    # Keep generation tiny so this verifier finishes quickly on CPU.
    mm = ModelManager(
        model_path=PRODUCTION_MODEL_PATH,
        max_batch_size=1,
        max_sequence_length=256,
    )


    print("=" * 70)
    print("VERIFY: Mistral merged model integration")
    print(f"Merged model path: {PRODUCTION_MODEL_PATH}")
    print("=" * 70)

    if not os.path.isdir(PRODUCTION_MODEL_PATH):
        raise SystemExit(
            f"Merged model directory not found: {PRODUCTION_MODEL_PATH}. "
            f"Run `python scripts/train_mistral_slm.py` first (and ensure it saves merged model there)."
        )

    mm.load_model_sync()
    status = await mm.get_status()
    print("Model status:", status)
    if status.get("is_mock"):
        raise SystemExit(
            "ModelManager loaded mock instead of real weights. "
            "Check the merged model folder contents (config.json, model.safetensors)."
        )

    # Tool-triggering query (calculator)
    prompt = "Calculate 12 * 7 step-by-step, then give the final result."
    context = {"user_name": "Aryan"}

    thoughts = await wm.think(prompt, context)
    scenarios = await wm.imagine_scenarios(prompt, context)

    result = await wm.generate_response(
        prompt,
        scenarios=scenarios,
        thoughts=thoughts,
        model_manager=mm,
        document_context=None,
        context=context,
    )

    # Self-evaluation helper (world_model already evaluates during generate_response,
    # but we also run the explicit function for consistency)
    re_eval = await wm.self_evaluate_and_improve(prompt, result["content"], mm)

    print("=" * 70)
    print("Prompt:", prompt)
    print("\n--- Response (truncated) ---")
    print(result["content"][:1200])

    print("\n--- Scores ---")
    print("self_eval (during response):", result.get("self_evaluation_score"))
    print("causal_score:", result.get("causal_score"))
    print("physics_score:", result.get("physics_score"))
    print("world_model self_evaluate_and_improve overall:", re_eval.get("overall"))

    print("\n--- Tool Results (if any) ---")
    tool_results = result.get("tool_results") or []
    if tool_results:
        for tr in tool_results:
            print("-", tr[:400])
    else:
        print("(none)")

    print("=" * 70)
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

