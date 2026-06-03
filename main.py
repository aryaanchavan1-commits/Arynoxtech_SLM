#!/usr/bin/env python3
"""
World Model SLM 2026 - Main Entry Point
Supports: UI (--ui), Server (--server), Training (--train)
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
from pathlib import Path

# Add parent directory and scripts subdirectory to path for imports
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

DEFAULT_MODEL_PATH = "./models/anonyllm-360m-trained"
TINY_MOBILE_PATH = "./models/tiny-mobile-slm"
OPTIMIZED_INT4_PATH = "./models/optimized/smollm2-360m/int4"
FALLBACK_MODEL_PATH = "./models/smollm2-360m-trained-slm"

os.environ.setdefault("MODEL_NAME", "AnonyLLM-360M-v2")
os.environ.setdefault("CREATOR", "Aryan Chavan")

def _auto_pick_model():
    """Auto-pick the best model available locally."""
    mode = os.environ.get("SLM_MODE", "auto").strip().lower()
    if mode == "mobile" or mode == "tiny":
        return TINY_MOBILE_PATH
    for p in [DEFAULT_MODEL_PATH, FALLBACK_MODEL_PATH, OPTIMIZED_INT4_PATH, TINY_MOBILE_PATH]:
        if os.path.isdir(p) and any(f.endswith((".safetensors", ".bin", ".pt")) for _, _, files in os.walk(p) for f in files):
            return p
    return FALLBACK_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="World Model SLM 2026")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI server")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--train-smoke", action="store_true", help="Run a tiny local training smoke test")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.train:
        print("=" * 60)
        print("  WORLD MODEL SLM 2026 - TRAINING PIPELINE")
        print("=" * 60)
        from scripts.train_mistral_slm import main as train_main
        train_main()
        return

    if args.train_smoke:
        print("=" * 60)
        print("  WORLD MODEL SLM 2026 - LOCAL TRAINING SMOKE TEST")
        print("=" * 60)
        from scripts.train_mistral_slm import smoke_train
        smoke_train()
        return

    if args.server:
        print("=" * 60)
        print(f"  WORLD MODEL SLM 2026 - API SERVER ({args.host}:{args.port})")
        print("=" * 60)
        from serving.server import ModelServer
        import asyncio
        model_path = _auto_pick_model()
        print(f"  Model: {model_path}")
        server = ModelServer(
            model_path=model_path,
            host=args.host,
            port=args.port,
        )
        server.setup()
        asyncio.run(server.start())
        return

    # Default: UI
    print("=" * 60)
    print("  WORLD MODEL SLM 2026 - STREAMLIT UI")
    print("=" * 60)
    model_path = _auto_pick_model()
    print(f"  Model: {model_path}")
    os.environ.setdefault("MODEL_PATH", model_path)
    print("\nLaunching with: streamlit run ui/app.py")
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])


if __name__ == "__main__":
    main()


