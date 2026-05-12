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


def main():
    parser = argparse.ArgumentParser(description="World Model SLM 2026")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI server")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.train:
        print("=" * 60)
        print("  WORLD MODEL SLM 2026 - TRAINING PIPELINE")
        print("=" * 60)
        from scripts.train_tinyllama_slm import main as train_main
        import asyncio
        asyncio.run(train_main())
        return

    if args.server:
        print("=" * 60)
        print(f"  WORLD MODEL SLM 2026 - API SERVER ({args.host}:{args.port})")
        print("=" * 60)
        from serving.server import ModelServer
        import asyncio
        server = ModelServer(
            model_path="./models/tinyllama-trained-slm",
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
    print("\nLaunching with: streamlit run ui/app.py")
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])


if __name__ == "__main__":
    main()


