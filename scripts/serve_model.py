#!/usr/bin/env python3
"""
Script to serve the custom LLM model using FastAPI.
"""

import argparse
import asyncio
import sys
import io
from pathlib import Path
from typing import Optional

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Fix OpenBLAS memory allocation error on Windows - MUST be before numpy/torch imports
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from serving.server import ModelServer
from utils.logger import setup_logging

async def main():
    setup_logging(level="INFO")

    parser = argparse.ArgumentParser(description="Serve custom LLM model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host address for server")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for server")
    parser.add_argument("--model_name", type=str, default="local-llm",
                       help="Model name for display")
    parser.add_argument("--max_batch_size", type=int, default=8,
                       help="Maximum batch size for inference")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                       help="Maximum sequence length for generation")

    args = parser.parse_args()

    # Initialize model server
    server = ModelServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        max_batch_size=args.max_batch_size,
        max_sequence_length=args.max_sequence_length
    )

    # Setup server
    await server.setup()

    # Start server
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())