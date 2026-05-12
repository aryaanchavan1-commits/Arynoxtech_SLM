#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from serving.model import ModelManager
import asyncio

async def quick_test():
    print("Testing model loading...")
    # Force use of fallback model
    manager = ModelManager(model_path="./models/tinyllama-trained-slm")
    await manager.load_model()
    status = await manager.get_status()
    print(f"Model loaded: {status['loaded']}")
    print(f"Path: {status['model_path']}")
    print(f"Device: {status['device']}")
    print(f"Is mock: {status.get('is_mock', 'N/A')}")
    
    print("\nTesting generation...")
    response = await manager.generate("What is 2+2?", max_tokens=50)
    print(f"Response: {response}")
    
    await manager.unload_model()
    print("\nDone!")

asyncio.run(quick_test())
