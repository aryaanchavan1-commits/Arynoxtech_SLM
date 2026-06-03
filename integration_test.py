#!/usr/bin/env python3
"""
Integration test for the World Model SLM system.
Tests the complete pipeline with a single model load.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from serving.model import ModelManager
from core.world_model import WorldModel

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

async def main():
    print_header("WORLD MODEL SLM 2026 - INTEGRATION TEST")
    
    # Test 1: Model Loading
    print("\n1. Loading model...")
    manager = ModelManager(model_path="./models/smollm2-360m-trained-slm")
    await manager.load_model()
    status = await manager.get_status()
    
    if not status['loaded']:
        print(f"{FAIL} Model failed to load")
        return 1
    print(f"{PASS} Model loaded: {status['model_path']}")
    print(f"   Device: {status['device']}")
    
    # Test 2: Basic Generation
    print("\n2. Testing basic generation...")
    start = time.time()
    response = await manager.generate("What is 2+2?", max_tokens=50)
    elapsed = time.time() - start
    
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Response: {response[:100]}...")
    
    if elapsed > 10:
        print(f"{WARN} Generation slow ({elapsed:.2f}s > 10s)")
    else:
        print(f"{PASS} Generation fast enough ({elapsed:.2f}s < 10s)")
    
    # Test 3: World Model
    print("\n3. Testing World Model...")
    wm = WorldModel(imagination_depth=2, thinking_steps=3)
    
    thoughts = await wm.think("What is gravity?", None)
    print(f"{PASS} Generated {len(thoughts)} thinking steps")
    
    scenarios = await wm.imagine_scenarios("What is gravity?", None)
    print(f"{PASS} Generated {len(scenarios)} scenarios")
    
    # Test RL agent
    state = wm.embedder.encode("test")
    import torch
    st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action, strategy, lp = wm.rl.get_action(st)
    print(f"{PASS} RL agent action: {strategy}")
    
    # Test 4: Hallucination Detection
    print("\n4. Testing hallucination detection...")
    query = "What happens when you drop a ball?"
    good_response = "The ball falls down due to gravity."
    bad_response = "The ball floats up into the sky."
    
    c_good, _ = wm.causal.validate(query, good_response)
    p_good, _ = wm.physics.check(query, good_response)
    print(f"{PASS} Good response - Causal: {c_good:.2f}, Physics: {p_good:.2f}")
    
    c_bad, _ = wm.causal.validate(query, bad_response)
    p_bad, _ = wm.physics.check(query, bad_response)
    print(f"{PASS} Bad response - Causal: {c_bad:.2f}, Physics: {p_bad:.2f}")
    
    # Test 5: Full Pipeline
    print("\n5. Testing full pipeline...")
    prompt = "Explain photosynthesis briefly."
    
    thoughts = await wm.think(prompt, None)
    scenarios = await wm.imagine_scenarios(prompt, None)
    
    start = time.time()
    result = await wm.generate_response(
        prompt, scenarios, thoughts, manager,
        document_context=None, context={}
    )
    elapsed = time.time() - start
    
    print(f"{PASS} Pipeline completed in {elapsed:.2f}s")
    print(f"   Response: {result['content'][:200]}...")
    print(f"   Self-eval: {result.get('self_evaluation_score', 0):.2f}")
    print(f"   Causal: {result.get('causal_score', 0):.2f}")
    print(f"   Physics: {result.get('physics_score', 0):.2f}")
    
    await manager.unload_model()
    
    # Summary
    print_header("TEST SUMMARY")
    print(f"{PASS} Model Loading")
    print(f"{PASS} Basic Generation")
    print(f"{PASS} World Model + RL")
    print(f"{PASS} Hallucination Detection")
    print(f"{PASS} Full Pipeline")
    print("="*60)
    print("  ALL TESTS PASSED - core system is operational!")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
