#!/usr/bin/env python3
"""
Demonstrate the system answering user queries.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.world_model import WorldModel
from serving.model import ModelManager

async def demo():
    print("="*70)
    print("  WORLD MODEL SLM 2026 - USER QUERY DEMONSTRATION")
    print("="*70)
    
    # Initialize
    print("\n[1/4] Loading model...")
    wm = WorldModel(imagination_depth=2, thinking_steps=3)
    model_path = "./models/anonyllm-360m-trained"
    if not os.path.exists(os.path.join(model_path, "config.json")):
        model_path = "./models/smollm2-360m-trained-slm"
    manager = ModelManager(model_path=model_path)
    await manager.load_model()
    print("      ✅ Model loaded")
    
    # User queries to test
    queries = [
        "What is an LLM?",
        "What is Llamaster?",
        "How do LLMs learn to explain things?",
        "How can LLMs improve written communication?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"  Query {i}: {query}")
        print(f"{'='*70}")
        
        # Process query
        thoughts = await wm.think(query, None)
        scenarios = await wm.imagine_scenarios(query, None)
        
        print(f"\n🧠 Thinking ({len(thoughts)} steps):")
        for t in thoughts[:2]:  # Show first 2
            print(f"   • {t.thought}")
        
        print(f"\n🌍 Scenarios ({len(scenarios)}):")
        for s in scenarios[:2]:  # Show first 2
            print(f"   • {s.description}: {s.outcome.value} (p={s.probability:.2f})")
        
        # Generate response
        print(f"\n💬 Generating response...")
        result = await wm.generate_response(
            query, scenarios, thoughts, manager,
            document_context=None, context={}
        )
        
        print(f"\n📝 Response:")
        print(f"{result['content']}")
        
        print(f"\n📊 Quality Metrics:")
        print(f"   Self-eval: {result.get('self_evaluation_score', 0):.2f}")
        print(f"   Causal: {result.get('causal_score', 0):.2f}")
        print(f"   Physics: {result.get('physics_score', 0):.2f}")
        print(f"   Strategy: {result.get('strategy', 'N/A')}")
        
        if result.get('tool_results'):
            print(f"\n🔧 Tools Used:")
            for tr in result['tool_results']:
                print(f"   {tr}")
    
    await manager.unload_model()
    
    print("\n" + "="*70)
    print("  DEMONSTRATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(demo())
