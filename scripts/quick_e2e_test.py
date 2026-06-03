"""Quick end-to-end system test."""
import os, sys, asyncio
os.environ['HF_HOME'] = 'D:/.hf_cache'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from serving.model import ModelManager
from core.world_model import WorldModel


async def test():
    print('=' * 60)
    print('END-TO-END SYSTEM TEST')
    print('=' * 60)

    # 1. Test model loading
    print('\n[1] Loading trained model...')
    model_path = './models/anonyllm-360m-trained'
    if not os.path.exists(os.path.join(model_path, 'config.json')):
        model_path = './models/smollm2-360m-trained-slm'
    manager = ModelManager(model_path=model_path)
    await manager.load_model()
    status = await manager.get_status()
    print(f'  Status: {status["status"]}')
    print(f'  Path: {status["model_path"]}')
    print(f'  Mock: {status["is_mock"]}')

    # 2. Test generation
    print('\n[2] Testing generation...')
    result = await manager.generate('What is photosynthesis?')
    print(f'  Result: {result[:100]}')

    # 3. Test world model
    print('\n[3] Testing WorldModel...')
    wm = WorldModel(imagination_depth=1, thinking_steps=2)

    thoughts = await wm.think('What is ML?')
    print(f'  Thoughts: {len(thoughts)}')

    scenarios = await wm.imagine_scenarios('What is ML?')
    print(f'  Scenarios: {len(scenarios)}')

    # 4. Test generate with world model
    print('\n[4] Testing world model generation...')
    result = await wm.generate_response(
        'What is ML?',
        scenarios=[], thoughts=[],
        model_manager=manager
    )
    print(f'  Response: {str(result)[:100]}')

    # 5. Test RL agent evaluation
    print('\n[5] Testing RL evaluation...')
    quality = wm.rl.evaluate('Machine learning is a field of AI.', 'What is ML?')
    print(f'  Quality score: {quality:.2f}')

    # 6. Test tools
    print('\n[6] Testing tools...')
    from core.tools import ToolRegistry
    tools = ToolRegistry()
    calc_result = await tools.execute('calculator', {'expression': '2+2'})
    print(f'  Calculator: {calc_result.result}')

    await manager.unload_model()
    print('\n' + '=' * 60)
    print('ALL TESTS PASSED!')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(test())
