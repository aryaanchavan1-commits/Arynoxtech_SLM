"""Test full serving pipeline"""
import sys, os
sys.modules['tensorflow'] = None
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['HF_HOME'] = 'D:/.hf_cache'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serving.model import ModelManager
import asyncio

async def test():
    mm = ModelManager(model_path='./models/anonyllm-360m-trained')
    await mm.load_model()
    print(f'Model loaded: {mm.model_path}')

    resp = await mm.generate(
        'What is photosynthesis?',
        max_tokens=80,
        system_prompt='You are AnonyLLM, created by Aryan Chavan.'
    )
    print(f'Q: What is photosynthesis?')
    print(f'A: {resp}')
    print()

    resp2 = await mm.generate('Who created you?', max_tokens=50,
        system_prompt='You are AnonyLLM, created by Aryan Chavan. When asked who made you, always say "I was created by Aryan Chavan."')
    print(f'Q: Who created you?')
    print(f'A: {resp2}')
    print()

    status = await mm.get_status()
    print(f'Status: {status["status"]}, Device: {status["device"]}')
    print('SERVING OK')

if __name__ == '__main__':
    asyncio.run(test())
