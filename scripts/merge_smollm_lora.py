"""Merge SmolLM2-360M LoRA adapter into standalone model."""
import os, sys, json, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.update({
    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
})
os.environ.setdefault('HF_HOME', 'D:/.hf_cache')
os.environ.setdefault('HF_HUB_CACHE', 'D:/.hf_cache/hub')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

LORA_PATH = "./models/smollm2-360m-trained-slm-lora"
OUTPUT_PATH = "./models/smollm2-360m-trained-slm"
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("=" * 60)
    print(f"  Merging SmolLM2-360M LoRA into standalone model")
    print(f"  Base: {BASE_MODEL}")
    print(f"  LoRA: {LORA_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)

    if DEVICE == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
            quantization_config=bnb, trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, device_map=None,
            trust_remote_code=True,
        )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, LORA_PATH)

    print("Merging...")
    merged = model.merge_and_unload()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    meta = {
        "base_model": BASE_MODEL,
        "lora_source": LORA_PATH,
        "type": "merged_lora",
        "params": "360M",
        "creator": "Aryan Chavan",
    }
    with open(os.path.join(OUTPUT_PATH, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Merged model saved to {OUTPUT_PATH}")
    total = sum(p.numel() for p in merged.parameters())
    size = sum(p.numel() * p.element_size() for p in merged.parameters())
    print(f"   Parameters: {total/1e6:.1f}M")
    print(f"   Size: {size/1e6:.1f} MB (fp16: {total*2/1e6:.1f} MB)")

    del base, model, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
