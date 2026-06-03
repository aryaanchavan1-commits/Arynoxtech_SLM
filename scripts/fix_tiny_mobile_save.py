"""Fix the shared tensor memory issue and save the trained tiny mobile model."""
import os, sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = "./models/tiny-mobile-slm"
CKPT_DIR = "./models/tiny-mobile-slm-checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

from core.slm_architecture import SLMModel
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

VOCAB_SIZE = 16000
DIM = 384
NUM_LAYERS = 8
NUM_HEADS = 6
NUM_KV_HEADS = 3
MAX_SEQ_LEN = 256
MLP_RATIO = 3.5

ANTI_HALLUCINATION_PROMPT = (
    "You are a precise, honest AI assistant. Follow these rules: "
    "1. Only state facts you are confident about. "
    "2. If uncertain, say 'I am not entirely sure.' "
    "3. Never invent names/dates. "
    "4. Ground answers in context. "
    "5. Show step-by-step reasoning. "
    "6. If asked who made you, say: 'I was created by Aryan Chavan.'"
)

TRAINING_CORPUS = [
    ANTI_HALLUCINATION_PROMPT,
    "[SYSTEM] [USER] [ASSISTANT]",
    "What is photosynthesis? Plants convert light into chemical energy.",
    "Who made you? I was created by Aryan Chavan.",
    "Explain gravity. Gravity is the attraction between masses.",
    "What is 15+27? 15+27=42.",
]

def build_tokenizer():
    special = ["<pad>", "<unk>", "<s>", "</s>"]
    words = set()
    for text in TRAINING_CORPUS:
        words.update(text.replace("\n", " ").split())
    vocab = {tok: i for i, tok in enumerate(special)}
    for word in sorted(words):
        if word not in vocab:
            vocab[word] = len(vocab)
    while len(vocab) < VOCAB_SIZE:
        vocab[f"<extra_{len(vocab)}>"] = len(vocab)
    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="<unk>", pad_token="<pad>",
        bos_token="<s>", eos_token="</s>",
    )
    return fast

tokenizer = build_tokenizer()
tokenizer.save_pretrained(OUTPUT_DIR)

checkpoints = [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")]
if not checkpoints:
    print("No checkpoints found!")
    sys.exit(1)

latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
ckpt_path = os.path.join(CKPT_DIR, latest)
print(f"Loading checkpoint: {ckpt_path}")

model = SLMModel(
    vocab_size=tokenizer.vocab_size,
    dim=DIM, num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
    max_seq_len=MAX_SEQ_LEN, mlp_ratio=MLP_RATIO,
    tie_weights=True,
)

state = torch.load(os.path.join(ckpt_path, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state, strict=False)

# Clone shared weight to avoid safetensors crash
model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

model = model.half()
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_fp16.pt"))

n_params = sum(p.numel() for p in model.parameters())
meta = {
    "architecture": "custom_slm_gqa_rope_swiglu",
    "params": n_params,
    "dim": DIM, "layers": NUM_LAYERS,
    "heads": NUM_HEADS, "kv_heads": NUM_KV_HEADS,
    "vocab_size": tokenizer.vocab_size, "max_seq_len": MAX_SEQ_LEN,
    "creator": "Aryan Chavan",
    "datasets": "alpaca+squad+gsm8k",
    "checkpoint": latest,
}
with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

fp16_size = os.path.getsize(os.path.join(OUTPUT_DIR, "model_fp16.pt"))
print(f"Model saved to {OUTPUT_DIR}")
print(f"Parameters: {n_params/1e6:.1f}M")
print(f"FP16 size: {fp16_size/1e6:.1f} MB")
