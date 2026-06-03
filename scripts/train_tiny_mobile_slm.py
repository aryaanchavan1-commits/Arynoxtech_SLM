"""Train an ultra-tiny SLM (50M params) from scratch for mobile devices.
Uses the custom SLM architecture from core/slm_architecture.py.
Output: ./models/tiny-mobile-slm (50M params, ~100MB fp16)
"""

import os, sys, json, gc, math
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ.update({
    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'KMP_DUPLICATE_LIB_OK': 'TRUE', 'TF_ENABLE_ONEDNN_OPTS': '0',
})
os.environ.setdefault('HF_HOME', 'D:/.hf_cache')
os.environ.setdefault('HF_HUB_CACHE', 'D:/.hf_cache/hub')

import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.slm_architecture import SLMModel
from transformers import PreTrainedTokenizerFast, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

OUTPUT_DIR = "./models/tiny-mobile-slm"
VOCAB_SIZE = 16000
DIM = 384
NUM_LAYERS = 8
NUM_HEADS = 6
NUM_KV_HEADS = 3
MAX_SEQ_LEN = 256
MLP_RATIO = 3.5

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-4"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "5000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    "How does a computer work? A computer processes data using a CPU, memory, and storage.",
    "What is machine learning? Machine learning is a subset of AI where models learn from data.",
    "Explain neural networks. Neural networks are computing systems inspired by biological brains.",
    "What is Python? Python is a high-level programming language.",
    "How does the internet work? The internet is a global network of connected computers.",
    "What is an API? An API allows different software applications to communicate.",
    "Explain databases. Databases store and organize data for efficient retrieval.",
    "What is the capital of France? The capital of France is Paris.",
    "How do plants grow? Plants grow through photosynthesis using sunlight, water, and nutrients.",
    "What is water made of? Water is made of two hydrogen atoms and one oxygen atom (H2O).",
    "Explain the water cycle. Water evaporates, condenses into clouds, and precipitates as rain.",
    "What is energy? Energy is the ability to do work.",
    "Explain electricity. Electricity is the flow of electrons through a conductor.",
    "What is the speed of light? The speed of light is approximately 300,000 km/s.",
    "What is DNA? DNA is the molecule that carries genetic instructions for life.",
    "What is an algorithm? An algorithm is a step-by-step procedure for solving a problem.",
    "Explain cloud computing. Cloud computing delivers computing services over the internet.",
    "What is cybersecurity? Cybersecurity protects computer systems from theft or damage.",
    "How does GPS work? GPS uses satellites to triangulate positions on Earth.",
    "What is blockchain? Blockchain is a distributed ledger technology.",
    "Explain quantum computing. Quantum computing uses quantum bits for computation.",
    "What is the Turing test? The Turing test evaluates a machine's ability to exhibit human intelligence.",
    "Explain the difference between AI and ML. AI is the broad field of intelligent machines; ML is a subset.",
    "What is 2+2? 2+2=4.",
    "What is the boiling point of water? Water boils at 100 degrees Celsius at sea level.",
    "Explain what a CPU does. A CPU executes instructions from computer programs.",
    "What is RAM? RAM is temporary memory that stores data for active programs.",
    "What is an operating system? An OS manages computer hardware and software resources.",
    "Explain the internet. The internet is a global network of computers that communicate via TCP/IP.",
    "What is a database? A database is an organized collection of structured information.",
    "Explain encryption. Encryption converts data into a coded form to prevent unauthorized access.",
    "What is the cloud? The cloud refers to servers accessed over the internet for computing services.",
    "Explain binary. Binary is a base-2 number system using only 0 and 1.",
    "What is HTTP? HTTP is the protocol used for transferring web pages.",
    "How does WiFi work? WiFi uses radio waves to transmit data between devices.",
    "What is SSL? SSL encrypts data between a web server and browser.",
    "Explain DNS. DNS converts domain names to IP addresses.",
    "What is a VPN? A VPN creates an encrypted tunnel for secure internet access.",
    "Explain AI ethics. AI ethics addresses the moral implications of artificial intelligence.",
    "What is robotics? Robotics designs and builds machines that can perform tasks autonomously.",
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

def format_example(instruction, inp, output):
    text = f"[SYSTEM]\n{ANTI_HALLUCINATION_PROMPT}\n\n[USER]\n{instruction}"
    if inp:
        text += f"\n\n{inp}"
    text += f"\n\n[ASSISTANT]\n{output}"
    return text

def load_real_datasets(tokenizer, max_samples=5000):
    parts = []

    loaders = [
        ("tatsu-lab/alpaca", "train", lambda ex: {
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "output": ex.get("output", ""),
        }),
        ("squad", "train", lambda ex: {
            "instruction": ex.get("question", ""),
            "input": (ex.get("answers") or {}).get("text", [""])[0] if isinstance(ex.get("answers"), dict) else "",
            "output": (ex.get("answers") or {}).get("text", [""])[0] if isinstance(ex.get("answers"), dict) else "",
        }),
        ("gsm8k", "main", lambda ex: {
            "instruction": ex.get("question", ""),
            "input": "",
            "output": ex.get("answer", ""),
        }),
    ]

    max_each = max(1, max_samples // len(loaders))

    for ds_name, split, converter in loaders:
        try:
            logger.info(f"Loading dataset: {ds_name} ({split})")
            ds = load_dataset(ds_name, split=split, streaming=True)
            samples = []
            for i, ex in enumerate(ds):
                if i >= max_each:
                    break
                samples.append(ex)
            raw = Dataset.from_list(samples)
            conv = raw.map(converter, remove_columns=raw.column_names)
            parts.append(conv)
            logger.info(f"  Loaded {ds_name}: {len(conv)} samples")
        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")

    if parts:
        dataset = concatenate_datasets(parts)
        if len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=42).select(range(max_samples))
        return dataset
    return None

def generate_synthetic_backup(tokenizer, num_samples=5000):
    import random
    pairs = [
        ("What is photosynthesis?", "Plants convert light into chemical energy using chlorophyll."),
        ("Explain gravity", "Gravity is the attractive force between objects with mass."),
        ("What is 15+27?", "15+27=42."),
        ("Who created you?", "I was created by Aryan Chavan."),
        ("What is machine learning?", "Machine learning is a field of AI where systems learn from data patterns."),
        ("How does the internet work?", "The internet connects computers globally through standardized protocols."),
        ("What is Python?", "Python is a high-level, interpreted programming language."),
        ("Explain the water cycle", "Water evaporates, condenses into clouds, and falls as precipitation."),
    ]
    texts = []
    for _ in range(num_samples):
        q, a = random.choice(pairs)
        texts.append({
            "instruction": q, "input": "", "output": a,
        })
    return Dataset.from_list(texts)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    setup_logging(level="INFO")

    print("=" * 60)
    print("  TINY MOBILE SLM - TRAINING FROM SCRATCH")
    print(f"  Architecture: {DIM}dim, {NUM_LAYERS}layers, {NUM_HEADS}heads")
    print(f"  Device: {DEVICE}")
    print(f"  Samples: {MAX_SAMPLES} | Epochs: {NUM_EPOCHS} | Seq: {MAX_SEQ_LEN}")
    print("=" * 60)

    tokenizer = build_tokenizer()
    logger.info(f"Tokenizer built: vocab_size={tokenizer.vocab_size}")

    model = SLMModel(
        vocab_size=tokenizer.vocab_size,
        dim=DIM, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
        max_seq_len=MAX_SEQ_LEN, mlp_ratio=MLP_RATIO,
        tie_weights=True,
    )

    n_params = count_parameters(model)
    logger.info(f"Model parameters: {n_params/1e6:.1f}M")
    logger.info(f"Footprint (fp16): {n_params*2/1e6:.1f} MB")
    logger.info(f"Footprint (int8): {n_params*1/1e6:.1f} MB")

    dataset = load_real_datasets(tokenizer, max_samples=MAX_SAMPLES)
    if dataset is None or len(dataset) < 100:
        logger.warning("Real datasets unavailable; using synthetic backup")
        dataset = generate_synthetic_backup(tokenizer, num_samples=MAX_SAMPLES)

    def tok_fn(examples):
        texts = [
            format_example(examples["instruction"][i], examples["input"][i], examples["output"][i])
            for i in range(len(examples["instruction"]))
        ]
        return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    logger.info(f"Tokenizing {len(dataset)} samples...")
    tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")
    split_idx = max(int(len(tokenized) * 0.95), 1)
    train_ds = tokenized.select(range(split_idx))
    val_ds = tokenized.select(range(split_idx, len(tokenized)))
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_fp16 = DEVICE == "cuda"
    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}-checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        report_to=["none"],
        dataloader_num_workers=0,
        fp16=use_fp16,
        bf16=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model, args=args, data_collator=collator,
        train_dataset=train_ds, eval_dataset=val_ds,
    )

    model.prepare_for_save()
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.prepare_for_save()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    model.prepare_for_save()
    if DEVICE == "cuda":
        model = model.half().cpu()
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_fp16.pt"))

    meta = {
        "architecture": "custom_slm_gqa_rope_swiglu",
        "params": n_params,
        "dim": DIM, "layers": NUM_LAYERS,
        "heads": NUM_HEADS, "kv_heads": NUM_KV_HEADS,
        "vocab_size": tokenizer.vocab_size, "max_seq_len": MAX_SEQ_LEN,
        "creator": "Aryan Chavan",
        "datasets": "alpaca+squad+gsm8k",
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print(f"  TINY MOBILE SLM TRAINED!")
    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  Model saved: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
