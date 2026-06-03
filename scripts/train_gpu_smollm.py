"""World Model SLM 2026 - GPU Optimized Training
Trains SmolLM2-360M with QLoRA on RTX 3050 4GB GPU.
Uses modern transformers API (BitsAndBytesConfig, dtype).
Output: ./models/smollm2-360m-trained-slm (merged) + ./models/smollm2-360m-trained-slm-lora (adapter)
"""

import os, sys, json, gc

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    'TF_ENABLE_ONEDNN_OPTS': '0',
})

# Force HF cache to D: drive
os.environ.setdefault('HF_HOME', 'D:/.hf_cache')
os.environ.setdefault('HF_HUB_CACHE', 'D:/.hf_cache/hub')

import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer, TrainerCallback,
    BitsAndBytesConfig
)
from transformers import PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, Dataset, concatenate_datasets
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

# === CONFIG ===
BASE_MODEL = os.environ.get("SLM_BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
OUTPUT_DIR = "./models/smollm2-360m-trained-slm"
LORA_OUTPUT_DIR = "./models/smollm2-360m-trained-slm-lora"

LORA_R, LORA_ALPHA = 16, 32
LORA_DROPOUT = 0.1

FINAL_DATASET = [
    ("tatsu-lab/alpaca", "train"),
    ("squad", "train"),
    ("gsm8k", "main"),
]

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "3e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "50"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "500"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "25"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "5000"))
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "-1"))
GC_INTERVAL = int(os.environ.get("GC_INTERVAL", "40"))

ANTI_HALLUCINATION_PROMPT = (
    "You are a precise, honest AI assistant. Follow these rules: "
    "1. Only state facts you are confident about. "
    "2. If uncertain, say 'I am not entirely sure.' "
    "3. Never invent names/dates. "
    "4. Ground answers in context. "
    "5. Show step-by-step reasoning. "
    "6. If asked who made you, say: 'I was created by Aryan Chavan.'"
)


def format_example(instruction, inp, output):
    text = f"[SYSTEM]\n{ANTI_HALLUCINATION_PROMPT}\n\n[USER]\n{instruction}"
    if inp:
        text += f"\n\n{inp}"
    text += f"\n\n[ASSISTANT]\n{output}"
    return text


def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        logger.info("Loading with 4-bit QLoRA for 4GB GPU...")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=quant,
            trust_remote_code=True,
        )
    else:
        logger.info("Loading in fp32 for CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )

    logger.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB" if device == "cuda" else "Model loaded on CPU")
    return model, tokenizer


def setup_lora(model):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    available = set()
    for name, _ in model.named_modules():
        for p in target_modules:
            if p in name:
                available.add(p)
    found = [p for p in target_modules if p in available]
    logger.info(f"LoRA targets: {found}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=found,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_datasets(tokenizer):
    def to_alpaca(ex):
        return {"instruction": ex.get("instruction", ""), "input": ex.get("input", ""), "output": ex.get("output", "")}
    def to_squad(ex):
        answers = ex.get("answers") or {}
        text_list = answers.get("text") if isinstance(answers, dict) else None
        out = text_list[0] if isinstance(text_list, list) and text_list else ""
        return {"instruction": ex.get("question", ""), "input": ex.get("context", ""), "output": out}
    def to_gsm8k(ex):
        return {"instruction": ex.get("question", ""), "input": "", "output": ex.get("answer", "")}
    loaders = {
        "tatsu-lab/alpaca": ("train", None, to_alpaca),
        "squad": ("train", None, to_squad),
        "gsm8k": ("train", "main", to_gsm8k),
    }

    parts = []
    max_each = max(1, MAX_SAMPLES // max(1, len(FINAL_DATASET)))

    for ds_name, split in FINAL_DATASET:
        try:
            split_name, ds_config, converter = loaders[ds_name]
            logger.info(f"Loading dataset: {ds_name} ({split_name})")
            kwargs = {"split": split_name, "streaming": True}
            if ds_config:
                kwargs["name"] = ds_config
            ds = load_dataset(ds_name, **kwargs)
            samples = []
            for i, ex in enumerate(ds):
                if i >= max_each:
                    break
                samples.append(ex)
            raw = Dataset.from_list(samples)
            conv = raw.map(converter, remove_columns=raw.column_names)
            parts.append(conv)
            logger.info(f"  Loaded {ds_name}: {len(conv)}")
        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")

    if not parts:
        logger.warning("Using synthetic fallback data")
        synthetic = [
            {"instruction": "What is photosynthesis?", "input": "", "output": "Plants convert light into chemical energy."},
            {"instruction": "Explain gravity.", "input": "", "output": "Gravity is the attraction between masses."},
            {"instruction": "What is 15+27?", "input": "", "output": "15+27=42."},
        ] * 200
        dataset = Dataset.from_list(synthetic)
    else:
        dataset = concatenate_datasets(parts)
        if len(dataset) > MAX_SAMPLES:
            dataset = dataset.shuffle(seed=42).select(range(MAX_SAMPLES))

    def tok_fn(examples):
        texts = [
            format_example(examples["instruction"][i], examples["input"][i], examples["output"][i])
            for i in range(len(examples["instruction"]))
        ]
        return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")

    tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")
    split_idx = max(int(len(tokenized) * 0.95), min(10, len(tokenized) - 1))
    return tokenized.select(range(split_idx)), tokenized.select(range(split_idx, len(tokenized)))


class PeriodicGCCallback(TrainerCallback):
    def __init__(self, interval=40):
        self.interval = interval
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def train_model(model, tokenizer, train_ds, val_ds):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    import shutil
    for p in [LORA_OUTPUT_DIR, OUTPUT_DIR]:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)

    max_steps_val = MAX_TRAIN_STEPS if MAX_TRAIN_STEPS > 0 else -1
    args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        max_steps=max_steps_val,
        num_train_epochs=NUM_EPOCHS if max_steps_val <= 0 else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        fp16=True,
        bf16=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[PeriodicGCCallback(interval=GC_INTERVAL)],
    )

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        raise
    finally:
        trainer.save_model(LORA_OUTPUT_DIR)
        tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    return trainer


def merge_and_save(model, tokenizer):
    logger.info("Merging LoRA into base model...")
    merged = None

    try:
        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            logger.info("In-memory merge OK")
    except Exception as e:
        logger.warning(f"In-memory merge failed: {e}")

    if merged is None:
        logger.info("Reloading base + adapter for merge")
        base, _ = load_model_and_tokenizer()
        merged = PeftModel.from_pretrained(base, LORA_OUTPUT_DIR)
        merged = merged.merge_and_unload()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    meta = {
        "base_model": BASE_MODEL,
        "source": "Alpaca + SQuAD + GSM8K + DailyDialog",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "anti_hallucination_prompt": True,
        "creator": "Aryan Chavan",
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Merged model saved to {OUTPUT_DIR}")


def main():
    setup_logging(level="INFO")

    print("=" * 60)
    print(f"  GPU TRAINING: {BASE_MODEL}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB" if torch.cuda.is_available() else "")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()
    model = setup_lora(model)
    train_ds, val_ds = load_datasets(tokenizer)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    train_model(model, tokenizer, train_ds, val_ds)
    merge_and_save(model, tokenizer)

    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Merged: {OUTPUT_DIR}")
    print(f"  Start: python main.py --ui")
    print("=" * 60)


if __name__ == "__main__":
    main()
