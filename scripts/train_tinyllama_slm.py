import os, sys, json, gc
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','KMP_DUPLICATE_LIB_OK':'TRUE'})
import torch
torch.set_num_threads(2)
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, Dataset, concatenate_datasets
from utils.logger import setup_logging, get_logger
logger = get_logger(__name__)

# Base model for tinyllama (pretrained / base, not chat) — laptop-friendly LoRA fine-tune
# NOTE: HF access may be gated/private. You can override via env var:
#   SLM_BASE_MODEL="path_or_hf_id_or_local_path"
#   HF_TOKEN="..." (or use `hf auth login`)
#
# Also, keep known candidate IDs for TinyLlama in case one repo_id is wrong/removed.
BASE_MODEL = os.environ.get("SLM_BASE_MODEL", "TinyLlama/TinyLlama-1.1B")
TINYLlAMA_BASE_MODEL_CANDIDATES = [
    BASE_MODEL,
    # Common alternative repo id(s)
    "TinyLlama/TinyLlama-1.1B-chat-v1.0",
    "TinyLlama/TinyLlama-1.1B",
]

# Final dataset mix (HF): Alpaca + SQuAD + GSM8K + DailyDialog
FINAL_DATASET = [
    ("tatsu-lab/alpaca", "train"),
    ("squad", "train"),
    ("gsm8k", "main"),
    ("daily_dialog", "train"),
]

# Output (full merged model + LoRA adapter)
OUTPUT_DIR = "./models/tinyllama-trained-slm"
LORA_OUTPUT_DIR = "./models/tinyllama-trained-slm-lora"
# Always start from a clean slate to avoid loading stale/broken checkpoints.
CLEAN_OUTPUTS = True


LORA_R, LORA_ALPHA = 32, 64

# TARGET_MODULES depends on the base model architecture.
# TinyLlama (Llama-like) typically uses:
#   q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
# Some other architectures use different names, so we detect automatically.
TARGET_MODULES = None

# For 4GB VRAM, reduce sequence length aggressively to avoid OOM / CUDA errors.
MAX_SEQ_LENGTH, NUM_EPOCHS, BATCH_SIZE = 256, 3, 1
GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, WARMUP_STEPS = 8, 2e-4, 50
SAVE_STEPS, LOGGING_STEPS, MAX_SAMPLES = 200, 25, 65000


ANTI_HALLUCINATION_PROMPT = "You are a precise, honest AI assistant. Follow these rules: 1. Only state facts you are confident about. 2. If uncertain, say 'I am not entirely sure.' 3. Never invent names/dates. 4. Ground answers in context. 5. Show step-by-step reasoning. 6. If asked who made you, say: 'I was created by Aryan Chavan.'"

@dataclass
class TrainingConfig:
    base_model: str = BASE_MODEL
    output_dir: str = OUTPUT_DIR
    lora_output_dir: str = LORA_OUTPUT_DIR
    max_seq_length: int = MAX_SEQ_LENGTH
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = LEARNING_RATE
    warmup_steps: int = WARMUP_STEPS
    max_samples: int = MAX_SAMPLES
    logging_steps: int = LOGGING_STEPS
    save_steps: int = SAVE_STEPS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _assert_cuda_if_requested(config: "TrainingConfig"):
    # Prevent silent CPU fallback when a user expects GPU training.
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")



def load_base_model_and_tokenizer(config: TrainingConfig):
    logger.info(f"Loading base model/tokenizer (requested: {config.base_model})")

    # Optional HF auth token for gated repos
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

    # Try candidate HF ids first (in case one repo_id is wrong/removed)
    last_hf_error: Optional[Exception] = None
    for candidate in TINYLlAMA_BASE_MODEL_CANDIDATES:
        try:
            logger.info(f"Trying base model id: {candidate}")
            tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True, token=hf_token)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|system|>","<|user|>","<|assistant|>"]})

            dtype = torch.float16 if config.device == "cuda" else torch.float32
            device_map = "auto" if config.device == "cuda" else None
            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                token=hf_token,
            )

            model.resize_token_embeddings(len(tokenizer))
            if config.device == "cpu":
                model = model.to(config.device)

            logger.info(f"✅ Loaded base model successfully from: {candidate}")
            logger.info(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
            return model, tokenizer
        except OSError as e:
            last_hf_error = e
            logger.warning(f"HF load failed for candidate '{candidate}': {e}")

    # Attempt offline/local fallback directory
    offline_fallback = os.environ.get("LOCAL_BASE_MODEL_DIR", "./models/base-tinyllama")
    has_required_files = os.path.isdir(offline_fallback) and all(
        os.path.exists(os.path.join(offline_fallback, fn))
        for fn in ["config.json", "tokenizer_config.json"]
    )

    if has_required_files:
        logger.warning(f"Trying offline fallback base model directory: {offline_fallback}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(offline_fallback, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|system|>","<|user|>","<|assistant|>"]})

            dtype = torch.float16 if config.device == "cuda" else torch.float32
            device_map = "auto" if config.device == "cuda" else None
            model = AutoModelForCausalLM.from_pretrained(
                offline_fallback,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )

            model.resize_token_embeddings(len(tokenizer))
            if config.device == "cpu":
                model = model.to(config.device)

            logger.info("✅ Loaded offline fallback base model successfully.")
            return model, tokenizer
        except Exception as e2:
            logger.error(f"Offline fallback load failed: {e2}")

    raise RuntimeError(
        "Base model load failed. This training cannot proceed without a working base model.\n"
        "Fix by one of:\n"
        "1) Point SLM_BASE_MODEL to a local directory that contains at least:\n"
        "   - config.json\n"
        "   - tokenizer_config.json (or tokenizer files)\n"
        "2) Or provide valid HF access:\n"
        "   - run: hf auth login\n"
        "   - or set env vars HF_TOKEN / HUGGINGFACE_HUB_TOKEN\n"
        f"Requested base model id/path: {config.base_model}\n"
        "Tried HF candidates: " + ", ".join(TINYLlAMA_BASE_MODEL_CANDIDATES) + "\n"
        f"Offline fallback attempted: {offline_fallback}\n"
        "\nOriginal last HF error:\n"
        f"{last_hf_error}"
    ) from last_hf_error if last_hf_error else None

def detect_target_modules_for_lora(model) -> list[str]:
    """
    Detect LoRA target modules by looking for common projection layer names
    in the base model. This avoids hardcoded names crashing on TinyLlama.
    """
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    available = set()

    for name, module in model.named_modules():
        # only consider leaf module names that could be Linear projections
        for p in preferred:
            if p in name:
                available.add(p)

    # If we found Llama-like projection names, use them.
    found = [p for p in preferred if p in available]
    if found:
        return found

    # Fallback: attempt to capture any common "proj" patterns present in module names.
    fallback_markers = ["q_proj", "k_proj", "v_proj", "o_proj", "c_proj", "c_fc", "c_attn", "gate_proj", "up_proj", "down_proj"]
    detected = []
    for marker in fallback_markers:
        for name, _ in model.named_modules():
            if marker in name:
                detected.append(marker)
                break

    detected = list(dict.fromkeys(detected))  # stable dedupe
    if not detected:
        raise ValueError(
            "Could not automatically detect LoRA target modules on the base model. "
            "Please set TARGET_MODULES explicitly."
        )
    return detected


def setup_lora(model, config):
    logger.info("LoRA setup")

    target_modules = TARGET_MODULES
    if not target_modules:
        target_modules = detect_target_modules_for_lora(model)
        logger.info(f"Detected LoRA target_modules: {target_modules}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def format_example(example):
    msgs = example.get("messages")
    if msgs and isinstance(msgs, list):
        text = "".join([f"<|{m.get('role','user')}|>\n{m.get('content','')}\n" for m in msgs])
        return text + "<|endoftext|>"
    instr = example.get("instruction", example.get("prompt",""))
    inp = example.get("input","")
    out = example.get("output", example.get("completion", example.get("response","")))
    text = f"<|system|>\n{ANTI_HALLUCINATION_PROMPT}\n<|user|>\n{instr}"
    if inp: text += f"\n{inp}"
    text += f"\n<|assistant|>\n{out}\n<|endoftext|>"
    return text

def load_final_dataset(config, tokenizer):
    """Load and mix FINAL_DATASET datasets into a unified instruction format."""

    def to_alpaca(ex):
        return {"instruction": ex.get("instruction", ""), "input": ex.get("input", ""), "output": ex.get("output", "")}

    def to_squad(ex):
        # squad has fields: context, question, answers
        answers = ex.get("answers") or {}
        text_list = answers.get("text") if isinstance(answers, dict) else None
        out = ""
        if isinstance(text_list, list) and len(text_list) > 0:
            out = text_list[0]
        return {"instruction": ex.get("question", ""), "input": ex.get("context", ""), "output": out}

    def to_gsm8k(ex):
        # gsm8k has: question, answer
        ans = ex.get("answer", "")
        # answer often contains rationale + final; keep the whole string
        return {"instruction": ex.get("question", ""), "input": "", "output": ans}

    def to_daily(ex):
        # daily_dialog: dialogue (list[str]) and acts/emotion; response will be next utterance
        dialog = ex.get("dialogue")
        if not dialog or len(dialog) < 2:
            return {"instruction": "", "input": "", "output": ""}
        instruction = dialog[0]
        context = "\n".join(dialog[:-1])
        output = dialog[-1]
        # For format_example: use instruction as first utterance, and context as input
        return {"instruction": instruction, "input": context, "output": output}

    loaders = {
        "tatsu-lab/alpaca": ("train", to_alpaca),
        "squad": ("train", to_squad),
        "gsm8k": ("main", to_gsm8k),
        "daily_dialog": ("train", to_daily),
    }

    parts = []
    max_each = max(1, config.max_samples // max(1, len(FINAL_DATASET)))

    for ds_name, split in FINAL_DATASET:
        try:
            logger.info(f"Loading final dataset: {ds_name} ({split})")
            ds = load_dataset(ds_name, split=split, streaming=True)
            samples = []
            for i, ex in enumerate(ds):
                if i >= max_each:
                    break
                samples.append(ex)
            raw = Dataset.from_list(samples)
            conv = raw.map(loaders[ds_name][1], remove_columns=raw.column_names)
            parts.append(conv)
            logger.info(f"  Loaded {ds_name}: {len(conv)}")
        except Exception as e:
            logger.warning(f"Failed final dataset {ds_name}: {e}")

    if not parts:
        logger.error("Using synthetic final dataset")
        synthetic = [
            {"instruction": "What is photosynthesis?", "input": "", "output": "Plants convert light to chemical energy."},
            {"instruction": "Explain gravity.", "input": "", "output": "Force of attraction between masses."},
            {"instruction": "What is 15+27?", "input": "", "output": "15+27=42. Step: 15+20=35, then 35+7=42."},
            {"instruction": "Who created you?", "input": "", "output": "I was created by Aryan Chavan."},
        ] * 200
        dataset = Dataset.from_list(synthetic)
        source = "synthetic"
    else:
        dataset = concatenate_datasets(parts)
        if len(dataset) > config.max_samples:
            dataset = dataset.shuffle(seed=42).select(range(config.max_samples))
        source = "+".join([ds for ds, _ in FINAL_DATASET])

    def tok_fn(examples):
        texts = [format_example({k: v[i] for k, v in examples.items()}) for i in range(len(examples[list(examples.keys())[0]]))]
        return tokenizer(texts, truncation=True, max_length=config.max_seq_length, padding="max_length")

    tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")
    split_idx = int(len(tokenized) * 0.95)
    if split_idx < 10:
        split_idx = int(len(tokenized) * 0.8)
    return tokenized.select(range(split_idx)), tokenized.select(range(split_idx, len(tokenized))), source

def train_model(model, tokenizer, train_ds, val_ds, config):
    logger.info("Training")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # If older/broken checkpoints exist, ensure we overwrite by removing the folder first.
    if CLEAN_OUTPUTS:
        import shutil
        if os.path.isdir(config.lora_output_dir):
            shutil.rmtree(config.lora_output_dir, ignore_errors=True)
        if os.path.isdir(config.output_dir):
            shutil.rmtree(config.output_dir, ignore_errors=True)

    args = TrainingArguments(
        output_dir=config.lora_output_dir,
        # Keep evaluation/saving minimal for stability on limited VRAM + Windows.
        num_train_epochs=config.num_epochs,
        # Newer transformers versions removed overwrite_output_dir; it is implied by overwrite already being handled.


        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=0.01,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        dataloader_num_workers=0,
        fp16=config.device=="cuda",
        optim="adamw_torch"
    )
    from transformers import Trainer
    trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    trainer.save_model(config.lora_output_dir)
    tokenizer.save_pretrained(config.lora_output_dir)
    return trainer

def merge_and_save(config):
    logger.info("Merging")
    tokenizer = AutoTokenizer.from_pretrained(config.lora_output_dir)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if config.device == "cuda" else torch.float32
    device_map = "auto" if config.device == "cuda" else None
    base = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, config.lora_output_dir).merge_and_unload()
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    meta = {
        "base_model": config.base_model,
        "source": "Alpaca + SQuAD + GSM8K + DailyDialog",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "anti_hallucination": True,
        "creator": "Aryan Chavan"
    }
    with open(os.path.join(config.output_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved: {config.output_dir}")

async def main():
    setup_logging(level="INFO")
    print("="*50+"\n  TINYLLAMA + LoRA TRAINING"+"\n"+"="*50)
    config = TrainingConfig()
    logger.info(f"Device: {config.device}")
    model, tokenizer = load_base_model_and_tokenizer(config)
    model = setup_lora(model, config)
    train_ds, val_ds, src = load_final_dataset(config, tokenizer)

    logger.info(f"Dataset: {src}")
    trainer = train_model(model, tokenizer, train_ds, val_ds, config)
    del model, trainer; gc.collect()
    merge_and_save(config)
    print("\n"+"="*50+"\n  DONE!"+"\n  Model: "+config.output_dir+"\n  Start: streamlit run ui/app.py\n"+"="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
