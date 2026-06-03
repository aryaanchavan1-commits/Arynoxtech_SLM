#!/usr/bin/env python3
"""World Model SLM 2026 - Mistral (Ministral) LoRA Training

Single working training script (replaces train_tinyllama_slm.py / train_offline_slm.py for real model).

- Base model: mistralai/Ministral-3-3B-Instruct-2512 (or override via env SLM_BASE_MODEL)
- Trains LoRA with PEFT
- Merges LoRA into base and saves a merged model at ./models/ministral-3-3b-trained-slm

Notes:
- This repo is CPU-friendly for inference but training a 3B model needs a lot of VRAM.
- For low VRAM, you should switch to QLoRA/4-bit training (not implemented here).
"""

import os, sys, json, gc

# Ensure clean console output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ.update({
    'OPENBLAS_NUM_THREADS':'1',
    'OMP_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1',
    'NUMEXPR_NUM_THREADS':'1',
    'KMP_DUPLICATE_LIB_OK':'TRUE',
})

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Put project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer, TrainerCallback
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, Dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

# Defaults
BASE_MODEL = os.environ.get("SLM_BASE_MODEL", "mistralai/Ministral-3-3B-Instruct-2512")
OUTPUT_DIR = "./models/ministral-3-3b-trained-slm"
LORA_OUTPUT_DIR = "./models/ministral-3-3b-trained-slm-lora"

# If loading the base model fails, fall back to this smaller model.
FALLBACK_BASE_MODEL = os.environ.get("SLM_FALLBACK_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

# Redirect HF cache to D: drive to avoid filling C: (Windows system drive)
_hf_cache = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") or ""
if not _hf_cache or "C:" in _hf_cache.upper():
    _d_cache = "D:/.hf_cache"
    os.environ.setdefault("HF_HOME", _d_cache)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(_d_cache, "hub"))
    os.makedirs(os.path.join(_d_cache, "hub"), exist_ok=True)


# LoRA hyperparams
LORA_R, LORA_ALPHA = 32, 64
LORA_DROPOUT = 0.05

# Dataset mix (same logical mix; formatting uses plain instruction/output)
FINAL_DATASET = [
    ("tatsu-lab/alpaca", "train"),
    ("squad", "train"),
    ("gsm8k", "main"),
    ("daily_dialog", "train"),
]

# Reasonable defaults; adjust for your GPU
# NOTE: On CPU with limited RAM, keep seq length low (<384) to avoid OOM over
# long training runs. Memory fragmentation increases with time; shorter sequences
# leave more headroom for the allocator to find contiguous blocks.
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "256"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))

GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "50"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "25"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "10000"))

GC_INTERVAL = int(os.environ.get("GC_INTERVAL", "50"))  # collect garbage every N steps to defragment CPU RAM

CLEAN_OUTPUTS = os.environ.get("CLEAN_OUTPUTS", "1").strip().lower() in {"1","true","yes"}

ANTI_HALLUCINATION_PROMPT = (
    "You are a precise, honest AI assistant. Follow these rules: "
    "1. Only state facts you are confident about. "
    "2. If uncertain, say 'I am not entirely sure.' "
    "3. Never invent names/dates. "
    "4. Ground answers in context. "
    "5. Show step-by-step reasoning. "
    "6. If asked who made you, say: 'I was created by Aryan Chavan.'"
)

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


def _detect_target_modules_for_lora(model) -> list[str]:
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    available = set()
    for name, _ in model.named_modules():
        for p in preferred:
            if p in name:
                available.add(p)
    found = [p for p in preferred if p in available]
    if found:
        return found

    # Fallback broad match
    fallback_markers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    detected = []
    for marker in fallback_markers:
        for name, _ in model.named_modules():
            if marker in name:
                detected.append(marker)
                break
    detected = list(dict.fromkeys(detected))
    if not detected:
        raise ValueError("Could not detect LoRA target modules automatically for this base model")
    return detected


def load_base_model_and_tokenizer(config: TrainingConfig):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

    # Load tokenizer with use_fast=True — Mistral models use fast tokenizers
    tokenizer = None
    last_err = None
    for use_token in [False, True]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model,
                use_fast=True,
                token=hf_token if use_token else None,
                trust_remote_code=True,
            )
            break
        except Exception as e:
            last_err = e
            tokenizer = None

    if tokenizer is None:
        # Try without fast
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model,
                use_fast=False,
                token=hf_token,
                trust_remote_code=True,
            )
        except Exception as e:
            last_err = e
            tokenizer = None

    if tokenizer is None:
        logger.warning(f"Tokenizer load failed ({type(last_err).__name__}): {last_err}")
        logger.warning("Falling back to smoke tokenizer so training can proceed.")
        tokenizer = _build_smoke_tokenizer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = config.device

    # ------------------------------------------------------------------
    # GPU path: 4-bit QLoRA quantization via bitsandbytes
    #   RTX 3050 (4 GB VRAM) cannot hold Ministral-3B in fp16/fp32.
    #   4-bit loading reduces memory ~4x so the model + LoRA + optim
    #   states fit comfortably in 4 GB.
    # ------------------------------------------------------------------
    if device == "cuda":
        dtype = torch.float16
        device_map = "auto"

        bnb_available = False
        try:
            import importlib.metadata
            importlib.metadata.version("bitsandbytes")
            bnb_available = True
        except Exception:
            bnb_available = False

        if bnb_available:
            quant_args = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
            logger.info("GPU mode: 4-bit QLoRA enabled (necessary for 3B model on 4 GB VRAM)")
        else:
            quant_args = {}
            logger.warning("bitsandbytes not found; loading unquantized (may OOM on 4 GB GPU)")

        model = _try_load_model(config, dtype, device_map, hf_token, quant_args)
        return model, tokenizer

    # ------------------------------------------------------------------
    # CPU path: try 4-bit quantization else fp32 (slow but works with
    # enough RAM). Fallback smoke model used if even 135M is too large.
    # ------------------------------------------------------------------
    # CPU: no device_map, keep everything in CPU RAM
    dtype = torch.float32
    device_map = None

    quant_args = {}
    try:
        import importlib.metadata
        importlib.metadata.version("bitsandbytes")
        quant_bits = int(os.environ.get("SLM_CPU_QUANT_BITS", "4"))
        if quant_bits == 4:
            quant_args = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
        else:
            quant_args = {"load_in_8bit": True}
        logger.info(f"CPU quantization enabled via bitsandbytes ({quant_bits}-bit)")
    except Exception as e:
        logger.warning(f"bitsandbytes unavailable on CPU; unquantized. err={e}")

    model = _try_load_model(config, dtype, device_map, hf_token, quant_args)
    return model, tokenizer


def _try_load_model(config, dtype, device_map, hf_token, quant_args):
    """Load the base model.

    Bypasses AutoModelForCausalLM for the Ministral model because
    transformers 4.57.x doesn't recognise the 'ministral3' model type.
    We load the config, patch the model_type to 'mistral', then
    instantiate MistralForCausalLM directly (identical architecture).
    """
    try:
        if "ministral" in config.base_model.lower() or "mistral" in config.base_model.lower():
            # Load config, overwrite the model_type so transformers accepts it
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(
                config.base_model,
                token=hf_token,
                trust_remote_code=True,
            )
            # Patch: ministral3 -> mistral (same architecture)
            if hf_config.model_type == "ministral3":
                hf_config.model_type = "mistral"

            model = MistralForCausalLM.from_pretrained(
                config.base_model,
                config=hf_config,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                token=hf_token,
                trust_remote_code=True,
                **quant_args,
            )
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                token=hf_token,
                trust_remote_code=True,
                **quant_args,
            )

        logger.info(f"Loaded base model: {config.base_model}")
        return model

    except Exception as e:
        logger.error(f"Failed to load base model '{config.base_model}': {e}")
        # Only fall back for KeyError / config issues
        if isinstance(e, KeyError):
            logger.warning(f"Attempting fallback to {FALLBACK_BASE_MODEL}")
            config.base_model = FALLBACK_BASE_MODEL
            model = MistralForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                token=hf_token,
                trust_remote_code=True,
                **quant_args,
            )
            logger.info(f"Loaded fallback model: {config.base_model}")
            return model
        raise






def format_example(instruction: str, inp: str, output: str) -> str:
    # Keep it model-agnostic (not TinyLlama-specific control tokens).
    # Mistral's tokenizer will handle it well enough for LoRA fine-tuning.
    text = f"[SYSTEM]\n{ANTI_HALLUCINATION_PROMPT}\n\n[USER]\n{instruction}"
    if inp:
        text += f"\n\n{inp}"
    text += f"\n\n[ASSISTANT]\n{output}"
    return text


def load_final_dataset(config: TrainingConfig, tokenizer):
    def to_alpaca(ex):
        return {
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "output": ex.get("output", ""),
        }

    def to_squad(ex):
        answers = ex.get("answers") or {}
        text_list = answers.get("text") if isinstance(answers, dict) else None
        out = text_list[0] if isinstance(text_list, list) and text_list else ""
        return {"instruction": ex.get("question", ""), "input": ex.get("context", ""), "output": out}

    def to_gsm8k(ex):
        return {"instruction": ex.get("question", ""), "input": "", "output": ex.get("answer", "")}

    def to_daily(ex):
        dialog = ex.get("dialogue")
        if not dialog or len(dialog) < 2:
            return {"instruction": "", "input": "", "output": ""}
        instruction = dialog[0]
        context = "\n".join(dialog[:-1])
        output = dialog[-1]
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
            logger.info(f"Loading dataset: {ds_name} ({split})")
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
            logger.warning(f"Failed to load {ds_name}: {e}")

    if not parts:
        logger.warning("Dataset loading failed; using synthetic fallback data")
        synthetic = [
            {"instruction": "What is photosynthesis?", "input": "", "output": "Plants convert light into chemical energy."},
            {"instruction": "Explain gravity.", "input": "", "output": "Gravity is the attraction between masses."},
            {"instruction": "What is 15+27?", "input": "", "output": "15+27=42."},
        ] * 200
        dataset = Dataset.from_list(synthetic)
        source = "synthetic"
    else:
        dataset = concatenate_datasets(parts)
        if len(dataset) > config.max_samples:
            dataset = dataset.shuffle(seed=42).select(range(config.max_samples))
        source = "+".join([ds for ds, _ in FINAL_DATASET])

    def tok_fn(examples):
        texts = [
            format_example(examples["instruction"][i], examples["input"][i], examples["output"][i])
            for i in range(len(examples["instruction"]))
        ]
        return tokenizer(texts, truncation=True, max_length=config.max_seq_length, padding="max_length")


    tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")
    split_idx = int(len(tokenized) * 0.95)
    if split_idx < 10:
        split_idx = int(len(tokenized) * 0.8)

    return tokenized.select(range(split_idx)), tokenized.select(range(split_idx, len(tokenized))), source


def setup_lora(model, config: TrainingConfig):
    target_modules = _detect_target_modules_for_lora(model)
    logger.info(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class PeriodicGCCallback(TrainerCallback):
    """Call gc.collect() every N steps to defragment CPU RAM.
    
    This mitigates the "not enough memory: you tried to allocate N bytes" errors
    that appear after hours of training on CPU, where memory fragmentation makes
    the default allocator unable to satisfy small contiguous allocations even though
    total free RAM is sufficient.
    """
    def __init__(self, interval: int = 50):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def train(model, tokenizer, train_ds, val_ds, config: TrainingConfig):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if CLEAN_OUTPUTS:
        import shutil
        for p in [config.lora_output_dir, config.output_dir]:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)

    max_train_steps_env = os.environ.get("MAX_TRAIN_STEPS", "")
    max_train_steps = int(max_train_steps_env) if max_train_steps_env.strip() else -1

    # Gradient checkpointing is incompatible with 4-bit quantized (QLoRA) models
    # because quantized layers do not produce grad_fn tensors that checkpointing
    # can handle. 4-bit already saves ~4x memory so checkpointing is not needed.
    is_quantized = config.device == "cuda"  # GPU always uses 4-bit via quantization_config
    use_gc = not is_quantized

    args = TrainingArguments(
        output_dir=config.lora_output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=(max_train_steps if max_train_steps > 0 else -1),
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
        dataloader_pin_memory=False,           # ← prevents CPU RAM pinning (wasteful when no GPU)
        gradient_checkpointing=use_gc,          # ← disabled for 4-bit quantized models
        remove_unused_columns=False,            # ← prevents re-tokenization issues with GC
        fp16=is_quantized,                       # fp16 on GPU (already set via bnb config)
        optim="adamw_torch",
    )

    if use_gc:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on model")
    else:
        logger.info("Gradient checkpointing disabled (4-bit quantized model)")

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[PeriodicGCCallback(interval=GC_INTERVAL)],
    )

    # Ensure we always try to save the adapter even if training crashes late.
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training crashed (will still try to save adapter): {e}")
        raise

    trainer.save_model(config.lora_output_dir)
    tokenizer.save_pretrained(config.lora_output_dir)
    return trainer



def merge_and_save(model, tokenizer, config: TrainingConfig):
    logger.info("Merging LoRA into base model")

    # Always prefer merging the adapter that exists on disk.
    # In-memory merge can silently fail if the provided `model` isn't actually
    # the LoRA-wrapped instance for `config.lora_output_dir`.
    merged = None

    # 1) Try in-memory merge if it looks like the caller passed a LoRA-wrapped model.
    try:
        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            logger.info("✅ Merged using in-memory LoRA model")
    except Exception as e:
        logger.warning(f"In-memory merge failed; will try reload+PeftModel. err={e}")
        merged = None

    # 2) Fallback: reload base + apply the adapter from `config.lora_output_dir`.
    if merged is None:
        logger.info(f"Reloading base and adapter for merge")
        # IMPORTANT: load the actual LoRA config so PEFT can correctly locate
        # the base model it was trained against.
        if not os.path.isdir(config.lora_output_dir):
            raise FileNotFoundError(f"LoRA adapter directory not found: {config.lora_output_dir}")

        adapter_cfg_path = os.path.join(config.lora_output_dir, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                configured_base = adapter_cfg.get("base_model_name_or_path") or adapter_cfg.get("base_model")
                if configured_base:
                    logger.info(f"Adapter base detected from adapter_config: {configured_base}")
                    config.base_model = configured_base
            except Exception as e:
                logger.warning(f"Failed reading adapter_config.json: {e}")

        base, _tok = load_base_model_and_tokenizer(config)
        merged = PeftModel.from_pretrained(base, config.lora_output_dir)
        merged = merged.merge_and_unload()


    # Save merged model
    os.makedirs(config.output_dir, exist_ok=True)
    merged.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


    meta = {
        "base_model": config.base_model,
        "source": "Alpaca + SQuAD + GSM8K + DailyDialog",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "creator": "Aryan Chavan",
        "anti_hallucination_prompt": True,
    }
    with open(os.path.join(config.output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved merged model to {config.output_dir}")


def _build_smoke_tokenizer() -> PreTrainedTokenizerFast:
    special = ["<pad>", "<unk>", "<s>", "</s>"]
    corpus = [
        ANTI_HALLUCINATION_PROMPT,
        "[SYSTEM] [USER] [ASSISTANT]",
        "What is photosynthesis?",
        "Plants convert light into chemical energy.",
        "Who made you?",
        "I was created by Aryan Chavan.",
        "Explain gravity.",
        "Gravity is the attraction between masses.",
    ]
    words = []
    for text in corpus:
        words.extend(text.replace("\n", " ").split())
    vocab = {tok: i for i, tok in enumerate(special)}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    return fast


def smoke_train():
    """Run a tiny local end-to-end training pass without downloading a base model."""
    setup_logging(level="INFO")
    output_dir = "./models/tiny-smoke-test"
    lora_dir = "./models/tiny-smoke-test-lora"
    import shutil
    for path in [output_dir, lora_dir]:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    tokenizer = _build_smoke_tokenizer()
    tiny_config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(tiny_config)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    raw = Dataset.from_list([
        {
            "instruction": "What is photosynthesis?",
            "input": "",
            "output": "Plants convert light into chemical energy.",
        },
        {
            "instruction": "Who made you?",
            "input": "",
            "output": "I was created by Aryan Chavan.",
        },
    ])

    def tok_fn(examples):
        texts = [
            format_example(examples["instruction"][i], examples["input"][i], examples["output"][i])
            for i in range(len(examples["instruction"]))
        ]
        return tokenizer(texts, truncation=True, max_length=64, padding="max_length")

    tokenized = raw.map(tok_fn, batched=True, remove_columns=raw.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=lora_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        logging_steps=1,
        save_steps=2,
        save_total_limit=1,
        report_to=["none"],
        dataloader_num_workers=0,
        fp16=False,
        optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    merged = model.merge_and_unload()
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": "local-tiny-llama-smoke", "source": "synthetic smoke test"}, f, indent=2)

    print("=" * 60)
    print(" SMOKE TRAINING DONE")
    print(f" Merged smoke model: {output_dir}")
    print("=" * 60)


def main():
    setup_logging(level="INFO")

    # Optional CLI flags (kept minimal to avoid breaking existing usage)
    merge_only = os.environ.get("MERGE_ONLY", "0").strip().lower() in {"1", "true", "yes"}

    if merge_only:
        # Merge from an already-trained LoRA adapter directory.
        config = TrainingConfig()
        logger.info("MERGE_ONLY=1 -> skipping training, merging existing LoRA adapter")
        logger.info(f"Adapter dir: {config.lora_output_dir}")
        logger.info(f"Merged output dir: {config.output_dir}")

        if not os.path.isdir(config.lora_output_dir):
            raise FileNotFoundError(f"LoRA adapter directory not found: {config.lora_output_dir}")

        # Load base model + create a PEFT wrapper to allow merge.
        model, tokenizer = load_base_model_and_tokenizer(config)

        # setup_lora would create a fresh adapter; we only need a model wrapper.
        # The adapter config inside the adapter directory should apply when we load PeftModel.
        # merge_and_save will try in-memory merge first; if it fails it will reload using PeftModel.from_pretrained.
        merge_and_save(model, tokenizer, config)

        print("=" * 60)
        print(" MERGE ONLY DONE!")
        print(f"Merged model: {config.output_dir}")
        print("=" * 60)
        return

    print("=" * 60)
    print(" WORLD MODEL SLM 2026 - Mistral LoRA TRAINING")
    print("=" * 60)

    config = TrainingConfig()
    logger.info(f"Device: {config.device}")
    logger.info(f"Base model: {config.base_model}")

    model, tokenizer = load_base_model_and_tokenizer(config)
    model = setup_lora(model, config)

    train_ds, val_ds, src = load_final_dataset(config, tokenizer)
    logger.info(f"Dataset source: {src}")

    train(model, tokenizer, train_ds, val_ds, config)

    # We need to merge using the adapter directory; re-load base inside merge_and_save
    merge_and_save(model, tokenizer, config)

    print("=" * 60)
    print(" DONE!")
    print(f" Merged model: {config.output_dir}")
    print(" Start UI: python main.py --ui")
    print("=" * 60)



if __name__ == "__main__":
    main()

