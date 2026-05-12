import os, sys
sys.modules['tensorflow']=None
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'; os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'; os.environ['TRANSFORMERS_NO_TF']='1'
os.environ['TRANSFORMERS_NO_FLAX']='1'; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['HF_HUB_DISABLE_SYMLINKS']='1'

import asyncio, json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",  # Smallest model
]

# Your trained production model (merged LoRA)
DEFAULT_MODEL_PATH = "./models/offline-trained-slm"
LORA_BASE_MODEL = "TinyLlama/TinyLlama-1.1B"

# Better default model for production (small, fits in memory)
PRODUCTION_FALLBACK = "HuggingFaceTB/SmolLM2-135M-Instruct"

def _detect_peft_adapter(model_path: str) -> bool:
    """Check if the path contains PEFT adapter files."""
    adapter_path = os.path.join(model_path, "adapter_config.json")
    return os.path.exists(adapter_path)


class ModelManager:
    def __init__(self, model_path: str = None, max_batch_size: int = 8,
                 max_sequence_length: int = 512, device: str = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.generation_config: Optional[Any] = None
        self._is_mock = False
        self._kv_cache = True
        self._chat_template = None

    async def load_model(self) -> None:
        self.load_model_sync()

    def load_model_sync(self) -> None:
        if self.model is not None:
            logger.info("Model already loaded")
            return
        logger.info(f"Loading model from {self.model_path}")
        loaded = False

        # Try to load the merged production model first
        merged_path = "./models/offline-trained-slm"
        if os.path.exists(merged_path):
            try:
                logger.info(f"Loading production merged model: {merged_path}")
                self._load_local(merged_path)
                loaded = True
                logger.info("✅ Production merged model loaded!")
                # Check if model is producing reasonable output
                if self._is_model_broken():
                    logger.warning("Model appears broken, trying fallback")
                    loaded = False
                    self.model = None
                    self.tokenizer = None
                    import gc
                    gc.collect()
            except Exception as e:
                logger.warning(f"Merged model load failed: {e}")
                self.model = None
                self.tokenizer = None

        # Fallback to LoRA
        lora_path = "./models/tinyllama-trained-slm-lora/checkpoint-500"
        alt_lora_path = "./scripts/models/tinyllama-trained-slm-lora/checkpoint-500"
        if not loaded:
            if os.path.exists(lora_path):
                chosen_lora_path = lora_path
            elif os.path.exists(alt_lora_path):
                chosen_lora_path = alt_lora_path
            else:
                chosen_lora_path = None

            if chosen_lora_path is not None:
                try:
                    logger.info(f"Trying LoRA checkpoint: {chosen_lora_path}")
                    self._load_local(chosen_lora_path)
                    loaded = True
                    logger.info("✅ Loaded YOUR TRAINED LoRA model!")
                    if self._is_model_broken():
                        logger.warning("LoRA model appears broken, trying fallback")
                        loaded = False
                        self.model = None
                        self.tokenizer = None
                        import gc
                        gc.collect()
                except Exception as e:
                    logger.warning(f"LoRA load failed: {e}")
                    self.model = None
                    self.tokenizer = None


        # Try specified path
        if not loaded and os.path.exists(self.model_path):
            try:
                self._load_local(self.model_path)
                loaded = True
                logger.info("Local model loaded")
                # Check if it's broken
                if self._is_model_broken():
                    logger.warning("Local model is broken, discarding")
                    loaded = False
                    self.model = None
                    self.tokenizer = None
                    import gc
                    gc.collect()
            except Exception as e:
                logger.warning(f"Local load failed: {e}")
                self.model = None
                self.tokenizer = None

        # Use fallback models (better quality)
        if not loaded:
            # Try production fallback first
            try:
                logger.info(f"Loading production fallback: {PRODUCTION_FALLBACK}")
                self._load_fallback(PRODUCTION_FALLBACK)
                loaded = True
                self.model_path = PRODUCTION_FALLBACK
                logger.info(f"Production fallback loaded: {PRODUCTION_FALLBACK}")
            except Exception as e:
                logger.warning(f"Production fallback failed: {e}")
                
            # Try other fallbacks
            for fallback in FALLBACK_MODELS:
                if fallback == PRODUCTION_FALLBACK:
                    continue
                try:
                    logger.info(f"Trying fallback: {fallback}")
                    self._load_fallback(fallback)
                    loaded = True
                    self.model_path = fallback
                    logger.info(f"Fallback loaded: {fallback}")
                    break
                except Exception as e:
                    logger.warning(f"Fallback {fallback} failed: {e}")

        if not loaded:
            logger.error("All real models failed. Loading mock as LAST RESORT.")
            self._load_mock()

        if not self._is_mock:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_sequence_length,
                do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
                repetition_penalty=1.1, pad_token_id=self.tokenizer.eos_token_id,
            )
            self.model.eval()

    def _is_model_broken(self) -> bool:
        """Quick check if model is producing garbage."""
        try:
            if not self.model or not self.tokenizer:
                return True
            test_input = self.tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(
                    **test_input, max_new_tokens=30, do_sample=False, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Check for repetitive garbage patterns
            unique_chars = len(set(decoded))
            total_chars = len(decoded)
            # If mostly punctuation or very few unique chars, it's broken
            if unique_chars < 10 or (total_chars > 20 and unique_chars / total_chars < 0.3):
                logger.warning(f"Model appears broken: decoded='{decoded[:100]}', unique={unique_chars}, total={total_chars}")
                return True
            # Check for repetitive patterns
            if decoded.count(':') > 5 or decoded.count('%') > 5:
                logger.warning(f"Model has repetitive punctuation: '{decoded[:100]}'")
                return True
            return False
        except Exception as e:
            logger.warning(f"Model check error: {e}")
            return True

    def _load_fallback(self, model_name: str):
        """Load a fallback model from HuggingFace."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device=="cuda" else torch.float32,
            device_map="auto" if self.device=="cuda" else None,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._chat_template = getattr(self.tokenizer, "chat_template", None)
        self._is_mock = False

    async def unload_model(self) -> None:
        logger.info("Unloading model")
        self.model = None; self.tokenizer = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    async def generate(self, prompt: str, max_tokens: int = 512,
                        temperature: float = 0.7, top_p: float = 0.9,
                        top_k: int = 50, do_sample: bool = True,
                        system_prompt: Optional[str] = None) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        formatted = self._build_prompt(prompt, system_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt",
                                truncation=True, max_length=min(self.max_sequence_length, 1024))
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=min(max_tokens, 512),
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                do_sample=do_sample, repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=self._kv_cache,
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    async def chat_completion(self, messages: List[Dict[str,str]],
                              max_tokens: int = 512, temperature: float = 0.7,
                              top_p: float = 0.9, top_k: int = 50,
                              do_sample: bool = True) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        if self._chat_template:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            parts = []
            for m in messages:
                if m["role"]=="system": parts.append(f"System: {m['content']}")
                elif m["role"]=="user": parts.append(f"User: {m['content']}")
                else: parts.append(f"Assistant: {m['content']}")
            parts.append("Assistant:")
            prompt = "\n".join(parts)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=self.max_sequence_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                do_sample=do_sample, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=self._kv_cache,
            )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    async def get_status(self) -> Dict[str, Any]:
        if not self.model:
            return {"status":"unloaded","model_path":self.model_path,"device":self.device,"loaded":False}
        status = {"status":"loaded","model_path":self.model_path,"device":self.device,
                  "max_batch_size":self.max_batch_size,"max_sequence_length":self.max_sequence_length,
                  "loaded":True,"is_mock":self._is_mock,"kv_cache":self._kv_cache,
                  "chat_template":self._chat_template is not None}
        if self.device=="cuda" and torch.cuda.is_available():
            status["gpu_total"]=torch.cuda.get_device_properties(0).total_memory
            status["gpu_allocated"]=torch.cuda.memory_allocated()
        return status

    def _load_local(self, model_path: str = None):
        if model_path is None:
            model_path = self.model_path
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        dtype = torch.float16 if self.device=="cuda" else torch.float32
        
        # Check for PEFT adapter - your LoRA model
        if _detect_peft_adapter(model_path):
            logger.info("PEFT adapter detected, loading with base model")
            from peft import PeftModel
            # Read base model from adapter config
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            base_model_name = LORA_BASE_MODEL  # TinyLlama
            try:
                with open(adapter_config_path, "r") as f:
                    adapter_cfg = json.load(f)
                if "base_model_name_or_path" in adapter_cfg:
                    base_model_name = adapter_cfg["base_model_name_or_path"]
            except Exception:
                pass
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=dtype,
                device_map="auto" if self.device=="cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model = self.model.merge_and_unload()
            logger.info("✅ LoRA merged and loaded successfully!")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._chat_template = getattr(self.tokenizer, "chat_template", None)
        if self.device=="cpu":
            self.model = self.model.to(self.device)

    def _load_mock(self):
        self._is_mock = True
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        config = GPT2Config(vocab_size=1000, n_positions=512, n_embd=256, n_layer=4, n_head=4)
        self.model = GPT2LMHeadModel(config)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.warning("Mock model loaded — responses will be gibberish!")

    def _build_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self._chat_template and system_prompt:
            msgs = [{"role":"system","content":system_prompt},{"role":"user","content":prompt}]
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        elif self._chat_template:
            return self.tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {prompt}\nAssistant:"
        return f"User: {prompt}\nAssistant:"
