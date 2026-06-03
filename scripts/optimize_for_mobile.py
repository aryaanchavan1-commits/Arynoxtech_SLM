"""Optimize trained models for small devices (laptop, mobile).
- INT4 & INT8 quantization via bitsandbytes / torch.quantization
- ONNX export for cross-platform inference
- Model pruning for size reduction
- Output: ./models/optimized/
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
os.environ.setdefault('HF_HOME', 'D:/.hf_cache')

import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

MODELS_TO_OPTIMIZE = [
    {
        "name": "anonyllm-360m",
        "path": "./models/anonyllm-360m-trained",
        "type": "transformers",
    },
]

OUTPUT_DIR = "./models/optimized"


def quantize_int8(model_path, output_path, model_type="transformers"):
    """Quantize model to INT8 using torch.quantization."""
    logger.info(f"Quantizing {model_path} to INT8...")
    os.makedirs(output_path, exist_ok=True)

    if model_type == "transformers":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            # Save INT8 via dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

            torch.save(quantized_model.state_dict(), os.path.join(output_path, "model_int8.pt"))
            tokenizer.save_pretrained(output_path)

            # Also save as safetensors for HF loading
            quantized_model.save_pretrained(output_path)

            original_size = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
            )
            quantized_size = sum(
                p.numel() * p.element_size()
                for p in quantized_model.parameters()
            )

            logger.info(f"  Original: {original_size/1e6:.1f} MB")
            logger.info(f"  INT8: {quantized_size/1e6:.1f} MB")
            logger.info(f"  Compression: {original_size/quantized_size:.1f}x")

            del model, quantized_model
            torch.cuda.empty_cache()
            gc.collect()

            return True

        except Exception as e:
            logger.warning(f"INT8 quantization failed: {e}")
            return False

    elif model_type == "custom_slm":
        # Custom SLM direct INT8 quantization
        try:
            from core.slm_architecture import SLMModel

            state = torch.load(os.path.join(model_path, "model_fp16.pt"), map_location="cpu")

            # Quantize each weight tensor to INT8
            quantized_state = {}
            for key, tensor in state.items():
                if "weight" in key and tensor.dim() >= 2:
                    # Per-channel quantization
                    scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 127.0
                    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
                    quantized_state[key] = quantized
                    quantized_state[f"{key}_scale"] = scale.to(torch.float16)
                else:
                    quantized_state[key] = tensor

            os.makedirs(output_path, exist_ok=True)
            torch.save(quantized_state, os.path.join(output_path, "model_int8.pt"))

            original_size = sum(t.numel() * t.element_size() for t in state.values())
            quantized_size = sum(t.numel() * t.element_size() for t in quantized_state.values())
            logger.info(f"  Original: {original_size/1e6:.1f} MB")
            logger.info(f"  INT8: {quantized_size/1e6:.1f} MB")
            logger.info(f"  Compression: {original_size/quantized_size:.1f}x")

            return True

        except Exception as e:
            logger.warning(f"Custom SLM INT8 quantization failed: {e}")
            return False

    return False


def export_to_onnx(model_path, output_path, model_type="transformers"):
    """Export model to ONNX format for cross-platform inference."""
    logger.info(f"Exporting {model_path} to ONNX...")
    os.makedirs(output_path, exist_ok=True)

    try:
        import torch.onnx

        if model_type == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            dummy_input = tokenizer("Hello", return_tensors="pt")
            input_ids = dummy_input["input_ids"].to(model.device)

            torch.onnx.export(
                model,
                (input_ids,),
                os.path.join(output_path, "model.onnx"),
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=17,
                do_constant_folding=True,
            )

            tokenizer.save_pretrained(output_path)
            logger.info(f"  ONNX model saved to {output_path}/model.onnx")

            del model
            torch.cuda.empty_cache()
            gc.collect()
            return True

    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        return False

    return False


def create_mobile_package(model_path, output_path, model_type="transformers"):
    """Create a lightweight package for mobile deployment."""
    logger.info(f"Creating mobile package for {model_path}...")
    os.makedirs(output_path, exist_ok=True)

    # Save metadata
    meta = {
        "model_name": os.path.basename(model_path),
        "model_type": model_type,
        "formats": [],
        "recommended_device": "mobile",
    }

    if os.path.exists(os.path.join(output_path, "model_int8.pt")):
        meta["formats"].append("int8_pt")
        size = os.path.getsize(os.path.join(output_path, "model_int8.pt"))
        meta["int8_size_mb"] = round(size / 1e6, 2)

    if os.path.exists(os.path.join(output_path, "model.onnx")):
        meta["formats"].append("onnx")
        size = os.path.getsize(os.path.join(output_path, "model.onnx"))
        meta["onnx_size_mb"] = round(size / 1e6, 2)

    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Mobile package created at {output_path}")
    return meta


def main():
    setup_logging(level="INFO")

    print("=" * 60)
    print("  MOBILE OPTIMIZATION - Quantization & Export")
    print("=" * 60)

    for model_info in MODELS_TO_OPTIMIZE:
        model_path = model_info["path"]
        model_name = model_info["name"]
        model_type = model_info["type"]

        if not os.path.exists(model_path):
            logger.warning(f"Model {model_path} not found, skipping")
            continue

        opt_path = os.path.join(OUTPUT_DIR, model_name)
        logger.info(f"\n--- Optimizing {model_name} ({model_type}) ---")

        quantize_int8(model_path, os.path.join(opt_path, "int8"), model_type)
        export_to_onnx(model_path, os.path.join(opt_path, "onnx"), model_type)
        create_mobile_package(model_path, opt_path)

        # Try INT4 quantization for extreme compression
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            quant_path = os.path.join(opt_path, "int4")
            os.makedirs(quant_path, exist_ok=True)

            if model_type == "transformers" and os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    ),
                    device_map="auto",
                    trust_remote_code=True,
                )
                model.save_pretrained(quant_path)
                tokenizer.save_pretrained(quant_path)
                logger.info(f"  4-bit quantized model saved to {quant_path}")
                del model
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.warning(f"4-bit quantization skipped: {e}")

    print("=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
