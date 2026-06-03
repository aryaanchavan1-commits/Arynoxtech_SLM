"""Convert safetensors model to regular torch checkpoint"""
import struct, json, torch, os, sys
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
CACHE_DIR = Path("D:/.hf_cache/hub")

def find_model_path():
    model_dir = CACHE_DIR / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots"
    if not model_dir.exists():
        print("Model not cached. Download first.")
        return None
    snapshots = list(model_dir.glob("*"))
    if not snapshots:
        return None
    return snapshots[0] / "model.safetensors"

def convert():
    safetensors_path = find_model_path()
    if not safetensors_path or not safetensors_path.exists():
        print("Model not found")
        return False

    print(f"Converting {safetensors_path}...")
    file_size = os.path.getsize(safetensors_path)
    print(f"File size: {file_size/1e6:.1f} MB")

    with open(safetensors_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_data = f.read(header_size)
        # Remove trailing null bytes
        header_data = header_data.rstrip(b'\x00')
        header = json.loads(header_data.decode("utf-8"))
        tensor_data_offset = 8 + header_size

    print(f"Header has {len(header)} tensors")

    OUTPUT_DIR = Path("./models/smollm2-360m-checkpoint")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy config and tokenizer files
    snapshot_dir = safetensors_path.parent
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "chat_template.jinja"]:
        src = snapshot_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(OUTPUT_DIR / fname))

    # Build state dict from file
    state_dict = {}
    print("Reading tensors...")
    with open(safetensors_path, "rb") as f:
        f.seek(tensor_data_offset)
        data = f.read()

    for name, info in header.items():
        start, end = info["data_offsets"]
        dtype_str = info.get("dtype", "F32")
        shape = info.get("shape", [])

        dtype_map = {"F16": torch.float16, "F32": torch.float32, "BF16": torch.bfloat16,
                     "I64": torch.int64, "I32": torch.int32, "I8": torch.int8}
        dtype = dtype_map.get(dtype_str, torch.float32)
        tensor = torch.frombuffer(data[start:end], dtype=dtype).reshape(shape)
        state_dict[name] = tensor

    print(f"Loaded {len(state_dict)} tensors, saving...")
    torch.save(state_dict, str(OUTPUT_DIR / "pytorch_model.bin"))
    print(f"Saved pytorch_model.bin to {OUTPUT_DIR} (size: {os.path.getsize(str(OUTPUT_DIR / 'pytorch_model.bin'))/1e6:.1f} MB)")
    return True

if __name__ == "__main__":
    convert()
