"""Check GPU capabilities and actually load/execute the NF4 base model."""

from common import SYSTEM_PROMPT, MODEL_ID, LARGER_MODEL_ID

import torch
import argparse
from model_utils import generate, load_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b")
    args = parser.parse_args()
    print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"total VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")
    model_id = LARGER_MODEL_ID if args.model_size == "1.5b" else MODEL_ID
    model, tokenizer = load_base(model_id)
    print(f"4-bit quantized: {getattr(model, 'is_loaded_in_4bit', False)}")
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Model did not load in 4-bit")
    reply = generate(model, tokenizer, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "用一句話向我說早安。"},
    ], sample=False, max_new_tokens=48)
    print(f"4-bit inference: {reply}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
