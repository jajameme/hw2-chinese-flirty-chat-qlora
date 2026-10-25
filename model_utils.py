"""One 4-bit model setup shared by GPU check, training and inference."""

from common import MODEL_ID, get_converter, trim_messages

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_base(model_id=MODEL_ID):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; use a CUDA-enabled PyTorch installation")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config(), dtype=torch.float16,
        device_map={"": 0},
    )
    return model, tokenizer


def generate(model, tokenizer, messages, *, sample=True, max_new_tokens=128, max_turns=6):
    model.eval()
    inputs = tokenizer.apply_chat_template(
        trim_messages(messages, max_turns=max_turns), add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        result = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=sample,
            **({"temperature": 0.8, "top_p": 0.9} if sample else {}),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    new_tokens = result[0, inputs["input_ids"].shape[-1]:]
    return get_converter().convert(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
