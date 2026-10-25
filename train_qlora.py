"""Fine-tune Qwen2.5-0.5B-Instruct with NF4 QLoRA on 6 GiB CUDA."""

import argparse
import json
import math
import random
from pathlib import Path

from common import ADAPTER_DIR, LARGER_ADAPTER_DIR, LARGER_MODEL_ID, MODEL_ID, ROOT

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from model_utils import load_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Use a small positive value only for a smoke run")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    large = args.model_size == "1.5b"
    if args.batch_size is None:
        args.batch_size = 1 if large else 4
    if args.grad_accum is None:
        args.grad_accum = 8 if large else 4
    if large:
        args.gradient_checkpointing = True
    model_id = LARGER_MODEL_ID if large else MODEL_ID
    output_dir = LARGER_ADAPTER_DIR if large else ADAPTER_DIR
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        parser.error("epochs, batch-size and grad-accum must be positive")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_path = ROOT / "data" / "sft" / "train.jsonl"
    valid_path = ROOT / "data" / "sft" / "valid.jsonl"
    for path in (train_path, valid_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run prepare_chat_data.py")
    data = load_dataset("json", data_files={"train": str(train_path), "validation": str(valid_path)})
    model, tokenizer = load_base(model_id)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing,
    )
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    ))
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}")
    print(f"trainable dtypes: {set(str(p.dtype) for p in model.parameters() if p.requires_grad)}")
    steps_per_epoch = math.ceil(len(data["train"]) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs if args.max_steps < 0 else args.max_steps
    check_steps = 10 if total_steps < 40 else 15
    if total_steps < check_steps:
        check_steps = max(1, total_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(output_dir), max_length=args.max_length,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=max(1, round(total_steps * 0.05)),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=True, bf16=False, tf32=False, optim="paged_adamw_8bit",
        logging_steps=5, eval_strategy="steps", eval_steps=check_steps,
        save_strategy="steps", save_steps=check_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, completion_only_loss=True,
        restore_callback_states_from_checkpoint=True,
        eos_token="<|im_end|>", report_to="none", seed=42,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=data["train"],
        eval_dataset=data["validation"], processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    # TRL may recast PEFT weights while preparing the model. Keep trainable
    # adapters in FP32 for FP16 AMP on Turing GPUs (no native BF16).
    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
    actual_dtypes = {str(p.dtype) for p in trainer.model.parameters() if p.requires_grad}
    if actual_dtypes != {"torch.float32"}:
        raise RuntimeError(f"Unexpected trainable dtypes: {actual_dtypes}")
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = {
        "train": result.metrics,
        "eval": trainer.evaluate(),
        "best_checkpoint": str(Path(trainer.state.best_model_checkpoint).relative_to(ROOT)).replace("\\", "/")
        if trainer.state.best_model_checkpoint else None,
        "best_metric": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
        "trainable_parameters": trainable,
        "config": {**vars(args), "resume": Path(args.resume).name if args.resume else None},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps({"global_step": trainer.state.global_step,
                    "best_eval_loss": trainer.state.best_metric,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved adapter and metrics to {output_dir}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
