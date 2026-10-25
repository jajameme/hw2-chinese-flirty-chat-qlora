"""Plot logged train and validation loss from a completed run."""

import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ADAPTER_DIR, LARGER_ADAPTER_DIR, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b")
    args = parser.parse_args()
    large = args.model_size == "1.5b"
    adapter_dir = LARGER_ADAPTER_DIR if large else ADAPTER_DIR
    metrics = json.loads((adapter_dir / "metrics.json").read_text(encoding="utf-8"))
    logs = metrics["log_history"]
    training = [(r["step"], r["loss"]) for r in logs if "loss" in r]
    validation = [(r["step"], r["eval_loss"]) for r in logs if "eval_loss" in r]
    if not training or not validation:
        raise ValueError("No training/validation loss logs found")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(*zip(*training), marker=".", label="Train loss")
    ax.plot(*zip(*validation), marker="o", label="Validation loss")
    ax.set(xlabel="Optimizer step", ylabel="Loss", title=f"Qwen2.5-{args.model_size.upper()} QLoRA learning curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = ROOT / "images" / ("loss_curve_15b.png" if large else "loss_curve.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
