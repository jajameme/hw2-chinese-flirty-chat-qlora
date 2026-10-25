"""Resume training and evaluation in a detached process with an atomic state.json."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from common import ADAPTER_DIR, LARGER_ADAPTER_DIR, ROOT


STATE_FILE = ROOT / "state.json"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "background_train.log"


@contextmanager
def runner_lock():
    """OS-managed lock: released automatically even if the runner crashes."""
    lock_path = ROOT / ".background.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise SystemExit("A background runner is already active")
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise SystemExit("A background runner is already active")
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_state(state):
    state["updated_at"] = timestamp()
    temporary = STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def tail_log(limit=5000, start=0):
    if not LOG_FILE.exists():
        return ""
    with LOG_FILE.open("rb") as file:
        file.seek(0, 2)
        file.seek(max(start, file.tell() - limit))
        return file.read().decode("utf-8", errors="replace")


def newest_checkpoint(output_dir):
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        required = ("trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt")
        if path.is_dir() and all((path / name).is_file() for name in required):
            try:
                candidates.append((int(path.name.split("-")[-1]), path))
            except ValueError:
                pass
    return max(candidates, default=(0, None))


def refresh_progress(state, output_dir):
    checkpoint_step, checkpoint = newest_checkpoint(output_dir)
    state["checkpoint_step"] = checkpoint_step
    if checkpoint is not None:
        try:
            trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
            state["best_eval_loss"] = trainer_state.get("best_metric")
        except (OSError, json.JSONDecodeError):
            pass
    matches = re.findall(rf"(\d+)/{state['total_steps']}\s*\[", tail_log(18000, state.get("log_start_offset", 0)))
    if matches:
        state["current_step"] = max(state.get("current_step", 0), max(map(int, matches)))
    else:
        state["current_step"] = max(state.get("current_step", 0), checkpoint_step)


def run_stage(state, name, args, output_dir, env, log):
    state["stage"] = name
    write_state(state)
    log.write(f"\n[{timestamp()}] START {name}: {' '.join(args)}\n".encode("utf-8"))
    log.flush()
    process = subprocess.Popen(
        args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    state["child_pid"] = process.pid
    write_state(state)
    while process.poll() is None:
        if name == "training":
            refresh_progress(state, output_dir)
        write_state(state)
        time.sleep(5)
    state["exit_code"] = process.returncode
    state["child_pid"] = None
    if name == "training":
        refresh_progress(state, output_dir)
    write_state(state)
    log.write(f"\n[{timestamp()}] END {name}: exit {process.returncode}\n".encode("utf-8"))
    log.flush()
    if process.returncode:
        raise RuntimeError(f"{name} exited with code {process.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="1.5b")
    args = parser.parse_args()
    large = args.model_size == "1.5b"
    output_dir = LARGER_ADAPTER_DIR if large else ADAPTER_DIR
    checkpoint_step, checkpoint = newest_checkpoint(output_dir)
    LOG_DIR.mkdir(exist_ok=True)
    suffix = "_15b" if large else ""
    state = {
        "status": "running", "stage": "starting", "model_size": args.model_size,
        "started_at": timestamp(), "updated_at": timestamp(),
        "current_step": checkpoint_step, "checkpoint_step": checkpoint_step,
        "total_steps": 231 if large else 117, "best_eval_loss": None,
        "resume_checkpoint": str(checkpoint) if checkpoint else None,
        "runner_pid": os.getpid(), "child_pid": None, "exit_code": None,
        "log": str(LOG_FILE.relative_to(ROOT)),
        "artifacts": {},
    }
    write_state(state)
    log_start = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0
    state["log_start_offset"] = log_start
    env = os.environ.copy()
    env["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    try:
        with LOG_FILE.open("ab") as log:
            train = [sys.executable, "train_qlora.py", "--model-size", args.model_size]
            if checkpoint:
                train += ["--resume", str(checkpoint)]
            run_stage(state, "training", train, output_dir, env, log)
            for stage, script in (("plot", "plot_loss.py"),
                                  ("comparison", "compare_base.py"),
                                  ("demo_image", "render_demo.py")):
                run_stage(state, stage, [sys.executable, script, "--model-size", args.model_size],
                          output_dir, env, log)
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        final_state = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
        state["status"] = "completed"
        state["stage"] = "completed"
        state["current_step"] = final_state["global_step"]
        state["best_eval_loss"] = metrics["best_metric"]
        state["artifacts"] = {
            "adapter": str((output_dir / "adapter_model.safetensors").relative_to(ROOT)),
            "metrics": str((output_dir / "metrics.json").relative_to(ROOT)),
            "loss_curve": f"images/loss_curve{suffix}.png",
            "comparison": f"results/comparison{suffix}.md",
            "demo_image": f"images/demo_chat{suffix}.png",
        }
        for artifact in state["artifacts"].values():
            if not (ROOT / artifact).is_file():
                raise FileNotFoundError(f"Expected artifact missing: {artifact}")
        record = (
            f"\n\n<!-- background-run-{state['started_at']} -->\n"
            f"背景流程於 {timestamp()} 完成 {args.model_size} 訓練與產圖。"
            f"實際 optimizer steps：{state['current_step']}；最低 validation loss：{state['best_eval_loss']:.4f}。"
            f"對照逐字稿：[comparison{suffix}.md](../results/comparison{suffix}.md)，"
            f"loss 圖：[loss_curve{suffix}.png](../images/loss_curve{suffix}.png)。"
            "此段由背景程序自動記錄；回答品質仍需檢視逐字稿。\n"
        )
        docs_dir = ROOT / "docs"
        docs_dir.mkdir(exist_ok=True)
        with (docs_dir / "作業二.md").open("a", encoding="utf-8") as doc:
            doc.write(record)
        state["completed_at"] = timestamp()
    except Exception as error:
        state["status"] = "failed"
        state["stage"] = "failed"
        state["error"] = str(error)
        state["log_tail"] = tail_log()
        state["completed_at"] = timestamp()
    finally:
        state["child_pid"] = None
        write_state(state)


if __name__ == "__main__":
    with runner_lock():
        main()
