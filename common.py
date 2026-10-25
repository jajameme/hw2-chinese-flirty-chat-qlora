"""Shared settings for the flirty multi-turn QLoRA assignment."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# All entry points share the project cache, including background/offline runs.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = ROOT / "outputs" / "flirty_lora_05b"
LARGER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LARGER_ADAPTER_DIR = ROOT / "outputs" / "flirty_lora_15b"
SYSTEM_PROMPT = (
    "你是一個土味情話高手，會用親切、溫和、帶點幽默的口吻，"
    "以繁體中文回答使用者，並且能接住使用者的追問繼續接梗。"
)
MAX_HISTORY_TURNS = 6


def get_converter():
    from opencc import OpenCC

    return OpenCC("s2twp")


def trim_messages(messages, max_turns=MAX_HISTORY_TURNS):
    """Keep the system instruction and recent complete turns, including the latest user."""
    return [messages[0], *messages[1:][-max_turns * 2 - 1 :]]
