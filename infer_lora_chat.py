"""Interactive multi-turn chat with the trained adapter."""

import argparse

from common import ADAPTER_DIR, LARGER_ADAPTER_DIR, LARGER_MODEL_ID, MODEL_ID, SYSTEM_PROMPT

from peft import PeftModel
from model_utils import generate, load_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=6, help="Maximum remembered turns")
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b")
    args = parser.parse_args()
    if args.turns < 1:
        parser.error("--turns must be positive")
    large = args.model_size == "1.5b"
    adapter_dir = LARGER_ADAPTER_DIR if large else ADAPTER_DIR
    model_id = LARGER_MODEL_ID if large else MODEL_ID
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter missing: {adapter_dir}; run train_qlora.py")
    base, tokenizer = load_base(model_id)
    model = PeftModel.from_pretrained(base, adapter_dir)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("土味情話聊天已開始。/reset 清除對話，/history 查看歷史，/quit 結束。")
    while True:
        try:
            user = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"/quit", "exit", "quit", "q"}:
            break
        if user == "/reset":
            messages = messages[:1]
            print("已清除對話歷史。")
            continue
        if user == "/history":
            for message in messages[1:]:
                print(f"{message['role']}：{message['content']}")
            continue
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        reply = generate(model, tokenizer, messages, max_turns=args.turns)
        print(f"助手：{reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
