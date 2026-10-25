"""Use the same five two-turn prompts for base and LoRA model."""

import json
import argparse

from common import (ADAPTER_DIR, LARGER_ADAPTER_DIR, LARGER_MODEL_ID,
                    MODEL_ID, ROOT, SYSTEM_PROMPT)


PROMPTS = [
    ("我和喜歡的人在天文館，幫我想一句不太油的情話。", "如果她說今天星星都看不清楚，怎麼接？"),
    ("朋友今天面試失利，想用一點幽默安慰她。", "她說現在完全笑不出來，接下來怎麼說？"),
    ("我想邀請對方週末一起逛花市，給我一句輕鬆的邀請。", "如果她說這週末要加班呢？"),
    ("我忘記回訊息讓對方不開心，怎麼道歉？", "她說你每次都說會改，怎麼回？"),
    ("遠距離的伴侶今天很忙，我想說一句暖心的。", "對方說可能三天都沒空視訊，怎麼接？"),
]

DEMO = [
    "我在夜市看到一盞小燈，突然想對喜歡的人說句情話。",
    "她說小燈一下就沒電了，我怎麼接？",
    "她又問，如果下雨了呢？",
    "她說你是不是每件事都能接成情話？",
    "最後她想知道我們週末能做什麼。",
]


def run_pair(model, tokenizer, first, followup):
    from model_utils import generate
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": first})
    answer1 = generate(model, tokenizer, messages, sample=False, max_new_tokens=96)
    messages.append({"role": "assistant", "content": answer1})
    messages.append({"role": "user", "content": followup})
    answer2 = generate(model, tokenizer, messages, sample=False, max_new_tokens=96)
    return answer1, answer2


def markdown_cell(value):
    return value.replace("|", "｜").replace("\r\n", "\n").replace("\n", "<br>")


def write_report(rows, demo, out, model_id, suffix):
    lines = [f"# {model_id} 原始模型與 QLoRA 比較", "", "固定相同 system prompt，採貪婪解碼；第二輪各自帶入第一輪輸出。", ""]
    for i, row in enumerate(rows, 1):
        lines += [f"## 題目 {i}", "", f"**開場：**{row['first']}", f"**追問：**{row['followup']}", "",
                  "| 模型 | 第一輪 | 追問回覆 |", "| --- | --- | --- |",
                  f"| 原始 | {markdown_cell(row['base'][0])} | {markdown_cell(row['base'][1])} |",
                  f"| QLoRA | {markdown_cell(row['lora'][0])} | {markdown_cell(row['lora'][1])} |", ""]
    lines += ["## 五輪展示", ""]
    for turn in demo:
        lines += [f"- 使用者：{turn['user']}", f"- 助手：{turn['assistant']}", ""]
    (out / f"comparison{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / f"comparison{suffix}.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b")
    parser.add_argument("--render-only", action="store_true", help="Regenerate Markdown from existing JSON without GPU")
    args = parser.parse_args()
    large = args.model_size == "1.5b"
    adapter_dir = LARGER_ADAPTER_DIR if large else ADAPTER_DIR
    model_id = LARGER_MODEL_ID if large else MODEL_ID
    suffix = "_15b" if large else ""
    out = ROOT / "results"
    if args.render_only:
        payload = json.loads((out / f"comparison{suffix}.json").read_text(encoding="utf-8"))
        write_report(payload["pairs"], payload["demo"], out, model_id, suffix)
        return
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter missing: {adapter_dir}")
    from peft import PeftModel
    from model_utils import generate, load_base
    base, tokenizer = load_base(model_id)
    rows = [{"first": first, "followup": followup,
             "base": run_pair(base, tokenizer, first, followup)}
            for first, followup in PROMPTS]
    tuned = PeftModel.from_pretrained(base, adapter_dir)
    for row in rows:
        row["lora"] = run_pair(tuned, tokenizer, row["first"], row["followup"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    demo = []
    for user in DEMO:
        messages.append({"role": "user", "content": user})
        reply = generate(tuned, tokenizer, messages, sample=False, max_new_tokens=96)
        messages.append({"role": "assistant", "content": reply})
        demo.append({"user": user, "assistant": reply})
    out.mkdir(exist_ok=True)
    (out / f"comparison{suffix}.json").write_text(
        json.dumps({"pairs": rows, "demo": demo}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(rows, demo, out, model_id, suffix)


if __name__ == "__main__":
    main()
