"""Render the actual five-turn model transcript as a shareable PNG."""

import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import ROOT

WIDTH = 1200
PADDING = 72
MAX_BUBBLE = 990


def font(size):
    candidates = [Path("C:/Windows/Fonts/msjh.ttc"), Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Install Microsoft JhengHei or Noto Sans CJK to render Chinese text")
    return ImageFont.truetype(str(path), size)


def wrap(draw, value, use_font, max_width):
    lines = []
    line = ""
    for char in value:
        if char == "\n" or draw.textlength(line + char, font=use_font) > max_width:
            if line:
                lines.append(line)
            line = "" if char == "\n" else char
        else:
            line += char
    if line:
        lines.append(line)
    return lines or [""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b")
    args = parser.parse_args()
    suffix = "_15b" if args.model_size == "1.5b" else ""
    source = ROOT / "results" / f"comparison{suffix}.json"
    demo = json.loads(source.read_text(encoding="utf-8"))["demo"]
    if len(demo) < 5:
        raise ValueError("Expected a five-turn demo")
    title_font, body_font, small_font = font(39), font(30), font(22)
    scratch = ImageDraw.Draw(Image.new("RGB", (WIDTH, 100), "white"))
    blocks = []
    for turn in demo:
        for role, value in (("使用者", turn["user"]), ("QLoRA 助手", turn["assistant"])):
            lines = wrap(scratch, value, body_font, MAX_BUBBLE - 52)
            height = 34 + len(lines) * 47 + 20
            blocks.append((role, lines, height))
    total_height = 180 + sum(h + 22 for _, _, h in blocks) + 95
    image = Image.new("RGB", (WIDTH, total_height), "#f6f8fc")
    draw = ImageDraw.Draw(image)
    draw.text((PADDING, 48), "土味情話專家：五輪接梗展示", font=title_font, fill="#17233c")
    draw.text((PADDING, 105), f"Qwen2.5-{args.model_size.upper()}-Instruct + 4-bit QLoRA  |  固定提示與貪婪解碼", font=small_font, fill="#61708b")
    y = 170
    for role, lines, height in blocks:
        user = role == "使用者"
        x = WIDTH - PADDING - MAX_BUBBLE if user else PADDING
        color = "#dcecff" if user else "#e9f7ed"
        draw.rounded_rectangle((x, y, x + MAX_BUBBLE, y + height), radius=20, fill=color)
        draw.text((x + 26, y + 13), role, font=small_font, fill="#335275" if user else "#326b50")
        for i, line in enumerate(lines):
            draw.text((x + 26, y + 45 + i * 47), line, font=body_font, fill="#202b3d")
        y += height + 22
    draw.text((PADDING, total_height - 56), f"逐字稿：results/comparison{suffix}.md  |  由實際模型輸出排版，非介面截圖", font=small_font, fill="#66758b")
    out = ROOT / "images" / f"demo_chat{suffix}.png"
    out.parent.mkdir(exist_ok=True)
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
