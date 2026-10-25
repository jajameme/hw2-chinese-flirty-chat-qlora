"""Validate raw conversations and produce one canonical JSONL corpus."""

import json
import re
import csv
import argparse
from collections import Counter

from common import ROOT, SYSTEM_PROMPT, get_converter


def validate(messages, source):
    if not isinstance(messages, list) or len(messages) < 4 or len(messages) % 2:
        raise ValueError(f"{source}: expected at least two user/assistant turns")
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError(f"{source}: message {index} must be {expected}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{source}: message {index} has empty content")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-only", action="store_true",
                        help="Include only seed IDs marked yes in data/review.csv")
    args = parser.parse_args()
    approved = None
    if args.approved_only:
        with (ROOT / "data" / "review.csv").open(encoding="utf-8-sig", newline="") as file:
            reviews = list(csv.DictReader(file))
        valid = {"", "yes", "no"}
        if any(row["approved"].strip().lower() not in valid for row in reviews):
            raise ValueError("review.csv approved column must be blank, yes or no")
        approved = {row["seed_id"] for row in reviews if row["approved"].strip().lower() == "yes"}
        if not approved:
            raise ValueError("No approved seeds in review.csv")
    raw_files = sorted((ROOT / "raw").glob("*.json"))
    if not raw_files:
        raise FileNotFoundError("raw/*.json is empty")
    convert = get_converter().convert
    seen = set()
    rows = []
    scenarios = Counter()
    duplicates = 0
    for path in raw_files:
        batch = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(batch, list):
            raise ValueError(f"{path}: top-level value must be a list")
        for number, item in enumerate(batch, 1):
            source = f"{path.name}#{number}"
            if approved is not None and item.get("seed_id") not in approved:
                continue
            if not isinstance(item, dict) or not isinstance(item.get("scenario"), str):
                raise ValueError(f"{source}: missing scenario")
            messages = item.get("messages")
            validate(messages, source)
            messages = [
                {"role": m["role"], "content": convert(m["content"].strip())}
                for m in messages
            ]
            first = re.sub(r"\s+", "", messages[0]["content"])
            if first in seen:
                duplicates += 1
                continue
            seen.add(first)
            scenarios[item["scenario"]] += 1
            rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages]})
    out = ROOT / "data" / "chat.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"files": len(raw_files), "conversations": len(rows),
                      "duplicate_first_messages": duplicates, "scenarios": scenarios},
                     ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
