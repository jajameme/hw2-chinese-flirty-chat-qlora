"""Split whole conversations before expanding assistant turns into SFT examples."""

import json
import random
from collections import defaultdict

from common import ROOT

SEED = 42
VALID_RATIO = 0.1
MAX_CONTEXT_MESSAGES = 6


def expand(conversations):
    examples = []
    for messages in conversations:
        for index in range(2, len(messages), 2):
            context = messages[max(1, index - MAX_CONTEXT_MESSAGES):index]
            examples.append({
                "prompt": [messages[0], *context],
                "completion": [messages[index]],
            })
    return examples


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main():
    source = ROOT / "data" / "chat.jsonl"
    conversations = [json.loads(line)["messages"] for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(conversations) < 10:
        raise ValueError("At least 10 conversations are needed for a meaningful validation split")
    # Two opening styles can share the same gag. Keep those variants together
    # so validation does not see an answer already present in training.
    grouped = defaultdict(list)
    for messages in conversations:
        answer_key = tuple(m["content"] for m in messages if m["role"] == "assistant")
        grouped[answer_key].append(messages)
    groups = list(grouped.values())
    random.Random(SEED).shuffle(groups)
    n_valid = max(1, round(len(groups) * VALID_RATIO))
    valid = [messages for group in groups[:n_valid] for messages in group]
    train = [messages for group in groups[n_valid:] for messages in group]
    out = ROOT / "data" / "sft"
    write_jsonl(out / "train.jsonl", expand(train))
    write_jsonl(out / "valid.jsonl", expand(valid))
    print(f"gag groups: train={len(groups) - n_valid}, valid={n_valid}")
    print(f"conversations: train={len(train)}, valid={len(valid)}")
    print(f"examples: train={len(expand(train))}, valid={len(expand(valid))}")


if __name__ == "__main__":
    main()
