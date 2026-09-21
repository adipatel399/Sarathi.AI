"""Audit lexical overlap between training and evaluation sets."""

import argparse
import json
import re
from pathlib import Path

_URL = re.compile(r"(?:https?://|www\.)\S+|\S+\.(?:com|in|top|xyz|online|info)\S*", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
_SPACE = re.compile(r"\s+")


def _load(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text(row: dict) -> str:
    if "text" in row:
        return row["text"]
    return row["meta"]["text"]


def normalize(text: str) -> str:
    text = _URL.sub(" <url> ", text.lower())
    text = _NUMBER.sub(" <number> ", text)
    return _SPACE.sub(" ", text).strip()


def token_similarity(left: str, right: str) -> float:
    a, b = set(normalize(left).split()), set(normalize(right).split())
    return len(a & b) / len(a | b) if a or b else 1.0


def audit(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    train_texts = [_text(row) for row in train_rows]
    train_normalized = {normalize(text) for text in train_texts}
    similarities = [max(token_similarity(_text(row), candidate) for candidate in train_texts) for row in eval_rows]
    ordered = sorted(similarities)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "evaluation_examples": len(eval_rows),
        "exact_normalized_duplicates": sum(normalize(_text(row)) in train_normalized for row in eval_rows),
        "nearest_train_token_similarity_median": round(percentile(0.5), 3),
        "nearest_train_token_similarity_p95": round(percentile(0.95), 3),
        "examples_above_0_8_similarity": sum(value >= 0.8 for value in similarities),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure train/evaluation lexical overlap.")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--eval", required=True)
    args = ap.parse_args()
    print(json.dumps(audit(_load(Path(args.train)), _load(Path(args.eval))), indent=2))


if __name__ == "__main__":
    main()
