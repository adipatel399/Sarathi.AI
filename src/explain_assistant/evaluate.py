import argparse
import json
import random
import re
from math import sqrt
from time import perf_counter
from collections import Counter, defaultdict
from pathlib import Path

from explain_assistant.model import Explainer, parse_verdict
from explain_assistant.prompt import build_messages
from explain_assistant.safety import apply_safety_net

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _from_simple(item: dict) -> dict:
    label = {
        "doc_type": item["doc_type"], "verdict": item["verdict"], "is_dangerous": item["verdict"] == "scam",
        "urgency": "", "red_flags": [], "explanation": "", "what_to_do": "",
    }
    messages = build_messages(item["sender"], item["text"], item["lang"])
    messages.append({"role": "assistant", "content": json.dumps(label, ensure_ascii=False)})
    meta = {"case": f"{item['verdict']}", "reply_language": item["lang"], "sender": item["sender"], "text": item["text"]}
    return {"messages": messages, "meta": meta}


def load_rows(path: Path, limit: int | None, seed: int = 0) -> list[dict]:
    if path.suffix == ".json":
        rows = [_from_simple(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit and limit < len(rows):
        by_case = defaultdict(list)
        for row in rows:
            by_case[row["meta"]["case"]].append(row)
        rng = random.Random(seed)
        picked, cases = [], sorted(by_case)
        while len(picked) < limit:
            for case in cases:
                if by_case[case] and len(picked) < limit:
                    picked.append(by_case[case].pop(rng.randrange(len(by_case[case]))))
        rows = picked
    return rows


def _wilson(successes: int, total: int) -> list[float] | None:
    """95% Wilson interval, which remains meaningful for small and perfect samples."""
    if not total:
        return None
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [round(100 * max(0, centre - margin), 1), round(100 * min(1, centre + margin), 1)]


def score(rows: list[dict], raw_outputs: list[str], use_safety_net: bool = False) -> dict:
    total = len(rows)
    counts = Counter()
    per_case = defaultdict(lambda: [0, 0])
    mistakes = []
    for row, raw in zip(rows, raw_outputs):
        gold = json.loads(row["messages"][-1]["content"])
        pred = parse_verdict(raw)
        if pred is not None and use_safety_net:
            pred = apply_safety_net(
                pred, row["meta"]["sender"], row["meta"]["text"], row["meta"]["reply_language"]
            )
        case = row["meta"]["case"]
        per_case[case][1] += 1
        if pred is None:
            counts["invalid_json"] += 1
            mistakes.append({"case": case, "text": row["meta"]["text"][:160], "raw": raw[:200]})
            if gold["is_dangerous"]:
                counts["dangerous_total"] += 1
            if gold["verdict"] == "safe":
                counts["safe_total"] += 1
            continue
        counts["valid_json"] += 1
        verdict_ok = pred.verdict == gold["verdict"]
        counts["verdict_correct"] += verdict_ok
        per_case[case][0] += verdict_ok
        counts["doc_type_correct"] += pred.doc_type == gold["doc_type"]
        if gold["is_dangerous"]:
            counts["dangerous_total"] += 1
            counts["dangerous_caught"] += pred.is_dangerous
        if gold["verdict"] == "safe":
            counts["safe_total"] += 1
            counts["safe_flagged_as_scam"] += pred.verdict == "scam" or pred.is_dangerous
        wants_hindi = row["meta"]["reply_language"] == "hi"
        has_hindi = bool(DEVANAGARI.search(pred.explanation + pred.what_to_do))
        counts["language_correct"] += wants_hindi == has_hindi
        if not verdict_ok:
            mistakes.append({"case": case, "gold": gold["verdict"], "pred": pred.verdict, "text": row["meta"]["text"][:160]})

    pct = lambda n, d: round(100 * n / d, 1) if d else None
    return {
        "examples": total,
        "valid_json_pct": pct(counts["valid_json"], total),
        "verdict_accuracy_pct": pct(counts["verdict_correct"], total),
        "verdict_accuracy_95ci_pct": _wilson(counts["verdict_correct"], total),
        "doc_type_accuracy_pct": pct(counts["doc_type_correct"], total),
        "dangerous_recall_pct": pct(counts["dangerous_caught"], counts["dangerous_total"]),
        "dangerous_recall_95ci_pct": _wilson(counts["dangerous_caught"], counts["dangerous_total"]),
        "safe_messages_flagged_as_scam": counts["safe_flagged_as_scam"],
        "safe_messages_total": counts["safe_total"],
        "reply_language_correct_pct": pct(counts["language_correct"], total),
        "per_case_verdict_accuracy_pct": {c: pct(ok, n) for c, (ok, n) in sorted(per_case.items())},
        "mistakes": mistakes[:25],
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate Sarathi.AI on the held-out test split.")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--adapter", default="adapters", help="LoRA adapter dir; pass 'none' for the base model")
    ap.add_argument("--limit", type=int, default=None, help="stratified subset size")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.data), args.limit)
    explainer = Explainer(adapter_path=None if args.adapter == "none" else args.adapter)
    items = [(r["meta"]["sender"], r["meta"]["text"], r["meta"]["reply_language"]) for r in rows]
    start = perf_counter()
    raw = explainer.generate_many(items, batch_size=args.batch_size)
    report = score(rows, raw)
    report["with_safety_net"] = score(rows, raw, use_safety_net=True)
    report["model"] = "base" if args.adapter == "none" else f"lora:{args.adapter}"
    report["seconds_per_example"] = round((perf_counter() - start) / len(rows), 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("mistakes", "per_case_verdict_accuracy_pct", "with_safety_net")}
    summary["with_safety_net"] = {
        k: v for k, v in report["with_safety_net"].items() if k not in ("mistakes", "per_case_verdict_accuracy_pct")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
