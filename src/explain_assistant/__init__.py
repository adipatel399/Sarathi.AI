import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="sarathiai", description="Explain a confusing Indian SMS/document in simple words.")
    ap.add_argument("text", nargs="?", help="message text (omit when using --image)")
    ap.add_argument("--sender", default="unknown", help="SMS sender, e.g. VM-HDFCBK or +91 98xxxxxx")
    ap.add_argument("--image", help="photo of a letter, bill or prescription")
    ap.add_argument("--lang", choices=["hi", "en"], default="hi")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (default: ./adapters)")
    ap.add_argument("--speak", action="store_true", help="read the answer aloud")
    args = ap.parse_args()

    from explain_assistant.assistant import build_reply
    from explain_assistant.model import DEFAULT_ADAPTER, Explainer

    if args.image:
        from explain_assistant.ocr import image_to_text

        text, sender = image_to_text(args.image), "photo of a document"
    elif args.text:
        text, sender = args.text, args.sender
    else:
        ap.error("give message text or --image")

    verdict = Explainer(adapter_path=args.adapter or DEFAULT_ADAPTER).explain(sender, text, args.lang)
    reply = build_reply(verdict, args.lang)
    print(json.dumps(asdict(verdict), ensure_ascii=False, indent=2) if verdict else "{}")
    print("\n" + reply.text)
    if args.speak:
        from explain_assistant.voice import synthesize

        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["afplay", str(synthesize(reply.speech, args.lang, Path(tmp) / "reply"))], check=False)
