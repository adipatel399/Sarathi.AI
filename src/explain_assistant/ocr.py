"""On-device OCR for photos of letters, bills and prescriptions (Apple Vision via ocrmac)."""

from pathlib import Path


def image_to_text(path: str | Path) -> str:
    try:
        from ocrmac import ocrmac
    except ImportError as exc:
        raise RuntimeError("Photo reading needs the 'ocr' extra: uv sync --extra ocr") from exc

    annotations = ocrmac.OCR(str(path), recognition_level="accurate", language_preference=["en-US"]).recognize()
    # Vision returns boxes with a bottom-left origin; sort top-to-bottom, then left-to-right.
    lines = sorted(annotations, key=lambda a: (-round(a[2][1] + a[2][3], 2), a[2][0]))
    return "\n".join(text for text, confidence, _ in lines if confidence >= 0.3)
