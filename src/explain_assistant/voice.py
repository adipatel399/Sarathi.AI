"""Offline text-to-speech with macOS system voices, encoded as Telegram-ready OGG/Opus."""

import shutil
import subprocess
import tempfile
from pathlib import Path

VOICES = {"hi": "Lekha", "en": "Rishi"}


def synthesize(text: str, language: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    voice = VOICES.get(language, VOICES["en"])
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "speech.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        if shutil.which("ffmpeg"):
            ogg = out_path.with_suffix(".ogg")
            result = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-c:a", "libopus", "-b:a", "32k", str(ogg)],
                capture_output=True,
            )
            if result.returncode == 0:
                return ogg
        m4a = out_path.with_suffix(".m4a")
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(aiff), str(m4a)], check=True)
    return m4a
