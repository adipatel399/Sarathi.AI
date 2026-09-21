"""Turns a forwarded message into the reply an elderly user sees, plus a family alert when needed."""

import re
from dataclasses import dataclass

from explain_assistant.model import Verdict

_SENDER_LINE = re.compile(r"^\s*(?:from|sender|भेजने वाला)\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE)

HEADINGS = {
    "hi": {"scam": "⚠️ सावधान: यह धोखा (स्कैम) है", "suspicious": "🤔 इस संदेश पर शक है", "safe": "✅ यह संदेश सुरक्षित लगता है",
           "todo": "क्या करें", "error": "माफ़ कीजिए, मैं यह संदेश समझ नहीं पाया। कृपया दोबारा भेजें या परिवार से पूछें।"},
    "en": {"scam": "⚠️ Warning: this is a scam", "suspicious": "🤔 This message looks suspicious", "safe": "✅ This message looks safe",
           "todo": "What to do", "error": "Sorry, I couldn't understand this message. Please send it again or ask your family."},
}

FLAG_LABELS = {
    "hi": "क्यों सावधान रहें",
    "en": "Why to be careful",
}

FLAG_NAMES_HI = {
    "personal_number_impersonation": "संस्था के नाम से निजी नंबर",
    "credential_request": "OTP या गुप्त जानकारी की मांग",
    "urgent_external_action": "जल्दबाज़ी में लिंक, कॉल या भुगतान का दबाव",
}


@dataclass
class Reply:
    text: str
    speech: str
    alert_family: bool


def split_sender(message: str) -> tuple[str, str]:
    """A forwarded SMS may start with 'From: VM-HDFCBK'. Returns (sender, body)."""
    lines = message.strip().splitlines()
    if lines:
        match = _SENDER_LINE.match(lines[0])
        if match and len(lines) > 1:
            return match.group(1), "\n".join(lines[1:]).strip()
    return "unknown", message.strip()


def build_reply(verdict: Verdict | None, language: str) -> Reply:
    h = HEADINGS.get(language, HEADINGS["en"])
    if verdict is None:
        return Reply(text=h["error"], speech=h["error"], alert_family=False)
    heading = h.get(verdict.verdict, h["suspicious"])
    flags = ""
    if verdict.verdict != "safe" and verdict.red_flags:
        readable = ", ".join(
            FLAG_NAMES_HI.get(flag, flag.replace("_", " ")) if language == "hi" else flag.replace("_", " ")
            for flag in verdict.red_flags[:3]
        )
        flags = f"\n\n🔎 {FLAG_LABELS.get(language, FLAG_LABELS['en'])}: {readable}"
    text = f"{heading}\n\n{verdict.explanation}{flags}\n\n👉 {h['todo']}: {verdict.what_to_do}"
    speech = f"{heading.lstrip('⚠️🤔✅ ')}. {verdict.explanation} {verdict.what_to_do}"
    alert = verdict.is_dangerous or (verdict.verdict == "scam" and verdict.urgency == "high")
    return Reply(text=text, speech=speech, alert_family=alert)


def family_alert(verdict: Verdict, sender: str, message: str, parent_name: str) -> str:
    flags = ", ".join(verdict.red_flags) or "—"
    snippet = message if len(message) <= 300 else message[:300] + "…"
    return (
        f"🚨 {parent_name} just received a likely scam ({verdict.doc_type}, urgency {verdict.urgency}).\n"
        f"Sender: {sender}\nRed flags: {flags}\n\nMessage:\n{snippet}\n\n"
        "They were told not to click, pay or share any OTP. A quick call to check on them would help."
    )
