"""Small, deterministic safety net for high-confidence scam patterns.

The model remains the primary classifier.  These rules only upgrade a verdict when
multiple, independently suspicious signals appear together; they never downgrade it.
"""

import re
from dataclasses import replace

from explain_assistant.model import Verdict

_PERSONAL_SENDER = re.compile(r"(?:\+?91[\s-]?)?[6-9]\d{9}$")
_CREDENTIAL = re.compile(r"\b(?:otp|upi\s*pin|cvv|password|mpin)\b", re.IGNORECASE)
_PAYMENT = re.compile(r"\b(?:pay|payment|fee|transfer|send\s+money|upi|रुपये|भुगतान|पैसे)\b", re.IGNORECASE)
_URGENCY = re.compile(
    r"\b(?:immediately|urgent|today|tonight|within\s+\d+\s*(?:min|hour)|blocked?|"
    r"disconnect(?:ed|ion)?|arrest(?:ed)?|तुरंत|आज|बंद|गिरफ्तार)\b",
    re.IGNORECASE,
)
_OFF_PLATFORM = re.compile(r"(?:https?://|www\.|\.apk\b|\bcall\b|\bwhatsapp\b|लिंक|फोन)", re.IGNORECASE)
_IMPERSONATION = re.compile(
    r"\b(?:bank|police|cbi|customs|electricity|power|kyc|aadhaar|pension|बैंक|पुलिस|बिजली)\b",
    re.IGNORECASE,
)


def _looks_personal(sender: str) -> bool:
    return bool(_PERSONAL_SENDER.fullmatch(re.sub(r"[\s()-]", "", sender.strip())))


def apply_safety_net(verdict: Verdict, sender: str, text: str, language: str) -> Verdict:
    """Upgrade likely false negatives using a conservative combination of signals."""
    signals: list[str] = []
    personal_impersonation = _looks_personal(sender) and bool(_IMPERSONATION.search(text))
    credential_request = bool(_CREDENTIAL.search(text) and _OFF_PLATFORM.search(text))
    coercive_action = bool(_URGENCY.search(text) and (_PAYMENT.search(text) or _OFF_PLATFORM.search(text)))

    if personal_impersonation:
        signals.append("personal_number_impersonation")
    if credential_request:
        signals.append("credential_request")
    if coercive_action:
        signals.append("urgent_external_action")

    # Require either a credential request, or two independent contextual signals.
    if verdict.verdict == "safe" and (credential_request or len(signals) >= 2):
        if language == "hi":
            explanation = "यह संदेश किसी संस्था का नाम लेकर निजी नंबर, जल्दबाज़ी या बाहरी लिंक/कॉल का दबाव डाल रहा है। यह धोखा हो सकता है।"
            action = "कोई लिंक न खोलें, पैसे या OTP न दें। संस्था के आधिकारिक नंबर पर खुद कॉल करें और परिवार को बताएं।"
        else:
            explanation = "This message uses an organisation's name with a personal number, urgency, or an external link/call. It may be a scam."
            action = "Do not click, pay, or share an OTP. Contact the organisation using its official number and tell your family."
        return replace(
            verdict,
            verdict="scam",
            is_dangerous=True,
            urgency="high",
            red_flags=list(dict.fromkeys([*verdict.red_flags, *signals])),
            explanation=explanation,
            what_to_do=action,
        )
    return verdict
