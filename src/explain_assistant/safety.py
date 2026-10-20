"""Small, deterministic safety net for high-confidence scam patterns.

These checks run before inference and also protect model failure paths. They
upgrade risk when actionable signals occur together; they never downgrade it.
"""

import re
import unicodedata
from dataclasses import replace

from explain_assistant.model import Verdict

_PERSONAL_SENDER = re.compile(r"(?:\+?91[\s-]?)?[6-9]\d{9}$")


def _words(pattern: str) -> re.Pattern:
    # Include Hindi combining marks: Python's word boundary splits Hindi words.
    return re.compile(r"(?<![\w\u0900-\u097f])(?:" + pattern + r")(?![\w\u0900-\u097f])", re.I)


_CREDENTIAL = _words(r"otp|upi\s*pin|cvv|password|mpin|ओटीपी|पासवर्ड|पिन")
_CREDENTIAL_REQUEST = _words(r"share|tell|enter|send|reply|give|provide|बताइए|बताएं|बताओ|डालें|भेजें|साझा")
_CREDENTIAL_NEGATION = _words(r"never|do\s+not|don't|not\s+ask|कभी\s+नहीं|नहीं|मत|न")
_PAYMENT = _words(r"pay|payment|fee|transfer|send\s+money|upi|रुपये|भुगतान|पैसे")
_URGENCY = _words(r"immediately|urgent|today|tonight|within\s+\d+\s*(?:min|hour)|blocked?|disconnect(?:ed|ion)?|arrest(?:ed)?|तुरंत|आज|बंद|गिरफ्तार")
_OFF_PLATFORM = re.compile(r"(?:https?://|www\.|\.apk\b|\bcall\b|\bwhatsapp\b|लिंक|फोन|कॉल)", re.IGNORECASE)
_IMPERSONATION = _words(r"bank|police|cbi|customs|electricity|power|kyc|aadhaar|pension|msedcl|bescom|uppcl|tnpdcl|बैंक|पुलिस|बिजली")
_DISCONNECTION = re.compile(r"disconnect(?:ed|ion)?|(?:connection|power|electricity).{0,60}\bcut\b|बिजली.{0,40}(?:बंद|काट)|कनेक्शन.{0,40}(?:बंद|काट)", re.I)
_WARNING = _words(r"scam\s+(?:example|warning)|example\s+of\s+a\s+scam|is\s+a\s+scam|धोखे\s+का\s+उदाहरण")
_OBFUSCATED_OTP = re.compile(r"(?<!\w)o[\s.\-]*t[\s.\-]*p(?!\w)", re.I)
_DELIVERY_AT_DOOR = re.compile(r"\bdelivery\s+otp\b.{0,100}\bonly\b.{0,30}\bdelivery\s+agent\b.{0,30}\b(?:door|doorstep)\b", re.I)


def _looks_personal(sender: str) -> bool:
    return bool(_PERSONAL_SENDER.fullmatch(re.sub(r"[\s()-]", "", sender.strip())))


def detect_risk(sender: str, text: str) -> list[str]:
    """Detect actionable patterns independently of model output.

    Sender headers are hints, not authentication. Negation is clause-local so
    an unrelated disclaimer cannot hide an explicit credential request.
    """
    text = _OBFUSCATED_OTP.sub("OTP", unicodedata.normalize("NFKC", text))
    clauses = re.split(r"[.!?;\n।]+|\b(?:but|instead|however)\b|लेकिन", text, flags=re.I)
    credential_request = any(
        _CREDENTIAL.search(clause) and _CREDENTIAL_REQUEST.search(clause)
        and not _CREDENTIAL_NEGATION.search(clause) and not _WARNING.search(clause)
        # A narrowly worded delivery handoff needs model judgement, not an
        # automatic scam label. Other credentials/remote requests still count.
        and not (
            _DELIVERY_AT_DOOR.search(clause)
            and all(word.lower() == "otp" for word in _CREDENTIAL.findall(clause))
            and not _OFF_PLATFORM.search(clause)
        )
        for clause in clauses
    )
    personal_impersonation = False
    coercive_action = False
    disconnection_action = False
    for index, clause in enumerate(clauses):
        if _CREDENTIAL_NEGATION.search(clause) or _WARNING.search(clause):
            continue
        following = clauses[index + 1] if index + 1 < len(clauses) else ""
        if _CREDENTIAL_NEGATION.search(following) or _WARNING.search(following):
            following = ""
        context = clause + " " + following
        institution = bool(_IMPERSONATION.search(clause))
        action = bool(_PAYMENT.search(context) or _OFF_PLATFORM.search(context))
        coercive = bool(_URGENCY.search(clause) and action)
        personal_impersonation |= _looks_personal(sender) and institution and coercive
        coercive_action |= coercive
        disconnection_action |= institution and bool(_DISCONNECTION.search(clause)) and bool(_OFF_PLATFORM.search(context))
    signals: list[str] = []
    if personal_impersonation:
        signals.append("personal_number_impersonation")
    if credential_request:
        signals.append("credential_request")
    if coercive_action and (personal_impersonation or disconnection_action):
        signals.append("urgent_external_action")
    if disconnection_action:
        signals.append("disconnection_action_request")
    return signals if credential_request or personal_impersonation or disconnection_action else []


def apply_safety_net(verdict: Verdict | None, sender: str, text: str, language: str) -> Verdict | None:
    """Warn even without model output; never downgrade an existing scam verdict."""
    signals = detect_risk(sender, text)
    if signals and (verdict is None or verdict.verdict != "scam"):
        if language == "hi":
            explanation = "इस संदेश में गुप्त जानकारी माँगी गई है या संस्था के नाम पर तुरंत कॉल या भुगतान का दबाव है। यह धोखा हो सकता है।"
            action = "अभी रुकें और परिवार से जाँच कराएँ। तब तक लिंक न खोलें, पैसे या OTP न दें।"
        else:
            explanation = "This message requests secret information or pressures you to call or pay in an organisation's name. It may be a scam."
            action = "Pause and ask your family to check. Until then, do not open links, pay, or share an OTP."
        verdict = verdict or Verdict("scam", "scam", True, "high")
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
