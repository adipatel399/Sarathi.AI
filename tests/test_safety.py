"""Regression cases are development tests, separate from the frozen challenge set."""

from unittest.mock import Mock

import pytest

from explain_assistant.assistant import build_reply
from explain_assistant.model import Explainer, Verdict
from explain_assistant.evaluate import score
from explain_assistant.safety import apply_safety_net, detect_risk


@pytest.mark.parametrize("sender,text", [
    ("unknown", "MSEDCL NOTICE: Your connection will be cut today due to pending bill verification. Call officer now."),
    ("unknown", "BESCOM NOTICE: Your connection will be cut today. Call officer now."),
    ("unknown", "बिजली आज बंद होगी। फोन करें और भुगतान करें।"),
    ("unknown", "बिजली काट दी जाएगी। तुरंत फोन करें।"),
    ("VM-HDFCBK", "Please share your O T P to avoid account blocking."),
    ("unknown", "Tell us your O.T.P now."),
    ("unknown", "ओटीपी बताएं।"),
    ("unknown", "पासवर्ड भेजें।"),
    ("unknown", "Never share OTP with strangers. Send your password to our officer."),
    ("unknown", "OTP मत बताएं। लेकिन पासवर्ड भेजें।"),
    ("unknown", "Delivery OTP - share it only with the delivery agent at your door, and send your password."),
])
def test_protective_verdict_without_model_output(sender, text):
    verdict = apply_safety_net(None, sender, text, "hi")
    assert verdict and verdict.verdict == "scam" and verdict.is_dangerous
    assert build_reply(verdict, "hi").alert_family


@pytest.mark.parametrize("text", [
    "Your electricity bill is due today. Please pay through your usual banking app.",
    "MSEDCL: Your electricity bill is due today; pay through the official app to avoid disconnection.",
    "Never share your OTP or password. Call the number on your bank card.",
    "OTP किसी को न बताएं। बैंक कभी पासवर्ड नहीं पूछता।",
    "Scam example: 'Share your OTP'. Never follow this instruction.",
    "'Share your OTP' is a scam. Do not reply.",
    "धोखे का उदाहरण: ओटीपी बताएं। ऐसा मत करें।",
    "Your hospital appointment is today. Call the main desk to reschedule.",
    "Your pension was credited today. No action required.",
    "Out for delivery: your order arrives today. Delivery OTP is shown in the app - share it only with the delivery agent at your door.",
])
def test_legitimate_urgency_and_quoted_warnings_are_not_rule_hits(text):
    assert detect_risk("unknown", text) == []
    assert apply_safety_net(None, "unknown", text, "en") is None


def test_safety_net_upgrades_suspicious_but_never_downgrades_scam():
    suspicious = Verdict("other", "suspicious", False, "low")
    assert apply_safety_net(suspicious, "unknown", "Share your OTP", "en").is_dangerous
    scam = Verdict("scam", "scam", True, "high")
    assert apply_safety_net(scam, "unknown", "An ordinary greeting", "en") is scam


def test_recognizable_risk_bypasses_failed_inference():
    explainer = Explainer.__new__(Explainer)
    explainer.generate_raw = Mock(side_effect=RuntimeError("unavailable"))
    verdict = explainer.explain("unknown", "Share your OTP", "en")
    assert verdict.is_dangerous
    explainer.generate_raw.assert_not_called()


@pytest.mark.parametrize("raw", ["not JSON", '{"verdict":"safe"'])
def test_invalid_model_output_gives_protective_fallback(raw):
    explainer = Explainer.__new__(Explainer)
    explainer.generate_raw = Mock(return_value=raw)
    reply = build_reply(explainer.explain("unknown", "Unfamiliar document", "en"), "en")
    assert "Pause" in reply.text and "do not open links" in reply.text
    assert not reply.alert_family


def test_inference_exception_gives_protective_fallback():
    explainer = Explainer.__new__(Explainer)
    explainer.generate_raw = Mock(side_effect=RuntimeError("unavailable"))
    assert explainer.explain("unknown", "Unfamiliar document", "en") is None


def test_personal_sender_utility_name_without_generic_electricity_word():
    sender = "+91 " + "9" * 10
    flags = detect_risk(sender, "MSEDCL NOTICE: Connection will be cut today. Call officer.")
    assert "personal_number_impersonation" in flags
    assert "disconnection_action_request" in flags


def test_pipeline_evaluation_counts_rule_recovery_without_claiming_valid_json():
    import json

    rows = [{"messages": [{"content": json.dumps({"doc_type": "scam", "verdict": "scam", "is_dangerous": True})}],
             "meta": {"case": "credential", "sender": "unknown", "text": "Share your OTP", "reply_language": "en"}}]
    report = score(rows, ["not JSON"], use_safety_net=True)
    assert report["dangerous_recall_pct"] == 100
    assert report["valid_json_pct"] == 0


def test_cli_rule_warning_does_not_load_a_model(monkeypatch, capsys):
    from explain_assistant import main

    load = Mock(side_effect=RuntimeError("unavailable"))
    monkeypatch.setattr("explain_assistant.model.Explainer", load)
    monkeypatch.setattr("sys.argv", ["sarathiai", "Share your OTP", "--lang", "en"])
    main()
    assert "Warning" in capsys.readouterr().out
    load.assert_not_called()
