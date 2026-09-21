import json
import re

from explain_assistant.assistant import build_reply, split_sender
from explain_assistant.audit import audit, normalize
from explain_assistant.bot import State
from explain_assistant.data_gen import CASES, LABEL_KEYS, _holdout_index, generate
from explain_assistant.model import Verdict, parse_verdict
from explain_assistant.prompt import DOC_TYPES
from explain_assistant.safety import apply_safety_net


def test_parse_verdict_handles_noise_around_json():
    raw = 'sure!\n{"doc_type": "scam", "verdict": "SCAM", "is_dangerous": true, "urgency": "high", "red_flags": ["x"], "explanation": "e", "what_to_do": "w"}\n'
    v = parse_verdict(raw)
    assert v and v.verdict == "scam" and v.is_dangerous and v.red_flags == ["x"]


def test_parse_verdict_rejects_garbage():
    assert parse_verdict("no json here") is None
    assert parse_verdict('{"verdict": "safe"') is None
    assert parse_verdict('{"explanation": "missing keys"}') is None
    assert parse_verdict('{"doc_type":"x","verdict":"maybe","red_flags":[]}') is None
    assert parse_verdict('{"doc_type":"x","verdict":"safe","is_dangerous":"false","red_flags":[]}') is None


def test_parse_verdict_handles_braces_and_multiple_objects():
    raw = 'note {not json} then {"doc_type":"otp","verdict":"safe","urgency":"low","red_flags":[],"explanation":"Use {care}","what_to_do":"wait"}'
    assert parse_verdict(raw).doc_type == "otp"


def test_scam_verdict_cannot_disable_family_alert():
    raw = '{"doc_type":"scam","verdict":"scam","is_dangerous":false,"urgency":"high","red_flags":[]}'
    assert parse_verdict(raw).is_dangerous


def test_safety_net_catches_personal_number_utility_false_negative():
    safe = Verdict("utility_bill", "safe", False, "low", [], "Normal bill", "Pay it")
    result = apply_safety_net(
        safe, "+91 9876543210", "MSEDCL electricity disconnected tonight. Call officer now", "en"
    )
    assert result.verdict == "scam" and result.is_dangerous
    assert "personal_number_impersonation" in result.red_flags


def test_safety_net_does_not_upgrade_normal_bill():
    safe = Verdict("utility_bill", "safe", False, "low", [], "Normal bill", "Pay it")
    result = apply_safety_net(safe, "VM-MSEDCL", "Your electricity bill is due on 25 June", "en")
    assert result == safe


def test_safety_net_does_not_misread_credential_safety_advice():
    safe = Verdict("bank_alert", "safe", False, "low", [], "Safety advice", "No action")
    english = apply_safety_net(
        safe, "VK-SBIINB", "SBI will NEVER ask you to share OTP, PIN or CVV. Call the number on your card.", "en"
    )
    hindi = apply_safety_net(
        safe, "VK-SBIINB", "बैंक कर्मचारी कभी नहीं पूछते। OTP या PIN किसी को मत बताएं।", "hi"
    )
    assert english == safe
    assert hindi == safe


def test_state_recovers_from_corruption_and_saves_atomically(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json")
    state = State(path)
    assert state.language(42) == "hi"
    state.set_language(42, "en")
    assert json.loads(path.read_text())["language"]["42"] == "en"
    assert not path.with_suffix(".json.tmp").exists()


def test_overlap_audit_normalizes_urls_and_numbers():
    assert normalize("Pay 500 at HTTP://bad.xyz/123") == "pay <number> at <url>"
    train = [{"meta": {"text": "Pay Rs 100 at http://bad.xyz/a"}}]
    evaluation = [{"text": "Pay Rs 999 at http://bad.xyz/b"}]
    assert audit(train, evaluation)["exact_normalized_duplicates"] == 1


def test_split_sender():
    assert split_sender("From: VM-HDFCBK\nRs 500 debited") == ("VM-HDFCBK", "Rs 500 debited")
    assert split_sender("Rs 500 debited") == ("unknown", "Rs 500 debited")
    assert split_sender("From: +91 9876543210") == ("unknown", "From: +91 9876543210")


def test_build_reply_alerts_family_only_for_danger():
    scam = Verdict("scam", "scam", True, "high", ["x"], "यह स्कैम है", "क्लिक न करें")
    safe = Verdict("otp", "safe", False, "low", [], "असली OTP", "किसी को न बताएं")
    assert build_reply(scam, "hi").alert_family
    assert not build_reply(safe, "hi").alert_family
    assert "सावधान" in build_reply(scam, "hi").text
    assert not build_reply(None, "en").alert_family


def test_every_case_label_is_well_formed():
    for case in CASES:
        label = case["label"]
        assert label["doc_type"] in DOC_TYPES, case["name"]
        assert label["verdict"] in {"safe", "suspicious", "scam"}
        assert label["is_dangerous"] == (label["verdict"] == "scam")


def _template_regex(template):
    parts = re.split(r"\{[a-z_0-9]+\}", template)
    return re.compile("^" + "(.+?)".join(re.escape(p) for p in parts) + "$", re.DOTALL)


def test_test_split_uses_only_held_out_phrasings():
    by_name = {c["name"]: c for c in CASES}
    for split, seed in (("train", 1), ("test", 3)):
        for row in generate(len(CASES) * 4, split, seed):
            assert tuple(json.loads(row["messages"][-1]["content"])) == LABEL_KEYS
            case = by_name[row["meta"]["case"]]
            if case["name"] == "prescription":
                continue
            held = _holdout_index(case)
            matches = {i for i, (_, t) in enumerate(case["variants"]) if _template_regex(t).match(row["meta"]["text"])}
            if split == "test":
                assert matches == {held}, case["name"]
            else:
                assert held not in matches and matches, case["name"]
