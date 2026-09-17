import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from explain_assistant.prompt import build_messages

BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_ADAPTER = Path(__file__).resolve().parents[2] / "adapters"

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Verdict:
    doc_type: str
    verdict: str
    is_dangerous: bool
    urgency: str
    red_flags: list[str] = field(default_factory=list)
    explanation: str = ""
    what_to_do: str = ""


def parse_verdict(raw: str) -> Verdict | None:
    """Pull the JSON object out of raw model output; None if it is missing or malformed."""
    match = _JSON_OBJECT.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        verdict = str(data["verdict"]).lower()
        return Verdict(
            doc_type=str(data["doc_type"]),
            verdict=verdict,
            is_dangerous=bool(data.get("is_dangerous", verdict == "scam")),
            urgency=str(data.get("urgency", "low")).lower(),
            red_flags=[str(f) for f in data.get("red_flags", [])],
            explanation=str(data.get("explanation", "")),
            what_to_do=str(data.get("what_to_do", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


class Explainer:
    def __init__(self, adapter_path: str | Path | None = DEFAULT_ADAPTER, base_model: str = BASE_MODEL):
        from mlx_lm import load

        adapter = str(adapter_path) if adapter_path and Path(adapter_path).exists() else None
        if adapter_path and adapter is None:
            raise FileNotFoundError(f"LoRA adapter not found at {adapter_path}")
        self.model, self.tokenizer = load(base_model, adapter_path=adapter)

    def generate_raw(self, sender: str, text: str, reply_language: str = "hi", max_tokens: int = 400) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            build_messages(sender, text, reply_language), add_generation_prompt=True
        )
        return generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0), verbose=False,
        )

    def explain(self, sender: str, text: str, reply_language: str = "hi") -> Verdict | None:
        return parse_verdict(self.generate_raw(sender, text, reply_language))

    def generate_many(self, items: list[tuple[str, str, str]], max_tokens: int = 400, batch_size: int = 8) -> list[str]:
        """Greedy batched generation for (sender, text, reply_language) tuples."""
        from mlx_lm import batch_generate

        prompts = [
            self.tokenizer.apply_chat_template(build_messages(s, t, lang), add_generation_prompt=True)
            for s, t, lang in items
        ]
        outputs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            response = batch_generate(
                self.model, self.tokenizer, chunk, max_tokens=max_tokens,
                completion_batch_size=batch_size, prefill_batch_size=batch_size,
            )
            outputs.extend(response.texts)
        return outputs
