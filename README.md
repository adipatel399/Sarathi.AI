# Samjhao (समझाओ) — "Explain this to me"

**An on-device AI helper for elderly Indians.** A parent forwards any confusing SMS, bill, prescription or
government letter (or a photo of it) to a Telegram bot. A fine-tuned 4B model running on a MacBook replies in
simple Hindi or English, with a voice note, tells them *what it is* and *what to do* — and flags scams. If it's
dangerous, their son or daughter gets an alert.

Nothing leaves the machine: model, OCR and speech all run locally.

Samjhao also has a conservative deterministic safety net around the model. It catches high-confidence combinations
such as an organisation impersonated from a personal number plus an urgent call, payment, credential or external-link
request. This directly guards the known utility-disconnection false-negative class without changing ordinary safe bills.

```
Parent (Telegram)  ── text / photo ──►  Apple Vision OCR (photos)
                                              │
                                              ▼
                           Qwen3-4B-Instruct + Samjhao LoRA (MLX, 4-bit)
                           → {doc_type, verdict, is_dangerous, urgency,
                              red_flags, explanation, what_to_do}
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
                    ▼                         ▼                          ▼
             text reply (hi/en)    voice note (macOS Lekha/Rishi)   family alert if dangerous
```

Suspicious replies show up to three plain-language reasons so a parent can understand *why* they should be careful,
not just see a red warning. Model output is schema-validated before use, and contradictory output cannot silently
disable a scam alert.

## Results

Same prompt for both models, greedy decoding, on an M4 MacBook Pro (16 GB).

**40 hand-written real-world-style messages** (`evals/realworld.json`) — scam types and senders *never seen in
training*: FASTag KYC, Aadhaar "update fee", stock-tip WhatsApp groups, fake electricity APKs, eSIM OTP theft,
"sent money by mistake", Ayushman card fee, sextortion "police", plus genuine Jio/IRCTC/lab-report/school-fee/
EMI/property-tax messages and OCR'd documents.

| | Base Qwen3-4B | **Samjhao (fine-tuned)** |
|---|---|---|
| Valid JSON | 85.0% | **100%** |
| Verdict accuracy (safe / suspicious / scam) | 30.0% | **100%** (40/40) |
| Dangerous messages caught | 78.9% | **100%** (19/19) |
| Genuine messages wrongly called scams | 9 of 19 | **0 of 19** |
| Reply in the requested language | 80.0% | **100%** |
| Seconds per message | 2.9 | 2.4 |

**Held-out test split** (315 messages, wording never seen in training):

| | Base Qwen3-4B (63-msg stratified subset) | **Samjhao** (all 315) |
|---|---|---|
| Valid JSON | 84.1% | **100%** |
| Verdict accuracy | 41.3% | **99.0%** |
| Document-type accuracy | — | **99.0%** |
| Dangerous messages caught | 73.3% | **98.7%** (148/150) |
| Genuine messages wrongly called scams | 13 of 30 | **1 of 150** |
| Reply in the requested language | 84.1% | **100%** |

What the numbers mean — and don't:
- The base model usually *senses* danger but hedges with "suspicious", calls genuine bank/OTP/hospital
  messages suspicious (the false alarms that make elderly users stop trusting a tool), and often runs out of
  its 400-token budget writing long red-flag lists, which is why some of its JSON is invalid.
- The real-world set is small (40) and written by the author, so treat 100% as "no failures found on 40",
  not as a guarantee.
- Samjhao's 3 test mistakes: two electricity-disconnection scams from a `+91` number that named the real
  utility (`MSEDCL NOTICE: … call officer`) were called safe, and one genuine Bank of Baroda debit alert was
  called a scam. Sender-vs-content mismatch on utility names is the clearest next thing to improve.
- Raw reports with every mistake: `results/`.

Training: 200 steps (≈800 examples, one-third of an epoch), validation loss 1.78 → 0.125 (step 100) → 0.035
(step 200); stopped there because it had converged and more steps would mostly memorise templates.

## What it handles

21 message families, generated with realistic Indian senders (DLT headers like `VM-HDFCBK` vs personal
`+91` numbers), amounts, dates, banks, utilities, hospitals and drugs — in English, Hindi and Hinglish:

| Genuine (hard negatives) | Scams |
|---|---|
| Bank OTPs, debit alerts, pension credits | Fake KYC / account-block links |
| Electricity bills with "pay by due date" urgency | Electricity disconnection "call officer" scam |
| LIC premium reminders, life-certificate notices | Digital arrest (fake CBI/police) |
| PM-KISAN installments, delivery updates | Lottery / advance-fee, fake job offers |
| Hospital appointments, **prescriptions** (explains each medicine) | UPI PIN "cashback/refund", family-impersonation |
| | Parcel redelivery fee, pension/OTP scams |
| | **Sender mismatch**: a real-looking bank alert from a personal number |

## How it was built

1. **Synthetic data** (`src/explain_assistant/data_gen.py`) — 2,400 train / 150 valid / 315 test examples (regenerate with `uv run samjhao-data`; the trained adapter is included in `adapters/`).
   Every family has several phrasings; one is **held out of training** and used only for the test split, so
   the test measures generalisation to unseen wording. Labels include simple Hindi/English explanations.
2. **LoRA fine-tune** of `Qwen3-4B-Instruct-2507` (4-bit, MLX) on a 16 GB M4 MacBook Pro: 16 layers,
   loss on the answer only (`--mask-prompt`), gradient checkpointing, peak memory ≈ 4.6 GB.
3. **Evaluation** on the held-out split **and** 40 hand-written real-world-style messages
   (`evals/realworld.json`) covering scam types and senders never seen in training (FASTag, Aadhaar fee,
   stock-tip groups, fake APKs, eSIM, Ayushman card, lab reports, IRCTC, school fees…).

## Run it

Requirements: Apple Silicon Mac, [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra ocr
uv run samjhao-data                      # generate data/{train,valid,test}.jsonl
./scripts/train.sh                        # LoRA fine-tune → adapters/
uv run samjhao-eval --out results/test.json
uv run samjhao-eval --data evals/realworld.json --out results/realworld.json
uv run samjhao-eval --adapter none --data evals/realworld.json --out results/realworld_base.json
```

Try a message:

```bash
uv run samjhao "Your electricity will be disconnected tonight 9.30pm. Call officer 9876543210" --sender "+91 9876543210" --lang hi --speak
uv run samjhao --image bill.jpg --lang en
```

Telegram bot (create a bot with [@BotFather](https://t.me/BotFather)):

```bash
TELEGRAM_BOT_TOKEN=... uv run samjhao-bot
```

- Parent: `/start` → pick हिंदी / English → forward messages or send photos. Add `From: VM-HDFCBK` as the
  first line when the SMS sender is known — sender-vs-content mismatch is one of the strongest scam signals.
- Child: `/guardian` → gives a `/family <code>` command for the parent to send. After linking, the child gets
  an alert whenever the parent receives a dangerous message.

## Limitations

- Training data is synthetic and template-based; real messages are messier. The real-world set is small (40).
- The model reasons in Hindi and English only; more Indian languages would need a translation/TTS layer
  (e.g. AI4Bharat IndicTrans2 + Indic-TTS).
- Photo OCR uses Apple Vision (English script); handwritten prescriptions are out of scope.
- It is a helper, not a guarantee — when in doubt the advice is always "don't click, don't pay, ask family".

## Tests

```bash
uv run pytest
```
