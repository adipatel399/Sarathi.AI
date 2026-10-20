"""Telegram bot: parents forward confusing messages or photos, get a simple text + voice reply.

Run:  TELEGRAM_BOT_TOKEN=... uv run sarathiai-bot
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
from time import monotonic
from uuid import uuid4
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from explain_assistant.assistant import build_reply, build_simple_reply, family_alert, split_sender
from explain_assistant.model import DEFAULT_ADAPTER, Explainer
from explain_assistant.safety import apply_safety_net
from explain_assistant.voice import synthesize

log = logging.getLogger("sarathiai")
_LEGACY_STATE_PATH = Path.home() / ".samjhao" / "state.json"
STATE_PATH = Path(
    os.environ.get("SARATHIAI_STATE")
    or os.environ.get("SAMJHAO_STATE")
    or (_LEGACY_STATE_PATH if _LEGACY_STATE_PATH.exists() else Path.home() / ".sarathiai" / "state.json")
)

WELCOME = (
    "नमस्ते! मैं Sarathi.AI हूँ—आपका भरोसेमंद AI मार्गदर्शक। कोई भी संदेश, बिल, दवा का पर्चा या सरकारी चिट्ठी जो समझ न आए, "
    "मुझे फ़ॉरवर्ड करें या उसकी फ़ोटो भेजें। मैं आसान शब्दों में बताऊँगा कि यह क्या है और क्या करना है।\n\n"
    "Hello! Forward any confusing message or send a photo of a letter, bill or prescription.\n\n"
    "पहले अपनी भाषा चुनें। Choose your language below."
)
MAX_MESSAGE_CHARS = 12_000
ANSWER_TTL = 30 * 60
MAX_ANSWERS = 8

GUIDE = {
    "hi": "अब कोई संदेश यहाँ भेजें या बिल/चिट्ठी की साफ़ फ़ोटो भेजें। मैं बताऊँगा कि आगे क्या करना है। भेजने वाले का नाम पता हो तो बाद में जोड़ सकते हैं। पैसे या OTP देने से पहले रुकें।",
    "en": "Send a message here, or a clear photo of a bill or letter. I'll explain what to do next. You can add the sender afterwards if you know it. Pause before paying or sharing an OTP.",
}


def action_buttons(key: str, lang: str) -> InlineKeyboardMarkup:
    labels = ("🔊 फिर सुनें", "👉 आसान शब्दों में", "👪 परिवार से पूछें", "✏️ भेजने वाला जोड़ें") if lang == "hi" else (
        "🔊 Read aloud", "👉 Explain more simply", "👪 Ask my family", "✏️ Add sender",
    )
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=f"help:{action}:{key}")]
        for label, action in zip(labels, ("voice", "simple", "family", "sender"))
    ])


def recent_answers(chat_data: dict) -> dict:
    """Keep only a bounded, short-lived in-memory history for action buttons."""
    answers = chat_data.setdefault("answers", {})
    now = monotonic()
    for key in list(answers):
        if now - answers[key]["created"] >= ANSWER_TTL:
            del answers[key]
    while len(answers) > MAX_ANSWERS:
        del answers[next(iter(answers))]
    return answers


class State:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"language": {}, "family": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state root must be an object")
            data.setdefault("language", {})
            data.setdefault("family", {})
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            log.exception("state file is unreadable; starting with empty state")
            return {"language": {}, "family": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def language(self, chat_id: int) -> str:
        return self.data["language"].get(str(chat_id), "hi")

    def set_language(self, chat_id: int, lang: str):
        with self.lock:
            self.data["language"][str(chat_id)] = lang
            self._save()

    def family(self, chat_id: int) -> list[int]:
        return self.data["family"].get(str(chat_id), [])

    def add_family(self, chat_id: int, family_chat_id: int):
        with self.lock:
            members = self.data["family"].setdefault(str(chat_id), [])
            if family_chat_id not in members:
                members.append(family_chat_id)
            self._save()


class SarathiAIBot:
    def __init__(self, explainer: Explainer | None, state: State):
        self.explainer = explainer
        self.state = state
        self.model_lock = asyncio.Lock()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.chat_data.pop("awaiting_sender", None)
        buttons = [[InlineKeyboardButton("हिंदी", callback_data="lang:hi"), InlineKeyboardButton("English", callback_data="lang:en")]]
        await update.message.reply_text(WELCOME, reply_markup=InlineKeyboardMarkup(buttons))

    async def choose_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if update.effective_chat.type != "private":
            await query.answer("Please open a private chat with the bot.", show_alert=True)
            return
        lang = query.data.split(":", 1)[1]
        self.state.set_language(update.effective_chat.id, lang)
        await query.answer()
        await query.edit_message_text(GUIDE[lang])

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.chat_data.pop("awaiting_sender", None)
        await update.message.reply_text(GUIDE[self.state.language(update.effective_chat.id)])

    async def guardian(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        await update.message.reply_text(
            "You'll get alerts when your parent receives a scam.\n"
            f"Ask them to send this to the bot from their phone:\n\n/family {chat_id}"
        )

    async def family(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args or not context.args[0].lstrip("-").isdigit():
            await update.message.reply_text("Usage: /family <code from your child's /guardian message>")
            return
        family_id = int(context.args[0])
        self.state.add_family(update.effective_chat.id, family_id)
        await update.message.reply_text("परिवार जुड़ गया ✅ Family linked.")
        name = update.effective_user.first_name or "Your parent"
        await context.bot.send_message(family_id, f"✅ {name} linked you for scam alerts.")

    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pending = context.chat_data.get("awaiting_sender")
        if pending:
            record = recent_answers(context.chat_data).get(pending)
            if record is None:
                context.chat_data.pop("awaiting_sender", None)
            else:
                sender = update.message.text.strip()
                if not sender or len(sender) > 100 or "\n" in sender:
                    await update.message.reply_text(
                        "एक लाइन में भेजने वाले का नाम/नंबर लिखें, या /cancel भेजें।\nType just the sender name/number on one line, or send /cancel."
                    )
                    return
                context.chat_data.pop("awaiting_sender", None)
                await self._answer(update, context, sender, record["body"])
                return
        sender, body = split_sender(update.message.text)
        if len(body) > MAX_MESSAGE_CHARS:
            await update.message.reply_text("यह संदेश बहुत लंबा है। कृपया छोटा हिस्सा भेजें।\nThis message is too long. Please send a shorter section.")
            return
        await self._answer(update, context, sender, body)

    async def on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from explain_assistant.ocr import image_to_text

        context.chat_data.pop("awaiting_sender", None)
        photo = update.message.photo[-1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "doc.jpg"
                await (await photo.get_file()).download_to_drive(path)
                text = await asyncio.to_thread(image_to_text, path)
        except Exception:
            log.exception("photo reading failed")
            text = ""
        caption_sender, _ = split_sender((update.message.caption or "") + "\n.")
        if not text.strip():
            await self._answer(update, context, "unknown", "")
            return
        await self._answer(update, context, caption_sender if caption_sender != "unknown" else "photo of a document", text)

    async def _answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE, sender: str, body: str):
        chat_id = update.effective_chat.id
        lang = self.state.language(chat_id)
        await context.bot.send_chat_action(chat_id, "typing")
        if len(body) > MAX_MESSAGE_CHARS:
            await update.effective_message.reply_text("कृपया छोटा हिस्सा भेजें।\nPlease send a shorter section.")
            return
        verdict = apply_safety_net(None, sender, body, lang)
        if verdict is None and body.strip() and self.explainer is not None:
            try:
                async with self.model_lock:
                    verdict = await asyncio.to_thread(self.explainer.explain, sender, body, lang)
            except Exception:
                log.exception("explanation failed")
        verdict = apply_safety_net(verdict, sender, body, lang)
        reply = build_reply(verdict, lang)
        key = uuid4().hex[:12]
        answers = recent_answers(context.chat_data)
        record = {"created": monotonic(), "sender": sender, "body": body, "lang": lang,
                  "verdict": verdict, "reply": reply, "busy": True, "shared": set()}
        answers[key] = record
        asyncio.get_running_loop().call_later(ANSWER_TTL, answers.pop, key, None)
        recent_answers(context.chat_data)
        await update.effective_message.reply_text(reply.text, reply_markup=action_buttons(key, lang))

        if reply.alert_family and verdict is not None:
            name = update.effective_user.first_name or "Your parent"
            for family_id in self.state.family(chat_id):
                try:
                    await context.bot.send_message(family_id, family_alert(verdict, sender, body, name))
                    record["shared"].add(family_id)
                except Exception:
                    log.exception("family alert to %s failed", family_id)
        record["busy"] = False
        await self._send_audio(update.effective_message, reply.speech, lang)

    async def _send_audio(self, message, speech: str, lang: str):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                audio = await asyncio.to_thread(synthesize, speech, lang, Path(tmp) / "reply")
                with open(audio, "rb") as fh:
                    if audio.suffix == ".ogg":
                        await message.reply_voice(fh)
                    else:
                        await message.reply_audio(fh)
            except Exception:
                log.exception("voice reply failed")
                await message.reply_text("आवाज़ नहीं बन पाई। ऊपर लिखा जवाब पढ़ें या फिर सुनें दबाएँ।\nAudio unavailable. Read the text above or tap Read aloud to retry.")

    async def help_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if update.effective_chat.type != "private":
            await query.answer("Please open a private chat with the bot.", show_alert=True)
            return
        await query.answer()
        _, action, key = query.data.split(":", 2)
        record = recent_answers(context.chat_data).get(key)
        if record is None:
            await query.message.reply_text("यह जवाब पुराना है। संदेश फिर भेजें।\nThis answer has expired. Please send the message again.")
            return
        lang = record["lang"]
        if record["busy"]:
            return
        record["busy"] = True
        try:
            if action == "voice":
                await self._send_audio(query.message, record["reply"].speech, lang)
            elif action == "simple":
                reply = build_simple_reply(record["verdict"], lang)
                await query.message.reply_text(reply.text, reply_markup=action_buttons(key, lang))
            elif action == "sender":
                context.chat_data["awaiting_sender"] = key
                label = "पता नहीं / छोड़ें" if lang == "hi" else "Don't know / Skip"
                await query.message.reply_text(
                    "SMS के ऊपर लिखा नाम या नंबर भेजें, जैसे VM-HDFCBK। पता न हो तो छोड़ें।" if lang == "hi" else
                    "Type the name or number shown above the SMS, for example VM-HDFCBK. Skip if you don't know it.",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=f"help:skip:{key}")]]),
                )
            elif action == "skip":
                if context.chat_data.get("awaiting_sender") == key:
                    context.chat_data.pop("awaiting_sender", None)
                await query.message.reply_text(record["reply"].text, reply_markup=action_buttons(key, lang))
            elif action == "family":
                await self._ask_family(update, context, record)
        finally:
            record["busy"] = False

    async def _ask_family(self, update: Update, context: ContextTypes.DEFAULT_TYPE, record: dict):
        message = update.callback_query.message
        members = self.state.family(update.effective_chat.id)
        lang = record["lang"]
        if not members:
            await message.reply_text(
                "अभी परिवार जुड़ा नहीं है। परिवार के सदस्य से इस बॉट में /guardian भेजने को कहें। फिर उनका दिया /family कमांड यहाँ भेजें।" if lang == "hi" else
                "No family member is linked yet. Ask them to send /guardian to this bot, then send their /family command here."
            )
            return
        failed = False
        name = update.effective_user.first_name or "Your parent"
        snippet = record["body"][:300]
        request = f"👪 {name} asks you to help check a message. Please call them.\n\nMessage:\n{snippet}\n\nSarathi.AI:\n{record['reply'].text}"
        for member in members:
            if member in record["shared"]:
                continue
            try:
                await context.bot.send_message(member, request)
                record["shared"].add(member)
            except Exception:
                failed = True
                log.exception("family help request failed")
        if failed:
            text = "कुछ लोगों को संदेश नहीं पहुँचा। परिवार को सीधे फोन करें, या फिर कोशिश करें।" if lang == "hi" else "Some messages could not be delivered. Call your family directly, or try again."
        else:
            text = "परिवार को बता दिया है। ज़रूरत हो तो उन्हें सीधे फोन करें।" if lang == "hi" else "Your family has been notified. You can also call them directly."
        await message.reply_text(text)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN (create a bot with @BotFather).")
    log.info("loading model…")
    adapter = os.environ.get("SARATHIAI_ADAPTER") or os.environ.get("SAMJHAO_ADAPTER") or DEFAULT_ADAPTER
    try:
        explainer = Explainer(adapter_path=adapter)
    except Exception:
        log.exception("model unavailable; running protective checks only, restart after restoring the model")
        explainer = None
    bot = SarathiAIBot(explainer, State(STATE_PATH))
    app = Application.builder().token(token).concurrent_updates(True).build()
    private = filters.ChatType.PRIVATE
    app.add_handler(CommandHandler("start", bot.start, filters=private))
    app.add_handler(CommandHandler("help", bot.cancel, filters=private))
    app.add_handler(CommandHandler("cancel", bot.cancel, filters=private))
    app.add_handler(CommandHandler("guardian", bot.guardian, filters=private))
    app.add_handler(CommandHandler("family", bot.family, filters=private))
    app.add_handler(CallbackQueryHandler(bot.choose_language, pattern=r"^lang:(hi|en)$"))
    app.add_handler(CallbackQueryHandler(bot.help_action, pattern=r"^help:(voice|simple|family|sender|skip):[a-f0-9]{12}$"))
    app.add_handler(MessageHandler(private & filters.PHOTO, bot.on_photo))
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, bot.on_text))
    log.info("Sarathi.AI bot running")
    app.run_polling()


if __name__ == "__main__":
    main()
