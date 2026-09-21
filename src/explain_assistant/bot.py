"""Telegram bot: parents forward confusing messages or photos, get a simple text + voice reply.

Run:  TELEGRAM_BOT_TOKEN=... uv run sarathiai-bot
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from explain_assistant.assistant import build_reply, family_alert, split_sender
from explain_assistant.model import DEFAULT_ADAPTER, Explainer
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
    "Tip: add the SMS sender on the first line, e.g. 'From: VM-HDFCBK'."
)
MAX_MESSAGE_CHARS = 12_000


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
    def __init__(self, explainer: Explainer, state: State):
        self.explainer = explainer
        self.state = state
        self.model_lock = asyncio.Lock()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        buttons = [[InlineKeyboardButton("हिंदी", callback_data="lang:hi"), InlineKeyboardButton("English", callback_data="lang:en")]]
        await update.message.reply_text(WELCOME, reply_markup=InlineKeyboardMarkup(buttons))

    async def choose_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        lang = query.data.split(":", 1)[1]
        self.state.set_language(query.message.chat.id, lang)
        await query.answer()
        await query.edit_message_text("भाषा: हिंदी ✅" if lang == "hi" else "Language: English ✅")

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
        sender, body = split_sender(update.message.text)
        if len(body) > MAX_MESSAGE_CHARS:
            await update.message.reply_text("यह संदेश बहुत लंबा है। कृपया छोटा हिस्सा भेजें।\nThis message is too long. Please send a shorter section.")
            return
        await self._answer(update, context, sender, body)

    async def on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from explain_assistant.ocr import image_to_text

        photo = update.message.photo[-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.jpg"
            await (await photo.get_file()).download_to_drive(path)
            text = await asyncio.to_thread(image_to_text, path)
        caption_sender, _ = split_sender((update.message.caption or "") + "\n.")
        if not text.strip():
            lang = self.state.language(update.effective_chat.id)
            await update.message.reply_text(build_reply(None, lang).text)
            return
        await self._answer(update, context, caption_sender if caption_sender != "unknown" else "photo of a document", text)

    async def _answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE, sender: str, body: str):
        chat_id = update.effective_chat.id
        lang = self.state.language(chat_id)
        await context.bot.send_chat_action(chat_id, "typing")
        try:
            async with self.model_lock:
                verdict = await asyncio.to_thread(self.explainer.explain, sender, body, lang)
        except Exception:
            log.exception("explanation failed")
            verdict = None
        reply = build_reply(verdict, lang)
        await update.message.reply_text(reply.text)

        with tempfile.TemporaryDirectory() as tmp:
            try:
                audio = await asyncio.to_thread(synthesize, reply.speech, lang, Path(tmp) / "reply")
                with open(audio, "rb") as fh:
                    if audio.suffix == ".ogg":
                        await update.message.reply_voice(fh)
                    else:
                        await update.message.reply_audio(fh)
            except Exception:
                log.exception("voice reply failed")

        if reply.alert_family and verdict is not None:
            name = update.effective_user.first_name or "Your parent"
            for family_id in self.state.family(chat_id):
                try:
                    await context.bot.send_message(family_id, family_alert(verdict, sender, body, name))
                except Exception:
                    log.exception("family alert to %s failed", family_id)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN (create a bot with @BotFather).")
    log.info("loading model…")
    adapter = os.environ.get("SARATHIAI_ADAPTER") or os.environ.get("SAMJHAO_ADAPTER") or DEFAULT_ADAPTER
    bot = SarathiAIBot(Explainer(adapter_path=adapter), State(STATE_PATH))
    app = Application.builder().token(token).concurrent_updates(True).build()
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("guardian", bot.guardian))
    app.add_handler(CommandHandler("family", bot.family))
    app.add_handler(CallbackQueryHandler(bot.choose_language, pattern=r"^lang:"))
    app.add_handler(MessageHandler(filters.PHOTO, bot.on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_text))
    log.info("Sarathi.AI bot running")
    app.run_polling()


if __name__ == "__main__":
    main()
