import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from explain_assistant.assistant import build_simple_reply
from explain_assistant.bot import ANSWER_TTL, MAX_ANSWERS, GUIDE, SarathiAIBot, recent_answers
from explain_assistant.model import Verdict


@pytest.fixture
def flow():
    state = Mock()
    state.language.return_value = "en"
    state.family.return_value = []
    explainer = Mock()
    explainer.explain.return_value = Verdict("other", "safe", False, "low", [], "A normal notice.", "Keep it for your records.")
    bot = SarathiAIBot(explainer, state)
    bot._send_audio = AsyncMock()
    message = SimpleNamespace(reply_text=AsyncMock(), text="A normal notice.")
    update = SimpleNamespace(
        message=message, effective_message=message,
        effective_chat=SimpleNamespace(id=42, type="private"),
        effective_user=SimpleNamespace(first_name="Parent"),
        callback_query=SimpleNamespace(message=message, answer=AsyncMock(), edit_message_text=AsyncMock()),
    )
    context = SimpleNamespace(chat_data={}, bot=SimpleNamespace(send_chat_action=AsyncMock(), send_message=AsyncMock()))
    return bot, update, context


def action(update, context, name):
    key = next(reversed(context.chat_data["answers"]))
    update.callback_query.data = f"help:{name}:{key}"
    return context.chat_data["answers"][key]


def test_language_selection_shows_next_step(flow):
    bot, update, context = flow
    update.callback_query.data = "lang:hi"
    asyncio.run(bot.choose_language(update, context))
    bot.state.set_language.assert_called_once_with(42, "hi")
    update.callback_query.edit_message_text.assert_awaited_once_with(GUIDE["hi"])


def test_message_needs_no_sender_prefix_and_has_all_actions(flow):
    bot, update, context = flow
    asyncio.run(bot.on_text(update, context))
    bot.explainer.explain.assert_called_once_with("unknown", "A normal notice.", "en")
    markup = update.message.reply_text.await_args.kwargs["reply_markup"]
    assert [row[0].text for row in markup.inline_keyboard] == [
        "🔊 Read aloud", "👉 Explain more simply", "👪 Ask my family", "✏️ Add sender",
    ]


def test_optional_sender_rechecks_original_message(flow):
    bot, update, context = flow

    async def exercise():
        await bot.on_text(update, context)
        action(update, context, "sender")
        await bot.help_action(update, context)
        update.message.text = "VM-HDFCBK"
        await bot.on_text(update, context)

    asyncio.run(exercise())
    assert "awaiting_sender" not in context.chat_data
    assert bot.explainer.explain.call_args.args == ("VM-HDFCBK", "A normal notice.", "en")


def test_sender_skip_and_cancel_allow_new_messages(flow):
    bot, update, context = flow

    async def exercise():
        await bot.on_text(update, context)
        for finish in ("skip", "cancel"):
            action(update, context, "sender")
            await bot.help_action(update, context)
            assert "awaiting_sender" in context.chat_data
            if finish == "skip":
                action(update, context, "skip")
                await bot.help_action(update, context)
            else:
                await bot.cancel(update, context)
            assert "awaiting_sender" not in context.chat_data

    asyncio.run(exercise())


def test_simple_view_preserves_risk_and_action():
    verdict = Verdict("scam", "scam", True, "high", [], "A long explanation", "Call your family.")
    reply = build_simple_reply(verdict, "en")
    assert reply.alert_family and "Warning" in reply.text and "Call your family." in reply.text
    assert "A long explanation" not in reply.text


def test_simple_and_voice_buttons_use_the_selected_answer(flow):
    bot, update, context = flow

    async def exercise():
        await bot.on_text(update, context)
        record = action(update, context, "simple")
        await bot.help_action(update, context)
        assert update.message.reply_text.await_args.args[0] == build_simple_reply(record["verdict"], "en").text
        action(update, context, "voice")
        await bot.help_action(update, context)
        bot._send_audio.assert_awaited_with(update.message, record["reply"].speech, "en")

    asyncio.run(exercise())


def test_family_button_explains_setup_without_linked_family(flow):
    bot, update, context = flow

    async def exercise():
        await bot.on_text(update, context)
        action(update, context, "family")
        await bot.help_action(update, context)

    asyncio.run(exercise())
    assert "/guardian" in update.message.reply_text.await_args.args[0]
    context.bot.send_message.assert_not_awaited()


def test_family_request_is_explicit_and_deduplicated(flow):
    bot, update, context = flow
    bot.state.family.return_value = [101, 102]

    async def exercise():
        await bot.on_text(update, context)
        context.bot.send_message.assert_not_awaited()
        action(update, context, "family")
        await bot.help_action(update, context)
        await bot.help_action(update, context)

    asyncio.run(exercise())
    assert context.bot.send_message.await_count == 2
    text = context.bot.send_message.await_args.args[1]
    assert "asks you to help" in text and "likely scam" not in text


def test_failed_family_delivery_is_retryable_without_resending_successes(flow):
    bot, update, context = flow
    bot.state.family.return_value = [101, 102]
    context.bot.send_message.side_effect = [None, RuntimeError("offline"), None]

    async def exercise():
        await bot.on_text(update, context)
        action(update, context, "family")
        await bot.help_action(update, context)
        assert "could not be delivered" in update.message.reply_text.await_args.args[0]
        await bot.help_action(update, context)

    asyncio.run(exercise())
    assert [call.args[0] for call in context.bot.send_message.await_args_list] == [101, 102, 102]


def test_expired_or_other_chat_buttons_cannot_share_a_message(flow):
    bot, update, context = flow
    bot.state.family.return_value = [101]

    async def exercise():
        await bot.on_text(update, context)
        record = action(update, context, "family")
        record["created"] -= ANSWER_TTL
        await bot.help_action(update, context)
        assert "expired" in update.message.reply_text.await_args.args[0]
        context.chat_data = {}
        await bot.help_action(update, context)

    asyncio.run(exercise())
    context.bot.send_message.assert_not_awaited()


def test_history_has_a_size_bound():
    data = {"answers": {str(i): {"created": float("inf")} for i in range(MAX_ANSWERS + 3)}}
    assert len(recent_answers(data)) == MAX_ANSWERS
    assert "0" not in data["answers"]


def test_safety_warning_and_alert_precede_audio_without_inference(flow):
    bot, update, context = flow
    bot.state.family.return_value = [101]
    events = []
    update.message.reply_text.side_effect = lambda *a, **k: events.append("text")
    context.bot.send_message.side_effect = lambda *a, **k: events.append("family")
    bot._send_audio.side_effect = lambda *a, **k: events.append("audio")
    asyncio.run(bot._answer(update, context, "unknown", "Share your O T P"))
    assert events == ["text", "family", "audio"]
    bot.explainer.explain.assert_not_called()


def test_group_callback_cannot_send_private_family_information(flow):
    bot, update, context = flow
    update.effective_chat.type = "group"
    update.callback_query.data = "help:family:abcdef123456"
    asyncio.run(bot.help_action(update, context))
    context.bot.send_message.assert_not_awaited()
    assert update.callback_query.answer.await_args.kwargs["show_alert"]


def test_photo_ocr_failure_offers_help_instead_of_crashing(flow, monkeypatch):
    bot, update, context = flow
    photo = SimpleNamespace(get_file=AsyncMock(return_value=SimpleNamespace(download_to_drive=AsyncMock())))
    update.message.photo = [photo]
    update.message.caption = None
    monkeypatch.setattr("explain_assistant.ocr.image_to_text", Mock(side_effect=RuntimeError("unreadable")))
    asyncio.run(bot.on_photo(update, context))
    assert "couldn't check" in update.message.reply_text.await_args.args[0]
    assert "reply_markup" in update.message.reply_text.await_args.kwargs
    bot.explainer.explain.assert_not_called()


def test_audio_failure_keeps_text_and_offers_retry(flow, monkeypatch):
    bot, update, context = flow
    monkeypatch.setattr("explain_assistant.bot.synthesize", Mock(side_effect=RuntimeError("missing voice")))
    asyncio.run(SarathiAIBot._send_audio(bot, update.message, "A normal notice", "en"))
    assert "Read aloud to retry" in update.message.reply_text.await_args.args[0]


def test_bot_remains_protective_without_a_loaded_model(flow):
    bot, update, context = flow
    bot.explainer = None

    async def exercise():
        await bot._answer(update, context, "unknown", "A normal notice")
        assert "couldn't check" in update.message.reply_text.await_args.args[0]
        await bot._answer(update, context, "unknown", "Share your OTP")
        assert "Warning" in update.message.reply_text.await_args.args[0]

    asyncio.run(exercise())
