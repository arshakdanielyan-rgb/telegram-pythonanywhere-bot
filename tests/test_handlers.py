from unittest.mock import patch, MagicMock


def make_message(text="hello", user_id=123, chat_id=456, chat_type="private"):
    msg = MagicMock()
    msg.text = text
    msg.from_user.id = user_id
    msg.chat.id = chat_id
    msg.chat.type = chat_type
    msg.reply_to_message = None
    return msg


HANDLER_PATCHES = {
    "bot.handlers.should_respond": True,
    "bot.handlers.is_rate_limited": False,
    "bot.handlers.BOT_INFO": MagicMock(id=42, username="testbot"),
}


def test_handle_message_streams_reply():
    """handle_message opens a stream for the user's message and pipes the
    deltas into stream_reply for live editing."""
    sentinel = object()  # stands in for the ask_ai_stream generator
    with (
        patch("bot.handlers.should_respond", return_value=True),
        patch("bot.handlers.is_rate_limited", return_value=False),
        patch("bot.handlers.BOT_INFO", MagicMock(username="testbot")),
        patch("bot.handlers.ask_ai_stream", return_value=sentinel) as mock_stream,
        patch("bot.handlers.stream_reply", return_value="AI reply") as mock_send,
        patch("bot.handlers.send_wrestler_images") as mock_images,
        patch("bot.handlers.bot"),
    ):
        from bot.handlers import handle_message

        msg = make_message(text="hello")
        handle_message(msg)
        mock_stream.assert_called_once_with(123, "hello")
        mock_send.assert_called_once_with(msg, sentinel)
        # After the text reply, the wrestler-image enrichment runs on the text.
        mock_images.assert_called_once_with(msg, "hello")


def test_handle_message_skips_when_not_responding():
    with (
        patch("bot.handlers.should_respond", return_value=False),
        patch("bot.handlers.ask_ai_stream") as mock_ask,
    ):
        from bot.handlers import handle_message

        handle_message(make_message())
        mock_ask.assert_not_called()


def test_handle_message_rate_limited():
    with (
        patch("bot.handlers.should_respond", return_value=True),
        patch("bot.handlers.is_rate_limited", return_value=True),
        patch("bot.handlers.BOT_INFO", MagicMock(username="testbot")),
        patch("bot.handlers.ask_ai_stream") as mock_ask,
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import handle_message

        handle_message(make_message())
        mock_ask.assert_not_called()
        mock_bot.send_message.assert_called_once()
        assert "daily limit" in mock_bot.send_message.call_args[0][1]


def test_handle_message_sends_generic_error():
    with (
        patch("bot.handlers.should_respond", return_value=True),
        patch("bot.handlers.is_rate_limited", return_value=False),
        patch("bot.handlers.BOT_INFO", MagicMock(username="testbot")),
        patch("bot.handlers.ask_ai_stream"),
        patch(
            "bot.handlers.stream_reply", side_effect=Exception("API key invalid")
        ),
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import handle_message

        handle_message(make_message())
        error_msg = mock_bot.send_message.call_args[0][1]
        assert "Something went wrong" in error_msg
        assert "API key" not in error_msg


def test_handle_message_none_text_skipped():
    """Stickers/photos/edits arriving with text=None must NOT call ask_ai
    (would burn rate limit and AI quota for no reason)."""
    with (
        patch("bot.handlers.should_respond", return_value=True),
        patch("bot.handlers.is_rate_limited", return_value=False),
        patch("bot.handlers.BOT_INFO", MagicMock(username="testbot")),
        patch("bot.handlers.ask_ai_stream") as mock_ask,
        patch("bot.handlers.stream_reply") as mock_send,
        patch("bot.handlers.bot"),
    ):
        from bot.handlers import handle_message

        msg = make_message()
        msg.text = None
        handle_message(msg)
        mock_ask.assert_not_called()
        mock_send.assert_not_called()


def test_handle_message_mention_only_skipped():
    """In a group, '@testbot' alone strips to empty — don't call ask_ai."""
    with (
        patch("bot.handlers.should_respond", return_value=True),
        patch("bot.handlers.is_rate_limited", return_value=False),
        patch("bot.handlers.BOT_INFO", MagicMock(username="testbot")),
        patch("bot.handlers.ask_ai_stream") as mock_ask,
        patch("bot.handlers.stream_reply"),
        patch("bot.handlers.bot"),
    ):
        from bot.handlers import handle_message

        msg = make_message(text="@testbot")
        handle_message(msg)
        mock_ask.assert_not_called()


# ── /about ────────────────────────────────────────────────────────────────────


def test_cmd_about_answers_via_ai():
    """/about now responds through the AI, streamed live: it opens an
    ask_ai_stream for the requesting user and pipes it into stream_reply."""
    sentinel = object()
    with (
        patch("bot.handlers.ask_ai_stream", return_value=sentinel) as mock_ask,
        patch("bot.handlers.stream_reply", return_value="I'm a wrestling bot.") as mock_send,
        patch("bot.handlers.bot"),
    ):
        from bot.handlers import cmd_about

        msg = make_message()
        cmd_about(msg)
        # AI is asked, on behalf of the requesting user, for an introduction.
        assert mock_ask.call_count == 1
        assert mock_ask.call_args[0][0] == 123
        mock_send.assert_called_once_with(msg, sentinel)


def test_cmd_about_sends_generic_error_on_failure():
    """If the AI call raises, /about must not crash the worker — it sends a
    generic error message instead."""
    with (
        patch("bot.handlers.ask_ai_stream"),
        patch("bot.handlers.stream_reply", side_effect=RuntimeError("boom")),
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import cmd_about

        cmd_about(make_message())
        sent = mock_bot.send_message.call_args[0][1]
        assert "went wrong" in sent.lower()


# ── /quote, /fact (AI canned-prompt commands) ──────────────────────


def _assert_streams_prompt_keyword(handler_name, keyword):
    """A canned-prompt command should stream an AI reply whose prompt mentions
    `keyword`, for the requesting user."""
    sentinel = object()
    with (
        patch("bot.handlers.ask_ai_stream", return_value=sentinel) as mock_ask,
        patch("bot.handlers.stream_reply", return_value="reply") as mock_send,
        patch("bot.handlers.bot"),
    ):
        import bot.handlers

        handler = getattr(bot.handlers, handler_name)
        msg = make_message()
        handler(msg)
        assert mock_ask.call_args[0][0] == 123
        assert keyword in mock_ask.call_args[0][1].lower()
        mock_send.assert_called_once_with(msg, sentinel)


def test_cmd_quote_streams_quote():
    _assert_streams_prompt_keyword("cmd_quote", "quote")


def test_cmd_fact_streams_fact():
    _assert_streams_prompt_keyword("cmd_fact", "fact")


def test_cmd_quote_survives_ai_error():
    """A failure in the AI stream must send a generic error, not crash."""
    with (
        patch("bot.handlers.ask_ai_stream"),
        patch("bot.handlers.stream_reply", side_effect=Exception("boom")),
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import cmd_quote

        cmd_quote(make_message())
        assert "went wrong" in mock_bot.send_message.call_args[0][1].lower()


# ── /predictor ──────────────────────────────────────────────────────


def test_cmd_predictor_streams_prediction():
    """/predictor <names> streams an AI prediction whose prompt carries the
    matchup, on behalf of the requesting user."""
    sentinel = object()
    with (
        patch("bot.handlers.ask_ai_stream", return_value=sentinel) as mock_ask,
        patch("bot.handlers.stream_reply", return_value="reply") as mock_send,
        patch("bot.handlers.send_wrestler_images") as mock_images,
        patch("bot.handlers.bot"),
    ):
        from bot.handlers import cmd_predictor

        msg = make_message(text="/predictor John Cena vs Brock Lesnar")
        cmd_predictor(msg)
        assert mock_ask.call_args[0][0] == 123
        assert "John Cena vs Brock Lesnar" in mock_ask.call_args[0][1]
        mock_send.assert_called_once_with(msg, sentinel)
        # Photos are requested for the wrestlers in the matchup text.
        mock_images.assert_called_once_with(msg, "John Cena vs Brock Lesnar")


def test_cmd_predictor_usage_hint_when_no_names():
    """A bare /predictor shows the usage hint and never calls the AI."""
    with (
        patch("bot.handlers.ask_ai_stream") as mock_ask,
        patch("bot.handlers.stream_reply") as mock_send,
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import cmd_predictor

        cmd_predictor(make_message(text="/predictor"))
        mock_ask.assert_not_called()
        mock_send.assert_not_called()
        assert "two wrestlers" in mock_bot.send_message.call_args[0][1].lower()


def test_cmd_predictor_survives_ai_error():
    """A failure in the AI stream must send a generic error, not crash."""
    with (
        patch("bot.handlers.ask_ai_stream"),
        patch("bot.handlers.stream_reply", side_effect=Exception("boom")),
        patch("bot.handlers.send_wrestler_images"),
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import cmd_predictor

        cmd_predictor(make_message(text="/predictor Alice vs Bob"))
        assert "went wrong" in mock_bot.send_message.call_args[0][1].lower()


# ── /language command + inline-button callback ──────────────────────────────────


def make_call(data="lang:hy", user_id=123, chat_id=456, message_id=789):
    call = MagicMock()
    call.data = data
    call.id = "callback-id"
    call.from_user.id = user_id
    call.message.chat.id = chat_id
    call.message.message_id = message_id
    return call


def test_cmd_language_sends_menu_with_inline_keyboard():
    with (
        patch("bot.handlers.get_language", return_value="en"),
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import cmd_language

        cmd_language(make_message(text="/language"))
        assert mock_bot.send_message.call_count == 1
        # Menu text is localized and an inline keyboard is attached.
        assert "Choose your language" in mock_bot.send_message.call_args[0][1]
        assert "reply_markup" in mock_bot.send_message.call_args[1]


def test_on_language_selected_saves_and_confirms_in_new_language():
    with (
        patch("bot.handlers.set_language", return_value=True) as mock_set,
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import on_language_selected

        on_language_selected(make_call("lang:hy"))
        mock_set.assert_called_once_with(123, "hy")
        # Confirmation edited into the menu message, in the chosen language.
        assert mock_bot.edit_message_text.call_args[0][0] == "✅ Լեզուն փոխվեց հայերենի։"
        mock_bot.answer_callback_query.assert_called_once()


def test_on_language_selected_ignores_unknown_code():
    with (
        patch("bot.handlers.set_language") as mock_set,
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import on_language_selected

        on_language_selected(make_call("lang:xx"))
        mock_set.assert_not_called()
        mock_bot.edit_message_text.assert_not_called()
        mock_bot.answer_callback_query.assert_called_once()


def test_on_language_selected_falls_back_to_send_when_edit_fails():
    with (
        patch("bot.handlers.set_language", return_value=True),
        patch("bot.handlers.bot") as mock_bot,
    ):
        mock_bot.edit_message_text.side_effect = Exception("message too old")
        from bot.handlers import on_language_selected

        on_language_selected(make_call("lang:ru"))
        assert mock_bot.send_message.call_args[0][1] == "✅ Язык был изменён на русский."


def test_interface_string_uses_selected_language():
    """Existing commands must render in the user's chosen language."""
    with (
        patch("bot.handlers.clear_history"),
        patch("bot.handlers.get_language", return_value="ru"),
        patch("bot.handlers.bot") as mock_bot,
    ):
        from bot.handlers import cmd_reset

        cmd_reset(make_message())
        sent = mock_bot.send_message.call_args[0][1]
        assert sent == "🧹 История разговора очищена. Начинаем заново!"


# ── /sha ─────────────────────────────────────────────────────────────────────


def test_cmd_sha_reports_live_commit_sha():
    with (
        patch("bot.handlers.bot") as mock_bot,
        patch("bot.handlers.COMMIT_SHA", "abc1234"),
    ):
        from bot.handlers import cmd_sha

        cmd_sha(make_message())
        mock_bot.send_message.assert_called_once_with(456, "Live SHA: abc1234")


def test_cmd_sha_reports_unknown_when_git_sha_unavailable():
    with (
        patch("bot.handlers.bot") as mock_bot,
        patch("bot.handlers.COMMIT_SHA", ""),
    ):
        from bot.handlers import cmd_sha

        cmd_sha(make_message())
        mock_bot.send_message.assert_called_once_with(456, "Live SHA: unknown")


# ── /model command ────────────────────────────────────────────────────────────


def _import_cmd_model_with_hf_enabled():
    """Re-import handlers module with HF_SPACE_ID set so cmd_model exists."""
    import importlib
    import bot.config
    import bot.handlers

    original = bot.config.HF_SPACE_ID
    bot.config.HF_SPACE_ID = "fake/space"
    # Also patch the import in handlers module (already imported via `from ... import HF_SPACE_ID`)
    bot.handlers.HF_SPACE_ID = "fake/space"
    importlib.reload(bot.handlers)
    cmd_model = getattr(bot.handlers, "cmd_model", None)
    # Restore
    bot.config.HF_SPACE_ID = original
    bot.handlers.HF_SPACE_ID = original
    return cmd_model


def test_cmd_model_no_args_shows_current():
    cmd_model = _import_cmd_model_with_hf_enabled()
    assert cmd_model is not None
    with (
        patch("bot.handlers.get_provider", return_value="main"),
        patch("bot.handlers.bot") as mock_bot,
    ):
        msg = make_message(text="/model")
        cmd_model(msg)
        sent = mock_bot.send_message.call_args[0][1]
        assert "Current provider: main" in sent
        assert "/model main" in sent
        assert "/model hf" in sent


def test_cmd_model_switch_to_hf():
    cmd_model = _import_cmd_model_with_hf_enabled()
    with (
        patch("bot.handlers.set_provider", return_value=True) as mock_set,
        patch("bot.handlers.bot") as mock_bot,
    ):
        msg = make_message(text="/model hf")
        cmd_model(msg)
        mock_set.assert_called_once_with(123, "hf")
        sent = mock_bot.send_message.call_args[0][1]
        assert "hf" in sent
        assert "Armenian" in sent


def test_cmd_model_switch_to_main():
    cmd_model = _import_cmd_model_with_hf_enabled()
    with (
        patch("bot.handlers.set_provider", return_value=True) as mock_set,
        patch("bot.handlers.bot") as mock_bot,
    ):
        msg = make_message(text="/model main")
        cmd_model(msg)
        mock_set.assert_called_once_with(123, "main")
        sent = mock_bot.send_message.call_args[0][1]
        assert "main" in sent.lower()


def test_cmd_model_invalid_choice():
    cmd_model = _import_cmd_model_with_hf_enabled()
    with (
        patch("bot.handlers.set_provider") as mock_set,
        patch("bot.handlers.bot") as mock_bot,
    ):
        msg = make_message(text="/model bogus")
        cmd_model(msg)
        mock_set.assert_not_called()
        assert "Invalid" in mock_bot.send_message.call_args[0][1]


def test_cmd_model_redis_error_reports_failure():
    cmd_model = _import_cmd_model_with_hf_enabled()
    with (
        patch("bot.handlers.set_provider", return_value=False),
        patch("bot.handlers.bot") as mock_bot,
    ):
        msg = make_message(text="/model hf")
        cmd_model(msg)
        assert "Could not save" in mock_bot.send_message.call_args[0][1]


def test_cmd_model_not_registered_without_hf_space_id():
    """When HF_SPACE_ID is empty, cmd_model should not exist."""
    import importlib
    import bot.config
    import bot.handlers

    bot.config.HF_SPACE_ID = ""
    bot.handlers.HF_SPACE_ID = ""
    # reload() doesn't delete existing attributes, so clear it first
    if hasattr(bot.handlers, "cmd_model"):
        delattr(bot.handlers, "cmd_model")
    importlib.reload(bot.handlers)
    assert not hasattr(bot.handlers, "cmd_model")


def test_handle_message_streams_after_rate_limit_check():
    """The stream must open only after should_respond + rate-limit pass, so
    the live-editing send never fires for a blocked message."""
    call_order = []
    with (
        patch("bot.handlers.should_respond", return_value=True),
        patch(
            "bot.handlers.is_rate_limited",
            side_effect=lambda uid: call_order.append("rate") or False,
        ),
        patch("bot.handlers.BOT_INFO", MagicMock(username="testbot")),
        patch(
            "bot.handlers.ask_ai_stream",
            side_effect=lambda *a: call_order.append("stream"),
        ),
        patch("bot.handlers.stream_reply", return_value="reply"),
        patch("bot.handlers.send_wrestler_images"),
        patch("bot.handlers.bot"),
    ):
        from bot.handlers import handle_message

        handle_message(make_message())
        assert call_order == ["rate", "stream"]
