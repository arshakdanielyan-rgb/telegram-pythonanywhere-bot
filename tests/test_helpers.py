from unittest.mock import patch, MagicMock


def make_message(chat_type="private", reply_from_id=None, text="hello"):
    message = MagicMock()
    message.chat.type = chat_type
    message.text = text
    message.reply_to_message = None
    if reply_from_id:
        message.reply_to_message = MagicMock()
        message.reply_to_message.from_user.id = reply_from_id
    return message


# ── send_reply ─────────────────────────────────────────────────────────────────


def test_send_reply_short_text():
    with patch("bot.helpers.bot") as mock_bot:
        from bot.helpers import send_reply

        msg = make_message()
        send_reply(msg, "Hello!")
        mock_bot.send_message.assert_called_once_with(
            msg.chat.id, "Hello!", parse_mode="Markdown"
        )


def test_send_reply_splits_long_text():
    with patch("bot.helpers.bot") as mock_bot:
        with patch("bot.helpers.MAX_MSG_LEN", 10):
            from bot.helpers import send_reply

            msg = make_message()
            send_reply(msg, "A" * 25)
            assert mock_bot.send_message.call_count == 3


def test_send_reply_falls_back_to_plain_text_on_markdown_failure():
    """If Markdown parse fails (unbalanced **/[), retry the same chunk as plain text."""
    with patch("bot.helpers.bot") as mock_bot:
        # First call (Markdown) fails; second call (plain) succeeds.
        mock_bot.send_message.side_effect = [Exception("can't parse entities"), None]
        from bot.helpers import send_reply

        msg = make_message()
        send_reply(msg, "an unbalanced * marker")
        assert mock_bot.send_message.call_count == 2
        first_kwargs = mock_bot.send_message.call_args_list[0][1]
        second_kwargs = mock_bot.send_message.call_args_list[1][1]
        assert first_kwargs.get("parse_mode") == "Markdown"
        # Retry has no parse_mode
        assert "parse_mode" not in second_kwargs


def test_send_reply_propagates_when_plain_text_also_fails():
    """If both Markdown and plain text fail, raise so the webhook can skip
    the dedupe marker and let Telegram retry instead of silently dropping."""
    with patch("bot.helpers.bot") as mock_bot:
        mock_bot.send_message.side_effect = [
            Exception("can't parse entities"),
            Exception("Telegram unavailable"),
        ]
        from bot.helpers import send_reply

        msg = make_message()
        try:
            send_reply(msg, "anything")
            raised = False
        except Exception as e:
            raised = True
            assert "Telegram unavailable" in str(e)
        assert raised, "send_reply must propagate when plain-text retry also fails"


def test_send_reply_prefers_newline_split():
    """Long text with newlines should be cut at a newline, not mid-line."""
    with patch("bot.helpers.bot") as mock_bot:
        with patch("bot.helpers.MAX_MSG_LEN", 30):
            from bot.helpers import send_reply

            msg = make_message()
            send_reply(msg, "first paragraph here\n\nsecond paragraph here")
            # Two chunks split on the paragraph break
            assert mock_bot.send_message.call_count == 2
            sent_first = mock_bot.send_message.call_args_list[0][0][1]
            sent_second = mock_bot.send_message.call_args_list[1][0][1]
            assert sent_first == "first paragraph here"
            assert sent_second == "second paragraph here"


# ── stream_reply ───────────────────────────────────────────────────────────────


def test_stream_reply_returns_full_text_and_finalizes():
    with patch("bot.helpers.bot") as mock_bot:
        from bot.helpers import stream_reply

        msg = make_message()
        result = stream_reply(msg, iter(["Hello ", "world", "!"]))
        assert result == "Hello world!"
        # A placeholder is sent first so something appears instantly.
        mock_bot.send_message.assert_called_once()
        # The final edit carries the complete text (Markdown pass).
        final = mock_bot.edit_message_text.call_args_list[-1]
        assert final[0][0] == "Hello world!"
        assert final[1].get("parse_mode") == "Markdown"


def test_stream_reply_final_edit_falls_back_to_plain():
    """If the Markdown final edit fails (unbalanced entity), retry plain."""
    with patch("bot.helpers.bot") as mock_bot:
        # Suppress intermediate edits so only the final pass runs.
        with patch("bot.helpers.STREAM_MIN_EDIT_CHARS", 10_000):
            mock_bot.edit_message_text.side_effect = [
                Exception("can't parse entities"),  # Markdown attempt
                None,  # plain retry succeeds
            ]
            from bot.helpers import stream_reply

            stream_reply(make_message(), iter(["an unbalanced * marker"]))
            assert mock_bot.edit_message_text.call_count == 2
            assert mock_bot.edit_message_text.call_args_list[0][1].get(
                "parse_mode"
            ) == "Markdown"
            assert "parse_mode" not in mock_bot.edit_message_text.call_args_list[1][1]


def test_stream_reply_splits_across_messages_on_overflow():
    """When the reply grows past one Telegram message, a new message opens."""
    with patch("bot.helpers.bot") as mock_bot:
        with patch("bot.helpers.MAX_MSG_LEN", 10):
            from bot.helpers import stream_reply

            result = stream_reply(make_message(), iter(["A" * 8, "B" * 8]))
            assert result == "A" * 8 + "B" * 8
            # Placeholder for the first message + placeholder for the overflow
            # continuation message.
            assert mock_bot.send_message.call_count == 2


def test_stream_reply_edit_error_is_non_fatal():
    """A transient edit failure mid-stream must not abort the stream."""
    with patch("bot.helpers.bot") as mock_bot:
        with patch("bot.helpers.STREAM_MIN_EDIT_CHARS", 1):
            mock_bot.edit_message_text.side_effect = Exception("429 Too Many Requests")
            from bot.helpers import stream_reply

            # Should complete and still return the full text despite edits failing.
            result = stream_reply(make_message(), iter(["a", "b", "c"]))
            assert result == "abc"


# ── should_respond ─────────────────────────────────────────────────────────────


def test_should_respond_private_chat():
    from bot.helpers import should_respond

    assert should_respond(make_message(chat_type="private")) is True


def test_should_respond_group_always_true():
    """should_respond now returns True unconditionally — bot replies to every message."""
    from bot.helpers import should_respond

    assert should_respond(make_message(chat_type="group", text="just chatting")) is True
    assert should_respond(make_message(chat_type="group", text="hey @testbot")) is True
    assert should_respond(make_message(chat_type="group", reply_from_id=99)) is True


# ── keep_typing ────────────────────────────────────────────────────────────────


def test_keep_typing_sends_typing_action():
    with (
        patch("bot.helpers.bot") as mock_bot,
        patch("bot.helpers.TYPING_REFRESH_SECONDS", 0.05),
    ):
        from bot.helpers import keep_typing

        with keep_typing(123):
            pass  # exits immediately
        # At least one typing action was sent before the context exited
        typing_calls = [
            c
            for c in mock_bot.send_chat_action.call_args_list
            if c[0] == (123, "typing")
        ]
        assert len(typing_calls) >= 1


def test_keep_typing_refreshes_while_block_runs():
    import time

    with (
        patch("bot.helpers.bot") as mock_bot,
        patch("bot.helpers.TYPING_REFRESH_SECONDS", 0.05),
    ):
        from bot.helpers import keep_typing

        with keep_typing(123):
            time.sleep(0.2)  # wait long enough for multiple refreshes
        typing_calls = [
            c
            for c in mock_bot.send_chat_action.call_args_list
            if c[0] == (123, "typing")
        ]
        assert len(typing_calls) >= 2


def test_keep_typing_stops_thread_on_exit():
    import time

    with (
        patch("bot.helpers.bot") as mock_bot,
        patch("bot.helpers.TYPING_REFRESH_SECONDS", 0.05),
    ):
        from bot.helpers import keep_typing

        with keep_typing(123):
            pass
        count_at_exit = mock_bot.send_chat_action.call_count
        time.sleep(0.15)
        # No further calls after the context exits
        assert mock_bot.send_chat_action.call_count == count_at_exit


def test_keep_typing_swallows_errors():
    """A failing typing call should not crash the generation path."""
    with (
        patch("bot.helpers.bot") as mock_bot,
        patch("bot.helpers.TYPING_REFRESH_SECONDS", 0.05),
    ):
        mock_bot.send_chat_action.side_effect = Exception("Telegram down")
        from bot.helpers import keep_typing

        # Should not raise
        with keep_typing(123):
            pass
