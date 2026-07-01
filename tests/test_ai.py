from unittest.mock import patch


def test_ask_ai_returns_reply():
    with (
        patch("bot.ai.generate", return_value="Hello there!"),
        patch("bot.ai.get_history", return_value=[]),
        patch("bot.ai.save_history"),
    ):
        from bot.ai import ask_ai

        reply = ask_ai(123, "hi")
        assert reply == "Hello there!"


def test_ask_ai_injects_language_directive_into_system_prompt():
    """The user's chosen language must reach the model via the system prompt."""
    with (
        patch("bot.ai.generate", return_value="ok") as mock_gen,
        patch("bot.ai.get_history", return_value=[]),
        patch("bot.ai.save_history"),
        patch("bot.ai.get_language", return_value="ru"),
    ):
        from bot.ai import ask_ai

        ask_ai(123, "hi")
        system_msg = mock_gen.call_args[0][1][0]
        assert system_msg["role"] == "system"
        assert "Russian" in system_msg["content"]


def test_ask_ai_saves_history():
    with (
        patch("bot.ai.generate", return_value="reply"),
        patch("bot.ai.get_history", return_value=[]),
        patch("bot.ai.save_history") as mock_save,
    ):
        from bot.ai import ask_ai

        ask_ai(123, "hi")
        mock_save.assert_called_once()
        saved_history = mock_save.call_args[0][1]
        assert saved_history[0] == {"role": "user", "content": "hi"}
        assert saved_history[1]["role"] == "assistant"


def test_ask_ai_passes_user_id_to_generate():
    with (
        patch("bot.ai.generate", return_value="hi") as mock_gen,
        patch("bot.ai.get_history", return_value=[]),
        patch("bot.ai.save_history"),
    ):
        from bot.ai import ask_ai

        ask_ai(456, "hello")
        assert mock_gen.call_args[0][0] == 456


# ── ask_ai_stream ───────────────────────────────────────────────────────────────


def test_ask_ai_stream_yields_deltas():
    with (
        patch("bot.ai.generate_stream", return_value=iter(["Hel", "lo", "!"])),
        patch("bot.ai.get_history", return_value=[]),
        patch("bot.ai.save_history"),
    ):
        from bot.ai import ask_ai_stream

        assert list(ask_ai_stream(123, "hi")) == ["Hel", "lo", "!"]


def test_ask_ai_stream_saves_joined_reply_after_consumption():
    with (
        patch("bot.ai.generate_stream", return_value=iter(["Hel", "lo"])),
        patch("bot.ai.get_history", return_value=[]),
        patch("bot.ai.save_history") as mock_save,
    ):
        from bot.ai import ask_ai_stream

        list(ask_ai_stream(123, "hi"))  # fully consume
        saved_history = mock_save.call_args[0][1]
        assert saved_history[0] == {"role": "user", "content": "hi"}
        assert saved_history[1] == {"role": "assistant", "content": "Hello"}


def test_ask_ai_stream_does_not_save_when_not_consumed():
    """A partial reply (generator abandoned before completion) must not be
    persisted — history stays clean for the next turn."""
    with (
        patch("bot.ai.generate_stream", return_value=iter(["Hel", "lo"])),
        patch("bot.ai.get_history", return_value=[]),
        patch("bot.ai.save_history") as mock_save,
    ):
        from bot.ai import ask_ai_stream

        gen = ask_ai_stream(123, "hi")
        next(gen)  # pull one delta, then abandon
        mock_save.assert_not_called()
