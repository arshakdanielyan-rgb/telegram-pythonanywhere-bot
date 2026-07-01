from unittest.mock import patch, MagicMock


# ── _call_main retry logic ──────────────────────────────────────────────────


def test_call_main_retries_on_failure():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "hello"
    with (
        patch("bot.providers.ai") as mock_ai,
        patch("bot.providers.time.sleep") as mock_sleep,
    ):
        mock_ai.chat.completions.create.side_effect = [
            Exception("network error"),
            mock_response,
        ]
        from bot.providers import _call_main

        result = _call_main([{"role": "user", "content": "hi"}])
        assert result == "hello"
        assert mock_ai.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1)


def test_call_main_raises_after_max_retries():
    with patch("bot.providers.ai") as mock_ai, patch("bot.providers.time.sleep"):
        mock_ai.chat.completions.create.side_effect = Exception("persistent")
        from bot.providers import _call_main

        try:
            _call_main([{"role": "user", "content": "hi"}], retries=3)
            assert False, "Should have raised"
        except Exception as e:
            assert str(e) == "persistent"
        assert mock_ai.chat.completions.create.call_count == 3


def test_call_main_succeeds_first_try():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    with (
        patch("bot.providers.ai") as mock_ai,
        patch("bot.providers.time.sleep") as mock_sleep,
    ):
        mock_ai.chat.completions.create.return_value = mock_response
        from bot.providers import _call_main

        assert _call_main([{"role": "user", "content": "hi"}]) == "ok"
        mock_sleep.assert_not_called()


# ── _last_user_message ────────────────────────────────────────────────────────


def test_last_user_message_skips_system():
    from bot.providers import _last_user_message

    result = _last_user_message(
        [
            {"role": "system", "content": "you are a bot"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert result == "hi"


def test_last_user_message_returns_most_recent():
    from bot.providers import _last_user_message

    messages = []
    for i in range(5):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    # No trailing user message — most recent user is u4
    assert _last_user_message(messages) == "u4"


def test_last_user_message_empty_when_no_user_turn():
    from bot.providers import _last_user_message

    assert _last_user_message([{"role": "system", "content": "x"}]) == ""


# ── _strip_html ───────────────────────────────────────────────────────────────


def test_strip_html_removes_tags():
    from bot.providers import _strip_html

    assert _strip_html("<div>hello <b>world</b></div>") == "hello world"


def test_strip_html_preserves_text_without_tags():
    from bot.providers import _strip_html

    assert _strip_html("plain text") == "plain text"


# ── _call_hf ──────────────────────────────────────────────────────────────────


def test_call_hf_calls_gradio_client():
    mock_client = MagicMock()
    mock_client.predict.return_value = ("<p>Armenian response</p>", "done")
    with patch("bot.providers.HF_SPACE_ID", "edisimon/armgpt-demo"):
        import gradio_client

        with patch.object(
            gradio_client, "Client", return_value=mock_client
        ) as mock_cls:
            from bot.providers import _call_hf, HF_REQUEST_TIMEOUT

            result = _call_hf([{"role": "user", "content": "Բարև"}])
            mock_cls.assert_called_once_with(
                "edisimon/armgpt-demo",
                hf_token=None,
                httpx_kwargs={"timeout": HF_REQUEST_TIMEOUT},
            )
            mock_client.predict.assert_called_once()
            assert "Armenian response" in result


def test_call_hf_handles_plain_string_result():
    mock_client = MagicMock()
    mock_client.predict.return_value = "just text"
    with patch("bot.providers.HF_SPACE_ID", "fake/space"):
        import gradio_client

        with patch.object(gradio_client, "Client", return_value=mock_client):
            from bot.providers import _call_hf

            assert _call_hf([{"role": "user", "content": "hi"}]) == "just text"


def test_call_hf_no_retry_on_failure():
    mock_client = MagicMock()
    mock_client.predict.side_effect = Exception("HF down")
    with patch("bot.providers.HF_SPACE_ID", "fake/space"):
        import gradio_client

        with patch.object(gradio_client, "Client", return_value=mock_client):
            from bot.providers import _call_hf

            try:
                _call_hf([{"role": "user", "content": "hi"}])
                assert False, "Should have raised"
            except Exception as e:
                assert "HF down" in str(e)
            # Only one call — no retry
            assert mock_client.predict.call_count == 1


# ── generate dispatch ─────────────────────────────────────────────────────────


def test_generate_dispatches_to_main():
    with (
        patch("bot.providers.get_provider", return_value="main"),
        patch("bot.providers._call_main", return_value="main reply") as mock_main,
        patch("bot.providers._call_hf") as mock_hf,
    ):
        from bot.providers import generate

        assert generate(123, [{"role": "user", "content": "hi"}]) == "main reply"
        mock_main.assert_called_once()
        mock_hf.assert_not_called()


def test_generate_dispatches_to_hf():
    with (
        patch("bot.providers.get_provider", return_value="hf"),
        patch("bot.providers._call_main") as mock_main,
        patch("bot.providers._call_hf", return_value="hf reply") as mock_hf,
    ):
        from bot.providers import generate

        assert generate(123, [{"role": "user", "content": "hi"}]) == "hf reply"
        mock_hf.assert_called_once()
        mock_main.assert_not_called()


# ── _stream_main ────────────────────────────────────────────────────────────────


def _chunk(content):
    """Build a fake OpenAI streaming chunk with one choice + delta."""
    c = MagicMock()
    c.choices[0].delta.content = content
    return c


def test_stream_main_yields_content_deltas():
    with patch("bot.providers.ai") as mock_ai:
        mock_ai.chat.completions.create.return_value = iter(
            [_chunk("Hel"), _chunk("lo"), _chunk(None), _chunk("!")]
        )
        from bot.providers import _stream_main

        # None deltas (e.g. role-only opening chunk) are skipped.
        assert list(_stream_main([{"role": "user", "content": "hi"}])) == [
            "Hel",
            "lo",
            "!",
        ]
        assert mock_ai.chat.completions.create.call_args[1]["stream"] is True


def test_stream_main_retries_before_first_token():
    """A failure to START the stream retries, since nothing has been shown."""
    with (
        patch("bot.providers.ai") as mock_ai,
        patch("bot.providers.time.sleep") as mock_sleep,
    ):
        mock_ai.chat.completions.create.side_effect = [
            Exception("network error"),
            iter([_chunk("ok")]),
        ]
        from bot.providers import _stream_main

        assert list(_stream_main([{"role": "user", "content": "hi"}])) == ["ok"]
        assert mock_ai.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1)


def test_stream_main_does_not_retry_after_first_token():
    """Once a token is yielded the user has seen partial text — re-raise
    instead of retrying (a retry would duplicate the visible output)."""

    def explode_mid_stream():
        yield _chunk("par")
        raise Exception("dropped mid-stream")

    with (
        patch("bot.providers.ai") as mock_ai,
        patch("bot.providers.time.sleep"),
    ):
        mock_ai.chat.completions.create.return_value = explode_mid_stream()
        from bot.providers import _stream_main

        got = []
        try:
            for delta in _stream_main([{"role": "user", "content": "hi"}], retries=3):
                got.append(delta)
            assert False, "should have raised"
        except Exception as e:
            assert "dropped mid-stream" in str(e)
        assert got == ["par"]
        # No second attempt — the stream was already partially delivered.
        assert mock_ai.chat.completions.create.call_count == 1


# ── generate_stream dispatch ────────────────────────────────────────────────────


def test_generate_stream_dispatches_to_main():
    with (
        patch("bot.providers.get_provider", return_value="main"),
        patch("bot.providers._stream_main", return_value=iter(["a", "b"])) as mock_main,
        patch("bot.providers._call_hf") as mock_hf,
    ):
        from bot.providers import generate_stream

        assert list(generate_stream(123, [{"role": "user", "content": "hi"}])) == [
            "a",
            "b",
        ]
        mock_main.assert_called_once()
        mock_hf.assert_not_called()


def test_generate_stream_hf_yields_single_delta():
    """hf has no streaming API — it yields its whole reply as one delta."""
    with (
        patch("bot.providers.get_provider", return_value="hf"),
        patch("bot.providers._call_hf", return_value="full hf reply") as mock_hf,
        patch("bot.providers._stream_main") as mock_main,
    ):
        from bot.providers import generate_stream

        assert list(generate_stream(123, [{"role": "user", "content": "hi"}])) == [
            "full hf reply"
        ]
        mock_hf.assert_called_once()
        mock_main.assert_not_called()
