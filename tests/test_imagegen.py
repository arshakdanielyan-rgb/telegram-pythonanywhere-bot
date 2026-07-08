import base64
from unittest.mock import MagicMock, patch

import pytest


def _resp(status=200, json_data=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data if json_data is not None else {}
    r.text = text
    return r


def _image_response(b64_data, mime="image/png"):
    """A well-formed generateContent response carrying one inline image."""
    return {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"mimeType": mime, "data": b64_data}}]}}
        ]
    }


# ── generate_image ──────────────────────────────────────────────────


def test_generate_image_returns_bytes_and_mime():
    raw = b"\x89PNG fake bytes"
    b64 = base64.b64encode(raw).decode()
    with (
        patch("bot.imagegen.IMAGE_API_KEY", "key"),
        patch("bot.imagegen.requests") as req,
    ):
        req.post.return_value = _resp(200, _image_response(b64, "image/png"))
        from bot.imagegen import generate_image

        data, mime = generate_image("a cat")
        assert data == raw
        assert mime == "image/png"


def test_generate_image_accepts_snake_case_inline_data():
    raw = b"jpegbytes"
    b64 = base64.b64encode(raw).decode()
    snake = {
        "candidates": [
            {"content": {"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}}
        ]
    }
    with (
        patch("bot.imagegen.IMAGE_API_KEY", "key"),
        patch("bot.imagegen.requests") as req,
    ):
        req.post.return_value = _resp(200, snake)
        from bot.imagegen import generate_image

        data, mime = generate_image("a dog")
        assert data == raw
        assert mime == "image/jpeg"


def test_generate_image_sends_prompt_and_auth_header():
    b64 = base64.b64encode(b"x").decode()
    with (
        patch("bot.imagegen.IMAGE_API_KEY", "secret-key"),
        patch("bot.imagegen.requests") as req,
    ):
        req.post.return_value = _resp(200, _image_response(b64))
        from bot.imagegen import generate_image

        generate_image("neon city")
        _, kwargs = req.post.call_args
        assert kwargs["headers"]["x-goog-api-key"] == "secret-key"
        body = kwargs["json"]
        assert body["contents"][0]["parts"][0]["text"] == "neon city"
        assert "IMAGE" in body["generationConfig"]["responseModalities"]


def test_generate_image_raises_quota_on_429():
    from bot.imagegen import ImageGenError

    with (
        patch("bot.imagegen.IMAGE_API_KEY", "key"),
        patch("bot.imagegen.requests") as req,
    ):
        req.post.return_value = _resp(429, {}, "rate limited")
        from bot.imagegen import generate_image

        with pytest.raises(ImageGenError) as exc:
            generate_image("x")
        assert exc.value.key == "image.quota"


def test_generate_image_raises_failed_on_auth_error():
    from bot.imagegen import ImageGenError

    with (
        patch("bot.imagegen.IMAGE_API_KEY", "key"),
        patch("bot.imagegen.requests") as req,
    ):
        req.post.return_value = _resp(403, {}, "forbidden")
        from bot.imagegen import generate_image

        with pytest.raises(ImageGenError) as exc:
            generate_image("x")
        assert exc.value.key == "image.failed"


def test_generate_image_raises_failed_on_network_error():
    from bot.imagegen import ImageGenError

    with (
        patch("bot.imagegen.IMAGE_API_KEY", "key"),
        patch("bot.imagegen.requests") as req,
    ):
        req.post.side_effect = Exception("timeout")
        from bot.imagegen import generate_image

        with pytest.raises(ImageGenError) as exc:
            generate_image("x")
        assert exc.value.key == "image.failed"


def test_generate_image_raises_blocked_on_safety_feedback():
    from bot.imagegen import ImageGenError

    blocked = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
    with (
        patch("bot.imagegen.IMAGE_API_KEY", "key"),
        patch("bot.imagegen.requests") as req,
    ):
        req.post.return_value = _resp(200, blocked)
        from bot.imagegen import generate_image

        with pytest.raises(ImageGenError) as exc:
            generate_image("something disallowed")
        assert exc.value.key == "image.blocked"


def test_generate_image_raises_no_image_when_only_text():
    from bot.imagegen import ImageGenError

    text_only = {"candidates": [{"content": {"parts": [{"text": "here you go"}]}}]}
    with (
        patch("bot.imagegen.IMAGE_API_KEY", "key"),
        patch("bot.imagegen.requests") as req,
    ):
        req.post.return_value = _resp(200, text_only)
        from bot.imagegen import generate_image

        with pytest.raises(ImageGenError) as exc:
            generate_image("x")
        assert exc.value.key == "image.no_image"


def test_generate_image_raises_when_not_configured():
    from bot.imagegen import ImageGenError

    with patch("bot.imagegen.IMAGE_API_KEY", ""):
        from bot.imagegen import generate_image

        with pytest.raises(ImageGenError):
            generate_image("x")


def test_extract_image_returns_none_when_no_image_part():
    from bot.imagegen import _extract_image

    # An inlineData part with no "data" is not a usable image.
    empty = {"candidates": [{"content": {"parts": [{"inlineData": {}}]}}]}
    assert _extract_image(empty) is None
    assert _extract_image({}) is None
