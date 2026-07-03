from unittest.mock import MagicMock, patch


def _ai_returning(content):
    """A fake OpenAI client whose completion returns `content`."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    ai = MagicMock()
    ai.chat.completions.create.return_value = resp
    return ai


def _wiki_response(status=200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    return r


# ── _extract_names ──────────────────────────────────────────────────


def test_extract_names_parses_json_array():
    with patch("bot.images.ai", _ai_returning('["John Cena", "Brock Lesnar"]')):
        from bot.images import _extract_names

        assert _extract_names("cena or lesnar?") == ["John Cena", "Brock Lesnar"]


def test_extract_names_finds_array_inside_prose():
    with patch("bot.images.ai", _ai_returning('Sure: ["John Cena"] is who.')):
        from bot.images import _extract_names

        assert _extract_names("tell me about cena") == ["John Cena"]


def test_extract_names_dedupes_case_insensitively_and_caps():
    with patch("bot.images.ai", _ai_returning('["A","a","B","C","D"]')):
        from bot.images import _extract_names

        # de-duped (A==a) then capped at _MAX_IMAGES (3)
        assert _extract_names("x") == ["A", "B", "C"]


def test_extract_names_empty_on_non_json():
    with patch("bot.images.ai", _ai_returning("no wrestlers here")):
        from bot.images import _extract_names

        assert _extract_names("what is a takedown?") == []


def test_extract_names_empty_on_api_error():
    ai = MagicMock()
    ai.chat.completions.create.side_effect = Exception("boom")
    with patch("bot.images.ai", ai):
        from bot.images import _extract_names

        assert _extract_names("x") == []


# ── _fetch_image ────────────────────────────────────────────────────


def test_fetch_image_returns_url_and_title():
    data = {
        "type": "standard",
        "description": "American professional wrestler",
        "extract": "John Cena is a wrestler and actor.",
        "title": "John Cena",
        "thumbnail": {"source": "https://upload.wikimedia.org/cena.jpg"},
    }
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, data)
        from bot.images import _fetch_image

        assert _fetch_image("John Cena") == (
            "https://upload.wikimedia.org/cena.jpg",
            "John Cena",
        )


def test_fetch_image_falls_back_to_originalimage():
    data = {
        "type": "standard",
        "description": "Olympic wrestler",
        "extract": "won gold in wrestling",
        "title": "Aleksandr Karelin",
        "originalimage": {"source": "https://upload.wikimedia.org/karelin.jpg"},
    }
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, data)
        from bot.images import _fetch_image

        assert _fetch_image("Aleksandr Karelin")[0].endswith("karelin.jpg")


def test_fetch_image_skips_disambiguation():
    data = {
        "type": "disambiguation",
        "description": "wrestler",
        "thumbnail": {"source": "x"},
    }
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, data)
        from bot.images import _fetch_image

        assert _fetch_image("Ambiguous") is None


def test_fetch_image_skips_non_wrestling_namesake():
    data = {
        "type": "standard",
        "description": "American singer",
        "extract": "a pop musician",
        "thumbnail": {"source": "x"},
    }
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, data)
        from bot.images import _fetch_image

        assert _fetch_image("Some Singer") is None


def test_fetch_image_skips_when_no_image():
    data = {"type": "standard", "description": "wrestler", "extract": "wrestling"}
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, data)
        from bot.images import _fetch_image

        assert _fetch_image("No Photo") is None


def test_fetch_image_none_on_404():
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(404, {})
        from bot.images import _fetch_image

        assert _fetch_image("Nobody") is None


def test_fetch_image_none_on_network_error():
    with patch("bot.images.requests") as req:
        req.get.side_effect = Exception("timeout")
        from bot.images import _fetch_image

        assert _fetch_image("X") is None


# ── send_wrestler_images ────────────────────────────────────────────


def test_send_wrestler_images_sends_a_photo_per_name():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["John Cena", "Rey Mysterio"]),
        patch(
            "bot.images._fetch_image",
            side_effect=[("u1", "John Cena"), ("u2", "Rey Mysterio")],
        ),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        msg = MagicMock()
        msg.chat.id = 456
        send_wrestler_images(msg, "cena vs rey")
        assert mock_bot.send_photo.call_count == 2
        mock_bot.send_photo.assert_any_call(456, "u1", caption="John Cena")


def test_send_wrestler_images_disabled_is_noop():
    with (
        patch("bot.images.WRESTLER_IMAGES", False),
        patch("bot.images._extract_names") as extract,
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        send_wrestler_images(MagicMock(), "cena")
        extract.assert_not_called()
        mock_bot.send_photo.assert_not_called()


def test_send_wrestler_images_empty_text_is_noop():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names") as extract,
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        send_wrestler_images(MagicMock(), "")
        extract.assert_not_called()
        mock_bot.send_photo.assert_not_called()


def test_send_wrestler_images_skips_names_without_an_image():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["A", "B"]),
        patch("bot.images._fetch_image", side_effect=[None, ("u", "B")]),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        msg = MagicMock()
        msg.chat.id = 1
        send_wrestler_images(msg, "x")
        mock_bot.send_photo.assert_called_once_with(1, "u", caption="B")


def test_send_wrestler_images_never_raises_on_send_failure():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["A"]),
        patch("bot.images._fetch_image", return_value=("u", "A")),
        patch("bot.images.bot") as mock_bot,
    ):
        mock_bot.send_photo.side_effect = Exception("telegram rejected url")
        from bot.images import send_wrestler_images

        # Must swallow the error — image enrichment can't break the reply path.
        send_wrestler_images(MagicMock(), "x")
