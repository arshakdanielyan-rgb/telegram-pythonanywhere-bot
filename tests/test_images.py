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


def test_fetch_image_retries_disambiguation_with_wrestler_qualifier():
    disambig = {
        "type": "disambiguation",
        "description": "Topics referred to by the same term",
        "extract": "Arsen Harutyunyan may refer to:",
    }
    wrestler = {
        "type": "standard",
        "description": "Armenian wrestler (born 1999)",
        "extract": "Arsen Harutyunyan is an Armenian freestyle wrestler.",
        "title": "Arsen Harutyunyan (wrestler)",
        "thumbnail": {"source": "https://upload.wikimedia.org/arsen.jpg"},
    }
    with patch("bot.images.requests") as req:
        # Bare name -> disambiguation; "(wrestler)" retry -> the athlete's page.
        req.get.side_effect = [
            _wiki_response(200, disambig),
            _wiki_response(200, wrestler),
        ]
        from bot.images import _fetch_image

        assert _fetch_image("Arsen Harutyunyan") == (
            "https://upload.wikimedia.org/arsen.jpg",
            "Arsen Harutyunyan (wrestler)",
        )
        assert req.get.call_count == 2


def test_fetch_image_no_retry_when_name_already_qualified():
    disambig = {"type": "disambiguation", "description": "same term", "extract": ""}
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, disambig)
        from bot.images import _fetch_image

        # Extractor already supplied a qualifier — don't append another.
        assert _fetch_image("Ali Aliyev (wrestler)") is None
        assert req.get.call_count == 1


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


# ── _fetch_wiki ─────────────────────────────────────────────────────


def test_fetch_wiki_returns_article_with_extract():
    data = {
        "type": "standard",
        "description": "American professional wrestler",
        "extract": "John Cena is a wrestler and actor.",
        "title": "John Cena",
        "thumbnail": {"source": "https://upload.wikimedia.org/cena.jpg"},
    }
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, data)
        from bot.images import _fetch_wiki

        info = _fetch_wiki("John Cena")
        assert info["title"] == "John Cena"
        assert info["extract"] == "John Cena is a wrestler and actor."
        assert info["image"] == "https://upload.wikimedia.org/cena.jpg"


def test_fetch_wiki_keeps_article_without_image():
    data = {"type": "standard", "description": "wrestler", "extract": "a wrestler"}
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, data)
        from bot.images import _fetch_wiki

        info = _fetch_wiki("No Photo")
        assert info is not None
        assert info["image"] is None


def test_fetch_wiki_none_on_non_wrestling_namesake():
    data = {"type": "standard", "description": "singer", "extract": "a musician"}
    with patch("bot.images.requests") as req:
        req.get.return_value = _wiki_response(200, data)
        from bot.images import _fetch_wiki

        assert _fetch_wiki("Some Singer") is None


# ── ground_wrestlers ────────────────────────────────────────────────


def test_ground_wrestlers_sends_photo_first_and_returns_grounding():
    info = {
        "title": "John Cena",
        "description": "American wrestler",
        "extract": "John Cena is a wrestler.",
        "image": "u1",
    }
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["John Cena"]),
        patch("bot.images._fetch_wiki", return_value=info),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import ground_wrestlers

        msg = MagicMock()
        msg.chat.id = 5
        grounding, missing = ground_wrestlers(msg, "tell me about cena")
        mock_bot.send_photo.assert_called_once_with(5, "u1", caption="John Cena")
        assert "John Cena is a wrestler." in grounding
        assert missing == []


def test_ground_wrestlers_notes_missing_article_in_grounding():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["Ghost Wrestler"]),
        patch("bot.images._fetch_wiki", return_value=None),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import ground_wrestlers

        grounding, missing = ground_wrestlers(MagicMock(), "about ghost wrestler")
        mock_bot.send_photo.assert_not_called()
        assert "No Wikipedia article was found" in grounding
        assert "Ghost Wrestler" in grounding
        assert missing == []


def test_ground_wrestlers_reports_article_without_photo_as_missing():
    info = {"title": "No Photo Guy", "description": "w", "extract": "x", "image": None}
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["No Photo Guy"]),
        patch("bot.images._fetch_wiki", return_value=info),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import ground_wrestlers

        grounding, missing = ground_wrestlers(MagicMock(), "x")
        mock_bot.send_photo.assert_not_called()
        assert grounding is not None
        assert missing == ["No Photo Guy"]


def test_ground_wrestlers_noop_when_disabled():
    with (
        patch("bot.images.WRESTLER_IMAGES", False),
        patch("bot.images._extract_names") as extract,
    ):
        from bot.images import ground_wrestlers

        assert ground_wrestlers(MagicMock(), "cena") == (None, [])
        extract.assert_not_called()


def test_ground_wrestlers_none_when_no_names():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=[]),
    ):
        from bot.images import ground_wrestlers

        assert ground_wrestlers(MagicMock(), "what is a takedown?") == (None, [])


# ── notify_missing_photos ───────────────────────────────────────────


def test_notify_missing_photos_sends_localized_note_without_qualifier():
    with (
        patch("bot.images.get_language", return_value="en"),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import notify_missing_photos

        msg = MagicMock()
        msg.chat.id = 8
        notify_missing_photos(msg, ["Ali Aliyev (wrestler)"])
        note = mock_bot.send_message.call_args[0][1]
        assert mock_bot.send_message.call_args[0][0] == 8
        assert "Ali Aliyev" in note
        assert "(wrestler)" not in note


def test_notify_missing_photos_noop_on_empty():
    with patch("bot.images.bot") as mock_bot:
        from bot.images import notify_missing_photos

        notify_missing_photos(MagicMock(), [])
        mock_bot.send_message.assert_not_called()


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


def test_send_wrestler_images_notes_when_no_image_found():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["Ghost Wrestler"]),
        patch("bot.images._fetch_image", return_value=None),
        patch("bot.images.get_language", return_value="en"),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        msg = MagicMock()
        msg.chat.id = 7
        send_wrestler_images(msg, "tell me about ghost wrestler")
        mock_bot.send_photo.assert_not_called()
        mock_bot.send_message.assert_called_once()
        args, _ = mock_bot.send_message.call_args
        assert args[0] == 7
        assert "Ghost Wrestler" in args[1]


def test_send_wrestler_images_note_strips_disambiguation_qualifier():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["Ali Aliyev (wrestler)"]),
        patch("bot.images._fetch_image", return_value=None),
        patch("bot.images.get_language", return_value="en"),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        msg = MagicMock()
        msg.chat.id = 3
        send_wrestler_images(msg, "who is ali aliyev the wrestler")
        note = mock_bot.send_message.call_args[0][1]
        assert "Ali Aliyev" in note
        assert "(wrestler)" not in note


def test_send_wrestler_images_no_note_when_all_found():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["John Cena"]),
        patch("bot.images._fetch_image", return_value=("u", "John Cena")),
        patch("bot.images.get_language", return_value="en"),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        msg = MagicMock()
        msg.chat.id = 9
        send_wrestler_images(msg, "tell me about cena")
        mock_bot.send_photo.assert_called_once()
        mock_bot.send_message.assert_not_called()
