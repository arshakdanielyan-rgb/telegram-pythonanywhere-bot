from unittest.mock import MagicMock, patch


def _ai_reply(content):
    """Build a fake OpenAI chat completion whose message content is `content`."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


def _wiki_response(status=200, payload=None):
    """Build a fake requests.Response for a Wikipedia/Commons call."""
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    return r


# ── _extract_names ──────────────────────────────────────────────────────────


def test_extract_names_parses_json_array():
    with patch("bot.images.ai") as mock_ai:
        mock_ai.chat.completions.create.return_value = _ai_reply(
            '["John Cena", "Aleksandr Karelin"]'
        )
        from bot.images import _extract_names

        assert _extract_names("cena vs karelin") == ["John Cena", "Aleksandr Karelin"]


def test_extract_names_empty_for_general_question():
    with patch("bot.images.ai") as mock_ai:
        mock_ai.chat.completions.create.return_value = _ai_reply("[]")
        from bot.images import _extract_names

        assert _extract_names("what is a takedown?") == []


def test_extract_names_dedupes_and_caps():
    with patch("bot.images.ai") as mock_ai:
        mock_ai.chat.completions.create.return_value = _ai_reply(
            '["A", "a", "B", "C", "D"]'
        )
        from bot.images import _extract_names

        # case-insensitive dedupe ("A"/"a") + cap at _MAX_IMAGES (3)
        assert _extract_names("x") == ["A", "B", "C"]


def test_extract_names_survives_junk_and_errors():
    from bot.images import _extract_names

    with patch("bot.images.ai") as mock_ai:
        mock_ai.chat.completions.create.return_value = _ai_reply("not json at all")
        assert _extract_names("x") == []
    with patch("bot.images.ai") as mock_ai:
        mock_ai.chat.completions.create.side_effect = RuntimeError("api down")
        assert _extract_names("x") == []


# ── _fetch_wiki ─────────────────────────────────────────────────────────────


def test_fetch_wiki_returns_title_and_image_for_wrestler():
    payload = {
        "type": "standard",
        "title": "John Cena",
        "description": "American professional wrestler",
        "extract": "John Cena is a professional wrestler and actor.",
        "thumbnail": {"source": "https://upload.wikimedia.org/cena.jpg"},
    }
    with patch("bot.images.requests.get", return_value=_wiki_response(200, payload)):
        from bot.images import _fetch_wiki

        info = _fetch_wiki("John Cena")
        assert info == {
            "title": "John Cena",
            "image": "https://upload.wikimedia.org/cena.jpg",
        }


def test_fetch_wiki_rejects_non_wrestler_namesake():
    payload = {
        "type": "standard",
        "title": "John Smith",
        "description": "British politician",
        "extract": "John Smith was a Labour Party leader.",
        "thumbnail": {"source": "https://upload.wikimedia.org/smith.jpg"},
    }
    with patch("bot.images.requests.get", return_value=_wiki_response(200, payload)):
        from bot.images import _fetch_wiki

        # No "wrestl" in description/extract → not this person's photo.
        assert _fetch_wiki("John Smith") is None


def test_fetch_wiki_retries_disambiguation_with_qualifier():
    disambig = {"type": "disambiguation", "title": "Ali Aliyev"}
    wrestler = {
        "type": "standard",
        "title": "Ali Aliyev",
        "description": "Soviet wrestler",
        "extract": "Ali Aliyev was a Soviet freestyle wrestler.",
        "originalimage": {"source": "https://upload.wikimedia.org/aliyev.jpg"},
    }
    with patch(
        "bot.images.requests.get",
        side_effect=[_wiki_response(200, disambig), _wiki_response(200, wrestler)],
    ) as mock_get:
        from bot.images import _fetch_wiki

        info = _fetch_wiki("Ali Aliyev")
        assert info["image"] == "https://upload.wikimedia.org/aliyev.jpg"
        # Second lookup used the "(wrestler)" qualifier.
        assert "wrestler" in mock_get.call_args_list[1].args[0]


def test_fetch_wiki_returns_none_image_when_article_has_no_photo():
    payload = {
        "type": "standard",
        "title": "Obscure Wrestler",
        "description": "amateur wrestler",
        "extract": "Obscure Wrestler competed in freestyle wrestling.",
    }
    with patch("bot.images.requests.get", return_value=_wiki_response(200, payload)):
        from bot.images import _fetch_wiki

        assert _fetch_wiki("Obscure Wrestler") == {
            "title": "Obscure Wrestler",
            "image": None,
        }


# ── _fetch_commons_image ────────────────────────────────────────────────────


def test_commons_image_returns_surname_matched_photo():
    payload = {
        "query": {
            "pages": {
                "1": {
                    "index": 1,
                    "title": "File:Obscure Wrestler 2019.jpg",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/commons/ow.jpg",
                            "mime": "image/jpeg",
                        }
                    ],
                }
            }
        }
    }
    with patch("bot.images.requests.get", return_value=_wiki_response(200, payload)):
        from bot.images import _fetch_commons_image

        url = _fetch_commons_image("Obscure Wrestler", "wrestler")
        assert url == "https://upload.wikimedia.org/commons/ow.jpg"


def test_commons_image_skips_wrong_person_and_non_images():
    payload = {
        "query": {
            "pages": {
                # svg logo — wrong mime, skipped even though surname matches
                "1": {
                    "index": 1,
                    "title": "File:Karelin logo.svg",
                    "imageinfo": [
                        {"url": "https://x/logo.svg", "mime": "image/svg+xml"}
                    ],
                },
                # a raster image but of a different person — surname mismatch
                "2": {
                    "index": 2,
                    "title": "File:Someone Else.jpg",
                    "imageinfo": [
                        {"url": "https://x/else.jpg", "mime": "image/jpeg"}
                    ],
                },
            }
        }
    }
    with patch("bot.images.requests.get", return_value=_wiki_response(200, payload)):
        from bot.images import _fetch_commons_image

        assert _fetch_commons_image("Aleksandr Karelin", "karelin") is None


def test_commons_image_returns_none_on_http_error():
    with patch("bot.images.requests.get", return_value=_wiki_response(500, {})):
        from bot.images import _fetch_commons_image

        assert _fetch_commons_image("x", "x") is None


# ── _resolve_image (Wikipedia → Commons fallback) ───────────────────────────


def test_resolve_image_prefers_wikipedia_photo():
    with patch(
        "bot.images._fetch_wiki",
        return_value={"title": "John Cena", "image": "https://x/cena.jpg"},
    ):
        from bot.images import _resolve_image

        assert _resolve_image("John Cena") == ("https://x/cena.jpg", "John Cena")


def test_resolve_image_falls_back_to_commons():
    with (
        patch(
            "bot.images._fetch_wiki",
            return_value={"title": "Ali Aliyev (wrestler)", "image": None},
        ),
        patch(
            "bot.images._fetch_commons_image", return_value="https://x/aliyev.jpg"
        ) as mock_commons,
    ):
        from bot.images import _resolve_image

        url, title = _resolve_image("Ali Aliyev")
        assert url == "https://x/aliyev.jpg"
        assert title == "Ali Aliyev (wrestler)"
        # Commons searched by the qualifier-stripped title, verified by surname.
        mock_commons.assert_called_once_with("Ali Aliyev", "aliyev")


def test_resolve_image_none_when_no_article():
    with patch("bot.images._fetch_wiki", return_value=None):
        from bot.images import _resolve_image

        assert _resolve_image("Not A Wrestler") is None


def test_resolve_image_none_when_no_image_anywhere():
    with (
        patch(
            "bot.images._fetch_wiki",
            return_value={"title": "Obscure Wrestler", "image": None},
        ),
        patch("bot.images._fetch_commons_image", return_value=None),
    ):
        from bot.images import _resolve_image

        assert _resolve_image("Obscure Wrestler") is None


# ── send_wrestler_images ────────────────────────────────────────────────────


def _msg():
    m = MagicMock()
    m.chat.id = 456
    m.from_user.id = 123
    return m


def test_send_wrestler_images_sends_photo_and_notes_missing():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", return_value=["John Cena", "Ghost Wrestler"]),
        patch(
            "bot.images._resolve_image",
            side_effect=[("https://x/cena.jpg", "John Cena"), None],
        ),
        patch("bot.images.get_language", return_value="en"),
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        send_wrestler_images(_msg(), "cena and ghost")
        mock_bot.send_photo.assert_called_once_with(
            456, "https://x/cena.jpg", caption="John Cena"
        )
        # The one wrestler with no verified image gets a localized note.
        note = mock_bot.send_message.call_args[0][1]
        assert "Ghost Wrestler" in note


def test_send_wrestler_images_disabled_is_noop():
    with (
        patch("bot.images.WRESTLER_IMAGES", False),
        patch("bot.images._extract_names") as mock_extract,
        patch("bot.images.bot") as mock_bot,
    ):
        from bot.images import send_wrestler_images

        send_wrestler_images(_msg(), "John Cena")
        mock_extract.assert_not_called()
        mock_bot.send_photo.assert_not_called()


def test_send_wrestler_images_never_raises():
    with (
        patch("bot.images.WRESTLER_IMAGES", True),
        patch("bot.images._extract_names", side_effect=RuntimeError("boom")),
        patch("bot.images.bot"),
    ):
        from bot.images import send_wrestler_images

        # Must swallow the error — enrichment can't break the text reply path.
        send_wrestler_images(_msg(), "anything")
