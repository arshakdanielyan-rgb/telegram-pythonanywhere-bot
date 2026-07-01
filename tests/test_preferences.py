from unittest.mock import patch


def test_get_provider_default_when_unset():
    with patch("bot.preferences.store") as mock_store:
        mock_store.get.return_value = None
        from bot.preferences import get_provider
        assert get_provider(123) == "main"


def test_get_provider_returns_saved_main():
    with patch("bot.preferences.store") as mock_store, \
         patch("bot.preferences.HF_SPACE_ID", "fake/space"):
        mock_store.get.return_value = "main"
        from bot.preferences import get_provider
        assert get_provider(123) == "main"


def test_get_provider_returns_saved_hf_when_configured():
    with patch("bot.preferences.store") as mock_store, \
         patch("bot.preferences.HF_SPACE_ID", "fake/space"):
        mock_store.get.return_value = "hf"
        from bot.preferences import get_provider
        assert get_provider(123) == "hf"


def test_get_provider_falls_back_to_default_when_hf_not_configured():
    """Saved value is 'hf' but HF_SPACE_ID is empty — fall back."""
    with patch("bot.preferences.store") as mock_store, \
         patch("bot.preferences.HF_SPACE_ID", ""):
        mock_store.get.return_value = "hf"
        from bot.preferences import get_provider
        assert get_provider(123) == "main"


def test_get_provider_ignores_invalid_value():
    with patch("bot.preferences.store") as mock_store:
        mock_store.get.return_value = "garbage"
        from bot.preferences import get_provider
        assert get_provider(123) == "main"


def test_get_provider_redis_down_returns_default():
    with patch("bot.preferences.store") as mock_store:
        mock_store.get.side_effect = Exception("connection refused")
        from bot.preferences import get_provider
        assert get_provider(123) == "main"


def test_set_provider_saves_to_redis():
    with patch("bot.preferences.store") as mock_store:
        from bot.preferences import set_provider
        assert set_provider(123, "hf") is True
        mock_store.set.assert_called_once_with("provider:123", "hf")


def test_set_provider_rejects_invalid():
    with patch("bot.preferences.store") as mock_store:
        from bot.preferences import set_provider
        assert set_provider(123, "bogus") is False
        mock_store.set.assert_not_called()


def test_set_provider_redis_down_returns_false():
    with patch("bot.preferences.store") as mock_store:
        mock_store.set.side_effect = Exception("connection refused")
        from bot.preferences import set_provider
        assert set_provider(123, "main") is False


# ── language preference ─────────────────────────────────────────────────────────


def test_get_language_default_is_english():
    with patch("bot.preferences.store", None), \
         patch.dict("bot.preferences._language_prefs", {}, clear=True):
        from bot.preferences import get_language
        assert get_language(999) == "en"


def test_set_language_works_without_a_store():
    """Core guarantee: a choice takes effect even in stateless mode."""
    with patch("bot.preferences.store", None), \
         patch.dict("bot.preferences._language_prefs", {}, clear=True):
        from bot.preferences import get_language, set_language
        assert set_language(123, "hy") is True
        assert get_language(123) == "hy"


def test_set_language_rejects_unsupported_code():
    with patch.dict("bot.preferences._language_prefs", {}, clear=True):
        from bot.preferences import set_language
        assert set_language(123, "xx") is False


def test_get_language_prefers_persisted_store_value():
    with patch("bot.preferences.store") as mock_store, \
         patch.dict("bot.preferences._language_prefs", {}, clear=True):
        mock_store.get.return_value = "ru"
        from bot.preferences import get_language
        assert get_language(123) == "ru"


def test_set_language_persists_to_store_when_available():
    with patch("bot.preferences.store") as mock_store, \
         patch.dict("bot.preferences._language_prefs", {}, clear=True):
        from bot.preferences import set_language
        assert set_language(123, "en") is True
        mock_store.set.assert_called_once_with("language:123", "en")


def test_set_language_survives_store_failure():
    with patch("bot.preferences.store") as mock_store, \
         patch.dict("bot.preferences._language_prefs", {}, clear=True):
        mock_store.set.side_effect = Exception("down")
        mock_store.get.side_effect = Exception("down")
        from bot.preferences import get_language, set_language
        assert set_language(123, "hy") is True
        assert get_language(123) == "hy"
