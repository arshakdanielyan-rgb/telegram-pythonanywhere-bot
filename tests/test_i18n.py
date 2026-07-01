from unittest.mock import patch


def test_t_returns_language_specific_string():
    from bot.i18n import t

    assert "English" in t("language.changed", "en")


def test_t_armenian_confirmation_matches_spec():
    from bot.i18n import t

    assert t("language.changed", "hy") == "✅ Լեզուն փոխվեց հայերենի։"


def test_t_russian_confirmation_matches_spec():
    from bot.i18n import t

    assert t("language.changed", "ru") == "✅ Язык был изменён на русский."


def test_t_unknown_language_falls_back_to_english():
    from bot.i18n import t

    assert t("reset.done", "xx") == t("reset.done", "en")


def test_t_missing_key_in_language_falls_back_to_english():
    import bot.i18n as i18n

    # Simulate a language whose file is missing the key entirely.
    with patch.dict(i18n._TRANSLATIONS, {"hy": {}}, clear=False):
        assert i18n.t("reset.done", "hy") == i18n.t("reset.done", "en")


def test_t_unknown_key_returns_the_key():
    from bot.i18n import t

    assert t("nope.not.a.key", "en") == "nope.not.a.key"


def test_t_fills_placeholders():
    from bot.i18n import t

    out = t("roll.result", "en", result=4, sides=20)
    assert "4" in out and "20" in out


def test_t_bad_placeholder_returns_unformatted_template():
    """A formatting error must never raise — return the raw template."""
    from bot.i18n import t

    # roll.result expects result/sides; omitting them shouldn't crash.
    assert t("roll.result", "en", wrong="x")


def test_english_name_lookup_and_default():
    from bot.i18n import english_name

    assert english_name("hy") == "Armenian"
    assert english_name("ru") == "Russian"
    assert english_name("xx") == "English"
