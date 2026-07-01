"""Lightweight localization for user-facing text.

Translations live in JSON files under ``bot/locales/`` — one file per
language, named ``<code>.json`` (e.g. ``en.json``). Look up a string with
``t(key, lang)``; missing keys fall back to English, then to the raw key so
a typo can never crash a handler.

Adding a new language is a two-step, code-free change:
    1. Add a row to ``SUPPORTED_LANGUAGES`` below.
    2. Drop a matching ``<code>.json`` into ``bot/locales/``.
The /language menu, the reply-language directive, and validation all read
from ``SUPPORTED_LANGUAGES``, so nothing else needs editing.
"""

import json
from pathlib import Path

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"

DEFAULT_LANGUAGE = "en"

# Ordered so the /language menu always renders the same way. Each row is
# code -> (flag emoji, native name, English name). The native name labels the
# selection button (a picker should show each language in its own script); the
# English name is fed to the model so AI replies come back in that language.
SUPPORTED_LANGUAGES = {
    "hy": ("🇦🇲", "Հայերեն", "Armenian"),
    "en": ("🇺🇸", "English", "English"),
    "ru": ("🇷🇺", "Русский", "Russian"),
}


def _load_translations() -> dict[str, dict[str, str]]:
    """Read every supported language's JSON file once at import time.

    A missing or malformed file degrades to an empty table for that language
    (its keys then fall back to English) rather than crashing the worker.
    """
    tables: dict[str, dict[str, str]] = {}
    for code in SUPPORTED_LANGUAGES:
        path = _LOCALES_DIR / f"{code}.json"
        try:
            with open(path, encoding="utf-8") as f:
                tables[code] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"i18n: could not load {path} ({e}); using empty table")
            tables[code] = {}
    return tables


_TRANSLATIONS = _load_translations()


def t(key: str, lang: str = DEFAULT_LANGUAGE, **kwargs) -> str:
    """Return the localized string for ``key`` in ``lang``.

    Resolution order: the requested language → English → the raw key. Any
    ``{placeholder}`` tokens are filled from ``kwargs``; a formatting error
    (e.g. a missing placeholder) returns the unformatted template rather than
    raising.
    """
    table = _TRANSLATIONS.get(lang) or {}
    template = table.get(key)
    if template is None:
        template = _TRANSLATIONS.get(DEFAULT_LANGUAGE, {}).get(key, key)
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template


def english_name(lang: str) -> str:
    """English name of a language code (for the model reply directive)."""
    row = SUPPORTED_LANGUAGES.get(lang) or SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]
    return row[2]
