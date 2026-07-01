from bot.clients import store
from bot.config import DEFAULT_PROVIDER, HF_SPACE_ID
from bot.i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES

VALID_PROVIDERS = ("main", "hf")

# In-process fallback for the reply-language choice. The SQLite store, when
# configured, is the durable home (survives restarts); this dict guarantees
# the choice takes effect immediately even in stateless mode (no SQLITE_PATH)
# or during a transient store outage, so /language never silently no-ops.
_language_prefs: dict[int, str] = {}


def get_provider(user_id: int) -> str:
    """Return the user's chosen provider, or DEFAULT_PROVIDER.

    Falls back to DEFAULT_PROVIDER if storage is not configured,
    storage is down, the user has no saved preference, or the saved
    preference is "hf" but HF_SPACE_ID is not configured.
    """
    if store is None:
        return DEFAULT_PROVIDER
    try:
        value = store.get(f"provider:{user_id}")
    except Exception as e:
        print(f"Store read error (preferences): {e}")
        return DEFAULT_PROVIDER
    if value not in VALID_PROVIDERS:
        return DEFAULT_PROVIDER
    if value == "hf" and not HF_SPACE_ID:
        return DEFAULT_PROVIDER
    return value


def set_provider(user_id: int, provider: str) -> bool:
    """Save the user's provider choice. Returns True on success."""
    if provider not in VALID_PROVIDERS:
        return False
    if store is None:
        return False
    try:
        store.set(f"provider:{user_id}", provider)
        return True
    except Exception as e:
        print(f"Store write error (preferences): {e}")
        return False


def get_language(user_id: int) -> str:
    """Return the user's chosen language code, or DEFAULT_LANGUAGE.

    Prefers the persisted store value when available; otherwise uses the
    in-process choice (covers stateless mode and store outages); finally
    defaults to DEFAULT_LANGUAGE for a user who has never chosen.
    """
    if store is not None:
        try:
            value = store.get(f"language:{user_id}")
            if value in SUPPORTED_LANGUAGES:
                return value
        except Exception as e:
            print(f"Store read error (language): {e}")
    return _language_prefs.get(user_id, DEFAULT_LANGUAGE)


def set_language(user_id: int, language: str) -> bool:
    """Save the user's language choice. Returns True for any supported code.

    Records the choice in-process immediately so it always takes effect, then
    best-effort persists it to the store for durability across restarts. A
    store failure does not fail the call — the in-process value still wins.
    """
    if language not in SUPPORTED_LANGUAGES:
        return False
    _language_prefs[user_id] = language
    if store is not None:
        try:
            store.set(f"language:{user_id}", language)
        except Exception as e:
            print(f"Store write error (language): {e}")
    return True
