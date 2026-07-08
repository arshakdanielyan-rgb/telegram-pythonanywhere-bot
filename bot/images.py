"""Optional wrestler-photo enrichment.

When ``WRESTLER_IMAGES`` is enabled, after answering a question that names a
specific wrestler the bot sends a photo of each wrestler alongside the text
reply. Photos come from Wikipedia's keyless REST summary endpoint (whitelisted
on PythonAnywhere's free tier).

Everything here is best-effort: feature disabled, no name found, no image,
disambiguation page, a non-wrestling namesake, or a network/whitelist failure
all degrade to sending nothing. The text answer is produced and sent by the
caller first, so image enrichment can never block or break a reply.
"""

import json
import re
from urllib.parse import quote

import requests

from bot.clients import ai, bot
from bot.config import MODEL, WRESTLER_IMAGES
from bot.i18n import t
from bot.preferences import get_language

# Keep enrichment cheap and bounded so it never eats into Telegram's ~60s
# webhook window: a short extraction call and at most a few quick lookups.
_EXTRACT_TIMEOUT = 8  # seconds for the name-extraction AI call
_WIKI_TIMEOUT = 6  # seconds per Wikipedia summary lookup
_MAX_IMAGES = 3  # cap photos per message (avoid spam / latency)
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
# Wikimedia asks API clients to send a descriptive User-Agent.
_USER_AGENT = "telegram-wrestling-bot/1.0 (educational; pyTelegramBotAPI)"

_EXTRACT_SYSTEM = (
    "You identify specific, real, named wrestlers in a user's message. "
    "Return ONLY a JSON array of the full names of the wrestlers the user is "
    'asking about, using canonical names suitable for a Wikipedia search (e.g. '
    '["John Cena", "Aleksandr Karelin"]). Include professional and '
    "amateur/Olympic wrestlers. If a name is shared by several notable people, "
    "use the context in the message (sport, country, weight class, era, "
    "competition) to pick the wrestler, and append the Wikipedia "
    'disambiguation qualifier so the name resolves to that person (e.g. '
    '"Ali Aliyev (wrestler)"). Return an empty array [] if the message names '
    "no specific wrestler or is a general question. Never invent names."
)


def _extract_names(user_text: str) -> list[str]:
    """Ask the model for the specific wrestlers named in ``user_text``.

    Returns a de-duplicated, order-preserving list capped at ``_MAX_IMAGES``.
    Any failure (API error, non-JSON reply) yields an empty list.
    """
    try:
        resp = ai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            timeout=_EXTRACT_TIMEOUT,
            max_tokens=80,
        )
        content = str(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"wrestler-image name extraction failed: {e}")
        return []
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        names = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    if not isinstance(names, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if not isinstance(name, str):
            continue
        name = name.strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
        if len(out) >= _MAX_IMAGES:
            break
    return out


def _fetch_summary(name: str) -> dict | None:
    """Fetch and parse Wikipedia's REST summary for ``name``, or ``None``."""
    title = quote(name.replace(" ", "_"), safe="")
    try:
        r = requests.get(
            _WIKI_SUMMARY_URL.format(title),
            headers={"User-Agent": _USER_AGENT},
            timeout=_WIKI_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        print(f"Wikipedia lookup failed for {name!r}: {e}")
        return None


def _fetch_image(name: str) -> tuple[str, str] | None:
    """Return ``(image_url, page_title)`` for a wrestler, or ``None``.

    Uses Wikipedia's REST summary endpoint. Skips disambiguation pages,
    entries with no image, and pages that don't look wrestling-related — the
    last guard stops a photo of a non-wrestler who happens to share the name.
    """
    data = _fetch_summary(name)
    if data is None:
        return None
    # A bare name can resolve to a disambiguation page when several notable
    # people share it (e.g. "Arsen Harutyunyan"). Retry once with the
    # "(wrestler)" qualifier so the lookup lands on the athlete instead of
    # giving up — unless the extractor already supplied a qualifier.
    if data.get("type") == "disambiguation" and "(" not in name:
        data = _fetch_summary(f"{name} (wrestler)")
        if data is None or data.get("type") == "disambiguation":
            return None
    elif data.get("type") == "disambiguation":
        return None
    blurb = f"{data.get('description', '')} {data.get('extract', '')}".lower()
    if "wrestl" not in blurb:
        return None
    image = (data.get("thumbnail") or {}).get("source") or (
        data.get("originalimage") or {}
    ).get("source")
    if not image:
        return None
    return image, data.get("title") or name


def send_wrestler_images(message, user_text: str) -> None:
    """Best-effort: send a photo for each specific wrestler named in ``user_text``.

    For any named wrestler with no verified photo, a short localized note is
    sent so the user knows the omission is deliberate (not a bug) — the text
    answer has already gone out, so this only adds context. Never raises —
    image enrichment must not break the text reply path.
    """
    if not WRESTLER_IMAGES or not user_text:
        return
    try:
        missing: list[str] = []
        for name in _extract_names(user_text):
            found = _fetch_image(name)
            if not found:
                missing.append(name)
                continue
            image_url, title = found
            try:
                bot.send_photo(message.chat.id, image_url, caption=title)
            except Exception as e:
                print(f"send_photo failed for {title!r}: {e}")
        if missing:
            lang = get_language(message.from_user.id)
            # Drop any "(wrestler)"-style disambiguation qualifier for display —
            # it helps the Wikipedia lookup but reads oddly to a user.
            display = [re.sub(r"\s*\([^)]*\)\s*$", "", n).strip() or n for n in missing]
            note = t("images.not_found", lang, names=", ".join(display))
            try:
                bot.send_message(message.chat.id, note)
            except Exception as e:
                print(f"images.not_found note failed: {e}")
    except Exception as e:
        print(f"send_wrestler_images error: {e}")
