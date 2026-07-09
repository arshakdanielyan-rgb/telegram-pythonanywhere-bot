"""Optional wrestler photo enrichment for wrestling questions.

When ``WRESTLER_IMAGES`` is enabled and a message names a specific wrestler,
``send_wrestler_images()`` sends that wrestler's photo into the chat. The
lookup, for each named wrestler:

1. Find the correct **Wikipedia** article (keyless REST summary endpoint).
   Disambiguation pages retry once with a ``(wrestler)`` qualifier, and pages
   that don't read as wrestling-related are rejected — so the photo always
   belongs to the right wrestler, not a same-named person.
2. Use the article's main profile image (the infobox photo, which Wikipedia
   already serves from Wikimedia Commons).
3. If the article has **no** image, fall back to searching **Wikimedia
   Commons** directly, keeping only image files whose name contains the
   wrestler's surname (a verification guard against wrong-person photos).
4. If no verified image is found anywhere, the wrestler is reported to the
   caller as "missing" so the user gets a localized "no official image" note.

Everything here is best-effort: feature disabled, no name found, no image,
disambiguation page, a non-wrestling namesake, or a network/whitelist failure
all degrade gracefully. Image enrichment must never block or break the text
reply, so every path is wrapped and returns quietly on failure.

Whitelist note: both ``en.wikipedia.org`` and ``commons.wikimedia.org`` are
covered by PythonAnywhere's free-tier outbound whitelist (``.wikipedia.org`` /
``.wikimedia.org``). The photo bytes themselves are fetched by Telegram from
the ``upload.wikimedia.org`` URL, so they never touch the PA whitelist.
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
_WIKI_TIMEOUT = 6  # seconds per Wikipedia / Commons lookup
_MAX_IMAGES = 3  # cap photos per message (avoid spam / latency)
_COMMONS_LIMIT = 10  # Commons search results to scan for a verified image
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
_COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
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


def _strip_qualifier(name: str) -> str:
    """Drop a trailing "(wrestler)"-style disambiguation qualifier for display."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip() or name


def _surname(name: str) -> str:
    """Return the last word of a (qualifier-stripped) name, lowercased.

    Used to verify a Wikimedia Commons file actually belongs to the wrestler:
    "Aleksandr Karelin" -> "karelin". Falls back to the whole cleaned name for
    mononyms."""
    cleaned = _strip_qualifier(name)
    parts = cleaned.split()
    return (parts[-1] if parts else cleaned).lower()


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


def _fetch_wiki(name: str) -> dict | None:
    """Return a wrestler's Wikipedia article as a dict, or ``None``.

    The dict has ``title`` and ``image`` (a photo URL or ``None``). Uses
    Wikipedia's REST summary endpoint. Skips disambiguation pages and pages
    that don't look wrestling-related — the latter stops showing a photo of a
    non-wrestler who happens to share the name. An article with no photo is
    still returned (``image`` is ``None``) so the caller can try Commons.
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
    return {"title": data.get("title") or name, "image": image}


def _fetch_commons_image(query: str, surname: str) -> str | None:
    """Search Wikimedia Commons for a wrestler's image, or return ``None``.

    Scans the File namespace for ``query`` and returns the first result that
    (a) is a raster image and (b) has ``surname`` in its filename — the
    surname check is a verification guard so a generic search can't return a
    photo of the wrong person. Called only as a fallback when the Wikipedia
    article has no infobox photo.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",  # File: namespace
        "gsrlimit": str(_COMMONS_LIMIT),
        "prop": "imageinfo",
        "iiprop": "url|mime",
    }
    try:
        r = requests.get(
            _COMMONS_API_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_WIKI_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        pages = (r.json().get("query") or {}).get("pages") or {}
    except Exception as e:
        print(f"Commons lookup failed for {query!r}: {e}")
        return None
    # `generator=search` tags each page with an `index` giving search rank;
    # sort by it so we consider the best matches first.
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        title = str(page.get("title") or "")
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("url")
        mime = str(info.get("mime") or "")
        # Skip SVGs (flags/logos) and non-image files; require the wrestler's
        # surname in the filename so the photo can't be of a different person.
        if not url or mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            continue
        if surname and surname not in title.lower():
            continue
        return url
    return None


def _resolve_image(name: str) -> tuple[str, str] | None:
    """Return ``(image_url, display_title)`` for a wrestler, or ``None``.

    Wikipedia first (which also verifies the name really is a wrestler), then
    a Wikimedia Commons fallback when the confirmed article has no photo. When
    Wikipedia can't confirm a wrestler by this name, returns ``None`` rather
    than risk showing the wrong person's photo.
    """
    info = _fetch_wiki(name)
    if info is None:
        return None
    if info["image"]:
        return info["image"], info["title"]
    # Confirmed wrestler article, but no infobox photo: fall back to Commons,
    # searching by the canonical article title and verifying by surname.
    title = info["title"]
    url = _fetch_commons_image(_strip_qualifier(title), _surname(title))
    if url:
        return url, title
    return None


def notify_missing_photos(message, names: list[str]) -> None:
    """Send the localized 'no official image' note for ``names`` (best-effort).

    Called after any photos so a named wrestler with no verified image reads as
    a deliberate omission, not a bug. No-op for an empty list.
    """
    if not names:
        return
    try:
        lang = get_language(message.from_user.id)
        display = [_strip_qualifier(n) for n in names]
        note = t("images.not_found", lang, names=", ".join(display))
        bot.send_message(message.chat.id, note)
    except Exception as e:
        print(f"images.not_found note failed: {e}")


def send_wrestler_images(message, user_text: str) -> None:
    """Best-effort: send a photo for each specific wrestler named in ``user_text``.

    For each named wrestler, tries Wikipedia then Wikimedia Commons for a
    verified image and sends it. Any wrestler with no verified image is
    collected and reported via a single localized note. Never raises — image
    enrichment must not break the text reply path.
    """
    if not WRESTLER_IMAGES or not user_text:
        return
    try:
        missing: list[str] = []
        for name in _extract_names(user_text):
            found = _resolve_image(name)
            if not found:
                missing.append(name)
                continue
            image_url, title = found
            try:
                bot.send_photo(message.chat.id, image_url, caption=title)
            except Exception as e:
                print(f"send_photo failed for {title!r}: {e}")
        notify_missing_photos(message, missing)
    except Exception as e:
        print(f"send_wrestler_images error: {e}")
