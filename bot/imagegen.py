"""Optional AI image generation via Google's Gemini image API (free tier).

Enabled when ``IMAGE_API_KEY`` is set (see ``bot/config.py``). The ``/image``
command sends the user's prompt to ``generativelanguage.googleapis.com`` —
whose host is under ``.googleapis.com``, which is on PythonAnywhere's outbound
whitelist — and returns the generated image bytes for the bot to send.

Uses the stable ``models:generateContent`` endpoint so it keeps working across
model versions: point ``IMAGE_MODEL`` at a newer image model and, as long as it
speaks ``generateContent`` (returning an ``inlineData`` image part), no code
change is needed. A model reachable only via a different endpoint would need
``_ENDPOINT`` adapted.

All failures raise :class:`ImageGenError`, which carries an i18n key so the
handler can show a short, localized, user-safe message while the technical
detail is logged server-side.
"""

import base64

import requests

from bot.config import IMAGE_API_KEY, IMAGE_MODEL

_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
# Image generation is slower than text; stay comfortably under Telegram's
# ~60s webhook window so a slow call frees the worker instead of wedging it.
_REQUEST_TIMEOUT = 45


class ImageGenError(Exception):
    """Image-generation failure carrying an i18n key for a user-safe message.

    ``key`` is looked up by the handler via the user's language; the exception
    message itself holds the technical detail for server-side logs only.
    """

    def __init__(self, key: str, detail: str = ""):
        super().__init__(detail or key)
        self.key = key


def _extract_image(data: dict) -> tuple[bytes, str] | None:
    """Pull ``(image_bytes, mime_type)`` from a generateContent response.

    Returns ``None`` when the response contains no decodable image part.
    Handles both the REST camelCase (``inlineData``/``mimeType``) and the
    snake_case (``inline_data``/``mime_type``) spellings defensively.
    """
    for candidate in data.get("candidates") or []:
        parts = ((candidate.get("content") or {}).get("parts")) or []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime = (
                    inline.get("mimeType") or inline.get("mime_type") or "image/png"
                )
                try:
                    return base64.b64decode(inline["data"]), mime
                except (ValueError, TypeError):
                    return None
    return None


def generate_image(prompt: str) -> tuple[bytes, str]:
    """Generate an image for ``prompt``; return ``(image_bytes, mime_type)``.

    Raises :class:`ImageGenError` (with a localizable ``key``) on any failure —
    not configured, network error, HTTP error, safety block, or a response with
    no image.
    """
    if not IMAGE_API_KEY:
        raise ImageGenError("image.failed", "IMAGE_API_KEY is not set")
    try:
        resp = requests.post(
            _ENDPOINT.format(model=IMAGE_MODEL),
            headers={
                "x-goog-api-key": IMAGE_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                # Ask for an image in the response. TEXT is included because
                # some image models reject an image-only modality list; any
                # text part is simply ignored below.
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            },
            timeout=_REQUEST_TIMEOUT,
        )
    except Exception as e:
        print(f"image gen request failed: {e}")
        raise ImageGenError("image.failed", str(e))

    if resp.status_code != 200:
        # Log the body server-side; map common statuses to a clear user message.
        print(f"image gen HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code == 429:
            raise ImageGenError("image.quota", "quota exceeded")
        if resp.status_code in (401, 403):
            raise ImageGenError("image.failed", "auth rejected")
        raise ImageGenError("image.failed", f"HTTP {resp.status_code}")

    try:
        data = resp.json()
    except ValueError as e:
        print(f"image gen: non-JSON response: {e}")
        raise ImageGenError("image.failed", "non-JSON response")

    result = _extract_image(data)
    if result is None:
        # A prompt blocked by the safety filter comes back with no image and a
        # blockReason — surface that distinctly so the user knows to reword.
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise ImageGenError("image.blocked", feedback.get("blockReason", ""))
        print(f"image gen: no image part in response: {str(data)[:300]}")
        raise ImageGenError("image.no_image", "no image part")
    return result
