import threading
import time
from contextlib import contextmanager
from bot.clients import bot
from bot.config import ALLOWED_USERS, MAX_MSG_LEN

# Pre-compute lookup sets so per-message is_allowed() is O(1).
# Numeric IDs are matched as strings against str(user.id).
_ALLOWED_USERNAMES = {u.lower() for u in ALLOWED_USERS if not u.isdigit()}
_ALLOWED_USER_IDS = {u for u in ALLOWED_USERS if u.isdigit()}

# Telegram "typing" chat action expires after ~5 seconds, so re-send it every
# 4 seconds while slow providers (e.g. HF ArmGPT) are generating.
TYPING_REFRESH_SECONDS = 4

# Live-streaming knobs. Telegram rate-limits edits to the same message, so we
# only push an edit every STREAM_MIN_EDIT_INTERVAL seconds and only once at
# least STREAM_MIN_EDIT_CHARS of new text have accumulated — this keeps the
# "typing" effect smooth without tripping 429 Too Many Requests.
STREAM_MIN_EDIT_INTERVAL = 1.1
STREAM_MIN_EDIT_CHARS = 12
# Placeholder shown before the first token arrives (and if the model returns
# nothing at all). Telegram rejects empty message text, so we never edit to "".
STREAM_PLACEHOLDER = "…"


def _split_once(text: str, limit: int) -> tuple[str, str]:
    """Split `text` into (head, rest) where head fits within `limit`.

    Prefers a paragraph break, then a line break, then a hard cut — the same
    boundary preference as _split_for_telegram, so we never slice through the
    middle of a Markdown entity.
    """
    window = text[:limit]
    cut = window.rfind("\n\n")
    if cut <= 0:
        cut = window.rfind("\n")
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip(), text[cut:].lstrip()


def _split_for_telegram(text: str, limit: int) -> list[str]:
    """Split text into chunks that each fit Telegram's per-message limit.

    Prefers paragraph and line breaks over hard cuts so we don't slice in
    the middle of a Markdown entity (which would make Telegram reject the
    whole chunk). Falls back to a hard cut only if a single line is too
    long to fit.
    """
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        head, remaining = _split_once(remaining, limit)
        chunks.append(head)
    if remaining:
        chunks.append(remaining)
    return chunks


def send_reply(message, text: str) -> None:
    """Send a reply, splitting and Markdown-fallback safely.

    Telegram's Markdown parser is strict — unbalanced ``*`` or ``[`` from
    the model or from search-result titles will reject the entire message.
    On parse errors we retry the same chunk as plain text. If even the
    plain-text send fails we re-raise: the webhook caller relies on this
    signal to skip the dedupe marker so Telegram can retry.
    """
    for chunk in _split_for_telegram(text, MAX_MSG_LEN):
        try:
            bot.send_message(message.chat.id, chunk, parse_mode="Markdown")
        except Exception as e:
            print(f"Markdown send failed, retrying as plain text: {e}")
            bot.send_message(message.chat.id, chunk)


def _finalize_message(chat_id: int, message_id: int, text: str) -> None:
    """Edit a message to its final text, Markdown-first with plain fallback.

    Matches send_reply()'s Markdown handling: the model may emit an unbalanced
    ``*`` or ``[`` that Telegram's strict parser rejects, so we retry as plain
    text. "message is not modified" (the text already matches) is harmless and
    just gets logged.
    """
    text = text or STREAM_PLACEHOLDER
    try:
        bot.edit_message_text(text, chat_id, message_id, parse_mode="Markdown")
    except Exception as e:
        try:
            bot.edit_message_text(text, chat_id, message_id)
        except Exception as e2:
            print(f"Final stream edit failed (markdown: {e}; plain: {e2})")


def stream_reply(message, deltas) -> str:
    """Stream text into Telegram, editing a message live as content arrives.

    Sends a placeholder immediately, then edits it in place as the `deltas`
    iterable yields text fragments — producing a word-by-word "typing" effect.
    Edits are throttled (see STREAM_MIN_EDIT_* above) to stay under Telegram's
    edit rate limit. When the reply grows past one Telegram message the filled
    part is finalized and a fresh message continues the stream. Each message's
    final edit gets a Markdown pass with a plain-text fallback.

    Intermediate edits are sent as plain text on purpose: a half-formed
    Markdown entity from a mid-stream token would make Telegram reject the
    edit. Returns the full accumulated text so the caller can log it.
    """
    chat_id = message.chat.id
    sent = bot.send_message(chat_id, STREAM_PLACEHOLDER)
    message_id = sent.message_id

    full = ""  # everything streamed so far (returned to the caller)
    current = ""  # text belonging to the currently-open message
    displayed = ""  # last text pushed to the open message
    last_edit = 0.0

    for delta in deltas:
        if not delta:
            continue
        full += delta
        current += delta

        # Overflowed one message: finalize the filled part, open a new one.
        while len(current) > MAX_MSG_LEN:
            head, current = _split_once(current, MAX_MSG_LEN)
            _finalize_message(chat_id, message_id, head)
            sent = bot.send_message(chat_id, STREAM_PLACEHOLDER)
            message_id = sent.message_id
            displayed = ""

        now = time.monotonic()
        if (
            current != displayed
            and len(current) - len(displayed) >= STREAM_MIN_EDIT_CHARS
            and now - last_edit >= STREAM_MIN_EDIT_INTERVAL
        ):
            try:
                bot.edit_message_text(current or STREAM_PLACEHOLDER, chat_id, message_id)
                displayed = current
                last_edit = now
            except Exception as e:
                # Transient 429 / "not modified" — retry on the next delta.
                print(f"Stream edit failed: {e}")

    # Final pass on the open message, with Markdown rendering restored.
    _finalize_message(chat_id, message_id, current)
    return full


@contextmanager
def keep_typing(chat_id: int):
    """Keep the Telegram "typing" indicator alive while the block runs.

    Spawns a background thread that re-sends the typing action every few
    seconds until the context exits, then joins the thread before returning
    so the serverless function can shut down cleanly.
    """
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                bot.send_chat_action(chat_id, "typing")
            except Exception as e:
                print(f"typing indicator error: {e}")
                return
            # Use wait() so we can exit early when stop is set
            if stop.wait(TYPING_REFRESH_SECONDS):
                return

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def should_respond(message) -> bool:
    """Respond to all messages in private chats and group chats."""
    return True


def is_allowed(message) -> bool:
    """Telegram-handler `func=` filter implementing the ALLOWED_USERS whitelist.

    Returns True when the whitelist is empty (default — everyone allowed)
    OR when the sender's username (case-insensitive) or numeric user_id
    is in the list. Non-matching messages cause telebot to skip every
    handler, so the bot stays silent for unauthorized users.
    """
    if not ALLOWED_USERS:
        return True
    user = getattr(message, "from_user", None)
    if user is None:
        return False
    if str(getattr(user, "id", "")) in _ALLOWED_USER_IDS:
        return True
    username = getattr(user, "username", "") or ""
    return username.lower() in _ALLOWED_USERNAMES
