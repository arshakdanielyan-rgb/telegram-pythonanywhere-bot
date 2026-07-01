import os
import random
from datetime import datetime
from telebot import types
from bot.clients import bot, BOT_INFO
from bot.config import HF_SPACE_ID, RATE_LIMIT
from bot.ai import ask_ai, ask_ai_stream
from bot.helpers import (
    is_allowed,
    keep_typing,
    send_reply,
    should_respond,
    stream_reply,
)
from bot.history import clear_history
from bot.i18n import SUPPORTED_LANGUAGES, t
from bot.notes import add_note, clear_notes, get_notes, remove_note
from bot.preferences import get_language, get_provider, set_language, set_provider
from bot.rate_limit import is_rate_limited


def _tr(user_id: int, key: str, **kwargs) -> str:
    """Translate a key into the user's chosen language (English fallback)."""
    return t(key, get_language(user_id), **kwargs)

# Verbose console logging for local dev and teaching. Enabled by
# BOT_VERBOSE_LOG=1 (run_local.py sets this automatically). Prints one
# line per inbound/outbound message so kids and teachers can see the
# conversation flow in their terminal while the bot is running.
VERBOSE_LOG = os.environ.get("BOT_VERBOSE_LOG", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _log(message, direction: str, text: str) -> None:
    """Print a one-line trace of a message in verbose mode.

    direction is "in" (user → bot) or "out" (bot → user). Text is
    truncated to 500 characters so long AI replies don't flood the
    terminal. Newlines are collapsed for single-line readability.
    """
    if not VERBOSE_LOG:
        return
    user = message.from_user
    user_name = (
        f"@{user.username}" if user.username else (user.first_name or f"user:{user.id}")
    )
    bot_name = f"@{BOT_INFO.username}"
    snippet = (text or "").replace("\n", " ").replace("\r", " ")
    if len(snippet) > 500:
        snippet = snippet[:500] + "..."
    if direction == "in":
        sender, receiver = user_name, bot_name
    else:
        sender, receiver = bot_name, user_name
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {sender} → {receiver}: {snippet}", flush=True)


def _stream_ai_command(message, prompt: str) -> None:
    """Stream a live AI reply for a fixed command prompt, with shared
    error handling. Used by the small "canned prompt" commands (/about,
    /quote, /fact, /compliment): each just feeds a fixed instruction to
    the AI and streams the result into the chat.
    """
    try:
        reply = stream_reply(message, ask_ai_stream(message.from_user.id, prompt))
        _log(message, "out", reply)
    except Exception as e:
        print(f"Error in AI command: {e}")
        bot.send_message(message.chat.id, _tr(message.from_user.id, "error.generic"))
        _log(message, "out", f"[error] {e}")


@bot.message_handler(commands=["start"], func=is_allowed)
def cmd_start(message):
    bot.send_message(message.chat.id, _tr(message.from_user.id, "start.greeting"))

@bot.message_handler(commands=["joke"], func=is_allowed)
def cmd_joke(message):
 reply = ask_ai(message.from_user.id, "Tell one short, joke about wrestling.")
 bot.send_message(message.chat.id, reply)

@bot.message_handler(commands=["help"], func=is_allowed)
def cmd_help(message):
    lang = get_language(message.from_user.id)
    keys = [
        "help.start",
        "help.help",
        "help.reset",
        "help.about",
        "help.joke",
        "help.quote",
        "help.fact",
        "help.compliment",
        "help.roll",
        "help.roast",
        "help.remember",
        "help.recall",
        "help.forget",
        "help.language",
    ]
    if HF_SPACE_ID:
        keys.append("help.model")
    command_list = t("help.header", lang) + "\n\n" + "\n".join(t(k, lang) for k in keys)
    # Generate a short AI intro, then send one message: AI text on top,
    # command list at the bottom. If the AI call fails, send just the list.
    prompt = "The user asked for help. Briefly explain what you can help them with and invite them to ask a question."
    try:
        with keep_typing(message.chat.id):
            intro = ask_ai(message.from_user.id, prompt)
        reply = f"{intro}\n\n{command_list}"
    except Exception as e:
        print(f"Error in cmd_help: {e}")
        reply = command_list
    send_reply(message, reply)
    _log(message, "out", reply)

@bot.message_handler(commands=["roast"], func=is_allowed)
def cmd_roast(message):
 name = message.text.split(maxsplit=1)[1] if " " in message.text else "you"
 reply = ask_ai(message.from_user.id, f"Write a short, playful, friendly roast of {name}.")
 bot.send_message(message.chat.id, reply)

@bot.message_handler(commands=["reset"], func=is_allowed)
def cmd_reset(message):
    clear_history(message.from_user.id)
    bot.send_message(message.chat.id, _tr(message.from_user.id, "reset.done"))


@bot.message_handler(commands=["about"], func=is_allowed)
def cmd_about(message):
    _stream_ai_command(
        message,
        "Briefly introduce yourself: say who you are and what you can help with.",
    )


@bot.message_handler(commands=["quote"], func=is_allowed)
def cmd_quote(message):
    _stream_ai_command(
        message,
        "Share one short, motivational wrestling quote (with the person who "
        "said it if you know it). Keep it to a sentence or two.",
    )


@bot.message_handler(commands=["fact"], func=is_allowed)
def cmd_fact(message):
    _stream_ai_command(
        message,
        "Share one short, surprising wrestling fact. Keep it to a sentence or two.",
    )


@bot.message_handler(commands=["compliment"], func=is_allowed)
def cmd_compliment(message):
    _stream_ai_command(
        message,
        "Give the user a short, genuine, wrestling-themed compliment to "
        "brighten their day. One or two sentences.",
    )


# Dice roll is fully local — no AI needed. Optional argument sets the number
# of sides: "/roll" is a d6, "/roll 20" is a d20. Out-of-range or non-numeric
# arguments are rejected with a hint rather than silently coerced.
ROLL_MIN_SIDES = 2
ROLL_MAX_SIDES = 1000


@bot.message_handler(commands=["roll"], func=is_allowed)
def cmd_roll(message):
    parts = (message.text or "").split(maxsplit=1)
    sides = 6
    if len(parts) > 1:
        arg = parts[1].strip()
        if not (arg.isdigit() and ROLL_MIN_SIDES <= int(arg) <= ROLL_MAX_SIDES):
            bot.send_message(
                message.chat.id,
                _tr(
                    message.from_user.id,
                    "roll.invalid",
                    min=ROLL_MIN_SIDES,
                    max=ROLL_MAX_SIDES,
                ),
            )
            return
        sides = int(arg)
    result = random.randint(1, sides)
    reply = _tr(message.from_user.id, "roll.result", result=result, sides=sides)
    bot.send_message(message.chat.id, reply)
    _log(message, "out", reply)


@bot.message_handler(commands=["remember"], func=is_allowed)
def cmd_remember(message):
    parts = (message.text or "").split(maxsplit=1)
    note = parts[1].strip() if len(parts) > 1 else ""
    if not note:
        bot.send_message(message.chat.id, _tr(message.from_user.id, "remember.usage"))
        return
    add_note(message.from_user.id, note)
    reply = _tr(message.from_user.id, "remember.saved")
    bot.send_message(message.chat.id, reply)
    _log(message, "out", reply)


@bot.message_handler(commands=["recall"], func=is_allowed)
def cmd_recall(message):
    notes = get_notes(message.from_user.id)
    if not notes:
        bot.send_message(message.chat.id, _tr(message.from_user.id, "recall.none"))
        return
    lang = get_language(message.from_user.id)
    numbered = "\n".join(f"{i}. {note}" for i, note in enumerate(notes, 1))
    reply = f"{t('recall.header', lang)}\n\n{numbered}"
    # send_reply splits long lists across messages and falls back to plain text
    # if a note contains Markdown that Telegram would reject.
    send_reply(message, reply)
    _log(message, "out", reply)


@bot.message_handler(commands=["forget"], func=is_allowed)
def cmd_forget(message):
    uid = message.from_user.id
    parts = (message.text or "").split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    if arg:
        # "/forget <n>" — drop a single note by its /recall number.
        if not arg.isdigit():
            bot.send_message(message.chat.id, _tr(uid, "forget.invalid"))
            return
        removed = remove_note(uid, int(arg))
        if removed is None:
            bot.send_message(message.chat.id, _tr(uid, "forget.invalid"))
            return
        reply = _tr(uid, "forget.one", note=removed)
        bot.send_message(message.chat.id, reply)
        _log(message, "out", reply)
        return
    # "/forget" with no argument — clear everything.
    if not get_notes(uid):
        bot.send_message(message.chat.id, _tr(uid, "forget.nothing"))
        return
    clear_notes(uid)
    reply = _tr(uid, "forget.all")
    bot.send_message(message.chat.id, reply)
    _log(message, "out", reply)


# Prefix for the inline-button callbacks emitted by the /language menu.
LANG_CALLBACK_PREFIX = "lang:"


def _language_keyboard():
    """Build the inline keyboard for the /language menu — one button per
    supported language, labelled with its flag + native name."""
    markup = types.InlineKeyboardMarkup(row_width=1)
    markup.add(
        *[
            types.InlineKeyboardButton(
                text=f"{flag} {native}",
                callback_data=f"{LANG_CALLBACK_PREFIX}{code}",
            )
            for code, (flag, native, _english) in SUPPORTED_LANGUAGES.items()
        ]
    )
    return markup


@bot.message_handler(commands=["language"], func=is_allowed)
def cmd_language(message):
    bot.send_message(
        message.chat.id,
        _tr(message.from_user.id, "language.menu"),
        reply_markup=_language_keyboard(),
    )



@bot.callback_query_handler(
    func=lambda c: str(getattr(c, "data", "") or "").startswith(LANG_CALLBACK_PREFIX)
    and is_allowed(c)
)
def on_language_selected(call):
    code = (call.data or "")[len(LANG_CALLBACK_PREFIX):]
    if code not in SUPPORTED_LANGUAGES:
        bot.answer_callback_query(call.id)
        return
    set_language(call.from_user.id, code)
    # Confirmation is rendered in the just-selected language. Editing the menu
    # message replaces the buttons with the confirmation in one clean step;
    # if the edit fails (message too old, etc.) fall back to a fresh message.
    confirmation = t("language.changed", code)
    try:
        bot.edit_message_text(
            confirmation, call.message.chat.id, call.message.message_id
        )
    except Exception as e:
        print(f"Language confirmation edit failed: {e}")
        bot.send_message(call.message.chat.id, confirmation)
    bot.answer_callback_query(call.id)


if HF_SPACE_ID:

    @bot.message_handler(commands=["model"], func=is_allowed)
    def cmd_model(message):
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) == 1:
            current = get_provider(message.from_user.id)
            bot.send_message(
                message.chat.id,
                _tr(message.from_user.id, "model.current", provider=current),
            )
            return
        choice = parts[1].strip().lower()
        if choice not in ("main", "hf"):
            bot.send_message(
                message.chat.id, _tr(message.from_user.id, "model.invalid")
            )
            return
        if not set_provider(message.from_user.id, choice):
            bot.send_message(
                message.chat.id, _tr(message.from_user.id, "model.save_failed")
            )
            return
        if choice == "hf":
            bot.send_message(
                message.chat.id, _tr(message.from_user.id, "model.switched_hf")
            )
        else:
            bot.send_message(
                message.chat.id, _tr(message.from_user.id, "model.switched_main")
            )


@bot.message_handler(content_types=["text"], func=is_allowed)
def handle_message(message):
    if not should_respond(message):
        return
    text = (message.text or "").replace(f"@{BOT_INFO.username}", "").strip()
    if not text:
        # Edited messages, forwards, or stickers-with-empty-caption can
        # arrive with no usable text. Don't burn rate-limit / AI calls on them.
        return
    _log(message, "in", text)
    if is_rate_limited(message.from_user.id):
        limit_msg = _tr(message.from_user.id, "rate_limit.reached", limit=RATE_LIMIT)
        bot.send_message(message.chat.id, limit_msg)
        _log(message, "out", f"[rate limited] {limit_msg}")
        return
    try:
        reply = stream_reply(message, ask_ai_stream(message.from_user.id, text))
        _log(message, "out", reply)
    except Exception as e:
        print(f"Error in handle_message: {e}")
        bot.send_message(message.chat.id, _tr(message.from_user.id, "error.generic"))
        _log(message, "out", f"[error] {e}")
