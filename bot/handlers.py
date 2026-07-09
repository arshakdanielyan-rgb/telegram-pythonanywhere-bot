import os
from datetime import datetime
from telebot import types
from bot.clients import bot, BOT_INFO
from bot.config import COMMIT_SHA, HF_SPACE_ID, RATE_LIMIT
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
    /quote, /fact, /quiz): each just feeds a fixed instruction to
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

@bot.message_handler(commands=["help"], func=is_allowed)
def cmd_help(message):
    lang = get_language(message.from_user.id)
    keys = [
        "help.start",
        "help.help",
        "help.reset",
        "help.about",
        "help.sha",
        "help.quote",
        "help.fact",
        "help.quiz",
        "help.predictor",
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


@bot.message_handler(commands=["sha"], func=is_allowed)
def cmd_sha(message):
    sha = COMMIT_SHA or "unknown"
    bot.send_message(message.chat.id, f"Live SHA: {sha}")


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


@bot.message_handler(commands=["quiz"], func=is_allowed)
def cmd_quiz(message):
    _stream_ai_command(
        message,
        "Ask the user one fun trivia question about wrestling. Give the "
        "question only — no answer, no multiple-choice options unless it "
        "helps. Keep it to one or two sentences and invite them to reply "
        "with their guess.",
    )


@bot.message_handler(commands=["predictor"], func=is_allowed)
def cmd_predictor(message):
    parts = (message.text or "").split(maxsplit=1)
    matchup = parts[1].strip() if len(parts) > 1 else ""
    if not matchup:
        bot.send_message(
            message.chat.id, _tr(message.from_user.id, "predictor.usage")
        )
        return
    prompt = (
        "The user wants a fun, hypothetical wrestling match prediction for "
        f"this matchup: {matchup}. Predict who is more likely to win and "
        "clearly explain why — compare their styles, strengths, signature "
        "techniques, and records if you know them. Make clear this is just a "
        "fun prediction, not a real result, and never invent statistics."
    )
    _stream_ai_command(message, prompt)


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
        reply = stream_reply(
            message, ask_ai_stream(message.from_user.id, text)
        )
        _log(message, "out", reply)
    except Exception as e:
        print(f"Error in handle_message: {e}")
        bot.send_message(message.chat.id, _tr(message.from_user.id, "error.generic"))
        _log(message, "out", f"[error] {e}")
