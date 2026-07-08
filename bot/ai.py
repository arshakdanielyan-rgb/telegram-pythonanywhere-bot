from bot.config import SYSTEM_PROMPT
from bot.history import get_history, save_history
from bot.i18n import english_name
from bot.preferences import get_language
from bot.providers import generate, generate_stream


def _build_system_prompt(user_id: int) -> str:
    """System prompt plus a directive to reply in the user's chosen language.

    Keeps AI answers consistent with the interface language selected via
    /language. English (the default) still gets an explicit directive so the
    model doesn't drift when the user writes in another language.
    """
    lang = get_language(user_id)
    directive = f"Always reply to the user in {english_name(lang)}."
    return f"{SYSTEM_PROMPT}\n\n{directive}"


def _build_messages(user_id: int, history: list, grounding: str | None) -> list:
    """Assemble the message list: system prompt, optional Wikipedia grounding,
    then the conversation history.

    ``grounding`` (when given) is inserted as its own system message so the
    model treats the supplied Wikipedia article as the primary source for the
    turn. It is deliberately kept out of ``history`` so it is never persisted —
    it is context for this single answer, not part of the conversation.
    """
    messages = [{"role": "system", "content": _build_system_prompt(user_id)}]
    if grounding:
        messages.append({"role": "system", "content": grounding})
    messages += history
    return messages


def ask_ai(user_id: int, user_message: str, grounding: str | None = None) -> str:
    history = get_history(user_id)
    history.append({"role": "user", "content": user_message})

    messages = _build_messages(user_id, history, grounding)

    reply = generate(user_id, messages)

    history.append({"role": "assistant", "content": reply})
    save_history(user_id, history)
    return reply


def ask_ai_stream(user_id: int, user_message: str, grounding: str | None = None):
    """Stream an AI reply, yielding text deltas as they are generated.

    Same history/system-prompt setup as ask_ai(), but the reply is
    produced incrementally so the caller can update the Telegram message
    live. History is saved only after the stream completes — the caller
    MUST fully consume the generator (a partial reply is never persisted).
    If generation raises mid-stream the exception propagates and nothing
    is saved, so a broken turn doesn't poison the conversation.

    ``grounding``, when supplied, is a Wikipedia source block injected as an
    ephemeral system message (see _build_messages) — not saved to history.
    """
    history = get_history(user_id)
    history.append({"role": "user", "content": user_message})

    messages = _build_messages(user_id, history, grounding)

    parts: list[str] = []
    for delta in generate_stream(user_id, messages):
        parts.append(delta)
        yield delta

    reply = "".join(parts)
    history.append({"role": "assistant", "content": reply})
    save_history(user_id, history)
