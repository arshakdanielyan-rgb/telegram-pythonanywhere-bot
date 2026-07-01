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


def ask_ai(user_id: int, user_message: str) -> str:
    history = get_history(user_id)
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": _build_system_prompt(user_id)}]
    messages += history

    reply = generate(user_id, messages)

    history.append({"role": "assistant", "content": reply})
    save_history(user_id, history)
    return reply


def ask_ai_stream(user_id: int, user_message: str):
    """Stream an AI reply, yielding text deltas as they are generated.

    Same history/system-prompt setup as ask_ai(), but the reply is
    produced incrementally so the caller can update the Telegram message
    live. History is saved only after the stream completes — the caller
    MUST fully consume the generator (a partial reply is never persisted).
    If generation raises mid-stream the exception propagates and nothing
    is saved, so a broken turn doesn't poison the conversation.
    """
    history = get_history(user_id)
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": _build_system_prompt(user_id)}]
    messages += history

    parts: list[str] = []
    for delta in generate_stream(user_id, messages):
        parts.append(delta)
        yield delta

    reply = "".join(parts)
    history.append({"role": "assistant", "content": reply})
    save_history(user_id, history)
