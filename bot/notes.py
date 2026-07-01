"""Per-user free-form notes, backing the /remember and /recall commands.

Notes are stored as a JSON list under ``notes:{user_id}`` in the SQLite store
(permanent — no TTL). As with language preferences, an in-process fallback
keeps the feature working in stateless mode (no ``SQLITE_PATH``) or during a
transient store outage, so /remember never silently drops a note.
"""

import json
from bot.clients import store

# Cap notes per user so one user can't grow the store unbounded. When full,
# the oldest note is dropped (FIFO) rather than rejecting the new one.
MAX_NOTES = 50

_notes_fallback: dict[int, list[str]] = {}


def _key(user_id: int) -> str:
    return f"notes:{user_id}"


def get_notes(user_id: int) -> list[str]:
    """Return the user's saved notes (most-recent last), or an empty list."""
    if store is not None:
        try:
            raw = store.get(_key(user_id))
            if raw:
                data = json.loads(raw)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"Store read error (notes): {e}")
    return list(_notes_fallback.get(user_id, []))


def add_note(user_id: int, text: str) -> list[str]:
    """Append a note and return the updated list (trimmed to MAX_NOTES).

    Records in-process immediately so the note is never lost, then best-effort
    persists to the store for durability across restarts.
    """
    notes = get_notes(user_id)
    notes.append(text)
    if len(notes) > MAX_NOTES:
        notes = notes[-MAX_NOTES:]
    _save(user_id, notes)
    return notes


def remove_note(user_id: int, index: int) -> str | None:
    """Remove the note at 1-based ``index`` (matching /recall's numbering).

    Returns the removed note's text, or None if the index is out of range.
    """
    notes = get_notes(user_id)
    if index < 1 or index > len(notes):
        return None
    removed = notes.pop(index - 1)
    _save(user_id, notes)
    return removed


def clear_notes(user_id: int) -> None:
    """Forget all of a user's notes (in-process and persisted)."""
    _notes_fallback.pop(user_id, None)
    if store is not None:
        try:
            store.delete(_key(user_id))
        except Exception as e:
            print(f"Store delete error (notes): {e}")


def _save(user_id: int, notes: list[str]) -> None:
    """Record notes in-process immediately, then best-effort persist to store."""
    _notes_fallback[user_id] = notes
    if store is not None:
        try:
            store.set(_key(user_id), json.dumps(notes))
        except Exception as e:
            print(f"Store write error (notes): {e}")
