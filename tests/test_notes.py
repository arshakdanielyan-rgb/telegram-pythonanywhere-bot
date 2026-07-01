import json
from unittest.mock import patch


def test_get_notes_empty_by_default():
    with patch("bot.notes.store", None), \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        from bot.notes import get_notes
        assert get_notes(123) == []


def test_add_note_works_without_a_store():
    """Stateless mode: the note is kept in-process and recallable."""
    with patch("bot.notes.store", None), \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        from bot.notes import add_note, get_notes
        add_note(123, "I train on Mondays")
        assert get_notes(123) == ["I train on Mondays"]


def test_add_note_appends_in_order():
    with patch("bot.notes.store", None), \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        from bot.notes import add_note, get_notes
        add_note(1, "first")
        add_note(1, "second")
        assert get_notes(1) == ["first", "second"]


def test_notes_are_per_user():
    with patch("bot.notes.store", None), \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        from bot.notes import add_note, get_notes
        add_note(1, "alice note")
        add_note(2, "bob note")
        assert get_notes(1) == ["alice note"]
        assert get_notes(2) == ["bob note"]


def test_add_note_persists_json_to_store():
    with patch("bot.notes.store") as mock_store, \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        mock_store.get.return_value = None
        from bot.notes import add_note
        add_note(123, "hello")
        key, value = mock_store.set.call_args[0][:2]
        assert key == "notes:123"
        assert json.loads(value) == ["hello"]


def test_get_notes_reads_json_from_store():
    with patch("bot.notes.store") as mock_store, \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        mock_store.get.return_value = json.dumps(["a", "b"])
        from bot.notes import get_notes
        assert get_notes(123) == ["a", "b"]


def test_add_note_caps_at_max():
    with patch("bot.notes.store", None), \
         patch.dict("bot.notes._notes_fallback", {}, clear=True), \
         patch("bot.notes.MAX_NOTES", 3):
        from bot.notes import add_note, get_notes
        for i in range(5):
            add_note(1, f"note{i}")
        notes = get_notes(1)
        # Oldest dropped, newest kept.
        assert notes == ["note2", "note3", "note4"]


def test_get_notes_falls_back_on_store_error():
    with patch("bot.notes.store") as mock_store, \
         patch.dict("bot.notes._notes_fallback", {123: ["cached"]}, clear=True):
        mock_store.get.side_effect = Exception("down")
        from bot.notes import get_notes
        assert get_notes(123) == ["cached"]


def test_get_notes_ignores_malformed_json():
    with patch("bot.notes.store") as mock_store, \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        mock_store.get.return_value = "{not valid json"
        from bot.notes import get_notes
        assert get_notes(123) == []


# ── remove_note / clear_notes ───────────────────────────────────────────────────


def test_remove_note_by_index():
    with patch("bot.notes.store", None), \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        from bot.notes import add_note, get_notes, remove_note
        add_note(1, "a")
        add_note(1, "b")
        add_note(1, "c")
        removed = remove_note(1, 2)  # 1-based → "b"
        assert removed == "b"
        assert get_notes(1) == ["a", "c"]


def test_remove_note_out_of_range_returns_none():
    with patch("bot.notes.store", None), \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        from bot.notes import add_note, get_notes, remove_note
        add_note(1, "only")
        assert remove_note(1, 5) is None
        assert remove_note(1, 0) is None
        assert get_notes(1) == ["only"]  # unchanged


def test_clear_notes_empties_everything():
    with patch("bot.notes.store", None), \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        from bot.notes import add_note, clear_notes, get_notes
        add_note(1, "a")
        add_note(1, "b")
        clear_notes(1)
        assert get_notes(1) == []


def test_clear_notes_deletes_from_store():
    with patch("bot.notes.store") as mock_store, \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        from bot.notes import clear_notes
        clear_notes(123)
        mock_store.delete.assert_called_once_with("notes:123")


def test_remove_note_persists_to_store():
    with patch("bot.notes.store") as mock_store, \
         patch.dict("bot.notes._notes_fallback", {}, clear=True):
        mock_store.get.return_value = json.dumps(["a", "b"])
        from bot.notes import remove_note
        assert remove_note(123, 1) == "a"
        _, value = mock_store.set.call_args[0][:2]
        assert json.loads(value) == ["b"]
