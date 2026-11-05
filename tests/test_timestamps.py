from datetime import datetime, timezone

from mcp_assistant.storage.timestamps import to_iso


def test_to_iso_removes_microseconds():
    value = datetime(2026, 10, 14, 12, 30, 45, 987654, tzinfo=timezone.utc)

    assert to_iso(value) == "2026-10-14T12:30:45+00:00"


def test_to_iso_preserves_timezone():
    value = datetime.fromisoformat("2026-10-14T15:30:45+03:00")

    assert to_iso(value) == "2026-10-14T15:30:45+03:00"
