from datetime import datetime


def to_iso(value: datetime) -> str:
    """Возвращает ISO 8601 без микросекунд."""
    return value.replace(microsecond=0).isoformat()
