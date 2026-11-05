"""
Трассировка агента в JSONL-формате.

Каждое событие — отдельная строка JSON в файле.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

class Tracer:
    """Пишет события сессии в JSONL-файл."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def log(self, event_type: str, **data: Any) -> None:
        """
        Записывает событие в файл.

        Args:
            event_type: Тип события (user_request, tool_call, llm_response и т.д.)
            **data: Произвольные поля события в виде <name> = <value>.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **data,
        }
        try:
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception as e:
            print(f"Ошибка записи трассировки: {e}")

    def close(self) -> None:
        """Закрывает файл."""
        self._file.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *args) -> None:
        self.close()