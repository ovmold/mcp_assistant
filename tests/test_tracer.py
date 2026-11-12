"""Тесты для Tracer."""

import json
from pathlib import Path

import pytest

from mcp_assistant.client.tracer import Tracer


# ============================================================================
# Хелперы
# ============================================================================

def read_jsonl(path: Path) -> list[dict]:
    """Читает JSONL-файл и возвращает список объектов."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ============================================================================
# Базовые сценарии
# ============================================================================

def test_creates_file(tmp_path) -> None:
    """Tracer создаёт файл, если его нет."""
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path)
    tracer.log("event", data=1)
    tracer.close()

    assert path.exists()


def test_creates_parent_dirs(tmp_path) -> None:
    """Tracer создаёт родительские папки, если их нет."""
    path = tmp_path / "nested" / "dir" / "trace.jsonl"
    tracer = Tracer(path)
    tracer.log("event", data=1)
    tracer.close()

    assert path.exists()


def test_each_line_is_valid_json(tmp_path) -> None:
    """Каждая строка файла — валидный JSON."""
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        t.log("event1", content="привет")
        t.log("event2", data={"x": 1})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # не должно упасть


def test_log_contains_type_and_timestamp(tmp_path) -> None:
    """Запись содержит поля type и timestamp."""
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        t.log("tool_call", server="tickets", tool="search")

    records = read_jsonl(path)
    assert len(records) == 1
    rec = records[0]
    assert rec["type"] == "tool_call"
    assert "timestamp" in rec
    assert rec["server"] == "tickets"
    assert rec["tool"] == "search"


def test_log_accepts_arbitrary_data(tmp_path) -> None:
    """log() принимает любые именованные аргументы."""
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        t.log(
            "complex",
            string="hello",
            number=42,
            boolean=True,
            nested={"a": [1, 2, 3]},
            nothing=None,
        )

    rec = read_jsonl(path)[0]
    assert rec["string"] == "hello"
    assert rec["number"] == 42
    assert rec["boolean"] is True
    assert rec["nested"] == {"a": [1, 2, 3]}
    assert rec["nothing"] is None


# ============================================================================
# Спецсимволы
# ============================================================================

def test_unicode_not_escaped(tmp_path) -> None:
    """Русские буквы сохраняются как есть (ensure_ascii=False)."""
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        t.log("event", text="привет мир")

    raw = path.read_text(encoding="utf-8")
    assert "привет мир" in raw
    # Проверяем, что нет \u-экранирования
    assert "\\u" not in raw


def test_newlines_in_values_escaped(tmp_path) -> None:
    """Переносы строк внутри значений экранируются, строка остаётся одной."""
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        t.log("event", text="строка 1\nстрока 2")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # одна строка в файле
    rec = json.loads(lines[0])
    assert rec["text"] == "строка 1\nстрока 2"


def test_quotes_escaped(tmp_path) -> None:
    """Кавычки внутри значений не ломают JSON."""
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        t.log("event", text='Он сказал "привет"')

    rec = read_jsonl(path)[0]
    assert rec["text"] == 'Он сказал "привет"'


# ============================================================================
# flush и добавление в файл
# ============================================================================

def test_flush_writes_immediately(tmp_path) -> None:
    """После log() данные уже видны в файле (flush)."""
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path)
    tracer.log("event", data=1)

    # Читаем, не закрывая tracer
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    json.loads(lines[0])

    tracer.close()


def test_appends_to_existing_file(tmp_path) -> None:
    """Tracer не перезаписывает файл, а дописывает в конец."""
    path = tmp_path / "trace.jsonl"
    path.write_text('{"type": "old"}\n', encoding="utf-8")

    with Tracer(path) as t:
        t.log("new", data=1)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "old"
    assert json.loads(lines[1])["type"] == "new"


# ============================================================================
# Контекстный менеджер
# ============================================================================

def test_context_manager_closes_file(tmp_path) -> None:
    """После выхода из with файл закрыт."""
    path = tmp_path / "trace.jsonl"
    with Tracer(path) as t:
        t.log("event", data=1)

    # Попытка записи после close должна упасть
    with pytest.raises(ValueError):
        t._file.write("test")


def test_close_idempotent(tmp_path) -> None:
    """Повторный close() не ломает Tracer."""
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path)
    tracer.log("event", data=1)
    tracer.close()
    tracer.close()  # не должно упасть


# ============================================================================
# Ошибки
# ============================================================================

def test_log_after_close_does_not_write(tmp_path) -> None:
    """log() после close() не должен добавлять строки в файл."""
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path)
    tracer.log("before_close", data=1)
    tracer.close()

    # После close() log() не должен писать
    tracer.log("after_close", data=2)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "before_close"