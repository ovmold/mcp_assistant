"""Тесты для config_mcp.load_config."""

import json
import os
from pathlib import Path

import pytest

from mcp_assistant.client import config_mcp
from mcp_assistant import config as conf


# ============================================================================
# Хелперы и фикстуры
# ============================================================================

def write_config(tmp_path: Path, data: dict) -> Path:
    """Пишет JSON-конфиг во временный файл и возвращает путь."""
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def patch_project_dir(tmp_path, monkeypatch):
    """Подменяет conf.PROJECT_DIR на временную папку."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(conf, "PROJECT_DIR", project_dir)
    return project_dir


# ============================================================================
# Успешная загрузка и дефолты
# ============================================================================

def test_load_minimal_config(tmp_path):
    """Минимальный конфиг с одним stdio-сервером."""
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {"type": "stdio", "command": "python", "args": []},
        }],
    })

    config = config_mcp.load_config(path)

    assert len(config["servers"]) == 1
    assert config["servers"][0]["name"] == "tickets"


def test_defaults_are_set(tmp_path):
    """Все дефолты проставляются, если их нет в JSON."""
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {"type": "stdio", "command": "python"},
        }],
    })

    config = config_mcp.load_config(path)

    # Глобальные дефолты
    assert config["max_steps"] == 8
    assert config["max_history_messages"] == 40
    assert config["session_timeout"] == 300.0
    assert config["llm_timeout"] == 60.0
    assert "max_retries" not in config

    # Дефолты сервера
    server = config["servers"][0]
    assert server["tool_timeout"] == 30.0
    assert server["always_confirm_by_server"] == []


def test_existing_values_not_overwritten(tmp_path):
    """Если значения заданы в JSON, дефолты их не перезаписывают."""
    path = write_config(tmp_path, {
        "model": "GigaChat-2-Max",
        "max_steps": 20,
        "max_history_messages": 15,
        "servers": [{
            "name": "tickets",
            "transport": {"type": "stdio", "command": "python"},
            "tool_timeout": 5.0,
            "always_confirm_by_server": ["update_status"],
        }],
    })

    config = config_mcp.load_config(path)

    assert config["max_steps"] == 20
    assert config["max_history_messages"] == 15
    assert config["servers"][0]["tool_timeout"] == 5.0
    assert config["servers"][0]["always_confirm_by_server"] == ["update_status"]


# ============================================================================
# -m-режим: добавление PYTHONPATH
# ============================================================================

def test_m_mode_adds_pythonpath(tmp_path, patch_project_dir):
    """В -m-режиме добавляется PYTHONPATH с PROJECT_DIR."""
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "mcp_assistant.servers.tickets"],
            },
        }],
    })

    config = config_mcp.load_config(path)
    env = config["servers"][0]["transport"]["env"]

    assert "PYTHONPATH" in env
    assert str(patch_project_dir) in env["PYTHONPATH"]


def test_m_mode_preserves_existing_pythonpath(tmp_path, patch_project_dir):
    """В -m-режиме существующий PYTHONPATH дополняется, а не затирается."""
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "mcp_assistant.servers.tickets"],
                "env": {"PYTHONPATH": "/existing/path"},
            },
        }],
    })

    config = config_mcp.load_config(path)
    pythonpath = config["servers"][0]["transport"]["env"]["PYTHONPATH"]

    # PROJECT_DIR идёт первым, затем os.pathsep, затем старое значение
    expected = f"{patch_project_dir}{os.pathsep}/existing/path"
    assert pythonpath == expected


def test_m_mode_does_not_modify_args(tmp_path):
    """В -m-режиме args не преобразуются в пути."""
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "mcp_assistant.servers.tickets"],
            },
        }],
    })

    config = config_mcp.load_config(path)
    args = config["servers"][0]["transport"]["args"]

    assert args == ["-m", "mcp_assistant.servers.tickets"]


# ============================================================================
# Обычный режим: разрешение путей в args
# ============================================================================

def test_relative_args_resolved(tmp_path, patch_project_dir):
    """Относительные пути в args разрешаются относительно PROJECT_DIR."""
    (patch_project_dir / "server.py").touch()
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["server.py"],
            },
        }],
    })

    config = config_mcp.load_config(path)
    args = config["servers"][0]["transport"]["args"]

    expected = str((patch_project_dir / "server.py").resolve())
    assert args[0] == expected


def test_absolute_args_not_changed(tmp_path):
    """Абсолютные пути в args остаются как есть."""
    abs_path = str(tmp_path / "absolute_server.py")
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": [abs_path],
            },
        }],
    })

    config = config_mcp.load_config(path)
    assert config["servers"][0]["transport"]["args"][0] == abs_path


def test_flags_and_numbers_not_touched(tmp_path):
    """Флаги (--port) и числа (8000) не преобразуются в пути."""
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": ["--port", "8000"],
            },
        }],
    })

    config = config_mcp.load_config(path)
    args = config["servers"][0]["transport"]["args"]
    assert args == ["--port", "8000"]


def test_url_in_args_not_touched(tmp_path):
    """URL в args не преобразуется в путь (нет ошибки resolve)."""
    url = "http://example.com/server.py"
    path = write_config(tmp_path, {
        "servers": [{
            "name": "tickets",
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": [url],
            },
        }],
    })

    config = config_mcp.load_config(path)
    assert config["servers"][0]["transport"]["args"][0] == url


# ============================================================================
# streamable-http
# ============================================================================

def test_streamable_http_server(tmp_path):
    """Streamable-http сервер с url загружается корректно."""
    path = write_config(tmp_path, {
        "servers": [{
            "name": "kb",
            "transport": {"type": "streamable-http", "url": "http://localhost:8000/mcp"},
        }],
    })

    config = config_mcp.load_config(path)
    assert config["servers"][0]["transport"]["url"] == "http://localhost:8000/mcp"


# ============================================================================
# Валидация (ошибки)
# ============================================================================

def test_file_not_found(tmp_path):
    """Несуществующий файл → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        config_mcp.load_config(tmp_path / "missing.json")


def test_empty_servers_raises(tmp_path):
    """Пустой список серверов → ValueError."""
    path = write_config(tmp_path, {"servers": []})

    with pytest.raises(ValueError, match="Список серверов пуст"):
        config_mcp.load_config(path)


def test_missing_servers_key_raises(tmp_path):
    """Отсутствие ключа servers → ValueError."""
    path = write_config(tmp_path, {})

    with pytest.raises(ValueError, match="Список серверов пуст"):
        config_mcp.load_config(path)


def test_server_without_name_raises(tmp_path):
    """Сервер без name → ValueError."""
    path = write_config(tmp_path, {
        "servers": [{"transport": {"type": "stdio", "command": "python"}}],
    })

    with pytest.raises(ValueError, match="нет поля 'name'"):
        config_mcp.load_config(path)


def test_duplicate_names_raise(tmp_path):
    """Дублирующиеся имена серверов → ValueError."""
    path = write_config(tmp_path, {
        "servers": [
            {"name": "tickets", "transport": {"type": "stdio", "command": "python"}},
            {"name": "tickets", "transport": {"type": "stdio", "command": "python"}},
        ],
    })

    with pytest.raises(ValueError, match="Дублирующееся имя сервера"):
        config_mcp.load_config(path)


def test_stdio_without_command_raises(tmp_path):
    """stdio-сервер без command → ValueError."""
    path = write_config(tmp_path, {
        "servers": [{"name": "tickets", "transport": {"type": "stdio"}}],
    })

    with pytest.raises(ValueError, match="для stdio нужен 'command'"):
        config_mcp.load_config(path)


def test_http_without_url_raises(tmp_path):
    """streamable-http без url → ValueError."""
    path = write_config(tmp_path, {
        "servers": [{"name": "kb", "transport": {"type": "streamable-http"}}],
    })

    with pytest.raises(ValueError, match="для streamable-http нужен 'url'"):
        config_mcp.load_config(path)


def test_unknown_transport_raises(tmp_path):
    """Неизвестный тип транспорта → ValueError."""
    path = write_config(tmp_path, {
        "servers": [{"name": "x", "transport": {"type": "websocket"}}],
    })

    with pytest.raises(ValueError, match="неизвестный транспорт"):
        config_mcp.load_config(path)