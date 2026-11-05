"""Загрузка конфигурации MCP-серверов."""

import json
import os
from pathlib import Path
from mcp_assistant import config as conf

def load_config(path: str | Path = "servers.json") -> dict:
    """
    Загружает конфиг и разрешает относительные пути в args.
    Для -m-режима добавляет PYTHONPATH, чтобы Python нашел пакет.

    Returns:
        dict с настройками серверов, лимитами агента и таймаутами.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if not config.get("servers"):
        raise ValueError("Список серверов пуст")

    names = set()
    for server in config["servers"]:
        name = server.get("name")
        if not name:
            raise ValueError("У сервера нет поля 'name'")
        if name in names:
            raise ValueError(f"Дублирующееся имя сервера: {name}")
        names.add(name)

        transport = server.get("transport", {})
        t_type = transport.get("type")

        if t_type == "stdio":
            if not transport.get("command"):
                raise ValueError(f"Сервер '{name}': для stdio нужен 'command'")
            
            args = transport.get("args", [])
            if "-m" in args:
                # -m-режим: добавляет PROJECT_DIR в PYTHONPATH,
                # чтобы Python нашел пакет 
                env = transport.setdefault("env", {})
                project = str(conf.PROJECT_DIR)
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (
                    f"{project}{os.pathsep}{existing}" if existing else project
                )
                # args не трогает: там имена модулей, а не файлы
            else:
                # Разрешает относительные пути в args
                transport["args"] = [
                    str((conf.PROJECT_DIR / arg).resolve())
                    if not Path(arg).is_absolute() and _looks_like_path(arg)
                    else arg
                    for arg in transport.get("args", [])
                ]
        elif t_type == "streamable-http":
            if not transport.get("url"):
                raise ValueError(f"Сервер '{name}': для {t_type} нужен 'url'")
        else:
            raise ValueError(f"Сервер '{name}': неизвестный транспорт '{t_type}'")

        server.setdefault("tool_timeout", 30.0)
        server.setdefault("always_confirm_by_server", [])

    config.setdefault("max_steps", 8)
    config.setdefault("max_history_messages", 40)
    config.setdefault("max_history_chars", 60_000)
    config.setdefault("max_tool_result_chars", 12_000)
    config.setdefault("session_timeout", 300.0)
    config.setdefault("llm_timeout", 60.0)

    return config


def _looks_like_path(value: str) -> bool:
    if "://" in value:
        return False
    if value.startswith("-"):
        return False
    return (
        "/" in value or "\\" in value
        or value.endswith((".py", ".js", ".ts", ".json"))
    )