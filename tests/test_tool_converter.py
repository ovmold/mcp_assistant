"""Тесты для ToolConverter."""

from types import SimpleNamespace

import pytest

from client.tool_converter import ToolConverter


def make_tool(name: str, description: str = "", schema: dict | None = None):
    """Создает фейковый MCP-инструмент (похожий на mcp.types.Tool)."""
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema=schema,
    )


# ============================================================================
# add_server_tools
# ============================================================================

def test_add_single_tool():
    """Один инструмент корректно преобразуется в формат OpenAI."""
    converter = ToolConverter()
    converter.add_server_tools("tickets", [
        make_tool("search", "Поиск тикетов", {"type": "object"}),
    ])

    assert len(converter.tools) == 1
    func = converter.tools[0]["function"]
    assert converter.tools[0]["type"] == "function"
    assert func["name"] == "tickets__search"
    assert func["description"] == "Поиск тикетов"
    assert func["parameters"] == {"type": "object"}


def test_add_multiple_tools():
    """Несколько инструментов с одного сервера."""
    converter = ToolConverter()
    converter.add_server_tools("tickets", [
        make_tool("search"),
        make_tool("update_status"),
    ])

    assert len(converter.tools) == 2
    names = [t["function"]["name"] for t in converter.tools]
    assert names == ["tickets__search", "tickets__update_status"]


def test_add_tools_from_multiple_servers():
    """Инструменты с разных серверов получают разные префиксы."""
    converter = ToolConverter()
    converter.add_server_tools("tickets", [make_tool("search")])
    converter.add_server_tools("kb", [make_tool("search")])

    names = [t["function"]["name"] for t in converter.tools]
    assert names == ["tickets__search", "kb__search"]


def test_missing_description_becomes_empty_string():
    """Если description=None, подставляется пустая строка."""
    converter = ToolConverter()
    converter.add_server_tools("srv", [make_tool("t", description=None)])

    assert converter.tools[0]["function"]["description"] == ""


def test_missing_schema_gets_default():
    """Если input_schema=None, подставляется пустая схема."""
    converter = ToolConverter()
    converter.add_server_tools("srv", [make_tool("t", schema=None)])

    params = converter.tools[0]["function"]["parameters"]
    assert params == {"type": "object", "properties": {}}


# ============================================================================
# resolve
# ============================================================================

def test_resolve_returns_pair():
    """resolve() возвращает (server_name, original_tool_name)."""
    converter = ToolConverter()
    converter.add_server_tools("tickets", [make_tool("search")])

    result = converter.resolve("tickets__search")
    assert result == ("tickets", "search")


def test_resolve_unknown_returns_none():
    """resolve() для неизвестного имени возвращает None."""
    converter = ToolConverter()
    converter.add_server_tools("tickets", [make_tool("search")])

    assert converter.resolve("unknown__tool") is None
    assert converter.resolve("") is None
    assert converter.resolve("tickets__unknown") is None


def test_resolve_is_case_sensitive():
    """resolve() чувствителен к регистру (как и имена инструментов)."""
    converter = ToolConverter()
    converter.add_server_tools("tickets", [make_tool("Search")])

    assert converter.resolve("tickets__Search") == ("tickets", "Search")
    assert converter.resolve("tickets__search") is None


# ============================================================================
# Интеграционные сценарии
# ============================================================================

def test_full_workflow():
    """Полный сценарий: добавить → получить список → resolve."""
    converter = ToolConverter()
    converter.add_server_tools("tickets", [
        make_tool("search", "Поиск"),
        make_tool("update_status", "Обновление"),
    ])
    converter.add_server_tools("kb", [
        make_tool("get_article", "Статья"),
    ])

    # Все имена уникальны
    names = [t["function"]["name"] for t in converter.tools]
    assert len(names) == len(set(names)) == 3

    # Все имена разрешаются обратно
    for name in names:
        server, tool = converter.resolve(name)
        assert server in ("tickets", "kb")
        assert tool in ("search", "update_status", "get_article")


def test_empty_tools_list():
    """Пустой список инструментов не ломает конвертер."""
    converter = ToolConverter()
    converter.add_server_tools("srv", [])

    assert converter.tools == []
    assert converter.resolve("srv__anything") is None