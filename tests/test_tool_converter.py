"""Тесты для ToolConverter."""

from types import SimpleNamespace

import pytest
from mcp.types import ToolAnnotations

from mcp_assistant.client.tool_converter import ToolConverter


# ============================================================================
# Хелперы
# ============================================================================

def make_tool(
    name: str,
    description: str | None = "",
    schema: dict | None = None,
    annotations: ToolAnnotations | None = None,
) -> SimpleNamespace:
    """Создаёт фейковый MCP-инструмент (похожий на mcp.types.Tool)."""
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema=schema,
        annotations=annotations,
    )


def _converter_with(tool: SimpleNamespace, server: str = "tickets") -> ToolConverter:
    """Конвертер с одним инструментом — сокращает boilerplate."""
    conv = ToolConverter()
    conv.register_server_tools(server, [tool])
    return conv


# ============================================================================
# register_server_tools
# ============================================================================

def test_add_single_tool() -> None:
    """Один инструмент корректно преобразуется в формат GigaChat."""
    converter = ToolConverter()
    converter.register_server_tools("tickets", [
        make_tool("search", "Поиск тикетов", {"type": "object"}),
    ])

    assert len(converter.tool_specs) == 1
    func = converter.tool_specs[0]
    assert func["name"] == "tickets__search"
    assert func["description"] == "Поиск тикетов"
    assert func["parameters"] == {"type": "object"}


def test_add_multiple_tools() -> None:
    """Несколько инструментов с одного сервера."""
    converter = ToolConverter()
    converter.register_server_tools("tickets", [
        make_tool("search"),
        make_tool("update_status"),
    ])

    assert len(converter.tool_specs) == 2
    names = [t["name"] for t in converter.tool_specs]
    assert names == ["tickets__search", "tickets__update_status"]


def test_add_tools_from_multiple_servers() -> None:
    """Инструменты с разных серверов получают разные префиксы."""
    converter = ToolConverter()
    converter.register_server_tools("tickets", [make_tool("search")])
    converter.register_server_tools("kb", [make_tool("search")])

    names = [t["name"] for t in converter.tool_specs]
    assert names == ["tickets__search", "kb__search"]


def test_missing_description_becomes_empty_string() -> None:
    """Если description=None, подставляется пустая строка."""
    converter = _converter_with(make_tool("t", description=None), server="srv")
    assert converter.tool_specs[0]["description"] == ""


def test_missing_schema_gets_default() -> None:
    """Если input_schema=None, подставляется пустая схема."""
    converter = _converter_with(make_tool("t", schema=None), server="srv")
    params = converter.tool_specs[0]["parameters"]
    assert params == {"type": "object", "properties": {}}


def test_empty_tools_list() -> None:
    """Пустой список инструментов не ломает конвертер."""
    converter = ToolConverter()
    converter.register_server_tools("srv", [])

    assert converter.tool_specs == []
    assert converter.resolve("srv__anything") is None


# ============================================================================
# get_tool_specs
# ============================================================================

def test_get_tool_specs_returns_all_tools() -> None:
    """get_tool_specs() возвращает все добавленные инструменты."""
    converter = _converter_with(make_tool("search"))

    functions = converter.get_tool_specs()
    assert len(functions) == 1
    assert functions[0]["name"] == "tickets__search"


def test_get_tool_specs_empty() -> None:
    """Для пустого конвертера возвращает пустой список."""
    converter = ToolConverter()
    assert converter.get_tool_specs() == []


# ============================================================================
# resolve
# ============================================================================

def test_resolve_returns_pair() -> None:
    """resolve() возвращает (server_name, original_tool_name)."""
    converter = _converter_with(make_tool("search"))

    assert converter.resolve("tickets__search") == ("tickets", "search")


def test_resolve_unknown_returns_none() -> None:
    """resolve() для неизвестного имени возвращает None."""
    converter = _converter_with(make_tool("search"))

    assert converter.resolve("unknown__tool") is None
    assert converter.resolve("") is None
    assert converter.resolve("tickets__unknown") is None


# ============================================================================
# _raw: хранение исходного Tool
# ============================================================================

def test_raw_stores_original_tool_object() -> None:
    """_raw хранит исходный Tool, а не урезанную копию."""
    ann = ToolAnnotations(readOnlyHint=False, destructiveHint=True)
    tool = make_tool("update", annotations=ann)

    converter = _converter_with(tool)

    stored = converter._raw["tickets__update"]
    assert stored is tool, "_raw должен хранить объект Tool целиком"
    assert stored.annotations is ann, (
        "аннотации должны сохраняться — иначе requires_confirmation не работает"
    )


# ============================================================================
# requires_confirmation
# ============================================================================

def test_requires_confirmation_unknown_tool_failsafe() -> None:
    """Неизвестный инструмент → подтверждение (fail-safe)."""
    converter = ToolConverter()
    needs, reason = converter.requires_confirmation("nope__nope")

    assert needs is True
    assert reason == "tool not found (fail-safe)"


def test_requires_confirmation_no_annotations_failsafe() -> None:
    """Инструмент без аннотаций → подтверждение (fail-safe)."""
    converter = _converter_with(make_tool("bare", annotations=None))

    needs, reason = converter.requires_confirmation("tickets__bare")
    assert needs is True
    assert reason == "no annotations (fail-safe)"


def test_requires_confirmation_readonly_returns_false() -> None:
    """readOnlyHint=True → подтверждение НЕ нужно."""
    converter = _converter_with(make_tool(
        "read",
        annotations=ToolAnnotations(readOnlyHint=True),
    ))

    needs, reason = converter.requires_confirmation("tickets__read")
    assert needs is False
    assert reason == "readOnlyHint=True"


def test_requires_confirmation_destructive_returns_true() -> None:
    """
    Главный тест: destructiveHint=True читается из аннотаций.

    Именно здесь падал test_agent.py::test_destructive_tool_confirmation_from_annotations:
    requires_confirmation возвращал 'write tool (readOnlyHint is not True)'
    вместо 'destructiveHint=True'.
    """
    converter = _converter_with(make_tool(
        "del",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True),
    ))

    needs, reason = converter.requires_confirmation("tickets__del")
    assert needs is True
    assert reason == "destructiveHint=True", (
        f"Ожидался 'destructiveHint=True', получено {reason!r}. "
        f"Проверьте имя поля в requires_confirmation — должно быть "
        f"destructiveHint (camelCase), как в ToolAnnotations."
    )


def test_requires_confirmation_additive_write_returns_true() -> None:
    """readOnlyHint=False и destructiveHint=False → подтверждение нужно."""
    converter = _converter_with(make_tool(
        "add",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
    ))

    needs, reason = converter.requires_confirmation("tickets__add")
    assert needs is True
    assert reason == "write tool (readOnlyHint is not True)"


def test_requires_confirmation_only_readonly_default() -> None:
    """
    Только readOnlyHint=False без destructiveHint (None по умолчанию)
    → трактуется как write tool.
    """
    converter = _converter_with(make_tool(
        "w",
        annotations=ToolAnnotations(readOnlyHint=False),
    ))

    needs, reason = converter.requires_confirmation("tickets__w")
    assert needs is True
    assert reason == "write tool (readOnlyHint is not True)"


# ============================================================================
# Интеграционные сценарии
# ============================================================================

def test_full_workflow() -> None:
    """Полный сценарий: добавить → получить список → resolve."""
    converter = ToolConverter()
    converter.register_server_tools("tickets", [
        make_tool("search", "Поиск"),
        make_tool("update_status", "Обновление"),
    ])
    converter.register_server_tools("kb", [
        make_tool("get_article", "Статья"),
    ])

    names = [t["name"] for t in converter.tool_specs]
    assert len(names) == len(set(names)) == 3

    for name in names:
        server, tool = converter.resolve(name)
        assert server in ("tickets", "kb")
        assert tool in ("search", "update_status", "get_article")


def test_mapping_and_tools_consistent() -> None:
    """Каждый инструмент в tool_specs имеет запись в mapping."""
    converter = ToolConverter()
    converter.register_server_tools("tickets", [
        make_tool("search"),
        make_tool("update_status"),
    ])
    converter.register_server_tools("kb", [make_tool("get_article")])

    names = [t["name"] for t in converter.tool_specs]
    assert set(names) == set(converter.mapping.keys())