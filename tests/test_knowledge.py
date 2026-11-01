"""Юнит-тесты инструментов сервера knowledge (без MCP-слоя)."""

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from servers.knowledge import (
    search_kb, 
    get_article, 
    get_article_resource,
)


# ============================================================================
# search_kb
# ============================================================================

def test_search_kb_finds_relevant(temp_kb):
    """Релевантный запрос должен находить статью про 502."""
    result = search_kb("502 оплата", top_k=3)
    assert "payment-502" in result


def test_search_kb_returns_empty_for_unknown(temp_kb):
    """Неизвестный запрос не должен ничего находить (или отсекаться по порогу)."""
    result = search_kb("квантовая хромодинамика")
    assert result == []


def test_search_kb_empty_query_raises(temp_kb):
    """Пустой запрос должен выбрасывать ToolError."""
    with pytest.raises(ToolError, match=r"(?i)query пуст"):
        search_kb("")


def test_search_kb_whitespace_query_raises(temp_kb):
    """Запрос из одних пробелов тоже должен выбрасывать ToolError."""
    with pytest.raises(ToolError, match=r"(?i)query пуст"):
        search_kb("   ")


def test_search_kb_limit(temp_kb):
    """Параметр top_k ограничивает количество результатов."""
    result = search_kb("календарь пароль 502", top_k=1)
    assert len(result) <= 1


# ============================================================================
# get_article
# ============================================================================

def test_get_article_found(temp_kb):
    """Существующая статья возвращается с корректным текстом."""
    text = get_article("payment-502")
    assert "Ошибка 502" in text
    assert "Bad Gateway" in text


def test_get_article_not_found(temp_kb):
    """Несуществующая статья вызывает ToolError."""
    with pytest.raises(ToolError, match=r"(?i)не найдена"):
        get_article("nonexisting-article")


def test_get_article_path_traversal(temp_kb):
    """Попытка выйти за пределы KB_DIR вызывает ToolError (валидация)."""
    with pytest.raises(ToolError, match=r"(?i)недопустим"):
        get_article("../../etc/passwd")


def test_get_article_with_dotdot_encoded(temp_kb):
    """URL-encoded path traversal тоже должен блокироваться."""
    with pytest.raises(ToolError, match=r"(?i)недопустим"):
        get_article("..%2Fsecret")


def test_get_article_with_spaces(temp_kb):
    """Имя статьи с пробелами не проходит валидацию."""
    with pytest.raises(ToolError, match=r"(?i)недопустим"):
        get_article("with spaces")


# ============================================================================
# get_article_resource
# ============================================================================

def test_get_article_resource_found(temp_kb):
    """Ресурс возвращает текст статьи."""
    text = get_article_resource("password-reset")
    assert "Сброс пароля" in text


def test_get_article_resource_invalid_name(temp_kb):
    """Невалидное имя ресурса (пробелы) вызывает ValueError."""
    with pytest.raises(ValueError, match=r"(?i)недопустим"):
        get_article_resource("with spaces")


def test_get_article_resource_missing(temp_kb):
    """Отсутствующий ресурс вызывает FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match=r"(?i)не найден"):
        get_article_resource("missing")