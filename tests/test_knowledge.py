"""Юнит-тесты инструментов сервера knowledge (без MCP-слоя)."""

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_assistant.servers import knowledge as knowledge_module
from mcp_assistant.servers.knowledge import (
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


def test_search_kb_exact_code_outranks_similar_code(temp_kb):
    (temp_kb / "upload-500.md").write_text(
        "# Ошибка 500 при загрузке\n\nОшибка при загрузке аватара.",
        encoding="utf-8",
    )

    result = search_kb("502")

    assert result[0] == "payment-502"


def test_search_kb_password_reset_excludes_unrelated_articles(temp_kb):
    (temp_kb / "password-reset-email.md").write_text(
        "# Сброс пароля\n\nИнструкция по сбросу пароля и письмо для восстановления.",
        encoding="utf-8",
    )
    (temp_kb / "security-report.md").write_text(
        "# Сообщение об уязвимости\n\nКак сообщить о проблеме безопасности.",
        encoding="utf-8",
    )
    (temp_kb / "double-charge.md").write_text(
        "# Двойное списание\n\nПовторное списание с карты клиента.",
        encoding="utf-8",
    )
    (temp_kb / "password-reset.md").unlink()

    result = search_kb("сброс пароля")

    assert result[0] == "password-reset-email"
    assert "security-report" not in result
    assert "double-charge" not in result


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


def test_get_article_resource_invalid_encoding(temp_kb):
    """Файл в некорректной кодировке вызывает UnicodeDecodeError."""
    (temp_kb / "broken.md").write_bytes(b"\xff\xfe\xff")

    with pytest.raises(UnicodeDecodeError):
        get_article_resource("broken")


def test_main_initializes_kb_before_serving(monkeypatch):
    events = []
    monkeypatch.setattr(
        knowledge_module.conf, "load_env", lambda path: events.append("env")
    )
    monkeypatch.setattr(
        knowledge_module, "_init_kb", lambda: events.append("init")
    )
    monkeypatch.setattr(
        knowledge_module.mcp,
        "run",
        lambda **kwargs: events.append("run"),
    )

    knowledge_module.main()

    assert events == ["env", "init", "run"]