"""Юнит-тесты инструментов сервера tickets (без MCP-слоя)."""

import sqlite3

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.prompts.base import UserMessage, AssistantMessage

from servers.tickets import (
    search_tickets, get_ticket, add_comment, update_status,
    get_ticket_resource, get_ticket_stats_resource,
    analyze_ticket
)


# ============================================================================
# search_tickets
# ============================================================================

def test_search_by_subject(temp_db):
    """Поиск по слову, которое есть в subject (тикеты 1 и 4)."""
    result = search_tickets("502")
    assert len(result) >= 2
    assert all("502" in t.subject or "502" in t.body for t in result)


def test_search_by_body(temp_db):
    """Поиск по слову, которое есть только в body."""
    result = search_tickets("Bad Gateway")
    assert len(result) >= 1
    assert any("502" in t.subject for t in result)


def test_search_phrase_with_extra_words(temp_db):
    """
    Регресс: запрос с лишним словом и другой словоформой должен находить тикет.
    
    Запрос «ошибка 502 оплата» должен находить «Ошибка 502 при оплате»,
    несмотря на слово «при» и форму «оплате» вместо «оплата».
    """
    result = search_tickets("ошибка 502 оплата")
    assert len(result) >= 1
    subjects = [t.subject for t in result]
    assert "Ошибка 502 при оплате" in subjects


@pytest.mark.parametrize("query", [
    "ошибка 502 оплата",
    "оплата ошибка 502",   # другой порядок
    "оплате 502",          # другая форма
    "502 при оплате",      # с «при» в запросе
])
def test_search_finds_502_ticket(temp_db, query):
    """Разные формулировки одного и того же запроса должны находить тикет про 502."""
    result = search_tickets(query)
    assert any("502" in t.subject for t in result), f"Не найдено для запроса: {query}"


def test_search_case_insensitive(temp_db):
    """Регистр не должен влиять на результат (для ASCII и кириллицы)."""
    r1 = search_tickets("ОШИБКА")
    r2 = search_tickets("ошибка")
    assert len(r1) == len(r2)


def test_search_empty_query_raises(temp_db):
    """Пустой запрос должен выбрасывать ToolError."""
    with pytest.raises(ToolError, match=r"(?i)пуст"):
        search_tickets("")


def test_search_whitespace_query_raises(temp_db):
    """Запрос из одних пробелов тоже должен выбрасывать ToolError."""
    with pytest.raises(ToolError, match=r"(?i)пуст"):
        search_tickets("   ")


def test_search_filter_by_status(temp_db):
    """Фильтр по статусу оставляет только тикеты с этим статусом."""
    result = search_tickets("502", status="open")
    assert len(result) >= 1
    assert all(t.status == "open" for t in result)


def test_search_filter_by_priority(temp_db):
    """Фильтр по приоритету оставляет только тикеты с этим приоритетом."""
    result = search_tickets("502", priority="high")
    assert len(result) >= 1
    assert all(t.priority == "high" for t in result)


def test_search_filter_by_customer(temp_db):
    """Фильтр по клиенту оставляет только тикеты этого клиента."""
    result = search_tickets("502", customer="Globex")
    assert len(result) >= 1
    assert all(t.client == "Globex" for t in result)


def test_search_limit(temp_db):
    """limit ограничивает количество результатов."""
    result = search_tickets("ошибка", limit=1)
    assert len(result) <= 1


def test_search_limit_clamped_to_min(temp_db):
    """limit=0 должен стать 1, а не упасть или вернуть пусто."""
    result = search_tickets("502", limit=0)
    assert 1 <= len(result) <= 1


def test_search_injection_in_ticket_does_not_break_search(temp_db):
    """
    Тикет-ловушка не должен ломать поиск и не должен исполняться.
    
    Текст из базы возвращается как данные. Агент (LLM) должен
    игнорировать инструкции внутри <tool_result>.
    """
    result = search_tickets("инструкции")
    assert len(result) >= 1
    assert any("Игнорируй" in t.body for t in result)


# ============================================================================
# get_ticket
# ============================================================================

def test_get_ticket_found(temp_db):
    """Существующий тикет возвращается с корректными данными."""
    t = get_ticket(1)
    assert t.id == 1
    assert t.subject == "Ошибка 502 при оплате"


def test_get_ticket_not_found(temp_db):
    """Несуществующий ID вызывает ToolError."""
    with pytest.raises(ToolError, match=r"(?i)не найден"):
        get_ticket(9999)


# ============================================================================
# add_comment
# ============================================================================

def test_add_comment(temp_db):
    """Комментарий сохраняется в таблицу messages."""
    add_comment(1, "Клиент подтвердил проблему", author="support")

    conn = sqlite3.connect(temp_db)
    rows = conn.execute(
        "SELECT author, text FROM messages WHERE ticket_id = 1"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0] == ("support", "Клиент подтвердил проблему")


def test_add_comment_to_missing_ticket(temp_db):
    """Комментарий к несуществующему тикету вызывает ToolError."""
    with pytest.raises(ToolError, match=r"(?i)не найден"):
        add_comment(9999, "текст")


# ============================================================================
# update_status
# ============================================================================

def test_update_status(temp_db):
    """Статус обновляется, и добавляется комментарий с причиной."""
    update_status(1, "closed", "Проблема решена")

    conn = sqlite3.connect(temp_db)
    status = conn.execute("SELECT status FROM tickets WHERE id = 1").fetchone()[0]
    msg = conn.execute(
        "SELECT text FROM messages WHERE ticket_id = 1"
    ).fetchone()
    conn.close()

    assert status == "closed"
    assert msg[0] == "Проблема решена"


def test_update_status_invalid_value(temp_db):
    """Недопустимый статус вызывает ToolError."""
    with pytest.raises(ToolError, match=r"(?i)неправильный"):
        update_status(1, "garbage", "причина")


def test_update_status_missing_ticket(temp_db):
    """Смена статуса у несуществующего тикета вызывает ToolError."""
    with pytest.raises(ToolError, match=r"(?i)не найден"):
        update_status(9999, "closed", "причина")


def test_update_status_fts_sync(temp_db):
    """
    Проверяет, что триггер FTS обновил индекс после смены subject.
    
    Если триггер не работает, поиск по новой теме не найдёт тикет.
    """
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE tickets SET subject = 'Новая тема про 502' WHERE id = 1")
    conn.commit()
    conn.close()

    result = search_tickets("Новая тема")
    assert any(t.id == 1 for t in result)

# ============================================================================
# Ресурс: get_ticket_resource (ticket://{ticket_id})
# ============================================================================

def test_get_ticket_resource_found(temp_db):
    """Существующий тикет возвращается в виде Markdown-карточки."""
    result = get_ticket_resource("1")
    
    # Проверяем заголовок и ключевые поля
    assert "# Тикет #1" in result
    assert "**Тема:** Ошибка 502 при оплате" in result
    assert "**Статус:** open" in result
    assert "**Приоритет:** high" in result
    assert "**Клиент:** Globex" in result
    assert "**Тип:** bug" in result
    assert "**Тэги:** payment,502" in result
    assert "## Описание" in result
    assert "При попытке оплаты возникает 502 Bad Gateway" in result


def test_get_ticket_resource_invalid_id_format(temp_db):
    """Нечисловой ID вызывает ValueError с понятным сообщением."""
    with pytest.raises(ValueError, match=r"(?i)неправильный id"):
        get_ticket_resource("abc")


def test_get_ticket_resource_not_found(temp_db):
    """Несуществующий ID вызывает ValueError."""
    with pytest.raises(ValueError, match=r"(?i)не найден"):
        get_ticket_resource("9999")


# ============================================================================
# Ресурс: get_ticket_stats_resource (tickets://stats)
# ============================================================================

def test_get_ticket_stats_resource(temp_db):
    """Ресурс возвращает корректную статистику по статусам и приоритетам."""
    result = get_ticket_stats_resource()
    
    # Проверяет общее количество
    assert "# Статистика по тикетам" in result
    assert "**Всего тикетов:** 5" in result
    
    # Проверяет статистику по статусам (из SEED_TICKETS: open=3, pending=1, resolved=1)
    assert "## По статусам" in result
    assert "- **open:** 3" in result
    assert "- **pending:** 1" in result
    assert "- **resolved:** 1" in result
    
    # Проверяет статистику по приоритетам (из SEED_TICKETS: high=2, low=2, medium=1)
    assert "## По приоритетам" in result
    assert "- **high:** 2" in result
    assert "- **low:** 2" in result
    assert "- **medium:** 1" in result


def test_get_ticket_stats_resource_empty_db(temp_db):
    """Пустая база данных (без тикетов) возвращает нулевую статистику."""
    # Очищаем таблицу tickets прямо в тестовой БД
    conn = sqlite3.connect(temp_db)
    conn.execute("DELETE FROM logs")
    conn.execute("DELETE FROM messages")
    conn.execute("DELETE FROM tickets")
    conn.commit()
    conn.close()

    result = get_ticket_stats_resource()
    
    assert "**Всего тикетов:** 0" in result
    # Секции "По статусам" и "По приоритетам" останутся, но без элементов списка
    assert "- **open:**" not in result
    assert "- **high:**" not in result


# ============================================================================
# Prompt: analyze_ticket
# ============================================================================

def test_analyze_ticket_returns_two_messages():
    """Промпт возвращает ровно два сообщения: запрос и подтверждение."""
    result = analyze_ticket("1042")
    assert isinstance(result, list)
    assert len(result) == 2


def test_analyze_ticket_contains_ticket_id():
    """ID тикета подставляется в текст запроса (а не остается шаблоном)."""
    result = analyze_ticket("1042")
    user_text = result[0].content.text  

    assert "1042" in user_text
    assert "{ticket_id}" not in user_text  # шаблон не должен остаться


def test_analyze_ticket_contains_required_sections():
    """Промпт содержит все три секции, которые ожидает LLM."""
    result = analyze_ticket("1042")
    user_text = result[0].content.text

    assert "## Суть" in user_text
    assert "## Вероятная причина" in user_text
    assert "## Следующий шаг" in user_text


def test_analyze_ticket_mentions_resource_uri():
    """Промпт явно указывает, что данные нужно взять из ресурса ticket://..."""
    result = analyze_ticket("1042")
    user_text = result[0].content.text

    assert "ticket://1042" in user_text


def test_analyze_ticket_message_types():
    """Первое сообщение — UserMessage, второе — AssistantMessage."""
    result = analyze_ticket("1042")
    assert isinstance(result[0], UserMessage)
    assert isinstance(result[1], AssistantMessage)