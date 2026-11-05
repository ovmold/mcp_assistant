"""Юнит-тесты инструментов сервера tickets (без MCP-слоя)."""

import sqlite3

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.prompts.base import UserMessage, AssistantMessage

from mcp_assistant.servers.tickets import (
    search_tickets, get_ticket, add_comment, update_status,
    get_ticket_resource, get_ticket_stats_resource,
    analyze_ticket, _db_cursor
)


def test_ticket_default_lists_are_independent():
    """
    Регресс: два Ticket, созданных без явных messages/logs,
    не должны делить один и тот же список.
    """
    from mcp_assistant.storage.models import Ticket, Message

    t1 = Ticket(id=1, subject="a", body="", status="open",
                priority="low", client="c", created_at="",
                updated_at="", tags="", issue_type="bug")
    t2 = Ticket(id=2, subject="b", body="", status="open",
                priority="low", client="c", created_at="",
                updated_at="", tags="", issue_type="bug")

    t1.messages.append(Message(id=1, author="x", text="y", created_at=""))
    assert t2.messages == [], "Ticket делит общий список messages между экземплярами"
    assert t1.logs is not t2.logs


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


@pytest.mark.parametrize("query", ["платёж", "платеж"])
def test_search_normalizes_russian_e_and_yo(temp_db, query):
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "INSERT INTO tickets (id, subject, body, status, priority, client, "
        "created_at, updated_at, tags, issue_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (6, "Платёжный шлюз недоступен", "Ошибка платёжного шлюза",
         "open", "high", "Acme", "2026-10-14T12:00:00", "2026-10-14T12:00:00",
         "payment", "bug"),
    )
    conn.commit()
    conn.close()

    result = search_tickets(query)

    assert any(ticket.id == 6 for ticket in result)


def test_search_leading_hyphen_is_literal_text(temp_db):
    result = search_tickets("-502")

    assert result
    assert all("502" in ticket.subject or "502" in ticket.body for ticket in result)


@pytest.mark.parametrize("query", ["оплата OR 502", "OR", "AND", "NOT"])
def test_search_fts_operators_are_literal_text(temp_db, query):
    assert isinstance(search_tickets(query), list)


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


@pytest.mark.parametrize("filters,expected_ids", [
    ({"status": "open"}, {1, 3, 5}),
    ({"priority": "high"}, {1, 4}),
    ({"customer": "Globex"}, {1, 4}),
])
def test_search_with_filters_only(temp_db, filters, expected_ids):
    result = search_tickets(**filters)

    assert {ticket.id for ticket in result} == expected_ids


@pytest.mark.parametrize("query", [None, "", "   "])
def test_search_with_empty_query_and_filters(temp_db, query):
    result = search_tickets(
        query=query,
        status="open",
        priority="high",
        customer="Globex",
    )

    assert [ticket.id for ticket in result] == [1]


@pytest.mark.parametrize("filters,expected_ids", [
    ({"created_from": "2026-10-14"}, {1, 2}),
    ({"created_to": "2026-10-13"}, {3, 4, 5}),
    ({"created_from": "2026-10-10", "created_to": "2026-10-14"}, {2, 3, 4}),
])
def test_search_by_created_date(temp_db, filters, expected_ids):
    result = search_tickets(**filters)

    assert {ticket.id for ticket in result} == expected_ids


def test_search_rejects_invalid_date(temp_db):
    with pytest.raises(ToolError, match=r"created_from.*ISO 8601"):
        search_tickets(created_from="не дата")


def test_search_rejects_reversed_date_range(temp_db):
    with pytest.raises(ToolError, match=r"created_from.*позже"):
        search_tickets(
            created_from="2026-10-15",
            created_to="2026-10-14",
        )


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


def test_get_ticket_includes_messages(temp_db):
    """get_ticket возвращает сообщения вместе с тикетом."""
    add_comment(1, "Комментарий 1", author="support")
    add_comment(1, "Комментарий 2", author="client")

    t = get_ticket(1)
    assert len(t.messages) >= 2
    texts = [m.text for m in t.messages]
    assert "Комментарий 1" in texts
    assert "Комментарий 2" in texts
    assert all(m.author in ("support", "client") for m in t.messages)


def test_get_ticket_includes_logs(temp_db):
    """get_ticket возвращает записи лога вместе с тикетом."""
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "INSERT INTO logs (ticket_id, level, message, created_at) "
        "VALUES (?, ?, ?, ?)",
        (1, "ERROR", "502 Bad Gateway", "2024-01-01T10:00:00Z"),
    )
    conn.commit()
    conn.close()

    t = get_ticket(1)
    assert len(t.logs) >= 1
    assert any("502" in l.message for l in t.logs)
    assert any(l.level == "ERROR" for l in t.logs)


def test_get_ticket_empty_messages_and_logs(temp_db):
    """Тикет без сообщений и логов возвращает пустые списки, а не None."""
    t = get_ticket(5)   # у тикета 5 нет активности
    assert t.messages == []
    assert t.logs == []


def test_get_ticket_messages_ordered_by_created_at(temp_db):
    """Сообщения отсортированы по created_at ASC, id ASC."""
    conn = sqlite3.connect(temp_db)
    conn.execute("DELETE FROM messages WHERE ticket_id = 1")
    conn.executemany(
        "INSERT INTO messages (ticket_id, author, text, created_at) "
        "VALUES (?, ?, ?, ?)",
        [
            (1, "a", "третье",  "2024-01-03T00:00:00Z"),
            (1, "a", "первое",  "2024-01-01T00:00:00Z"),
            (1, "a", "второе",  "2024-01-02T00:00:00Z"),
        ],
    )
    conn.commit()
    conn.close()

    t = get_ticket(1)
    texts = [m.text for m in t.messages]
    assert texts == ["первое", "второе", "третье"]


def test_search_tickets_returns_empty_messages_and_logs(temp_db):
    """
    Контракт: search_tickets НЕ грузит messages/logs.
    Полную карточку дает get_ticket(id).
    """
    result = search_tickets("502")
    assert result
    for t in result:
        assert t.messages == []
        assert t.logs == []


# ============================================================================
# add_comment
# ============================================================================

def test_add_comment(temp_db):
    """Комментарий сохраняется и обновляет updated_at тикета."""
    id = 2
    previous_updated_at = get_ticket(id).updated_at
    result = add_comment(id, "Клиент подтвердил проблему", author="support")

    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT id, author, text, created_at FROM messages WHERE ticket_id = ?",
        (id,),
    ).fetchone()
    ticket_updated_at = conn.execute(
        "SELECT updated_at FROM tickets WHERE id = ?",
        (id,),
    ).fetchone()[0]
    conn.close()

    assert row[1:3] == ("support", "Клиент подтвердил проблему")
    assert result.ticket_id == 2
    assert result.comment_id == row[0]
    assert result.added is True
    assert ticket_updated_at == row[3]
    assert ticket_updated_at != previous_updated_at


def test_add_comment_requires_nonempty_author(temp_db):
    with pytest.raises(ToolError, match=r"(?i)author пуст"):
        add_comment(2, "Комментарий", author="   ")


def test_add_comment_to_missing_ticket(temp_db):
    """Комментарий к несуществующему тикету вызывает ToolError."""
    with pytest.raises(ToolError, match=r"(?i)не найден"):
        add_comment(9999, "текст", author="support")

# ============================================================================
# update_status
# ============================================================================

def test_update_status(temp_db):
    """Статус обновляется, добавляется комментарий и возвращается результат."""
    result = update_status(1, "closed", "Проблема решена")

    conn = sqlite3.connect(temp_db)
    status = conn.execute("SELECT status FROM tickets WHERE id = 1").fetchone()[0]
    msg = conn.execute(
        "SELECT id, text FROM messages WHERE ticket_id = 1"
    ).fetchone()
    conn.close()

    assert status == "closed"
    assert msg[1] == "Проблема решена"
    assert result.ticket_id == 1
    assert result.old_status == "open"
    assert result.new_status == "closed"
    assert result.updated is True
    assert result.comment_id == msg[0]


def test_update_status_invalid_value(temp_db):
    """Недопустимый статус вызывает ToolError."""
    with pytest.raises(ToolError, match=r"(?i)неправильный"):
        update_status(1, "garbage", "причина")


def test_update_status_missing_ticket(temp_db):
    """Смена статуса у несуществующего тикета вызывает ToolError."""
    with pytest.raises(ToolError, match=r"(?i)не найден"):
        update_status(9999, "closed", "причина")


def test_update_status_rejects_unchanged_status(temp_db):
    previous_updated_at = get_ticket(1).updated_at

    with pytest.raises(ToolError, match=r"уже имеет статус open"):
        update_status(1, "open", "Статус уже установлен")

    ticket = get_ticket(1)
    assert ticket.status == "open"
    assert ticket.updated_at == previous_updated_at
    assert ticket.messages == []


def test_update_status_cannot_reopen_closed_ticket(temp_db):
    update_status(1, "closed", "Закрыт")

    with pytest.raises(ToolError, match=r"closed на open"):
        update_status(1, "open", "Повторное открытие")

    ticket = get_ticket(1)
    assert ticket.status == "closed"
    assert len(ticket.messages) == 1


def test_db_cursor_enables_foreign_keys(temp_db):
    with _db_cursor() as cursor:
        assert cursor.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                "INSERT INTO messages (ticket_id, author, text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (9999, "support", "orphan", "2026-10-14T12:00:00"),
            )


def test_fts_update_trigger_is_limited_to_indexed_columns(temp_db):
    conn = sqlite3.connect(temp_db)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'tickets_au'"
    ).fetchone()[0]
    conn.close()

    assert "AFTER UPDATE OF subject, body" in " ".join(sql.split())


def test_update_status_fts_sync(temp_db):
    """
    Проверяет, что триггер FTS обновил индекс после смены subject.
    
    Если триггер не работает, поиск по новой теме не найдет тикет.
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
    assert "## Переписка" in result
    assert "## Лог" in result


def test_get_ticket_resource_invalid_id_format(temp_db):
    """Нечисловой ID вызывает ValueError с понятным сообщением."""
    with pytest.raises(ValueError, match=r"(?i)неправильный id"):
        get_ticket_resource("abc")


def test_get_ticket_resource_not_found(temp_db):
    """Несуществующий ID вызывает ValueError."""
    with pytest.raises(ValueError, match=r"(?i)не найден"):
        get_ticket_resource("9999")


def test_get_ticket_resource_shows_messages(temp_db):
    """Сообщения попадают в карточку ресурса."""
    add_comment(1, "Уникальный текст сообщения", author="support")
    result = get_ticket_resource("1")

    assert "Уникальный текст сообщения" in result
    assert "support" in result


def test_get_ticket_resource_empty_sections(temp_db):
    """Для тикета без активности отображаются заглушки."""
    t = get_ticket_resource("5")   # тикет без messages/logs
    assert "_Сообщений нет._" in t
    assert "_Записей лога нет._" in t


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