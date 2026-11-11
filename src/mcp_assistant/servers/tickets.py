"""
MCP-сервер `tickets` (транспорт stdio)

Сервер работает поверх SQLite. Основная таблица tickets хранит данные тикета,
включая тему, текст, клиента, приоритет, статус, даты, теги и тип проблемы.
Комментарии и логи хранятся в связанных таблицах messages и logs.

- **Tools:** `search_tickets(query?, status?, priority?, customer?,
  created_from?, created_to?, limit)`, `get_ticket(ticket_id)`,
  `add_comment(ticket_id, text, author)`,
  `update_status(ticket_id, status, comment, author?)`.
- **Resources:** `ticket://{ticket_id}` — полная карточка тикета;
  `tickets://stats` — сводка по статусам и приоритетам.
- **Prompts:** `analyze_ticket(ticket_id)` — шаблон разбора тикета:
  суть, вероятная причина, следующий шаг.
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.prompts.base import UserMessage, AssistantMessage
from mcp.types import ToolAnnotations
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import sqlite3
import re

from mcp_assistant import config as conf
from mcp_assistant.storage.models import (
    TICKET_STATUSES,
    CommentAddResult,
    LogEntry,
    Message,
    StatusUpdateResult,
    Ticket,
    TicketPriority,
    TicketStatus,
)
from mcp_assistant.storage.timestamps import to_iso

mcp = MCPServer("tickets")


@contextmanager
def _db_cursor() -> Iterator[sqlite3.Cursor]:
    """Гарантирует закрытие соединения и курсора при любом исходе."""
    conn = sqlite3.connect(conf.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        conn.close()

def _parse_date_filter(value: str | None, name: str) -> tuple[datetime, bool] | None:
    """Разбирает ISO 8601 и возвращает (время UTC, признак даты без времени) или None.

    Дата-время без часового пояса считается заданной в UTC.
    """
    if value is None or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as e:
        raise ToolError(
            f"Параметр {name} должен быть датой или датой-временем в ISO 8601."
        ) from e

    date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc), date_only


# ----------------------------- Tools ------------------------

@mcp.tool(
    annotations = ToolAnnotations(
        readOnlyHint=True
    )
)
def search_tickets(
    query: str | None = None,
    limit: int = 10,
    status: TicketStatus | None = None,
    priority: TicketPriority | None = None,
    customer: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
) -> list[Ticket]:
    """Ищет тикеты по ключевым словам и необязательным фильтрам.

    Фильтры: status, priority, customer и диапазон created_at через
    created_from/created_to в ISO 8601. Дата без времени в created_to включает
    весь этот день. query необязателен при наличии любого фильтра.
    """
    limit = max(1, min(limit, conf.N_SEARCH_LIMIT))
    filters = [
        ("status", status.strip() if status else None),
        ("priority", priority.strip() if priority else None),
        ("client", customer.strip() if customer else None),
    ]
    filters = [(column, value) for column, value in filters if value]

    from_filter = _parse_date_filter(created_from, "created_from")
    to_filter = _parse_date_filter(created_to, "created_to")
    if from_filter and to_filter and from_filter[0] > to_filter[0]:
        raise ToolError("created_from не может быть позже created_to.")

    to_operator = "<="
    if to_filter is not None:
        to_date, date_only = to_filter
        if date_only:
            to_date += timedelta(days=1)
            to_operator = "<"
        to_filter = (to_date, date_only)

    safe_query = ""
    if query and query.strip():
        words = re.findall(r"\w+", query.lower().replace("ё", "е"), flags=re.UNICODE)
        words = [word[:5] if len(word) >= 6 else word for word in words]
        safe_query = " ".join(f'"{word}"*' for word in words)

    if not safe_query and not filters and not from_filter and not to_filter:
        raise ToolError(
            "Параметр query пуст и не заданы фильтры. Уточните у пользователя "
            "ключевые слова, статус, приоритет, клиента или диапазон дат."
        )

    where_parts = []
    params = []
    if safe_query:
        where_parts.append("tickets_fts MATCH ?")
        params.append(safe_query)
    for column, value in filters:
        where_parts.append(f"t.{column} = ?")
        params.append(value)
    if from_filter:
        where_parts.append("datetime(t.created_at) >= datetime(?)")
        params.append(to_iso(from_filter[0]))
    if to_filter:
        where_parts.append(f"datetime(t.created_at) {to_operator} datetime(?)")
        params.append(to_iso(to_filter[0]))
    params.append(limit)

    if safe_query:
        from_clause = "tickets t JOIN tickets_fts fts ON t.id = fts.rowid"
        order_clause = "bm25(tickets_fts), t.created_at DESC"
    else:
        from_clause = "tickets t"
        order_clause = "t.created_at DESC"

    sql = f"""
        SELECT t.*
        FROM {from_clause}
        WHERE {" AND ".join(where_parts)}
        ORDER BY {order_clause}
        LIMIT ?
    """

    try:
        with _db_cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [Ticket(**dict(row)) for row in rows]
    except Exception as e:
        raise ToolError(f"Ошибка поиска тикетов: {e}") from e

def _load_ticket(ticket_id: int) -> Ticket | None:
    with _db_cursor() as cursor:
        cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        if row is None:
            return None

        cursor.execute(
            "SELECT id, author, text, created_at FROM messages "
            "WHERE ticket_id = ? ORDER BY created_at ASC, id ASC",
            (ticket_id,),
        )
        messages = [Message(**dict(message)) for message in cursor.fetchall()]

        cursor.execute(
            "SELECT id, level, message, created_at FROM logs "
            "WHERE ticket_id = ? ORDER BY created_at ASC, id ASC",
            (ticket_id,),
        )
        logs = [LogEntry(**dict(log)) for log in cursor.fetchall()]

    return Ticket(**dict(row), messages=messages, logs=logs)


@mcp.tool(
    annotations = ToolAnnotations(
        readOnlyHint=True
    )
)
def get_ticket(ticket_id: int) -> Ticket:
    """Возвращает полный тикет по ID, включая переписку и логи."""
    ticket = _load_ticket(ticket_id)
    if ticket is None:
        raise ToolError(f"Тикет {ticket_id} не найден.")
    return ticket
    

@mcp.tool(
    annotations = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True
    )
)
def add_comment(
    ticket_id: int,
    text: str,
    author: str,
) -> CommentAddResult:
    """
    Добавляет комментарий в таблицу messages к существующему тикету.
    
    Используйте этот инструмент, когда нужно оставить сообщение, заметку 
    или ответ в рамках тикета. Комментарий сохраняется с указанием автора 
    и текущего времени (UTC).
    
    Инструмент сначала проверяет, что тикет с указанным ID существует.
    Если тикет не найден — операция отменяется с ошибкой.
    
    Args:
        ticket_id: Уникальный идентификатор тикета (целое число), 
            к которому добавляется комментарий. Пример: 1042.
        text: Текст комментария. Может содержать любой текст, 
            включая переносы строк и специальные символы.
        author: Обязательный автор комментария, например 'support',
            'user_123', 'manager' или 'customer'. Если автор неизвестен,
            агент должен запросить его у пользователя до вызова инструмента.
    
    Returns:
        CommentAddResult: ID тикета и нового комментария, added=True после commit.
    
    Raises:
        ToolError:
            Если тикет с указанным ticket_id не найден в базе данных.
            Если текст комментария или автор пусты.
    """
    if not author or not author.strip():
        raise ToolError(
            "Параметр author пуст. Уточните у пользователя автора комментария."
        )

    if not text or not text.strip():
        raise ToolError(
            "Параметр text пуст. Пожалуйста, уточните у пользователя "
            "комментарий для добавления к тикету "
            "(например: 'Клиент подтвердил проблему')."
        )
    
    with _db_cursor() as cursor:
        cursor.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,))
        if cursor.fetchone() is None:
            raise ToolError(f"Тикет {ticket_id} не найден.")

        created_at = to_iso(datetime.now(tz=timezone.utc))
        cursor.execute(
            "INSERT INTO messages (ticket_id, author, text, created_at) "
            "VALUES (?, ?, ?, ?)",
            (ticket_id, author, text, created_at),
        )
        comment_id = cursor.lastrowid
        cursor.execute(
            "UPDATE tickets SET updated_at = ? WHERE id = ?",
            (created_at, ticket_id),
        )
        cursor.connection.commit()

    return CommentAddResult(ticket_id=ticket_id, comment_id=comment_id, added=True)


@mcp.tool(
    annotations = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True
    )
)
def update_status(
    ticket_id: int,
    status: TicketStatus,
    comment: str,
    author: str = "support",
) -> StatusUpdateResult:
    """
    Обновляет статус тикета и добавляет комментарий с объяснением изменения.
    
    Используйте этот инструмент, когда нужно перевести тикет в новый статус
    (например, закрыть, отложить или пометить как решенный). Инструмент
    выполняет два действия в одной транзакции:
    1. Меняет статус тикета и обновляет поле updated_at в таблице tickets.
    2. Добавляет комментарий с указанием автора и причины изменения в таблицу messages.
    
    Инструмент сначала проверяет, что тикет существует и что переданный
    статус является допустимым. Вызов с неизменным статусом и переход
    closed → open запрещены. Если проверка не пройдена — изменения не применяются.
    
    Args:
        ticket_id: Уникальный идентификатор тикета (целое число),
            статус которого нужно изменить. Пример: 1042.
        status: Новый статус тикета. Допустимые значения:
            - "open" — тикет открыт, требует внимания
            - "pending" — тикет в ожидании (ответа клиента, ресурсов и т.д.)
            - "resolved" — проблема решена, но тикет ещё не закрыт
            - "closed" — тикет закрыт окончательно
            Значение должно точно совпадать с одним из перечисленных.
        comment: Текст комментария, объясняющий причину смены статуса.
            Может содержать любой текст, включая переносы строк.
            Пример: "Проблема решена, клиент подтвердил."
        author: Автор комментария. По умолчанию 'support'.
            Используйте другие значения для указания конкретного
            пользователя, например 'user_123' или 'manager' или клиента - 'customer'.
    
    Returns:
        StatusUpdateResult: старый и новый статус, ID комментария,
        updated=True после успешного commit.
    
    Raises:
        ToolError:
            - Если тикет с указанным ticket_id не найден.
            - Если переданный status не входит в список допустимых значений.
            - Если текст комментария пуст.
    """

    # Проверка на пустой комментарий
    if not comment or not comment.strip():
        raise ToolError(
            "Параметр comment пуст. Пожалуйста, уточните у пользователя "
            "комментарий для обновления статуса тикета "
            "(например: 'Проблема решена, клиент подтвердил.')."
        )
    
    with _db_cursor() as cursor:
        cursor.execute("SELECT status FROM tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        if row is None:
            raise ToolError(f"Тикет {ticket_id} не найден.")
        old_status = row["status"]

        if status not in TICKET_STATUSES:
            raise ToolError(f"Неправильный статус: {status}.")
        if old_status == status:
            raise ToolError(f"Тикет {ticket_id} уже имеет статус {status}.")
        if old_status == "closed" and status == "open":
            raise ToolError("Запрещена смена статуса с closed на open.")

        changed_at = to_iso(datetime.now(tz=timezone.utc))
        cursor.execute(
            "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
            (status, changed_at, ticket_id),
        )
        cursor.execute(
            "INSERT INTO messages (ticket_id, author, text, created_at) "
            "VALUES (?, ?, ?, ?)",
            (ticket_id, author, comment, changed_at),
        )
        comment_id = cursor.lastrowid
        cursor.connection.commit()

    return StatusUpdateResult(
        ticket_id=ticket_id,
        old_status=old_status,
        new_status=status,
        updated=True,
        comment_id=comment_id,
    )

# ----------------------------- Resources ------------------------
@mcp.resource("ticket://{ticket_id}")
def get_ticket_resource(ticket_id: str) -> str:
    """
    Возвращает полную карточку тикета по его ID.

    Ресурс доступен по адресу: ticket://{ticket_id}
    Пример: ticket://1042

    Args:
        ticket_id: ID тикета (в URI передается как строка).

    Returns:
        str: Текстовая карточка тикета со всеми полями, перепиской
        и записями лога в формате Markdown.

    Raises:
        ValueError:
            - Если тикет с таким ID не найден
            - Если ID тикета не может быть преобразован в int
    """

    try:
        tid = int(ticket_id)
    except ValueError as e:
        raise ValueError(f"Неправильный ID тикета: {ticket_id}.") from e

    ticket = _load_ticket(tid)
    if ticket is None:
        raise ValueError(f"Тикет {ticket_id} не найден.")

    ticket_card = [
        f"# Тикет #{ticket.id}",
        "",
        f"**Тема:** {ticket.subject}",
        f"**Статус:** {ticket.status}",
        f"**Приоритет:** {ticket.priority}",
        f"**Клиент:** {ticket.client}",
        f"**Тип:** {ticket.issue_type}",
        f"**Тэги:** {ticket.tags}",
        "",
        f"**Создан:** {ticket.created_at}",
        f"**Изменен:** {ticket.updated_at}",
        "",
        "## Описание",
        "",
        ticket.body,
        "",
        f"## Переписка ({len(ticket.messages)})",
        "",
    ]

    if not ticket.messages:
        ticket_card.extend(["_Сообщений нет._", ""])
    else:
        for message in ticket.messages:
            ticket_card.extend([
                f"### {message.author} — {message.created_at} (сообщение #{message.id})",
                "",
                message.text,
                "",
            ])

    ticket_card.extend([f"## Лог ({len(ticket.logs)})", ""])
    if not ticket.logs:
        ticket_card.append("_Записей лога нет._")
    else:
        for log in ticket.logs:
            ticket_card.append(
                f"- **[{log.level}]** `{log.created_at}` "
                f"(запись #{log.id}): {log.message}"
            )

    return "\n".join(ticket_card)

@mcp.resource("tickets://stats")
def get_ticket_stats_resource() -> str:
    """
    Возвращает сводку по всем тикетам: количество по статусам и приоритетам.
    
    Ресурс доступен по адресу: tickets://stats
    
    Returns:
        str: Текстовая сводка в формате Markdown.
    """
    with _db_cursor() as cursor:
        cursor.execute(
            "SELECT status, COUNT(*) FROM tickets "
            "GROUP BY status ORDER BY COUNT(*) DESC"
        )
        by_status = cursor.fetchall()

        cursor.execute(
            "SELECT priority, COUNT(*) FROM tickets "
            "GROUP BY priority ORDER BY COUNT(*) DESC"
        )
        by_priority = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) FROM tickets")
        total = cursor.fetchone()[0]

    stats = [
        "# Статистика по тикетам",
        "",
        f"**Всего тикетов:** {total}",
        "",
        "## По статусам",
        "",
    ]
    stats.extend(f"- **{status}:** {count}" for status, count in by_status)
    stats.extend(["", "## По приоритетам", ""])
    stats.extend(f"- **{priority}:** {count}" for priority, count in by_priority)
    return "\n".join(stats)

# ----------------------------- Prompts ------------------------
@mcp.prompt()
def analyze_ticket(ticket_id: str) -> list:
    """
    Шаблон разбора тикета: суть, вероятная причина, следующий шаг.
    
    Этот промпт предназначен для быстрого анализа тикета. Он формирует
    диалог, в котором пользователь просит разобрать тикет, а ассистент
    подтверждает готовность. Модель затем сама получает карточку тикета
    через ресурс `ticket://{ticket_id}` и выдает разбор в заданном формате.
    
    Используйте этот промпт, когда нужно:
    - Быстро понять суть проблемы.
    - Получить гипотезу о причине.
    - Определить конкретный следующий шаг.
    
    Args:
        ticket_id: ID тикета в виде строки (например, "1042").
            Передается как часть URI ресурса `ticket://{ticket_id}`.
    
    Returns:
        list: Список из двух сообщений:
            - UserMessage: запрос на разбор тикета с описанием формата.
            - AssistantMessage: подтверждение готовности выполнить разбор.
    
    Note:
        Промпт не выполняет код и не читает данные сам. Он только
        формирует текст запроса. Модель должна самостоятельно вызвать
        ресурс `ticket://{ticket_id}` для получения данных.
    """
    
    return [
        UserMessage(
            f"Разбери тикет #{ticket_id}.\n\n"
            f"Сначала получи полную карточку тикета через ресурс `ticket://{ticket_id}`. "
            "Если нужно, посмотри комментарии и связанные данные.\n\n"
            "Затем дай разбор в следующем формате:\n\n"
            "## Суть\n"
            "Одно-два предложения: в чем проблема, что просит клиент.\n\n"
            "## Вероятная причина\n"
            "Одно-два предложения: почему это могло произойти. "
            "Если данных недостаточно — предположи наиболее вероятный сценарий.\n\n"
            "## Следующий шаг\n"
            "Одно конкретное действие, которое нужно сделать прямо сейчас. "
            "Например: 'Запросить у клиента логи', 'Воспроизвести на Safari 17.4', "
            "'Передать в команду бэкенда'.\n\n"
            "Будь кратким и конкретным. Не пересказывай тикет целиком."
        ),
        AssistantMessage(
            "Понял. Сейчас получу карточку тикета и сделаю разбор."
        )
    ]

if __name__ == "__main__":
    # Загружает локальную конфигурацию и обновляет пути перед запуском сервера
    conf.load_env(conf.PROJECT_DIR / ".env")
    mcp.run()