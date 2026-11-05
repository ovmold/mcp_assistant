"""
MCP-сервер `tickets` (транспорт stdio)

Сервер работает поверх SQLite с тикетами. В таблице хранятся id, клиент, тема, текст, приоритет, статус, даты создания и обновления.

- **Tools:** `search_tickets(query, status?, priority?, customer?, limit)`, `get_ticket(id)`, `add_comment(id, text)`, `update_status(id, status, comment)`.
- **Resources:** `ticket://{id}` — полная карточка тикета; `tickets://stats` — сводка по статусам и приоритетам.
- **Prompts:** `analyze_ticket(id)` — шаблон разбора тикета: суть, вероятная причина, следующий шаг.
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.prompts.base import UserMessage, AssistantMessage
from mcp.types import ToolAnnotations
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite3
import re

from mcp_assistant import config as conf
from mcp_assistant import seed_data
from mcp_assistant.storage.models import Ticket, Message, LogEntry

mcp = MCPServer("tickets")


@contextmanager
def _db_cursor() -> Iterator[sqlite3.Cursor]:
    """Гарантирует закрытие соединения и курсора при любом исходе."""
    conn = sqlite3.connect(conf.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        conn.close()

# ----------------------------- Tools ------------------------

@mcp.tool(
    annotations = ToolAnnotations(
        readOnlyHint=True
    )
)
def search_tickets(
    query: str,
    limit: int = 10,
    status: str = None,
    priority: str = None,
    customer: str = None,
) -> list[Ticket]:
    """
    Ищет тикеты по ключевому слову в теме или описании (полнотекстовый поиск FTS5).
    Дополнительно может фильтровать по статусу, приоритету или заказчику.

    Args:
        query: Ключевое слово или фраза для поиска. Не может быть пустым.
        limit: Максимальное количество возвращаемых тикетов (1–100).
        status: Необязательный фильтр по статусу ('open', 'closed' и т.д.).
        priority: Необязательный фильтр по приоритету ('high', 'low' и т.д.).
        customer: Необязательный фильтр по имени заказчика.

    Returns:
        Список найденных тикетов, отсортированных по релевантности (bm25),
        затем по дате создания (сначала новые). Поля messages и logs в результатах поиска
        всегда пустые — для получения переписки и лога вызовите get_ticket(id).
    Raises:
        ToolError: Если ключевое слово или фраза для поиска (query) пустое.
    """
    # Ограничивает limit
    limit = max(1, min(limit, conf.N_SEARCH_LIMIT))

    # Проверка на пустой запрос
    if not query or not query.strip():
        raise ToolError(
            "Параметр query пуст. Пожалуйста, уточните у пользователя, "
            "что именно искать в тикетах (например: 'ошибка 502', "
            "'сброс пароля', 'оплата')."
        )

    # Очищает запрос от спецсимволов FTS5 (кавычки, звёздочки, скобки)
    # иначе MATCH может выдать синтаксическую ошибку.
    safe_query = re.sub(r'[^\w\s\-]', ' ', query).strip()
    if not safe_query:
        return []

    # Грубый стемминг: обрезает длинные слова до 5 символов:
    # "оплата" -> "оплат", "ошибка" -> "ошиб"
    words = [w[:5] + '*' if len(w) >= 6 else w + '*' for w in safe_query.split()]
    safe_query = ' '.join(words)

    # Базовый WHERE: MATCH по FTS5
    where_clause = "tickets_fts MATCH ?"
    params = [safe_query]

    # Дополнительные фильтры (по основной таблице tickets)
    if status is not None:
        where_clause += " AND t.status = ?"
        params.append(status)
    if priority is not None:
        where_clause += " AND t.priority = ?"
        params.append(priority)
    if customer is not None:
        where_clause += " AND t.client = ?"
        params.append(customer)

    params.append(limit)

    # Запрос: JOIN с FTS5, ранжирование по bm25
    sql = f"""
        SELECT t.*
        FROM tickets t
        JOIN tickets_fts fts ON t.id = fts.rowid
        WHERE {where_clause}
        ORDER BY bm25(tickets_fts), t.created_at DESC
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


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
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
    author: str = "support"
) -> None:
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
        author: Автор комментария. По умолчанию 'support'. 
            Используйте другие значения для указания конкретного пользователя, 
            например 'user_123' или 'manager' или клиента - 'customer'.
    
    Returns:
        None: Функция ничего не возвращает при успешном выполнении.
    
    Raises:
        ToolError: 
            Если тикет с указанным ticket_id не найден в базе данных.
            Если текст комментария пуст.
    """
    # Проверка на пустой комментарий
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

        cursor.execute(
            "INSERT INTO messages (ticket_id, author, text, created_at) "
            "VALUES (?, ?, ?, ?)",
            (ticket_id, author, text, seed_data._iso(datetime.now(tz=timezone.utc)))
        )
        cursor.connection.commit()


@mcp.tool(
    annotations = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True
    )
)
def update_status(
    ticket_id: int, 
    status: str, 
    comment: str, 
    author: str = "support"
) -> None:
    """
    Обновляет статус тикета и добавляет комментарий с объяснением изменения.
    
    Используйте этот инструмент, когда нужно перевести тикет в новый статус
    (например, закрыть, отложить или пометить как решенный). Инструмент
    выполняет два действия в одной транзакции:
    1. Меняет статус тикета и обновляет поле updated_at в таблице tickets.
    2. Добавляет комментарий с указанием автора и причины изменения в таблицу messages.
    
    Инструмент сначала проверяет, что тикет существует и что переданный
    статус является допустимым. Если проверка не пройдена — изменения
    не применяются.
    
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
        None: Функция ничего не возвращает при успешном выполнении.
    
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
        cursor.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,))
        if cursor.fetchone() is None:
            raise ToolError(f"Тикет {ticket_id} не найден.")

        if status not in seed_data.STATUSES:
            raise ToolError(f"Неправильный статус: {status}.")

        changed_at = seed_data._iso(datetime.now(tz=timezone.utc))
        cursor.execute(
            "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
            (status, changed_at, ticket_id),
        )
        cursor.execute(
            "INSERT INTO messages (ticket_id, author, text, created_at) "
            "VALUES (?, ?, ?, ?)",
            (ticket_id, author, comment, changed_at),
        )
        cursor.connection.commit()

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
    conf.load_env(conf.PROJECT_DIR / ".env")
    mcp.run()