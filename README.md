# MCP-ассистент инженера поддержки

Ассистент на базе GigaChat и Model Context Protocol (MCP). Он ищет тикеты в SQLite, работает с Markdown-базой знаний, добавляет комментарии и изменяет статусы. Независимые read-only инструменты могут выполняться параллельно через GigaChat SDK 0.2.4. Операции записи выполняются последовательно и требуют подтверждения пользователя.

## Требования

- Python 3.12+;
- пакетный менеджер [uv](https://docs.astral.sh/uv/);
- ключ GigaChat из [личного кабинета](https://developers.sber.ru/portal/products/gigachat-api);
- около 3 ГБ свободного места для зависимостей, включая PyTorch.

## Установка и настройка

```bash
git clone https://github.com/ovmold/mcp_assistant.git
cd mcp_assistant
uv sync
```

Скопируйте `.env.example` в `.env` и заполните как минимум:

```dotenv
GIGACHAT_AUTH_KEY=...
GIGACHAT_BASE_URL=https://api.giga.chat/v1
MODEL=GigaChat-3-Ultra
```

В `.env` также можно изменить `DATA_DIR`, `TRACE_DIR`, `KB_DIR` и `DB_PATH`.

## Тестовые данные

```bash
uv run python -m mcp_assistant.seed
```

Каждый запуск полностью пересоздаёт БД и базу знаний. Дополнительные Markdown-файлы в `KB_DIR` удаляются. Не запускайте seed с данными, которые нужно сохранить.

Создаются:

- `data/tickets.db` — тикеты, сообщения и логи;
- `data/kb/*.md` — статьи базы знаний;
- `traces/` — каталог трассировки.

Основные таблицы:

```text
tickets(id, subject, body, status, priority, client,
        created_at, updated_at, tags, issue_type)
messages(id, ticket_id, author, text, created_at)
logs(id, ticket_id, level, message, created_at)
```

## Структура проекта

- `src/mcp_assistant/client/agent.py` — агентный цикл, история и вызовы инструментов.
- `src/mcp_assistant/client/agent_prompts.py` — системный промпт агента.
- `src/mcp_assistant/client/mcp_client.py` — подключения к MCP-серверам, вызовы инструментов, чтение ресурсов и получение промптов.
- `src/mcp_assistant/client/ui/` — CLI и Gradio.
- `src/mcp_assistant/client/servers.json` — серверы, лимиты и таймауты.
- `src/mcp_assistant/servers/tickets.py` — поиск и изменение тикетов.
- `src/mcp_assistant/servers/knowledge.py` — гибридный поиск BM25 + эмбеддинги и чтение статей.
- `src/mcp_assistant/storage/` — модели, SQL-схема и timestamps.
- `src/mcp_assistant/seed.py`, `seed_data.py` — тестовые данные.
- `tests/` — unit- и интеграционные тесты.
- `evals/` — сценарии оценки агента.

Поток обработки запроса:

```text
CLI / Gradio → Agent → GigaChat → MCP Client → MCP Server → SQLite / Markdown
```

Сервер тикетов использует stdio, сервер базы знаний — Streamable HTTP.

## Запуск

Сначала в отдельном терминале запустите сервер базы знаний:

```bash
uv run python -m mcp_assistant.servers.knowledge
```

При первом запуске сервер скачивает модель эмбеддингов и кодирует статьи до приёма MCP-запросов. Последующие запуски используют локальный кэш модели.

Для запуска агента должны подключиться все настроенные MCP-серверы. При недоступности любого из них запуск прекращается с конкретной причиной ошибки.

Затем запустите ассистента в CLI:

```bash
uv run python -m mcp_assistant.client.main --ui cli
```

Или с веб-интерфейсом:

```bash
uv run python -m mcp_assistant.client.main --ui gradio
```

Gradio по умолчанию доступен по адресу `http://127.0.0.1:7860`; используйте `--host` и `--port` для изменения.

В CLI доступны `/reset` и `/clear` для очистки истории, `/exit` и `/quit` для выхода.

Примеры запросов:

```text
Найди тикет 1 и выведи его содержимое.
Найди открытые тикеты high от Globex, созданные на этой неделе.
Найди тикет 1 и статью про ошибку 502 в одном пакете параллельных вызовов.
```

## Тестирование

Все тесты:

```bash
uv run pytest -q
```

Выборочно:

```bash
uv run pytest tests/test_agent.py -q
uv run pytest tests/test_tickets.py -q
uv run pytest tests/test_knowledge.py -q
uv run pytest tests/test_mcp_layer.py -q
uv run pytest tests/test_gradio_ui.py -q
```

Проверка сборки:

```bash
uv build
```

## Оценка агента

Evals выполняют реальные запросы к GigaChat и используют отдельную временную БД. Перед запуском необходимы настройки из `.env` и запущенный сервер базы знаний.

```bash
uv run python evals/run_evals.py
```

Параметры:

```bash
uv run python evals/run_evals.py \
  --scenario SCENARIO_ID \
  --file evals/scenarios.yaml \
  --runs 3
```

- `--scenario` — ID одного сценария;
- `--file` — YAML-файл со сценариями;
- `--runs` — количество повторений, по умолчанию 3.

Отчёт включает Success Rate, точность выбора инструментов, среднее число шагов, время и расход токенов. Примеры вывода приведены в `EVALS.md`.

## Технологический стек

- `gigachat` — GigaChat SDK;
- `mcp[cli]` — Model Context Protocol;
- `gradio`, `fastapi`, `uvicorn` — веб-интерфейс;
- `sentence-transformers`, `torch`, `numpy` — семантический поиск;
- `anyio` — асинхронная работа;
- `python-dotenv`, `pyyaml` — окружение и сценарии;
- `pytest`, `pytest-asyncio` — тесты.

## Известные ограничения

- Один долгоживущий `MCPClient` обслуживает приложение; для параллельной многопользовательской работы со stateful-инструментами нужны дополнительные проверки.
- При интенсивных параллельных записях SQLite может отвечать `database is locked`; для production рассмотрите PostgreSQL.
- Лимиты сообщений, символов истории и результатов инструментов уменьшают риск превышения контекста, но не заменяют учёт токенов модели.
- Стриминг отсутствует: ответ появляется целиком после завершения работы агента.
- Все события записываются в общий `traces/session.jsonl`; события разных Gradio-сессий могут перемешиваться.
- `query` в `search_tickets` ищет только по `subject` и `body`; каждый фильтр принимает одно значение.
- Evals используют статьи из текущей рабочей базы знаний.
