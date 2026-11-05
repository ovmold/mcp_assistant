# MCP-ассистент инженера поддержки

Ассистент на базе GigaChat и Model Context Protocol (MCP). Он ищет тикеты в SQLite, находит статьи в Markdown-базе знаний, изменяет статусы и добавляет комментарии. Операции изменения данных выполняются только после подтверждения пользователя.

## Структура проекта

- `src/mcp_assistant/client/agent.py` — агентный цикл, обращение к GigaChat, история диалога и вызов инструментов.
- `src/mcp_assistant/client/agent_prompts.py` — системный промпт и правила работы агента.
- `src/mcp_assistant/client/mcp_client.py` — подключение к MCP-серверам.
- `src/mcp_assistant/client/ui/` — консольный и Gradio-интерфейсы.
- `src/mcp_assistant/client/servers.json` — конфигурация MCP-серверов, лимиты и таймауты.
- `src/mcp_assistant/servers/tickets.py` — инструменты поиска и изменения тикетов.
- `src/mcp_assistant/servers/knowledge.py` — семантический поиск и чтение статей базы знаний.
- `src/mcp_assistant/config.py` — общие пути к данным и настройки окружения.
- `src/mcp_assistant/storage/` — модели данных и SQL-схема.
- `src/mcp_assistant/seed.py` — создание тестовой БД и базы знаний.
- `tests/` — unit- и интеграционные тесты.
- `evals/` — сценарии оценки работы агента.

Поток обработки запроса:

```text
CLI / Gradio → Agent → GigaChat → MCP Client → MCP Server → SQLite / Markdown
```

## Установка и настройка

Требуются Python 3.12+, [uv](https://docs.astral.sh/uv/) и ключ GigaChat.

```bash
uv sync
```

Скопируйте `.env.example` в `.env` и заполните `GIGACHAT_AUTH_KEY`. В `.env` также можно изменить модель, адрес GigaChat API и пути к данным.

Для создания тестовых данных:

```bash
uv run python -m mcp_assistant.seed
```

Для удаления старых данных и повторного создания БД и базы знаний:

```bash
uv run python -m mcp_assistant.seed --reset
```

## Запуск

Сначала в отдельном терминале запустите MCP-сервер базы знаний:

```bash
uv run python -m mcp_assistant.servers.knowledge
```

Затем запустите ассистента в консоли:

```bash
uv run python -m mcp_assistant.client.main --ui cli
```

Или с веб-интерфейсом:

```bash
uv run python -m mcp_assistant.client.main --ui gradio
```

Gradio по умолчанию доступен по адресу `http://127.0.0.1:7860`. Адрес и порт можно изменить параметрами `--host` и `--port`.

В CLI доступны команды `/reset` или `/clear` для очистки истории и `/exit` или `/quit` для выхода.

## Тестирование

Запуск всех тестов:

```bash
uv run pytest -q
```

Примеры выборочного запуска:

```bash
uv run pytest tests/test_agent.py -q
uv run pytest tests/test_tickets.py -q
uv run pytest tests/test_knowledge.py -q
uv run pytest tests/test_mcp_layer.py -q
uv run pytest tests/test_gradio_ui.py -q
```

## Оценка агента

При запущенном сервере базы знаний выполните:

```bash
uv run python evals/run_evals.py
```

Можно указать сценарий, YAML-файл и количество повторений:

```bash
uv run python evals/run_evals.py --scenario SCENARIO_ID --file evals/scenarios.yaml --runs 3
```

Результаты показывают успешность сценариев, точность выбора инструментов и среднее количество шагов агента.
