import asyncio
import json
from unittest.mock import patch
from types import SimpleNamespace
from mcp.types import ToolAnnotations

import pytest

from mcp_assistant.client.agent import Agent, LLMClient
from mcp_assistant.client.confirmation import CLIConfirmation
from mcp_assistant.client import agent_prompts


# ============================================================================
# ФИКСТУРЫ И МОКИ
# ============================================================================

class MockConfirmation:
    """Заглушка ConfirmationProvider: записывает вызовы, возвращает заданный ответ."""
    def __init__(self, approved: bool = True):
        self.approved = approved
        self.calls: list[tuple[str, dict]] = []
        self.calls_with_reason: list[tuple[str, dict, str]] = []

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        self.calls.append((tool_name, arguments))
        return self.approved

    async def confirm_with_reason(
        self, tool_name: str, arguments: dict, reason: str = "",
    ) -> bool:
        self.calls.append((tool_name, arguments))
        self.calls_with_reason.append((tool_name, arguments, reason))
        return self.approved


@pytest.fixture
def mock_confirmation() -> MockConfirmation:
    return MockConfirmation(approved=True)

@pytest.fixture
def anyio_backend():
    """Указывает anyio использовать asyncio (а не trio)."""
    return "asyncio"


class MockTracer:
    """Заглушка для трассировщика."""
    def __init__(self):
        self.logs = []

    def log(self, event_type, **data) -> None:
        self.logs.append((event_type, data))


class MockMCPClient:
    """Заглушка для MCP-клиента."""
    def __init__(self, tools: dict, call_tool_result=None, call_tool_error=None):
        self._tools = tools
        self._call_tool_result = call_tool_result or {"result": "ok"}
        self._call_tool_error = call_tool_error

    def get_all_tools(self) -> dict:
        return self._tools

    async def call_tool(self, server_name, tool_name, arguments) -> SimpleNamespace:
        if self._call_tool_error:
            raise self._call_tool_error
        return SimpleNamespace(
            is_error=False,
            content=[SimpleNamespace(text=json.dumps(self._call_tool_result, ensure_ascii=False))],
        )


class MockLLMClient(LLMClient):
    """Заглушка для LLM-клиента, возвращающая заранее заданные ответы."""
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        self.history: list[list[dict]] = []  # история messages на каждом вызове

    async def chat(self, messages, functions=None):
        self.history.append([dict(m) for m in messages])  # копия messages
        if self.call_count >= len(self.responses):
            return self._make_final_response("Конец теста")
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

    @staticmethod
    def _make_final_response(content: str) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, function_call=None)
                )
            ]
        )

    @staticmethod
    def _make_function_call_response(name: str, arguments: dict) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        function_call=SimpleNamespace(
                            name=name,
                            arguments=json.dumps(arguments),
                        ),
                    )
                )
            ]
        )


def _make_tool(name: str, description: str, input_schema: dict, **ann) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema=input_schema,
        annotations=ToolAnnotations(**ann) if ann else None,
    )


@pytest.fixture
def mock_tools() -> dict:
    return {
        "tickets": [
            _make_tool(
                "search_tickets", "Поиск тикетов",
                {"type": "object", "properties": {"query": {"type": "string"}},
                 "required": ["query"]},
                readOnlyHint=True,
            ),
            _make_tool(
                "update_status", "Обновление статуса",
                {"type": "object", "properties": {
                    "ticket_id": {"type": "integer"},
                    "status": {"type": "string"},
                }, "required": ["ticket_id", "status"]},
                readOnlyHint=False, destructiveHint=True,
            ),
            _make_tool(
                "add_comment", "Добавить комментарий",
                {"type": "object", "properties": {
                    "ticket_id": {"type": "integer"},
                    "text": {"type": "string"},
                }, "required": ["ticket_id", "text"]},
                readOnlyHint=False, destructiveHint=True,
            ),
            _make_tool(
                "some_unannotated_tool", "Без аннотаций",
                {"type": "object", "properties": {}},
                # annotations отсутствуют → fail-safe
            ),
        ]
    }


@pytest.fixture
def mock_mcp_client(mock_tools) -> MockMCPClient:
    return MockMCPClient(mock_tools)


@pytest.fixture
def mock_tracer() -> MockTracer:
    return MockTracer()


@pytest.fixture(autouse=True)
def mock_system_prompt(monkeypatch) -> None:
    """Подменяет генератор системного промпта на статическую строку."""
    monkeypatch.setattr(
        agent_prompts,
        "build_system_prompt",
        lambda tools: "Ты — тестовый агент."
    )


# ============================================================================
# ТЕСТЫ
# ============================================================================

@pytest.mark.anyio
async def test_normal_flow(mock_mcp_client, mock_tracer, mock_confirmation):
    """Один вызов инструмента, затем финальный ответ."""
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__search_tickets", {"query": "ошибка"}
        ),
        MockLLMClient._make_final_response("Найдено 2 тикета"),
    ])
    agent = Agent(
        mcp_client=mock_mcp_client,
        llm=llm,
        tracer=mock_tracer,
        confirmation=mock_confirmation, 
        max_steps=5,
        always_confirm_tools={},
    )

    result = await agent.run("Найди тикеты про ошибку")
    assert result == "Найдено 2 тикета"
    assert llm.call_count == 2
    assert mock_confirmation.calls == []


@pytest.mark.anyio
async def test_step_limit(mock_mcp_client, mock_tracer, mock_confirmation):
    """Модель бесконечно запрашивает инструменты, срабатывает лимит шагов."""
    response = MockLLMClient._make_function_call_response(
        "tickets__search_tickets", {"query": "test"}
    )
    llm = MockLLMClient([response] * 10)
    agent = Agent(
        mcp_client=mock_mcp_client,
        llm=llm,
        tracer=mock_tracer,
        confirmation=mock_confirmation,
        max_steps=3,
        always_confirm_tools={},
    )

    result = await agent.run("Бесконечный цикл")
    assert "Достигнут лимит шагов" in result
    assert llm.call_count == 3
    assert mock_confirmation.calls == []


@pytest.mark.anyio
async def test_invalid_json_retries_once(mock_mcp_client, mock_tracer, mock_confirmation):
    """Невалидный JSON: агент ровно один раз сообщает модели об ошибке."""
    bad_call = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    function_call=SimpleNamespace(
                        name="tickets__search_tickets",
                        arguments='{"query": "test"',  # незакрытая скобка
                    ),
                )
            )
        ]
    )
    final = MockLLMClient._make_final_response("Исправил JSON")
    llm = MockLLMClient([bad_call, final])

    agent = Agent(
        mcp_client=mock_mcp_client,
        llm=llm,
        tracer=mock_tracer,
        confirmation=mock_confirmation,
        max_steps=5,
        always_confirm_tools={},
    )

    result = await agent.run("Вызови инструмент")

    # 1. Агент вернул финальный ответ
    assert result == "Исправил JSON"

    # 2. Ровно два вызова LLM (первый — плохой JSON, второй — финал)
    assert llm.call_count == 2

    # 3. Событие invalid_json залогировано ровно один раз
    invalid_events = [e for e in mock_tracer.logs if e[0] == "invalid_json"]
    assert len(invalid_events) == 1, (
        f"Ожидался ровно один повтор, но invalid_json логировался "
        f"{len(invalid_events)} раз(а)"
    )

    # 4. Инструмент search_tickets не вызывался (JSON был битым)
    tool_calls = [d["tool"] for ev, d in mock_tracer.logs if ev == "tool_call"]
    assert "search_tickets" not in tool_calls
    assert mock_confirmation.calls == []

@pytest.mark.anyio
async def test_tool_execution_error(mock_tracer, mock_confirmation):
    """Инструмент выбрасывает исключение."""
    mcp = MockMCPClient(
        tools={"tickets": [_make_tool(
            "search_tickets", "",
            {"type": "object", "properties": {}},
            readOnlyHint=True,
        )]},
        call_tool_error=ValueError("DB error"),
    )
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response("tickets__search_tickets", {}),
        MockLLMClient._make_final_response("Ошибка БД"),
    ])
    agent = Agent(
        mcp_client=mcp,
        llm=llm,
        tracer=mock_tracer,
        confirmation=mock_confirmation,
        max_steps=5,
        always_confirm_tools={},
    )

    result = await agent.run("Вызови инструмент с ошибкой")
    assert result == "Ошибка БД"
    assert llm.call_count == 2
    assert mock_confirmation.calls == []


@pytest.mark.anyio
async def test_tool_timeout_error_has_no_hardcoded_duration(mock_tracer, mock_confirmation):
    mcp = MockMCPClient(
        tools={"tickets": [_make_tool(
            "search_tickets", "",
            {"type": "object", "properties": {}},
            readOnlyHint=True,
        )]},
        call_tool_error=asyncio.TimeoutError(),
    )
    agent = Agent(
        mcp_client=mcp,
        llm=MockLLMClient([]),
        tracer=mock_tracer,
        confirmation=mock_confirmation,
    )

    result = await agent._execute_function_call(
        SimpleNamespace(name="tickets__search_tickets", arguments="{}"),
        mock_confirmation,
    )

    error = json.loads(result)["error"]
    assert "превышен таймаут ожидания инструмента" in error
    assert "60 сек" not in error


@pytest.mark.anyio
async def test_readonly_tool_no_confirmation(mock_mcp_client, mock_tracer):
    """search_tickets (readOnlyHint=True) не должен требовать подтверждения."""
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__search_tickets", {"query": "502"}
        ),
        MockLLMClient._make_final_response("Найдено"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mock_mcp_client,
        llm=llm,
        tracer=mock_tracer,
        confirmation=confirmation,
        max_steps=5,
        always_confirm_tools={},     # конфиг пуст — решает аннотация
    )

    await agent.run("найди 502")

    assert confirmation.calls == [], (
        "read-only инструмент не должен спрашивать подтверждение"
    )

@pytest.mark.anyio
async def test_destructive_tool_confirmation_from_annotations(mock_mcp_client, mock_tracer):
    """update_status (destructiveHint=True) требует подтверждения, даже если
    его нет в always_confirm_tools."""
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__update_status", {"ticket_id": 1, "status": "closed"}
        ),
        MockLLMClient._make_final_response("Ок"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mock_mcp_client,
        llm=llm,
        tracer=mock_tracer,
        confirmation=confirmation,
        max_steps=5,
        always_confirm_tools={},     # пусто — аннотация решает
    )

    await agent.run("закрой тикет")

    assert len(confirmation.calls) == 1
    assert confirmation.calls[0][0] == "update_status"
    # reason должен явно указывать на деструктивность
    reason = confirmation.calls_with_reason[0][2]
    assert "destructive" in reason.lower()

@pytest.mark.anyio
async def test_unannotated_tool_failsafe_confirmation(mock_mcp_client, mock_tracer):
    """Инструмент без аннотаций должен требовать подтверждения (fail-safe)."""
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__some_unannotated_tool", {}
        ),
        MockLLMClient._make_final_response("Ок"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mock_mcp_client,
        llm=llm,
        tracer=mock_tracer,
        confirmation=confirmation,
        max_steps=5,
        always_confirm_tools={},
    )

    await agent.run("вызови")

    assert len(confirmation.calls) == 1
    reason = confirmation.calls_with_reason[0][2]
    assert "no annotations" in reason or "fail-safe" in reason.lower()

@pytest.mark.anyio
async def test_config_always_confirm_overrides_readonly(mock_mcp_client, mock_tracer):
    """Если инструмент в always_confirm_tools — подтверждение нужно,
    даже если readOnlyHint=True (не доверяем серверу)."""
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__search_tickets", {"query": "x"}
        ),
        MockLLMClient._make_final_response("Ок"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mock_mcp_client,
        llm=llm,
        tracer=mock_tracer,
        confirmation=confirmation,
        max_steps=5,
        always_confirm_tools={"tickets": {"search_tickets"}},
    )

    await agent.run("ищи")

    assert len(confirmation.calls) == 1
    reason = confirmation.calls_with_reason[0][2]
    assert "always_confirm" in reason

@pytest.mark.anyio
async def test_add_comment_confirmation_from_destructive_hint(mock_mcp_client, mock_tracer):
    """add_comment помечен destructiveHint=True (комментарий не откатить)."""
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__add_comment", {"ticket_id": 1, "text": "ответ"}
        ),
        MockLLMClient._make_final_response("Ок"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=confirmation, max_steps=5, always_confirm_tools={},
    )

    await agent.run("добавь комментарий")

    assert len(confirmation.calls) == 1
    reason = confirmation.calls_with_reason[0][2]
    assert "destructive" in reason.lower()

@pytest.mark.anyio
async def test_confirmation_requested_trace_contains_reason(mock_mcp_client, mock_tracer):
    """При запросе подтверждения в трассу попадает reason."""
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__update_status", {"ticket_id": 1, "status": "closed"}
        ),
        MockLLMClient._make_final_response("Ок"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=confirmation, max_steps=5, always_confirm_tools={},
    )

    await agent.run("закрой")

    events = [e for e in mock_tracer.logs if e[0] == "confirmation_requested"]
    assert len(events) == 1, "должно быть ровно одно событие confirmation_requested"
    _, data = events[0]
    assert data["tool"] == "update_status"
    assert data["server"] == "tickets"
    assert "destructive" in data["reason"].lower()

@pytest.mark.anyio
async def test_denied_confirmation_does_not_call_tool(mock_tracer):
    """Если пользователь отказал — call_tool не должен вызываться."""
    mcp = MockMCPClient(
        tools={
            "tickets": [
                _make_tool(
                    "update_status", "",
                    {"type": "object", "properties": {}},
                    title="Обновить статус",
                    readOnlyHint=False, destructiveHint=True,
                ),
            ]
        },
    )
    called = []
    original_call = mcp.call_tool
    async def spy(server, tool, args):
        called.append((server, tool, args))
        return await original_call(server, tool, args)
    mcp.call_tool = spy

    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__update_status", {"ticket_id": 1, "status": "closed"}
        ),
        MockLLMClient._make_final_response("Отменено"),
    ])
    confirmation = MockConfirmation(approved=False)
    agent = Agent(
        mcp_client=mcp, llm=llm, tracer=mock_tracer,
        confirmation=confirmation, max_steps=5, always_confirm_tools={},
    )

    result = await agent.run("закрой")

    assert result == "Отменено"
    assert called == [], "инструмент не должен вызываться при отказе"

@pytest.mark.anyio
async def test_unknown_tool_does_not_ask_confirmation(mock_mcp_client, mock_tracer):
    """Неизвестный инструмент не должен доходить до подтверждения."""
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response("unknown__tool", {}),
        MockLLMClient._make_final_response("Не найдено"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=confirmation, max_steps=5, always_confirm_tools={},
    )

    await agent.run("вызови неизвестный")

    assert confirmation.calls == []
    assert confirmation.calls_with_reason == []

@pytest.mark.anyio
async def test_write_tool_confirmation_denied(mock_tracer):
    """Подтверждение отклонено, инструмент НЕ выполняется."""
    mcp = MockMCPClient(
        tools={"tickets": [_make_tool(
            "update_status", "",
            {"type": "object", "properties": {}},
            readOnlyHint=False, destructiveHint=True,
        )]},
        call_tool_result={"result": "статус обновлён"},
    )

    # Spy: фиксируем фактические вызовы call_tool
    called: list[tuple[str, str, dict]] = []
    original_call = mcp.call_tool
    async def spy(server, tool, args):
        called.append((server, tool, args))
        return await original_call(server, tool, args)
    mcp.call_tool = spy

    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__update_status", {"ticket_id": 1, "status": "closed"}
        ),
        MockLLMClient._make_final_response("Отменено"),
    ])
    confirmation = MockConfirmation(approved=False)
    agent = Agent(
        mcp_client=mcp, llm=llm, tracer=mock_tracer,
        confirmation=confirmation, max_steps=5,
        always_confirm_tools={"tickets": {"update_status"}},
    )

    result = await agent.run("Закрой тикет 1")

    assert result == "Отменено"
    assert len(confirmation.calls) == 1
    assert confirmation.calls[0] == ("update_status", {"ticket_id": 1, "status": "closed"})
    assert "destructiveHint=True" in confirmation.calls_with_reason[0][2]

    # Ключевой assert: инструмент не выполнился
    assert called == [], "call_tool не должен вызываться при отказе"

    # И в трассе должно быть confirmation_denied, а не tool_call
    events = [e[0] for e in mock_tracer.logs]
    assert "confirmation_denied" in events
    assert "tool_call" not in events

@pytest.mark.anyio
async def test_write_tool_confirmation_approved(mock_tracer):
    """Подтверждение получено, инструмент выполняется, результат доходит до LLM."""
    mcp = MockMCPClient(
        tools={"tickets": [_make_tool(
        "update_status", "",
        {"type": "object", "properties": {}},
        readOnlyHint=False, destructiveHint=True,
    )]},
        call_tool_result={"result": "статус обновлён"},
    )

    called: list[tuple[str, str, dict]] = []
    original_call = mcp.call_tool
    async def spy(server, tool, args):
        called.append((server, tool, args))
        return await original_call(server, tool, args)
    mcp.call_tool = spy

    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__update_status", {"ticket_id": 1, "status": "closed"}
        ),
        MockLLMClient._make_final_response("Закрыто"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mcp, llm=llm, tracer=mock_tracer,
        confirmation=confirmation, max_steps=5,
        always_confirm_tools={"tickets": {"update_status"}},
    )

    result = await agent.run("Закрой тикет 1")

    assert result == "Закрыто"
    assert len(confirmation.calls) == 1
    assert "destructiveHint=True" in confirmation.calls_with_reason[0][2]

    # Инструмент вызван ровно один раз с правильными аргументами
    assert called == [
        ("tickets", "update_status", {"ticket_id": 1, "status": "closed"})
    ]

    # Результат инструмента попал в историю как role="function"
    second_call_messages = llm.history[1]   # второй вызов LLM
    function_msgs = [m for m in second_call_messages if m["role"] == "function"]
    assert len(function_msgs) == 1
    assert "статус обновлён" in function_msgs[0]["content"]
    assert function_msgs[0]["name"] == "tickets__update_status"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer,expected",
    [("да", True), ("нет", False), ("y", True), ("  ДА  ", True), ("nonsense", False)],
)
async def test_cli_confirmation_with_reason(answer, expected):
    """confirm_with_reason возвращает ожидаемое значение."""
    with patch("builtins.input", return_value=answer):
        result = await CLIConfirmation().confirm_with_reason(
            "update_status", {"ticket_id": 1}, reason="destructiveHint=True",
        )
    assert result is expected


@pytest.mark.anyio
async def test_cli_confirmation_shows_reason_in_prompt():
    """Причина запроса отображается пользователю в тексте prompt."""
    captured: list[str] = []

    def fake_input(prompt: str) -> str:
        captured.append(prompt)
        return "да"

    with patch("builtins.input", side_effect=fake_input):
        await CLIConfirmation().confirm_with_reason(
            "update_status", {"ticket_id": 1}, reason="destructiveHint=True",
        )

    assert captured, "input() должен быть вызван"
    assert "destructiveHint=True" in captured[0], (
        "reason должен быть виден пользователю"
    )


@pytest.mark.anyio
async def test_agent_remembers_previous_turn(mock_mcp_client, mock_tracer, mock_confirmation):
    """Второй run() видит историю первого."""
    llm = MockLLMClient([
        MockLLMClient._make_final_response("Какой тикет?"),
        MockLLMClient._make_final_response("Закрыл 1042"),
    ])
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
    )

    assert await agent.run("Закрой тикет") == "Какой тикет?"
    assert await agent.run("Тикет 1042") == "Закрыл 1042"

    # Второй вызов LLM получил всю историю
    roles = [m["role"] for m in llm.history[1]]
    assert roles == ["system", "user", "assistant", "user"]
    contents = [m["content"] for m in llm.history[1]]
    assert "Закрой тикет" in contents
    assert "Какой тикет?" in contents
    assert "Тикет 1042" in contents


@pytest.mark.anyio
async def test_sessions_isolated(mock_mcp_client, mock_tracer, mock_confirmation):
    """Разные session_id не видят историю друг друга."""
    llm = MockLLMClient([
        MockLLMClient._make_final_response("a"),
        MockLLMClient._make_final_response("b"),
    ])
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
    )

    await agent.run("вопрос alice", session_id="alice")
    await agent.run("вопрос bob", session_id="bob")

    # bob видит только свой вопрос
    assert [m["role"] for m in llm.history[1]] == ["system", "user"]
    assert llm.history[1][1]["content"] == "вопрос bob"


@pytest.mark.anyio
async def test_reset_clears_history(mock_mcp_client, mock_tracer, mock_confirmation):
    """reset() удаляет историю — следующий run() стартует с чистого листа."""
    llm = MockLLMClient([
        MockLLMClient._make_final_response("a"),
        MockLLMClient._make_final_response("b"),
    ])
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
    )

    await agent.run("привет", session_id="s1")
    agent.reset("s1")
    await agent.run("пока", session_id="s1")

    assert [m["role"] for m in llm.history[1]] == ["system", "user"]
    assert llm.history[1][1]["content"] == "пока"


@pytest.mark.anyio
async def test_reset_only_affects_given_session(mock_mcp_client, mock_tracer, mock_confirmation):
    """reset("a") не должен затрагивать сессию "b"."""
    llm = MockLLMClient([
        MockLLMClient._make_final_response("ok"),
        MockLLMClient._make_final_response("ok"),
        MockLLMClient._make_final_response("ok"),
    ])
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
    )

    await agent.run("a1", session_id="a")
    await agent.run("b1", session_id="b")
    agent.reset("a")
    await agent.run("b2", session_id="b")

    # В истории "b" остался b1
    assert [m["content"] for m in llm.history[2] if m["role"] == "user"] == ["b1", "b2"]


@pytest.mark.anyio
async def test_get_history_returns_defensive_copy(mock_mcp_client, mock_tracer, mock_confirmation):
    """Мутация результата get_history() не меняет внутреннее состояние."""
    llm = MockLLMClient([MockLLMClient._make_final_response("ok")])
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
    )
    await agent.run("hi", session_id="s")

    hist = agent.get_history("s")
    hist[0]["content"] = "hacked"

    assert agent.get_history("s")[0]["content"] != "hacked"


@pytest.mark.anyio
async def test_run_appends_final_answer_to_history(mock_mcp_client, mock_tracer, mock_confirmation):
    llm = MockLLMClient([MockLLMClient._make_final_response("ответ")])
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
    )
    await agent.run("вопрос")

    hist = agent.get_history()
    assert hist[-1] == {"role": "assistant", "content": "ответ"}


@pytest.mark.anyio
async def test_run_appends_error_to_history(mock_mcp_client, mock_tracer, mock_confirmation):
    class FailingLLM(LLMClient):
        async def chat(self, messages, functions=None):
            raise RuntimeError("network down")

    agent = Agent(
        mcp_client=mock_mcp_client, llm=FailingLLM(None, "m"), tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
    )
    result = await agent.run("привет")

    assert "network down" in result
    hist = agent.get_history()
    assert hist[-1]["role"] == "assistant"
    assert "network down" in hist[-1]["content"]


@pytest.mark.anyio
async def test_run_appends_max_steps_message_to_history(mock_mcp_client, mock_tracer, mock_confirmation):
    response = MockLLMClient._make_function_call_response(
        "tickets__search_tickets", {"query": "x"}
    )
    llm = MockLLMClient([response] * 10)
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=2, always_confirm_tools={},
    )
    result = await agent.run("бесконечно")

    assert "Достигнут лимит шагов" in result
    assert agent.get_history()[-1] == {"role": "assistant", "content": result}


@pytest.mark.anyio
async def test_history_trimmed_removes_oldest_turns(mock_mcp_client, mock_tracer, mock_confirmation):
    """Старые ходы вытесняются, свежие остаются, system — всегда первый."""
    llm = MockLLMClient([])   # дефолтный "Конец теста" на каждый вызов
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
        max_history_messages=4,
    )

    for i in range(6):
        await agent.run(f"msg-{i}", session_id="s")

    hist = agent.get_history("s")
    assert hist[0]["role"] == "system"   # system не выброшен

    users = [m["content"] for m in hist if m["role"] == "user"]
    assert "msg-0" not in users          # самый старый выброшен
    assert "msg-5" in users              # самый свежий на месте


@pytest.mark.anyio
async def test_history_not_trimmed_below_limit(mock_mcp_client, mock_tracer, mock_confirmation):
    llm = MockLLMClient([])
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
        max_history_messages=100,
    )
    await agent.run("a")
    await agent.run("b")

    users = [m["content"] for m in agent.get_history() if m["role"] == "user"]
    assert users == ["a", "b"]


@pytest.mark.anyio
async def test_default_session_id(mock_mcp_client, mock_tracer, mock_confirmation):
    llm = MockLLMClient([
        MockLLMClient._make_final_response("a"),
        MockLLMClient._make_final_response("b"),
    ])
    agent = Agent(
        mcp_client=mock_mcp_client, llm=llm, tracer=mock_tracer,
        confirmation=mock_confirmation, max_steps=5, always_confirm_tools={},
    )
    await agent.run("hi")   # без session_id
    await agent.run("again")  # без session_id

    # Оба попали в "default"
    hist = agent.get_history("default")
    users = [m["content"] for m in hist if m["role"] == "user"]
    assert users == ["hi", "again"]