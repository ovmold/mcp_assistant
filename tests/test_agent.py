import json
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from client.agent import Agent, LLMClient
from client.confirmation import CLIConfirmation
from scripts import conf


# ============================================================================
# ФИКСТУРЫ И МОКИ
# ============================================================================

class MockConfirmation:
    """Заглушка ConfirmationProvider: записывает вызовы, возвращает заданный ответ."""
    def __init__(self, approved: bool = True):
        self.approved = approved
        self.calls: list[tuple[str, dict]] = []

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        self.calls.append((tool_name, arguments))
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
            content=[SimpleNamespace(text=json.dumps(self._call_tool_result))],
        )


class MockLLMClient(LLMClient):
    """Заглушка для LLM-клиента, возвращающая заранее заданные ответы."""
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0

    async def chat(self, messages, functions=None) -> SimpleNamespace:
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


@pytest.fixture
def mock_tools() -> dict:
    return {
        "tickets": [
            SimpleNamespace(
                name="search_tickets",
                description="Поиск тикетов",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            SimpleNamespace(
                name="update_status",
                description="Обновление статуса",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticket_id": {"type": "integer"},
                        "status": {"type": "string"},
                    },
                    "required": ["ticket_id", "status"],
                },
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
    monkeypatch.setattr(conf, "SYSTEM_PROMPT", "Ты — тестовый агент.")


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
        write_tools_by_server={},
    )

    result = await agent.run("Найди тикеты про ошибку")
    assert result == "Найдено 2 тикета"
    assert llm.call_count == 2


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
        write_tools_by_server={},
    )

    result = await agent.run("Бесконечный цикл")
    assert "Достигнут лимит шагов" in result
    assert llm.call_count == 3


@pytest.mark.anyio
async def test_invalid_json(mock_mcp_client, mock_tracer, mock_confirmation):
    """Невалидный JSON в аргументах инструмента."""
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
        write_tools_by_server={},
    )

    result = await agent.run("Вызови инструмент")
    assert result == "Исправил JSON"
    assert llm.call_count == 2


@pytest.mark.anyio
async def test_unknown_tool(mock_mcp_client, mock_tracer, mock_confirmation):
    """Модель вызывает несуществующий инструмент."""
    unknown_call = MockLLMClient._make_function_call_response(
        "unknown__tool", {"arg": "value"}
    )
    final = MockLLMClient._make_final_response("Инструмент не найден")
    llm = MockLLMClient([unknown_call, final])
    agent = Agent(
        mcp_client=mock_mcp_client,
        llm=llm,
        tracer=mock_tracer,
        confirmation=mock_confirmation,
        max_steps=5,
        write_tools_by_server={},
    )

    result = await agent.run("Вызови неизвестный инструмент")
    assert result == "Инструмент не найден"
    assert llm.call_count == 2


@pytest.mark.anyio
async def test_tool_execution_error(mock_tracer, mock_confirmation):
    """Инструмент выбрасывает исключение."""
    mcp = MockMCPClient(
        tools={
            "tickets": [
                SimpleNamespace(name="search_tickets", description="", input_schema={})
            ]
        },
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
        write_tools_by_server={},
    )

    result = await agent.run("Вызови инструмент с ошибкой")
    assert result == "Ошибка БД"
    assert llm.call_count == 2


@pytest.mark.anyio
async def test_write_tool_confirmation_denied(mock_tracer):
    """Подтверждение отклонено, инструмент не выполняется."""
    mcp = MockMCPClient(
        tools={
            "tickets": [
                SimpleNamespace(name="update_status", description="", input_schema={})
            ]
        },
        call_tool_result={"result": "статус обновлён"},
    )
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__update_status", {"ticket_id": 1, "status": "closed"}
        ),
        MockLLMClient._make_final_response("Отменено"),
    ])
    confirmation = MockConfirmation(approved=False)
    agent = Agent(
        mcp_client=mcp,
        llm=llm,
        tracer=mock_tracer,
        confirmation=confirmation,
        max_steps=5,
        write_tools_by_server={"tickets": {"update_status"}},
    )

    result = await agent.run("Закрой тикет 1")

    assert result == "Отменено"
    assert llm.call_count == 2
    # Проверяем, что агент реально спросил подтверждение с правильными аргументами
    assert len(confirmation.calls) == 1
    tool_name, arguments = confirmation.calls[0]
    assert tool_name == "update_status"
    assert arguments == {"ticket_id": 1, "status": "closed"}


@pytest.mark.anyio
async def test_write_tool_confirmation_approved(mock_tracer):
    """Подтверждение получено, инструмент выполняется."""
    mcp = MockMCPClient(
        tools={
            "tickets": [
                SimpleNamespace(name="update_status", description="", input_schema={})
            ]
        },
        call_tool_result={"result": "статус обновлён"},
    )
    llm = MockLLMClient([
        MockLLMClient._make_function_call_response(
            "tickets__update_status", {"ticket_id": 1, "status": "closed"}
        ),
        MockLLMClient._make_final_response("Закрыто"),
    ])
    confirmation = MockConfirmation(approved=True)
    agent = Agent(
        mcp_client=mcp,
        llm=llm,
        tracer=mock_tracer,
        confirmation=confirmation,
        max_steps=5,
        write_tools_by_server={"tickets": {"update_status"}},
    )

    result = await agent.run("Закрой тикет 1")

    assert result == "Закрыто"
    assert llm.call_count == 2
    assert len(confirmation.calls) == 1
    assert confirmation.calls[0][0] == "update_status"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer,expected",
    [("да", True), ("нет", False), ("y", True), ("  ДА  ", True), ("nonsense", False)],
)
async def test_cli_confirmation(answer, expected):
    with patch("builtins.input", return_value=answer):
        result = await CLIConfirmation().confirm("update_status", {"ticket_id": 1})
    assert result is expected