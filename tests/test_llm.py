import json
from types import SimpleNamespace

import pytest

from mcp_assistant.client.agent import Agent
from mcp_assistant.client import llm as llm_module
from mcp_assistant.client.llm import GigaChatLLMClient, OpenAILLMClient, create_llm


class FakeGigaChat:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.__class__.instances.append(self)

    def close(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(self, responses=None) -> None:
        self.responses = list(responses or [])
        self.requests = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeOpenAI:
    instances = []

    def __init__(self, responses=None, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)
        self.__class__.instances.append(self)

    def close(self) -> None:
        self.closed = True


def make_openai_response(content=None, calls=None) -> SimpleNamespace:
    tool_calls = [
        SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        for call_id, name, arguments in calls or []
    ]
    return SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-test",
        created=123,
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
        ))],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


@pytest.mark.anyio
async def test_openai_chat_sends_tools_and_maps_text_response() -> None:
    """Проверяет преобразование инструментов и текстового ответа OpenAI во внутренний формат."""
    sdk = FakeOpenAI(responses=[make_openai_response(content="Готово")])
    client = OpenAILLMClient(sdk, "gpt-test")
    functions = [{
        "name": "tickets__search_tickets",
        "description": "Поиск",
        "parameters": {"type": "object", "properties": {}},
    }]

    response = await client.chat(
        [{"role": "user", "content": "Найди тикеты"}],
        functions,
    )

    request = sdk.completions.requests[0]
    assert request["model"] == "gpt-test"
    assert request["messages"] == [{"role": "user", "content": "Найди тикеты"}]
    assert request["tools"] == [{"type": "function", "function": functions[0]}]
    assert request["parallel_tool_calls"] is True
    assert Agent._response_text(response) == "Готово"
    assert response.usage.total_tokens == 15


@pytest.mark.anyio
async def test_openai_parallel_tool_calls_keep_ids_and_raw_arguments() -> None:
    """Проверяет сохранение ID и JSON-аргументов параллельных вызовов OpenAI."""
    sdk = FakeOpenAI(responses=[make_openai_response(
        content="Проверю оба источника",
        calls=[
            ("call-1", "tickets__get_ticket", '{"ticket_id": 1}'),
            ("call-2", "knowledge__search_articles", '{"query": "502"}'),
        ],
    )])
    client = OpenAILLMClient(sdk, "gpt-test")

    response = await client.chat([{"role": "user", "content": "test"}])
    batches = Agent._get_call_batches(response)

    assert len(batches) == 1
    calls = batches[0][1]
    assert [call.id_ for call in calls] == ["call-1", "call-2"]
    assert calls[0].arguments == '{"ticket_id": 1}'
    assert Agent._response_text(response) == "Проверю оба источника"


def test_openai_history_expands_parallel_results_into_tool_messages() -> None:
    """Проверяет разворачивание внутренней истории в assistant- и tool-сообщения OpenAI."""
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "tools_state_id": "chatcmpl-test",
            "content": [
                {"text": "Сначала поиск"},
                {"function_call": {
                    "id": "call-1",
                    "name": "tickets__search_tickets",
                    "arguments": {"query": "502"},
                }},
                {"function_call": {
                    "id": "call-2",
                    "name": "knowledge__search_articles",
                    "arguments": '{"query": "502"}',
                }},
            ],
        },
        {
            "role": "tool",
            "tools_state_id": "chatcmpl-test",
            "content": [
                {"function_result": {
                    "id": "call-1",
                    "name": "tickets__search_tickets",
                    "result": {"result": "ticket"},
                }},
                {"function_result": {
                    "id": "call-2",
                    "name": "knowledge__search_articles",
                    "result": {"result": "article"},
                }},
            ],
        },
        {"role": "user", "content": "prompt from MCP"},
    ]

    converted = OpenAILLMClient._to_openai_messages(messages)

    assert converted[2]["content"] == "Сначала поиск"
    assert converted[2]["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "tickets__search_tickets",
                "arguments": '{"query": "502"}',
            },
        },
        {
            "id": "call-2",
            "type": "function",
            "function": {
                "name": "knowledge__search_articles",
                "arguments": '{"query": "502"}',
            },
        },
    ]
    assert converted[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": json.dumps({"result": "ticket"}, ensure_ascii=False),
    }
    assert converted[4]["tool_call_id"] == "call-2"
    assert converted[5] == {"role": "user", "content": "prompt from MCP"}


def test_create_llm_defaults_to_gigachat(monkeypatch) -> None:
    """Проверяет выбор GigaChat по умолчанию и передачу настроек SDK."""
    FakeGigaChat.instances = []
    monkeypatch.setattr(llm_module, "GigaChat", FakeGigaChat)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("MODEL", "GigaChat-3-Ultra")
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "secret")
    monkeypatch.setenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1")

    client = create_llm(timeout=42)

    assert isinstance(client, GigaChatLLMClient)
    assert FakeGigaChat.instances[0].kwargs == {
        "base_url": "https://api.giga.chat/v1",
        "model": "GigaChat-3-Ultra",
        "credentials": "secret",
        "scope": "GIGACHAT_API_PERS",
        "verify_ssl_certs": False,
        "timeout": 42,
    }
    client.close()
    assert FakeGigaChat.instances[0].closed


def test_create_llm_selects_openai(monkeypatch) -> None:
    """Проверяет выбор OpenAI и передачу ключа, модели, base_url и таймаута."""
    FakeOpenAI.instances = []
    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("LLM_PROVIDER", " OpenAI ")
    monkeypatch.setenv("MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

    client = create_llm(timeout=42)

    assert isinstance(client, OpenAILLMClient)
    assert FakeOpenAI.instances[0].kwargs == {
        "api_key": "secret",
        "timeout": 42,
        "base_url": "https://example.test/v1",
    }
    client.close()
    assert FakeOpenAI.instances[0].closed


def test_create_llm_rejects_unknown_provider(monkeypatch) -> None:
    """Проверяет ошибку при неизвестном LLM_PROVIDER."""
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    monkeypatch.setenv("MODEL", "test")

    with pytest.raises(ValueError, match="Неизвестный LLM_PROVIDER"):
        create_llm(timeout=10)


def test_openai_requires_api_key(monkeypatch) -> None:
    """Проверяет обязательность OPENAI_API_KEY для OpenAI."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MODEL", "gpt-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_llm(timeout=10)


@pytest.mark.anyio
async def test_agent_completes_openai_tool_round_trip() -> None:
    """Проверяет полный цикл вызова инструмента агентом через OpenAI-адаптер."""
    sdk = FakeOpenAI(responses=[
        make_openai_response(calls=[(
            "call-1",
            "tickets__search_tickets",
            '{"query": "502"}',
        )]),
        make_openai_response(content="Найден один тикет"),
    ])
    client = OpenAILLMClient(sdk, "gpt-test")

    class MCP:
        def get_all_tools(self) -> dict:
            return {"tickets": [SimpleNamespace(
                name="search_tickets",
                description="Поиск",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                annotations=SimpleNamespace(read_only_hint=True),
            )]}

        def get_all_resources(self) -> dict:
            return {}

        def get_all_resource_templates(self) -> dict:
            return {}

        def get_all_prompts(self) -> dict:
            return {}

        async def call_tool(self, server, tool, arguments) -> SimpleNamespace:
            assert (server, tool, arguments) == (
                "tickets", "search_tickets", {"query": "502"}
            )
            return SimpleNamespace(
                is_error=False,
                content=[SimpleNamespace(text='{"tickets": [{"id": 1}]}')],
            )

    class Tracer:
        def log(self, event_type, **data) -> None:
            pass

    agent = Agent(MCP(), client, Tracer(), confirmation=None)

    answer = await agent.run("Найди ошибку 502")

    assert answer == "Найден один тикет"
    second_request = sdk.completions.requests[1]
    assert [message["role"] for message in second_request["messages"]] == [
        "system", "user", "assistant", "tool",
    ]
    tool_message = second_request["messages"][-1]
    assert tool_message["tool_call_id"] == "call-1"
    assert "&quot;id&quot;: 1" in tool_message["content"]



@pytest.mark.anyio
async def test_gigachat_chat_without_functions() -> None:
    """Проверяет GigaChat-запрос без инструментов и включение parallel_tool_calls."""
    captured = {}

    def create(request):
        captured["request"] = request
        return "response"

    sdk = SimpleNamespace(chat=SimpleNamespace(create=create))
    client = GigaChatLLMClient(sdk, "giga-test")

    result = await client.chat([{"role": "user", "content": "Привет"}])

    assert result == "response"
    assert captured["request"].tools is None
    assert captured["request"].model_options.parallel_tool_calls is True


def test_openai_history_preserves_plain_assistant_content() -> None:
    """Проверяет сохранение обычного текста assistant и пустых tool-результатов при преобразовании."""
    converted = OpenAILLMClient._to_openai_messages([
        {"role": "assistant", "content": "Обычный ответ"},
        {"role": "assistant", "content": [{"text": "Только текст"}]},
        {"role": "tool", "content": None},
    ])

    assert converted == [
        {"role": "assistant", "content": "Обычный ответ"},
        {"role": "assistant", "content": "Только текст"},
    ]


def test_openai_response_without_content_and_usage() -> None:
    """Проверяет преобразование ответа OpenAI без текста, вызовов и статистики токенов."""
    response = make_openai_response()
    response.usage = None

    converted = OpenAILLMClient._from_openai_response(response)

    assert converted.usage is None
    assert converted.messages[0].content is None
    assert converted.messages[0].tools_state_id is None


def test_create_llm_requires_model(monkeypatch) -> None:
    """Проверяет обязательность переменной MODEL при создании LLM-клиента."""
    monkeypatch.delenv("MODEL", raising=False)

    with pytest.raises(ValueError, match="MODEL"):
        create_llm(timeout=10)


def test_gigachat_requires_auth_key(monkeypatch) -> None:
    """Проверяет обязательность GIGACHAT_AUTH_KEY для GigaChat."""
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("MODEL", "giga-test")
    monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)

    with pytest.raises(ValueError, match="GIGACHAT_AUTH_KEY"):
        create_llm(timeout=10)


def test_gigachat_requires_base_url(monkeypatch) -> None:
    """Проверяет обязательность GIGACHAT_BASE_URL для GigaChat."""
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("MODEL", "giga-test")
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "secret")
    monkeypatch.delenv("GIGACHAT_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="GIGACHAT_BASE_URL"):
        create_llm(timeout=10)


def test_create_llm_openai_without_base_url(monkeypatch) -> None:
    """Проверяет создание OpenAI-клиента без необязательного OPENAI_BASE_URL."""
    FakeOpenAI.instances = []
    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    client = create_llm(timeout=42)

    assert isinstance(client, OpenAILLMClient)
    assert FakeOpenAI.instances[0].kwargs == {"api_key": "secret", "timeout": 42}
