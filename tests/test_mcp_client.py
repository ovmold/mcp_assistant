import asyncio
from types import SimpleNamespace

import pytest

from mcp_assistant.client import mcp_client as mcp_client_module
from mcp_assistant.client.mcp_client import MCPClient


@pytest.mark.anyio
async def test_stdio_server_does_not_inherit_parent_secrets(monkeypatch) -> None:
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "test-secret")
    captured = {}

    def fake_stdio_client(params) -> object:
        captured["params"] = params
        return object()

    class FakeClient:
        def __init__(self, transport) -> None:
            self.transport = transport

        async def __aenter__(self) -> 'FakeClient':
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def list_tools(self) -> SimpleNamespace:
            return SimpleNamespace(tools=[])

        async def list_resources(self) -> SimpleNamespace:
            return SimpleNamespace(resources=[])

        async def list_resource_templates(self) -> SimpleNamespace:
            return SimpleNamespace(resource_templates=[])

        async def list_prompts(self) -> SimpleNamespace:
            return SimpleNamespace(prompts=[])

    monkeypatch.setattr(mcp_client_module, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp_client_module, "Client", FakeClient)
    client = MCPClient({"servers": []})

    try:
        await client._connect("tickets", {
            "transport": {
                "type": "stdio",
                "command": "python",
                "args": [],
                "env": {"EXPLICIT_SETTING": "preserved"},
            }
        })
    finally:
        await client.close_all()

    env = captured["params"].env
    assert "GIGACHAT_AUTH_KEY" not in env
    assert env["EXPLICIT_SETTING"] == "preserved"


@pytest.mark.anyio
async def test_connect_all_reports_root_causes_and_rejects_partial_startup(monkeypatch) -> None:
    client = MCPClient({"servers": [
        {"name": "tickets"},
        {"name": "knowledge"},
    ]})

    async def fake_connect(name, config) -> None:
        if name == "knowledge":
            raise ExceptionGroup("network", [
                ExceptionGroup("connect", [ConnectionRefusedError("connection refused")])
            ])
        client._clients[name] = SimpleNamespace()
        client._tools[name] = []

    monkeypatch.setattr(client, "_connect", fake_connect)

    with pytest.raises(RuntimeError) as exc_info:
        await client.connect_all()

    message = str(exc_info.value)
    assert "knowledge: ConnectionRefusedError: connection refused" in message
    assert "unhandled errors in a TaskGroup" not in message
    assert client._clients == {}
    assert client._tools == {}


@pytest.mark.anyio
async def test_read_resource_and_get_prompt(tickets_client) -> None:
    client = MCPClient({"servers": []})
    client._clients["tickets"] = tickets_client
    client._servers["tickets"] = {"tool_timeout": 5.0}

    resource = await client.read_resource("tickets", "ticket://1")
    prompt = await client.get_prompt(
        "tickets", "analyze_ticket", {"ticket_id": "1"}
    )

    assert "Тикет #1" in resource.contents[0].text
    assert len(prompt.messages) == 2
    assert "тикет #1" in prompt.messages[0].content.text



@pytest.mark.anyio
async def test_connect_rejects_unknown_transport() -> None:
    client = MCPClient({"servers": []})

    with pytest.raises(ValueError, match="Неизвестный транспорт"):
        await client._connect("unknown", {"transport": {"type": "unknown"}})


@pytest.mark.anyio
async def test_connect_all_rejects_empty_configuration() -> None:
    client = MCPClient({"servers": []})

    with pytest.raises(RuntimeError, match="ни к одному серверу"):
        await client.connect_all()


@pytest.mark.anyio
async def test_close_all_clears_state_after_close_error(capsys) -> None:
    class BrokenStack:
        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    client = MCPClient({"servers": []})
    client._exit_stack = BrokenStack()
    client._clients["tickets"] = SimpleNamespace()
    client._tools["tickets"] = []
    client._resources["tickets"] = []
    client._resource_templates["tickets"] = []
    client._prompts["tickets"] = []

    await client.close_all()

    assert "close failed" in capsys.readouterr().out
    assert client._clients == {}
    assert client._tools == {}
    assert client._resources == {}
    assert client._resource_templates == {}
    assert client._prompts == {}


def test_capability_getters_return_independent_dictionaries() -> None:
    client = MCPClient({"servers": []})
    client._tools["tickets"] = []
    client._resources["tickets"] = []
    client._resource_templates["tickets"] = []
    client._prompts["tickets"] = []

    getters = (
        client.get_all_tools,
        client.get_all_resources,
        client.get_all_resource_templates,
        client.get_all_prompts,
    )
    for getter in getters:
        result = getter()
        assert result == {"tickets": []}
        result.clear()
        assert getter() == {"tickets": []}


def test_get_client_and_timeout_validation() -> None:
    client = MCPClient({"servers": [{"name": "tickets", "tool_timeout": 5.0}]})

    with pytest.raises(ValueError, match="не подключён"):
        client._get_client("tickets")

    assert client._get_timeout("tickets", None) == 5.0
    assert client._get_timeout("tickets", 1.5) == 1.5
    assert client._get_timeout("tickets", 0) == 0


@pytest.mark.anyio
async def test_call_tool_and_other_operations_use_connection() -> None:
    class Connection:
        async def call_tool(self, name, arguments) -> SimpleNamespace:
            return SimpleNamespace(name=name, arguments=arguments)

        async def read_resource(self, uri) -> SimpleNamespace:
            return SimpleNamespace(uri=uri)

        async def get_prompt(self, name, arguments) -> SimpleNamespace:
            return SimpleNamespace(name=name, arguments=arguments)

    client = MCPClient({"servers": [{"name": "tickets"}]})
    client._clients["tickets"] = Connection()

    tool_result = await client.call_tool("tickets", "get_ticket", {"ticket_id": 1})
    resource = await client.read_resource("tickets", "ticket://1", timeout=1)
    prompt = await client.get_prompt("tickets", "analyze", {"ticket_id": "1"})

    assert tool_result.name == "get_ticket"
    assert resource.uri == "ticket://1"
    assert prompt.name == "analyze"


@pytest.mark.anyio
async def test_call_tool_timeout_uses_server_limit() -> None:
    class SlowConnection:
        async def call_tool(self, name, arguments) -> None:
            await asyncio.sleep(1)

    client = MCPClient({"servers": [{"name": "tickets", "tool_timeout": 0.01}]})
    client._clients["tickets"] = SlowConnection()

    with pytest.raises(asyncio.TimeoutError):
        await client.call_tool("tickets", "slow_tool", {})
