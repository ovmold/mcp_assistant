from types import SimpleNamespace

import pytest

from mcp_assistant.client import mcp_client as mcp_client_module
from mcp_assistant.client.mcp_client import MCPClient


@pytest.mark.anyio
async def test_stdio_server_does_not_inherit_parent_secrets(monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "test-secret")
    captured = {}

    def fake_stdio_client(params):
        captured["params"] = params
        return object()

    class FakeClient:
        def __init__(self, transport):
            self.transport = transport

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def list_tools(self):
            return SimpleNamespace(tools=[])

        async def list_resources(self):
            return SimpleNamespace(resources=[])

        async def list_resource_templates(self):
            return SimpleNamespace(resource_templates=[])

        async def list_prompts(self):
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
async def test_connect_all_reports_root_causes_and_rejects_partial_startup(monkeypatch):
    client = MCPClient({"servers": [
        {"name": "tickets"},
        {"name": "knowledge"},
    ]})

    async def fake_connect(name, config):
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
async def test_read_resource_and_get_prompt(tickets_client):
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
