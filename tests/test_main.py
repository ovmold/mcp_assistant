import sys

import pytest

from mcp_assistant.client import llm as llm_module
from mcp_assistant.client import main as main_module


class FakeGigaChat:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.closed = False
        self.__class__.instances.append(self)

    def close(self) -> None:
        self.closed = True


class FakeMCPClient:
    instances = []
    connect_error = None

    def __init__(self, config) -> None:
        self.closed = False
        self.__class__.instances.append(self)

    async def connect_all(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error

    def get_all_tools(self) -> dict:
        return {}

    def get_all_resources(self) -> dict:
        return {}

    def get_all_resource_templates(self) -> dict:
        return {}

    def get_all_prompts(self) -> dict:
        return {}

    async def close_all(self) -> None:
        self.closed = True


class FakeTracer:
    instances = []

    def __init__(self, path) -> None:
        self.closed = False
        self.__class__.instances.append(self)

    def log(self, event_type, **data) -> None:
        pass

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def main_mocks(monkeypatch) -> None:
    for cls in (FakeGigaChat, FakeMCPClient, FakeTracer):
        cls.instances = []
    FakeMCPClient.connect_error = None

    monkeypatch.setattr(sys, "argv", ["mcp-assistant", "--ui", "cli"])
    monkeypatch.setattr(main_module.conf, "load_env", lambda path: None)
    monkeypatch.setattr(main_module, "load_config", lambda path: {"servers": []})
    monkeypatch.setattr(llm_module, "GigaChat", FakeGigaChat)
    monkeypatch.setattr(main_module, "MCPClient", FakeMCPClient)
    monkeypatch.setattr(main_module, "Tracer", FakeTracer)
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "test-key")
    monkeypatch.setenv("GIGACHAT_BASE_URL", "http://localhost")
    monkeypatch.setenv("MODEL", "test-model")

    async def run_cli(agent) -> None:
        pass

    monkeypatch.setattr(main_module, "run_cli", run_cli)


@pytest.mark.anyio
async def test_main_closes_resources_after_ui_stops(main_mocks) -> None:
    result = await main_module.main()

    assert result == 0
    assert FakeGigaChat.instances[0].closed
    assert FakeMCPClient.instances[0].closed
    assert FakeTracer.instances[0].closed


@pytest.mark.anyio
async def test_main_closes_resources_when_mcp_connection_fails(main_mocks) -> None:
    FakeMCPClient.connect_error = RuntimeError("connection failed")

    result = await main_module.main()

    assert result == 1
    assert FakeGigaChat.instances[0].closed
    assert FakeMCPClient.instances[0].closed
    assert FakeTracer.instances == []
