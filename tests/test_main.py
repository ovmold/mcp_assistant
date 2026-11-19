import runpy
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
    """Проверяет закрытие LLM, MCP и Tracer после завершения UI."""
    result = await main_module.main()

    assert result == 0
    assert FakeGigaChat.instances[0].closed
    assert FakeMCPClient.instances[0].closed
    assert FakeTracer.instances[0].closed


@pytest.mark.anyio
async def test_main_closes_resources_when_mcp_connection_fails(main_mocks) -> None:
    """Проверяет закрытие созданных ресурсов при ошибке подключения MCP."""
    FakeMCPClient.connect_error = RuntimeError("connection failed")

    result = await main_module.main()

    assert result == 1
    assert FakeGigaChat.instances[0].closed
    assert FakeMCPClient.instances[0].closed
    assert FakeTracer.instances == []



@pytest.mark.anyio
async def test_main_reports_configuration_error(main_mocks, monkeypatch, capsys) -> None:
    """Проверяет код 1 и сообщение при ошибке загрузки конфигурации."""
    def fail_config(path):
        raise ValueError("invalid config")

    monkeypatch.setattr(main_module, "load_config", fail_config)

    assert await main_module.main() == 1
    assert "Ошибка конфигурации: invalid config" in capsys.readouterr().out
    assert FakeGigaChat.instances == []


@pytest.mark.anyio
async def test_main_reports_llm_initialization_error(main_mocks, monkeypatch, capsys) -> None:
    """Проверяет код 1 и сообщение при ошибке создания LLM-клиента."""
    def fail_llm(timeout):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(main_module, "create_llm", fail_llm)

    assert await main_module.main() == 1
    assert "Ошибка инициализации LLM: LLM unavailable" in capsys.readouterr().out
    assert FakeMCPClient.instances == []


@pytest.mark.anyio
async def test_main_starts_gradio_and_closes_resources(main_mocks, monkeypatch) -> None:
    """Проверяет запуск Gradio с host/port и последующее закрытие ресурсов."""
    captured = {}

    async def run_gradio(agent, host, port, session_timeout) -> None:
        captured.update(host=host, port=port, session_timeout=session_timeout)

    monkeypatch.setattr(sys, "argv", [
        "mcp-assistant", "--ui", "gradio", "--host", "127.0.0.2", "--port", "9000",
    ])
    monkeypatch.setattr(main_module, "run_gradio", run_gradio)

    assert await main_module.main() == 0
    assert captured == {"host": "127.0.0.2", "port": 9000, "session_timeout": 300.0}
    assert FakeGigaChat.instances[0].closed
    assert FakeMCPClient.instances[0].closed
    assert FakeTracer.instances[0].closed


@pytest.mark.anyio
async def test_main_closes_llm_when_mcp_shutdown_fails(main_mocks, monkeypatch) -> None:
    """Проверяет закрытие LLM даже при ошибке закрытия MCP."""
    async def fail_close(self) -> None:
        self.closed = True
        raise RuntimeError("close failed")

    monkeypatch.setattr(FakeMCPClient, "close_all", fail_close)

    with pytest.raises(RuntimeError, match="close failed"):
        await main_module.main()

    assert FakeGigaChat.instances[0].closed
    assert FakeMCPClient.instances[0].closed


def test_main_module_returns_130_on_keyboard_interrupt(monkeypatch, capsys) -> None:
    """Проверяет код завершения 130 при прерывании пользователем Ctrl+C."""
    def interrupt(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module.asyncio, "run", interrupt)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(main_module.__file__, run_name="__main__")

    assert exc_info.value.code == 130
    assert "Прервано пользователем" in capsys.readouterr().out
