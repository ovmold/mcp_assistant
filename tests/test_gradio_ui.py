import asyncio
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_assistant.client.ui import gradio as gradio_ui
from mcp_assistant.client.confirmation import GradioConfirmation
from mcp_assistant.client.ui.gradio import build_gradio_ui


@pytest.fixture(autouse=True)
def clean_gradio_sessions() -> Iterator[None]:
    gradio_ui._sessions.clear()
    yield
    gradio_ui._sessions.clear()


def get_handler(demo, name) -> Callable[..., Any]:
    return next(
        block_function.fn
        for block_function in demo.fns.values()
        if block_function.fn.__name__ == name
    )


class CancellableAgent:
    def __init__(self) -> None:
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def run(self, message, confirmation=None, session_id="default") -> str:
        self.calls.append(message)
        if len(self.calls) == 1:
            self.started.set()
            await self.release.wait()
            self.finished.set()
            return "Первый ответ"
        return "Второй ответ"


class ResettableAgent:
    def __init__(self) -> None:
        self.reset_calls = []
        self.calls = []

    async def run(self, message, confirmation=None, session_id="default") -> str:
        self.calls.append(message)
        return "Ответ"

    def reset(self, session_id="default") -> None:
        self.reset_calls.append(session_id)


@pytest.mark.anyio
async def test_cancelled_request_does_not_block_next_message() -> None:
    """Проверяет, что отмена HTTP-обработчика не отменяет задачу агента и не блокирует следующий запрос."""
    agent = CancellableAgent()
    demo = build_gradio_ui(agent)
    respond = get_handler(demo, "respond")
    request = SimpleNamespace(session_hash="cancelled-session")

    handler_task = asyncio.create_task(respond("Первый запрос", [], request))
    await asyncio.wait_for(agent.started.wait(), timeout=1)
    handler_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler_task

    agent.release.set()
    await asyncio.wait_for(agent.finished.wait(), timeout=1)

    history, text = await asyncio.wait_for(
        respond("Второй запрос", [], request), timeout=1
    )

    assert text == ""
    assert agent.calls == ["Первый запрос", "Второй запрос"]
    assert history[-1] == {"role": "assistant", "content": "Второй ответ"}


def test_clear_chat_removes_session_state() -> None:
    """Проверяет удаление состояния сессии и истории при очистке чата."""
    agent = ResettableAgent()
    demo = build_gradio_ui(agent)
    clear_chat = get_handler(demo, "clear_chat")
    request = SimpleNamespace(session_hash="cleared-session")
    gradio_ui._sessions[request.session_hash] = {
        "conf": GradioConfirmation(),
        "task": None,
    }

    result = clear_chat(request)

    assert result == ([], "")
    assert agent.reset_calls == [request.session_hash]
    assert request.session_hash not in gradio_ui._sessions


@pytest.mark.anyio
async def test_expired_session_is_reset_before_new_request() -> None:
    """Проверяет удаление просроченной сессии перед новым сообщением."""
    agent = ResettableAgent()
    demo = build_gradio_ui(agent, session_timeout=10.0)
    respond = get_handler(demo, "respond")
    request = SimpleNamespace(session_hash="expired-session")
    old_state = {
        "conf": GradioConfirmation(),
        "task": None,
        "last_active": asyncio.get_running_loop().time() - 20.0,
    }
    gradio_ui._sessions[request.session_hash] = old_state

    history, _ = await respond("Новый запрос", [], request)

    assert agent.reset_calls == [request.session_hash]
    assert agent.calls == ["Новый запрос"]
    assert history[-1] == {"role": "assistant", "content": "Ответ"}
    assert gradio_ui._sessions[request.session_hash] is not old_state



@pytest.mark.anyio
async def test_respond_ignores_blank_message() -> None:
    """Проверяет, что respond() не запускает агента для пустого сообщения."""
    demo = build_gradio_ui(ResettableAgent())
    respond = get_handler(demo, "respond")

    history, text = await respond("   ", None, None)

    assert history == []
    assert text == ""


@pytest.mark.anyio
async def test_respond_rejects_parallel_request_in_same_session() -> None:
    """Проверяет запрет второй задачи, пока текущая сессия занята."""
    agent = ResettableAgent()
    demo = build_gradio_ui(agent)
    respond = get_handler(demo, "respond")
    request = SimpleNamespace(session_hash="busy")
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    gradio_ui._sessions["busy"] = {
        "conf": GradioConfirmation(),
        "task": task,
        "last_active": gradio_ui.time.monotonic(),
    }

    history, text = await respond("Новый запрос", [], request)

    assert text == ""
    assert history[-1]["content"] == "Подождите, агент ещё работает..."
    assert agent.calls == []
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_respond_handles_confirmation_in_next_message() -> None:
    """Проверяет показ подтверждения и продолжение той же задачи после ответа «да»."""
    class ConfirmingAgent(ResettableAgent):
        async def run(self, message, confirmation=None, session_id="default") -> str:
            self.calls.append(message)
            approved = await confirmation.confirm_with_reason(
                "update_status",
                {"ticket_id": 1},
                "write operation",
            )
            return "Выполнено" if approved else "Отменено"

    agent = ConfirmingAgent()
    demo = build_gradio_ui(agent)
    respond = get_handler(demo, "respond")
    request = SimpleNamespace(session_hash="confirmation")

    history, _ = await asyncio.wait_for(respond("Закрой тикет", [], request), timeout=1)
    assert "Запрос на подтверждение" in history[-1]["content"]
    assert "update_status" in history[-1]["content"]

    history, _ = await asyncio.wait_for(respond("да", history, request), timeout=1)
    assert history[-1] == {"role": "assistant", "content": "Выполнено"}
    assert agent.calls == ["Закрой тикет"]


@pytest.mark.anyio
async def test_clear_chat_cancels_running_task() -> None:
    """Проверяет отмену активной задачи и очистку сессии кнопкой «Очистить»."""
    agent = ResettableAgent()
    demo = build_gradio_ui(agent)
    clear_chat = get_handler(demo, "clear_chat")
    request = SimpleNamespace(session_hash="running")
    task = asyncio.create_task(asyncio.Event().wait())
    gradio_ui._sessions["running"] = {
        "conf": GradioConfirmation(),
        "task": task,
    }

    assert clear_chat(request) == ([], "")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.reset_calls == ["running"]


@pytest.mark.anyio
async def test_completed_failed_task_is_cleared() -> None:
    """Проверяет очистку завершённой задачи и извлечение её исключения."""
    async def fail() -> None:
        raise RuntimeError("background failure")

    task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    state = {"task": task}

    gradio_ui._clear_completed_task(state)

    assert state["task"] is None
    assert isinstance(task.exception(), RuntimeError)


@pytest.mark.anyio
async def test_run_gradio_starts_uvicorn_with_mounted_app(monkeypatch) -> None:
    """Проверяет монтирование Gradio в FastAPI и запуск Uvicorn с нужными host и port."""
    events = []

    class Demo:
        def queue(self) -> None:
            events.append("queue")

    class Config:
        def __init__(self, app, host, port, log_level) -> None:
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level

    class Server:
        def __init__(self, config) -> None:
            self.config = config

        async def serve(self) -> None:
            events.append(("serve", self.config.host, self.config.port))

    app = SimpleNamespace(name="FastAPI")
    monkeypatch.setattr(gradio_ui, "build_gradio_ui", lambda agent, session_timeout: Demo())
    monkeypatch.setattr(gradio_ui, "FastAPI", lambda: app)
    monkeypatch.setattr(gradio_ui.gr, "mount_gradio_app", lambda app, demo, path: app)
    monkeypatch.setattr(gradio_ui.uvicorn, "Config", Config)
    monkeypatch.setattr(gradio_ui.uvicorn, "Server", Server)

    await gradio_ui.run_gradio(ResettableAgent(), "127.0.0.1", 7860)

    assert events == ["queue", ("serve", "127.0.0.1", 7860)]
