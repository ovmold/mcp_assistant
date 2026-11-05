import asyncio
from types import SimpleNamespace

import pytest

from mcp_assistant.client.ui import gradio as gradio_ui
from mcp_assistant.client.confirmation import GradioConfirmation
from mcp_assistant.client.ui.gradio import build_gradio_ui


@pytest.fixture(autouse=True)
def clean_gradio_sessions():
    gradio_ui._sessions.clear()
    yield
    gradio_ui._sessions.clear()


def get_handler(demo, name):
    return next(
        block_function.fn
        for block_function in demo.fns.values()
        if block_function.fn.__name__ == name
    )


class CancellableAgent:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def run(self, message, confirmation=None, session_id="default"):
        self.calls.append(message)
        if len(self.calls) == 1:
            self.started.set()
            await self.release.wait()
            self.finished.set()
            return "Первый ответ"
        return "Второй ответ"


class ResettableAgent:
    def __init__(self):
        self.reset_calls = []
        self.calls = []

    async def run(self, message, confirmation=None, session_id="default"):
        self.calls.append(message)
        return "Ответ"

    def reset(self, session_id="default"):
        self.reset_calls.append(session_id)


@pytest.mark.anyio
async def test_cancelled_request_does_not_block_next_message():
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


def test_clear_chat_removes_session_state():
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
async def test_expired_session_is_reset_before_new_request():
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
