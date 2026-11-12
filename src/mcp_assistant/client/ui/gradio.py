import asyncio
import json
import time

import gradio as gr
import uvicorn
from fastapi import FastAPI

from mcp_assistant.client.agent import Agent
from mcp_assistant.client.confirmation import GradioConfirmation

_sessions = {}


def _new_session() -> dict:
    return {
        "conf": GradioConfirmation(),
        "task": None,
        "last_active": time.monotonic(),
    }


def _remove_session(agent: Agent, session_id: str) -> None:
    state = _sessions.pop(session_id, None)
    if state is not None:
        task = state.get("task")
        if task is not None and not task.done():
            task.cancel()
    agent.reset(session_id)


def _remove_expired_sessions(agent: Agent, timeout: float) -> None:
    """Очищает неактивные сессии при следующем обращении к UI."""
    now = time.monotonic()
    for session_id, state in list(_sessions.items()):
        if now - state.get("last_active", now) > timeout:
            _remove_session(agent, session_id)


def _clear_completed_task(state: dict) -> None:
    """Очищает ссылку на завершённую задачу и извлекает её исключение."""
    task = state.get("task")
    if task is None or not task.done():
        return
    state["task"] = None
    if not task.cancelled():
        task.exception()


def build_gradio_ui(agent: Agent, session_timeout: float = 300.0) -> gr.Blocks:
    with gr.Blocks(title="MCP Agent") as demo:
        gr.Markdown("### MCP Agent\nАгент может попросить подтверждение — просто напишите «да» или «нет».")

        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(placeholder="Введите сообщение...", show_label=False)

        with gr.Row():
            submit_btn = gr.Button("Отправить", variant="primary")
            clear_btn = gr.Button("Очистить")

        async def respond(
            message: str,
            history: list[dict[str, str]] | None,
            request: gr.Request,
        ) -> tuple[list[dict[str, str]], str]:
            # Проверяет входные данные и убирает пробелы по краям сообщения
            if history is None:
                history = []
            if not message or not message.strip():
                return history, ""

            message = message.strip()

            # Определяет сессию и удаляет давно неактивные
            session_id = request.session_hash if request else "default"
            _remove_expired_sessions(agent, session_timeout)

            # Создает состояние новой сессии или обновляем активность текущей
            state = _sessions.get(session_id)
            if state is None:
                state = _new_session()
                _sessions[session_id] = state
            else:
                state["last_active"] = time.monotonic()
            _clear_completed_task(state)

            conf = state["conf"]

            # Если агент ждет подтверждения, передает ответ текущей задаче
            if state["task"] is not None and conf.pending is not None:
                history.append({"role": "user", "content": message})
                approved = message.lower() in {"да", "yes", "y", "д"}
                conf.resolve(approved)
            else:
                # Не запускает вторую задачу, пока предыдущая еще выполняется
                if state["task"] is not None:
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": "Подождите, агент ещё работает..."})
                    return history, ""

                # Запускает новый запрос агента в фоне
                history.append({"role": "user", "content": message})
                state["task"] = asyncio.create_task(
                    agent.run(message, confirmation=conf, session_id=session_id)
                )

            # Ждет либо завершения агента, либо запроса подтверждения
            task = state["task"]
            # Shield не дает отмене HTTP-запроса остановить работу агента
            shielded_task = asyncio.shield(task)
            event_task = asyncio.create_task(conf.wait_requested())

            try:
                done, _ = await asyncio.wait(
                    [shielded_task, event_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                # Освобождает вспомогательное ожидание независимо от результата
                event_task.cancel()

            # Показывает конечный ответ либо данные операции для подтверждения
            if shielded_task in done:
                state["task"] = None
                try:
                    result = task.result()
                    history.append({"role": "assistant", "content": result})
                except asyncio.CancelledError:
                    history.append({"role": "assistant", "content": "Операция отменена."})
                except Exception as e:
                    history.append({"role": "assistant", "content": f"Ошибка: {e}"})
            else:
                history.append({"role": "assistant", "content": _format_pending(conf.pending)})

            return history, ""

        def _format_pending(pending: tuple[str, dict]) -> str:
            tool, args = pending
            args_json = json.dumps(args, ensure_ascii=False, indent=2)
            return (
                "**Запрос на подтверждение**\n\n"
                f"Агент хочет выполнить: `{tool}`\n\n"
                f"Аргументы:\n```json\n{args_json}\n```\n\n"
                "Напишите **да** или **нет**."
            )

        def clear_chat(request: gr.Request) -> tuple[list, str]:
            session_id = request.session_hash if request else "default"
            _remove_session(agent, session_id)
            return [], ""

        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(clear_chat, None, [chatbot, msg])

    return demo


async def run_gradio(
    agent: Agent,
    host: str,
    port: int,
    session_timeout: float = 300.0,
) -> None:
    """Запускает Gradio внутри текущего event loop.

    Использует uvicorn.Server + mount_gradio_app, а НЕ demo.launch(),
    чтобы обработчики работали в том же цикле, что и MCPClient.
    """
    demo = build_gradio_ui(agent, session_timeout=session_timeout)
    demo.queue()  # очередь нужна для async-обработчиков

    app = FastAPI()
    app = gr.mount_gradio_app(app, demo, path="/")

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="info")
    )
    await server.serve()
