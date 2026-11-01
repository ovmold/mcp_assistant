# client/gradio_ui.py

import asyncio
import json
import gradio as gr

from client.agent import Agent
from client.confirmation import GradioConfirmation

# Глобальный словарь для хранения сессий (объекты asyncio нельзя хранить в gr.State)
_sessions = {}


def build_gradio_ui(agent: Agent) -> gr.Blocks:
    with gr.Blocks(title="MCP Agent") as demo:
        gr.Markdown("### MCP Agent\nАгент может попросить подтверждение — просто напишите «да» или «нет».")
        
        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(placeholder="Введите сообщение...", show_label=False)
        
        with gr.Row():
            submit_btn = gr.Button("Отправить", variant="primary")
            clear_btn = gr.Button("Очистить")

        async def respond(message: str, history: list, request: gr.Request):
            if history is None:
                history = []
                
            session_id = request.session_hash if request else "default"
            
            if session_id not in _sessions:
                _sessions[session_id] = {
                    "conf": GradioConfirmation(),
                    "task": None
                }
            
            state = _sessions[session_id]
            conf = state["conf"]

            if not message or not message.strip():
                return history, ""

            message = message.strip()

            # Проверяет: это ответ на подтверждение или новый запрос?
            if state["task"] is not None and conf.pending is not None:
                # Случай А: пользователь отвечает на запрос подтверждения
                history.append({"role": "user", "content": message})
                approved = message.lower() in {"да", "yes", "y", "д"}
                conf.resolve(approved)
            else:
                # Случай Б: новый запрос
                if state["task"] is not None:
                    # Защита: если задача еще выполняется (не ждет подтверждения)
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": "Подождите, агент ещё работает..."})
                    return history, ""

                history.append({"role": "user", "content": message})
                
                inner_task = asyncio.create_task(agent.run(message, confirmation=conf))
                state["task"] = inner_task

            # --- ОБЩАЯ ЛОГИКА ОЖИДАНИЯ ---
            # Ждет либо завершения задачи, либо запроса на подтверждение
            task = state["task"]
            shielded_task = asyncio.shield(task)
            event_task = asyncio.create_task(conf.wait_requested())
            
            try:
                done, _ = await asyncio.wait(
                    [shielded_task, event_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                event_task.cancel()

            if shielded_task in done:
                # Задача полностью завершилась (получен финальный ответ)
                state["task"] = None
                try:
                    result = task.result()
                    history.append({"role": "assistant", "content": result})
                except Exception as e:
                    history.append({"role": "assistant", "content": f"Ошибка: {e}"})
            else:
                # Задача еще работает — агент снова просит подтверждения
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

        # Привязка событий
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
        
        # Очистка истории
        clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

    return demo