"""
Точка входа MCP-клиента.

Загружает конфигурацию, инициализирует компоненты и запускает выбранный UI.
"""

import asyncio
import os
import sys
import argparse

import uvicorn
from fastapi import FastAPI
import gradio as gr

from gigachat import GigaChat

from mcp_assistant.client.ui.gradio import build_gradio_ui
from mcp_assistant.client.ui.cli import run_cli
from mcp_assistant.client.agent import Agent, LLMClient
from mcp_assistant.client.config_mcp import load_config
from mcp_assistant.client.mcp_client import MCPClient
from mcp_assistant.client.confirmation import CLIConfirmation, GradioConfirmation
from mcp_assistant.client.tracer import Tracer
from mcp_assistant import config as conf

async def main() -> int:
    """Асинхронная точка входа."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", choices=["cli", "gradio"], default="cli",
                        help="Какой интерфейс запустить")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # 1. Проверяет токен, api_base_url и модель
    conf.load_env(conf.PROJECT_DIR / ".env")
    auth_key = os.environ.get("GIGACHAT_AUTH_KEY")
    if not auth_key:
        print("Переменная окружения GIGACHAT_AUTH_KEY не задана")
        return 1
    api_base = os.environ.get("GIGACHAT_BASE_URL")
    if not api_base:
        print("Переменная окружения GIGACHAT_BASE_URL не задана")
        return 1
    model = os.environ.get("MODEL")
    if not model:
        print("Переменная окружения MODEL не задана")
        return 1

    # 2. Загружает конфиг
    try:
        config = load_config(conf.CLIENT_DIR / "servers.json")
    except Exception as e:
        print(f"Ошибка конфигурации: {e}")
        return 1

    # 3. Создает LLM-клиент через официальный SDK
    try:
        giga = GigaChat(
            base_url=api_base,
            model=model,
            credentials=auth_key, # Ключ авторизации из личного кабинета
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False,
            timeout=config.get("llm_timeout", 60.0),
        )
       
        # Передаем готовый экземпляр GigaChat в асинхронный клиент-обертку
        llm = LLMClient(giga_instance=giga, model=model)
    except Exception as e:
        print(f"Ошибка инициализации GigaChat: {e}")
        return 1

    # 4. Подключается к MCP-серверам
    mcp = MCPClient(config)
    tracer = None
    try:
        try:
            await mcp.connect_all()
        except Exception as e:
            print(f"Ошибка подключения к MCP-серверам: {e}")
            return 1

        # 5. Создает трассировщик
        tracer = Tracer(conf.TRACE_DIR / "session.jsonl")

        # Собирает write_tools для каждого сервера из конфига
        always_confirm_by_server = {
            server["name"]: set(server.get("always_confirm_by_server", []))
            for server in config["servers"]
        }

        if args.ui == "gradio":
            confirm = GradioConfirmation()
        else:
            confirm = CLIConfirmation()

        # 6. Создает агент
        agent = Agent(
            mcp_client=mcp,
            llm=llm,
            tracer=tracer,
            confirmation=confirm,
            max_steps=config.get("max_steps", 8),
            always_confirm_tools=always_confirm_by_server,
            max_history_messages=config.get("max_history_messages", 40),
        )

        # 7. Запуск выбранного интерфейса
        if args.ui == "gradio":
            await run_gradio(
                agent,
                host=args.host,
                port=args.port,
                session_timeout=config.get("session_timeout", 300.0),
            )
        else:
            await run_cli(agent)

    finally:
        # 8. Закрывает все созданные ресурсы
        try:
            if tracer is not None:
                tracer.close()
        finally:
            try:
                await mcp.close_all()
            finally:
                giga.close()

    return 0


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


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        exit_code = 130
    sys.exit(exit_code)