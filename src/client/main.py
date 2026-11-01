"""
Точка входа MCP-клиента.

Загружает конфигурацию, инициализирует компоненты и запускает CLI-чат.
"""

import asyncio
import os
import sys
import argparse

import uvicorn
from fastapi import FastAPI
import gradio as gr

from dotenv import load_dotenv
from gigachat import GigaChat

from client.gradio_ui import build_gradio_ui
from client.agent import Agent, LLMClient
from client.config_mcp import load_config
from client.mcp_client import MCPClient
from client.confirmation import CLIConfirmation, GradioConfirmation
from client.tracer import Tracer
from scripts import conf

async def main() -> int:
    """Асинхронная точка входа."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", choices=["cli", "gradio"], default="cli",
                        help="Какой интерфейс запустить")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # 1. Проверяет токен
    load_dotenv()
    auth_key = os.environ.get("GIGACHAT_AUTH_KEY")
    if not auth_key:
        print("Переменная окружения GIGACHAT_AUTH_KEY не задана")
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
            base_url=config["api_base"],
            model=config["model"],
            credentials=auth_key, # Ключ авторизации из личного кабинета
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False,
            timeout=config.get("llm_timeout", 60.0),
        )
       
        # Передаем готовый экземпляр GigaChat в наш асинхронный клиент-обертку
        llm = LLMClient(giga_instance=giga, model=config["model"])
    except Exception as e:
        print(f"Ошибка инициализации GigaChat: {e}")
        return 1

    # 4. Подключается к MCP-серверам
    mcp = MCPClient(config)
    try:
        await mcp.connect_all()
    except Exception as e:
        print(f"Ошибка подключения к MCP-серверам: {e}")
        return 1

    try:
        # 5. Создает трассировщик
        tracer = Tracer(conf.TRACE_DIR / "session.jsonl")

        # Собирает write_tools для каждого сервера из конфига
        write_tools_by_server = {
            server["name"]: set(server.get("write_tools", []))
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
            write_tools_by_server=write_tools_by_server,
        )

        # 7. Запуск выбранного интерфейса
        if args.ui == "gradio":
            await run_gradio(agent, host=args.host, port=args.port)
        else:
            await run_cli(agent)

    finally:
        # 8. Закрывает соединения
        await mcp.close_all()

    return 0


async def run_cli(agent: Agent) -> None:
    """Простой CLI-чат."""
    print("=" * 60)
    print("MCP-агент готов. Введите запрос или /exit для выхода.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            print("Выход.")
            break

        try:
            response = await agent.run(user_input)
            print(f"\nАгент: {response}")
        except Exception as e:
            print(f"\nОшибка: {e}")


async def run_gradio(agent: Agent, host: str, port: int) -> None:
    """Запускает Gradio внутри текущего event loop.

    Использует uvicorn.Server + mount_gradio_app, а НЕ demo.launch(),
    чтобы обработчики работали в том же цикле, что и MCPClient.
    """
    demo = build_gradio_ui(agent)
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