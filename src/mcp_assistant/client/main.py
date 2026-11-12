"""
Точка входа MCP-клиента.

Загружает конфигурацию, инициализирует компоненты и запускает выбранный UI.
"""

import asyncio
import sys
import argparse

from mcp_assistant.client.ui.gradio import run_gradio
from mcp_assistant.client.ui.cli import run_cli
from mcp_assistant.client.agent import Agent
from mcp_assistant.client.llm import create_llm
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
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Адрес веб-сервера (только для --ui gradio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Порт веб-сервера (только для --ui gradio)",
    )
    args = parser.parse_args()

    # 1. Загружает окружение и конфигурацию
    conf.load_env(conf.PROJECT_DIR / ".env")
    try:
        config = load_config(conf.CLIENT_DIR / "servers.json")
    except Exception as e:
        print(f"Ошибка конфигурации: {e}")
        return 1

    # 2. Создает клиент выбранного LLM-провайдера
    try:
        llm = create_llm(timeout=config.get("llm_timeout", 60.0))
    except Exception as e:
        print(f"Ошибка инициализации LLM: {e}")
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
            max_history_chars=config.get("max_history_chars", 60000),
            max_tool_result_chars=config.get("max_tool_result_chars", 12000),
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
                llm.close()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        exit_code = 130
    sys.exit(exit_code)