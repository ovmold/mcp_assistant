"""
Управление подключениями к MCP-серверам.

Отвечает за:
- Подключение к нескольким серверам (stdio, streamable-http).
- Получение списков инструментов.
- Вызов инструментов с таймаутом.
- Корректное закрытие соединений.
"""

import asyncio
import sys
import os

from contextlib import AsyncExitStack
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool, CallToolResult

class MCPClient:
    def __init__(self, config: dict):
        self._servers = {s["name"]: s for s in config["servers"]}
        self._clients: dict[str, Client] = {}
        self._tools: dict[str, list] = {}
        self._exit_stack = AsyncExitStack()  # ← Единый стек для всего

    async def connect_all(self) -> None:
        """Подключается ко всем серверам по очереди."""
        for name, cfg in self._servers.items():
            try:
                await self._connect(name, cfg)
            except Exception as e:
                print(f"'{name}' не подключен: {e}")
        if not self._clients:
            raise RuntimeError("Не удалось подключиться ни к одному серверу")

    async def _connect(self, name: str, config: dict) -> None:
        transport = config["transport"]

        if transport["type"] == "stdio":
            merged_env = {
            **os.environ,                   # ← обязательно первым
            **(transport.get("env") or {}), # ← явный env поверх
}
            params = StdioServerParameters(
                command=transport["command"],
                args=transport.get("args", []),
                env=merged_env,
            )
            # stdio_client возвращает асинхронный контекстный менеджер
            transport_ctx = stdio_client(params)

        elif transport["type"] == "streamable-http":
            transport_ctx = streamable_http_client(
                url=transport["url"],
            )
        else:
            raise ValueError(f"Неизвестный транспорт: {transport['type']}")

        # Создает Client, передавая ему транспортный контекст
        client = Client(transport_ctx)

        # Входим в контекст клиента. 
        # Client сам выполнит initialize() (рукопожатие) при входе.
        await self._exit_stack.enter_async_context(client)

        # Получаем список инструментов
        result = await client.list_tools()

        self._clients[name] = client
        self._tools[name] = result.tools

    async def close_all(self) -> None:
        """Закрывает все соединения через единый стек."""
        # AsyncExitStack корректно закроет все Client и их транспорты 
        # в обратном порядке, в той же задаче, где они открывались.
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            print(f"Ошибка при закрытии соединений: {e}")
        finally:
            self._clients.clear()
            self._tools.clear()

    def get_all_tools(self) -> dict[str, list[Tool]]:
        """{server_name: [tools]}"""
        return dict(self._tools)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
        timeout: float | None = None,
    ) -> CallToolResult:
        """Вызывает инструмент с таймаутом."""
        client = self._clients.get(server_name)
        if not client:
            raise ValueError(f"Сервер '{server_name}' не подключён")

        if timeout is None:
            timeout = self._servers[server_name].get("tool_timeout", 30.0)

        return await asyncio.wait_for(
            client.call_tool(tool_name, arguments),
            timeout=timeout,
        )

    

