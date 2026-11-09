"""Подключения к MCP-серверам и вызовы их возможностей."""

import asyncio
from contextlib import AsyncExitStack

from mcp import Client
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    Prompt,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    Tool,
)


def _format_exception(error: Exception) -> str:
    if isinstance(error, ExceptionGroup):
        return "; ".join(
            _format_exception(suberror) for suberror in error.exceptions
        )
    message = str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


class MCPClient:
    def __init__(self, config: dict):
        self._servers = {s["name"]: s for s in config["servers"]}
        self._clients: dict[str, Client] = {}
        self._tools: dict[str, list[Tool]] = {}
        self._resources: dict[str, list[Resource]] = {}
        self._resource_templates: dict[str, list[ResourceTemplate]] = {}
        self._prompts: dict[str, list[Prompt]] = {}
        self._exit_stack = AsyncExitStack()

    async def connect_all(self) -> None:
        """Подключает все серверы или сообщает обо всех ошибках подключения."""
        errors = {}
        for name, cfg in self._servers.items():
            try:
                await self._connect(name, cfg)
            except Exception as e:
                errors[name] = e

        if errors:
            details = "; ".join(
                f"{name}: {_format_exception(error)}" for name, error in errors.items()
            )
            await self.close_all()
            raise RuntimeError(f"Ошибки подключения к MCP-серверам: {details}")
        if not self._clients:
            raise RuntimeError("Не удалось подключиться ни к одному серверу")

    async def _connect(self, name: str, config: dict) -> None:
        transport = config["transport"]

        if transport["type"] == "stdio":
            merged_env = {
                **get_default_environment(),
                **(transport.get("env") or {}),
            }
            params = StdioServerParameters(
                command=transport["command"],
                args=transport.get("args", []),
                env=merged_env,
            )
            transport_ctx = stdio_client(params)
        elif transport["type"] == "streamable-http":
            transport_ctx = streamable_http_client(url=transport["url"])
        else:
            raise ValueError(f"Неизвестный транспорт: {transport['type']}")

        client = Client(transport_ctx)
        await self._exit_stack.enter_async_context(client)

        tools = await client.list_tools()
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        prompts = await client.list_prompts()

        self._clients[name] = client
        self._tools[name] = tools.tools
        self._resources[name] = resources.resources
        self._resource_templates[name] = templates.resource_templates
        self._prompts[name] = prompts.prompts

    async def close_all(self) -> None:
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            print(f"Ошибка при закрытии соединений: {e}")
        finally:
            self._clients.clear()
            self._tools.clear()
            self._resources.clear()
            self._resource_templates.clear()
            self._prompts.clear()

    def get_all_tools(self) -> dict[str, list[Tool]]:
        return dict(self._tools)

    def get_all_resources(self) -> dict[str, list[Resource]]:
        return dict(self._resources)

    def get_all_resource_templates(self) -> dict[str, list[ResourceTemplate]]:
        return dict(self._resource_templates)

    def get_all_prompts(self) -> dict[str, list[Prompt]]:
        return dict(self._prompts)

    def _get_client(self, server_name: str) -> Client:
        client = self._clients.get(server_name)
        if not client:
            raise ValueError(f"Сервер '{server_name}' не подключён")
        return client

    def _get_timeout(self, server_name: str, timeout: float | None) -> float:
        if timeout is not None:
            return timeout
        return self._servers[server_name].get("tool_timeout", 30.0)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
        timeout: float | None = None,
    ) -> CallToolResult:
        client = self._get_client(server_name)
        return await asyncio.wait_for(
            client.call_tool(tool_name, arguments),
            timeout=self._get_timeout(server_name, timeout),
        )

    async def read_resource(
        self,
        server_name: str,
        uri: str,
        timeout: float | None = None,
    ) -> ReadResourceResult:
        client = self._get_client(server_name)
        return await asyncio.wait_for(
            client.read_resource(uri),
            timeout=self._get_timeout(server_name, timeout),
        )

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: dict[str, str],
        timeout: float | None = None,
    ) -> GetPromptResult:
        client = self._get_client(server_name)
        return await asyncio.wait_for(
            client.get_prompt(prompt_name, arguments),
            timeout=self._get_timeout(server_name, timeout),
        )
