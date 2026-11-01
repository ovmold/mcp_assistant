import asyncio
import json
from typing import Any

from gigachat.models import Chat, Messages, MessagesRole

from client.mcp_client import MCPClient
from client.tool_converter import ToolConverter
from client.confirmation import ConfirmationProvider
from client.tracer import Tracer
from scripts import conf

# ============================================================================
# 1. ЗАЩИТА ОТ PROMPT INJECTION
# ============================================================================

def wrap_tool_result(server: str, tool: str, result: str) -> str:
    """Оборачивает результат инструмента в теги для защиты от injection."""
    return f"<tool_result server='{server}' tool='{tool}'>\n{result}\n</tool_result>"


# ============================================================================
# 2. КЛИЕНТ LLM (GIGACHAT SDK)
# ============================================================================

class LLMClient:
    """Обертка над официальным SDK GigaChat (асинхронная)."""

    def __init__(self, giga_instance, model):
        self.giga = giga_instance
        self.model = model  

    async def chat(
        self,
        messages: list[dict],
        functions: list[dict] | None = None,
    ) -> Any:
        """Отправляет запрос в GigaChat (синхронный вызов в отдельном потоке)."""

        def _sync_chat():
            gc_messages = []
            for msg in messages:
                role_str = msg["role"]
                if role_str == "system":
                    role = MessagesRole.SYSTEM
                elif role_str == "user":
                    role = MessagesRole.USER
                elif role_str == "assistant":
                    role = MessagesRole.ASSISTANT
                elif role_str in ("tool", "function"):
                    role = MessagesRole.FUNCTION
                else:
                    role = MessagesRole.USER

                gc_messages.append(Messages(
                    role=role,
                    content=msg.get("content") or "",
                    function_call=msg.get("function_call"),
                    name=msg.get("name"),
                ))

            chat_request = Chat(
                messages=gc_messages,
                model=self.model,
                functions=functions or None,
            )
            return self.giga.chat(chat_request)

        return await asyncio.to_thread(_sync_chat)


# ============================================================================
# 3. АГЕНТНЫЙ ЦИКЛ
# ============================================================================

class Agent:
    """Агент, который связывает GigaChat, MCP-инструменты и защиту."""

    def __init__(
        self,
        mcp_client: MCPClient,
        llm:LLMClient,
        tracer: Tracer,
        confirmation: ConfirmationProvider,
        max_steps: int = 8,
        write_tools_by_server: dict[str, set[str]] | None = None,
    ):
        self.mcp = mcp_client
        self.llm = llm
        self.tracer = tracer
        self.confirmation = confirmation
        self.max_steps = max_steps
        self.write_tools_by_server = write_tools_by_server or {}

        self.converter = ToolConverter()
        for server, tools in self.mcp.get_all_tools().items():
            self.converter.add_server_tools(server, tools)

        self.gigachat_functions = self.converter.get_gigachat_functions()

    async def run(
        self,
        user_input: str,
        confirmation: ConfirmationProvider | None = None,
    ) -> str:
        confirmation = confirmation or self.confirmation
        messages = [
            {"role": "system", "content": conf.SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        self.tracer.log("user_request", content=user_input)

        for step in range(1, self.max_steps + 1):
            try:
                response = await self.llm.chat(messages, self.gigachat_functions)
            except Exception as e:
                return f"Ошибка: не удалось получить ответ от модели ({e})"

            message = response.choices[0].message

            if not message.function_call:
                final = message.content or ""
                self.tracer.log("final_answer", content=final, step=step)
                return final

            self.tracer.log(
                "llm_response",
                content=message.content,
                function_call={
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments,
                },
                tokens=response.usage.total_tokens if getattr(response, "usage", None) else 0,
            )

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "function_call": {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments,
                }
            })

            result = await self._execute_function_call(message.function_call, confirmation)

            messages.append({
                "role": "function",
                "content": result,
                "name": message.function_call.name,
            })

        self.tracer.log("max_steps_reached", step=self.max_steps)
        return "Достигнут лимит шагов. Задача не завершена."

    async def _execute_function_call(
        self,
        function_call,
        confirmation: ConfirmationProvider,
    ) -> str:
        prefixed_name = function_call.name
        args_str = function_call.arguments

        try:
            arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError as e:
            self.tracer.log("invalid_json", tool=prefixed_name, error=str(e))
            return json.dumps({"error": f"Невалидный JSON в аргументах: {e}"}, ensure_ascii=False)

        resolved = self.converter.resolve(prefixed_name)
        if not resolved:
            return json.dumps({"error": f"Неизвестный инструмент: {prefixed_name}"}, ensure_ascii=False)
        server, tool_name = resolved

        write_tools = self.write_tools_by_server.get(server, set())

        if tool_name in write_tools:
            approved = await self._confirm_action(tool_name, arguments, confirmation)
            if not approved:
                self.tracer.log("confirmation_denied", tool=tool_name, arguments=arguments)
                return json.dumps({"result": "Пользователь отказался выполнять это действие."}, ensure_ascii=False)

        self.tracer.log("tool_call", server=server, tool=tool_name, arguments=arguments)
        try:
            # Добавляем таймаут на случай, если MCP-сервер завис
            result = await asyncio.wait_for(
                self.mcp.call_tool(server, tool_name, arguments), 
                timeout=60.0
            )
            text = self._extract_text(result)
            self.tracer.log("tool_result", server=server, tool=tool_name, result=text[:500])
            return json.dumps({"result": wrap_tool_result(server, tool_name, text)}, ensure_ascii=False)
        except asyncio.TimeoutError:
            self.tracer.log("tool_error", server=server, tool=tool_name, error="timeout")
            return json.dumps({"error": "Ошибка вызова: превышен таймаут ожидания инструмента (60 сек)"}, ensure_ascii=False)
        except Exception as e:
            self.tracer.log("tool_error", server=server, tool=tool_name, error=str(e))
            return json.dumps({"error": f"Ошибка вызова: {e}"}, ensure_ascii=False)

    async def _confirm_action(
        self,
        tool_name: str,
        arguments: dict,
        confirmation: ConfirmationProvider,
    ) -> bool:
        approved = await confirmation.confirm(tool_name, arguments)
        self.tracer.log("confirmation", tool=tool_name, approved=approved)
        return approved

    @staticmethod
    def _extract_text(result) -> str:
        is_error = getattr(result, "is_error", getattr(result, "isError", False))
        if is_error:
            return f"[Ошибка] {result.content[0].text if result.content else 'unknown'}"
        if not result.content:
            return "(пустой результат)"
        return "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )