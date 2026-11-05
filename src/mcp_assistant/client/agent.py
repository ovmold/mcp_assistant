import asyncio
import json
from typing import Any

from gigachat.models import Chat, Messages, MessagesRole

from mcp_assistant.client.mcp_client import MCPClient
from mcp_assistant.client.tool_converter import ToolConverter
from mcp_assistant.client.confirmation import ConfirmationProvider
from mcp_assistant.client.tracer import Tracer
from mcp_assistant.client import agent_prompts

# ============================================================================
# 1. ЗАЩИТА ОТ PROMPT INJECTION
# ============================================================================

def wrap_tool_result(server: str, tool: str, result: str) -> str:
    """Помечает результат как данные; правила разбора заданы в system prompt."""
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
        always_confirm_tools: dict[str, set[str]] | None = None,
        max_history_messages: int = 40,
    ):
        self.mcp = mcp_client
        self.llm = llm
        self.tracer = tracer
        self.confirmation = confirmation
        self.max_steps = max_steps
        self.always_confirm_tools = always_confirm_tools or {}
        self.max_history_messages = max_history_messages
        self._histories: dict[str, list[dict]] = {}

        self.converter = ToolConverter()
        for server, tools in self.mcp.get_all_tools().items():
            self.converter.add_server_tools(server, tools)

        self.gigachat_functions = self.converter.get_gigachat_functions()
        self.system_prompt = agent_prompts.build_system_prompt(self.gigachat_functions)

    # --- управление историей ------------------------------------------------

    def reset(self, session_id: str = "default") -> None:
        """Очистить историю диалога (кнопка «Очистить» / новый разговор)."""
        self._histories.pop(session_id, None)

    def get_history(self, session_id: str = "default") -> list[dict]:
        """Копия истории — для тестов и отладки."""
        return [dict(m) for m in self._histories.get(session_id, [])]

    def _trim(self, session_id: str) -> None:
        """
        Обрезает историю до max_history_messages, сохраняя:
          - system-сообщение (индекс 0),
          - целые ходы: граница режется только по сообщению role="user",
            чтобы не разорвать пары assistant(function_call) → function.
        """
        messages = self._histories.get(session_id)
        if not messages:
            return

        system = messages[0] if messages[0].get("role") == "system" else None
        head = [system] if system else []
        body = messages[len(head):]

        if len(body) <= self.max_history_messages:
            return

        cut = len(body) - self.max_history_messages
        while cut < len(body) and body[cut].get("role") != "user":
            cut += 1
        if cut >= len(body):
            return   # не найдена безопасная граница — остается, как есть

        messages[:] = head + body[cut:]

    async def run(
        self,
        user_input: str,
        confirmation: ConfirmationProvider | None = None,
        session_id: str = "default",
    ) -> str:
        confirmation = confirmation or self.confirmation

        messages = self._histories.setdefault(session_id, [])
        if not messages:
            messages.append({"role": "system", "content": self.system_prompt})

        self._trim(session_id)                    # обрезает ДО нового хода
        messages.append({"role": "user", "content": user_input})
        self.tracer.log("user_request", content=user_input, session_id=session_id)

        for step in range(1, self.max_steps + 1):
            try:
                response = await self.llm.chat(messages, self.gigachat_functions)
            except Exception as e:
                error_msg = f"Ошибка: не удалось получить ответ от модели ({e})"
                messages.append({"role": "assistant", "content": error_msg})
                return error_msg

            message = response.choices[0].message

            if not message.function_call:
                final = message.content or ""
                messages.append({"role": "assistant", "content": final})
                self.tracer.log("final_answer", content=final, step=step, session_id=session_id)
                return final

            self.tracer.log(
                "llm_response",
                content=message.content,
                function_call={
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments,
                },
                tokens=response.usage.total_tokens if getattr(response, "usage", None) else 0,
                session_id=session_id,
            )

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "function_call": {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments,
                },
            })

            result = await self._execute_function_call(message.function_call, confirmation)

            messages.append({
                "role": "function",
                "content": result,
                "name": message.function_call.name,
            })

        max_msg = "Достигнут лимит шагов. Задача не завершена."
        messages.append({"role": "assistant", "content": max_msg})
        self.tracer.log("max_steps_reached", step=self.max_steps, session_id=session_id)
        return max_msg

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

         # Базовая проверка по аннотациям MCP (fail-safe)
        needs_confirm, reason = self.converter.requires_confirmation(prefixed_name)

        # Дополнительная страховка из конфига:
        # всегда подтверждать перечисленные инструменты, даже если аннотации
        # утверждают readOnlyHint=true (например, не доверяем серверу).
        if tool_name in self.always_confirm_tools.get(server, set()):
            needs_confirm = True
            reason = f"config always_confirm ({reason})"

        if needs_confirm:
            self.tracer.log(
                "confirmation_requested",
                server=server, tool=tool_name, reason=reason, arguments=arguments,
            )
            approved = await self._confirm_action(tool_name, arguments, confirmation, reason)
            if not approved:
                self.tracer.log("confirmation_denied", tool=tool_name, arguments=arguments)
                return json.dumps(
                    {"result": "Пользователь отказался выполнять это действие."},
                    ensure_ascii=False,
                )

        self.tracer.log("tool_call", server=server, tool=tool_name, arguments=arguments)
        try:
            result = await self.mcp.call_tool(server, tool_name, arguments)
            text = self._extract_text(result)
            self.tracer.log("tool_result", server=server, tool=tool_name, result=text[:500])
            return json.dumps({"result": wrap_tool_result(server, tool_name, text)}, ensure_ascii=False)
        except asyncio.TimeoutError:
            self.tracer.log("tool_error", server=server, tool=tool_name, error="timeout")
            return json.dumps({"error": "Ошибка вызова: превышен таймаут ожидания инструмента"}, ensure_ascii=False)
        except Exception as e:
            self.tracer.log("tool_error", server=server, tool=tool_name, error=str(e))
            return json.dumps({"error": f"Ошибка вызова: {e}"}, ensure_ascii=False)

    async def _confirm_action(
        self,
        tool_name: str,
        arguments: dict,
        confirmation: ConfirmationProvider,
        reason: str = "",
    ) -> bool:
        if hasattr(confirmation, "confirm_with_reason"):
            approved = await confirmation.confirm_with_reason(tool_name, arguments, reason)
        else:
            approved = await confirmation.confirm(tool_name, arguments)
        self.tracer.log("confirmation", tool=tool_name, approved=approved, reason=reason)
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