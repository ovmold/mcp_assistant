import asyncio
import html
import json
from typing import Any

from mcp.types import Tool, ToolAnnotations

from mcp_assistant.client.llm import ChatLLM
from mcp_assistant.client.mcp_client import MCPClient
from mcp_assistant.client.tool_converter import ToolConverter
from mcp_assistant.client.confirmation import ConfirmationProvider
from mcp_assistant.client.tracer import Tracer
from mcp_assistant.client import agent_prompts

# ============================================================================
# 1. ЗАЩИТА ОТ PROMPT INJECTION
# ============================================================================

def wrap_tool_result(server: str, tool: str, result: str) -> str:
    """Помечает результат как данные и экранирует XML-подобную разметку в нём."""
    escaped_result = html.escape(result)
    return f"<tool_result server='{server}' tool='{tool}'>\n{escaped_result}\n</tool_result>"


class InvalidToolArgumentsError(ValueError):
    pass


class UserConfirmationDeniedError(Exception):
    pass


# ============================================================================
# 2. АГЕНТНЫЙ ЦИКЛ
# ============================================================================

class Agent:
    """Агент, который связывает LLM, MCP-инструменты и защиту."""

    def __init__(
        self,
        mcp_client: MCPClient,
        llm: ChatLLM,
        tracer: Tracer,
        confirmation: ConfirmationProvider,
        max_steps: int = 8,
        always_confirm_tools: dict[str, set[str]] | None = None,
        max_history_messages: int = 40,
        max_history_chars: int = 60_000,
        max_tool_result_chars: int = 12_000,
    ) -> None:
        self.mcp = mcp_client
        self.llm = llm
        self.tracer = tracer
        self.confirmation = confirmation
        self.max_steps = max_steps
        self.always_confirm_tools = always_confirm_tools or {}
        self.max_history_messages = max_history_messages
        self.max_history_chars = max_history_chars
        self.max_tool_result_chars = max_tool_result_chars
        self._histories: dict[str, list[dict]] = {}

        self.converter = ToolConverter()
        for server, tools in self.mcp.get_all_tools().items():
            self.converter.add_server_tools(server, tools)
        self._add_capability_tools()

        self.gigachat_functions = self.converter.get_gigachat_functions()
        self.system_prompt = agent_prompts.build_system_prompt(self.gigachat_functions)

    def _add_capability_tools(self) -> None:
        resources = self.mcp.get_all_resources()
        templates = self.mcp.get_all_resource_templates()
        prompts = self.mcp.get_all_prompts()

        for server in self.mcp.get_all_tools():
            capability_tools = []
            uris = [str(resource.uri) for resource in resources.get(server, [])]
            uris.extend(template.uri_template for template in templates.get(server, []))
            if uris:
                capability_tools.append(Tool(
                    name="read_resource",
                    description=(
                        "Читает MCP-ресурс по URI. Доступные ресурсы и шаблоны: "
                        + ", ".join(uris)
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {"uri": {"type": "string"}},
                        "required": ["uri"],
                    },
                    annotations=ToolAnnotations(readOnlyHint=True),
                ))

            for prompt in prompts.get(server, []):
                arguments = prompt.arguments or []
                capability_tools.append(Tool(
                    name=f"get_prompt_{prompt.name}",
                    description=(
                        f"Применяет MCP-промпт {prompt.name}: "
                        f"{prompt.description or 'без описания'}"
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            argument.name: {
                                "type": "string",
                                "description": argument.description or "",
                            }
                            for argument in arguments
                        },
                        "required": [
                            argument.name for argument in arguments if argument.required
                        ],
                    },
                    annotations=ToolAnnotations(readOnlyHint=True),
                ))

            self.converter.add_server_tools(server, capability_tools)

    # --- управление историей ------------------------------------------------

    def reset(self, session_id: str = "default") -> None:
        """Очистить историю диалога (кнопка «Очистить» / новый разговор)."""
        self._histories.pop(session_id, None)

    def get_history(self, session_id: str = "default") -> list[dict]:
        """Копия истории — для тестов и отладки."""
        return [dict(m) for m in self._histories.get(session_id, [])]

    @staticmethod
    def _history_size(messages: list[dict]) -> int:
        return sum(len(json.dumps(message, ensure_ascii=False)) for message in messages)

    def _trim(self, session_id: str) -> None:
        """Удаляет старые целые ходы до соблюдения лимитов сообщений и символов."""
        messages = self._histories.get(session_id)
        if not messages:
            return

        system = messages[0] if messages[0].get("role") == "system" else None
        head = [system] if system else []
        body = messages[len(head):]

        while body and (
            len(body) > self.max_history_messages
            or self._history_size(head + body) > self.max_history_chars
        ):
            cut = 1
            while cut < len(body) and body[cut].get("role") != "user":
                cut += 1
            body = body[cut:]

        messages[:] = head + body

    async def run(
        self,
        user_input: str,
        confirmation: ConfirmationProvider | None = None,
        session_id: str = "default",
    ) -> str:
        confirmation = confirmation or self.confirmation
        self.system_prompt = agent_prompts.build_system_prompt(self.gigachat_functions)

        messages = self._histories.setdefault(session_id, [])
        if not messages:
            messages.append({"role": "system", "content": self.system_prompt})
        else:
            messages[0]["content"] = self.system_prompt

        self._trim(session_id)                    # обрезает ДО нового хода
        messages.append({"role": "user", "content": user_input})
        self.tracer.log("user_request", content=user_input, session_id=session_id)
        invalid_argument_attempts = 0

        for step in range(1, self.max_steps + 1):
            try:
                response = await self.llm.chat(messages, self.gigachat_functions)
            except Exception as e:
                self.tracer.log("llm_error", error=str(e), session_id=session_id)
                error_msg = f"Ошибка: не удалось получить ответ от модели ({e})"
                messages.append({"role": "assistant", "content": error_msg})
                self._trim(session_id)
                return error_msg

            batches = self._get_call_batches(response)
            if not batches:
                final = self._response_text(response)
                messages.append({"role": "assistant", "content": final})
                self.tracer.log("final_answer", content=final, step=step, session_id=session_id)
                self._trim(session_id)
                return final

            calls = [call for _, batch_calls in batches for call in batch_calls]
            self.tracer.log(
                "llm_response",
                content=self._response_text(response),
                function_calls=[
                    {"id": call.id_, "name": call.name, "arguments": call.arguments}
                    for call in calls
                ],
                tokens=response.usage.total_tokens if getattr(response, "usage", None) else 0,
                session_id=session_id,
            )
            messages.extend(
                message.model_dump(mode="python", by_alias=True, exclude_none=True)
                for message in response.messages
            )

            for assistant_message, batch_calls in batches:
                outcomes = await self._execute_call_batch(batch_calls, confirmation)
                invalid_argument_attempts += sum(invalid for _, invalid, _ in outcomes)
                tool_content = []
                prompt_messages = []
                for call, (result, _, _) in zip(batch_calls, outcomes):
                    parsed_result = json.loads(result)
                    prompt_messages.extend(parsed_result.pop("_prompt_messages", []))
                    tool_content.append({
                        "function_result": {
                            "id": call.id_,
                            "name": call.name,
                            "result": parsed_result,
                        }
                    })
                messages.append({
                    "role": "tool",
                    "tools_state_id": assistant_message.tools_state_id,
                    "content": tool_content,
                })
                messages.extend(prompt_messages)
                if prompt_messages and prompt_messages[-1]["role"] == "assistant":
                    # GigaChat не принимает функции после завершающего сообщения assistant.
                    messages.append({
                        "role": "user",
                        "content": (
                            "Выполни приведённые выше инструкции, "
                            "используя доступные инструменты."
                        ),
                    })

                if any(denied for _, _, denied in outcomes):
                    final = "Операция отменена пользователем. Оставшиеся действия не выполнялись."
                    messages.append({"role": "assistant", "content": final})
                    self.tracer.log("final_answer", content=final, step=step, session_id=session_id)
                    self._trim(session_id)
                    return final

            if invalid_argument_attempts >= 2:
                error_msg = "Модель дважды вернула некорректные аргументы инструмента. Задача не завершена."
                messages.append({"role": "assistant", "content": error_msg})
                self.tracer.log("invalid_arguments_limit_reached", step=step, session_id=session_id)
                self._trim(session_id)
                return error_msg

        self.tracer.log("max_steps_reached", step=self.max_steps, session_id=session_id)
        return await self._finalize_after_limit(messages, session_id)

    async def _finalize_after_limit(self, messages: list[dict], session_id: str) -> str:
        max_msg = "Достигнут лимит шагов. Задача не завершена."
        final_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Лимит шагов исчерпан. Дай частичный ответ только на основе "
                    "уже собранных данных и честно укажи, что задача не завершена. "
                    "Не выдумывай недостающие сведения."
                ),
            },
        ]
        try:
            response = await self.llm.chat(final_messages, functions=None)
            if self._get_call_batches(response):
                raise ValueError("Модель запросила инструмент во время финализации")
            final = self._response_text(response)
            if not final:
                raise ValueError("Модель вернула пустой ответ во время финализации")
            messages.append({"role": "assistant", "content": final})
            self.tracer.log(
                "final_answer", content=final, step=self.max_steps,
                finalization=True, session_id=session_id,
            )
            self._trim(session_id)
            return final
        except Exception as e:
            self.tracer.log("finalization_error", error=str(e), session_id=session_id)
            messages.append({"role": "assistant", "content": max_msg})
            self._trim(session_id)
            return max_msg

    @staticmethod
    def _response_text(response) -> str:
        return "".join(
            part.text or ""
            for message in response.messages
            if message.role == "assistant"
            for part in message.content or []
        ).strip()

    @staticmethod
    def _get_call_batches(response) -> list[tuple[Any, list[Any]]]:
        batches = []
        for message in response.messages:
            if message.role != "assistant":
                continue
            calls = [
                part.function_call
                for part in message.content or []
                if part.function_call is not None
            ]
            if calls:
                batches.append((message, calls))

        calls = [call for _, batch_calls in batches for call in batch_calls]
        if len(calls) > 1:
            ids = [call.id_ for call in calls]
            if any(not call_id for call_id in ids) or len(set(ids)) != len(ids):
                raise ValueError("Параллельные вызовы должны иметь уникальные непустые ID")
        return batches

    def _needs_confirmation(self, function_call) -> bool:
        resolved = self.converter.resolve(function_call.name)
        if not resolved:
            return False
        server, tool_name = resolved
        needs_confirm, _ = self.converter.requires_confirmation(function_call.name)
        return needs_confirm or tool_name in self.always_confirm_tools.get(server, set())

    async def _execute_one(self, function_call, confirmation) -> tuple[str, bool, bool]:
        try:
            result = await self._execute_function_call(function_call, confirmation)
            return result, False, False
        except InvalidToolArgumentsError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False), True, False
        except UserConfirmationDeniedError:
            result = json.dumps(
                {"result": "Пользователь отказался выполнять это действие."},
                ensure_ascii=False,
            )
            return result, False, True

    async def _execute_call_batch(self, calls, confirmation) -> list[tuple[str, bool, bool]]:
        outcomes = []
        index = 0
        while index < len(calls):
            if self._needs_confirmation(calls[index]):
                outcome = await self._execute_one(calls[index], confirmation)
                outcomes.append(outcome)
                if outcome[2]:
                    skipped = json.dumps(
                        {"result": "Действие не выполнено: операция отменена пользователем."},
                        ensure_ascii=False,
                    )
                    outcomes.extend(
                        (skipped, False, False) for _ in calls[index + 1:]
                    )
                    break
                index += 1
                continue

            end = index
            while end < len(calls) and not self._needs_confirmation(calls[end]):
                end += 1
            outcomes.extend(await asyncio.gather(*(
                self._execute_one(call, confirmation)
                for call in calls[index:end]
            )))
            index = end
        return outcomes

    async def _execute_function_call(
        self,
        function_call,
        confirmation: ConfirmationProvider,
    ) -> str:
        prefixed_name = function_call.name
        raw_arguments = function_call.arguments

        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError as e:
            self.tracer.log("invalid_json", tool=prefixed_name, error=str(e))
            raise InvalidToolArgumentsError(f"Невалидный JSON в аргументах: {e}") from e

        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            self.tracer.log("invalid_arguments", tool=prefixed_name, arguments=arguments)
            raise InvalidToolArgumentsError("Аргументы инструмента должны быть объектом JSON")

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
                raise UserConfirmationDeniedError

        self.tracer.log("tool_call", server=server, tool=tool_name, arguments=arguments)
        try:
            if tool_name == "read_resource":
                uri = arguments.get("uri")
                if not isinstance(uri, str) or not uri.strip():
                    raise InvalidToolArgumentsError("Для read_resource нужен непустой uri")
                result = await self.mcp.read_resource(server, uri)
                text = self._truncate_tool_result(self._extract_resource_text(result))
            elif tool_name.startswith("get_prompt_"):
                prompt_name = tool_name.removeprefix("get_prompt_")
                result = await self.mcp.get_prompt(
                    server,
                    prompt_name,
                    {key: str(value) for key, value in arguments.items()},
                )
                prompt_messages = [
                    {
                        "role": message.role,
                        "content": getattr(message.content, "text", str(message.content)),
                    }
                    for message in result.messages
                ]
                self.tracer.log(
                    "tool_result", server=server, tool=tool_name, result="prompt applied"
                )
                return json.dumps({
                    "result": "MCP-промпт применён.",
                    "_prompt_messages": prompt_messages,
                }, ensure_ascii=False)
            else:
                result = await self.mcp.call_tool(server, tool_name, arguments)
                text = self._truncate_tool_result(self._extract_text(result))

            is_error = getattr(result, "is_error", getattr(result, "isError", False))
            self.tracer.log(
                "tool_result", server=server, tool=tool_name,
                result=text[:500], is_error=is_error,
            )
            return json.dumps({"result": wrap_tool_result(server, tool_name, text)}, ensure_ascii=False)
        except InvalidToolArgumentsError:
            raise
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

    def _truncate_tool_result(self, text: str) -> str:
        if len(text) <= self.max_tool_result_chars:
            return text
        return (
            f"{text[:self.max_tool_result_chars]}\n"
            f"[Результат сокращён: показаны первые {self.max_tool_result_chars} "
            f"из {len(text)} символов. Уточните запрос или уменьшите limit при поиске.]"
        )

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

    @staticmethod
    def _extract_resource_text(result) -> str:
        texts = [
            content.text
            for content in result.contents
            if hasattr(content, "text")
        ]
        if not texts:
            raise ValueError("Ресурс не содержит текстового представления")
        return "\n".join(texts)