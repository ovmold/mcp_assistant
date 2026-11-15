"""LLM-провайдеры и преобразование истории агента для OpenAI."""

import asyncio
import json
import os
from typing import Protocol

from gigachat import GigaChat
from gigachat.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatContentPart,
    ChatFunctionsTool,
    ChatMessage,
    ChatModelOptions,
    ChatTool,
    ChatUsage,
    PrimaryChatFunctionCall,
)
from openai import OpenAI


class ChatLLM(Protocol):
    async def chat(
        self,
        messages: list[dict],
        functions: list[dict] | None = None,
    ) -> ChatCompletionResponse: ...

    def close(self) -> None: ...


class GigaChatLLMClient:
    """Обертка над основным chat-completions контрактом GigaChat SDK."""

    def __init__(self, giga_instance, model) -> None:
        self.giga = giga_instance
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        functions: list[dict] | None = None,
    ) -> ChatCompletionResponse:
        """Отправляет запрос в GigaChat в отдельном потоке."""

        def _sync_chat() -> ChatCompletionResponse:
            tools = None
            if functions:
                tools = [ChatTool(
                    functions=ChatFunctionsTool(specifications=functions)
                )]
            request = ChatCompletionRequest(
                model=self.model,
                messages=[ChatMessage.model_validate(message) for message in messages],
                model_options=ChatModelOptions(parallel_tool_calls=True),
                tools=tools,
            )
            return self.giga.chat.create(request)

        return await asyncio.to_thread(_sync_chat)

    def close(self) -> None:
        self.giga.close()


class OpenAILLMClient:
    """Адаптер OpenAI Chat Completions к внутреннему контракту агента."""

    def __init__(self, openai_instance: OpenAI, model: str) -> None:
        self.client = openai_instance
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        functions: list[dict] | None = None,
    ) -> ChatCompletionResponse:
        def _sync_chat() -> ChatCompletionResponse:
            request = {
                "model": self.model,
                "messages": self._to_openai_messages(messages),
            }
            if functions:
                request["tools"] = [
                    {"type": "function", "function": function}
                    for function in functions
                ]
                request["parallel_tool_calls"] = True
            response = self.client.chat.completions.create(**request)
            return self._from_openai_response(response)

        return await asyncio.to_thread(_sync_chat)

    def close(self) -> None:
        self.client.close()

    @classmethod
    def _to_openai_messages(cls, messages: list[dict]) -> list[dict]:
        """Преобразует внутреннюю историю агента в сообщения OpenAI."""
        converted = []
        for message in messages:
            role = message["role"]
            if role == "assistant":
                converted.append(cls._assistant_to_openai(message))
            elif role == "tool":
                converted.extend(cls._tool_results_to_openai(message))
            else:
                converted.append({"role": role, "content": message.get("content")})
        return converted

    @staticmethod
    def _assistant_to_openai(message: dict) -> dict:
        """Преобразует текст и вызовы функций ассистента в формат OpenAI."""
        content = message.get("content")
        if not isinstance(content, list):
            return {"role": "assistant", "content": content}

        text = "".join(part.get("text") or "" for part in content)
        tool_calls = []
        for part in content:
            call = part.get("function_call")
            if call is None:
                continue
            arguments = call.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append({
                "id": call.get("id", call.get("id_")),
                "type": "function",
                "function": {"name": call["name"], "arguments": arguments},
            })

        converted = {"role": "assistant", "content": text or None}
        if tool_calls:
            converted["tool_calls"] = tool_calls
        return converted

    @staticmethod
    def _tool_results_to_openai(message: dict) -> list[dict]:
        """Разворачивает результаты вызовов в отдельные tool-сообщения OpenAI."""
        converted = []
        for part in message.get("content") or []:
            result = part["function_result"]
            converted.append({
                "role": "tool",
                "tool_call_id": result.get("id", result.get("id_")),
                "content": json.dumps(result.get("result"), ensure_ascii=False),
            })
        return converted

    @staticmethod
    def _from_openai_response(response) -> ChatCompletionResponse:
        """Преобразует ответ OpenAI SDK во внутренний ChatCompletionResponse."""
        messages = []
        for choice in response.choices:
            source = choice.message
            content = []
            if source.content:
                content.append(ChatContentPart(text=source.content))
            for call in source.tool_calls or []:
                content.append(ChatContentPart(
                    function_call=PrimaryChatFunctionCall(
                        id_=call.id,
                        name=call.function.name,
                        arguments=call.function.arguments,
                    )
                ))
            messages.append(ChatMessage(
                role="assistant",
                content=content or None,
                tools_state_id=response.id if source.tool_calls else None,
            ))

        usage = None
        if response.usage is not None:
            usage = ChatUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return ChatCompletionResponse(
            model=response.model,
            created_at=response.created,
            messages=messages,
            usage=usage,
        )


def create_llm(timeout: float) -> ChatLLM:
    """Создает выбранный через LLM_PROVIDER клиент; по умолчанию GigaChat."""
    provider = os.environ.get("LLM_PROVIDER", "gigachat").strip().lower()
    model = os.environ.get("MODEL")
    if not model:
        raise ValueError("Переменная окружения MODEL не задана")

    if provider == "gigachat":
        auth_key = os.environ.get("GIGACHAT_AUTH_KEY")
        if not auth_key:
            raise ValueError("Переменная окружения GIGACHAT_AUTH_KEY не задана")
        api_base = os.environ.get("GIGACHAT_BASE_URL")
        if not api_base:
            raise ValueError("Переменная окружения GIGACHAT_BASE_URL не задана")
        giga = GigaChat(
            base_url=api_base,
            model=model,
            credentials=auth_key,
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False,
            timeout=timeout,
        )
        return GigaChatLLMClient(giga_instance=giga, model=model)

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Переменная окружения OPENAI_API_KEY не задана")
        options = {"api_key": api_key, "timeout": timeout}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            options["base_url"] = base_url
        return OpenAILLMClient(OpenAI(**options), model=model)

    raise ValueError(
        f"Неизвестный LLM_PROVIDER: {provider!r}. Доступны: gigachat, openai"
    )
