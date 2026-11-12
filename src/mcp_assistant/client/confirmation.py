"""Способ спросить у пользователя подтверждение действия."""

import json
import asyncio
from typing import Protocol


class ConfirmationProvider(Protocol):
    """Интерфейс для всех механизмов подтверждения действий"""
    async def confirm(self, tool_name: str, arguments: dict) -> bool: ...


# --- реализация для CLI ---

class CLIConfirmation:
    """Спрашивает через stdin. Работает в фоновом потоке, чтобы не блокировать loop."""

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        return await self.confirm_with_reason(tool_name, arguments, "")

    async def confirm_with_reason(
        self, tool_name: str, arguments: dict, reason: str = "",
    ) -> bool:
        extra = f"\n   Причина: {reason}" if reason else ""
        prompt = (
            "\nЗапрос на подтверждение действия\n"
            f"Агент хочет выполнить: {tool_name}{extra}\n"
            f"   Аргументы: {json.dumps(arguments, ensure_ascii=False, indent=2)}\n"
            "Подтвердить? (да/нет): "
        )
        answer = await asyncio.to_thread(input, prompt)
        return answer.strip().lower() in {"да", "yes", "y", "д"}


# --- реализация для Gradio ---

class GradioConfirmation:
    """Подтверждение приходит следующим сообщением пользователя в чате."""

    def __init__(self, timeout: float = 600.0) -> None:
        self._future: asyncio.Future[bool] | None = None
        self._request: tuple[str, dict] | None = None
        self._requested = asyncio.Event()
        self._timeout = timeout
        self._reason: str = "" 

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        return await self.confirm_with_reason(tool_name, arguments, "")

    async def confirm_with_reason(
        self, tool_name: str, arguments: dict, reason: str = "",
    ) -> bool:
        loop = asyncio.get_running_loop()
        self._future = loop.create_future()
        self._request = (tool_name, arguments)
        self._reason = reason                         
        self._requested.set()
        try:
            return await asyncio.wait_for(self._future, timeout=self._timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._request = None
            self._reason = ""                         
            self._future = None
            self._requested.clear()

    def resolve(self, approved: bool) -> None:
        if self._future and not self._future.done():
            self._future.set_result(approved)

    @property
    def pending(self) -> tuple[str, dict] | None:
        return self._request

    @property
    def pending_reason(self) -> str:
        """Причина запроса — доступна, пока pending не None."""
        return self._reason

    async def wait_requested(self) -> None:
        await self._requested.wait()