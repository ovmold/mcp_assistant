import asyncio

import pytest

from mcp_assistant.client.confirmation import CLIConfirmation, GradioConfirmation


@pytest.mark.anyio
async def test_cli_confirm_delegates_without_reason(monkeypatch) -> None:
    """Проверяет делегирование CLI confirm() в confirm_with_reason() с пустой причиной."""
    confirmation = CLIConfirmation()
    captured = {}

    async def confirm_with_reason(tool_name, arguments, reason="") -> bool:
        captured.update(tool=tool_name, arguments=arguments, reason=reason)
        return True

    monkeypatch.setattr(confirmation, "confirm_with_reason", confirm_with_reason)

    assert await confirmation.confirm("update_status", {"ticket_id": 1})
    assert captured == {
        "tool": "update_status",
        "arguments": {"ticket_id": 1},
        "reason": "",
    }


@pytest.mark.anyio
async def test_gradio_confirmation_request_resolve_and_cleanup() -> None:
    """Проверяет создание, подтверждение и очистку Gradio-запроса подтверждения."""
    confirmation = GradioConfirmation(timeout=1)
    request_task = asyncio.create_task(
        confirmation.confirm_with_reason(
            "update_status",
            {"ticket_id": 1},
            "destructiveHint=True",
        )
    )

    await asyncio.wait_for(confirmation.wait_requested(), timeout=1)
    assert confirmation.pending == ("update_status", {"ticket_id": 1})
    assert confirmation.pending_reason == "destructiveHint=True"

    confirmation.resolve(True)

    assert await request_task is True
    assert confirmation.pending is None
    assert confirmation.pending_reason == ""
    confirmation.resolve(False)


@pytest.mark.anyio
async def test_gradio_confirmation_timeout_denies_and_cleans_state() -> None:
    """Проверяет отказ по таймауту и очистку состояния GradioConfirmation."""
    confirmation = GradioConfirmation(timeout=0.01)

    approved = await confirmation.confirm("add_comment", {"ticket_id": 1})

    assert approved is False
    assert confirmation.pending is None
    assert confirmation.pending_reason == ""


@pytest.mark.anyio
async def test_gradio_confirmation_cancellation_cleans_state() -> None:
    """Проверяет очистку состояния GradioConfirmation после отмены задачи."""
    confirmation = GradioConfirmation(timeout=1)
    request_task = asyncio.create_task(confirmation.confirm("update_status", {}))
    await asyncio.wait_for(confirmation.wait_requested(), timeout=1)

    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert confirmation.pending is None
    assert confirmation.pending_reason == ""
