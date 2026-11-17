import builtins

import pytest

from mcp_assistant.client.ui.cli import run_cli


class FakeAgent:
    def __init__(self, fail_on=None) -> None:
        self.calls = []
        self.reset_count = 0
        self.fail_on = fail_on

    def reset(self) -> None:
        self.reset_count += 1

    async def run(self, message) -> str:
        self.calls.append(message)
        if message == self.fail_on:
            raise RuntimeError("agent failed")
        return f"Ответ на {message}"


def install_inputs(monkeypatch, values) -> None:
    inputs = iter(values)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))


@pytest.mark.anyio
async def test_cli_handles_empty_input_reset_and_exit(monkeypatch, capsys) -> None:
    agent = FakeAgent()
    install_inputs(monkeypatch, ["", "/reset", "/clear", "вопрос", "/exit"])

    await run_cli(agent)

    output = capsys.readouterr().out
    assert agent.reset_count == 2
    assert agent.calls == ["вопрос"]
    assert output.count("История очищена") == 2
    assert "Агент: Ответ на вопрос" in output
    assert "Выход." in output


@pytest.mark.anyio
async def test_cli_reports_agent_error_and_quits(monkeypatch, capsys) -> None:
    agent = FakeAgent(fail_on="ошибка")
    install_inputs(monkeypatch, ["ошибка", "/quit"])

    await run_cli(agent)

    output = capsys.readouterr().out
    assert "Ошибка: agent failed" in output
    assert "Выход." in output


@pytest.mark.anyio
@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
async def test_cli_exits_on_input_interruption(monkeypatch, capsys, exception) -> None:
    def interrupted_input(prompt=""):
        raise exception

    monkeypatch.setattr(builtins, "input", interrupted_input)

    await run_cli(FakeAgent())

    assert "Выход." in capsys.readouterr().out
