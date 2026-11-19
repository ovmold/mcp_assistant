import sqlite3
from pathlib import Path

import pytest
import yaml

from evals.conf_evals import configure_eval_tickets_server, init_eval_db
from evals.scenario_eval import apply_setup, run_scenario, validate_scenarios
from mcp_assistant import config as conf


def test_init_eval_db_creates_parent_directory(tmp_path, monkeypatch) -> None:
    """Проверяет создание каталога и заполнение временной eval-БД."""
    db_path = tmp_path / "nested" / "evals.db"
    monkeypatch.setattr(conf, "DB_PATH", db_path)

    init_eval_db()

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    assert count == 10


def test_all_scenarios_use_valid_new_format() -> None:
    """Проверяет валидность и уникальность ID всех YAML-сценариев."""
    path = Path(__file__).resolve().parent.parent / "evals" / "scenarios.yaml"
    scenarios = yaml.safe_load(path.read_text(encoding="utf-8"))

    validate_scenarios(scenarios)

    assert len(scenarios) == 29
    assert len({scenario["id"] for scenario in scenarios}) == len(scenarios)


def test_scenario_id_cannot_contain_spaces() -> None:
    """Проверяет отклонение ID сценария с пробелами."""
    with pytest.raises(ValueError, match="пробелы"):
        validate_scenarios([{
            "id": "invalid id",
            "category": "read",
            "severity": "major",
            "turns": [{"user": "test"}],
        }])


@pytest.mark.anyio
async def test_multi_turn_scenario_checks_calls_and_db(tmp_path, monkeypatch) -> None:
    """Проверяет многоходовой сценарий, вызовы, подтверждение и новый комментарий в БД."""
    db_path = tmp_path / "evals.db"
    monkeypatch.setattr(conf, "DB_PATH", db_path)
    init_eval_db()

    class FakeAgent:
        def __init__(self) -> None:
            self.tracer = None
            self.confirmation = None
            self.turns = []

        async def run(self, user_input) -> str:
            self.turns.append(user_input)
            if len(self.turns) == 1:
                return "От чьего имени добавить комментарий?"

            arguments = {"ticket_id": 1, "text": "Проверено", "author": "support"}
            approved = await self.confirmation.confirm("add_comment", arguments)
            assert approved
            self.tracer.log("confirmation_requested", tool="add_comment")
            self.tracer.log("tool_call", tool="add_comment", arguments=arguments)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "INSERT INTO messages (ticket_id, author, text, created_at) "
                    "VALUES (1, 'support', 'Проверено', '2026-10-17T12:00:00')"
                )
                comment_id = cursor.lastrowid
                conn.commit()
            self.tracer.log(
                "tool_result",
                tool="add_comment",
                result={"comment_id": comment_id},
                is_error=False,
            )
            return "Комментарий добавлен"

    scenario = {
        "id": "multi_turn",
        "category": "write",
        "severity": "critical",
        "turns": [
            {"user": "Добавь комментарий"},
            {"user": "support", "confirm": ["да"]},
        ],
        "expect": {
            "calls": [{
                "tool": "add_comment",
                "args": {"ticket_id": 1},
                "succeeded": True,
            }],
            "confirmations": {"count": 1, "tools": ["add_comment"]},
            "answer": {"any": ["добавлен"]},
            "db": {
                "new_messages": {
                    1: {"count": 1, "contains": "Проверено", "author": "support"}
                }
            },
        },
    }

    result = await run_scenario(FakeAgent(), scenario)

    assert result["success"], result["reasons"]
    assert result["tool_calls"] == ["add_comment"]
    assert result["call_details"] == [{
        "tool": "add_comment",
        "args": {"ticket_id": 1, "text": "Проверено", "author": "support"},
        "succeeded": True,
        "result": {"comment_id": 1},
        "error": None,
    }]


@pytest.mark.anyio
async def test_scenario_releases_db_for_next_run(tmp_path, monkeypatch) -> None:
    """Проверяет закрытие БД между повторными прогонами сценария."""
    db_path = tmp_path / "evals.db"
    monkeypatch.setattr(conf, "DB_PATH", db_path)

    class FakeAgent:
        tracer = None
        confirmation = None

        async def run(self, user_input) -> str:
            return "Готово"

    scenario = {
        "id": "repeated_run",
        "category": "read",
        "severity": "major",
        "setup": {"tickets": {1: {"priority": "low"}}},
        "turns": [{"user": "Проверь тикет"}],
        "expect": {"db": {"unchanged_tickets": [1, 2, 3]}},
    }

    for _ in range(3):
        init_eval_db()
        apply_setup(scenario)
        result = await run_scenario(FakeAgent(), scenario)
        assert result["success"], result["reasons"]


def test_configure_eval_tickets_server_passes_only_eval_paths(tmp_path, monkeypatch) -> None:
    """Проверяет передачу временных DATA_DIR и DB_PATH только stdio-серверу tickets."""
    data_dir = tmp_path / "eval_data"
    db_path = data_dir / "tickets.db"
    monkeypatch.setattr(conf, "DATA_DIR", data_dir)
    monkeypatch.setattr(conf, "DB_PATH", db_path)
    config = {
        "servers": [
            {
                "name": "tickets",
                "transport": {
                    "type": "stdio",
                    "env": {"PYTHONPATH": "src"},
                },
            },
            {
                "name": "knowledge",
                "transport": {"type": "streamable-http"},
            },
        ]
    }

    configure_eval_tickets_server(config)

    assert config["servers"][0]["transport"]["env"] == {
        "PYTHONPATH": "src",
        "DATA_DIR": str(data_dir),
        "DB_PATH": str(db_path),
    }
    assert "env" not in config["servers"][1]["transport"]


@pytest.mark.parametrize(
    ("ticket_tool", "ticket_args"),
    [
        ("get_ticket", {"ticket_id": 1}),
        ("read_resource", {"uri": "ticket://1"}),
    ],
)
@pytest.mark.anyio
async def test_parallel_ticket_and_kb_accepts_both_read_methods(
    tmp_path, monkeypatch, ticket_tool, ticket_args
) -> None:
    """Проверяет, что сценарий принимает get_ticket или read_resource в параллельном пакете."""
    monkeypatch.setattr(conf, "DB_PATH", tmp_path / "evals.db")
    init_eval_db()

    class FakeAgent:
        tracer = None
        confirmation = None

        async def run(self, user_input) -> str:
            calls = [
                (ticket_tool, ticket_args, "tickets"),
                ("search_kb", {"query": "502"}, "knowledge"),
            ]
            self.tracer.log(
                "llm_response",
                function_calls=[
                    {
                        "name": f"{server}__{tool}",
                        "arguments": arguments,
                    }
                    for tool, arguments, server in calls
                ],
            )
            for tool, arguments, _ in calls:
                self.tracer.log("tool_call", tool=tool, arguments=arguments)
                self.tracer.log("tool_result", tool=tool, is_error=False)
            return "Тикет 1 и статья про 502"

    path = Path(__file__).resolve().parent.parent / "evals" / "scenarios.yaml"
    scenarios = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenario = next(s for s in scenarios if s["id"] == "parallel_ticket_and_kb")

    result = await run_scenario(FakeAgent(), scenario)

    assert result["success"], result["reasons"]



@pytest.mark.anyio
async def test_scenario_metrics_distinguish_check_types(tmp_path, monkeypatch) -> None:
    """Проверяет независимые результаты проверок вызовов, подтверждений, ответа, БД и токенов."""
    monkeypatch.setattr(conf, "DB_PATH", tmp_path / "evals.db")
    init_eval_db()

    class FakeAgent:
        tracer = None
        confirmation = None

        async def run(self, user_input) -> str:
            self.tracer.log("llm_response", tokens=7)
            self.tracer.log("llm_response", tokens=5)
            self.tracer.log("tool_call", tool="forbidden_tool", arguments={})
            self.tracer.log("tool_result", tool="forbidden_tool", is_error=True)
            return "неожиданный ответ"

    scenario = {
        "id": "separate_metrics",
        "category": "write",
        "severity": "critical",
        "turns": [{"user": "Выполни действие"}],
        "expect": {
            "calls": [{"tool": "required_tool", "succeeded": True}],
            "forbidden_tools": ["forbidden_tool"],
            "confirmations": {"count": 1},
            "answer": {"all": ["ожидаемый текст"]},
            "db": {"unchanged_tickets": [1]},
        },
    }

    result = await run_scenario(FakeAgent(), scenario)

    assert not result["success"]
    assert result["category"] == "write"
    assert result["severity"] == "critical"
    assert not result["required_calls_ok"]
    assert result["forbidden_called"] == ["forbidden_tool"]
    assert not result["confirmations_ok"]
    assert not result["answer_ok"]
    assert result["db_ok"]
    assert result["successful_tool_calls"] == 0
    assert result["total_tool_calls"] == 1
    assert result["tokens"] == 12



def test_print_results_includes_detailed_and_grouped_metrics(monkeypatch, capsys) -> None:
    """Проверяет сводные метрики, стабильность и разрезы по category и severity."""
    for name in ("DATA_DIR", "TRACE_DIR", "DB_PATH"):
        monkeypatch.setenv(name, "unused-in-report-test")

    from evals.run_evals import print_results

    def run(success, category, severity, **overrides):
        result = {
            "success": success,
            "category": category,
            "severity": severity,
            "latency": 1.0,
            "steps": 1,
            "tokens": 10,
            "required_calls_ok": True,
            "confirmations_ok": True,
            "answer_ok": True,
            "db_ok": True,
            "successful_tool_calls": 0,
            "total_tool_calls": 0,
            "forbidden_called": [],
        }
        result.update(overrides)
        return result

    results = {
        "stable_success": [
            run(True, "read", "major", successful_tool_calls=2, total_tool_calls=2),
            run(True, "read", "major", successful_tool_calls=2, total_tool_calls=2),
        ],
        "mixed": [
            run(True, "write", "critical", successful_tool_calls=1, total_tool_calls=1),
            run(
                False,
                "write",
                "critical",
                required_calls_ok=False,
                confirmations_ok=False,
                answer_ok=False,
                db_ok=False,
                successful_tool_calls=0,
                total_tool_calls=1,
                forbidden_called=["bad_tool"],
            ),
        ],
        "stable_failure": [run(False, "safety", "minor")],
    }

    print_results(results, 3)
    output = capsys.readouterr().out

    assert "Доля успешных прогонов (Run Success Rate): 60.0%" in output
    assert "Доля успешных сценариев (все прогоны успешны): 33.3% (1/3)" in output
    assert "Правильные обязательные вызовы: 80.0% (4/5)" in output
    assert "Прогоны без запрещённых вызовов: 80.0% (4/5)" in output
    assert "Правильные подтверждения: 80.0% (4/5)" in output
    assert "Корректный текст ответа: 80.0% (4/5)" in output
    assert "Корректное состояние БД: 80.0% (4/5)" in output
    assert "Доля успешных вызовов инструментов: 83.3% (5/6)" in output
    assert "Стабильный успех (2/2)" in output
    assert "Стабильный провал (0/1)" in output
    assert "Нестабильно (1/2)" in output
    assert "Доля успешных прогонов по категориям" in output
    assert "Доля успешных прогонов по критичности" in output
    assert "чтение (read): 100.0% (2/2)" in output
    assert "запись (write): 50.0% (1/2)" in output
    assert "безопасность (safety): 0.0% (0/1)" in output
    assert "устойчивость (robustness)" not in output
    assert "критические (critical): 50.0% (1/2)" in output
    assert "значительные (major): 100.0% (2/2)" in output
    assert "незначительные (minor): 0.0% (0/1)" in output
    assert "Среднее число шагов агентного цикла" in output
    assert "Средняя латентность выполнения сценария" in output
