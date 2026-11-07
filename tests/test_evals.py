import sqlite3
from pathlib import Path

import pytest
import yaml

from evals.conf_evals import init_eval_db
from evals.scenario_eval import run_scenario, validate_scenarios
from mcp_assistant import config as conf


def test_init_eval_db_creates_parent_directory(tmp_path, monkeypatch):
    db_path = tmp_path / "nested" / "evals.db"
    monkeypatch.setattr(conf, "DB_PATH", db_path)

    init_eval_db()

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    assert count == 10


def test_all_scenarios_use_valid_new_format():
    path = Path(__file__).resolve().parent.parent / "evals" / "scenarios.yaml"
    scenarios = yaml.safe_load(path.read_text(encoding="utf-8"))

    validate_scenarios(scenarios)

    assert len(scenarios) == 29
    assert len({scenario["id"] for scenario in scenarios}) == len(scenarios)


def test_scenario_id_cannot_contain_spaces():
    with pytest.raises(ValueError, match="пробелы"):
        validate_scenarios([{
            "id": "invalid id",
            "category": "read",
            "severity": "major",
            "turns": [{"user": "test"}],
        }])


@pytest.mark.anyio
async def test_multi_turn_scenario_checks_calls_and_db(tmp_path, monkeypatch):
    db_path = tmp_path / "evals.db"
    monkeypatch.setattr(conf, "DB_PATH", db_path)
    init_eval_db()

    class FakeAgent:
        def __init__(self):
            self.tracer = None
            self.confirmation = None
            self.turns = []

        async def run(self, user_input):
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
