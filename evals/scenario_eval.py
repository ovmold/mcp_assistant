import sqlite3
import time
from collections import Counter

from mcp_assistant import config as conf


class EvalTracer:
    def __init__(self):
        self.events = []

    def log(self, event_type: str, **data):
        self.events.append((event_type, data))


class EvalConfirmation:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.index = 0

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        if self.index >= len(self.responses):
            return False
        response = str(self.responses[self.index]).lower()
        self.index += 1
        return response in ("да", "yes", "true", "1")


def validate_scenarios(scenarios: list[dict]) -> None:
    if not isinstance(scenarios, list):
        raise ValueError("Файл evals должен содержать список сценариев")

    ids = set()
    for scenario in scenarios:
        scenario_id = scenario.get("id")
        if not scenario_id or not isinstance(scenario_id, str):
            raise ValueError("У каждого сценария должен быть строковый id")
        if scenario_id != scenario_id.strip() or any(c.isspace() for c in scenario_id):
            raise ValueError(f"ID сценария не должен содержать пробелы: {scenario_id!r}")
        if scenario_id in ids:
            raise ValueError(f"Дублирующийся ID сценария: {scenario_id}")
        ids.add(scenario_id)

        if scenario.get("category") not in {"read", "write", "safety", "robustness"}:
            raise ValueError(f"{scenario_id}: неизвестная category")
        if scenario.get("severity") not in {"critical", "major", "minor"}:
            raise ValueError(f"{scenario_id}: неизвестная severity")
        turns = scenario.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{scenario_id}: нужен непустой список turns")
        for turn in turns:
            if not isinstance(turn.get("user"), str) or not turn["user"].strip():
                raise ValueError(f"{scenario_id}: каждый ход должен содержать user")


def apply_setup(scenario: dict) -> None:
    setup = scenario.get("setup", {})
    tickets = setup.get("tickets", {})
    if not tickets:
        return

    allowed_fields = {
        "subject", "body", "status", "priority", "client", "created_at",
        "updated_at", "tags", "issue_type",
    }
    with sqlite3.connect(conf.DB_PATH) as conn:
        for ticket_id, fields in tickets.items():
            unknown = set(fields) - allowed_fields
            if unknown:
                raise ValueError(f"Неизвестные поля setup.tickets: {sorted(unknown)}")
            assignments = ", ".join(f"{field} = ?" for field in fields)
            conn.execute(
                f"UPDATE tickets SET {assignments} WHERE id = ?",
                [*fields.values(), ticket_id],
            )
        conn.commit()


def _snapshot_db() -> dict:
    with sqlite3.connect(conf.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        tickets = {
            row["id"]: dict(row)
            for row in conn.execute("SELECT * FROM tickets")
        }
        max_message_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages"
        ).fetchone()[0]
    return {"tickets": tickets, "max_message_id": max_message_id}


def _args_match(expected: dict, actual: dict) -> bool:
    for key, value in expected.items():
        if key not in actual:
            return False
        if isinstance(value, dict):
            if not isinstance(actual[key], dict) or not _args_match(value, actual[key]):
                return False
        elif actual[key] != value:
            return False
    return True


def _collect_calls(tracer: EvalTracer) -> list[dict]:
    calls = [
        {
            "tool": data["tool"],
            "args": data.get("arguments", {}),
            "succeeded": None,
        }
        for event, data in tracer.events
        if event == "tool_call"
    ]

    def mark_result(data, succeeded):
        for call in calls:
            if call["tool"] == data["tool"] and call["succeeded"] is None:
                call["succeeded"] = succeeded
                return

    for event, data in tracer.events:
        if event == "tool_result":
            mark_result(data, not data.get("is_error", False))
        elif event == "tool_error":
            mark_result(data, False)
    return calls


def _evaluate_calls(expect: dict, actual_calls: list[dict]) -> list[str]:
    reasons = []
    for expected in expect.get("calls", []):
        tool = expected["tool"]
        matching = [
            call for call in actual_calls
            if call["tool"] == tool and _args_match(expected.get("args", {}), call["args"])
        ]
        if not matching:
            reasons.append(f"Не найден вызов {tool} с аргументами {expected.get('args', {})}")
            continue
        if "succeeded" in expected and not any(
            call["succeeded"] is expected["succeeded"] for call in matching
        ):
            reasons.append(
                f"Вызов {tool} не имеет succeeded={expected['succeeded']}"
            )

    for tool in expect.get("forbidden_tools", []):
        if any(call["tool"] == tool for call in actual_calls):
            reasons.append(f"Выполнен запрещённый инструмент: {tool}")
    return reasons


def _evaluate_batches(expect: dict, tracer: EvalTracer) -> list[str]:
    actual_batches = [
        data.get("function_calls", [])
        for event, data in tracer.events
        if event == "llm_response" and data.get("function_calls")
    ]
    reasons = []
    position = 0
    for expected in expect.get("batches", []):
        expected_tools = Counter(expected["tools"])
        found = False
        while position < len(actual_batches):
            batch = actual_batches[position]
            position += 1
            actual_tools = Counter(call["name"].split("__", 1)[-1] for call in batch)
            if all(actual_tools[tool] >= count for tool, count in expected_tools.items()):
                if not expected.get("parallel", False) or len(batch) > 1:
                    found = True
                    break
        if not found:
            reasons.append(f"Не найден пакет вызовов: {expected['tools']}")
    return reasons


def _evaluate_confirmations(expect: dict, tracer: EvalTracer) -> list[str]:
    requested = [
        data["tool"]
        for event, data in tracer.events
        if event == "confirmation_requested"
    ]
    expected = expect.get("confirmations", {})
    reasons = []
    if "count" in expected and len(requested) != expected["count"]:
        reasons.append(
            f"Запросов подтверждения: {len(requested)}, ожидалось {expected['count']}"
        )
    for tool in expected.get("tools", []):
        if tool not in requested:
            reasons.append(f"Не запрошено подтверждение для {tool}")
    return reasons


def _evaluate_answer(expect: dict, answer: str) -> list[str]:
    rules = expect.get("answer", {})
    text = answer.casefold()
    reasons = []
    for value in rules.get("all", []):
        if value.casefold() not in text:
            reasons.append(f"В ответе нет обязательного текста: {value}")
    if rules.get("any") and not any(value.casefold() in text for value in rules["any"]):
        reasons.append(f"В ответе нет ни одной строки: {rules['any']}")
    for value in rules.get("none", []):
        if value.casefold() in text:
            reasons.append(f"В ответе есть запрещённый текст: {value}")
    return reasons


def _evaluate_db(expect: dict, baseline: dict) -> list[str]:
    expected = expect.get("db", {})
    if not expected:
        return []

    reasons = []
    with sqlite3.connect(conf.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for ticket_id, fields in expected.get("tickets", {}).items():
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            if row is None:
                reasons.append(f"Тикет {ticket_id} отсутствует в БД")
                continue
            for field, value in fields.items():
                if row[field] != value:
                    reasons.append(
                        f"Тикет {ticket_id}: {field}={row[field]!r}, ожидалось {value!r}"
                    )

        for ticket_id in expected.get("unchanged_tickets", []):
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            if row is None or dict(row) != baseline["tickets"].get(ticket_id):
                reasons.append(f"Тикет {ticket_id} изменился, хотя должен остаться прежним")

        for ticket_id, rules in expected.get("new_messages", {}).items():
            rows = conn.execute(
                "SELECT * FROM messages WHERE ticket_id = ? AND id > ? ORDER BY id",
                (ticket_id, baseline["max_message_id"]),
            ).fetchall()
            if "count" in rules and len(rows) != rules["count"]:
                reasons.append(
                    f"Тикет {ticket_id}: новых сообщений {len(rows)}, ожидалось {rules['count']}"
                )
            if "contains" in rules:
                expected_text = rules["contains"]
                texts = [row["text"] for row in rows]
                values = expected_text if isinstance(expected_text, list) else [expected_text]
                for value in values:
                    if not any(value in text for text in texts):
                        reasons.append(f"В новых сообщениях тикета {ticket_id} нет {value!r}")
            if "author" in rules and any(row["author"] != rules["author"] for row in rows):
                reasons.append(f"Новые сообщения тикета {ticket_id} имеют другого автора")
    return reasons


async def run_scenario(agent, scenario: dict) -> dict:
    tracer = EvalTracer()
    agent.tracer = tracer
    baseline = _snapshot_db()
    final_answer = ""
    errors = []

    start_time = time.perf_counter()
    for turn in scenario["turns"]:
        agent.confirmation = EvalConfirmation(turn.get("confirm", []))
        try:
            final_answer = await agent.run(turn["user"])
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            break
    latency = time.perf_counter() - start_time

    expect = scenario.get("expect", {})
    actual_calls = _collect_calls(tracer)
    reasons = [*errors]
    reasons.extend(_evaluate_calls(expect, actual_calls))
    reasons.extend(_evaluate_batches(expect, tracer))
    reasons.extend(_evaluate_confirmations(expect, tracer))
    reasons.extend(_evaluate_answer(expect, final_answer))
    reasons.extend(_evaluate_db(expect, baseline))

    if "max_steps" in expect:
        steps = max((data.get("step", 0) for _, data in tracer.events), default=0)
        if steps > expect["max_steps"]:
            reasons.append(f"Шагов {steps}, допустимо {expect['max_steps']}")
    else:
        steps = max((data.get("step", 0) for _, data in tracer.events), default=0)

    tokens = sum(
        data.get("tokens", 0)
        for event, data in tracer.events
        if event == "llm_response"
    )
    denied = [
        data["tool"]
        for event, data in tracer.events
        if event == "confirmation_denied"
    ]

    return {
        "success": not reasons,
        "latency": latency,
        "steps": steps,
        "tokens": tokens,
        "reasons": reasons,
        "final_answer": final_answer,
        "tool_calls": [call["tool"] for call in actual_calls],
        "confirmation_asked": [
            data["tool"]
            for event, data in tracer.events
            if event == "confirmation_requested"
        ],
        "confirmation_denied": denied,
        "events": tracer.events,
        "missing_tools": [],
        "forbidden_called": [
            tool for tool in expect.get("forbidden_tools", [])
            if tool in [call["tool"] for call in actual_calls]
        ],
    }
