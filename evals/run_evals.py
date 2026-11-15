import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

temp_eval_dir = os.path.join(tempfile.gettempdir(), "mcp_evals")
data_dir = os.path.join(temp_eval_dir, "data")
os.environ["DATA_DIR"] = data_dir
os.environ["TRACE_DIR"] = os.path.join(temp_eval_dir, "traces")
os.environ["DB_PATH"] = os.path.join(data_dir, "tickets.db")

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp_assistant import config as conf
from mcp_assistant.client.agent import Agent
from mcp_assistant.client.llm import create_llm
from mcp_assistant.client.config_mcp import load_config
from mcp_assistant.client.mcp_client import MCPClient
from evals.conf_evals import (
    SCENARIOS_FILE,
    configure_eval_tickets_server,
    init_eval_db,
)
from evals.scenario_eval import apply_setup, run_scenario, validate_scenarios


async def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск Evals для MCP Agent")
    parser.add_argument("--scenario", "-s", type=str, help="ID конкретного сценария для запуска")
    parser.add_argument("--file", "-f", type=str, default=SCENARIOS_FILE, help="Путь к YAML файлу")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Количество прогонов на сценарий")
    args = parser.parse_args()

    config = load_config(conf.CLIENT_DIR / "servers.json")
    configure_eval_tickets_server(config)
    try:
        llm = create_llm(timeout=config.get("llm_timeout", 60.0))
    except Exception as e:
        print(f"Ошибка инициализации LLM: {e}")
        return
    always_confirm_tools = {
        server["name"]: set(server.get("always_confirm_by_server", []))
        for server in config["servers"]
    }

    with open(args.file, "r", encoding="utf-8") as file:
        scenarios = yaml.safe_load(file)
    validate_scenarios(scenarios)

    if args.scenario:
        scenarios = [s for s in scenarios if s.get("id") == args.scenario]
        if not scenarios:
            print(f"Сценарий '{args.scenario}' не найден в {args.file}")
            return
        print(f"Запуск только одного сценария: {args.scenario}")

    results_by_scenario = {}
    print(f"Запуск {len(scenarios)} сценариев по {args.runs} раз(а)...")
    print("-" * 80)

    try:
        for scenario in scenarios:
            scenario_id = scenario["id"]
            results_by_scenario[scenario_id] = []
            print(f"[{scenario_id}] {scenario.get('description', '')}...", end=" ", flush=True)

            for _ in range(args.runs):
                try:
                    init_eval_db()
                except PermissionError:
                    print("\nФайл БД заблокирован. Ждем 1 секунду...", flush=True)
                    await asyncio.sleep(1)
                    init_eval_db()

                apply_setup(scenario)
                mcp = MCPClient(config)
                try:
                    await mcp.connect_all()
                except RuntimeError as e:
                    print(f"\n{e}")
                    print(
                        "Запустите сервер базы знаний в отдельном терминале: "
                        "uv run python -m mcp_assistant.servers.knowledge"
                    )
                    print("После его запуска повторите evals.")
                    raise SystemExit(1) from None
                try:
                    agent = Agent(
                        mcp_client=mcp,
                        llm=llm,
                        tracer=None,
                        confirmation=None,
                        max_steps=config.get("max_steps", 8),
                        always_confirm_tools=always_confirm_tools,
                        max_history_messages=config.get("max_history_messages", 40),
                        max_history_chars=config.get("max_history_chars", 60000),
                        max_tool_result_chars=config.get("max_tool_result_chars", 12000),
                    )
                    result = await run_scenario(agent, scenario)
                    results_by_scenario[scenario_id].append(result)
                    print("[OK]" if result["success"] else "[FAIL]", end="", flush=True)
                finally:
                    await mcp.close_all()

            print()
            for run_index, result in enumerate(results_by_scenario[scenario_id], start=1):
                if result["success"]:
                    continue
                print(f"  ├─ Прогон #{run_index} — ПРОВАЛ")
                for reason in result["reasons"]:
                    print(f"  │    • {reason}")
                print(f"  │    Инструменты: {result['tool_calls'] or '(не вызывались)'}")
                for call in result["call_details"]:
                    print(f"  │    Вызов {call['tool']}:")
                    print(f"  │      arguments: {call['args']}")
                    print(f"  │      succeeded: {call['succeeded']}")
                    detail = call["error"] if call["error"] is not None else call["result"]
                    if detail is not None:
                        label = "tool_error" if call["error"] is not None else "tool_result"
                        detail = str(detail).replace("\n", " ")[:500]
                        print(f"  │      {label}: {detail!r}")
                print(f"  │    Шаги: {result['steps']}, токены: {result['tokens']}")
                answer = (result["final_answer"] or "").replace("\n", " ")[:500]
                print(f"  │    Ответ агента: {answer!r}")
            print()
    finally:
        llm.close()

    print_results(results_by_scenario, args.runs)


def print_results(results_by_scenario: dict, runs_count: int) -> None:
    """Печатает результаты сценариев и сводные метрики по прогонам и группам."""
    print("\n" + "=" * 105)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 105)

    total_runs = 0
    total_success = 0
    total_latency = 0.0
    total_steps = 0
    total_tokens = 0
    stable_success_count = 0
    stable_failure_count = 0
    unstable_count = 0
    successful_scenarios = 0
    total_scenarios = len(results_by_scenario)

    print(f"{'Сценарий':<25} | {'Успех':<7} | {'Стабильность':<22} | {'Шаги':<5} | {'Время(с)':<8} | {'Токены':<6}")
    print("-" * 105)

    for scenario_id, runs in results_by_scenario.items():
        actual_runs = len(runs)
        if actual_runs == 0:
            continue

        # Полный успех, полный провал и смешанные результаты различаются отдельно.
        success_count = sum(1 for run in runs if run["success"])
        if success_count == actual_runs:
            successful_scenarios += 1
            stable_success_count += 1
            stability = f"Стабильный успех ({success_count}/{actual_runs})"
        elif success_count == 0:
            stable_failure_count += 1
            stability = f"Стабильный провал ({success_count}/{actual_runs})"
        else:
            unstable_count += 1
            stability = f"Нестабильно ({success_count}/{actual_runs})"

        # Средние показатели рассчитываются по фактическим прогонам сценария.
        avg_latency = sum(run["latency"] for run in runs) / actual_runs
        avg_steps = sum(run["steps"] for run in runs) / actual_runs
        avg_tokens = sum(run["tokens"] for run in runs) / actual_runs
        print(f"{scenario_id:<25} | {f'{success_count}/{actual_runs}':<7} | {stability:<22} | {avg_steps:<5.1f} | {avg_latency:<8.2f} | {avg_tokens:<6.0f}")

        # Умножение на actual_runs возвращает суммы для общих средних по всем прогонам.
        total_runs += actual_runs
        total_success += success_count
        total_latency += avg_latency * actual_runs
        total_steps += avg_steps * actual_runs
        total_tokens += avg_tokens * actual_runs

    print("-" * 105)
    if total_runs == 0:
        print("Нет данных для отображения.")
        return

    # Cписок всех прогонов используется для общих и групповых процентов.
    all_runs = [run for runs in results_by_scenario.values() for run in runs]
    print(f"Всего сценариев: {total_scenarios} (прогонов: {total_runs})")
    print(f"Доля успешных прогонов (Run Success Rate): {total_success / total_runs * 100:.1f}%")
    print(
        f"Доля успешных сценариев (все прогоны успешны): "
        f"{successful_scenarios / total_scenarios * 100:.1f}% "
        f"({successful_scenarios}/{total_scenarios})"
    )

    # Каждая доля — число прогонов, прошедших конкретную проверку, к общему числу прогонов.
    check_metrics = [
        ("Правильные обязательные вызовы", sum(run["required_calls_ok"] for run in all_runs)),
        ("Прогоны без запрещённых вызовов", sum(not run["forbidden_called"] for run in all_runs)),
        ("Правильные подтверждения", sum(run["confirmations_ok"] for run in all_runs)),
        ("Корректный текст ответа", sum(run["answer_ok"] for run in all_runs)),
        ("Корректное состояние БД", sum(run["db_ok"] for run in all_runs)),
    ]
    for label, passed in check_metrics:
        print(f"{label}: {passed / total_runs * 100:.1f}% ({passed}/{total_runs})")

    # Успешные вызовы инструментов / Общее кол-во вызовов инструментов
    successful_tool_calls = sum(run["successful_tool_calls"] for run in all_runs)
    total_tool_calls = sum(run["total_tool_calls"] for run in all_runs)
    if total_tool_calls:
        print(
            f"Доля успешных вызовов инструментов: "
            f"{successful_tool_calls / total_tool_calls * 100:.1f}% "
            f"({successful_tool_calls}/{total_tool_calls})"
        )
    else:
        print("Доля успешных вызовов инструментов: нет вызовов")

    def print_group_success(field: str, title: str, order: tuple[str, ...]) -> None:
        """Выводит долю успешных прогонов по значениям поля в заданном порядке."""
        labels = {
            "read": "чтение (read)",
            "write": "запись (write)",
            "safety": "безопасность (safety)",
            "robustness": "устойчивость (robustness)",
            "critical": "критические (critical)",
            "major": "значительные (major)",
            "minor": "незначительные (minor)",
        }
        print(f"\n{title}")
        for group in order:
            # В группу попадают все прогоны с текущим category или severity.
            group_runs = [run for run in all_runs if run[field] == group]
            if not group_runs:
                continue
            passed = sum(run["success"] for run in group_runs)
            print(
                f"  {labels.get(group, group)}: {passed / len(group_runs) * 100:.1f}% "
                f"({passed}/{len(group_runs)})"
            )

    print_group_success("category", "Доля успешных прогонов по категориям", ("read", "write", "safety", "robustness"))
    print_group_success("severity", "Доля успешных прогонов по критичности", ("critical", "major", "minor"))

    print(f"\nСреднее число шагов агентного цикла: {total_steps / total_runs:.1f}")
    print(f"Средняя латентность выполнения сценария: {total_latency / total_runs:.2f} сек")
    print(f"Средний расход токенов: {total_tokens / total_runs:.0f}")
    print(
        f"Стабильность сценариев: успех — {stable_success_count}, "
        f"провал — {stable_failure_count}, нестабильно — {unstable_count}"
    )


if __name__ == "__main__":
    asyncio.run(main())
