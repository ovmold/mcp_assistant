import asyncio
import time
import yaml
import os
import sys
import argparse
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# --- ПОДГОТОВКА ОКРУЖЕНИЯ (ДО ИМПОРТА МОДУЛЕЙ ПРОЕКТА) ---
load_dotenv()

temp_eval_dir = os.path.join(tempfile.gettempdir(), "mcp_evals")
data_dir = os.path.join(temp_eval_dir, "data")

# Изменяет пути в переменных среды для изоляции БД
os.environ["DATA_DIR"] = data_dir
os.environ["TRACE_DIR"] = os.path.join(temp_eval_dir, "traces")
os.environ["DB_PATH"] = os.path.join(data_dir, "tickets.db")

# Добавляет корень проекта в sys.path, чтобы импорты работали
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- ИМПОРТ МОДУЛЕЙ ПРОЕКТА ---
from gigachat import GigaChat
from scripts import conf
from client.agent import Agent, LLMClient
from client.config_mcp import load_config
from client.mcp_client import MCPClient
from client.confirmation import ConfirmationProvider
from conf_evals import init_eval_db, SCENARIOS_FILE


# --- ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ ---
class EvalTracer:
    def __init__(self):
        self.events = []

    def log(self, event_type: str, **data):
        self.events.append((event_type, data))


class EvalConfirmation(ConfirmationProvider):
    def __init__(self, scenario: dict):
        self.scenario = scenario

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        resp = self.scenario.get("confirmation_response", "да")
        if resp is None:
            return True
        return str(resp).lower() in ("да", "yes", "true", "1")


# --- ЛОГИКА ЗАПУСКА ОДНОГО СЦЕНАРИЯ ---
async def run_scenario(agent: Agent, scenario: dict) -> dict:
    tracer = EvalTracer()
    agent.tracer = tracer
    agent.confirmation = EvalConfirmation(scenario)

    start_time = time.perf_counter()
    try:
        final_answer = await agent.run(scenario["user_input"])
        error = None
    except Exception as e:
        final_answer = ""
        error = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - start_time

    # Сбор метрик
    tool_calls = [d["tool"] for ev, d in tracer.events if ev == "tool_call"]
    confirmation_asked = [d["tool"] for ev, d in tracer.events if ev == "confirmation"]
    confirmation_denied = [d["tool"] for ev, d in tracer.events if ev == "confirmation_denied"]
    steps = max((d.get("step", 0) for ev, d in tracer.events), default=0)
    tokens = sum(d.get("tokens", 0) for ev, d in tracer.events if ev == "llm_response")

    # Проверки
    expected_tools = scenario.get("expected_tools", [])
    expected_confirmation_tools = scenario.get("expected_confirmation_tools", [])
    forbidden_tools = scenario.get("forbidden_tools", [])
    expected_text = scenario.get("expected_response_contains")

    text_match = True
    if expected_text:
        if isinstance(expected_text, str):
            text_match = expected_text.lower() in final_answer.lower()
        elif isinstance(expected_text, list):
            text_match = any(text.lower() in final_answer.lower() for text in expected_text)

    missing_tools = [t for t in expected_tools if t not in tool_calls]
    missing_confirm = [t for t in expected_confirmation_tools if t not in confirmation_asked]
    forbidden_called = [t for t in forbidden_tools if t in tool_calls]

    reasons = []
    if error: reasons.append(f"Исключение: {error}")
    if missing_tools: reasons.append(f"Не выполнены ожидаемые: {missing_tools}")
    if missing_confirm: reasons.append(f"Не запрошено подтверждение для: {missing_confirm}")
    if forbidden_called: reasons.append(f"Выполнены запрещенные: {forbidden_called}")
    if not text_match and expected_text:
        reasons.append(f"В ответе нет ни одной из подстрок: {expected_text}")

    return {
        "success": not reasons,
        "latency": latency, "steps": steps, "tokens": tokens,
        "reasons": reasons,
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "confirmation_asked": confirmation_asked,
        "confirmation_denied": confirmation_denied,
        "events": tracer.events,
        "missing_tools": missing_tools,
        "forbidden_called": forbidden_called,
    }


# --- MAIN ---
async def main():
    parser = argparse.ArgumentParser(description="Запуск Evals для MCP Agent")
    parser.add_argument("--scenario", "-s", type=str, help="ID конкретного сценария для запуска")
    parser.add_argument("--file", "-f", type=str, default=SCENARIOS_FILE, help="Путь к YAML файлу")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Количество прогонов на сценарий")
    args = parser.parse_args()

    auth_key = os.environ.get("GIGACHAT_AUTH_KEY")
    if not auth_key:
        print("Ошибка: GIGACHAT_AUTH_KEY не задан")
        return

    # Загрузка конфигурации
    config = load_config(conf.CLIENT_DIR / "servers.json")
    
    # Инициализация LLM
    giga = GigaChat(
        base_url=config["api_base"],
        model=config["model"],
        credentials=auth_key,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
        timeout=config.get("llm_timeout", 60.0),
    )
    llm = LLMClient(giga_instance=giga, model=config["model"])

    write_tools_by_server = {
        server["name"]: set(server.get("write_tools", []))
        for server in config["servers"]
    }

    # Загрузка сценариев
    with open(args.file, "r", encoding="utf-8") as f:
        scenarios = yaml.safe_load(f)

    if args.scenario:
        scenarios = [s for s in scenarios if s.get("id") == args.scenario]
        if not scenarios:
            print(f"Сценарий '{args.scenario}' не найден в {args.file}")
            return
        print(f"Запуск только одного сценария: {args.scenario}")

    results_by_scenario = {}
    runs_count = args.runs

    print(f"Запуск {len(scenarios)} сценариев по {runs_count} раз(а)...")
    print("-" * 80)

    for scenario in scenarios:
        sid = scenario["id"]
        results_by_scenario[sid] = []
        print(f"[{sid}] {scenario.get('description', '')}...", end=" ", flush=True)

        for _ in range(runs_count):
            # 1. Очищаем и создаем БД перед запуском
            try:
                init_eval_db()
            except PermissionError:
                print("\nФайл БД заблокирован. Ждем 1 секунду...", flush=True)
                await asyncio.sleep(1)
                init_eval_db()

            # 2. Перезапускаем MCP-сервер (чтобы он открыл свежую БД)
            mcp = MCPClient(config)
            await mcp.connect_all()

            try:
                # 3. Создаем агента
                agent = Agent(
                    mcp_client=mcp,
                    llm=llm,
                    tracer=EvalTracer(), 
                    confirmation=EvalConfirmation(scenario),
                    max_steps=config.get("max_steps", 8),
                    write_tools_by_server=write_tools_by_server,
                )
                
                # 4. Запуск
                res = await run_scenario(agent, scenario)
                results_by_scenario[sid].append(res)
                print("[OK]" if res["success"] else "[FAIL]", end="", flush=True)
            finally:
                # 5. Обязательно гасим сервер, чтобы отпустить файл БД
                await mcp.close_all()

        print()

        # Вывод деталей по провалам
        for run_idx, r in enumerate(results_by_scenario[sid], start=1):
            if r["success"]:
                continue
            print(f"  ├─ Прогон #{run_idx} — ПРОВАЛ")
            for reason in r["reasons"]:
                print(f"  │    • {reason}")
            print(f"  │    Инструменты: {r['tool_calls'] or '(не вызывались)'}")
            print(f"  │    Шаги: {r['steps']}, токены: {r['tokens']}")
            ans = (r["final_answer"] or "").replace("\n", " ")[:500]
            print(f"  │    Ответ агента: {ans!r}")
        print()

    # --- Вывод итогов ---
    print_results(results_by_scenario, runs_count)


def print_results(results_by_scenario: dict, runs_count: int):
    print("\n" + "="*105)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*105)
    
    total_runs = 0
    total_success = 0
    total_latency = 0.0
    total_steps = 0
    total_tokens = 0
    stable_count = 0
    total_scenarios = len(results_by_scenario)
    
    print(f"{'Сценарий':<25} | {'Успех':<7} | {'Стабильность':<15} | {'Шаги':<5} | {'Время(с)':<8} | {'Токены':<6}")
    print("-" * 105)
    
    for sid, runs in results_by_scenario.items():
        actual_runs = len(runs)
        if actual_runs == 0:
            continue
            
        success_count = sum(1 for r in runs if r["success"])
        avg_latency = sum(r["latency"] for r in runs) / actual_runs
        avg_steps = sum(r["steps"] for r in runs) / actual_runs
        avg_tokens = sum(r["tokens"] for r in runs) / actual_runs
        
        if success_count == actual_runs:
            stability = f"Стабильно ({success_count}/{actual_runs})"
            stable_count += 1
        elif success_count == 0:
            stability = f"Стабильно ({success_count}/{actual_runs})"
            stable_count += 1
        else:
            stability = f"Флаки ({success_count}/{actual_runs})"
            
        print(f"{sid:<25} | {f'{success_count}/{actual_runs}':<7} | {stability:<15} | {avg_steps:<5.1f} | {avg_latency:<8.2f} | {avg_tokens:<6.0f}")
        
        total_runs += actual_runs
        total_success += success_count
        total_latency += avg_latency * actual_runs
        total_steps += avg_steps * actual_runs
        total_tokens += avg_tokens * actual_runs
        
    print("-" * 105)
    
    if total_runs == 0:
        print("Нет данных для отображения.")
        return
        
    tool_accuracy = sum(1 for runs in results_by_scenario.values() for r in runs if not r['missing_tools'] and not r['forbidden_called'])
    
    print(f"Всего сценариев: {total_scenarios} (прогонов: {total_runs})")
    print(f"Общая доля успешных сценариев (Success Rate): {total_success / total_runs * 100:.1f}%")
    print(f"Точность выбора инструментов (без запрещенных): {tool_accuracy / total_runs * 100:.1f}%")
    print(f"Среднее число шагов: {total_steps / total_runs:.1f}")
    print(f"Средняя латентность: {total_latency / total_runs:.2f} сек")
    print(f"Средний расход токенов: {total_tokens / total_runs:.0f}")
    print(f"Стабильных сценариев (без флаков): {stable_count} из {total_scenarios}")

if __name__ == "__main__":
    asyncio.run(main())