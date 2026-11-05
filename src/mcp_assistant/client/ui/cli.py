from mcp_assistant.client.agent import Agent


async def run_cli(agent: Agent) -> None:
    """Простой CLI-чат."""
    print("=" * 60)
    print("MCP-агент готов. Введите запрос или /reset для очистки истории; /exit для выхода.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_input:
            continue

        if user_input in ("/reset", "/clear"):
            agent.reset()
            print("История очищена.")
            continue

        if user_input in ("/exit", "/quit"):
            print("Выход.")
            break

        try:
            response = await agent.run(user_input)
            print(f"\nАгент: {response}")
        except Exception as e:
            print(f"\nОшибка: {e}")
