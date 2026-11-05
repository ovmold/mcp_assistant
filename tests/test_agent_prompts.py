from mcp_assistant.client.agent_prompts import build_system_prompt


def test_system_prompt_supports_relative_date_filters():
    prompt = build_system_prompt([])

    assert "Текущая дата и время в UTC:" in prompt
    assert "created_from" in prompt
    assert "created_to" in prompt
    assert "за последний месяц" in prompt
