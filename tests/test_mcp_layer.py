"""
Интеграционные тесты MCP-слоя проверяют, как инструменты ведут себя через MCP-протокол
"""

import pytest

# ---------- tickets ----------

@pytest.mark.anyio
async def test_tickets_search_via_mcp(tickets_client):
    result = await tickets_client.call_tool("search_tickets", {"query": "502"})
    assert not result.is_error
    assert result.content
    assert "502" in result.content[0].text


@pytest.mark.anyio
async def test_tickets_get_ticket_via_mcp(tickets_client):
    result = await tickets_client.call_tool("get_ticket", {"ticket_id": 1})
    assert not result.is_error
    assert "Ошибка 502" in result.content[0].text


@pytest.mark.anyio
async def test_tickets_empty_query_returns_error_via_mcp(tickets_client):
    result = await tickets_client.call_tool("search_tickets", {"query": ""})
    assert result.is_error
    assert result.content
    assert "query пуст" in result.content[0].text.lower()


@pytest.mark.anyio
async def test_tickets_update_status_via_mcp(tickets_client):
    result = await tickets_client.call_tool(
        "update_status",
        {"ticket_id": 1, "status": "closed", "comment": "Закрыто"},
    )
    assert not result.is_error


@pytest.mark.anyio
async def test_tickets_update_status_invalid_via_mcp(tickets_client):
    result = await tickets_client.call_tool(
        "update_status",
        {"ticket_id": 1, "status": "garbage", "comment": "test"},
    )

    assert result.is_error
    assert result.content
    # Проверяет, что текст ошибки упоминает недопустимый статус
    assert "неправильный" in result.content[0].text.lower()


# ---------- knowledge ----------

@pytest.mark.anyio
async def test_knowledge_search_via_mcp(knowledge_client):
    result = await knowledge_client.call_tool(
        "search_kb", {"query": "оплата", "top_k": 3},
    )
    assert not result.is_error
    assert result.content
    assert "payment-502" in result.content[0].text    


@pytest.mark.anyio
async def test_knowledge_path_traversal_via_mcp(knowledge_client):
    result = await knowledge_client.call_tool(
                "get_article", {"article_type": "../secret"},
            )
    assert result.is_error