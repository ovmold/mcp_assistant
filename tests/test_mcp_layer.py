"""
Интеграционные тесты MCP-слоя проверяют, как инструменты ведут себя через MCP-протокол
"""

import pytest

from mcp_assistant.storage.models import TICKET_PRIORITIES, TICKET_STATUSES

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
async def test_tickets_filter_only_search_via_mcp(tickets_client):
    result = await tickets_client.call_tool("search_tickets", {
        "status": "open",
        "priority": "high",
        "customer": "Globex",
    })

    assert not result.is_error
    assert result.content
    assert '"id": 1' in result.content[0].text
    assert '"id": 3' not in result.content[0].text


@pytest.mark.anyio
async def test_tickets_date_filter_via_mcp(tickets_client):
    result = await tickets_client.call_tool("search_tickets", {
        "created_from": "2026-10-14",
    })

    assert not result.is_error
    text = "\n".join(block.text for block in result.content)
    assert '"id": 1' in text
    assert '"id": 2' in text
    assert '"id": 3' not in text


@pytest.mark.anyio
async def test_add_comment_author_is_required_via_mcp(tickets_client):
    result = await tickets_client.list_tools()
    add_comment = next(tool for tool in result.tools if tool.name == "add_comment")

    assert "author" in add_comment.input_schema["properties"]
    assert set(add_comment.input_schema["required"]) == {"ticket_id", "text", "author"}


@pytest.mark.anyio
async def test_ticket_tools_expose_status_and_priority_enums(tickets_client):
    result = await tickets_client.list_tools()
    tools = {tool.name: tool for tool in result.tools}

    search_properties = tools["search_tickets"].input_schema["properties"]
    assert search_properties["status"]["anyOf"][0]["enum"] == list(TICKET_STATUSES)
    assert search_properties["priority"]["anyOf"][0]["enum"] == list(TICKET_PRIORITIES)
    update_status = tools["update_status"].input_schema["properties"]["status"]
    assert update_status["enum"] == list(TICKET_STATUSES)


@pytest.mark.anyio
async def test_tickets_empty_query_returns_error_via_mcp(tickets_client):
    result = await tickets_client.call_tool("search_tickets", {"query": ""})
    assert result.is_error
    assert result.content
    assert "query пуст" in result.content[0].text.lower()


@pytest.mark.anyio
async def test_tickets_add_comment_returns_result_via_mcp(tickets_client):
    result = await tickets_client.call_tool("add_comment", {
        "ticket_id": 2,
        "text": "Комментарий",
        "author": "support",
    })

    assert not result.is_error
    text = result.content[0].text
    assert '"ticket_id": 2' in text
    assert '"comment_id":' in text
    assert '"added": true' in text


@pytest.mark.anyio
async def test_tickets_update_status_via_mcp(tickets_client):
    result = await tickets_client.call_tool(
        "update_status",
        {"ticket_id": 1, "status": "closed", "comment": "Закрыто"},
    )
    assert not result.is_error
    text = result.content[0].text
    assert '"ticket_id": 1' in text
    assert '"old_status": "open"' in text
    assert '"new_status": "closed"' in text
    assert '"updated": true' in text
    assert '"comment_id":' in text


@pytest.mark.anyio
async def test_tickets_update_status_invalid_via_mcp(tickets_client):
    result = await tickets_client.call_tool(
        "update_status",
        {"ticket_id": 1, "status": "garbage", "comment": "test"},
    )

    assert result.is_error
    assert result.content
    # MCP/Pydantic отклоняет значение до входа в обработчик инструмента
    assert "input should be" in result.content[0].text.lower()


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