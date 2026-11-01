from mcp.types import Tool

class ToolConverter:
    def __init__(self):
        self.gigachat_tools = []  # Список для GigaChat
        self.mapping = {}         # "tickets__search": ("tickets", "search")

    def add_server_tools(self, server_name: str, mcp_tools: list[Tool]) -> None:
        for tool in mcp_tools:
            prefixed_name = f"{server_name}__{tool.name}"
            
            func_dict = {
                "name": prefixed_name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            }
            
            self.gigachat_tools.append(func_dict)
            
            self.mapping[prefixed_name] = (server_name, tool.name)

    def get_gigachat_functions(self) -> list[dict]:
        """Возвращает готовый список для GigaChat."""
        return self.gigachat_tools

    def resolve(self, prefixed_name: str) -> tuple[str, str] | None:
        """Возвращает (server_name, original_tool_name)."""
        return self.mapping.get(prefixed_name)