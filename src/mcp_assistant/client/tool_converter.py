"""Преобразует MCP-инструменты во внутренние спецификации для LLM.

Хранит соответствие имён с префиксом сервера исходным вызовам MCP
и метаданные для проверки необходимости подтверждения.
"""

from mcp.types import Tool

class ToolConverter:
    def __init__(self) -> None:
        self.tool_specs = []  # Общие спецификации инструментов для LLM
        self.mapping = {}         # "tickets__search": ("tickets", "search")
        self._raw: dict[str, Tool] = {} # prefixed → raw Tool

    def register_server_tools(self, server_name: str, mcp_tools: list[Tool]) -> None:
        """Регистрирует MCP-инструменты сервера для вызовов через LLM.

        Сохраняет спецификации с префиксом сервера, соответствие имён и исходные Tool.
        """
        for tool in mcp_tools:
            prefixed_name = f"{server_name}__{tool.name}"
            
            func_dict = {
                "name": prefixed_name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            }
            
            self.tool_specs.append(func_dict)
            
            self.mapping[prefixed_name] = (server_name, tool.name)
            self._raw[prefixed_name] = tool

    def get_tool_specs(self) -> list[dict]:
        """Возвращает внутренние спецификации инструментов для LLM."""
        return self.tool_specs

    def resolve(self, prefixed_name: str) -> tuple[str, str] | None:
        """Возвращает (server_name, original_tool_name)."""
        return self.mapping.get(prefixed_name)

    def requires_confirmation(self, prefixed_name: str) -> tuple[bool, str]:
        """
        Решает, нужно ли подтверждение по MCP-аннотациям.

        Returns:
            (needs_confirmation, reason) — reason годится для логов и UI.

        Fail-safe: если инструмент или аннотации отсутствуют — спрашиваем.
        """
        tool = self._raw.get(prefixed_name)
        if tool is None:
            return True, "tool not found (fail-safe)"

        ann = getattr(tool, "annotations", None)
        if ann is None:
            return True, "no annotations (fail-safe)"

        # В mcp[cli]>=2.3.0 поля называются read_only_hint / destructive_hint
        if getattr(ann, "read_only_hint", None) is True:
            return False, "readOnlyHint=True"

        if getattr(ann, "destructive_hint", None) is True:
            return True, "destructiveHint=True"

        return True, "write tool (readOnlyHint is not True)"
