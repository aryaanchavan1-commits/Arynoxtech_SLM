"""
Tool-use registry for the 2026 SLM.
Allows the model to call external tools and functions.
"""
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ToolType(Enum):
    CALCULATOR = "calculator"
    WEB_SEARCH = "web_search"
    CODE_EXECUTOR = "code_executor"
    KNOWLEDGE_BASE = "knowledge_base"
    TIME_DATE = "time_date"


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools."""
        self.register(ToolType.CALCULATOR.value, self._calculator)
        self.register(ToolType.TIME_DATE.value, self._time_date)
        self.register(ToolType.KNOWLEDGE_BASE.value, self._knowledge_base)
    
    def register(self, name: str, func: Callable):
        """Register a new tool."""
        self._tools[name] = func
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with descriptions."""
        return [
            {
                "name": ToolType.CALCULATOR.value,
                "description": "Perform mathematical calculations",
                "parameters": {"expression": "string"}
            },
            {
                "name": ToolType.TIME_DATE.value,
                "description": "Get current time and date",
                "parameters": {"timezone": "string (optional)"}
            },
            {
                "name": ToolType.KNOWLEDGE_BASE.value,
                "description": "Query the world knowledge base",
                "parameters": {"query": "string"}
            }
        ]
    
    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool with given parameters."""
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        try:
            func = self._tools[tool_name]
            if asyncio.iscoroutinefunction(func):
                result = await func(**parameters)
            else:
                result = func(**parameters)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )
    
    def _calculator(self, expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        # Only allow safe operations
        allowed_names = {
            "abs": abs, "max": max, "min": min,
            "sum": sum, "pow": pow, "round": round,
            "len": len
        }
        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _time_date(self, timezone: Optional[str] = None) -> str:
        """Get current time and date."""
        from datetime import datetime
        now = datetime.now()
        if timezone:
            return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"
        return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _knowledge_base(self, query: str) -> str:
        """Query world knowledge base."""
        # This will be integrated with the WorldModel's knowledge base
        return f"Knowledge base query: {query}"


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from model output.
    
    Expected format:
    <tool_call>
    {"name": "calculator", "parameters": {"expression": "2+2"}}
    </tool_call>
    """
    import re
    tool_calls = []
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match.strip())
            tool_calls.append(data)
        except json.JSONDecodeError:
            continue
    
    return tool_calls


def format_tool_response(tool_results: List[ToolResult]) -> str:
    """Format tool results for inclusion in the conversation."""
    responses = []
    for result in tool_results:
        if result.success:
            responses.append(f"[{result.tool_name}] Result: {result.result}")
        else:
            responses.append(f"[{result.tool_name}] Error: {result.error}")
    
    return "\n".join(responses)

