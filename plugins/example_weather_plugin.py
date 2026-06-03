"""
Example plugin for AnonyLLM.
Demonstrates how to add a custom tool.
Usage: Drop any .py file in plugins/ with a register(registry) function.
"""
import os, json, urllib.request
from core.plugin_system import BaseTool, ToolResult, ToolType


class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a city (use wttr.in)"
    tool_type = ToolType.API
    requires_network = True
    parameters = {
        "city": {"type": "string", "description": "City name"},
    }

    async def execute(self, city: str = "london", **kw) -> ToolResult:
        try:
            url = f"https://wttr.in/{city}?format=%C+%t+%h+%w"
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read().decode()
            return ToolResult(success=True, result=f"Weather for {city}: {data.strip()}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def register(registry):
    """Auto-called by plugin loader."""
    registry.register(WeatherTool())
    return {"name": "weather", "version": "1.0"}


def register_tool(registry):
    """Alternative entry point."""
    register(registry)
