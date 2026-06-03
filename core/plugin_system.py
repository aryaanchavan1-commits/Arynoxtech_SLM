"""
AnonyLLM Plugin & Tool System
Extensible framework for adding capabilities: voice, vision, APIs, custom tools
"""
import os, sys, json, importlib, inspect, asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent.parent / "plugins"
os.makedirs(PLUGIN_DIR, exist_ok=True)


class PluginType(Enum):
    TOOL = "tool"
    INTEGRATION = "integration"
    SERVICE = "service"
    MODIFIER = "modifier"


class ToolType(Enum):
    CALCULATOR = "calculator"
    WEB_SEARCH = "web_search"
    CODE_EXECUTOR = "code_executor"
    VOICE = "voice"
    VISION = "vision"
    FILE = "file"
    API = "api"
    CUSTOM = "custom"


@dataclass
class ToolSpec:
    name: str
    description: str
    tool_type: ToolType
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    requires_network: bool = False
    requires_auth: bool = False
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool:
    """Base class for all tools. Override execute()."""
    name: str = ""
    description: str = ""
    tool_type: ToolType = ToolType.CUSTOM
    parameters: Dict[str, Any] = {}
    requires_network: bool = False
    requires_auth: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def execute(self, **params) -> ToolResult:
        raise NotImplementedError

    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description,
            tool_type=self.tool_type, parameters=self.parameters,
            requires_network=self.requires_network,
            requires_auth=self.requires_auth, config=self.config,
        )


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate mathematical expressions"
    tool_type = ToolType.CALCULATOR
    parameters = {"expression": {"type": "string", "description": "Math expression"}}

    async def execute(self, expression: str = "", **kw) -> ToolResult:
        try:
            import math, re
            safe = re.sub(r'[^0-9+\-*/.()%^ ]', '', expression)
            safe = safe.replace('^', '**')
            allowed = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
            allowed.update({"abs": abs, "round": round, "max": max, "min": min})
            result = eval(safe, {"__builtins__": {}}, allowed)
            return ToolResult(success=True, result=str(result))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CodeExecutorTool(BaseTool):
    name = "code_executor"
    description = "Execute Python code securely"
    tool_type = ToolType.CODE_EXECUTOR
    parameters = {"code": {"type": "string", "description": "Python code"}}

    async def execute(self, code: str = "", **kw) -> ToolResult:
        try:
            import io, contextlib
            out = io.StringIO()
            restricted = {"__builtins__": {"print": print, "len": len, "range": range,
                          "int": int, "float": float, "str": str, "list": list,
                          "dict": dict, "tuple": tuple, "bool": bool, "True": True,
                          "False": False, "None": None}}
            with contextlib.redirect_stdout(out):
                exec(code, restricted, {})
            return ToolResult(success=True, result=out.getvalue() or "(no output)")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the internet for information"
    tool_type = ToolType.WEB_SEARCH
    requires_network = True
    parameters = {"query": {"type": "string", "description": "Search query"}}

    async def execute(self, query: str = "", **kw) -> ToolResult:
        try:
            from utils.web_search import WebLearner
            learner = WebLearner(enabled=True)
            results = await learner.search(query, num_results=5)
            if results:
                formatted = "\n".join(
                    f"- {r.title}: {r.snippet[:200]}" for r in results
                )
                return ToolResult(success=True, result=formatted)
            return ToolResult(success=True, result="No results found.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class VoiceTool(BaseTool):
    name = "voice"
    description = "Text-to-speech and speech-to-text"
    tool_type = ToolType.VOICE
    parameters = {
        "action": {"type": "string", "description": "speak or listen"},
        "text": {"type": "string", "description": "Text to speak (for speak action)"},
    }

    async def execute(self, action: str = "speak", text: str = "", **kw) -> ToolResult:
        try:
            from utils.voice import VoiceEngine
            engine = VoiceEngine()
            if action == "speak" and text:
                engine.speak(text)
                return ToolResult(success=True, result="Spoken successfully")
            elif action == "listen":
                result = engine.listen(timeout=5)
                if result:
                    return ToolResult(success=True, result=result)
                return ToolResult(success=False, error="No speech detected")
            return ToolResult(success=False, error="Invalid action")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class PluginRegistry:
    """Central registry for all tools, integrations, and plugins."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(CalculatorTool())
        self.register(CodeExecutorTool())
        self.register(WebSearchTool())
        self.register(VoiceTool())

    def register(self, tool: BaseTool):
        if not tool.name:
            raise ValueError("Tool must have a name")
        self._tools[tool.name] = tool
        self._trigger_hook("tool_registered", tool)

    def unregister(self, name: str):
        self._tools.pop(name, None)
        self._trigger_hook("tool_unregistered", name)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self, enabled_only: bool = True) -> List[ToolSpec]:
        specs = []
        for tool in self._tools.values():
            spec = tool.get_spec()
            if enabled_only and not spec.enabled:
                continue
            specs.append(spec)
        return specs

    async def execute(self, tool_name: str, **params) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{tool_name}' not found")
        if not tool.get_spec().enabled:
            return ToolResult(success=False, error=f"Tool '{tool_name}' is disabled")
        try:
            result = tool.execute(**params)
            if asyncio.iscoroutine(result):
                result = await result
            self._trigger_hook("tool_executed", tool_name, result)
            return result
        except Exception as e:
            self._trigger_hook("tool_error", tool_name, str(e))
            return ToolResult(success=False, error=str(e))

    def execute_sync(self, tool_name: str, **params) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{tool_name}' not found")
        try:
            result = tool.execute(**params)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    result = loop.run_until_complete(result)
                except RuntimeError:
                    result = asyncio.run(result)
            return result
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def on(self, event: str, callback: Callable):
        self._hooks.setdefault(event, []).append(callback)

    def _trigger_hook(self, event: str, *args, **kwargs):
        for cb in self._hooks.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception:
                pass

    def load_plugins_from(self, directory: Optional[Path] = None):
        """Dynamically load plugin files from a directory."""
        directory = directory or PLUGIN_DIR
        if not directory.exists():
            return
        sys.path.insert(0, str(directory.parent))
        for pyfile in sorted(directory.glob("*.py")):
            if pyfile.name.startswith("_"):
                continue
            try:
                module_name = f"plugins.{pyfile.stem}"
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                else:
                    importlib.import_module(module_name)
                self._trigger_hook("plugin_loaded", str(pyfile))
            except Exception as e:
                print(f"[Plugin] Failed to load {pyfile.name}: {e}")

    def enable_tool(self, name: str):
        tool = self._tools.get(name)
        if tool:
            spec = tool.get_spec()
            spec.enabled = True

    def disable_tool(self, name: str):
        tool = self._tools.get(name)
        if tool:
            spec = tool.get_spec()
            spec.enabled = False

    def get_all_specs(self) -> List[Dict[str, Any]]:
        return [{
            "name": s.name, "description": s.description,
            "type": s.tool_type.value, "parameters": s.parameters,
            "enabled": s.enabled, "requires_network": s.requires_network,
        } for s in self.list_tools(enabled_only=False)]


# Global singleton
_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        _registry.load_plugins_from()
    return _registry


# === Decorator API for creating tools ===
def tool(name: str, description: str = "", tool_type: ToolType = ToolType.CUSTOM):
    """Decorator to register a function as a tool."""
    def decorator(func):
        class FuncTool(BaseTool):
            name = name
            description = description or func.__doc__ or ""
            tool_type = tool_type
            async def execute(self, **params):
                try:
                    result = func(**params)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return ToolResult(success=True, result=result)
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
        get_registry().register(FuncTool())
        return func
    return decorator


# === Example: Image analysis tool ===
class VisionTool(BaseTool):
    name = "vision"
    description = "Analyze images using OCR and description"
    tool_type = ToolType.VISION
    parameters = {
        "file_path": {"type": "string", "description": "Path to image file"},
    }

    async def execute(self, file_path: str = "", **kw) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(success=False, error=f"File not found: {file_path}")
            from PIL import Image
            img = Image.open(file_path)
            info = f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}"
            try:
                import pytesseract
                text = pytesseract.image_to_string(img)
                if text.strip():
                    info += f"\nOCR Text: {text.strip()[:500]}"
            except ImportError:
                info += "\nOCR: pytesseract not installed"
            return ToolResult(success=True, result=info)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
