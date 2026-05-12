from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional

from pydantic import BaseModel


class AgentResponse(BaseModel):
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def execute(self, prompt: str, context: Optional[dict[str, Any]] = None) -> AgentResponse:
        pass
    
    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]