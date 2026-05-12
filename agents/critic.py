from typing import Any, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from serving.model import ModelManager
from core.world_model import WorldModel
from .base import AgentResponse, BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class CriticAgent(BaseAgent):
    def __init__(
        self,
        name: str = "Critic",
        model_path: str = "./models/tinyllama-trained-slm",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        critical_thinking: bool = True,
    ):
        super().__init__(name, model_path, "", temperature, max_tokens)
        self.model_manager = ModelManager(model_path=model_path)
        self._model_loaded = False

        # Native World Model Engine for evaluation
        self.world_model = WorldModel(
            imagination_depth=2,
            thinking_steps=3,
            enable_simulation=True
        )

    def _ensure_model_loaded(self):
        """Lazy-load the underlying LLM on first use."""
        if not self._model_loaded:
            try:
                self.model_manager.load_model_sync()
                self._model_loaded = True
                logger.info("CriticAgent model loaded successfully")
            except Exception as e:
                logger.error(f"CriticAgent failed to load model: {e}")
                raise

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None
    ) -> AgentResponse:
        try:
            self._ensure_model_loaded()
            system_prompt = self._build_system_prompt()

            # Validate against world model first
            evaluation_data = await self.world_model.evaluate_response(
                response=prompt,
                original_query=context.get('user_query', '') if context else '',
                model_manager=self.model_manager
            )

            content = evaluation_data["critique"]
            scores = self._parse_scores(content)

            logger.info(f"Critic evaluated output with scores: {scores}")

            return AgentResponse(
                content=content,
                metadata={
                    "model": self.model,
                    "scores": scores,
                    "validation_steps": 3,
                    "world_model_check": True,
                    "native_world_model": True
                }
            )
        except Exception as e:
            logger.error(f"Critic error: {e}")
            return AgentResponse(
                content="",
                success=False,
                error=str(e)
            )

    def _build_system_prompt(self) -> str:
        return """You are a strict Critic agent specialized in evaluating output quality.
Provide honest, detailed critiques focusing on accuracy, clarity, and completeness.
Rate each dimension objectively on a scale of 0-10.

CRITICAL CHECKS:
1. FACTUAL ACCURACY: Are all claims verifiable and true? Deduct heavily for false claims.
2. HALLUCINATION CHECK: Did the response invent any facts, names, dates, or events? 
3. CAUSAL CONSISTENCY: Does the response violate known cause-effect relationships?
4. PHYSICS VIOLATIONS: Does it contradict basic physics (gravity, thermodynamics, etc.)?
5. COMPLETENESS: Does it fully address all parts of the query?
6. CLARITY: Is the response well-structured and easy to understand?

If ANY hallucination is detected, score accuracy as 0."""

    def _parse_scores(self, content: str) -> dict[str, float]:
        scores = {"accuracy": 5.0, "clarity": 5.0, "completeness": 5.0, "overall": 5.0}

        for line in content.split("\n"):
            line = line.strip().upper()
            for key in scores:
                if key.upper() in line:
                    try:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            scores[key] = float(parts[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass

        weights = {"accuracy": 0.4, "clarity": 0.3, "completeness": 0.3}
        scores["overall"] = sum(scores[k] * weights[k] for k in weights)

        return scores

    async def close(self) -> None:
        if self.model_manager:
            await self.model_manager.unload_model()
