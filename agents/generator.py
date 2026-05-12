from typing import Any, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from serving.model import ModelManager
from core.world_model import WorldModel
from .base import AgentResponse, BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class GeneratorAgent(BaseAgent):
    def __init__(
        self,
        name: str = "Generator",
        model_path: str = "./models/tinyllama-trained-slm",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        imagination_depth: int = 3,
        thinking_steps: int = 5,
    ):
        super().__init__(name, model_path, "", temperature, max_tokens)
        self.model_manager = ModelManager(model_path=model_path)
        self._model_loaded = False

        # Native World Model Engine
        self.world_model = WorldModel(
            imagination_depth=imagination_depth,
            thinking_steps=thinking_steps,
            enable_simulation=True
        )

    def _ensure_model_loaded(self):
        """Lazy-load the underlying LLM on first use."""
        if not self._model_loaded:
            try:
                self.model_manager.load_model_sync()
                self._model_loaded = True
                logger.info("GeneratorAgent model loaded successfully")
            except Exception as e:
                logger.error(f"GeneratorAgent failed to load model: {e}")
                raise

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None
    ) -> AgentResponse:
        try:
            self._ensure_model_loaded()
            system_prompt = self._build_system_prompt(context)

            # 1. THINKING PHASE - Internal reasoning
            thoughts = await self.world_model.think(prompt, context)

            # 2. IMAGINATION PHASE - Simulate multiple scenarios
            scenarios = await self.world_model.imagine_scenarios(prompt, context)

            # 3. GENERATION PHASE - Create response after thinking
            document_context = context.get("document_context") if context else None
            response_data = await self.world_model.generate_response(
                prompt=prompt,
                scenarios=scenarios,
                thoughts=thoughts,
                model_manager=self.model_manager,
                document_context=document_context,
                context=context
            )

            content = response_data["content"]
            logger.info(f"Generator produced output ({len(content)} chars)")

            return AgentResponse(
                content=content,
                metadata={
                    "model": self.model,
                    "thought_steps": response_data.get("thinking_steps", 0),
                    "scenarios_explored": response_data.get("scenarios", response_data.get("scenarios_explored", 0)),
                    "best_scenario_probability": response_data.get("best_prob", response_data.get("best_scenario_probability", 0.7)),
                    "world_simulation_used": True,
                    "native_world_model": True
                }
            )
        except Exception as e:
            logger.error(f"Generator error: {e}")
            return AgentResponse(
                content="",
                success=False,
                error=str(e)
            )

    def _build_system_prompt(self, context: Optional[dict[str, Any]] = None) -> str:
        base_prompt = """You are a Creative Generator agent specialized in producing high-quality,
accurate, and comprehensive responses. Your goal is to generate the best possible output
for the given query."""

        if context and context.get("retrieved_knowledge"):
            retrieved = context["retrieved_knowledge"]
            base_prompt += f"\n\nUse the following retrieved knowledge to inform your response:\n{retrieved}"

        return base_prompt

    async def close(self) -> None:
        if self.model_manager:
            await self.model_manager.unload_model()
