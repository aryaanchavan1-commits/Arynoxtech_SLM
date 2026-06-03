from typing import Any, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from serving.model import ModelManager
from core.world_model import WorldModel
from .base import AgentResponse, BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)
DEFAULT_MODEL_PATH = "./models/smollm2-360m-trained-slm"


class PromptOptimizerAgent(BaseAgent):
    def __init__(
        self,
        name: str = "PromptOptimizer",
        model_path: str = DEFAULT_MODEL_PATH,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        model_manager: Optional[ModelManager] = None,
    ):
        super().__init__(name, model_path, "", temperature, max_tokens)
        self.model_manager = model_manager or ModelManager(model_path=model_path)
        self._model_loaded = model_manager is not None

        # Native World Model Engine for prompt optimization
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
                logger.info("PromptOptimizerAgent model loaded successfully")
            except Exception as e:
                logger.error(f"PromptOptimizerAgent failed to load model: {e}")
                raise

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None
    ) -> AgentResponse:
        try:
            self._ensure_model_loaded()
            current_prompt = context.get("current_prompt", "") if context else ""
            critique = context.get("critique", "") if context else ""
            scores = context.get("scores", {}) if context else {}

            optimize_prompt = f"""Analyze the current system prompt and critique, then improve it.

Current System Prompt:
{current_prompt}

Critique from evaluator:
{critique}

Current Scores:
- Accuracy: {scores.get('accuracy', 'N/A')}
- Clarity: {scores.get('clarity', 'N/A')}
- Completeness: {scores.get('completeness', 'N/A')}
- Overall: {scores.get('overall', 'N/A')}

Provide an improved system prompt that addresses the critique and aims to improve the scores.
Focus on:
1. Adding specific instructions to improve weak areas
2. Clarifying ambiguous requirements
3. Adding constraints for better accuracy
4. Ensuring comprehensive coverage

Output ONLY the improved system prompt, nothing else."""

            # Use world model to generate improved prompt
            thoughts = await self.world_model.think(optimize_prompt)
            scenarios = await self.world_model.imagine_scenarios(optimize_prompt)

            response_data = await self.world_model.generate_response(
                prompt=optimize_prompt,
                scenarios=scenarios,
                thoughts=thoughts,
                model_manager=self.model_manager
            )

            content = response_data["content"]

            # If response is too short or empty, generate a sensible default
            if not content or len(content) < 20:
                content = self._generate_default_improvement(current_prompt, critique, scores)

            logger.info("Prompt optimizer improved system prompt")

            return AgentResponse(
                content=content,
                metadata={
                    "model": self.model,
                    "thought_steps": len(thoughts),
                    "scenarios_explored": len(scenarios),
                    "native_world_model": True
                }
            )
        except Exception as e:
            logger.error(f"Prompt optimizer error: {e}")
            current_prompt = context.get("current_prompt", "") if context else ""
            critique = context.get("critique", "") if context else ""
            scores = context.get("scores", {}) if context else {}
            fallback = self._generate_default_improvement(current_prompt, critique, scores)
            return AgentResponse(
                content=fallback,
                metadata={
                    "model": self.model,
                    "fallback": True,
                    "error": str(e)
                }
            )

    def _generate_default_improvement(self, current_prompt: str, critique: str, scores: dict) -> str:
        """Generate a default improved prompt based on critique."""
        base = current_prompt if current_prompt else "You are a helpful AI assistant."

        improvements = []
        if scores.get("accuracy", 5.0) < 7.0:
            improvements.append("Always verify facts before stating them. Provide specific details and cite reasoning.")
        if scores.get("clarity", 5.0) < 7.0:
            improvements.append("Structure your responses clearly with headings, bullet points, or numbered steps.")
        if scores.get("completeness", 5.0) < 7.0:
            improvements.append("Ensure comprehensive coverage by addressing all aspects of the question thoroughly.")

        if not improvements:
            improvements.append("Maintain high standards of accuracy, clarity, and completeness in all responses.")

        return f"""{base}

Additional instructions:
{chr(10).join('- ' + imp for imp in improvements)}

Critique to address: {critique[:200] if critique else 'None'}"""

    def _build_system_prompt(self) -> str:
        return """You are a Prompt Optimizer agent specializing in improving system prompts.
Your goal is to analyze critiques and enhance prompts to achieve better outputs."""

    async def close(self) -> None:
        if self.model_manager:
            await self.model_manager.unload_model()

