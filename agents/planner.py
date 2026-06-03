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


class PlannerAgent(BaseAgent):
    def __init__(
        self,
        name: str = "Planner",
        model_path: str = DEFAULT_MODEL_PATH,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        model_manager: Optional[ModelManager] = None,
    ):
        super().__init__(name, model_path, "", temperature, max_tokens)
        self.model_manager = model_manager or ModelManager(model_path=model_path)
        self._model_loaded = model_manager is not None

        # Native World Model Engine for planning
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
                logger.info("PlannerAgent model loaded successfully")
            except Exception as e:
                logger.error(f"PlannerAgent failed to load model: {e}")
                raise

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None
    ) -> AgentResponse:
        try:
            self._ensure_model_loaded()
            plan_prompt = f"""Break down the following task into clear, actionable steps.

Task: {prompt}

Provide a structured plan with numbered steps. Each step should:
- Be specific and actionable
- Have clear inputs and outputs
- Be completable in a single iteration

Format:
STEP 1: <description>
STEP 2: <description>
..."""

            # Use world model to generate plan
            thoughts = await self.world_model.think(plan_prompt)
            scenarios = await self.world_model.imagine_scenarios(plan_prompt)

            response_data = await self.world_model.generate_response(
                prompt=plan_prompt,
                scenarios=scenarios,
                thoughts=thoughts,
                model_manager=self.model_manager
            )

            content = response_data["content"]
            steps = self._parse_steps(content)

            logger.info(f"Planner created {len(steps)} steps")

            return AgentResponse(
                content=content,
                metadata={
                    "model": self.model,
                    "steps": steps,
                    "thought_steps": len(thoughts),
                    "scenarios_explored": len(scenarios),
                    "native_world_model": True
                }
            )
        except Exception as e:
            logger.error(f"Planner error: {e}")
            fallback = self._generate_default_plan(prompt)
            return AgentResponse(
                content=fallback,
                metadata={
                    "model": self.model,
                    "steps": self._parse_steps(fallback),
                    "fallback": True,
                    "error": str(e)
                }
            )

    def _generate_default_plan(self, prompt: str) -> str:
        """Generate a default plan for the given task."""
        return f"""STEP 1: Understand the requirements of the task: {prompt[:100]}
STEP 2: Research and gather relevant information
STEP 3: Analyze the information and identify key points
STEP 4: Formulate a structured response
STEP 5: Review and refine the output for accuracy and clarity
STEP 6: Deliver the final response"""

    def _build_system_prompt(self) -> str:
        return """You are a Planner agent specialized in breaking down complex tasks
into clear, actionable steps. Create structured plans that can be executed sequentially."""

    def _parse_steps(self, content: str) -> list[str]:
        steps = []
        for line in content.split("\n"):
            if "STEP" in line and ":" in line:
                steps.append(line.split(":", 1)[1].strip())
        return steps

    async def close(self) -> None:
        if self.model_manager:
            await self.model_manager.unload_model()

