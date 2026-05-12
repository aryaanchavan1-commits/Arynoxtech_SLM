from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

# Fix OpenBLAS memory allocation error on Windows
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'

from utils.logger import get_logger

if TYPE_CHECKING:
    from .generator import GeneratorAgent
    from .critic import CriticAgent
    from .optimizer import PromptOptimizerAgent

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    iteration: int
    output: str
    critique: str
    scores: dict[str, float]
    system_prompt: str
    improvement: float = 0.0


class FeedbackLoop:
    def __init__(
        self,
        generator: "GeneratorAgent",
        critic: "CriticAgent",
        optimizer: "PromptOptimizerAgent",
        max_iterations: int = 5,
        min_improvement: float = 0.05,
    ):
        self.generator = generator
        self.critic = critic
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.results: list[EvaluationResult] = []
        self.current_prompt = self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        return """You are a helpful AI assistant that provides accurate, clear, and comprehensive responses.
Always verify your information and structure your answers clearly."""
    
    async def run(
        self,
        user_query: str,
        context: Optional[dict[str, Any]] = None,
        max_iterations: Optional[int] = None,
    ) -> list[EvaluationResult]:
        """Run the feedback loop for a given query.
        
        Args:
            user_query: The user's query to process
            context: Optional context dictionary
            max_iterations: Override the default max_iterations for this run
        """
        self.results = []
        previous_score = 0.0
        
        # Use provided max_iterations or fall back to instance default
        iterations = max_iterations if max_iterations is not None else self.max_iterations
        
        logger.info(f"Starting feedback loop for query: {user_query[:100]}...")
        
        for iteration in range(1, iterations + 1):
            logger.info(f"Running iteration {iteration}")
            
            output_response = await self.generator.execute(
                f"Query: {user_query}",
                context={"user_query": user_query, ** (context or {})},
            )
            
            if not output_response.success:
                logger.error(f"Generator failed: {output_response.error}")
                continue
            
            critique_response = await self.critic.execute(
                output_response.content,
                context={"user_query": user_query},
            )
            
            if not critique_response.success:
                logger.error(f"Critic failed: {critique_response.error}")
                continue
            
            scores = critique_response.metadata.get("scores", {
                "accuracy": 5.0,
                "clarity": 5.0,
                "completeness": 5.0,
                "overall": 5.0,
            })
            
            current_score = scores.get("overall", 5.0)
            improvement = current_score - previous_score
            
            result = EvaluationResult(
                iteration=iteration,
                output=output_response.content,
                critique=critique_response.content,
                scores=scores,
                system_prompt=self.current_prompt,
                improvement=improvement,
            )
            
            self.results.append(result)
            previous_score = current_score
            
            if iteration < iterations:
                optimizer_response = await self.optimizer.execute(
                    "",
                    context={
                        "current_prompt": self.current_prompt,
                        "critique": critique_response.content,
                        "scores": scores,
                    },
                )
                
                if optimizer_response.success:
                    self.current_prompt = optimizer_response.content
            
            logger.info(
                f"Iteration {iteration}: score={current_score:.2f}, "
                f"improvement={improvement:.2f}"
            )
        
        return self.results
    
    def get_best_result(self) -> Optional[EvaluationResult]:
        if not self.results:
            return None
        
        return max(self.results, key=lambda r: r.scores.get("overall", 0))
    
    def get_score_history(self) -> list[dict[str, Any]]:
        return [
            {
                "iteration": r.iteration,
                "accuracy": r.scores.get("accuracy", 0),
                "clarity": r.scores.get("clarity", 0),
                "completeness": r.scores.get("completeness", 0),
                "overall": r.scores.get("overall", 0),
                "improvement": r.improvement,
            }
            for r in self.results
        ]
    
    def get_best_prompt(self) -> str:
        best = self.get_best_result()
        return best.system_prompt if best else self.current_prompt
