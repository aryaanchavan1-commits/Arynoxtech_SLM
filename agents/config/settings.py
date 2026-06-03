import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class WorldModelConfig(BaseModel):
    model_path: str = "./models/world_model_v1"
    imagination_depth: int = 3
    thinking_steps: int = 5
    enable_world_simulation: bool = True
    temperature: float = 0.7
    max_tokens: int = 4096
    confidence_threshold: float = 0.7
    use_semantic_embeddings: bool = True
    enable_ppo: bool = True
    enable_grpo: bool = True


class MemoryConfig(BaseModel):
    vector_db: str = "faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = "./data/memory"
    top_k: int = 5


class DocumentProcessingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 128
    max_file_size_mb: int = 50
    supported_extensions: list[str] = [".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"]


class WebSearchConfig(BaseModel):
    enabled: bool = False
    serpapi_key: Optional[str] = None
    scraper_timeout: int = 30


class EvaluationConfig(BaseModel):
    max_iterations: int = 5
    min_improvement: float = 0.05
    accuracy_weight: float = 0.4
    clarity_weight: float = 0.3
    completeness_weight: float = 0.3


class ModelConfig(BaseModel):
    model_path: str = "./models/smollm2-360m-trained-slm"
    device: str = "auto"
    max_batch_size: int = 8
    max_sequence_length: int = 512
    kv_cache_enabled: bool = True
    fallback_models: list[str] = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
    ]
    auto_download: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_file: str = "./logs/agent_system.log"


class Settings(BaseSettings):
    world_model: WorldModelConfig = WorldModelConfig()
    memory: MemoryConfig = MemoryConfig()
    document_processing: DocumentProcessingConfig = DocumentProcessingConfig()
    web_search: WebSearchConfig = WebSearchConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    model: ModelConfig = ModelConfig()
    logging: LoggingConfig = LoggingConfig()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()
