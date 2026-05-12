# Fix OpenBLAS memory allocation error on Windows - MUST be before numpy/torch
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OPENBLAS_CORETYPE'] = 'Generic'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

from .model import ModelManager
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelServer:
    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        model_name: str = "local-llm",
        max_batch_size: int = 8,
        max_sequence_length: int = 512,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length

        self.model_manager: Optional[ModelManager] = None
        self.app: Optional[FastAPI] = None

    def setup(self) -> None:
        """Initialize FastAPI app, model manager, and routes."""
        if self.app is not None:
            return

        self.app = FastAPI(
            title="Local LLM Server",
            description="Inference server for custom-trained LLM with world model capabilities",
            version="2.0.0"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.model_manager = ModelManager(
            model_path=self.model_path,
            max_batch_size=self.max_batch_size,
            max_sequence_length=self.max_sequence_length,
        )

        router = self.create_routes()
        self.app.include_router(router)

        logger.info("Model server setup completed")

    def create_routes(self) -> Any:
        """Create API routes for the model server."""
        from fastapi import APIRouter

        router = APIRouter()

        @router.get("/")
        async def health_check() -> Dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy", "model": self.model_name}

        @router.post("/generate")
        async def generate_text(request: GenerateRequest) -> JSONResponse:
            """Generate text based on input prompt."""
            try:
                response = await self.model_manager.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    do_sample=request.do_sample
                )
                return JSONResponse(content={"response": response})
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @router.post("/chat")
        async def chat(request: ChatRequest) -> JSONResponse:
            """Chat completion endpoint."""
            try:
                response = await self.model_manager.chat_completion(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    do_sample=request.do_sample
                )
                return JSONResponse(content={"response": response})
            except Exception as e:
                logger.error(f"Chat error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @router.get("/status")
        async def get_status() -> Dict[str, Any]:
            """Get model status and metrics."""
            if self.model_manager is None:
                return {"status": "not_initialized", "loaded": False}
            return await self.model_manager.get_status()

        @router.on_event("startup")
        async def startup_event():
            if self.model_manager:
                await self.model_manager.load_model()
                logger.info("Model loaded on server startup")

        @router.on_event("shutdown")
        async def shutdown_event():
            if self.model_manager:
                await self.model_manager.unload_model()
                logger.info("Model unloaded on server shutdown")

        return router

    async def start(self) -> None:
        """Start the model server."""
        if not self.app:
            raise RuntimeError("Server not setup properly. Call setup() first.")

        logger.info(f"Starting model server on {self.host}:{self.port}")
        import uvicorn
        from uvicorn.config import Config

        # Uvicorn cannot be run inside existing asyncio loop
        config = Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def shutdown(self) -> None:
        """Shutdown the model server."""
        logger.info("Shutting down model server")
        if self.model_manager:
            await self.model_manager.unload_model()
        logger.info("Model server shutdown completed")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

if __name__ == "__main__":
    async def run_main():
        server = ModelServer(
            model_path="./models/tinyllama-trained-slm",
            host="0.0.0.0",
            port=8000
        )
        server.setup()
        await server.start()

    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        pass

