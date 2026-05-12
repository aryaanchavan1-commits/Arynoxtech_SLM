"""
FastAPI application factory for the LLM serving API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .server import ModelServer


def create_app(model_path: str = "./models", host: str = "0.0.0.0", port: int = 8000) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        model_path: Path to the trained model
        host: Server host
        port: Server port

    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="Local LLM Server",
        description="Inference server for custom-trained LLM with world model capabilities",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create model server instance
    model_server = ModelServer(
        model_path=model_path,
        host=host,
        port=port
    )

    # Include routes
    app.include_router(model_server.create_routes())

    return app