from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Fix OpenBLAS memory allocation error on Windows
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)


class MemoryItem(BaseModel):
    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def add(self, items: list[MemoryItem]) -> None:
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass


class FAISSVectorStore(VectorStore):
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.items: list[MemoryItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self._index = None
        self._load_embedding_model()
    
    def _load_embedding_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.embedding_model}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using mock embeddings")
            self._model = None
    
    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        if self._model:
            return self._model.encode(texts, show_progress_bar=False)
        return np.random.rand(len(texts), self.dimension).astype(np.float32)
    
    def add(self, items: list[MemoryItem]) -> None:
        if not items:
            return
        
        texts = [item.content for item in items]
        embeddings = self._get_embeddings(texts)
        
        for i, item in enumerate(items):
            item.embedding = embeddings[i].tolist()
            self.items.append(item)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self._build_index()
        logger.info(f"Added {len(items)} items to vector store")
    
    def _build_index(self) -> None:
        try:
            import faiss
            if self.embeddings is not None:
                dim = self.embeddings.shape[1]
                self._index = faiss.IndexFlatL2(dim)
                self._index.add(self.embeddings)
        except ImportError:
            logger.warning("faiss not installed, search will use simple matching")
            self._index = None
    
    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        if not self.items or self._index is None:
            return []
        
        query_embedding = self._get_embeddings([query])
        
        try:
            import faiss
            distances, indices = self._index.search(query_embedding, min(top_k, len(self.items)))
            
            results = []
            for idx in indices[0]:
                if idx < len(self.items):
                    results.append(self.items[idx])
            return results
        except ImportError:
            return self.items[:top_k]
    
    def save(self, path: str) -> None:
        import json
        from pathlib import Path
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "items": [item.model_dump() for item in self.items],
            "embedding_model": self.embedding_model,
        }
        
        with open(save_path / "memory.json", "w") as f:
            json.dump(data, f, indent=2)
        
        if self.embeddings is not None:
            np.save(save_path / "embeddings.npy", self.embeddings)
        
        logger.info(f"Saved {len(self.items)} items to {path}")
    
    def load(self, path: str) -> None:
        import json
        from pathlib import Path
        
        load_path = Path(path)
        
        if not load_path.exists():
            logger.warning(f"Memory path {path} does not exist")
            return
        
        data_file = load_path / "memory.json"
        if data_file.exists():
            with open(data_file) as f:
                data = json.load(f)
            
            self.items = [MemoryItem(**item) for item in data["items"]]
            self.embedding_model = data.get("embedding_model", self.embedding_model)
        
        embeddings_file = load_path / "embeddings.npy"
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
            self._build_index()
        
        logger.info(f"Loaded {len(self.items)} items from {path}")
    
    def clear(self) -> None:
        self.items = []
        self.embeddings = None
        self._index = None
        logger.info("Cleared vector store")


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./data/chroma",
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.items: list[MemoryItem] = []
        self._client = None
        self._collection = None
        self._load_client()
    
    def _load_client(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.Client(Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            ))
            self._collection = self._client.get_or_create_collection("memory")
            logger.info("Loaded ChromaDB client")
        except ImportError:
            logger.warning("chromadb not installed")
    
    def add(self, items: list[MemoryItem]) -> None:
        if not items or self._collection is None:
            return
        
        for item in items:
            self._collection.add(
                ids=[item.id],
                documents=[item.content],
                metadatas=[item.metadata],
            )
            self.items.append(item)
        
        logger.info(f"Added {len(items)} items to ChromaDB")
    
    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        if not self._collection:
            return []
        
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.items)),
        )
        
        items = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                items.append(MemoryItem(
                    id=results["ids"][0][i],
                    content=doc,
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                ))
        
        return items
    
    def save(self, path: str) -> None:
        import json
        from pathlib import Path
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "items": [item.model_dump() for item in self.items],
            "embedding_model": self.embedding_model,
        }
        
        with open(save_path / "memory.json", "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.items)} items to {path}")
    
    def load(self, path: str) -> None:
        import json
        from pathlib import Path
        
        load_path = Path(path)
        
        if not load_path.exists():
            return
        
        data_file = load_path / "memory.json"
        if data_file.exists():
            with open(data_file) as f:
                data = json.load(f)
            
            self.items = [MemoryItem(**item) for item in data["items"]]
            self.embedding_model = data.get("embedding_model", self.embedding_model)
            
            if self._collection and self.items:
                self._collection.delete(ids=[item.id for item in self.items])
                self.add(self.items)
        
        logger.info(f"Loaded {len(self.items)} items from {path}")
    
    def clear(self) -> None:
        if self._collection:
            self._collection.delete(ids=[item.id for item in self.items])
        self.items = []


def create_vector_store(
    backend: str = "faiss",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str = "./data/memory",
) -> VectorStore:
    if backend.lower() == "chroma":
        return ChromaVectorStore(embedding_model, persist_directory)
    return FAISSVectorStore(embedding_model)