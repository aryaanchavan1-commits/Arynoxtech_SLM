import uuid
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

from .vector_store import FAISSVectorStore, MemoryItem, VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryEntry:
    user_query: str
    output: str
    critique: str
    scores: dict[str, float]
    system_prompt: str
    iteration: int
    timestamp: str = ""


class DocumentProcessor:
    """Extract text from PDF, images, and text files for RAG ingestion."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, file_path: str, file_type: Optional[str] = None) -> str:
        path = Path(file_path)
        ext = (file_type or path.suffix).lower()

        if ext in (".pdf", ".application/pdf"):
            return self._extract_pdf(file_path)
        elif ext in (".txt", ".text", ".md", ".markdown"):
            return self._extract_text(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            return self._extract_image(file_path)
        elif ext in (".docx", ".doc"):
            return self._extract_docx(file_path)
        else:
            # Try text extraction as fallback
            try:
                return self._extract_text(file_path)
            except Exception as e:
                logger.warning(f"Unsupported file type {ext}, skipping: {e}")
                return ""

    def _extract_pdf(self, file_path: str) -> str:
        text_parts = []
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"pdfplumber failed ({e}), trying PyPDF2")
            try:
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text_parts.append(page.extract_text() or "")
            except Exception as e2:
                logger.error(f"PDF extraction failed: {e2}")
        return "\n".join(text_parts)

    def _extract_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _extract_image(self, file_path: str) -> str:
        try:
            from PIL import Image
            img = Image.open(file_path)
            # Try OCR if available
            try:
                import pytesseract
                text = pytesseract.image_to_string(img)
                if text.strip():
                    return text
            except ImportError:
                pass
            # Fallback: describe image dimensions
            return f"[Image: {img.format}, {img.size[0]}x{img.size[1]}, mode={img.mode}]"
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return ""

    def _extract_docx(self, file_path: str) -> str:
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            logger.warning("python-docx not installed, cannot process .docx files")
            return ""

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence ending
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                    end = sentence_break + 1
                else:
                    word_break = text.rfind(' ', start, end)
                    if word_break != -1:
                        end = word_break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap if end < len(text) else len(text)
        return chunks


class MemoryManager:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        persist_directory: str = "./data/memory",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        self.vector_store = vector_store or FAISSVectorStore()
        self.persist_directory = persist_directory
        self.document_store: Optional[VectorStore] = None
        self.entries: list[MemoryEntry] = []
        self.document_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._document_metadata: list[dict[str, Any]] = []
        self._load_memory()

    def _get_document_store(self) -> VectorStore:
        """Lazy-init a separate vector store for documents."""
        if self.document_store is None:
            doc_path = os.path.join(self.persist_directory, "documents")
            os.makedirs(doc_path, exist_ok=True)
            self.document_store = FAISSVectorStore(
                embedding_model=getattr(self.vector_store, 'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            )
            # Try to load existing document embeddings
            try:
                self.document_store.load(doc_path)
            except Exception:
                pass
        return self.document_store

    def _load_memory(self) -> None:
        memory_path = Path(self.persist_directory)
        if memory_path.exists():
            self.vector_store.load(str(memory_path))
            logger.info("Loaded existing chat memory")

    def add_entry(
        self,
        user_query: str,
        output: str,
        critique: str,
        scores: dict[str, float],
        system_prompt: str,
        iteration: int,
    ) -> None:
        import datetime
        entry_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()

        entry = MemoryEntry(
            user_query=user_query,
            output=output,
            critique=critique,
            scores=scores,
            system_prompt=system_prompt,
            iteration=iteration,
            timestamp=timestamp,
        )

        memory_item = MemoryItem(
            id=entry_id,
            content=f"Query: {user_query}\nOutput: {output}\nCritique: {critique}\nScores: {scores}",
            metadata={
                "user_query": user_query,
                "output": output,
                "critique": critique,
                "scores": scores,
                "system_prompt": system_prompt,
                "iteration": iteration,
                "timestamp": timestamp,
            },
        )

        self.vector_store.add([memory_item])
        self.entries.append(entry)
        logger.info(f"Added memory entry for iteration {iteration}")

    def process_document(self, file_path: str, file_name: Optional[str] = None, file_type: Optional[str] = None) -> dict[str, Any]:
        """Ingest a document: extract text, chunk it, embed, and store for RAG."""
        result = {
            "file_name": file_name or os.path.basename(file_path),
            "file_type": file_type,
            "chunks_ingested": 0,
            "text_length": 0,
            "success": False,
            "error": None,
        }

        try:
            text = self.document_processor.extract_text(file_path, file_type)
            if not text.strip():
                result["error"] = "No text could be extracted from the file."
                return result

            result["text_length"] = len(text)
            chunks = self.document_processor.chunk_text(text)

            doc_store = self._get_document_store()
            items = []
            for i, chunk in enumerate(chunks):
                item_id = f"doc_{uuid.uuid4().hex[:8]}_chunk_{i}"
                items.append(MemoryItem(
                    id=item_id,
                    content=chunk,
                    metadata={
                        "source_file": result["file_name"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": file_type,
                    },
                ))

            if items:
                doc_store.add(items)
                doc_store.save(os.path.join(self.persist_directory, "documents"))
                result["chunks_ingested"] = len(items)
                result["success"] = True
                self._document_metadata.append({
                    "file_name": result["file_name"],
                    "chunks": len(items),
                    "text_length": len(text),
                })
                logger.info(f"Ingested document {result['file_name']} with {len(items)} chunks")
            else:
                result["error"] = "Document produced no chunks after processing."

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Document processing failed for {file_path}: {e}")

        return result

    def get_document_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant document chunks for a query."""
        if self.document_store is None or not getattr(self.document_store, 'items', None):
            return ""

        results = self.document_store.search(query, top_k=top_k)
        if not results:
            return ""

        parts = []
        for r in results:
            meta = r.metadata or {}
            source = meta.get("source_file", "Unknown")
            parts.append(f"[From {source}]: {r.content}")

        return "\n\n---\n\n".join(parts)

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        return self.vector_store.search(query, top_k)

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        items = self.retrieve(query, top_k)
        if not items:
            return ""

        context_parts = []
        for item in items:
            context_parts.append(
                f"Past Query: {item.metadata.get('user_query', '')}\n"
                f"Output: {item.metadata.get('output', '')}\n"
                f"Critique: {item.metadata.get('critique', '')}\n"
                f"Scores: {item.metadata.get('scores', {})}"
            )

        return "\n\n---\n\n".join(context_parts)

    def save(self) -> None:
        self.vector_store.save(self.persist_directory)
        if self.document_store is not None:
            self.document_store.save(os.path.join(self.persist_directory, "documents"))
        logger.info("Saved memory to disk")

    def get_best_prompt(self) -> Optional[tuple[str, float]]:
        if not self.entries:
            return None
        best_entry = max(self.entries, key=lambda e: e.scores.get("overall", 0))
        return (
            best_entry.system_prompt,
            best_entry.scores.get("overall", 0),
        )

    def get_statistics(self) -> dict[str, Any]:
        chat_count = len(self.entries)
        doc_count = len(self._document_metadata)
        total_doc_chunks = sum(d.get("chunks", 0) for d in self._document_metadata)

        if not self.entries:
            return {
                "total_entries": chat_count,
                "average_score": 0.0,
                "best_score": 0.0,
                "iterations": 0,
                "documents": doc_count,
                "document_chunks": total_doc_chunks,
            }

        scores = [e.scores.get("overall", 0) for e in self.entries]
        return {
            "total_entries": chat_count,
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "iterations": sum(e.iteration for e in self.entries),
            "documents": doc_count,
            "document_chunks": total_doc_chunks,
        }

    def clear(self) -> None:
        self.vector_store.clear()
        self.entries = []
        if self.document_store is not None:
            self.document_store.clear()
        self._document_metadata = []
        logger.info("Cleared memory")

