import json
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel

from memory.vector_store import MemoryItem
from utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchResult(BaseModel):
    title: str
    snippet: str
    url: str
    source: str = "web"


class WebLearner:
    def __init__(
        self,
        enabled: bool = False,
        serpapi_key: Optional[str] = None,
        vector_store: Optional[Any] = None,
    ):
        self.enabled = enabled
        self.serpapi_key = serpapi_key
        self.vector_store = vector_store
        self.session = httpx.AsyncClient(timeout=30.0)
    
    async def search(self, query: str, num_results: int = 5) -> list[WebSearchResult]:
        if not self.enabled:
            return []
        
        results = []
        
        if self.serpapi_key:
            results = await self._search_serpapi(query, num_results)
        else:
            results = await self._search_duckduckgo(query, num_results)
        
        if results and self.vector_store:
            await self._store_results(results)
        
        return results
    
    async def _search_serpapi(
        self,
        query: str,
        num_results: int = 5,
    ) -> list[WebSearchResult]:
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.serpapi_key,
                "num": num_results,
            }
            
            response = await self.session.get(url, params=params)
            data = response.json()
            
            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("link", ""),
                    source="serpapi",
                ))
            
            return results
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return []
    
    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int = 5,
    ) -> list[WebSearchResult]:
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = await self.session.get(url)
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            for result in soup.select(".result")[:num_results]:
                title_elem = result.select_one(".result__title")
                snippet_elem = result.select_one(".result__snippet")
                link_elem = result.select_one(".result__url")
                
                if title_elem:
                    results.append(WebSearchResult(
                        title=title_elem.get_text(strip=True),
                        snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                        url=link_elem.get_text(strip=True) if link_elem else "",
                        source="duckduckgo",
                    ))
            
            return results
        except ImportError:
            logger.warning("beautifulsoup4 not installed, using fallback search")
            return await self._fallback_search(query, num_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    async def _fallback_search(
        self,
        query: str,
        num_results: int = 3,
    ) -> list[WebSearchResult]:
        try:
            url = f"https://ddg-api.vercel.app/search?q={quote_plus(query)}&num={num_results}"
            
            response = await self.session.get(url)
            data = response.json()
            
            results = []
            for item in data:
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("url", ""),
                    source="ddg-api",
                ))
            
            return results
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []
    
    async def _store_results(self, results: list[WebSearchResult]) -> None:
        import uuid
        
        for result in results:
            item = MemoryItem(
                id=str(uuid.uuid4()),
                content=f"[RETRIEVED KNOWLEDGE]\nTitle: {result.title}\n{result.snippet}\nSource: {result.url}",
                metadata={
                    "type": "retrieved_knowledge",
                    "source": result.source,
                    "url": result.url,
                },
            )
            
            self.vector_store.add([item])
        
        logger.info(f"Stored {len(results)} retrieved knowledge items")
    
    def format_knowledge(self, results: list[WebSearchResult]) -> str:
        if not results:
            return ""
        
        parts = ["[RETRIEVED KNOWLEDGE FROM WEB SEARCH]"]
        
        for i, result in enumerate(results, 1):
            parts.append(f"\n{i}. {result.title}")
            parts.append(f"   {result.snippet}")
            parts.append(f"   Source: {result.url}")
        
        return "\n".join(parts)
    
    async def close(self) -> None:
        await self.session.aclose()