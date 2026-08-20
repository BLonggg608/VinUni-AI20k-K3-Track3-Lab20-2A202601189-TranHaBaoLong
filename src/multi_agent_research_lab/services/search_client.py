"""Search client abstraction for ResearcherAgent."""

import json
import urllib.parse
import urllib.request

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily if TAVILY_API_KEY is set, otherwise falls back to Wikipedia search.
    """

    def __init__(self):
        settings = get_settings()
        self.tavily_key = settings.tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self.tavily_key:
            return self._search_tavily(query, max_results)
        return self._search_wikipedia(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        """Search using Tavily API."""

        url = "https://api.tavily.com/search"
        payload = json.dumps({"query": query, "max_results": max_results})
        headers = {
            "Authorization": f"ApiKey {self.tavily_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            url, data=payload.encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        docs = []
        for item in data.get("results", [])[:max_results]:
            docs.append(
                SourceDocument(
                    title=item.get("title", ""),
                    url=item.get("url"),
                    snippet=item.get("content", "")[:300],
                    metadata={"score": item.get("score")},
                )
            )
        return docs

    def _search_wikipedia(self, query: str, max_results: int) -> list[SourceDocument]:
        """Free fallback: search Wikipedia API."""
        encoded = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded}"
            f"&format=json&srlimit={max_results}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "MALab/1.0 (educational)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        docs = []
        for item in data.get("query", {}).get("search", [])[:max_results]:
            snippet = item.get("snippet", "")
            # Remove HTML tags from snippet
            import re
            snippet = re.sub(r"<[^>]+>", "", snippet)
            docs.append(
                SourceDocument(
                    title=item.get("title", ""),
                    url=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(item['title'])}",
                    snippet=snippet[:300],
                    metadata={"word_count": item.get("wordcount", 0)},
                )
            )
        return docs
