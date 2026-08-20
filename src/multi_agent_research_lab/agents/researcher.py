"""Researcher agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        logger.info("Researcher searching for: %s", query)

        # Search for sources
        search_client = SearchClient()
        sources = search_client.search(query, max_sources=max_sources)

        state.sources = sources
        state.add_trace_event("researcher_sources", {"count": len(sources)})

        # Build context for LLM to synthesize notes
        source_context = "\n\n".join(
            f"[{i+1}] {s.title}: {s.snippet}"
            + (f" ({s.url})" if s.url else "")
            for i, s in enumerate(sources)
        )

        system_prompt = (
            "You are a research assistant. Given the user's query and retrieved sources, "
            "write concise research notes that:\n"
            "1. Directly address the query\n"
            "2. Synthesize information across sources\n"
            "3. Flag conflicting information between sources\n"
            "4. Note the most reliable/authoritative sources\n"
            "Be factual and cite which source(s) each claim comes from."
        )

        user_prompt = (
            f"User query: {query}\n\nRetrieved sources:\n{source_context}\n\n"
            "Write research notes addressing the query."
        )

        llm = LLMClient(temperature=0.2)
        response = llm.complete(system_prompt, user_prompt)

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(agent="researcher", content=response.content, metadata={})
        )
        state.add_trace_event(
            "researcher_llm",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )

        logger.info("Researcher produced %d sources and notes", len(sources))
        return state
