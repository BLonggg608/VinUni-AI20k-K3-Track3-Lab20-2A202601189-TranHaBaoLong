"""Analyst agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        if not state.research_notes:
            state.errors.append("Analyst: no research_notes to analyze")
            return state

        logger.info("Analyst processing research notes")

        system_prompt = (
            "You are a critical analyst. Given research notes and sources, produce a "
            "structured analysis that:\n"
            "1. Extracts key claims and facts\n"
            "2. Compares viewpoints across sources\n"
            "3. Rates the credibility of each source (high/medium/low)\n"
            "4. Identifies gaps, contradictions, or weak evidence\n"
            "5. Highlights the most important insights for the final answer\n"
            "Be concise and use bullet points where appropriate."
        )

        source_list = "\n".join(
            f"- {s.title}" + (f": {s.url}" if s.url else "") for s in state.sources
        )

        user_prompt = (
            f"User query: {state.request.query}\n\n"
            f"Sources:\n{source_list}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            "Provide a structured analysis."
        )

        llm = LLMClient(temperature=0.1)
        response = llm.complete(system_prompt, user_prompt)

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(agent="analyst", content=response.content, metadata={})
        )
        state.add_trace_event(
            "analyst_llm",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )

        logger.info("Analyst produced analysis notes")
        return state
