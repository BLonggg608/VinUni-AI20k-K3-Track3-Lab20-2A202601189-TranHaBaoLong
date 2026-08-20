"""Writer agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        if not state.analysis_notes:
            state.errors.append("Writer: no analysis_notes to synthesize")
            return state

        logger.info("Writer synthesizing final answer")

        system_prompt = (
            "You are a clear, authoritative writer. Given research notes, analysis, and "
            "sources, produce a well-structured final answer that:\n"
            "1. Directly answers the user's query\n"
            "2. Cites sources using [1], [2], etc. inline\n"
            "3. Is accessible to the stated audience\n"
            "4. Acknowledges limitations or conflicting evidence\n"
            "Format: start with a concise answer, then expand with evidence."
        )

        source_list = "\n".join(
            f"[{i+1}] {s.title}" + (f" — {s.url}" if s.url else "")
            for i, s in enumerate(state.sources)
        )

        user_prompt = (
            f"User query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Sources:\n{source_list}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis:\n{state.analysis_notes}\n\n"
            "Write the final answer with inline citations."
        )

        llm = LLMClient(temperature=0.4)
        response = llm.complete(system_prompt, user_prompt)

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(agent="writer", content=response.content, metadata={})
        )
        state.add_trace_event(
            "writer_llm",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )

        logger.info("Writer produced final answer (%d chars)", len(response.content))
        return state
