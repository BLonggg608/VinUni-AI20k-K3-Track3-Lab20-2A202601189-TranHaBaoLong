"""Optional critic agent skeleton for bonus work."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        if not state.final_answer:
            state.errors.append("Critic: no final_answer to review")
            return state

        logger.info("Critic reviewing final answer")

        system_prompt = (
            "You are a critical reviewer. Given a final answer and its sources, "
            "evaluate the answer on:\n"
            "1. Factual accuracy — are claims supported by sources?\n"
            "2. Citation coverage — are all major claims cited?\n"
            "3. Hallucination risk — any unsupported or exaggerated claims?\n"
            "4. Completeness — does it fully address the user's query?\n"
            "Be brief. List any concerns as bullet points."
        )

        source_list = "\n".join(
            f"[{i+1}] {s.title}: {s.snippet[:150]}"
            + (f" — {s.url}" if s.url else "")
            for i, s in enumerate(state.sources)
        )

        user_prompt = (
            f"User query: {state.request.query}\n\n"
            f"Sources:\n{source_list}\n\n"
            f"Final answer:\n{state.final_answer}\n\n"
            "Critique the answer."
        )

        llm = LLMClient(temperature=0.1)
        response = llm.complete(system_prompt, user_prompt)

        state.agent_results.append(
            AgentResult(
                agent="critic",
                content=response.content,
                metadata={"review": response.content},
            )
        )
        state.add_trace_event(
            "critic_llm",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )

        logger.info("Critic completed review")
        return state
