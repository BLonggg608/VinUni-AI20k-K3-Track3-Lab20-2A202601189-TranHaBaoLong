"""Supervisor / router skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routing policy:
        - If max_iterations reached → done
        - If no sources → route to researcher
        - If no analysis_notes → route to analyst
        - If no final_answer → route to writer
        - Otherwise → done
        """
        settings = get_settings()
        max_iter = settings.max_iterations

        # Record this step
        state.record_route(self.name)

        # Guard: stop if too many iterations
        if state.iteration >= max_iter:
            logger.info("Max iterations (%s) reached, stopping workflow", max_iter)
            state.record_route("done")
            return state

        # Routing decision based on missing state
        if not state.sources:
            next_route = "researcher"
        elif not state.analysis_notes:
            next_route = "analyst"
        elif not state.final_answer:
            next_route = "writer"
        else:
            next_route = "done"

        logger.info(
            "Supervisor iteration=%d routing to=%s (sources=%s analysis=%s answer=%s)",
            state.iteration,
            next_route,
            bool(state.sources),
            bool(state.analysis_notes),
            bool(state.final_answer),
        )

        state.record_route(next_route)
        return state
