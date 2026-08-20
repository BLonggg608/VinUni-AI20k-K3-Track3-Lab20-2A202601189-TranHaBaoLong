"""LangGraph workflow skeleton."""

import logging

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

AGENT_NAMES = {"supervisor", "researcher", "analyst", "writer", "done"}


def _supervisor_node(state: ResearchState) -> ResearchState:
    return SupervisorAgent().run(state)


def _researcher_node(state: ResearchState) -> ResearchState:
    return ResearcherAgent().run(state)


def _analyst_node(state: ResearchState) -> ResearchState:
    return AnalystAgent().run(state)


def _writer_node(state: ResearchState) -> ResearchState:
    return WriterAgent().run(state)


def _route(state: ResearchState) -> str:
    """Determine next node based on route_history tail."""
    history = state.route_history
    if not history:
        return "supervisor"
    return history[-1]


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def build(self) -> StateGraph:
        """Create a LangGraph StateGraph with conditional routing."""
        builder = StateGraph(ResearchState)

        # Add nodes
        builder.add_node("supervisor", _supervisor_node)
        builder.add_node("researcher", _researcher_node)
        builder.add_node("analyst", _analyst_node)
        builder.add_node("writer", _writer_node)

        # Entry point
        builder.set_entry_point("supervisor")

        # Conditional edges from supervisor
        # After supervisor, route based on what supervisor wrote to route_history[-1]
        def _supervisor_route(state: ResearchState) -> str:
            history = state.route_history
            if not history:
                return "supervisor"
            last = history[-1]
            if last in {"researcher", "analyst", "writer"}:
                return last
            # done or unknown → end
            return END

        builder.add_conditional_edges("supervisor", _supervisor_route)

        # After each worker, always return to supervisor for next decision
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", END)

        return builder

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        graph = self.build()
        compiled = graph.compile()
        logger.info("Starting multi-agent workflow")
        result_dict = compiled.invoke(state)
        # Convert dict back to Pydantic model
        result = ResearchState.model_validate(result_dict)
        logger.info(
            "Workflow complete. iterations=%d, route_history=%s",
            result.iteration,
            result.route_history,
        )
        return result
