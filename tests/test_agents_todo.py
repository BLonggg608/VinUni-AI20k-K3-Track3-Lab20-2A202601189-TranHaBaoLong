"""Agent routing tests.

Tests verify that supervisor routing and all agent run methods work correctly
and do not raise StudentTodoError.
"""

import pytest

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


@pytest.fixture
def initial_state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="What is GraphRAG?"))


def test_supervisor_routes_to_researcher_when_no_sources(initial_state: ResearchState) -> None:
    state = initial_state
    agent = SupervisorAgent()
    result = agent.run(state)
    # Supervisor should route to researcher first
    assert result.route_history[-1] == "researcher"


def test_supervisor_routes_to_done_when_max_iterations_reached(
    initial_state: ResearchState,
) -> None:
    state = initial_state
    state.iteration = 99  # Simulate hitting max iterations
    agent = SupervisorAgent()
    result = agent.run(state)
    # Should route to done
    assert result.route_history[-1] == "done"


def test_supervisor_routes_to_analyst_when_sources_exist(initial_state: ResearchState) -> None:
    state = initial_state
    state.sources = [SourceDocument(title="Test", url="http://test.com", snippet="test")]
    agent = SupervisorAgent()
    result = agent.run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_when_analysis_exists(initial_state: ResearchState) -> None:
    state = initial_state
    state.sources = [SourceDocument(title="Test", url="http://test.com", snippet="test")]
    state.analysis_notes = "Some analysis"
    agent = SupervisorAgent()
    result = agent.run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_to_done_when_answer_exists(initial_state: ResearchState) -> None:
    state = initial_state
    state.sources = [SourceDocument(title="Test", url="http://test.com", snippet="test")]
    state.analysis_notes = "Some analysis"
    state.final_answer = "Done"
    agent = SupervisorAgent()
    result = agent.run(state)
    assert result.route_history[-1] == "done"


def test_route_history_grows(initial_state: ResearchState) -> None:
    state = initial_state
    agent = SupervisorAgent()
    result = agent.run(state)
    assert len(result.route_history) >= 2  # supervisor + next route
