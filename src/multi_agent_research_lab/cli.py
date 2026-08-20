"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using one LLM call."""

    _init()
    request = _parse_query(query)

    system_prompt = (
        "You are a helpful research assistant. Answer the user's question "
        "clearly and concisely. If you cite facts, note that this is a "
        "general knowledge response."
    )
    user_prompt = (
        f"Query: {request.query}\n\n"
        f"Audience: {request.audience}\n\n"
        "Provide a thorough, well-structured answer."
    )

    llm = LLMClient(temperature=0.3)
    import time
    started = time.perf_counter()
    response = llm.complete(system_prompt, user_prompt)
    elapsed = time.perf_counter() - started

    # Build state for consistency
    state = ResearchState(request=request)
    state.final_answer = response.content

    # Show metrics table
    table = Table(title="Single-Agent Baseline Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Latency", f"{elapsed:.2f}s")
    table.add_row("Input tokens", str(response.input_tokens or "N/A"))
    table.add_row("Output tokens", str(response.output_tokens or "N/A"))
    table.add_row("Est. cost", f"${response.cost_usd:.4f}" if response.cost_usd else "N/A")
    console.print(table)
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    # Show route history
    console.print(Panel.fit(
        " → ".join(result.route_history),
        title="Route History",
        style="cyan",
    ))

    # Show metrics
    table = Table(title="Multi-Agent Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Iterations", str(result.iteration))
    table.add_row("Sources found", str(len(result.sources)))
    table.add_row("Has analysis", "Yes" if result.analysis_notes else "No")

    # Sum up costs
    total_cost = sum(
        e.get("payload", {}).get("cost_usd", 0) or 0
        for e in result.trace
    )
    total_input = sum(
        e.get("payload", {}).get("input_tokens", 0) or 0
        for e in result.trace
    )
    total_output = sum(
        e.get("payload", {}).get("output_tokens", 0) or 0
        for e in result.trace
    )
    table.add_row("Total input tokens", str(total_input))
    table.add_row("Total output tokens", str(total_output))
    table.add_row("Est. total cost", f"${total_cost:.4f}")
    console.print(table)

    if result.final_answer:
        console.print(Panel.fit(result.final_answer, title="Final Answer"))
    else:
        console.print(Panel.fit(
            f"No final answer produced. Route history: {' → '.join(result.route_history)}",
            title="Warning",
            style="yellow",
        ))


if __name__ == "__main__":
    app()
