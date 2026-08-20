"""Benchmark skeleton for single-agent vs multi-agent."""

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return metrics.

    Captures token usage and cost from the workflow trace if available.
    """
    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:
        latency = perf_counter() - started
        logger.warning("Benchmark run '%s' failed: %s", run_name, exc)
        return (
            ResearchState(
                request={"query": query, "max_sources": 5, "audience": "technical learners"}
            ),
            BenchmarkMetrics(
                run_name=run_name,
                latency_seconds=latency,
                failure_rate=1.0,
                notes=f"Runner raised: {exc}",
            ),
        )

    latency = perf_counter() - started

    # Aggregate trace data
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cost: float | None = None

    costs = []
    input_tokens_list = []
    output_tokens_list = []
    for event in state.trace:
        payload = event.get("payload", {})
        if payload.get("cost_usd"):
            costs.append(payload["cost_usd"])
        if payload.get("input_tokens"):
            input_tokens_list.append(payload["input_tokens"])
        if payload.get("output_tokens"):
            output_tokens_list.append(payload["output_tokens"])

    if costs:
        total_cost = sum(costs)
    if input_tokens_list:
        total_input_tokens = sum(input_tokens_list)
    if output_tokens_list:
        total_output_tokens = sum(output_tokens_list)

    # Citation coverage: fraction of sources actually cited in the answer
    citation_coverage: float | None = None
    if state.final_answer and state.sources:
        cited = sum(1 for s in state.sources if s.title.lower() in state.final_answer.lower())
        citation_coverage = cited / len(state.sources)

    # Quality score: simple heuristic (can be replaced with LLM-as-judge)
    quality_score: float | None = None
    if state.final_answer:
        length = len(state.final_answer)
        has_sources = len(state.sources) > 0
        has_analysis = bool(state.analysis_notes)
        # Rough heuristic: longer answer + sources + analysis → better
        score = 0.0
        if length > 200:
            score += 3
        elif length > 100:
            score += 2
        else:
            score += 1
        if has_sources:
            score += 3
        if has_analysis:
            score += 2
        quality_score = min(score, 10.0)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=0.0,
        notes=(
            f"tokens_in={total_input_tokens}, tokens_out={total_output_tokens}, "
            f"iterations={state.iteration}, sources={len(state.sources)}"
        ),
    )
    return state, metrics
