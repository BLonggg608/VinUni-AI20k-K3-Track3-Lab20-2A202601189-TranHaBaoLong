"""Benchmark report rendering."""

from datetime import datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a detailed markdown report."""
    lines = [
        "# Multi-Agent Research System — Benchmark Report",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Summary",
        "",
        "This report compares a **single-agent baseline** (one LLM call for search + synthesis) "
        "against a **multi-agent workflow** (Supervisor → Researcher → Analyst → Writer) on the "
        "same benchmark queries.",
        "",
        "## Metrics Explanation",
        "",
        "| Metric | Description |",
        "|---|---|",
        "| Latency (s) | Wall-clock time from query submission to final answer |",
        "| Cost (USD) | Estimated token cost based on gpt-4o-mini pricing |",
        "| Quality | Heuristic score (0–10) based on answer length, sources, and analysis |",
        "| Citation cov. | Fraction of retrieved sources actually cited in the answer |",
        "| Failure rate | Fraction of runs that raised an error |",
        "",
        "## Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.append("")
    lines.append("## Analysis")

    # Compute aggregate stats
    multi_metrics = [m for m in metrics if "multi" in m.run_name.lower()]
    baseline_metrics = [m for m in metrics if "baseline" in m.run_name.lower()]

    if multi_metrics and baseline_metrics:
        avg_multi_latency = sum(m.latency_seconds for m in multi_metrics) / len(multi_metrics)
        avg_base_latency = sum(m.latency_seconds for m in baseline_metrics) / len(baseline_metrics)

        avg_multi_cost = sum(
            m.estimated_cost_usd or 0 for m in multi_metrics
        ) / len(multi_metrics)
        avg_base_cost = sum(
            m.estimated_cost_usd or 0 for m in baseline_metrics
        ) / len(baseline_metrics)

        avg_multi_quality = sum(
            m.quality_score or 0 for m in multi_metrics
        ) / len(multi_metrics)
        avg_base_quality = sum(
            m.quality_score or 0 for m in baseline_metrics
        ) / len(baseline_metrics)

        lines.append("")
        lines.append(
            f"- **Latency**: Multi-agent {avg_multi_latency:.1f}s vs "
            f"baseline {avg_base_latency:.1f}s. "
            "Multi-agent is slower due to multiple LLM calls but produces structured output."
        )
        lines.append(
            f"- **Cost**: Multi-agent ~${avg_multi_cost:.4f} vs "
            f"baseline ~${avg_base_cost:.4f}. "
            "Multi-agent uses more tokens due to intermediate steps."
        )
        lines.append(
            f"- **Quality**: Multi-agent {avg_multi_quality:.1f}/10 vs "
            f"baseline {avg_base_quality:.1f}/10. "
            "Multi-agent benefits from dedicated research and analysis phases."
        )
        lines.append("")

    lines.append("## Failure Mode Analysis")
    lines.append("")
    lines.append(
        "Common failure modes and mitigations:\n"
    )
    lines.append(
        "1. **No sources found**: Tavily/Wikipedia may return empty results for niche queries. "
        "Mitigation: implement fallback to a different search provider or use a mock.\n"
    )
    lines.append(
        "2. **LLM timeout or rate limit**: OpenAI API may throttle. "
        "Mitigation: add retry logic with `tenacity` (already in dependencies).\n"
    )
    lines.append(
        "3. **Max iterations reached without final answer**: Routing logic may loop. "
        "Mitigation: verify supervisor routing policy and increase `max_iterations` if needed.\n"
    )
    lines.append(
        "4. **Citation coverage < 100%**: Writer may not reference all sources. "
        "Mitigation: prompt engineer the writer system prompt to explicitly cite all sources.\n"
    )
    lines.append("")
    lines.append("## When to Use Multi-Agent")
    lines.append("")
    lines.append(
        "- Complex queries requiring specialized tools (search + analysis + writing)\n"
        "- Tasks where separating concerns improves quality or debuggability\n"
        "- When trace/explainability of intermediate steps is required\n"
    )
    lines.append("")
    lines.append("## When NOT to Use Multi-Agent")
    lines.append("")
    lines.append(
        "- Simple, single-step queries (e.g., 'What is X?')\n"
        "- Latency-critical applications (multi-agent adds overhead)\n"
        "- Cost-sensitive applications (each agent call costs tokens)\n"
        "- When a single well-crafted prompt achieves the same quality\n"
    )
    return "\n".join(lines) + "\n"
