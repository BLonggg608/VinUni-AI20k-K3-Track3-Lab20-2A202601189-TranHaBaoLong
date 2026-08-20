# Multi-Agent Research System — Benchmark Report

_Generated: 2026-08-20_
_Query: "What is GraphRAG?"_

## Summary

This report compares a **single-agent baseline** (one LLM call for synthesis)
against a **multi-agent workflow** (Supervisor → Researcher → Analyst → Writer)
on the same benchmark query.

## Metrics Explanation

| Metric | Description |
|---|---|
| Latency (s) | Wall-clock time from query submission to final answer |
| Cost (USD) | Estimated token cost based on gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output) |
| Quality | Heuristic score (0–10) based on answer length, sources, and analysis |
| Citation cov. | Fraction of retrieved sources actually cited in the answer |
| Iterations | Number of supervisor routing steps |

## Results

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---|---:|---:|---:|---:|---|
| baseline | 10.69 | 0.0004 | 5.0 | 0% | 0% | tokens_in=64, tokens_out=598 |
| multi-agent | 20.01 | 0.0012 | 8.0 | 60% | 0% | tokens_in=2164, tokens_out=1430, sources=5 |

### Route Trace (multi-agent)

```
supervisor → researcher → supervisor → analyst → supervisor → writer
(6 steps total)
```

- **Researcher**: searched Wikipedia → 5 sources found, research notes generated
- **Analyst**: processed notes, identified limitations and gaps
- **Writer**: synthesized final answer with citations [1][2][3]

## Analysis

### Latency

Multi-agent is **~1.9× slower** than the baseline (20.0s vs 10.7s) because it makes
3 separate LLM calls (researcher + analyst + writer) plus supervisor overhead and
Wikipedia search latency (~7s for the search call).

**Trade-off**: The extra latency is acceptable when the quality improvement justifies it.

### Cost

Multi-agent uses **~3× more tokens** ($0.0012 vs $0.0004), driven by:
- Researcher's LLM call (synthesizing notes from sources)
- Analyst's LLM call (structured analysis)
- Writer's LLM call (final synthesis with citations)

Baseline uses only 64 input + 598 output tokens for a single call.

### Quality

Multi-agent scores **8/10** vs baseline **5/10** on the heuristic scale:
- Baseline: short answer (598 tokens), no sources
- Multi-agent: longer structured answer (1430 tokens), 5 sources, citations

Multi-agent produces a more thorough, well-structured answer with sections and
explicit citations.

### Citation Coverage

- **Baseline**: 0% — no citations, no sources referenced
- **Multi-agent**: 60% — 3 of 5 Wikipedia sources cited inline as [1][2][3]

The writer cited most authoritative sources, though not all 5. This is a
prompt-engineering gap — a stronger citation instruction could improve this.

## Failure Mode Analysis

No failures occurred during this benchmark run. However, common failure modes
and mitigations are documented:

| Failure Mode | Cause | Mitigation |
|---|---|---|
| No sources found | Wikipedia/Tavily returns empty for niche queries | Add Tavily API key for better coverage |
| LLM timeout / rate limit | OpenAI throttles | Use `tenacity` for retry with backoff |
| Max iterations reached without answer | Routing loops | Verify supervisor policy; increase `max_iterations` |
| Citation coverage < 100% | Writer misses some sources | Prompt: "cite ALL sources by number" |

## When to Use Multi-Agent

Multi-agent is worth the extra latency and cost when:

- The query requires **multiple distinct capabilities** (search + analysis + writing)
- **Traceability** matters — you need to see which agent produced which output
- **Quality over speed** — structured, multi-step reasoning outperforms single-call
- **Debugging** is important — intermediate state (`research_notes`, `analysis_notes`)
  makes it easy to locate where a wrong answer was generated

## When NOT to Use Multi-Agent

Single-agent is better when:

- **Latency is critical** — any user-facing application needing sub-second responses
- **Simple, factual queries** — "What is X?" requires no multi-step reasoning
- **Cost is tightly constrained** — multi-agent uses 3× more tokens per query
- **A single well-crafted prompt achieves the same quality** — don't add
  orchestration complexity if it's not needed

## Conclusion

For the query "What is GraphRAG?", the multi-agent workflow produces a **substantially
better answer** — structured, sourced, and cited — at the cost of **3× more tokens
and 2× latency**. For production research assistants, this trade-off is often
worthwhile. For lightweight Q&A, a well-prompted single agent remains the
right default.
