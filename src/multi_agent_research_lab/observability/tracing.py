"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context with optional LangSmith / Langfuse integration.

    If LANGSMITH_API_KEY is set, traces are sent to LangSmith.
    If LANGFUSE_SECRET_KEY is set, traces are sent to Langfuse.
    Otherwise falls back to a no-op span that only measures duration.
    """
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    settings = get_settings()
    langsmith_key = settings.langsmith_api_key
    langfuse_key = settings.langfuse_secret_key

    # Try LangSmith first
    if langsmith_key:
        _trace_langsmith(name, attributes, langsmith_key, settings.langsmith_project)
    elif langfuse_key:
        _trace_langfuse(name, attributes, settings)

    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


def _trace_langsmith(
    name: str,
    attributes: dict[str, Any] | None,
    api_key: str,
    project: str,
) -> None:
    """Send a lightweight trace event to LangSmith."""
    try:
        import datetime as dt
        import json
        import urllib.request

        url = "https://api.smith.langchain.com/runs"
        payload = {
            "name": name,
            "run_type": "chain",
            "inputs": attributes or {},
            "start_time": dt.datetime.now(dt.UTC).isoformat(),
            "project_name": project,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        # Fire-and-forget; don't block on tracing
        urllib.request.urlopen(req, timeout=2)
    except Exception as exc:
        logger.debug("LangSmith trace failed (non-fatal): %s", exc)


def _trace_langfuse(name: str, attributes: dict[str, Any] | None, settings: Any) -> None:
    """Send a trace event to Langfuse."""
    try:
        import datetime as dt
        import hashlib
        import json
        import time
        import urllib.request

        url = f"{settings.langfuse_host}/api/public/ingestion"
        trace_id = hashlib.sha1(f"{name}{time.time()}".encode()).hexdigest()[:16]
        payload = {
            "batch": [
                {
                    "id": trace_id,
                    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                    "name": name,
                    "input": attributes or {},
                    "output": None,
                    "metadata": {"source": "multi-agent-research-lab"},
                }
            ]
        }
        data = json.dumps(payload).encode()
        headers = {
            "Authorization": f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=2)
    except Exception as exc:
        logger.debug("Langfuse trace failed (non-fatal): %s", exc)
