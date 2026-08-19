"""
Shared Langfuse observability helper.

Uses the Langfuse core Python SDK directly (not the LangChain CallbackHandler,
which has a version incompatibility with langgraph >=1.2).

Provides:
- get_langfuse_client(): singleton Langfuse client, or None if disabled.
- traced_ollama_call(): wraps a raw httpx.post Ollama call in a Langfuse
  generation span with real prompt, completion (capped at ~2000 chars),
  real token counts (from Ollama's response, never fabricated), and real
  latency.  On failure, closes the span as an explicit error and re-raises.
- create_trace(): creates a Langfuse trace for a pipeline run.
- add_trace_event(): attaches a named event (e.g. approval decision) to
  an existing trace.

Design rules:
1. Never fabricate a metric — omit what isn't real, never invent it.
2. Never swallow errors — failed LLM calls close as error spans, then re-raise.
3. Never call flush() per-request — let the SDK's background thread batch.
   flush() is called only once, at FastAPI shutdown, via shutdown_langfuse().
"""

import time
import logging
from typing import Optional, Callable, Any

from backend.config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    OBSERVABILITY_ENABLED,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)

# Cap logged completion text to avoid bloated payloads.
# This is a size cap on real data, not a truthfulness change.
_MAX_COMPLETION_CHARS = 2000

# ---------------------------------------------------------------------------
# Singleton Langfuse client
# ---------------------------------------------------------------------------
_langfuse_client = None


def get_langfuse_client():
    """Return the singleton Langfuse client, or None if observability is off."""
    global _langfuse_client
    if not OBSERVABILITY_ENABLED:
        return None
    if _langfuse_client is None:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        logger.info("Langfuse observability enabled — client initialized.")
    return _langfuse_client


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

def create_trace(
    session_id: str,
    name: str = "agent_pipeline",
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[dict] = None
):
    """
    Create a new Langfuse trace tied to *session_id* (typically thread_id).
    Returns the trace object, or None if observability is off.
    """
    client = get_langfuse_client()
    if client is None:
        return None
    return client.trace(
        name=name,
        session_id=session_id,
        input=input,
        output=output,
        metadata=metadata or {},
    )


def add_trace_event(trace, event_name: str, metadata: Optional[dict] = None):
    """
    Attach a named event to an existing trace (e.g. 'approval_decision').
    No-op if trace is None.
    """
    if trace is None:
        return
    trace.event(name=event_name, metadata=metadata or {})


def flush_langfuse():
    """Flush pending events to Langfuse Cloud."""
    client = get_langfuse_client()
    if client is not None:
        client.flush()


# ---------------------------------------------------------------------------
# traced_ollama_call — the core instrumentation wrapper
# ---------------------------------------------------------------------------

def traced_ollama_call(
    prompt: str,
    session_id: str,
    span_name: str,
    call_fn: Callable[[], Any],
    trace=None,
):
    """
    Wrap *call_fn* (an httpx.post call to Ollama) in a Langfuse generation span.

    Parameters
    ----------
    prompt : str
        The real prompt sent to Ollama.
    session_id : str
        Used as the Langfuse session_id (typically thread_id or request ID).
    span_name : str
        Human-readable name for the generation (e.g. 'forecast_explanation',
        'intent_parsing').
    call_fn : callable
        A zero-arg callable that performs the actual httpx.post and returns
        the httpx.Response.  This function wraps it; on success it extracts
        the completion and token counts from Ollama's JSON response.
    trace : optional
        An existing Langfuse trace to nest this generation under.
        If None, a standalone trace is created (or skipped if disabled).

    Returns
    -------
    httpx.Response
        The raw response from call_fn, so the caller can continue its
        existing parsing logic unchanged.

    Raises
    ------
    Any exception from call_fn — re-raised after closing the span as an error.
    """
    client = get_langfuse_client()

    # --- Observability off: just call through, zero overhead ---
    if client is None:
        return call_fn()

    # --- Create or reuse trace ---
    if trace is None:
        trace = client.trace(name=span_name, session_id=session_id)

    # Open generation span
    generation = trace.generation(
        name=span_name,
        model=OLLAMA_MODEL,
        input=prompt,
    )

    start_time = time.perf_counter()

    try:
        response = call_fn()
        elapsed_s = time.perf_counter() - start_time

        # Parse Ollama's JSON for real completion and real token counts
        try:
            body = response.json()
        except Exception:
            body = {}

        completion_text = body.get("response", "")
        # Cap completion text to avoid bloated payloads (real data, not a truthfulness change)
        if len(completion_text) > _MAX_COMPLETION_CHARS:
            completion_text = completion_text[:_MAX_COMPLETION_CHARS] + "… [truncated]"

        # Only log token counts if Ollama actually reports them — never fabricate
        usage = {}
        if "prompt_eval_count" in body:
            usage["input"] = body["prompt_eval_count"]
        if "eval_count" in body:
            usage["output"] = body["eval_count"]

        generation.end(
            output=completion_text,
            usage=usage if usage else None,
            metadata={"latency_s": round(elapsed_s, 3)},
        )

        return response

    except Exception as exc:
        elapsed_s = time.perf_counter() - start_time
        # Close span as an explicit error with the real exception
        generation.end(
            output=None,
            level="ERROR",
            status_message=f"{type(exc).__name__}: {exc}",
            metadata={"latency_s": round(elapsed_s, 3)},
        )
        raise  # Re-raise so the caller's existing fallback logic runs unchanged


# ---------------------------------------------------------------------------
# Shutdown hook — called once at FastAPI shutdown, never per-request
# ---------------------------------------------------------------------------

def shutdown_langfuse():
    """Flush any pending Langfuse events.  Called once at app shutdown."""
    client = get_langfuse_client()
    if client is not None:
        logger.info("Flushing Langfuse client on shutdown…")
        client.flush()
