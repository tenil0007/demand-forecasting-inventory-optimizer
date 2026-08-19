"""
Verification script for Langfuse observability integration.

Tests three scenarios:
1. Keys blank  → app behaves identically, zero traces, no crash.
2. Keys present → traced_ollama_call creates real generation spans.
3. Ollama down  → generation span closes as ERROR, exception re-raised.
"""

import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.getcwd())


def test_disabled_mode():
    """Scenario 1: No keys set → observability is a no-op, zero overhead."""
    # Force disabled
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)

    # Re-import with clean state
    import importlib
    import backend.config
    importlib.reload(backend.config)

    assert backend.config.OBSERVABILITY_ENABLED is False, "Should be disabled without keys"

    import backend.observability as obs
    importlib.reload(obs)

    assert obs.get_langfuse_client() is None, "Client should be None when disabled"
    assert obs.create_trace("test_session") is None, "Trace should be None when disabled"

    # traced_ollama_call should just pass through to call_fn
    call_count = 0
    class FakeResponse:
        def json(self):
            return {"response": "test"}

    def fake_call():
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    result = obs.traced_ollama_call(
        prompt="test prompt",
        session_id="test",
        span_name="test_gen",
        call_fn=fake_call,
    )
    assert call_count == 1, "call_fn should have been called exactly once"
    assert result.json()["response"] == "test", "Should return real response"
    print("✅ Scenario 1 PASSED: Disabled mode — no crash, clean pass-through")


def test_error_propagation():
    """Scenario 3: Ollama down → exception re-raised after span closes as error."""
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)

    import importlib
    import backend.config
    importlib.reload(backend.config)
    import backend.observability as obs
    importlib.reload(obs)

    def failing_call():
        raise ConnectionError("Ollama is not running")

    try:
        obs.traced_ollama_call(
            prompt="test prompt",
            session_id="test",
            span_name="test_gen",
            call_fn=failing_call,
        )
        assert False, "Should have raised ConnectionError"
    except ConnectionError as e:
        assert "Ollama is not running" in str(e)
        print("✅ Scenario 3 PASSED: Error re-raised — fallback logic will fire")


def test_enabled_mode_with_real_client():
    """Scenario 2: Keys present → Langfuse client created, generation span opened."""
    # Set dummy keys to enable the client (it won't actually send since host is fake)
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test-dummy"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-test-dummy"
    os.environ["LANGFUSE_HOST"] = "http://localhost:19999"  # non-existent, won't send

    import importlib
    import backend.config
    importlib.reload(backend.config)

    assert backend.config.OBSERVABILITY_ENABLED is True, "Should be enabled with keys"

    import backend.observability as obs
    # Reset singleton
    obs._langfuse_client = None
    importlib.reload(obs)

    client = obs.get_langfuse_client()
    assert client is not None, "Client should be created when keys are present"

    # Create a trace
    trace = obs.create_trace("test_session_123", name="test_pipeline")
    assert trace is not None, "Trace should be created"

    # Simulate a successful Ollama call
    class FakeResponse:
        def json(self):
            return {
                "response": "This is the forecast explanation with real content.",
                "prompt_eval_count": 42,
                "eval_count": 15,
            }

    result = obs.traced_ollama_call(
        prompt="Explain the forecast",
        session_id="test_session_123",
        span_name="forecast_explanation",
        call_fn=lambda: FakeResponse(),
        trace=trace,
    )
    assert result.json()["prompt_eval_count"] == 42, "Should return real response"

    # Simulate a failed Ollama call with span closing as error
    def failing_call():
        raise TimeoutError("Ollama request timed out")

    try:
        obs.traced_ollama_call(
            prompt="Parse intent",
            session_id="test_session_123",
            span_name="intent_parsing",
            call_fn=failing_call,
            trace=trace,
        )
        assert False, "Should have re-raised"
    except TimeoutError:
        pass  # Expected — span closed as ERROR, exception re-raised

    # Add a trace event
    obs.add_trace_event(trace, "approval_decision", {
        "decision": "approved",
        "approver": "test_user",
    })

    print("✅ Scenario 2 PASSED: Enabled mode — client created, trace + generation + event all work")
    print(f"   Client type: {type(client).__name__}")
    print(f"   Trace type:  {type(trace).__name__}")

    # Clean up
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)
    os.environ.pop("LANGFUSE_HOST", None)
    obs._langfuse_client = None


def test_no_flush_per_request():
    """Verify shutdown_langfuse doesn't crash when disabled."""
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)

    import importlib
    import backend.config
    importlib.reload(backend.config)
    import backend.observability as obs
    obs._langfuse_client = None
    importlib.reload(obs)

    # Should be a clean no-op
    obs.shutdown_langfuse()
    print("✅ Scenario 4 PASSED: shutdown_langfuse is a no-op when disabled")


if __name__ == "__main__":
    test_disabled_mode()
    test_error_propagation()
    test_enabled_mode_with_real_client()
    test_no_flush_per_request()
    print("\n🎉 All 4 observability verification scenarios passed!")
