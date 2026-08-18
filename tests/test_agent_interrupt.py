"""
test_agent_interrupt.py — Unit tests for LangGraph native interrupt() and Command(resume=...) pattern.
"""
import uuid
import pytest
from backend.agents.graph import agent_app, run_agent_pipeline, resume_agent
from backend.db.database import SessionLocal, init_db
from backend.db.models import AuditLog

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()

def test_langgraph_interrupt_and_approve_resume():
    """Test starting the graph, verifying it pauses at interrupt, and resuming with approval."""
    thread_id = f"test_thread_approve_{uuid.uuid4().hex[:8]}"
    store_id = "S001"
    product_id = "P001"

    # 1. Start graph execution -> reaches interrupt in approval_node
    paused_state = run_agent_pipeline(store_id, product_id, thread_id)
    
    # State should have computed forecast and recommendation
    assert paused_state.get("store_id") == store_id
    assert paused_state.get("product_id") == product_id
    assert "recommendation" in paused_state
    assert "risk_level" in paused_state

    # Verify checkpointer state indicates interrupted/pending
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent_app.get_state(config)
    assert snapshot is not None
    # Interrupt was reached
    assert len(snapshot.tasks) > 0 or snapshot.next == ("approval_node",)

    # 2. Resume graph with approval
    approver_name = "SupplyChainLead"
    resumed_state = resume_agent(thread_id, decision="approved", approver=approver_name)

    assert resumed_state.get("approved") is True
    assert resumed_state.get("approver") == approver_name
    assert resumed_state.get("decision") == "approved"

    # 3. Verify SQLite audit log reflects approval
    db = SessionLocal()
    audit_entry = db.query(AuditLog).filter(AuditLog.thread_id == thread_id).first()
    assert audit_entry is not None
    assert audit_entry.decision == "approved"
    assert audit_entry.approver == approver_name
    db.close()


def test_langgraph_interrupt_and_reject_resume():
    """Test starting the graph, pausing at interrupt, and resuming with rejection."""
    thread_id = f"test_thread_reject_{uuid.uuid4().hex[:8]}"
    store_id = "S002"
    product_id = "P002"

    # 1. Start graph
    paused_state = run_agent_pipeline(store_id, product_id, thread_id)
    assert paused_state.get("store_id") == store_id
    assert paused_state.get("product_id") == product_id

    # 2. Resume with rejection
    approver_name = "OperationsManager"
    resumed_state = resume_agent(thread_id, decision="rejected", approver=approver_name)

    assert resumed_state.get("approved") is False
    assert resumed_state.get("approver") == approver_name
    assert resumed_state.get("decision") == "rejected"

    # 3. Verify DB record
    db = SessionLocal()
    audit_entry = db.query(AuditLog).filter(AuditLog.thread_id == thread_id).first()
    assert audit_entry is not None
    assert audit_entry.decision == "rejected"
    assert audit_entry.approver == approver_name
    db.close()
