from datetime import datetime
from langgraph.types import interrupt
from backend.db.database import SessionLocal
from backend.db.models import AuditLog

def approval_node(state: dict) -> dict:
    """
    Agent node to handle human approval for recommendations.
    1. Pauses execution via interrupt() to await human review.
    2. Upon resume, writes the final audit log decision from within the graph.
    """
    store_id = state.get("store_id", "S001")
    product_id = state.get("product_id", "P001")
    rec = state.get("recommendation", {})
    recommended_qty = float(rec.get("economic_order_quantity", 0.0))
    risk_level = state.get("risk_level", "LOW")
    reasoning = state.get("reorder_reasoning", "")
    thread_id = state.get("thread_id")
    rec_id = state.get("recommendation_id")

    # Pause execution for human review via LangGraph native interrupt()
    human_input = interrupt({
        "action": "approve_recommendation",
        "thread_id": thread_id,
        "store_id": store_id,
        "product_id": product_id,
        "recommended_qty": recommended_qty,
        "risk_level": risk_level,
        "reasoning": reasoning,
        "recommendation_id": rec_id
    })

    # When resumed via Command(resume={...}), execution continues here
    decision = human_input.get("decision", "approved")
    approver = human_input.get("approver", "Human Reviewer")
    approved = (decision == "approved")

    # Resumed graph node commits final decision to audit log
    db = SessionLocal()
    log = None
    if rec_id:
        log = db.query(AuditLog).filter(AuditLog.id == rec_id).first()
    elif thread_id:
        log = db.query(AuditLog).filter(AuditLog.thread_id == thread_id).first()

    if log:
        log.decision = decision
        log.approver = approver
        log.decision_timestamp = datetime.utcnow()
        db.commit()
    db.close()

    return {
        "approved": approved,
        "approver": approver,
        "decision": decision,
        "recommendation_id": rec_id,
        "thread_id": thread_id,
        "store_id": store_id,
        "product_id": product_id,
        "recommended_qty": recommended_qty
    }
