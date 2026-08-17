from datetime import datetime
from langgraph.types import interrupt
from backend.db.database import SessionLocal
from backend.db.models import AuditLog

def approval_node(state: dict) -> dict:
    """
    Agent node to handle human approval for recommendations.
    Uses interrupt() to pause the graph for human input.
    """
    # Pause for human input
    human_input = interrupt({
        "action": "approve_recommendation",
        "store_id": state.get("store_id"),
        "product_id": state.get("product_id"),
        "recommended_qty": state.get("recommendation", {}).get("recommended_order_qty"),
        "risk_level": state.get("risk_level"),
        "reasoning": state.get("reorder_reasoning")
    })

    # When resumed, the human_input will be provided
    decision = human_input.get("decision")
    approver = human_input.get("approver")
    approved = decision == "approved"

    # Update audit log
    recommendation_id = state.get("recommendation_id")
    if recommendation_id:
        db = SessionLocal()
        audit_entry = db.query(AuditLog).filter(AuditLog.id == recommendation_id).first()
        if audit_entry:
            audit_entry.decision = decision
            audit_entry.approver = approver
            audit_entry.decision_timestamp = datetime.utcnow()
            db.commit()
        db.close()

    return {
        "approved": approved,
        "approver": approver
    }
