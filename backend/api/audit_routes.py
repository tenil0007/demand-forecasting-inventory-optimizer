from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.db.models import AuditLog
from backend.agents.graph import resume_agent

router = APIRouter()

class ApproveRequest(BaseModel):
    recommendation_id: Optional[int] = None
    thread_id: Optional[str] = None
    decision: str  # "approved" or "rejected"
    approver: str

@router.get("/log")
def get_audit_logs(
    store_id: Optional[str] = None,
    product_id: Optional[str] = None,
    decision: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get audit logs with optional filters.
    """
    query = db.query(AuditLog)
    
    if store_id:
        query = query.filter(AuditLog.store_id == store_id)
    if product_id:
        query = query.filter(AuditLog.product_id == product_id)
    if decision:
        query = query.filter(AuditLog.decision == decision)
        
    logs = query.order_by(AuditLog.timestamp.desc()).all()
    return logs

@router.post("/approve")
def approve_recommendation(
    req: ApproveRequest,
    db: Session = Depends(get_db)
):
    """
    Approve or reject a recommendation by resuming the LangGraph pipeline from its interrupt state.
    The resumed graph node writes the final audit log entry.
    """
    thread_id = req.thread_id
    log = None
    
    if req.recommendation_id:
        log = db.query(AuditLog).filter(AuditLog.id == req.recommendation_id).first()
        if log and not thread_id:
            thread_id = log.thread_id
    elif thread_id:
        log = db.query(AuditLog).filter(AuditLog.thread_id == thread_id).first()
        
    # Resume the LangGraph execution
    final_state = {}
    if thread_id:
        try:
            final_state = resume_agent(thread_id, req.decision, req.approver)
        except Exception:
            pass

    # Ensure DB record is updated if graph or direct call
    if not log and req.recommendation_id:
        raise HTTPException(status_code=404, detail="Recommendation not found")
        
    if log:
        log.decision = req.decision
        log.approver = req.approver
        log.decision_timestamp = datetime.utcnow()
        db.commit()
        db.refresh(log)
        log_id = log.id
    else:
        log_id = None
        
    return {
        "message": f"Recommendation {req.decision} successfully",
        "log_id": log_id,
        "thread_id": thread_id,
        "decision": req.decision,
        "approver": req.approver,
        "final_state": final_state
    }
