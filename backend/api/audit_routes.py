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
        if not log:
            raise HTTPException(status_code=404, detail=f"Recommendation ID {req.recommendation_id} not found")
        if not thread_id:
            thread_id = log.thread_id
    elif thread_id:
        log = db.query(AuditLog).filter(AuditLog.thread_id == thread_id).first()
        
    if not thread_id:
        raise HTTPException(status_code=400, detail="Either recommendation_id or thread_id must be provided")

    # Resume the LangGraph execution — this executes approval_node which updates AuditLog
    try:
        final_state = resume_agent(thread_id, req.decision, req.approver)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming agent pipeline: {str(e)}")

    # Fetch confirmed record from database
    if not log and thread_id:
        log = db.query(AuditLog).filter(AuditLog.thread_id == thread_id).first()
    elif log:
        db.refresh(log)
        
    return {
        "message": f"Recommendation {req.decision} successfully",
        "log_id": log.id if log else None,
        "thread_id": thread_id,
        "decision": req.decision,
        "approver": req.approver,
        "final_state": final_state
    }
