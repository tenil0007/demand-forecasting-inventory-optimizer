from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.db.models import AuditLog

router = APIRouter()

class ApproveRequest(BaseModel):
    recommendation_id: int
    decision: str
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
    Approve or reject a recommendation, updating the audit log.
    If the agent pipeline was paused, this would theoretically resume it,
    but here we handle the DB update directly.
    """
    from datetime import datetime
    
    log = db.query(AuditLog).filter(AuditLog.id == req.recommendation_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Recommendation not found")
        
    log.decision = req.decision
    log.approver = req.approver
    log.decision_timestamp = datetime.utcnow()
    
    db.commit()
    db.refresh(log)
    
    return {"message": "Recommendation updated successfully", "log_id": log.id}
