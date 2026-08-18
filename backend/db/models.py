from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from backend.db.database import Base

class AuditLog(Base):
    """
    SQLAlchemy model for the audit_log table.
    Tracks recommendations made by the system and human approval actions.
    """
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    thread_id = Column(String, index=True, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    store_id = Column(String, index=True, nullable=False)
    product_id = Column(String, index=True, nullable=False)
    
    recommended_qty = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    reasoning_snapshot = Column(Text, nullable=False)
    
    approver = Column(String, nullable=True)
    decision = Column(String, nullable=True)  # "approved", "rejected", "PENDING"
    decision_timestamp = Column(DateTime, nullable=True)
