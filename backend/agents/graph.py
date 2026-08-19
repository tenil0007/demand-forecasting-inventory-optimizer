import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from backend.config import CHECKPOINT_DB_PATH
from backend.agents.forecast_agent import forecast_node
from backend.agents.risk_agent import risk_node
from backend.agents.reorder_agent import reorder_node
from backend.agents.approval_agent import approval_node
from backend.observability import create_trace, add_trace_event

logger = logging.getLogger(__name__)

# Ensure directory for persistent SQLite checkpointer exists
Path(CHECKPOINT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)
checkpointer.setup()

class AgentState(TypedDict, total=False):
    thread_id: Optional[str]
    store_id: str
    product_id: str
    forecast: Dict[str, Any]
    explanation: str
    risk_level: str
    risk_reasoning: str
    recommendation: Dict[str, Any]
    reorder_reasoning: str
    approved: Optional[bool]
    approver: Optional[str]
    decision: Optional[str]
    recommendation_id: Optional[int]

def create_agent_graph(saver=None):
    """Build and compile the LangGraph agent pipeline with persistent SqliteSaver checkpointer."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("forecast_node", forecast_node)
    graph.add_node("risk_node", risk_node)
    graph.add_node("reorder_node", reorder_node)
    graph.add_node("approval_node", approval_node)

    # Add edges
    graph.add_edge(START, "forecast_node")
    graph.add_edge("forecast_node", "risk_node")
    graph.add_edge("risk_node", "reorder_node")
    graph.add_edge("reorder_node", "approval_node")
    graph.add_edge("approval_node", END)

    # Compile the graph with checkpointer
    active_saver = saver if saver is not None else checkpointer
    return graph.compile(checkpointer=active_saver)

# Global graph instance
agent_app = create_agent_graph()

def run_agent_pipeline(store_id: str, product_id: str, thread_id: str):
    """
    Run the agent pipeline for a given store and product.
    Executes through forecast, risk, and reorder nodes, then pauses at the approval node via interrupt().
    """
    # Create a Langfuse trace for the entire pipeline run (no-op if disabled)
    trace = create_trace(
        session_id=thread_id,
        name="agent_pipeline",
        metadata={"store_id": store_id, "product_id": product_id},
    )

    initial_state = {
        "thread_id": thread_id,
        "store_id": store_id,
        "product_id": product_id
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Runs up to interrupt in approval_node
        agent_app.invoke(initial_state, config=config)
    except Exception as e:
        logger.debug(f"Pipeline paused or reached interrupt: {e}")

    # Log pipeline pause as a trace event
    add_trace_event(trace, "pipeline_paused_at_approval", {
        "store_id": store_id,
        "product_id": product_id,
        "status": "PENDING_APPROVAL",
    })

    current_snapshot = agent_app.get_state(config)
    return current_snapshot.values if current_snapshot and current_snapshot.values else initial_state

def resume_agent(thread_id: str, decision: str, approver: str):
    """
    Resume the agent pipeline with human approval/rejection decision.
    Resumes from the exact interrupt point in approval_node.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Verify the thread exists in the persistent checkpointer
    snapshot = agent_app.get_state(config)
    if not snapshot or not snapshot.values:
        raise ValueError(f"No active or interrupted state found for thread_id '{thread_id}'")

    # Create a Langfuse trace for the resume action (no-op if disabled)
    trace = create_trace(
        session_id=thread_id,
        name="agent_resume",
        metadata={"decision": decision, "approver": approver},
    )

    state = agent_app.invoke(
        Command(resume={"decision": decision, "approver": approver}),
        config=config
    )

    # Attach the real decision and approver as a trace event
    add_trace_event(trace, "approval_decision", {
        "decision": decision,
        "approver": approver,
        "thread_id": thread_id,
    })

    return state
