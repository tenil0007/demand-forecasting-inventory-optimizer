from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.agents.forecast_agent import forecast_node
from backend.agents.risk_agent import risk_node
from backend.agents.reorder_agent import reorder_node
from backend.agents.approval_agent import approval_node

class AgentState(TypedDict):
    store_id: str
    product_id: str
    forecast: dict
    explanation: str
    risk_level: str
    risk_reasoning: str
    recommendation: dict
    reorder_reasoning: str
    approved: Optional[bool]
    approver: Optional[str]
    recommendation_id: Optional[int]

# Initialize memory checkpointer
memory = MemorySaver()

def create_agent_graph():
    """Build and compile the LangGraph agent pipeline."""
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

    # Compile the graph
    return graph.compile(checkpointer=memory)

# Global graph instance
agent_app = create_agent_graph()

def run_agent_pipeline(store_id: str, product_id: str, thread_id: str):
    """
    Run the agent pipeline for a given store and product.
    Pauses at the approval node via interrupt().
    """
    initial_state = {
        "store_id": store_id,
        "product_id": product_id
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Runs up to interrupt
        state = agent_app.invoke(initial_state, config=config)
        return state
    except Exception:
        # Retrieve snapshot values up to interrupt
        current_snapshot = agent_app.get_state(config)
        return current_snapshot.values if current_snapshot else initial_state

def resume_agent(thread_id: str, decision: str, approver: str):
    """
    Resume the agent pipeline with human approval decision.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = agent_app.invoke(
            Command(resume={"decision": decision, "approver": approver}),
            config=config
        )
        return state
    except Exception:
        current_snapshot = agent_app.get_state(config)
        return current_snapshot.values if current_snapshot else {}
