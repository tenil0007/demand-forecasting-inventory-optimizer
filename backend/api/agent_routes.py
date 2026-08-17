import uuid
import httpx
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from backend.agents.graph import run_agent_pipeline
from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

def parse_intent(query: str):
    """
    Parse the user's natural language query to extract store_id and product_id.
    """
    prompt = f"Extract the store_id and product_id from this query: '{query}'. Return ONLY a JSON object with 'store_id' and 'product_id' keys."
    
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=5.0
        )
        response.raise_for_status()
        text = response.json().get("response", "")
        import json
        import re
        # Find JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
        
    # Fallback keyword matching
    words = query.split()
    store_id = "S001"  # Default
    product_id = "P001"  # Default
    for i, word in enumerate(words):
        if word.lower() in ["store", "store_id"] and i + 1 < len(words):
            store_id = words[i+1].strip(",.")
        if word.lower() in ["product", "product_id"] and i + 1 < len(words):
            product_id = words[i+1].strip(",.")
            
    return {"store_id": store_id, "product_id": product_id}

@router.post("/query")
def agent_query(req: QueryRequest):
    """
    Endpoint for natural language queries to the agent pipeline.
    """
    try:
        intent = parse_intent(req.query)
        store_id = intent.get("store_id", "S001")
        product_id = intent.get("product_id", "P001")
        
        thread_id = str(uuid.uuid4())
        
        # Run agent pipeline
        state = run_agent_pipeline(store_id, product_id, thread_id)
        
        # Return the resulting state/reasoning chain
        return {
            "thread_id": thread_id,
            "store_id": store_id,
            "product_id": product_id,
            "reasoning_chain": {
                "forecast_explanation": state.get("explanation"),
                "risk_reasoning": state.get("risk_reasoning"),
                "reorder_reasoning": state.get("reorder_reasoning")
            },
            "answer": f"Forecast for {product_id} at Store {store_id}: Risk assessed as {state.get('risk_level')}. "
                      f"{state.get('reorder_reasoning', '')}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing agent pipeline: {str(e)}")
