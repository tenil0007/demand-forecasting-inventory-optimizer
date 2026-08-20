import uuid
import httpx
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from backend.agents.graph import run_agent_pipeline
from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from backend.observability import traced_ollama_call
from backend.security.prompt_guard import wrap_user_content, validate_entity_id, STORE_ID_RE, PRODUCT_ID_RE

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

def parse_intent(query: str):
    """
    Parse the user's natural language query to extract store_id and product_id.
    """
    prompt = (
        "Extract the store_id and product_id from the user query below. "
        "Return ONLY a JSON object with 'store_id' and 'product_id' keys.\n"
        f"{wrap_user_content(query)}"
    )
    
    try:
        response = traced_ollama_call(
            prompt=prompt,
            session_id=f"intent_{uuid.uuid4().hex[:8]}",
            span_name="intent_parsing",
            call_fn=lambda: httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=3.0,
            ),
        )
        response.raise_for_status()
        text = response.json().get("response", "")
        import json
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
        
    # Fallback keyword matching
    words = query.split()
    store_id = "S001"
    product_id = "P001"
    for i, word in enumerate(words):
        if word.lower() in ["store", "store_id"] and i + 1 < len(words):
            store_id = words[i+1].strip(",.?!;:'\"")
        if word.lower() in ["product", "product_id"] and i + 1 < len(words):
            product_id = words[i+1].strip(",.?!;:'\"")
            
    return {"store_id": store_id, "product_id": product_id}

@router.post("/query")
def agent_query_post(req: QueryRequest):
    """
    Initiates the agent graph from a natural language query, pauses at the approval interrupt,
    and returns thread_id with the pending recommendation.
    """
    try:
        intent = parse_intent(req.query)
        store_id = intent.get("store_id", "S001")
        product_id = intent.get("product_id", "P001")
        
        try:
            validate_entity_id(store_id, STORE_ID_RE, "store_id")
            validate_entity_id(product_id, PRODUCT_ID_RE, "product_id")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        thread_id = str(uuid.uuid4())
        state = run_agent_pipeline(store_id, product_id, thread_id)
        
        return {
            "thread_id": thread_id,
            "store_id": store_id,
            "product_id": product_id,
            "status": "PENDING_APPROVAL",
            "risk_level": state.get("risk_level", "LOW"),
            "forecast": state.get("forecast", {}),
            "recommendation": state.get("recommendation", {}),
            "reasoning_chain": {
                "forecast_explanation": state.get("explanation"),
                "risk_reasoning": state.get("risk_reasoning"),
                "reorder_reasoning": state.get("reorder_reasoning")
            },
            "answer": f"Forecast generated for {product_id} at Store {store_id}. Risk assessed as {state.get('risk_level')}. "
                      f"{state.get('reorder_reasoning', '')} [Pending Human Approval]"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing agent pipeline: {str(e)}")

@router.get("/query/{store_id}/{product_id}")
def agent_query_get(store_id: str, product_id: str):
    """
    Direct endpoint to start the agent pipeline for a specific store and product.
    Pauses at the interrupt and returns the thread_id + pending recommendation.
    """
    try:
        try:
            validate_entity_id(store_id, STORE_ID_RE, "store_id")
            validate_entity_id(product_id, PRODUCT_ID_RE, "product_id")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        thread_id = f"{store_id}_{product_id}_{uuid.uuid4().hex[:8]}"
        state = run_agent_pipeline(store_id, product_id, thread_id)
        
        return {
            "thread_id": thread_id,
            "store_id": store_id,
            "product_id": product_id,
            "status": "PENDING_APPROVAL",
            "risk_level": state.get("risk_level", "LOW"),
            "forecast": state.get("forecast", {}),
            "recommendation": state.get("recommendation", {}),
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
