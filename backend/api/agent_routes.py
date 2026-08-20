import uuid
import re
import json
import httpx
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from backend.agents.graph import run_agent_pipeline
from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from backend.observability import traced_ollama_call
from backend.security.prompt_guard import wrap_user_content, validate_entity_id, STORE_ID_RE, PRODUCT_ID_RE

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language query with Store ID and Product ID")

class ExtractedEntities(BaseModel):
    store_id: str
    product_id: str

def parse_intent(query: str) -> ExtractedEntities:
    """
    Parse the user's natural language query to extract store_id and product_id.
    Raises HTTPException(400) if no valid Store and Product entities are present.
    """
    # 1. Try regex extraction first for fast, exact identification
    store_match = re.search(r'\b(S\d{3})\b', query, re.IGNORECASE)
    prod_match = re.search(r'\b(P\d{4})\b', query, re.IGNORECASE)
    
    if store_match and prod_match:
        return ExtractedEntities(
            store_id=store_match.group(1).upper(),
            product_id=prod_match.group(1).upper()
        )

    # 2. Try LLM extraction with traced call
    prompt = (
        "Extract the store_id and product_id from the user query below.\n"
        "Valid formats: store_id is like 'S001', product_id is like 'P0001'.\n"
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
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            s_id = str(parsed.get("store_id", "")).upper().strip()
            p_id = str(parsed.get("product_id", "")).upper().strip()
            if STORE_ID_RE.match(s_id) and PRODUCT_ID_RE.match(p_id):
                return ExtractedEntities(store_id=s_id, product_id=p_id)
    except Exception:
        pass
        
    # If regex or LLM found one or neither, check keyword proximity
    words = query.replace(",", " ").replace(".", " ").replace("?", " ").split()
    found_s = store_match.group(1).upper() if store_match else None
    found_p = prod_match.group(1).upper() if prod_match else None
    
    for i, w in enumerate(words):
        if not found_s and w.lower() in ["store", "store_id", "location"] and i + 1 < len(words):
            candidate = words[i+1].upper()
            if STORE_ID_RE.match(candidate):
                found_s = candidate
        if not found_p and w.lower() in ["product", "product_id", "sku", "item"] and i + 1 < len(words):
            candidate = words[i+1].upper()
            if PRODUCT_ID_RE.match(candidate):
                found_p = candidate
                
    if found_s and found_p:
        return ExtractedEntities(store_id=found_s, product_id=found_p)
        
    missing = []
    if not found_s:
        missing.append("Store ID (e.g. S001)")
    if not found_p:
        missing.append("Product SKU (e.g. P0001)")
    raise HTTPException(
        status_code=400, 
        detail=f"Could not identify required entity from query: missing {', '.join(missing)}. Please include both Store (S001-S005) and Product (P0001-P0020)."
    )

@router.post("/query")
def agent_query_post(req: QueryRequest):
    """
    Initiates the agent graph from a natural language query, pauses at the approval interrupt,
    and returns thread_id with the pending recommendation.
    """
    entities = parse_intent(req.query)
    store_id = entities.store_id
    product_id = entities.product_id
    
    validate_entity_id(store_id, STORE_ID_RE, "store_id")
    validate_entity_id(product_id, PRODUCT_ID_RE, "product_id")
    
    thread_id = str(uuid.uuid4())
    state = run_agent_pipeline(store_id, product_id, thread_id)
    
    if not state or "risk_level" not in state:
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline failed to produce risk assessment for Store {store_id}, Product {product_id}."
        )
    
    risk_level = state["risk_level"]
    return {
        "thread_id": thread_id,
        "store_id": store_id,
        "product_id": product_id,
        "status": "PENDING_APPROVAL",
        "risk_level": risk_level,
        "forecast": state.get("forecast", {}),
        "recommendation": state.get("recommendation", {}),
        "reasoning_chain": {
            "forecast_explanation": state.get("explanation"),
            "risk_reasoning": state.get("risk_reasoning"),
            "reorder_reasoning": state.get("reorder_reasoning")
        },
        "answer": f"Forecast generated for {product_id} at Store {store_id}. Risk assessed as {risk_level}. "
                  f"{state.get('reorder_reasoning', '')} [Pending Human Approval]"
    }

@router.get("/query/{store_id}/{product_id}")
def agent_query_get(store_id: str, product_id: str):
    """
    Direct endpoint to start the agent pipeline for a specific store and product.
    Pauses at the interrupt and returns the thread_id + pending recommendation.
    """
    try:
        validate_entity_id(store_id, STORE_ID_RE, "store_id")
        validate_entity_id(product_id, PRODUCT_ID_RE, "product_id")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    thread_id = f"{store_id}_{product_id}_{uuid.uuid4().hex[:8]}"
    state = run_agent_pipeline(store_id, product_id, thread_id)
    
    if not state or "risk_level" not in state:
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline failed to produce risk assessment for Store {store_id}, Product {product_id}."
        )
    
    risk_level = state["risk_level"]
    return {
        "thread_id": thread_id,
        "store_id": store_id,
        "product_id": product_id,
        "status": "PENDING_APPROVAL",
        "risk_level": risk_level,
        "forecast": state.get("forecast", {}),
        "recommendation": state.get("recommendation", {}),
        "reasoning_chain": {
            "forecast_explanation": state.get("explanation"),
            "risk_reasoning": state.get("risk_reasoning"),
            "reorder_reasoning": state.get("reorder_reasoning")
        },
        "answer": f"Forecast for {product_id} at Store {store_id}: Risk assessed as {risk_level}. "
                  f"{state.get('reorder_reasoning', '')}"
    }
