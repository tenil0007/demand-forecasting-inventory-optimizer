import json
import httpx
from typing_extensions import TypedDict
from backend.models.forecast_model import ForecastModel
from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL

class AgentState(TypedDict):
    store_id: str
    product_id: str
    forecast: dict
    explanation: str
    risk_level: str
    risk_reasoning: str
    recommendation: dict
    reorder_reasoning: str
    approved: bool
    approver: str
    recommendation_id: int

def forecast_node(state: AgentState) -> dict:
    """
    Agent node to run the forecasting model and provide an explanation.
    """
    store_id = state.get("store_id")
    product_id = state.get("product_id")

    # Run the model
    model = ForecastModel()
    forecast_results = model.predict(store_id, product_id)
    
    # Try to get an explanation from Ollama
    explanation = ""
    try:
        prompt = f"Explain this demand forecast for Store {store_id}, Product {product_id}:\n{json.dumps(forecast_results, indent=2)}\nKeep it brief and actionable."
        
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=5.0
        )
        response.raise_for_status()
        explanation = response.json().get("response", "")
    except Exception as e:
        # Fallback to deterministic SHAP explanation
        explanation = model.explain_forecast(store_id=store_id, product_id=product_id)

    return {
        "forecast": forecast_results,
        "explanation": str(explanation)
    }
