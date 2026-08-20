import json
import logging
import httpx
from fastapi import APIRouter, HTTPException
from backend.models.forecast_model import ForecastModel, ModelUnavailableError, EntityNotFoundError
from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from backend.observability import traced_ollama_call
from backend.security.prompt_guard import wrap_user_content, validate_entity_id, STORE_ID_RE, PRODUCT_ID_RE

logger = logging.getLogger(__name__)
router = APIRouter()
model = ForecastModel()

@router.get("/{store_id}/{product_id}")
def get_forecast(
    store_id: str, 
    product_id: str
):
    """
    Get the forecasted demand for a specific store and product.
    Returns the recursive forecast, empirical prediction intervals, and genuine SHAP / LLM explanation.
    """
    # 1. Validate entity IDs
    try:
        validate_entity_id(store_id, STORE_ID_RE, "store_id")
        validate_entity_id(product_id, PRODUCT_ID_RE, "product_id")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Get prediction from the model
    try:
        forecast_results = model.predict_demand(store_id, product_id, days_ahead=14)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

    # 3. Generate explanation using Ollama with traced wrapper
    safe_forecast = {
        "dates": forecast_results.get("dates", []),
        "predicted_demand": forecast_results.get("predicted_demand", []),
        "lower_bound": forecast_results.get("lower_bound", []),
        "upper_bound": forecast_results.get("upper_bound", [])
    }
    explanation = ""
    explanation_available = False
    sku_label = wrap_user_content(f"Store {store_id}, Product {product_id}")
    prompt = (
        f"Explain this demand forecast for the store and product identified below:\n"
        f"{sku_label}\n"
        f"Forecast data:\n{json.dumps(safe_forecast, indent=2)}\n"
        f"Keep it brief and actionable."
    )
    
    try:
        response = traced_ollama_call(
            prompt=prompt,
            session_id=f"fc_{store_id}_{product_id}",
            span_name="forecast_explanation",
            call_fn=lambda: httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=5.0
            ),
        )
        response.raise_for_status()
        explanation = response.json().get("response", "")
        explanation_available = bool(explanation)
    except Exception as e:
        logger.warning(f"Ollama explanation call failed: {e}")
        explanation = "LLM explanation unavailable (Ollama offline)."
        explanation_available = False

    return {
        "store_id": store_id,
        "product_id": product_id,
        "forecast": forecast_results,
        "forecast_data": forecast_results,
        "explanation": explanation,
        "explanation_available": explanation_available
    }
