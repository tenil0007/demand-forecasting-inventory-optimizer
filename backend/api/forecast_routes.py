import json
import httpx
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from backend.models.forecast_model import ForecastModel
from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL

router = APIRouter()
model = ForecastModel()

@router.get("/{store_id}/{product_id}")
def get_forecast(
    store_id: str, 
    product_id: str, 
    start_date: Optional[str] = Query(None, description="Start date for forecast (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for forecast (YYYY-MM-DD)")
):
    """
    Get the forecasted demand for a specific store and product.
    Returns the forecast, confidence intervals, and an LLM-generated explanation.
    """
    try:
        # Get prediction from the model
        forecast_results = model.predict(store_id, product_id)
        
        # Optionally, generate explanation using Ollama
        explanation = ""
        prompt = f"Explain this demand forecast for Store {store_id}, Product {product_id}:\n{json.dumps(forecast_results, indent=2)}\nKeep it brief and actionable."
        
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=5.0
            )
            response.raise_for_status()
            explanation = response.json().get("response", "")
        except Exception:
            # Fallback explanation if Ollama fails
            explanation = f"Forecasted {forecast_results.get('forecast')} units with a confidence interval between {forecast_results.get('lower_bound')} and {forecast_results.get('upper_bound')}."

        return {
            "store_id": store_id,
            "product_id": product_id,
            "forecast": forecast_results,
            "forecast_data": forecast_results,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")
