import pandas as pd
from fastapi import APIRouter, HTTPException
from backend.models.forecast_model import ForecastModel
from backend.optimization.inventory_policy import generate_recommendation, stockout_risk_flag, safety_stock, reorder_point
from backend.config import RAW_DATA_PATH, COL_STORE_ID, COL_PRODUCT_ID, COL_INVENTORY_LEVEL, COL_PRICE, LEAD_TIME_DAYS

router = APIRouter()
model = ForecastModel()

@router.get("/{store_id}/{product_id}")
def get_reorder_recommendation(store_id: str, product_id: str):
    """
    Get reorder recommendation for a specific store and product.
    Includes risk level, recommended quantity, and reasoning.
    """
    try:
        # Get forecast
        forecast_results = model.predict_demand(store_id, product_id, days_ahead=14)
        predictions = forecast_results.get("predicted_demand", [])
        avg_demand = sum(predictions) / len(predictions) if predictions else 20.0

        # Get current inventory
        current_inventory = 50.0
        price = 25.0
        try:
            df = pd.read_csv(RAW_DATA_PATH)
            filtered_df = df[(df[COL_STORE_ID] == store_id) & (df[COL_PRODUCT_ID] == product_id)]
            if not filtered_df.empty:
                current_inventory = float(filtered_df[COL_INVENTORY_LEVEL].iloc[-1])
                price = float(filtered_df[COL_PRICE].iloc[-1])
        except Exception as e:
            print(f"Error loading inventory: {e}")

        # Check stockout risk & recommendation
        recommendation = generate_recommendation(store_id, product_id, forecast_results, current_inventory, price)
        risk_level = recommendation.get("stockout_risk", "LOW")

        return {
            "store_id": store_id,
            "product_id": product_id,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "reasoning": recommendation.get("reasoning", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")
