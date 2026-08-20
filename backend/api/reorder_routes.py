import pandas as pd
from fastapi import APIRouter, HTTPException
from backend.models.forecast_model import ForecastModel, EntityNotFoundError, ModelUnavailableError
from backend.optimization.inventory_policy import generate_recommendation
from backend.config import RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID, COL_INVENTORY_LEVEL, COL_PRICE, COL_DEMAND
from backend.security.prompt_guard import validate_entity_id, STORE_ID_RE, PRODUCT_ID_RE

router = APIRouter()
model = ForecastModel()

@router.get("/{store_id}/{product_id}")
def get_reorder_recommendation(store_id: str, product_id: str):
    """
    Get reorder recommendation for a specific store and product.
    Includes risk level, recommended quantity, safety stock, and reasoning.
    """
    # 1. Validate entity IDs
    try:
        validate_entity_id(store_id, STORE_ID_RE, "store_id")
        validate_entity_id(product_id, PRODUCT_ID_RE, "product_id")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Get forecast
    try:
        forecast_results = model.predict_demand(store_id, product_id, days_ahead=14)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast for reorder: {str(e)}")

    # 3. Load actual inventory and price from dataset
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
        filtered_df = df[(df[COL_STORE_ID] == store_id) & (df[COL_PRODUCT_ID] == product_id)].sort_values(COL_DATE)
        if filtered_df.empty:
            raise HTTPException(status_code=404, detail=f"Store {store_id} / Product {product_id} not found in inventory dataset.")
        
        current_inventory = float(filtered_df[COL_INVENTORY_LEVEL].iloc[-1])
        price = float(filtered_df[COL_PRICE].iloc[-1])
        recent_demands = filtered_df[COL_DEMAND].tail(28)
        demand_std = float(recent_demands.std()) if len(recent_demands) > 1 else None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading inventory records: {str(e)}")

    # 4. Check stockout risk & generate single-source-of-truth recommendation
    recommendation = generate_recommendation(store_id, product_id, forecast_results, current_inventory, price, demand_std=demand_std)
    risk_level = recommendation.get("stockout_risk", "LOW")

    return {
        "store_id": store_id,
        "product_id": product_id,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "reasoning": recommendation.get("reasoning", "")
    }
