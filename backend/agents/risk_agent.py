import logging
import numpy as np
import pandas as pd
from backend.optimization.inventory_policy import stockout_risk_flag, safety_stock, reorder_point
from backend.config import RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID, COL_INVENTORY_LEVEL, COL_DEMAND, LEAD_TIME_DAYS
from backend.security.prompt_guard import validate_entity_id, STORE_ID_RE, PRODUCT_ID_RE

logger = logging.getLogger(__name__)

def risk_node(state: dict) -> dict:
    """
    Agent node to evaluate the stockout risk based on the current inventory and forecast.
    """
    store_id = state.get("store_id")
    product_id = state.get("product_id")
    
    validate_entity_id(store_id, STORE_ID_RE, "store_id")
    validate_entity_id(product_id, PRODUCT_ID_RE, "product_id")
    
    forecast_data = state.get("forecast", {})
    predictions = forecast_data.get("predicted_demand", [])
    if not predictions:
        raise ValueError(f"No forecast predictions available for Store {store_id} / SKU {product_id}")
        
    avg_demand = float(np.mean(predictions))

    # Load real current inventory and historical demand standard deviation from the dataset
    df = pd.read_csv(RAW_DATA_PATH)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    filtered_df = df[(df[COL_STORE_ID] == store_id) & (df[COL_PRODUCT_ID] == product_id)].sort_values(COL_DATE)
    
    if filtered_df.empty:
        raise ValueError(f"SKU {product_id} at Store {store_id} not found in inventory dataset.")
        
    current_inventory = float(filtered_df[COL_INVENTORY_LEVEL].iloc[-1])
    recent_history = filtered_df[COL_DEMAND].tail(28)
    demand_std = float(recent_history.std()) if len(recent_history) > 1 and recent_history.std() > 0 else float(np.std(predictions)) if len(predictions) > 1 else max(avg_demand * 0.15, 1.0)

    # Calculate ROP and Risk
    ss = safety_stock(demand_std, LEAD_TIME_DAYS)
    rop = reorder_point(avg_demand, LEAD_TIME_DAYS, ss)
    risk_level = stockout_risk_flag(current_inventory, avg_demand * LEAD_TIME_DAYS, rop)
    
    risk_reasoning = f"Current inventory ({current_inventory:.0f}) vs ROP ({rop:.1f}). Safety stock buffer: {ss:.1f}. Risk assessed as {risk_level}."

    return {
        "risk_level": risk_level,
        "risk_reasoning": risk_reasoning,
        "demand_std": demand_std
    }
