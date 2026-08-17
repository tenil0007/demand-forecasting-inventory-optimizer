import pandas as pd
from backend.optimization.inventory_policy import stockout_risk_flag, safety_stock, reorder_point
from backend.config import RAW_DATA_PATH, COL_STORE_ID, COL_PRODUCT_ID, COL_INVENTORY_LEVEL, LEAD_TIME_DAYS

def risk_node(state: dict) -> dict:
    """
    Agent node to evaluate the stockout risk based on the current inventory and forecast.
    """
    store_id = state.get("store_id")
    product_id = state.get("product_id")
    forecast_data = state.get("forecast", {})
    
    predictions = forecast_data.get("predicted_demand", [])
    avg_demand = sum(predictions) / len(predictions) if predictions else 20.0
    demand_std = avg_demand * 0.2

    # Load current inventory from the dataset
    current_inventory = 50.0
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        filtered_df = df[(df[COL_STORE_ID] == store_id) & (df[COL_PRODUCT_ID] == product_id)]
        if not filtered_df.empty:
            current_inventory = float(filtered_df[COL_INVENTORY_LEVEL].iloc[-1])
    except Exception as e:
        print(f"Error loading inventory data: {e}")

    # Calculate ROP
    ss = safety_stock(demand_std, LEAD_TIME_DAYS)
    rop = reorder_point(avg_demand, LEAD_TIME_DAYS, ss)
    risk_level = stockout_risk_flag(current_inventory, avg_demand * LEAD_TIME_DAYS, rop)
    
    risk_reasoning = f"Current inventory ({current_inventory:.0f}) vs ROP ({rop:.1f}). Risk assessed as {risk_level}."

    return {
        "risk_level": risk_level,
        "risk_reasoning": risk_reasoning
    }
