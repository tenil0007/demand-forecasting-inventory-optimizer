import pandas as pd
from backend.optimization.inventory_policy import generate_recommendation
from backend.db.database import SessionLocal
from backend.db.models import AuditLog
from backend.config import RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID, COL_INVENTORY_LEVEL, COL_PRICE
from backend.security.prompt_guard import validate_entity_id, STORE_ID_RE, PRODUCT_ID_RE

def reorder_node(state: dict) -> dict:
    """
    Agent node to run inventory optimization and set recommendation.
    Creates an initial PENDING audit log entry for the recommendation.
    """
    store_id = state.get("store_id")
    product_id = state.get("product_id")
    validate_entity_id(store_id, STORE_ID_RE, "store_id")
    validate_entity_id(product_id, PRODUCT_ID_RE, "product_id")
    
    forecast_data = state.get("forecast", {})
    risk_level = state.get("risk_level", "LOW")
    thread_id = state.get("thread_id")
    demand_std = state.get("demand_std")

    df = pd.read_csv(RAW_DATA_PATH)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    filtered_df = df[(df[COL_STORE_ID] == store_id) & (df[COL_PRODUCT_ID] == product_id)].sort_values(COL_DATE)
    
    if filtered_df.empty:
        raise ValueError(f"SKU {product_id} at Store {store_id} not found in dataset.")
        
    current_inventory = float(filtered_df[COL_INVENTORY_LEVEL].iloc[-1])
    price = float(filtered_df[COL_PRICE].iloc[-1])

    # Use inventory policy to generate recommendation
    recommendation = generate_recommendation(store_id, product_id, forecast_data, current_inventory, price, demand_std=demand_std)
    recommended_qty = recommendation.get("recommended_qty", recommendation.get("economic_order_quantity", 0))
    reorder_reasoning = recommendation.get("reasoning", f"Order {recommended_qty} units.")

    db = SessionLocal()
    try:
        audit_entry = AuditLog(
            thread_id=thread_id,
            store_id=store_id,
            product_id=product_id,
            recommended_qty=float(recommended_qty),
            risk_level=risk_level,
            reasoning_snapshot=state.get("risk_reasoning", "") + "\n" + reorder_reasoning,
            decision="PENDING"
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        recommendation_id = audit_entry.id
    finally:
        db.close()

    return {
        "recommendation": recommendation,
        "reorder_reasoning": reorder_reasoning,
        "recommendation_id": recommendation_id
    }
