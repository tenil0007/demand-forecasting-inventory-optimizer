import pandas as pd
from backend.optimization.inventory_policy import generate_recommendation
from backend.db.database import SessionLocal
from backend.db.models import AuditLog
from backend.config import RAW_DATA_PATH, COL_STORE_ID, COL_PRODUCT_ID, COL_INVENTORY_LEVEL, COL_PRICE

def reorder_node(state: dict) -> dict:
    """
    Agent node to run inventory optimization and set recommendation.
    Creates an initial PENDING audit log entry for the recommendation.
    """
    store_id = state.get("store_id")
    product_id = state.get("product_id")
    forecast_data = state.get("forecast", {})
    risk_level = state.get("risk_level", "LOW")
    thread_id = state.get("thread_id")

    current_inventory = 50.0
    price = 25.0
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        filtered_df = df[(df[COL_STORE_ID] == store_id) & (df[COL_PRODUCT_ID] == product_id)]
        if not filtered_df.empty:
            current_inventory = float(filtered_df[COL_INVENTORY_LEVEL].iloc[-1])
            price = float(filtered_df[COL_PRICE].iloc[-1])
    except Exception:
        pass

    # Use inventory policy to generate recommendation
    recommendation = generate_recommendation(store_id, product_id, forecast_data, current_inventory, price)
    recommended_qty = recommendation.get("economic_order_quantity", 0)
    reorder_reasoning = recommendation.get("reasoning", f"Order {recommended_qty} units.")

    # Create initial audit log entry with PENDING status
    db = SessionLocal()
    audit_entry = AuditLog(
        thread_id=thread_id,
        store_id=store_id,
        product_id=product_id,
        recommended_qty=recommended_qty,
        risk_level=risk_level,
        reasoning_snapshot=state.get("risk_reasoning", "") + "\n" + reorder_reasoning,
        decision="PENDING"
    )
    db.add(audit_entry)
    db.commit()
    db.refresh(audit_entry)
    recommendation_id = audit_entry.id
    db.close()

    return {
        "recommendation": recommendation,
        "reorder_reasoning": reorder_reasoning,
        "recommendation_id": recommendation_id
    }
