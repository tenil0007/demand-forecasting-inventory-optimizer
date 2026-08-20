"""
predict.py — Prediction interface for trained demand forecasting model.
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.models.forecast_model import forecast_model

def predict_sku_demand(store_id: str = "S001", product_id: str = "P0001", days_ahead: int = 14):
    """Run demand forecast for a specified store and product."""
    if not forecast_model.is_loaded:
        forecast_model.load()
    result = forecast_model.predict_demand(store_id=store_id, product_id=product_id, days_ahead=days_ahead)
    return result

if __name__ == "__main__":
    store = sys.argv[1] if len(sys.argv) > 1 else "S001"
    product = sys.argv[2] if len(sys.argv) > 2 else "P0001"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    res = predict_sku_demand(store, product, days)
    print(f"Predictions for {store} - {product} ({days} days):")
    for d, p, lb, ub in zip(res["dates"], res["predicted_demand"], res["lower_bound"], res["upper_bound"]):
        print(f"  {d}: {p} (95% Prediction Interval: [{lb}, {ub}])")
