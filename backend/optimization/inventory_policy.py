import math
from typing import Dict, Any, Optional

from backend.config import (
    LEAD_TIME_DAYS, ORDER_COST, HOLDING_COST_PERCENT, SERVICE_LEVEL_Z
)

def safety_stock(demand_std: float, lead_time_days: int = LEAD_TIME_DAYS, service_level_z: float = SERVICE_LEVEL_Z) -> float:
    """
    Calculate Safety Stock.
    SS = Z * demand_std * sqrt(lead_time_days)
    """
    return service_level_z * demand_std * math.sqrt(lead_time_days)

def reorder_point(avg_daily_demand: float, lead_time_days: int = LEAD_TIME_DAYS, safety_stock_units: float = 0.0) -> float:
    """
    Calculate Reorder Point.
    ROP = (avg_daily_demand * lead_time_days) + safety_stock_units
    """
    return (avg_daily_demand * lead_time_days) + safety_stock_units

def economic_order_quantity(annual_demand: float, order_cost: float = ORDER_COST, holding_cost_per_unit: float = 1.0) -> float:
    """
    Calculate Economic Order Quantity (EOQ).
    EOQ = sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
    """
    if holding_cost_per_unit <= 0:
        return 0.0
    return math.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)

def stockout_risk_flag(current_inventory: float, forecasted_demand: float, reorder_point_val: float) -> str:
    """
    Determine stockout risk.
    HIGH: current inventory < ROP
    MEDIUM: current inventory < ROP + 20%
    LOW: otherwise
    """
    if current_inventory < reorder_point_val:
        return "HIGH"
    elif current_inventory < (reorder_point_val * 1.20):
        return "MEDIUM"
    else:
        return "LOW"

def generate_recommendation(
    store_id: str,
    product_id: str,
    forecast_data: Dict[str, Any],
    current_inventory: float,
    price: float,
    demand_std: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate single-source-of-truth inventory recommendation based on forecast and current state.
    """
    if not isinstance(forecast_data, dict):
        return {"error": "Invalid forecast data structure."}
    if "error" in forecast_data:
        return {"error": forecast_data["error"]}
        
    predictions = forecast_data.get("predicted_demand", [])
    if not predictions:
        return {"error": "No predictions found."}
        
    # Stats from forecast
    avg_daily_demand = float(sum(predictions) / len(predictions))
    if demand_std is None or demand_std <= 0:
        if len(predictions) > 1:
            demand_std = float(math.sqrt(sum((x - avg_daily_demand)**2 for x in predictions) / (len(predictions) - 1)))
        else:
            demand_std = max(float(avg_daily_demand * 0.15), 1.0)
    
    annual_demand = max(avg_daily_demand * 365.0, 1.0)
    holding_cost_per_unit = max(float(price * HOLDING_COST_PERCENT), 0.01)
    
    # Calculate policies
    ss = safety_stock(demand_std, LEAD_TIME_DAYS, SERVICE_LEVEL_Z)
    rop = reorder_point(avg_daily_demand, LEAD_TIME_DAYS, ss)
    eoq = economic_order_quantity(annual_demand, ORDER_COST, holding_cost_per_unit)
    
    risk_flag = stockout_risk_flag(current_inventory, avg_daily_demand * LEAD_TIME_DAYS, rop)
    
    # Recommended quantity: EOQ for HIGH risk, 50% EOQ for MEDIUM risk buffer, 0 for LOW risk
    if risk_flag == "HIGH":
        recommended_qty = int(round(eoq))
        reasoning = f"Current inventory ({current_inventory:.0f} units) is below ROP ({rop:.1f}). Order {recommended_qty} units (EOQ) immediately to cover {LEAD_TIME_DAYS}-day lead time."
    elif risk_flag == "MEDIUM":
        recommended_qty = int(round(eoq * 0.5))
        reasoning = f"Current inventory ({current_inventory:.0f} units) is approaching safety buffer ({rop:.1f} ROP). Order {recommended_qty} units to maintain buffer."
    else:
        recommended_qty = 0
        reasoning = f"Current inventory ({current_inventory:.0f} units) is healthy and above ROP ({rop:.1f}). No replenishment order required."

    recommendation = {
        "store_id": store_id,
        "product_id": product_id,
        "current_inventory": round(current_inventory, 1),
        "price": round(price, 2),
        "avg_daily_demand": round(avg_daily_demand, 2),
        "demand_std": round(demand_std, 2),
        "safety_stock": round(ss, 2),
        "reorder_point": round(rop, 2),
        "economic_order_quantity": round(eoq, 2),
        "recommended_qty": recommended_qty,
        "stockout_risk": risk_flag,
        "reasoning": reasoning
    }
    
    return recommendation
