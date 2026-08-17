import math
from typing import Dict, Any

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

def generate_recommendation(store_id: str, product_id: str, forecast_data: Dict[str, Any], current_inventory: float, price: float) -> Dict[str, Any]:
    """
    Generate inventory recommendation based on forecast and current state.
    """
    if "error" in forecast_data:
        return {"error": forecast_data["error"]}
        
    predictions = forecast_data.get("predicted_demand", [])
    if not predictions:
        return {"error": "No predictions found."}
        
    # Stats from forecast
    avg_daily_demand = sum(predictions) / len(predictions)
    demand_std = math.sqrt(sum((x - avg_daily_demand)**2 for x in predictions) / len(predictions)) if len(predictions) > 1 else avg_daily_demand * 0.1
    
    annual_demand = avg_daily_demand * 365
    holding_cost_per_unit = price * HOLDING_COST_PERCENT
    
    # Calculate policies
    ss = safety_stock(demand_std, LEAD_TIME_DAYS, SERVICE_LEVEL_Z)
    rop = reorder_point(avg_daily_demand, LEAD_TIME_DAYS, ss)
    eoq = economic_order_quantity(annual_demand, ORDER_COST, holding_cost_per_unit)
    
    risk_flag = stockout_risk_flag(current_inventory, avg_daily_demand * LEAD_TIME_DAYS, rop)
    
    # Construct reasoning
    if risk_flag == "HIGH":
        reasoning = f"Current inventory ({current_inventory}) is below the reorder point ({rop:.2f}). Order {eoq:.0f} units immediately."
    elif risk_flag == "MEDIUM":
        reasoning = f"Current inventory ({current_inventory}) is approaching the reorder point ({rop:.2f}). Monitor closely; an order of {eoq:.0f} units may be needed soon."
    else:
        reasoning = f"Current inventory ({current_inventory}) is healthy and above the reorder point ({rop:.2f}). No immediate action required."

    recommendation = {
        "store_id": store_id,
        "product_id": product_id,
        "safety_stock": round(ss, 2),
        "reorder_point": round(rop, 2),
        "economic_order_quantity": round(eoq, 2),
        "stockout_risk": risk_flag,
        "reasoning": reasoning
    }
    
    return recommendation
