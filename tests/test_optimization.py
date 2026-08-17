import pytest
import math
from backend.optimization.inventory_policy import (
    safety_stock, reorder_point, economic_order_quantity, stockout_risk_flag, generate_recommendation
)

def test_safety_stock():
    # SS = Z * demand_std * sqrt(lead_time_days)
    # Z=1.65, demand_std=10, lead_time_days=4 -> 1.65 * 10 * 2 = 33
    ss = safety_stock(demand_std=10, lead_time_days=4, service_level_z=1.65)
    assert math.isclose(ss, 33.0, rel_tol=1e-5)

def test_reorder_point():
    # ROP = (avg_daily_demand * lead_time_days) + safety_stock_units
    # avg=20, lead_time=5, ss=15 -> (20 * 5) + 15 = 115
    rop = reorder_point(avg_daily_demand=20, lead_time_days=5, safety_stock_units=15)
    assert rop == 115.0

def test_economic_order_quantity():
    # EOQ = sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
    # annual_demand=1000, order_cost=50, holding_cost_per_unit=10
    # EOQ = sqrt((2 * 1000 * 50) / 10) = sqrt(10000) = 100
    eoq = economic_order_quantity(annual_demand=1000, order_cost=50, holding_cost_per_unit=10)
    assert math.isclose(eoq, 100.0, rel_tol=1e-5)
    
    # Test zero/negative holding cost
    assert economic_order_quantity(1000, 50, 0) == 0.0

def test_stockout_risk_flag():
    rop = 100.0
    
    # HIGH: < ROP
    assert stockout_risk_flag(90.0, 50.0, rop) == "HIGH"
    
    # MEDIUM: >= ROP but < ROP * 1.20 (120)
    assert stockout_risk_flag(110.0, 50.0, rop) == "MEDIUM"
    
    # LOW: >= ROP * 1.20
    assert stockout_risk_flag(125.0, 50.0, rop) == "LOW"

def test_generate_recommendation():
    # Mock forecast data
    forecast_data = {
        "dates": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "predicted_demand": [10.0, 15.0, 20.0]  # avg = 15.0
    }
    
    result = generate_recommendation(
        store_id="S1",
        product_id="P1",
        forecast_data=forecast_data,
        current_inventory=50.0,
        price=100.0 # holding cost percent default is 0.20 -> 20.0 holding cost
    )
    
    assert "error" not in result
    assert result["store_id"] == "S1"
    assert result["product_id"] == "P1"
    assert "safety_stock" in result
    assert "reorder_point" in result
    assert "economic_order_quantity" in result
    assert "stockout_risk" in result
    assert "reasoning" in result
    
    # Since avg demand is 15.0, annual demand is 15.0 * 365 = 5475.0
    # Holding cost is 20.0, Order cost is 50.0
    # EOQ = sqrt((2 * 5475.0 * 50.0) / 20.0) = sqrt(27375) ~ 165.45
    assert math.isclose(result["economic_order_quantity"], 165.45, rel_tol=1e-3)
