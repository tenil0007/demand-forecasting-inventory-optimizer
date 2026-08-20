"""
backtest.py — Backtesting: Naive Inventory Policy vs AI Agent Policy

Simulates two inventory management policies over the test period and
quantifies the business value of the AI-driven approach.

Run: python backend/models/backtest.py
"""
import sys
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np

from backend.config import (
    BASE_DIR, RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID, COL_CATEGORY,
    COL_INVENTORY_LEVEL, COL_UNITS_SOLD, COL_DEMAND, COL_PRICE,
    LEAD_TIME_DAYS, ORDER_COST, HOLDING_COST_PERCENT, SERVICE_LEVEL_Z
)
from backend.optimization.inventory_policy import (
    safety_stock, reorder_point, economic_order_quantity, stockout_risk_flag
)

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def simulate_naive_policy(sku_data, static_rop=50, fixed_order_qty=100, lead_time=LEAD_TIME_DAYS):
    """
    Naive policy: reorder a fixed quantity when inventory position drops below a static threshold,
    with realistic lead-time pipeline delay.
    """
    results = []
    inventory = float(sku_data.iloc[0][COL_INVENTORY_LEVEL])
    pending_orders = []  # list of (arrival_day_idx, qty)

    for day_idx, (_, row) in enumerate(sku_data.iterrows()):
        # 1. Arriving orders
        arrived_qty = sum(qty for arr_idx, qty in pending_orders if arr_idx == day_idx)
        inventory += arrived_qty
        pending_orders = [(arr_idx, qty) for arr_idx, qty in pending_orders if arr_idx > day_idx]

        demand = float(row[COL_DEMAND])
        price = float(row[COL_PRICE])

        # 2. Customer fulfillment
        fulfilled = min(inventory, demand)
        stockout_units = max(0.0, demand - inventory)
        lost_sales = stockout_units * price

        # 3. End-of-day physical inventory
        inventory = max(0.0, inventory - demand)

        # 4. Inventory position = on_hand + on_order
        on_order = sum(qty for _, qty in pending_orders)
        inv_position = inventory + on_order

        # 5. Order decision
        ordered = 0
        if inv_position <= static_rop:
            ordered = fixed_order_qty
            pending_orders.append((day_idx + lead_time, ordered))

        holding = inventory * price * HOLDING_COST_PERCENT / 365.0

        results.append({
            'date': row[COL_DATE],
            'demand': demand,
            'fulfilled': fulfilled,
            'stockout_units': stockout_units,
            'lost_sales': lost_sales,
            'inventory': inventory,
            'ordered': ordered,
            'order_cost': ORDER_COST if ordered > 0 else 0.0,
            'holding_cost': holding
        })

    return pd.DataFrame(results)


def simulate_agent_policy(sku_data, lead_time=LEAD_TIME_DAYS):
    """
    Agent policy: dynamic ROP/EOQ driven by rolling demand statistics, service level,
    and realistic lead-time pipeline delay.
    """
    results = []
    inventory = float(sku_data.iloc[0][COL_INVENTORY_LEVEL])
    pending_orders = []  # list of (arrival_day_idx, qty)
    demand_history = []

    for day_idx, (_, row) in enumerate(sku_data.iterrows()):
        # 1. Arriving orders
        arrived_qty = sum(qty for arr_idx, qty in pending_orders if arr_idx == day_idx)
        inventory += arrived_qty
        pending_orders = [(arr_idx, qty) for arr_idx, qty in pending_orders if arr_idx > day_idx]

        demand = float(row[COL_DEMAND])
        price = float(row[COL_PRICE])
        demand_history.append(demand)

        # 2. Compute dynamic stats strictly from past observed history
        recent = demand_history[-28:] if len(demand_history) >= 7 else demand_history
        avg_demand = float(np.mean(recent))
        std_demand = float(np.std(recent)) if len(recent) > 1 else max(avg_demand * 0.2, 1.0)

        # 3. Dynamic optimization formulas
        ss = safety_stock(std_demand, lead_time, SERVICE_LEVEL_Z)
        rop = reorder_point(avg_demand, lead_time, ss)
        annual_demand = max(avg_demand * 365.0, 1.0)
        holding_cost_unit = max(price * HOLDING_COST_PERCENT, 0.01)
        eoq = economic_order_quantity(annual_demand, ORDER_COST, holding_cost_unit)

        # 4. Customer fulfillment
        fulfilled = min(inventory, demand)
        stockout_units = max(0.0, demand - inventory)
        lost_sales = stockout_units * price

        # 5. End-of-day physical inventory
        inventory = max(0.0, inventory - demand)

        # 6. Inventory position = on_hand + on_order
        on_order = sum(qty for _, qty in pending_orders)
        inv_position = inventory + on_order

        # 7. Dynamic order decision
        ordered = 0
        if inv_position <= rop:
            ordered = max(int(round(eoq)), 10)
            pending_orders.append((day_idx + lead_time, ordered))

        holding = inventory * price * HOLDING_COST_PERCENT / 365.0

        results.append({
            'date': row[COL_DATE],
            'demand': demand,
            'fulfilled': fulfilled,
            'stockout_units': stockout_units,
            'lost_sales': lost_sales,
            'inventory': inventory,
            'ordered': ordered,
            'order_cost': ORDER_COST if ordered > 0 else 0.0,
            'holding_cost': holding,
            'dynamic_rop': rop,
            'dynamic_ss': ss,
            'dynamic_eoq': eoq
        })

    return pd.DataFrame(results)


def run_full_backtest():
    print("=" * 60)
    print("RUNNING INVENTORY POLICY BACKTEST (NAIVE vs AGENT)")
    print("=" * 60)

    df = pd.read_csv(RAW_DATA_PATH)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    # Time-based split: last 60 days
    max_date = df[COL_DATE].max()
    test_start = max_date - pd.Timedelta(days=60)
    test_df = df[df[COL_DATE] >= test_start].copy()

    print(f"Test period: {test_start.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({len(test_df)} rows)")

    naive_totals = {'lost_sales': 0, 'holding_cost': 0, 'order_cost': 0, 'stockout_units': 0, 'total_demand': 0}
    agent_totals = {'lost_sales': 0, 'holding_cost': 0, 'order_cost': 0, 'stockout_units': 0, 'total_demand': 0}

    sku_groups = test_df.groupby([COL_STORE_ID, COL_PRODUCT_ID])
    num_skus = len(sku_groups)
    print(f"Evaluating across {num_skus} store-product combinations...")

    for (store, prod), sku_data in sku_groups:
        sku_sorted = sku_data.sort_values(COL_DATE).reset_index(drop=True)
        if len(sku_sorted) < 14:
            continue

        naive_res = simulate_naive_policy(sku_sorted)
        agent_res = simulate_agent_policy(sku_sorted)

        for k in ['lost_sales', 'holding_cost', 'order_cost', 'stockout_units', 'demand']:
            target_k = 'total_demand' if k == 'demand' else k
            naive_totals[target_k] += naive_res[k].sum()
            agent_totals[target_k] += agent_res[k].sum()

    naive_total_cost = naive_totals['lost_sales'] + naive_totals['holding_cost'] + naive_totals['order_cost']
    agent_total_cost = agent_totals['lost_sales'] + agent_totals['holding_cost'] + agent_totals['order_cost']
    cost_savings = naive_total_cost - agent_total_cost
    savings_pct = (cost_savings / naive_total_cost * 100) if naive_total_cost > 0 else 0

    naive_fill_rate = 1.0 - (naive_totals['stockout_units'] / naive_totals['total_demand']) if naive_totals['total_demand'] > 0 else 1.0
    agent_fill_rate = 1.0 - (agent_totals['stockout_units'] / agent_totals['total_demand']) if agent_totals['total_demand'] > 0 else 1.0

    print("\n" + "-" * 50)
    print(f"{'Metric':<30} {'Naive Policy':<15} {'Agent Policy':<15}")
    print("-" * 50)
    print(f"{'Total Cost ($)':<30} ${naive_total_cost:>12,.2f} ${agent_total_cost:>12,.2f}")
    print(f"{'  Lost Sales ($)':<30} ${naive_totals['lost_sales']:>12,.2f} ${agent_totals['lost_sales']:>12,.2f}")
    print(f"{'  Holding Cost ($)':<30} ${naive_totals['holding_cost']:>12,.2f} ${agent_totals['holding_cost']:>12,.2f}")
    print(f"{'  Ordering Cost ($)':<30} ${naive_totals['order_cost']:>12,.2f} ${agent_totals['order_cost']:>12,.2f}")
    print(f"{'Fill Rate':<30} {naive_fill_rate:>13.1%} {agent_fill_rate:>13.1%}")
    print("-" * 50)
    print(f"Total Cost Reduction: ${cost_savings:,.2f} ({savings_pct:.1f}% savings)")

    results_summary = {
        'naive_policy': {
            'total_cost': round(naive_total_cost, 2),
            'lost_sales': round(naive_totals['lost_sales'], 2),
            'holding_cost': round(naive_totals['holding_cost'], 2),
            'order_cost': round(naive_totals['order_cost'], 2),
            'fill_rate': round(naive_fill_rate, 4),
            'stockout_units': int(naive_totals['stockout_units'])
        },
        'agent_policy': {
            'total_cost': round(agent_total_cost, 2),
            'lost_sales': round(agent_totals['lost_sales'], 2),
            'holding_cost': round(agent_totals['holding_cost'], 2),
            'order_cost': round(agent_totals['order_cost'], 2),
            'fill_rate': round(agent_fill_rate, 4),
            'stockout_units': int(agent_totals['stockout_units'])
        },
        'comparison': {
            'cost_savings_dollars': round(cost_savings, 2),
            'cost_savings_pct': round(savings_pct, 2),
            'fill_rate_improvement_pct': round((agent_fill_rate - naive_fill_rate) * 100, 2)
        }
    }

    with open(ARTIFACTS_DIR / "backtest_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    return results_summary

if __name__ == '__main__':
    run_full_backtest()
