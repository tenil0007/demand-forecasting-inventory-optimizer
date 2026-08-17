"""
03_model_experiments.py — Backtesting: Naive Policy vs AI Agent Policy

Simulates two inventory management policies over the test period and
quantifies the business value of the AI-driven approach.

Run: python notebooks/03_model_experiments.py
"""
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.config import (
    RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID, COL_CATEGORY,
    COL_INVENTORY_LEVEL, COL_UNITS_SOLD, COL_DEMAND, COL_PRICE,
    LEAD_TIME_DAYS, ORDER_COST, HOLDING_COST_PERCENT, SERVICE_LEVEL_Z
)
from backend.optimization.inventory_policy import (
    safety_stock, reorder_point, economic_order_quantity, stockout_risk_flag
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def simulate_naive_policy(sku_data, static_rop=50, fixed_order_qty=100):
    """
    Naive policy: reorder a fixed quantity when inventory drops below a static threshold.
    This is what most retailers do without ML.
    """
    results = []
    inventory = sku_data.iloc[0][COL_INVENTORY_LEVEL]

    for _, row in sku_data.iterrows():
        demand = row[COL_DEMAND]
        price = row[COL_PRICE]

        # Fulfill what we can
        fulfilled = min(inventory, demand)
        stockout_units = max(0, demand - inventory)
        lost_sales = stockout_units * price

        # Update inventory
        inventory = max(0, inventory - demand)

        # Reorder decision: static threshold
        ordered = 0
        if inventory <= static_rop:
            ordered = fixed_order_qty
            inventory += ordered  # simplified: instant arrival after lead time ignored for comparison

        # Holding cost
        holding = inventory * price * HOLDING_COST_PERCENT / 365

        results.append({
            'date': row[COL_DATE],
            'demand': demand,
            'fulfilled': fulfilled,
            'stockout_units': stockout_units,
            'lost_sales': lost_sales,
            'inventory': inventory,
            'ordered': ordered,
            'order_cost': ORDER_COST if ordered > 0 else 0,
            'holding_cost': holding
        })

    return pd.DataFrame(results)


def simulate_agent_policy(sku_data):
    """
    Agent policy: dynamic ROP/EOQ driven by rolling demand statistics.
    Uses the same optimization formulas as the agent's inventory_policy module.
    """
    results = []
    inventory = sku_data.iloc[0][COL_INVENTORY_LEVEL]

    # Rolling window for demand stats
    demand_history = []

    for _, row in sku_data.iterrows():
        demand = row[COL_DEMAND]
        price = row[COL_PRICE]
        demand_history.append(demand)

        # Compute dynamic stats from recent history
        recent = demand_history[-28:] if len(demand_history) >= 7 else demand_history
        avg_demand = np.mean(recent)
        std_demand = np.std(recent) if len(recent) > 1 else avg_demand * 0.3

        # Dynamic optimization
        ss = safety_stock(std_demand, LEAD_TIME_DAYS, SERVICE_LEVEL_Z)
        rop = reorder_point(avg_demand, LEAD_TIME_DAYS, ss)
        annual_demand = avg_demand * 365
        holding_cost_per_unit = price * HOLDING_COST_PERCENT
        eoq = economic_order_quantity(annual_demand, ORDER_COST,
                                      max(holding_cost_per_unit, 0.01))

        # Fulfill what we can
        fulfilled = min(inventory, demand)
        stockout_units = max(0, demand - inventory)
        lost_sales = stockout_units * price

        # Update inventory
        inventory = max(0, inventory - demand)

        # Reorder decision: dynamic threshold
        ordered = 0
        if inventory <= rop:
            ordered = max(eoq, demand * LEAD_TIME_DAYS)  # at least cover lead time
            inventory += ordered

        # Holding cost
        holding = inventory * price * HOLDING_COST_PERCENT / 365

        results.append({
            'date': row[COL_DATE],
            'demand': demand,
            'fulfilled': fulfilled,
            'stockout_units': stockout_units,
            'lost_sales': lost_sales,
            'inventory': inventory,
            'ordered': ordered,
            'order_cost': ORDER_COST if ordered > 0 else 0,
            'holding_cost': holding
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("  BACKTEST: NAIVE POLICY vs AI AGENT POLICY")
    print("=" * 60)

    # Load data
    df = pd.read_csv(RAW_DATA_PATH)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values(COL_DATE)

    # Use last 60 days as test period
    test_start = df[COL_DATE].max() - pd.Timedelta(days=60)
    test_df = df[df[COL_DATE] >= test_start].copy()

    print(f"\nTest period: {test_start.date()} -> {df[COL_DATE].max().date()}")
    print(f"Test records: {len(test_df)}")

    # Select top 10 SKUs by demand volume for simulation
    top_skus = (test_df.groupby([COL_STORE_ID, COL_PRODUCT_ID])[COL_DEMAND]
                .sum().nlargest(10).reset_index())

    print(f"Simulating {len(top_skus)} SKUs...\n")

    total_naive = {'stockout_days': 0, 'lost_sales': 0, 'holding_cost': 0, 'order_cost': 0}
    total_agent = {'stockout_days': 0, 'lost_sales': 0, 'holding_cost': 0, 'order_cost': 0}
    sku_results = []

    for _, sku in top_skus.iterrows():
        store_id = sku[COL_STORE_ID]
        product_id = sku[COL_PRODUCT_ID]

        sku_data = test_df[(test_df[COL_STORE_ID] == store_id) &
                           (test_df[COL_PRODUCT_ID] == product_id)].sort_values(COL_DATE)

        if len(sku_data) < 7:
            continue

        naive_results = simulate_naive_policy(sku_data)
        agent_results = simulate_agent_policy(sku_data)

        naive_stockouts = (naive_results['stockout_units'] > 0).sum()
        agent_stockouts = (agent_results['stockout_units'] > 0).sum()

        naive_cost = naive_results['lost_sales'].sum() + naive_results['holding_cost'].sum() + naive_results['order_cost'].sum()
        agent_cost = agent_results['lost_sales'].sum() + agent_results['holding_cost'].sum() + agent_results['order_cost'].sum()

        total_naive['stockout_days'] += naive_stockouts
        total_naive['lost_sales'] += naive_results['lost_sales'].sum()
        total_naive['holding_cost'] += naive_results['holding_cost'].sum()
        total_naive['order_cost'] += naive_results['order_cost'].sum()

        total_agent['stockout_days'] += agent_stockouts
        total_agent['lost_sales'] += agent_results['lost_sales'].sum()
        total_agent['holding_cost'] += agent_results['holding_cost'].sum()
        total_agent['order_cost'] += agent_results['order_cost'].sum()

        sku_results.append({
            'store_id': store_id, 'product_id': product_id,
            'naive_stockout_days': naive_stockouts,
            'agent_stockout_days': agent_stockouts,
            'naive_total_cost': naive_cost,
            'agent_total_cost': agent_cost,
            'savings': naive_cost - agent_cost
        })

        print(f"  {store_id}/{product_id}: Naive={naive_stockouts} stockout days, "
              f"Agent={agent_stockouts} stockout days, Savings=${naive_cost - agent_cost:,.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    naive_total_cost = total_naive['lost_sales'] + total_naive['holding_cost'] + total_naive['order_cost']
    agent_total_cost = total_agent['lost_sales'] + total_agent['holding_cost'] + total_agent['order_cost']
    total_savings = naive_total_cost - agent_total_cost

    stockout_reduction = ((total_naive['stockout_days'] - total_agent['stockout_days']) /
                          max(total_naive['stockout_days'], 1) * 100)

    print(f"\n  Naive Policy:")
    print(f"    Stockout days:  {total_naive['stockout_days']}")
    print(f"    Lost sales:     ${total_naive['lost_sales']:,.2f}")
    print(f"    Holding cost:   ${total_naive['holding_cost']:,.2f}")
    print(f"    Order cost:     ${total_naive['order_cost']:,.2f}")
    print(f"    TOTAL COST:     ${naive_total_cost:,.2f}")

    print(f"\n  AI Agent Policy:")
    print(f"    Stockout days:  {total_agent['stockout_days']}")
    print(f"    Lost sales:     ${total_agent['lost_sales']:,.2f}")
    print(f"    Holding cost:   ${total_agent['holding_cost']:,.2f}")
    print(f"    Order cost:     ${total_agent['order_cost']:,.2f}")
    print(f"    TOTAL COST:     ${agent_total_cost:,.2f}")

    print(f"\n  [RESULT] SAVINGS:            ${total_savings:,.2f}")
    print(f"  [RESULT] STOCKOUT REDUCTION: {stockout_reduction:.1f}%")

    # ── Visualization ─────────────────────────────────────────────────────────
    print("\nGenerating comparison charts...")

    # Chart 1: Cost comparison
    fig = go.Figure()
    categories = ['Lost Sales', 'Holding Cost', 'Order Cost', 'TOTAL']
    naive_vals = [total_naive['lost_sales'], total_naive['holding_cost'],
                  total_naive['order_cost'], naive_total_cost]
    agent_vals = [total_agent['lost_sales'], total_agent['holding_cost'],
                  total_agent['order_cost'], agent_total_cost]

    fig.add_trace(go.Bar(name='Naive Policy', x=categories, y=naive_vals,
                         marker_color='#EF5350', text=[f'${v:,.0f}' for v in naive_vals],
                         textposition='outside'))
    fig.add_trace(go.Bar(name='AI Agent Policy', x=categories, y=agent_vals,
                         marker_color='#26A69A', text=[f'${v:,.0f}' for v in agent_vals],
                         textposition='outside'))

    fig.update_layout(
        title=f'Naive Policy vs AI Agent Policy — Cost Comparison<br>'
              f'<sub>Total savings: ${total_savings:,.0f} | Stockout reduction: {stockout_reduction:.0f}%</sub>',
        yaxis_title='Cost ($)',
        barmode='group',
        template='plotly_white',
        font=dict(size=14),
        height=500
    )
    save_fig(fig, 'backtest_cost_comparison')

    # Chart 2: Stockout days per SKU
    sku_df = pd.DataFrame(sku_results)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Naive Policy', x=sku_df['product_id'],
                         y=sku_df['naive_stockout_days'], marker_color='#EF5350'))
    fig.add_trace(go.Bar(name='AI Agent Policy', x=sku_df['product_id'],
                         y=sku_df['agent_stockout_days'], marker_color='#26A69A'))
    fig.update_layout(title='Stockout Days by SKU — Naive vs AI Agent',
                      yaxis_title='Stockout Days', barmode='group',
                      template='plotly_white', height=400)
    save_fig(fig, 'backtest_stockout_by_sku')

    # Save results
    backtest_results = {
        'test_period': [str(test_start.date()), str(df[COL_DATE].max().date())],
        'n_skus_tested': len(sku_results),
        'naive_policy': {k: round(v, 2) for k, v in total_naive.items()},
        'agent_policy': {k: round(v, 2) for k, v in total_agent.items()},
        'total_savings': round(total_savings, 2),
        'stockout_reduction_pct': round(stockout_reduction, 1),
        'sku_results': sku_results
    }
    with open(os.path.join(ARTIFACTS_DIR, 'backtest_results.json'), 'w') as fp:
        json.dump(backtest_results, fp, indent=2, default=str)

    print(f"\n[OK] Charts saved to {ARTIFACTS_DIR}/")
    print(f"[OK] Results saved to {ARTIFACTS_DIR}/backtest_results.json")


def save_fig(fig, name):
    """Save plotly figure."""
    fig.write_html(os.path.join(ARTIFACTS_DIR, f"{name}.html"))
    try:
        fig.write_image(os.path.join(ARTIFACTS_DIR, f"{name}.png"), width=1200, height=600, scale=2)
    except Exception:
        print(f"  [Notice] HTML saved. Could not save static PNG for {name} (optional: pip install kaleido)")


if __name__ == '__main__':
    main()
