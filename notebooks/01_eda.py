"""
01_eda.py — Exploratory Data Analysis for Retail Store Inventory & Demand Forecasting

Produces and saves key charts and findings for the presentation deck.
Run: python notebooks/01_eda.py
"""
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.config import (
    RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID, COL_CATEGORY,
    COL_REGION, COL_INVENTORY_LEVEL, COL_UNITS_SOLD, COL_UNITS_ORDERED,
    COL_PRICE, COL_DISCOUNT, COL_WEATHER, COL_PROMOTION, COL_COMPETITOR_PRICING,
    COL_SEASONALITY, COL_EPIDEMIC, COL_DEMAND
)

# ── Output directory ──────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def save_plot(fig, name):
    """Save a plotly figure as HTML and static PNG."""
    fig.write_html(os.path.join(ARTIFACTS_DIR, f"{name}.html"))
    try:
        fig.write_image(os.path.join(ARTIFACTS_DIR, f"{name}.png"), width=1200, height=600, scale=2)
    except Exception:
        print(f"  [Notice] HTML saved. Could not save static PNG for {name} (optional: pip install kaleido)")


def main():
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    df = pd.read_csv(RAW_DATA_PATH)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values(COL_DATE)

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df[COL_DATE].min().date()} -> {df[COL_DATE].max().date()}")
    print(f"  Stores: {df[COL_STORE_ID].nunique()}")
    print(f"  Products: {df[COL_PRODUCT_ID].nunique()}")
    print(f"  Categories: {df[COL_CATEGORY].nunique()} -- {df[COL_CATEGORY].unique().tolist()}")
    print(f"  Regions: {df[COL_REGION].nunique()} -- {df[COL_REGION].unique().tolist()}")

    # ── 2. Missing Data & Dtypes ──────────────────────────────────────────────
    print("\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  None -- dataset is complete")
    else:
        print(missing[missing > 0])

    print("\nSummary statistics:")
    print(df.describe().round(2).to_string())

    findings = []

    # ── 3. Demand / Units Sold Over Time ──────────────────────────────────────
    print("\nChart 1: Demand over time...")
    daily_agg = df.groupby(COL_DATE).agg({
        COL_DEMAND: 'sum',
        COL_UNITS_SOLD: 'sum'
    }).reset_index()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=daily_agg[COL_DATE], y=daily_agg[COL_DEMAND],
                             mode='lines', name='Demand', line=dict(color='#2196F3')))
    fig.add_trace(go.Scatter(x=daily_agg[COL_DATE], y=daily_agg[COL_UNITS_SOLD],
                             mode='lines', name='Units Sold', line=dict(color='#FF9800', dash='dot')))
    fig.update_layout(title='Daily Aggregate Demand vs Units Sold Over Time',
                      xaxis_title='Date', yaxis_title='Units',
                      template='plotly_white', legend=dict(x=0.01, y=0.99))
    save_plot(fig, 'eda_demand_over_time')

    gap = (daily_agg[COL_DEMAND].sum() - daily_agg[COL_UNITS_SOLD].sum()) / daily_agg[COL_DEMAND].sum() * 100
    findings.append(f"Demand-Sales gap: {gap:.1f}% of demand goes unfulfilled (potential stockout indicator)")

    # ── 4. Demand by Category ─────────────────────────────────────────────────
    print("Chart 2: Demand by Category...")
    cat_agg = df.groupby(COL_CATEGORY)[COL_DEMAND].sum().sort_values(ascending=True).reset_index()
    fig = px.bar(cat_agg, x=COL_DEMAND, y=COL_CATEGORY, orientation='h',
                 title='Total Demand by Product Category',
                 color=COL_DEMAND, color_continuous_scale='Blues',
                 template='plotly_white')
    fig.update_layout(coloraxis_showscale=False)
    save_plot(fig, 'eda_demand_by_category')

    top_cat = cat_agg.iloc[-1][COL_CATEGORY]
    findings.append(f"Top demand category: {top_cat}")

    # ── 5. Demand by Region ───────────────────────────────────────────────────
    print("Chart 3: Demand by Region...")
    reg_agg = df.groupby(COL_REGION)[COL_DEMAND].sum().sort_values(ascending=True).reset_index()
    fig = px.bar(reg_agg, x=COL_DEMAND, y=COL_REGION, orientation='h',
                 title='Total Demand by Region',
                 color=COL_DEMAND, color_continuous_scale='Teal',
                 template='plotly_white')
    fig.update_layout(coloraxis_showscale=False)
    save_plot(fig, 'eda_demand_by_region')

    # ── 6. Price & Discount vs Demand ─────────────────────────────────────────
    print("Chart 4: Price/Discount vs Demand...")
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Price vs Demand', 'Discount vs Demand'])

    sample = df.sample(min(5000, len(df)), random_state=42)
    fig.add_trace(go.Scatter(x=sample[COL_PRICE], y=sample[COL_DEMAND],
                             mode='markers', marker=dict(size=3, opacity=0.3, color='#2196F3'),
                             name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sample[COL_DISCOUNT], y=sample[COL_DEMAND],
                             mode='markers', marker=dict(size=3, opacity=0.3, color='#FF5722'),
                             name='Discount'), row=1, col=2)
    fig.update_layout(title='Price & Discount Impact on Demand', template='plotly_white',
                      showlegend=False)
    save_plot(fig, 'eda_price_discount_impact')

    price_corr = df[COL_PRICE].corr(df[COL_DEMAND])
    disc_corr = df[COL_DISCOUNT].corr(df[COL_DEMAND])
    findings.append(f"Price-Demand correlation: {price_corr:.3f} | Discount-Demand correlation: {disc_corr:.3f}")

    # ── 7. Weather Condition vs Demand ────────────────────────────────────────
    print("Chart 5: Weather vs Demand...")
    fig = px.box(df, x=COL_WEATHER, y=COL_DEMAND,
                 title='Demand Distribution by Weather Condition',
                 color=COL_WEATHER, template='plotly_white')
    fig.update_layout(showlegend=False)
    save_plot(fig, 'eda_weather_vs_demand')

    # ── 8. Epidemic Impact ────────────────────────────────────────────────────
    print("Chart 6: Epidemic impact on demand (KEY FINDING)...")
    epi_agg = df.groupby(COL_EPIDEMIC)[COL_DEMAND].agg(['mean', 'std', 'count']).reset_index()
    epi_agg[COL_EPIDEMIC] = epi_agg[COL_EPIDEMIC].map({0: 'No Epidemic', 1: 'Epidemic'})

    fig = px.bar(epi_agg, x=COL_EPIDEMIC, y='mean',
                 error_y='std',
                 title='Average Daily Demand: Epidemic vs Non-Epidemic Periods',
                 color=COL_EPIDEMIC, color_discrete_map={'No Epidemic': '#4CAF50', 'Epidemic': '#F44336'},
                 template='plotly_white')
    fig.update_layout(showlegend=False, yaxis_title='Average Demand')
    save_plot(fig, 'eda_epidemic_impact')

    epi_mean = df[df[COL_EPIDEMIC] == 1][COL_DEMAND].mean()
    non_epi_mean = df[df[COL_EPIDEMIC] == 0][COL_DEMAND].mean()
    epi_shift = (epi_mean - non_epi_mean) / non_epi_mean * 100
    findings.append(f"[KEY FINDING] Epidemic demand shift: {epi_shift:+.1f}% vs normal periods")

    # ── 9. Inventory vs Sold — Sample SKUs (Stockout Detection) ───────────────
    print("Chart 7: Inventory vs Units Sold for sample SKUs...")
    # Pick 3 store-product combinations with high demand
    top_skus = (df.groupby([COL_STORE_ID, COL_PRODUCT_ID])[COL_DEMAND]
                .sum().nlargest(3).reset_index())

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=[f"Store {row[COL_STORE_ID]} / Product {row[COL_PRODUCT_ID]}"
                                        for _, row in top_skus.iterrows()])

    for i, (_, sku) in enumerate(top_skus.iterrows(), 1):
        sku_data = df[(df[COL_STORE_ID] == sku[COL_STORE_ID]) &
                      (df[COL_PRODUCT_ID] == sku[COL_PRODUCT_ID])].sort_values(COL_DATE)

        fig.add_trace(go.Scatter(x=sku_data[COL_DATE], y=sku_data[COL_INVENTORY_LEVEL],
                                 mode='lines', name='Inventory', line=dict(color='#2196F3'),
                                 showlegend=(i == 1)), row=i, col=1)
        fig.add_trace(go.Scatter(x=sku_data[COL_DATE], y=sku_data[COL_UNITS_SOLD],
                                 mode='lines', name='Units Sold', line=dict(color='#FF9800'),
                                 showlegend=(i == 1)), row=i, col=1)

    fig.update_layout(title='Inventory Level vs Units Sold — Top 3 SKUs (Stockout Detection)',
                      template='plotly_white', height=800)
    save_plot(fig, 'eda_inventory_vs_sold_skus')

    # Count potential stockout events (inventory < 10)
    stockout_events = len(df[df[COL_INVENTORY_LEVEL] < 10])
    findings.append(f"Potential stockout events (inventory < 10 units): {stockout_events} records ({stockout_events/len(df)*100:.1f}%)")

    # ── 10. Promotion Impact ──────────────────────────────────────────────────
    print("Chart 8: Promotion impact...")
    promo_agg = df.groupby(COL_PROMOTION)[COL_DEMAND].mean().reset_index()
    promo_agg[COL_PROMOTION] = promo_agg[COL_PROMOTION].map({0: 'No Promotion', 1: 'Promotion'})

    fig = px.bar(promo_agg, x=COL_PROMOTION, y=COL_DEMAND,
                 title='Average Demand: Promotion vs No Promotion',
                 color=COL_PROMOTION, color_discrete_map={'No Promotion': '#9E9E9E', 'Promotion': '#4CAF50'},
                 template='plotly_white')
    fig.update_layout(showlegend=False, yaxis_title='Average Demand')
    save_plot(fig, 'eda_promotion_impact')

    promo_lift = (df[df[COL_PROMOTION] == 1][COL_DEMAND].mean() / 
                  df[df[COL_PROMOTION] == 0][COL_DEMAND].mean() - 1) * 100
    findings.append(f"Promotion demand lift: {promo_lift:+.1f}%")

    # ── 11. Competitor Pricing Analysis ───────────────────────────────────────
    print("Chart 9: Competitor pricing analysis...")
    sample['price_ratio'] = sample[COL_PRICE] / sample[COL_COMPETITOR_PRICING].replace(0, np.nan)
    fig = px.scatter(sample, x='price_ratio',
                     y=COL_DEMAND,
                     title='Price/Competitor Ratio vs Demand',
                     template='plotly_white',
                     opacity=0.3)
    save_plot(fig, 'eda_competitor_pricing')

    # ── 12. Seasonality Analysis ──────────────────────────────────────────────
    print("Chart 10: Seasonality analysis...")
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    season_agg = df.groupby(COL_SEASONALITY)[COL_DEMAND].mean().reindex(season_order).reset_index()

    fig = px.bar(season_agg, x=COL_SEASONALITY, y=COL_DEMAND,
                 title='Average Demand by Season',
                 color=COL_DEMAND, color_continuous_scale='RdYlBu_r',
                 template='plotly_white')
    fig.update_layout(coloraxis_showscale=False, yaxis_title='Average Demand')
    save_plot(fig, 'eda_seasonality')

    best_season = season_agg.loc[season_agg[COL_DEMAND].idxmax(), COL_SEASONALITY]
    findings.append(f"Highest demand season: {best_season}")

    # ── Save Key Findings ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  KEY FINDINGS")
    print("=" * 60)
    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")

    findings_path = os.path.join(ARTIFACTS_DIR, 'eda_findings.md')
    with open(findings_path, 'w') as fp:
        fp.write("# EDA Key Findings\n\n")
        for f in findings:
            fp.write(f"- {f}\n")

    print(f"\n[OK] All charts saved to {ARTIFACTS_DIR}/")
    print(f"[OK] Findings saved to {findings_path}")
    print("\nDataset overview saved to artifacts/dataset_overview.json")

    overview = {
        'shape': list(df.shape),
        'date_range': [str(df[COL_DATE].min().date()), str(df[COL_DATE].max().date())],
        'n_stores': int(df[COL_STORE_ID].nunique()),
        'n_products': int(df[COL_PRODUCT_ID].nunique()),
        'n_categories': int(df[COL_CATEGORY].nunique()),
        'categories': df[COL_CATEGORY].unique().tolist(),
        'n_regions': int(df[COL_REGION].nunique()),
        'regions': df[COL_REGION].unique().tolist(),
        'columns': df.columns.tolist(),
        'findings': findings
    }
    with open(os.path.join(ARTIFACTS_DIR, 'dataset_overview.json'), 'w') as fp:
        json.dump(overview, fp, indent=2, default=str)


if __name__ == '__main__':
    main()
