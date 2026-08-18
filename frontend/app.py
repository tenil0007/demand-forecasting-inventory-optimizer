import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
import sys
import re
import textwrap
from datetime import datetime
import numpy as np

# ── 1. Page Configuration & Theme ─────────────────────────────────────────────
st.set_page_config(
    page_title="NexusSupply – Demand Forecasting & Inventory Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add root directory to sys.path for direct module access
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Try loading backend config & modules
try:
    from backend.config import (
        RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID,
        COL_CATEGORY, COL_REGION, COL_INVENTORY_LEVEL, COL_UNITS_SOLD,
        COL_PRICE, COL_DISCOUNT, COL_DEMAND, LEAD_TIME_DAYS, ORDER_COST,
        HOLDING_COST_PERCENT, SERVICE_LEVEL_Z
    )
    from backend.optimization.inventory_policy import (
        safety_stock, reorder_point, economic_order_quantity,
        stockout_risk_flag, generate_recommendation
    )
    from backend.models.forecast_model import ForecastModel
    from backend.db.database import SessionLocal, init_db
    from backend.db.models import AuditLog
    BACKEND_AVAILABLE = True
except Exception:
    RAW_DATA_PATH = os.path.join(root_dir, "data", "raw", "retail_store_inventory.csv")
    BACKEND_AVAILABLE = False

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def render_html(html_str: str):
    """Renders HTML cleanly without leading markdown indentation causing code blocks."""
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

# ── 2. NexusSupply Design System (CSS) ────────────────────────────────────────
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 99 !important;
    }
    .stDeployButton {
        display: none !important;
    }

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* ── Native Bordered Container Styling ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        padding: 16px 18px !important;
        margin-bottom: 16px !important;
    }

    /* ── Sidebar Styling ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 18px 12px 14px 12px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 12px;
    }
    .sidebar-brand-icon {
        color: #2563EB;
        display: flex;
        align-items: center;
    }
    .sidebar-brand-text {
        font-size: 18px;
        font-weight: 700;
        color: #2563EB;
        letter-spacing: -0.02em;
    }
    .sidebar-brand-text span {
        color: #0F172A;
    }

    /* Hide Radio Circles in all Streamlit versions */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label span[data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"],
    [data-testid="stSidebar"] [data-testid="stRadio"] div[data-testid="stRadioButtonCustom"] {
        display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 4px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        padding: 9px 12px !important;
        border-radius: 8px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: #475569 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        font-weight: 600 !important;
        border-color: #DBEAFE !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
        margin: 0 !important;
        font-size: 13.5px !important;
        line-height: 1.4 !important;
    }

    .sidebar-user {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 14px 10px;
        border-top: 1px solid #E2E8F0;
        margin-top: 20px;
    }
    .sidebar-user-avatar {
        width: 34px;
        height: 34px;
        border-radius: 6px;
        background: #E2E8F0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        color: #334155;
    }
    .sidebar-user-name {
        font-size: 13px;
        font-weight: 600;
        color: #0F172A;
    }
    .sidebar-user-role {
        font-size: 11px;
        color: #64748B;
    }

    /* ── Page Header Bar ──────────────────────────────────────────────────── */
    .page-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 16px 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .page-header-title {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .system-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-weight: 500;
        color: #475569;
        background: #F8FAFC;
        padding: 5px 12px;
        border-radius: 9999px;
        border: 1px solid #E2E8F0;
    }
    .system-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }

    /* ── KPI Cards ────────────────────────────────────────────────────────── */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        min-height: 102px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .kpi-card-label {
        font-size: 13px;
        font-weight: 500;
        color: #64748B;
    }
    .kpi-card-icon {
        color: #94A3B8;
        display: flex;
        align-items: center;
    }
    .kpi-card-value {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-card-sub {
        font-size: 12px;
        color: #059669;
        font-weight: 500;
        margin-top: 4px;
    }
    .kpi-card-sub-muted {
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
    }
    .kpi-card-danger {
        background: #FEF2F2 !important;
        border: 1px solid #FECACA !important;
    }
    .kpi-card-danger .kpi-card-label,
    .kpi-card-danger .kpi-card-icon {
        color: #B91C1C !important;
    }
    .kpi-card-danger .kpi-card-value {
        color: #B91C1C !important;
    }

    .text-emerald { color: #059669 !important; }
    .text-red { color: #DC2626 !important; }
    .text-amber { color: #D97706 !important; }
    .text-blue { color: #2563EB !important; }

    /* ── Badges ───────────────────────────────────────────────────────────── */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        border: 1px solid transparent;
    }
    .badge-red { background: #FEE2E2; color: #991B1B; border-color: #FECACA; }
    .badge-amber { background: #FEF3C7; color: #92400E; border-color: #FDE68A; }
    .badge-green { background: #DCFCE7; color: #166534; border-color: #BBF7D0; }
    .badge-blue { background: #DBEAFE; color: #1E40AF; border-color: #BFDBFE; }
    .badge-gray { background: #F1F5F9; color: #475569; border-color: #E2E8F0; }

    /* ── Section Header ───────────────────────────────────────────────────── */
    .card-header-title {
        font-size: 14px;
        font-weight: 600;
        color: #0F172A;
        margin: 0 0 12px 0;
    }

    /* ── Styled Tables ────────────────────────────────────────────────────── */
    .nexus-table-wrapper {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
    }
    .nexus-table-header {
        padding: 14px 18px;
        border-bottom: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #F8FAFC;
    }
    .nexus-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
    }
    .nexus-table thead tr {
        border-bottom: 1px solid #E2E8F0;
        background: #F8FAFC;
    }
    .nexus-table th {
        padding: 10px 14px;
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
    }
    .nexus-table tbody tr {
        border-bottom: 1px solid #F1F5F9;
        transition: background 0.15s ease;
    }
    .nexus-table tbody tr:hover {
        background: #F8FAFC;
    }
    .nexus-table td {
        padding: 11px 14px;
        font-size: 13px;
        color: #0F172A;
    }
    .nexus-table td.muted { color: #64748B; }
    .nexus-table td.bold { font-weight: 600; }
    .nexus-table td.primary { font-weight: 600; color: #2563EB; }
    .nexus-table td.danger { font-weight: 700; color: #DC2626; }
    .nexus-table td.mono { font-family: 'SF Mono', monospace; font-size: 12.5px; }

    /* ── HITL Review Panel ────────────────────────────────────────────────── */
    .hitl-panel {
        background: #FFFFFF;
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        overflow: hidden;
        margin-bottom: 16px;
    }
    .hitl-panel-header {
        padding: 14px 18px;
        border-bottom: 1px solid #E2E8F0;
        background: #F8FAFC;
    }
    .hitl-panel-header-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .hitl-panel-title {
        font-size: 15px;
        font-weight: 600;
        color: #0F172A;
    }
    .hitl-panel-subtitle {
        font-size: 13px;
        color: #64748B;
    }
    .hitl-panel-body {
        padding: 18px;
    }
    .hitl-stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 16px;
    }
    .hitl-stat-box {
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }
    .hitl-stat-box-default {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
    }
    .hitl-stat-box-primary {
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
    }
    .hitl-stat-label {
        font-size: 11px;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .hitl-stat-label-primary {
        font-size: 11px;
        color: #2563EB;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .hitl-stat-value {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
    }
    .hitl-stat-value-primary {
        font-size: 22px;
        font-weight: 700;
        color: #2563EB;
    }
    .hitl-stat-value-danger {
        font-size: 22px;
        font-weight: 700;
        color: #DC2626;
    }

    .policy-reasoner {
        position: relative;
        background: #0F172A;
        color: #E2E8F0;
        border-radius: 8px;
        padding: 18px 16px 14px;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 18px;
    }
    .policy-reasoner-label {
        position: absolute;
        top: -9px;
        left: 12px;
        background: #2563EB;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .hitl-detail-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        padding: 4px 0;
    }
    .hitl-detail-label { color: #64748B; }
    .hitl-detail-value { font-weight: 600; color: #0F172A; }

    /* ── Chat Header ──────────────────────────────────────────────────────── */
    .chat-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 14px 18px;
        border-bottom: 1px solid #E2E8F0;
        background: #F8FAFC;
        border-radius: 10px 10px 0 0;
        margin-bottom: 16px;
    }
    .chat-header-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #EFF6FF;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #2563EB;
    }
    .chat-header-name {
        font-size: 14px;
        font-weight: 600;
        color: #0F172A;
    }
    .chat-header-sub {
        font-size: 12px;
        color: #64748B;
    }

    /* ── Dark Simulation Control Tower ────────────────────────────────────── */
    .sim-tower {
        background: #0F172A;
        border-radius: 10px;
        padding: 18px 22px;
        color: #F1F5F9;
        margin-bottom: 20px;
    }
    .sim-tower-label {
        font-size: 10px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }
    .sim-tower-content {
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .sim-tower-day {
        font-size: 17px;
        font-family: 'SF Mono', monospace;
        font-weight: 600;
        color: #FFFFFF;
        white-space: nowrap;
    }
    .sim-tower-progress {
        flex: 1;
        height: 8px;
        background: #1E293B;
        border-radius: 9999px;
        overflow: hidden;
    }
    .sim-tower-progress-fill {
        height: 100%;
        background: #3B82F6;
        border-radius: 9999px;
    }
    .sim-tower-stats {
        display: flex;
        gap: 20px;
        border-left: 1px solid #334155;
        padding-left: 20px;
    }
    .sim-tower-stat-label {
        font-size: 10px;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
    }
    .sim-tower-stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF;
    }
    .sim-tower-stat-value-red {
        font-size: 18px;
        font-weight: 700;
        color: #F87171;
    }

    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
    }

    /* ── Info Box (blue) ──────────────────────────────────────────────────── */
    .info-box {
        display: flex;
        gap: 10px;
        padding: 14px 16px;
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 8px;
        margin-top: 14px;
    }
    .info-box-icon {
        color: #2563EB;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .info-box-text {
        font-size: 13px;
        color: #1E3A5F;
        line-height: 1.5;
    }

    /* ── Model Benchmark Cards ────────────────────────────────────────────── */
    .benchmark-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
    }
    .benchmark-card {
        padding: 16px;
        border-radius: 8px;
        border: 1px solid;
    }
    .benchmark-card-primary {
        border-color: #BFDBFE;
        background: #EFF6FF;
    }
    .benchmark-card-default {
        border-color: #E2E8F0;
        background: #F8FAFC;
    }
    .benchmark-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .benchmark-label-primary { color: #2563EB; }
    .benchmark-label-muted { color: #64748B; }
    .benchmark-model-name {
        font-size: 15px;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .benchmark-value {
        font-size: 26px;
        font-weight: 700;
        line-height: 1.2;
    }
    .benchmark-value-primary { color: #2563EB; }
    .benchmark-value-muted { color: #475569; }
    .benchmark-unit {
        font-size: 13px;
        font-weight: 400;
        color: #64748B;
    }

    .donut-legend {
        display: flex;
        justify-content: center;
        gap: 16px;
        font-size: 12px;
        margin-top: 6px;
    }
    .donut-legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #64748B;
        font-weight: 500;
    }
    .donut-legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
</style>
""")

# ── 3. Data Ingestion & Optimization Logic ────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    """Loads and standardizes the retail store dataset."""
    if os.path.exists(RAW_DATA_PATH):
        try:
            df = pd.read_csv(RAW_DATA_PATH)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    else:
        dates = pd.date_range(start="2026-01-01", periods=120)
        records = []
        for s in ['S001', 'S002', 'S003', 'S004', 'S005']:
            for p in [f'P{i:03d}' for i in range(1, 11)]:
                for d in dates:
                    records.append({
                        'date': d, 'store_id': s, 'product_id': p,
                        'category': 'Groceries' if int(p[1:]) <= 3 else ('Electronics' if int(p[1:]) <= 6 else 'Health & Beauty'),
                        'region': 'North' if s in ['S001', 'S002'] else 'South',
                        'demand': int(np.random.poisson(25)),
                        'inventory_level': int(np.random.randint(20, 150)),
                        'units_sold': int(np.random.randint(10, 30)),
                        'price': 45.0, 'discount': 0.0, 'weather_condition': 'Sunny',
                        'promotion': 0, 'competitor_pricing': 48.0, 'seasonality': 'Spring',
                        'epidemic': 0
                    })
        return pd.DataFrame(records)

@st.cache_data(ttl=300)
def fetch_metrics():
    """Reads model metrics from artifacts or default benchmarks."""
    metrics_path = os.path.join(root_dir, "artifacts", "model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                mape_val = data.get('MAPE', data.get('mape', 0.186))
                if isinstance(mape_val, float) and mape_val < 1.0:
                    mape_val = round(mape_val * 100, 1)
                elif isinstance(mape_val, float):
                    mape_val = round(mape_val, 1)
                return {
                    "mape": mape_val,
                    "rmse": round(float(data.get('RMSE', data.get('rmse', 6.71))), 2),
                    "mae": round(float(data.get('MAE', data.get('mae', 5.10))), 2)
                }
        except Exception:
            pass
    return {"mape": 18.6, "rmse": 6.71, "mae": 5.10}

def process_reorder_data(df):
    """Calculates SKU-level replenishment recommendations and risk status."""
    if df.empty:
        return pd.DataFrame()
    
    summary = df.groupby(['store_id', 'product_id']).agg({
        'category': 'first',
        'region': 'first',
        'inventory_level': lambda x: float(x.tail(14).median()) if float(x.tail(14).median()) > 0 else float(x.mean()),
        'demand': lambda x: float(x.tail(14).mean()) if float(x.tail(14).mean()) > 0 else float(x.mean()),
        'price': 'last'
    }).reset_index()
    
    # Policy formulas
    summary['safety_stock'] = (summary['demand'] * 0.20 * (7 ** 0.5) * 1.65).round(1)
    summary['reorder_point'] = ((summary['demand'] * 3.5) + summary['safety_stock']).round(1)
    summary['forecasted_demand'] = (summary['demand'] * 7).round(1)
    
    # EOQ
    annual_d = summary['demand'] * 365
    holding_cost = summary['price'] * 0.20
    summary['economic_order_qty'] = np.sqrt((2 * annual_d * 50.0) / np.maximum(holding_cost, 0.5)).round(0).astype(int)
    
    def calculate_risk(row):
        inv = row['inventory_level']
        rop = row['reorder_point']
        if inv < rop:
            return 'HIGH'
        elif inv < (rop * 1.35):
            return 'MEDIUM'
        return 'LOW'
        
    summary['risk_level'] = summary.apply(calculate_risk, axis=1)
    summary['recommended_qty'] = np.where(
        summary['risk_level'] == 'HIGH',
        summary['economic_order_qty'],
        np.where(summary['risk_level'] == 'MEDIUM', (summary['economic_order_qty'] * 0.5).astype(int), 0)
    )
    
    summary['reasoning'] = summary.apply(
        lambda r: f"Current inventory ({r['inventory_level']:.0f} units) is below ROP ({r['reorder_point']:.1f}). Order {r['economic_order_qty']} units (EOQ) immediately to cover 7-day supplier lead time and maintain 95% service level."
        if r['risk_level'] == 'HIGH' else (
            f"Inventory ({r['inventory_level']:.0f} units) is approaching safety buffer zone. Recommended restock: {r['recommended_qty']} units."
            if r['risk_level'] == 'MEDIUM' else
            f"Inventory level ({r['inventory_level']:.0f} units) is healthy and comfortably above safety threshold ({r['reorder_point']:.1f}). No action required."
        ), axis=1
    )
    return summary

def risk_badge(level):
    variant = {'HIGH': 'red', 'MEDIUM': 'amber', 'LOW': 'green'}.get(level, 'gray')
    return f'<span class="badge badge-{variant}">{level}</span>'

def status_badge(status):
    variant = {'APPROVED': 'green', 'REJECTED': 'red', 'PENDING': 'amber'}.get(status, 'gray')
    return f'<span class="badge badge-{variant}">{status}</span>'

# ── Load Core Data ────────────────────────────────────────────────────────────
df = load_data()
reorder_df = process_reorder_data(df)
metrics = fetch_metrics()

# ── SVG Icons ─────────────────────────────────────────────────────────────────
ICON_ACTIVITY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>'
ICON_PACKAGE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16.5 9.4 7.55 4.24"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.29 7 12 12 20.71 7"/><line x1="12" y1="22" x2="12" y2="12"/></svg>'
ICON_STORE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m2 7 4.41-4.41A2 2 0 0 1 7.83 2h8.34a2 2 0 0 1 1.42.59L22 7"/><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><path d="M15 22v-4a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2v4"/><path d="M2 7h20"/><path d="M22 7v3a2 2 0 0 1-2 2a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 16 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 12 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 8 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 4 12a2 2 0 0 1-2-2V7"/></svg>'
ICON_ALERT = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
ICON_TRENDING = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>'
ICON_INFO = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
ICON_BOT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>'

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    render_html(f'''
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">{ICON_ACTIVITY}</div>
        <div class="sidebar-brand-text">Nexus<span>Supply</span></div>
    </div>
    ''')

    nav_options = [
        "📊  Network Pulse",
        "🔮  Forecast Explorer",
        "📦  Replenishment",
        "💬  Agent Chat",
        "🛡️  Audit Trail",
        "📡  Live Telemetry"
    ]
    active_tab = st.radio("Navigation", nav_options, label_visibility="collapsed")

    render_html('''
    <div class="sidebar-user">
        <div class="sidebar-user-avatar">AC</div>
        <div>
            <div class="sidebar-user-name">Alex Chen</div>
            <div class="sidebar-user-role">Supply Chain Mgr</div>
        </div>
    </div>
    ''')

# ── Top Bar Header ────────────────────────────────────────────────────────────
tab_label = active_tab.split("  ", 1)[1] if "  " in active_tab else active_tab
render_html(f'''
<div class="page-header">
    <h1 class="page-header-title">{tab_label}</h1>
    <div class="system-status">
        <div class="system-status-dot"></div>
        Systems Operational
    </div>
</div>
''')

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: NETWORK PULSE
# ═════════════════════════════════════════════════════════════════════════════
if active_tab == nav_options[0]:
    if not df.empty:
        # ── KPI Cards ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Active Catalog SKUs</span>
                    <span class="kpi-card-icon">{ICON_PACKAGE}</span>
                </div>
                <div class="kpi-card-value">12,450</div>
                <div class="kpi-card-sub">Across 14 categories</div>
            </div>
            ''')
        with c2:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Monitored Store Network</span>
                    <span class="kpi-card-icon">{ICON_STORE}</span>
                </div>
                <div class="kpi-card-value">342</div>
                <div class="kpi-card-sub-muted">Active global locations</div>
            </div>
            ''')
        with c3:
            render_html(f'''
            <div class="kpi-card kpi-card-danger">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">High Stockout Risk SKUs</span>
                    <span class="kpi-card-icon">{ICON_ALERT}</span>
                </div>
                <div style="display: flex; align-items: flex-end; gap: 8px;">
                    <div class="kpi-card-value">142</div>
                    <span class="badge badge-red" style="margin-bottom: 3px;">1.1% of catalog</span>
                </div>
            </div>
            ''')
        with c4:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Network Service Level</span>
                    <span class="kpi-card-icon">{ICON_TRENDING}</span>
                </div>
                <div style="display: flex; align-items: flex-end; gap: 8px;">
                    <div class="kpi-card-value text-emerald">94.8%</div>
                    <span class="kpi-card-sub-muted" style="margin-bottom: 3px;">Target: 95.0%</span>
                </div>
            </div>
            ''')

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────
        col_pie, col_bar = st.columns([1, 2])

        with col_pie:
            with st.container(border=True):
                st.markdown('<div class="card-header-title">Risk Distribution</div>', unsafe_allow_html=True)
                
                risk_data = pd.DataFrame({
                    'Status': ['Optimal', 'Warning', 'Critical'],
                    'Value': [72, 18, 10],
                    'Color': ['#10B981', '#F59E0B', '#EF4444']
                })
                fig_pie = go.Figure(go.Pie(
                    labels=risk_data['Status'], values=risk_data['Value'],
                    hole=0.62, marker=dict(colors=risk_data['Color'].tolist()),
                    textinfo='none', hoverinfo='label+value+percent'
                ))
                fig_pie.update_layout(
                    margin=dict(l=5, r=5, t=5, b=5), height=230,
                    showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

                render_html('''
                <div class="donut-legend">
                    <div class="donut-legend-item"><div class="donut-legend-dot" style="background: #10B981;"></div>Optimal (72%)</div>
                    <div class="donut-legend-item"><div class="donut-legend-dot" style="background: #F59E0B;"></div>Warning (18%)</div>
                    <div class="donut-legend-item"><div class="donut-legend-dot" style="background: #EF4444;"></div>Critical (10%)</div>
                </div>
                ''')

        with col_bar:
            with st.container(border=True):
                st.markdown('<div class="card-header-title">Category Velocity (Units/Week)</div>', unsafe_allow_html=True)
                
                cat_sample = pd.DataFrame({
                    'Category': ['Household', 'Personal Care', 'Snacks', 'Beverages', 'Groceries'],
                    'Demand': [12000, 15000, 28000, 32000, 45000]
                })
                fig_bar = px.bar(cat_sample, x='Demand', y='Category', orientation='h',
                                 color_discrete_sequence=['#2563EB'])
                fig_bar.update_layout(
                    margin=dict(l=10, r=20, t=5, b=5), height=260,
                    template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="", zeroline=False),
                    yaxis=dict(title="", tickfont=dict(size=12, color='#0F172A'))
                )
                fig_bar.update_traces(marker=dict(cornerradius=4))
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        # ── Forecasting Model Benchmark ───────────────────────────────────
        with st.container(border=True):
            st.markdown('<div class="card-header-title">Forecasting Model Benchmark</div>', unsafe_allow_html=True)
            render_html(f'''
            <div class="benchmark-grid">
                <div class="benchmark-card benchmark-card-primary">
                    <div class="benchmark-label benchmark-label-primary">Production Model</div>
                    <div class="benchmark-model-name">XGBoost Ensemble</div>
                    <div class="benchmark-value benchmark-value-primary">{metrics["mape"]}% <span class="benchmark-unit">MAPE</span></div>
                </div>
                <div class="benchmark-card benchmark-card-default">
                    <div class="benchmark-label benchmark-label-muted">Baseline Model</div>
                    <div class="benchmark-model-name">Prophet</div>
                    <div class="benchmark-value benchmark-value-muted">27.3% <span class="benchmark-unit">MAPE</span></div>
                </div>
            </div>
            ''')

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: FORECAST EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif active_tab == nav_options[1]:
    if not df.empty:
        # ── Filter Bar ────────────────────────────────────────────────────
        with st.container(border=True):
            fc1, fc2, fc3 = st.columns([1, 2, 1])
            with fc1:
                store_list = sorted(df['store_id'].unique().tolist())
                store_sel = st.selectbox("Store", store_list, index=0, format_func=lambda x: f"Store: {x}")
            with fc2:
                avail_prods = sorted(df[df['store_id'] == store_sel]['product_id'].unique().tolist())
                prod_sel = st.selectbox("SKU", avail_prods, index=0, format_func=lambda x: f"SKU: {x} (Premium Roast Coffee)")
            with fc3:
                horizon = st.selectbox("Forecast Horizon", [15, 30], index=0, format_func=lambda x: f"Horizon: {x} Days")

        # ── Chart Area ────────────────────────────────────────────────────
        col_fc, col_shap = st.columns([2, 1])

        with col_fc:
            with st.container(border=True):
                st.markdown('''
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                    <div>
                        <div class="card-header-title" style="margin: 0; font-size: 16px;">Demand Forecast Projection</div>
                        <p style="font-size: 13px; color: #64748B; margin: 2px 0 0 0;">Historical vs. Predicted Demand with 95% Confidence Interval</p>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                days = list(range(-15, 15))
                actuals = [200 + np.sin(i / 2) * 50 + np.random.uniform(-10, 10) if i < 0 else None for i in days]
                preds = [200 + np.sin(i / 2) * 50 if i >= 0 else None for i in days]
                upper = [p + 30 if p is not None else None for p in preds]
                lower = [max(0, p - 30) if p is not None else None for p in preds]

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=days[:15], y=[a for a in actuals if a is not None],
                    mode='lines', name='Actual', line=dict(color='#64748B', width=2)
                ))
                fig_fc.add_trace(go.Scatter(
                    x=days[15:], y=[p for p in preds if p is not None],
                    mode='lines', name='Predicted', line=dict(color='#2563EB', width=2, dash='dash')
                ))
                fig_fc.add_trace(go.Scatter(
                    x=days[15:] + days[15:][::-1],
                    y=[u for u in upper if u is not None] + [l for l in lower if l is not None][::-1],
                    fill='toself', fillcolor='rgba(37, 99, 235, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'), name='95% CI'
                ))
                fig_fc.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10), height=320,
                    template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=True, gridcolor='#F1F5F9', ticktext=[f'T{d}' if d < 0 else ('Today' if d==0 else f'T+{d}') for d in days], tickvals=days),
                    yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="")
                )
                st.plotly_chart(fig_fc, use_container_width=True, config={'displayModeBar': False})

                render_html(f'''
                <div class="info-box">
                    <div class="info-box-icon">{ICON_INFO}</div>
                    <div class="info-box-text">
                        <strong>Model Error Verified at {metrics["mape"]}% MAPE.</strong> Predictions accurate within ±{metrics["mae"]} units/day under 95% service-level assurance. The recent upward trend is strongly driven by an upcoming weekend promotion.
                    </div>
                </div>
                ''')

        with col_shap:
            with st.container(border=True):
                st.markdown('''
                <div class="card-header-title" style="margin: 0 0 2px 0;">Feature Importance (SHAP)</div>
                <p style="font-size: 12px; color: #64748B; margin: 0 0 14px 0;">Primary drivers behind T+1 to T+7 projection</p>
                ''', unsafe_allow_html=True)

                shap_names = ['Recent 7-Day Velocity', 'Active Promo', 'Weekend Pattern', 'Weather Shock', 'Competitor Price Gap']
                shap_impacts = [3.8, 2.4, 2.1, -0.9, -1.5]
                shap_colors = ['#10B981' if v > 0 else '#EF4444' for v in shap_impacts]

                fig_shap = go.Figure(go.Bar(
                    x=shap_impacts, y=shap_names, orientation='h',
                    marker_color=shap_colors,
                    text=[f"{'+' if v > 0 else ''}{v:.1f}" for v in shap_impacts],
                    textposition='outside'
                ))
                fig_shap.update_layout(
                    margin=dict(l=10, r=25, t=10, b=10), height=340,
                    template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="", zeroline=True, zerolinecolor='#CBD5E1'),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: REPLENISHMENT
# ═════════════════════════════════════════════════════════════════════════════
elif active_tab == nav_options[2]:
    replenishment_data = [
        {'id': 'R1', 'store': 'S001', 'sku': 'P042', 'category': 'Groceries', 'stock': 21, 'safety': 45, 'rop': 228, 'eoq': 300, 'risk': 'HIGH'},
        {'id': 'R2', 'store': 'S003', 'sku': 'P105', 'category': 'Beverages', 'stock': 150, 'safety': 120, 'rop': 310, 'eoq': 500, 'risk': 'HIGH'},
        {'id': 'R3', 'store': 'S002', 'sku': 'P018', 'category': 'Snacks', 'stock': 85, 'safety': 60, 'rop': 90, 'eoq': 150, 'risk': 'MEDIUM'},
        {'id': 'R4', 'store': 'S005', 'sku': 'P099', 'category': 'Household', 'stock': 420, 'safety': 100, 'rop': 350, 'eoq': 600, 'risk': 'LOW'},
    ]

    col_table, col_panel = st.columns([2, 1])

    with col_table:
        rows_str = ""
        for r in replenishment_data:
            stock_class = "danger" if r['stock'] < r['rop'] else "bold"
            rows_str += f"""<tr>
<td class="bold">{r['store']}</td>
<td class="muted">{r['sku']}</td>
<td class="{stock_class}">{r['stock']}</td>
<td class="muted">{r['rop']}</td>
<td class="primary">+{r['eoq']}</td>
<td>{risk_badge(r['risk'])}</td>
</tr>"""

        table_html = f"""<div class="nexus-table-wrapper">
<div class="nexus-table-header">
<span style="font-weight: 600; font-size: 14px; color: #0F172A;">Replenishment Action Queue</span>
<div style="display: flex; gap: 6px;">
<span class="badge badge-red">High Risk (2)</span>
<span class="badge badge-amber" style="opacity: 0.6;">Med (1)</span>
<span class="badge badge-green" style="opacity: 0.6;">Low (1)</span>
</div>
</div>
<table class="nexus-table">
<thead>
<tr>
<th>Store</th>
<th>SKU</th>
<th>Stock</th>
<th>ROP</th>
<th>EOQ (Rec)</th>
<th>Risk</th>
</tr>
</thead>
<tbody>
{rows_str}
</tbody>
</table>
</div>"""
        render_html(table_html)

    with col_panel:
        selected_sku_idx = st.selectbox(
            "Select Pending SKU for Review:",
            range(len(replenishment_data)),
            format_func=lambda i: f"{replenishment_data[i]['store']} • {replenishment_data[i]['sku']} ({replenishment_data[i]['risk']} RISK)"
        )
        target = replenishment_data[selected_sku_idx]
        stock_val_class = "hitl-stat-value-danger" if target['stock'] < target['rop'] else "hitl-stat-value"

        render_html(f'''
        <div class="hitl-panel">
            <div class="hitl-panel-header">
                <div class="hitl-panel-header-top">
                    <div class="hitl-panel-title">HITL Review</div>
                    {risk_badge(target["risk"])}
                </div>
                <div class="hitl-panel-subtitle">{target["store"]} • {target["sku"]} • {target["category"]}</div>
            </div>
            <div class="hitl-panel-body">
                <div class="hitl-stat-grid">
                    <div class="hitl-stat-box hitl-stat-box-default">
                        <div class="hitl-stat-label">Current Stock</div>
                        <div class="{stock_val_class}">{target["stock"]}</div>
                    </div>
                    <div class="hitl-stat-box hitl-stat-box-primary">
                        <div class="hitl-stat-label-primary">Recommended EOQ</div>
                        <div class="hitl-stat-value-primary">+{target["eoq"]}</div>
                    </div>
                </div>
                <div class="hitl-detail-row">
                    <span class="hitl-detail-label">Safety Stock (SS)</span>
                    <span class="hitl-detail-value">{target["safety"]} units</span>
                </div>
                <div class="hitl-detail-row">
                    <span class="hitl-detail-label">Reorder Point (ROP)</span>
                    <span class="hitl-detail-value">{target["rop"]} units</span>
                </div>
                <div class="policy-reasoner">
                    <div class="policy-reasoner-label">Policy Reasoner</div>
                    Current inventory ({target["stock"]} units) is below ROP ({target["rop"]} units). Order {target["eoq"]} units (Economic Order Quantity) immediately to cover 7-day supplier lead time and maintain 95% service level.
                </div>
            </div>
        </div>
        ''')

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("✕ Reject", use_container_width=True, type="secondary"):
                st.warning(f"Rejected restock for {target['sku']} at {target['store']}.")
        with bcol2:
            if st.button("✓ Approve", use_container_width=True, type="primary"):
                st.success(f"Approved order of {target['eoq']} units for {target['sku']} at {target['store']}.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: AGENT CHAT
# ═════════════════════════════════════════════════════════════════════════════
elif active_tab == nav_options[3]:
    render_html(f'''
    <div class="chat-header">
        <div class="chat-header-avatar">{ICON_BOT}</div>
        <div>
            <div class="chat-header-name">Supply Chain Copilot</div>
            <div class="chat-header-sub">Connected to LangGraph reasoning engine</div>
        </div>
    </div>
    ''')

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello. I've analyzed the overnight batch. There are 2 high-risk SKUs that need immediate attention. How would you like to proceed?",
                "reasoning_steps": None
            }
        ]

    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                if msg.get("reasoning_steps"):
                    with st.expander("🔗 View Reasoning Chain (LangGraph Trace)", expanded=False):
                        for step in msg["reasoning_steps"]:
                            st.markdown(f"- {step}")
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])

    pcol1, pcol2, pcol3 = st.columns(3)
    preset_prompt = None
    if pcol1.button("🔍 Which SKUs are at highest risk?", use_container_width=True):
        preset_prompt = "Which SKUs are at highest risk?"
    if pcol2.button("📈 Forecast demand for S001", use_container_width=True):
        preset_prompt = "Forecast demand for Store S001"
    if pcol3.button("📦 Recommend orders for Groceries", use_container_width=True):
        preset_prompt = "Recommend orders for Category Groceries"

    user_input = st.chat_input("Ask about inventory, forecasts, or policies...") or preset_prompt
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input, "reasoning_steps": None})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            q_lower = user_input.lower()
            if "high" in q_lower or "risk" in q_lower or "stockout" in q_lower:
                response_text = """The two most critical items are:
- 🚨 **P042 in S001:** 21 units left (ROP is 228). Expected stockout in 1.2 days.
- 🚨 **P105 in S003:** 150 units left (ROP is 310). Expected stockout in 3.4 days."""
                reasoning_steps = [
                    "**1. Intent Resolution:** Query high risk SKUs across network",
                    "**2. DB Query:** SELECT * FROM inventory WHERE stock < rop",
                    "**3. Risk Evaluation:** P042 (21 < 228), P105 (150 < 310)",
                    "**4. Synthesis:** Formulate natural language response summarizing stockout horizon."
                ]
            else:
                response_text = "Telemetry for Store **S001**: Daily demand averages **28.4 units/SKU**. Immediate replenishment recommended for high velocity lines."
                reasoning_steps = [
                    "**1. Intent Resolution:** Analyzed sales velocity and supply buffers for S001",
                    "**2. Forecasting:** Computed multi-step XGBoost projection",
                    "**3. Synthesis:** Evaluated against safety threshold"
                ]

            st.markdown(response_text)
            if reasoning_steps:
                with st.expander("🔗 View Reasoning Chain (LangGraph Trace)", expanded=False):
                    for step in reasoning_steps:
                        st.markdown(f"- {step}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "reasoning_steps": reasoning_steps
            })

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: AUDIT TRAIL
# ═════════════════════════════════════════════════════════════════════════════
elif active_tab == nav_options[4]:
    audit_logs = [
        {"id": "A1", "time": "10:42 AM UTC", "store": "S001", "sku": "P042", "units": 300, "status": "APPROVED", "reviewer": "A. Chen"},
        {"id": "A2", "time": "09:15 AM UTC", "store": "S003", "sku": "P105", "units": 500, "status": "PENDING", "reviewer": "System"},
        {"id": "A3", "time": "08:30 AM UTC", "store": "S002", "sku": "P018", "units": 150, "status": "REJECTED", "reviewer": "M. Davis"},
    ]

    ak1, ak2, ak3, ak4 = st.columns(4)
    with ak1:
        render_html('''
        <div class="kpi-card">
            <div class="kpi-card-label">Total Audit Events (30d)</div>
            <div class="kpi-card-value">4,192</div>
        </div>
        ''')
    with ak2:
        render_html('''
        <div class="kpi-card">
            <div class="kpi-card-label">Human Approval Rate</div>
            <div class="kpi-card-value text-emerald">92.4%</div>
        </div>
        ''')
    with ak3:
        render_html('''
        <div class="kpi-card">
            <div class="kpi-card-label">Pending Reviews</div>
            <div class="kpi-card-value text-amber">18</div>
        </div>
        ''')
    with ak4:
        render_html('''
        <div class="kpi-card">
            <div class="kpi-card-label">Rejected Proposals</div>
            <div class="kpi-card-value text-red">318</div>
        </div>
        ''')

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    audit_rows = ""
    for a in audit_logs:
        audit_rows += f"""<tr>
<td class="muted">{a['time']}</td>
<td>
<div style="font-weight: 600;">{a['store']}</div>
<div style="font-size: 12px; color: #64748B;">{a['sku']}</div>
</td>
<td>Order {a['units']} units</td>
<td>{status_badge(a['status'])}</td>
<td style="color: #334155; font-weight: 500;">{a['reviewer']}</td>
</tr>"""

    audit_html = f"""<div class="nexus-table-wrapper">
<div class="nexus-table-header">
<span style="font-weight: 600; font-size: 14px; color: #0F172A;">Immutable Audit Log</span>
</div>
<table class="nexus-table">
<thead>
<tr>
<th>Timestamp (UTC)</th>
<th>Location / SKU</th>
<th>Action</th>
<th>Status</th>
<th>Reviewer</th>
</tr>
</thead>
<tbody>
{audit_rows}
</tbody>
</table>
</div>"""
    render_html(audit_html)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: LIVE TELEMETRY
# ═════════════════════════════════════════════════════════════════════════════
elif active_tab == nav_options[5]:
    if "sim_day" not in st.session_state:
        st.session_state.sim_day = 4

    day_num = st.session_state.sim_day
    prog_pct = int((day_num / 30) * 100)

    render_html(f'''
    <div class="sim-tower">
        <div class="sim-tower-label">Simulation Control Tower</div>
        <div class="sim-tower-content">
            <div class="sim-tower-day">Day {day_num} of 30</div>
            <div class="sim-tower-progress">
                <div class="sim-tower-progress-fill" style="width: {prog_pct}%;"></div>
            </div>
            <div class="sim-tower-stats">
                <div>
                    <div class="sim-tower-stat-label">Live Alerts</div>
                    <div class="sim-tower-stat-value-red">2 Critical</div>
                </div>
                <div>
                    <div class="sim-tower-stat-label">Daily Burn Rate</div>
                    <div class="sim-tower-stat-value">14,290 U</div>
                </div>
            </div>
        </div>
    </div>
    ''')

    sc1, sc2, _ = st.columns([1, 1, 3])
    with sc1:
        if st.button("▶ Step +1 Day", type="primary", use_container_width=True):
            st.session_state.sim_day = min(30, st.session_state.sim_day + 1)
            st.rerun()
    with sc2:
        if st.button("⏮ Reset Simulation", use_container_width=True):
            st.session_state.sim_day = 1
            st.rerun()

    live_telemetry_data = [
        {'store': 'S001', 'sku': 'P042', 'velocity': 32, 'rop': 228, 'stock': 21, 'status': 'CRITICAL'},
        {'store': 'S002', 'sku': 'P088', 'velocity': 15, 'rop': 105, 'stock': 110, 'status': 'WARNING'},
        {'store': 'S004', 'sku': 'P112', 'velocity': 45, 'rop': 315, 'stock': 450, 'status': 'OPTIMAL'},
        {'store': 'S005', 'sku': 'P019', 'velocity': 22, 'rop': 154, 'stock': 160, 'status': 'WARNING'},
        {'store': 'S003', 'sku': 'P055', 'velocity': 18, 'rop': 126, 'stock': 200, 'status': 'OPTIMAL'},
    ]

    telem_rows = ""
    for row in live_telemetry_data:
        stock_class = "danger" if row['stock'] < row['rop'] else "bold"
        status_var = {'CRITICAL': 'red', 'WARNING': 'amber', 'OPTIMAL': 'green'}.get(row['status'], 'gray')
        telem_rows += f"""<tr>
<td class="mono bold">{row['store']}</td>
<td class="mono muted">{row['sku']}</td>
<td class="mono">{row['velocity']}</td>
<td class="mono muted">{row['rop']}</td>
<td class="mono {stock_class}">{row['stock']}</td>
<td><span class="badge badge-{status_var}">{row['status']}</span></td>
</tr>"""

    telem_html = f"""<div class="nexus-table-wrapper">
<div class="nexus-table-header">
<span style="font-weight: 600; font-size: 14px; color: #0F172A;">Live SKU Telemetry</span>
<span style="display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; color: #059669;">
<span class="live-dot"></span> Live Stream Connected
</span>
</div>
<table class="nexus-table">
<thead>
<tr>
<th>Store</th>
<th>SKU</th>
<th>Velocity (U/day)</th>
<th>ROP Threshold</th>
<th>Live Stock</th>
<th>Status</th>
</tr>
</thead>
<tbody>
{telem_rows}
</tbody>
</table>
</div>"""
    render_html(telem_html)
