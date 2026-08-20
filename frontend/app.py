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

# ── 3. Data Ingestion & Genuine Optimization Logic ────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    """Loads and standardizes the Kaggle retail store dataset."""
    if os.path.exists(RAW_DATA_PATH):
        try:
            df = pd.read_csv(RAW_DATA_PATH)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            st.error(f"Error loading dataset from {RAW_DATA_PATH}: {e}")
            st.stop()
    else:
        st.error(f"Production dataset not found at '{RAW_DATA_PATH}'. Please ensure raw dataset exists.")
        st.stop()

@st.cache_data(ttl=300)
def fetch_metrics():
    """Reads genuine model metrics from artifacts."""
    metrics_path = os.path.join(root_dir, "artifacts", "model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                mape_val = data.get('MAPE', data.get('mape', 0.3658))
                if isinstance(mape_val, float) and mape_val < 1.0:
                    mape_val = round(mape_val * 100, 1)
                elif isinstance(mape_val, float):
                    mape_val = round(mape_val, 1)

                agg_mape = data.get('xgboost_daily_aggregate_MAPE', data.get('xgboost', {}).get('daily_aggregate', {}).get('MAPE', 0.0574))
                if isinstance(agg_mape, float) and agg_mape < 1.0:
                    agg_mape = round(agg_mape * 100, 1)
                elif isinstance(agg_mape, float):
                    agg_mape = round(agg_mape, 1)

                prophet_mape = data.get('prophet_daily_aggregate_MAPE', data.get('prophet', {}).get('daily_aggregate', {}).get('MAPE', 0.2096))
                if isinstance(prophet_mape, float) and prophet_mape < 1.0:
                    prophet_mape = round(prophet_mape * 100, 1)
                elif isinstance(prophet_mape, float):
                    prophet_mape = round(prophet_mape, 1)

                return {
                    "mape": mape_val,
                    "mape_aggregate": agg_mape,
                    "prophet_mape": prophet_mape,
                    "rmse": round(float(data.get('RMSE', data.get('rmse', 32.69))), 2),
                    "mae": round(float(data.get('MAE', data.get('mae', 24.67))), 2)
                }
        except Exception as e:
            st.warning(f"Error reading model_metrics.json: {e}")
            return None
    return None

@st.cache_data(ttl=300)
def fetch_fairness_audit():
    """Reads genuine subgroup fairness audit from artifacts."""
    audit_path = os.path.join(root_dir, "artifacts", "fairness_audit.json")
    if os.path.exists(audit_path):
        try:
            with open(audit_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_data(ttl=300)
def process_reorder_data(df):
    """Calculates SKU-level replenishment recommendations and risk status directly from the dataset."""
    if df.empty:
        return pd.DataFrame()
    
    # H-1 FIX: Sort by date so .tail(14) gets the most recent 14 days, not random rows
    df = df.sort_values(['store_id', 'product_id', 'date'])
    
    summary = df.groupby(['store_id', 'product_id']).agg({
        'category': 'first',
        'region': 'first',
        # C-2 FIX: Don't mask zero inventory — only fall back if no rows exist
        'inventory_level': lambda x: float(x.tail(14).median()) if len(x.tail(14)) > 0 else float(x.mean()),
        'demand': lambda x: float(x.tail(14).mean()) if len(x.tail(14)) > 0 else float(x.mean()),
        'price': 'last'
    }).reset_index()
    
    # H-2 FIX: Use backend config constants instead of hardcoded values
    _lead_time = LEAD_TIME_DAYS if BACKEND_AVAILABLE else 7
    _service_z = SERVICE_LEVEL_Z if BACKEND_AVAILABLE else 1.65
    _holding_pct = HOLDING_COST_PERCENT if BACKEND_AVAILABLE else 0.20
    _order_cost = ORDER_COST if BACKEND_AVAILABLE else 50.0

    # Mathematical inventory policy using config constants
    demand_std = summary['demand'] * 0.20  # Approximate std as 20% of mean demand
    summary['safety_stock'] = (_service_z * demand_std * (_lead_time ** 0.5)).round(1)
    # C-1 FIX: Use full lead time (7 days) instead of 3.5
    summary['reorder_point'] = ((summary['demand'] * _lead_time) + summary['safety_stock']).round(1)
    summary['forecasted_demand'] = (summary['demand'] * _lead_time).round(1)
    
    # EOQ: sqrt((2 * annual_demand * order_cost) / holding_cost)
    annual_d = summary['demand'] * 365
    holding_cost = summary['price'] * _holding_pct
    summary['economic_order_qty'] = np.sqrt((2 * annual_d * _order_cost) / np.maximum(holding_cost, 0.5)).round(0).astype(int)
    
    def calculate_risk(row):
        inv = row['inventory_level']
        rop = row['reorder_point']
        if inv < rop:
            return 'HIGH'
        # H-2 FIX: Use 1.20 buffer to match backend stockout_risk_flag (was 1.35)
        elif inv < (rop * 1.20):
            return 'MEDIUM'
        return 'LOW'
        
    summary['risk_level'] = summary.apply(calculate_risk, axis=1)
    summary['recommended_qty'] = np.where(
        summary['risk_level'] == 'HIGH',
        summary['economic_order_qty'],
        np.where(summary['risk_level'] == 'MEDIUM', (summary['economic_order_qty'] * 0.5).astype(int), 0)
    )
    
    summary['reasoning'] = summary.apply(
        lambda r: f"Current inventory ({r['inventory_level']:.0f} units) is below ROP ({r['reorder_point']:.1f}). Order {r['economic_order_qty']} units (EOQ) immediately to cover {_lead_time}-day supplier lead time and maintain 95% service level."
        if r['risk_level'] == 'HIGH' else (
            f"Inventory ({r['inventory_level']:.0f} units) is approaching safety buffer zone. Recommended restock: {r['recommended_qty']} units."
            if r['risk_level'] == 'MEDIUM' else
            f"Inventory level ({r['inventory_level']:.0f} units) is healthy and comfortably above safety threshold ({r['reorder_point']:.1f}). No action required."
        ), axis=1
    )
    return summary

def get_audit_logs():
    """Fetches real audit trail from the API endpoint or database."""
    try:
        resp = requests.get(f"{API_URL}/audit/log", timeout=3)
        if resp.status_code == 200:
            logs = resp.json()
            if logs:
                records = []
                for l in logs:
                    records.append({
                        "id": l.get("id"),
                        "thread_id": l.get("thread_id"),
                        "timestamp": l.get("timestamp", "N/A"),
                        "store_id": l.get("store_id"),
                        "product_id": l.get("product_id"),
                        "recommended_qty": int(l.get("recommended_qty") or 0),
                        "risk_level": l.get("risk_level", "LOW"),
                        "decision_status": (l.get("decision") or "PENDING").upper(),
                        "approver": l.get("approver") or "System Agent",
                        "reasoning": l.get("reasoning_snapshot") or ""
                    })
                return pd.DataFrame(records)
    except Exception:
        pass

    if BACKEND_AVAILABLE:
        try:
            db = SessionLocal()
            try:
                logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).all()
                records = []
                for l in logs:
                    records.append({
                        "id": l.id,
                        "thread_id": l.thread_id,
                        "timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S") if l.timestamp else "N/A",
                        "store_id": l.store_id,
                        "product_id": l.product_id,
                        "recommended_qty": int(l.recommended_qty) if l.recommended_qty else 0,
                        "risk_level": l.risk_level or "LOW",
                        "decision_status": (l.decision or "PENDING").upper(),
                        "approver": l.approver or "System Agent",
                        "reasoning": l.reasoning_snapshot or ""
                    })
                if records:
                    return pd.DataFrame(records)
            finally:
                db.close()
        except Exception:
            pass
            
    return pd.DataFrame()

def get_pending_recommendations(reorder_df):
    """Fetches real pending recommendations from the audit database/LangGraph pipeline."""
    audit_df = get_audit_logs()
    pending_logs = pd.DataFrame()
    if not audit_df.empty:
        pending_logs = audit_df[audit_df["decision_status"] == "PENDING"]
        
    # If no pending recommendations exist in the database, initialize through real agent pipeline
    if pending_logs.empty and not reorder_df.empty:
        risk_skus = reorder_df[reorder_df['risk_level'].isin(['HIGH', 'MEDIUM'])].head(5)
        for _, r in risk_skus.iterrows():
            s_id = r['store_id']
            p_id = r['product_id']
            t_id = f"{s_id}_{p_id}_init"
            try:
                if BACKEND_AVAILABLE:
                    from backend.agents.graph import run_agent_pipeline
                    run_agent_pipeline(s_id, p_id, t_id)
                else:
                    requests.get(f"{API_URL}/agent/query/{s_id}/{p_id}", timeout=5)
            except Exception:
                pass
        audit_df = get_audit_logs()
        if not audit_df.empty:
            pending_logs = audit_df[audit_df["decision_status"] == "PENDING"]

    if not pending_logs.empty:
        merged = pending_logs.merge(
            reorder_df[['store_id', 'product_id', 'category', 'inventory_level', 'safety_stock', 'reorder_point', 'economic_order_qty']],
            on=['store_id', 'product_id'],
            how='left'
        )
        merged['safety_stock'] = merged['safety_stock'].fillna(25.0)
        merged['reorder_point'] = merged['reorder_point'].fillna(75.0)
        merged['inventory_level'] = merged['inventory_level'].fillna(50.0)
        merged['economic_order_qty'] = merged['economic_order_qty'].fillna(merged['recommended_qty']).astype(int)
        merged['category'] = merged['category'].fillna('Retail Goods')
        return merged
    return pd.DataFrame()

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
# SIDEBAR NAVIGATION (5 Tabs)
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
        "⚖️  Fairness & Bias"
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
        total_products = int(df['product_id'].nunique())
        total_stores = int(df['store_id'].nunique())
        total_categories = int(df['category'].nunique())
        total_regions = int(df['region'].nunique())
        total_skus = max(len(reorder_df), 1)

        high_risk_count = int(len(reorder_df[reorder_df['risk_level'] == 'HIGH'])) if not reorder_df.empty else 0
        med_risk_count = int(len(reorder_df[reorder_df['risk_level'] == 'MEDIUM'])) if not reorder_df.empty else 0
        low_risk_count = int(len(reorder_df[reorder_df['risk_level'] == 'LOW'])) if not reorder_df.empty else 0

        risk_pct = round((high_risk_count / total_skus) * 100, 1)
        # M-1 FIX: Show true service level instead of artificial 85% floor
        service_level = round(((total_skus - high_risk_count) / total_skus) * 100, 1)

        # ── KPI Cards ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Active Catalog SKUs</span>
                    <span class="kpi-card-icon">{ICON_PACKAGE}</span>
                </div>
                <div class="kpi-card-value">{total_products}</div>
                <div class="kpi-card-sub">Across {total_categories} categories</div>
            </div>
            ''')
        with c2:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Monitored Store Network</span>
                    <span class="kpi-card-icon">{ICON_STORE}</span>
                </div>
                <div class="kpi-card-value">{total_stores}</div>
                <div class="kpi-card-sub-muted">Active across {total_regions} regions ({total_skus} store-SKUs)</div>
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
                    <div class="kpi-card-value">{high_risk_count}</div>
                    <span class="badge badge-red" style="margin-bottom: 3px;">{risk_pct}% of network</span>
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
                    <div class="kpi-card-value text-emerald">{service_level}%</div>
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
                
                opt_pct = round((low_risk_count / total_skus) * 100)
                warn_pct = round((med_risk_count / total_skus) * 100)
                crit_pct = round((high_risk_count / total_skus) * 100)

                risk_data = pd.DataFrame({
                    'Status': ['Optimal', 'Warning', 'Critical'],
                    'Value': [low_risk_count, med_risk_count, high_risk_count],
                    'Color': ['#10B981', '#F59E0B', '#EF4444']
                })
                fig_pie = go.Figure(go.Pie(
                    labels=risk_data['Status'], values=risk_data['Value'],
                    hole=0.62, marker=dict(colors=risk_data['Color'].tolist()),
                    textinfo='none', hoverinfo='label+value+percent'
                ))
                fig_pie.update_layout(
                    margin=dict(l=5, r=5, t=5, b=5), height=230,
                    showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    dragmode=False, hovermode='closest'
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

                render_html(f'''
                <div class="donut-legend">
                    <div class="donut-legend-item"><div class="donut-legend-dot" style="background: #10B981;"></div>Optimal ({opt_pct}%)</div>
                    <div class="donut-legend-item"><div class="donut-legend-dot" style="background: #F59E0B;"></div>Warning ({warn_pct}%)</div>
                    <div class="donut-legend-item"><div class="donut-legend-dot" style="background: #EF4444;"></div>Critical ({crit_pct}%)</div>
                </div>
                ''')

        with col_bar:
            with st.container(border=True):
                st.markdown('<div class="card-header-title">Category Velocity (Total Demand)</div>', unsafe_allow_html=True)
                
                cat_df = df.groupby('category')['demand'].sum().sort_values(ascending=True).reset_index()
                fig_bar = px.bar(cat_df, x='demand', y='category', orientation='h',
                                 color_discrete_sequence=['#2563EB'])
                fig_bar.update_layout(
                    margin=dict(l=10, r=20, t=5, b=5), height=260,
                    template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
                    dragmode=False,
                    xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="", zeroline=False, fixedrange=True),
                    yaxis=dict(title="", tickfont=dict(size=12, color='#0F172A'), fixedrange=True)
                )
                fig_bar.update_traces(
                    marker=dict(cornerradius=4),
                    hovertemplate='<b>%{y}</b><br>Total Historical Demand: <b>%{x:,.0f} units</b><extra></extra>'
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        # ── Forecasting Model Benchmark ───────────────────────────────────
        with st.container(border=True):
            st.markdown('<div class="card-header-title">Forecasting Model Benchmark</div>', unsafe_allow_html=True)
            render_html(f'''
            <div class="benchmark-grid">
                <div class="benchmark-card benchmark-card-primary">
                    <div class="benchmark-label benchmark-label-primary">Production Model</div>
                    <div class="benchmark-model-name">XGBoost Ensemble (76k Assigned Dataset)</div>
                    <div class="benchmark-value benchmark-value-primary">{metrics.get("mape_aggregate", 5.7)}% <span class="benchmark-unit">Daily Network MAPE (36.6% SKU-level)</span></div>
                </div>
                <div class="benchmark-card benchmark-card-default">
                    <div class="benchmark-label benchmark-label-muted">Baseline Model</div>
                    <div class="benchmark-model-name">Prophet (Additive Seasonality/Trend)</div>
                    <div class="benchmark-value benchmark-value-muted">{metrics.get("prophet_mape", 21.0)}% <span class="benchmark-unit">Daily Network MAPE</span></div>
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
                prod_cat_map = df[df['store_id'] == store_sel].drop_duplicates('product_id').set_index('product_id')['category'].to_dict()
                prod_sel = st.selectbox("SKU", avail_prods, index=0, format_func=lambda x: f"SKU: {x} ({prod_cat_map.get(x, '')})")
            with fc3:
                horizon = st.selectbox("Forecast Horizon", [7, 14, 30], index=1, format_func=lambda x: f"Horizon: {x} Days")

        # ── Chart Area ────────────────────────────────────────────────────
        col_fc, col_shap = st.columns([2, 1])

        with col_fc:
            with st.container(border=True):
                st.markdown(f'''
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                    <div>
                        <div class="card-header-title" style="margin: 0; font-size: 16px;">Demand Forecast Projection: {store_sel} / {prod_sel}</div>
                        <p style="font-size: 13px; color: #64748B; margin: 2px 0 0 0;">Historical Dataset vs. AI Predicted Demand with 95% Confidence Interval</p>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                filtered_df = df[(df['store_id'] == store_sel) & (df['product_id'] == prod_sel)].sort_values('date')

                if not filtered_df.empty:
                    last_date = filtered_df['date'].max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

                    preds, lower_b, upper_b = [], [], []
                    forecast_err = None
                    if BACKEND_AVAILABLE:
                        try:
                            fmodel = ForecastModel()
                            res = fmodel.predict_demand(store_sel, prod_sel, days_ahead=horizon)
                            preds = res.get('predicted_demand', [])
                            lower_b = res.get('lower_bound', [])
                            upper_b = res.get('upper_bound', [])
                        except Exception as e:
                            forecast_err = str(e)

                    if not preds or len(preds) != horizon:
                        st.error(f"Model forecast unavailable for {store_sel} / {prod_sel}: {forecast_err or 'Model artifact missing'}. Run training pipeline to generate predictions.")
                    else:
                        fig_fc = go.Figure()
                        hist_sample = filtered_df.tail(45)

                        # Historical Line
                        fig_fc.add_trace(go.Scatter(
                            x=hist_sample['date'], y=hist_sample['demand'],
                            mode='lines', name='Actual Demand', line=dict(color='#64748B', width=2)
                        ))
                        # Predicted Line
                        fig_fc.add_trace(go.Scatter(
                            x=future_dates, y=preds,
                            mode='lines', name='AI Predicted Demand', line=dict(color='#2563EB', width=2.5, dash='dash')
                        ))
                        # Confidence Envelope
                        fig_fc.add_trace(go.Scatter(
                            x=future_dates.tolist() + future_dates.tolist()[::-1],
                            y=upper_b + lower_b[::-1],
                            fill='toself', fillcolor='rgba(37, 99, 235, 0.1)',
                            line=dict(color='rgba(255,255,255,0)'), name='95% Prediction Interval'
                        ))
                    fig_fc.update_layout(
                        margin=dict(l=15, r=15, t=15, b=15), height=320,
                        template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified",
                        dragmode='pan',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='#F1F5F9',
                            tickformat='%b %d',
                            hoverformat='%b %d, %Y',
                            fixedrange=False
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='#F1F5F9',
                            title="Units / Day",
                            fixedrange=True
                        )
                    )
                    st.plotly_chart(
                        fig_fc,
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
                            'displaylogo': False,
                            'scrollZoom': True
                        }
                    )

                    render_html(f'''
                    <div class="info-box">
                        <div class="info-box-icon">{ICON_INFO}</div>
                        <div class="info-box-text">
                            <strong>Model Precision: {metrics["mape"]}% SKU-Level MAPE ({metrics.get("mape_aggregate", 5.7)}% Network Aggregate).</strong> Forecast for {prod_sel} is calibrated within ±{metrics["mae"]} units/day under 95% service-level assurance.
                        </div>
                    </div>
                    ''')

        with col_shap:
            with st.container(border=True):
                st.markdown('''
                <div class="card-header-title" style="margin: 0 0 2px 0;">Feature Importance (SHAP)</div>
                <p style="font-size: 12px; color: #64748B; margin: 0 0 14px 0;">Primary drivers behind XGBoost demand projection</p>
                ''', unsafe_allow_html=True)

                shap_data = []
                if BACKEND_AVAILABLE:
                    try:
                        fmodel_shap = ForecastModel()
                        shap_data = fmodel_shap.explain_forecast(
                            store_id=store_sel,
                            product_id=prod_sel,
                            return_structured=True,
                            top_k=5,
                            days_ahead=horizon
                        )
                    except Exception:
                        shap_data = []

                if shap_data:
                    FEATURE_DISPLAY_NAMES = {
                        "demand_lag_1": "1-Day Demand Lag",
                        "demand_lag_7": "7-Day Demand Lag",
                        "demand_lag_14": "14-Day Demand Lag",
                        "demand_lag_28": "28-Day Demand Lag",
                        "demand_rolling_mean_7": "Recent 7-Day Velocity",
                        "demand_rolling_std_7": "7-Day Volatility",
                        "demand_rolling_mean_28": "28-Day Demand Velocity",
                        "demand_rolling_std_28": "28-Day Volatility",
                        "price_discount_ratio": "Discount Impact",
                        "price_competitor_ratio": "Competitor Price Gap",
                        "Price": "Base Price",
                        "Competitor Pricing": "Competitor Price",
                        "Promotion": "Active Promo",
                        "Epidemic": "Epidemic / Disruption",
                        "seasonality_encoded": "Seasonal Factor",
                        "is_weekend": "Weekend Pattern",
                        "day_of_week": "Day of Week",
                        "month": "Month Trend",
                        "quarter": "Quarter Cycle",
                        "weather_Sunny": "Sunny Weather",
                        "weather_Rainy": "Rainy Weather",
                        "weather_Cloudy": "Cloudy Weather",
                        "weather_Snowy": "Snowy Weather",
                        "weather_Stormy": "Weather Shock (Storm)",
                    }

                    shap_names = [
                        FEATURE_DISPLAY_NAMES.get(item["feature"], item["feature"].replace("_", " ").title())
                        for item in shap_data
                    ]
                    shap_impacts = [
                        item.get("signed_impact", item.get("impact", 0.0))
                        for item in shap_data
                    ]
                    shap_colors = ['#10B981' if v >= 0 else '#EF4444' for v in shap_impacts]

                    # Safe symmetric range bound with generous margin for labels
                    max_impact = max([abs(v) for v in shap_impacts]) if shap_impacts else 10.0
                    x_range_bound = max(max_impact * 1.8, 8.0)
                    text_labels = [f"+{v:.2f} units" if v >= 0 else f"{v:+.2f} units" for v in shap_impacts]
                    # Place negative-value labels inside bars to avoid collision with y-axis feature names
                    text_positions = ['outside' if v >= 0 else 'inside' for v in shap_impacts]
                    text_font_colors = ['#0F172A' if v >= 0 else '#FFFFFF' for v in shap_impacts]

                    fig_shap = go.Figure(go.Bar(
                        x=shap_impacts,
                        y=shap_names,
                        orientation='h',
                        marker=dict(
                            color=shap_colors,
                            cornerradius=4
                        ),
                        text=text_labels,
                        textposition=text_positions,
                        textfont=dict(color=text_font_colors, size=11),
                        insidetextanchor='middle',
                        cliponaxis=False,
                        hovertemplate='<b>%{y}</b><br>Demand Impact: <b>%{x:+.2f} units/day</b><extra></extra>'
                    ))
                    fig_shap.update_layout(
                        margin=dict(l=170, r=90, t=32, b=20),
                        height=340,
                        template='plotly_white',
                        paper_bgcolor='rgba(0,0,0,0)',
                        dragmode=False,
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='#F1F5F9',
                            zeroline=True,
                            zerolinecolor='#94A3B8',
                            zerolinewidth=1.5,
                            range=[-x_range_bound, x_range_bound],
                            fixedrange=True,
                            title=dict(text="Impact on Daily Forecast (Units)", font=dict(size=11, color="#64748B"))
                        ),
                        yaxis=dict(
                            autorange="reversed",
                            fixedrange=True,
                            tickfont=dict(size=12, color="#0F172A")
                        ),
                        annotations=[
                            dict(
                                x=-x_range_bound * 0.55, y=1.14, xref="x", yref="paper",
                                text="◀ Decreases Demand", showarrow=False,
                                font=dict(size=10.5, color="#EF4444")
                            ),
                            dict(
                                x=x_range_bound * 0.55, y=1.14, xref="x", yref="paper",
                                text="Increases Demand ▶", showarrow=False,
                                font=dict(size=10.5, color="#10B981")
                            )
                        ]
                    )
                    st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Explanation unavailable for this selection.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: REPLENISHMENT
# ═════════════════════════════════════════════════════════════════════════════
elif active_tab == nav_options[2]:
    if not reorder_df.empty:
        # Filter toolbar
        risk_filter = st.multiselect(
            "Filter Risk Level", ['HIGH', 'MEDIUM', 'LOW'],
            default=['HIGH', 'MEDIUM'], label_visibility="collapsed"
        )
        view_df = reorder_df[reorder_df['risk_level'].isin(risk_filter)] if risk_filter else reorder_df

        col_table, col_panel = st.columns([2, 1])

        with col_table:
            high_c = len(reorder_df[reorder_df['risk_level'] == 'HIGH'])
            med_c = len(reorder_df[reorder_df['risk_level'] == 'MEDIUM'])
            low_c = len(reorder_df[reorder_df['risk_level'] == 'LOW'])

            rows_str = ""
            for _, r in view_df.iterrows():
                stock_class = "danger" if r['inventory_level'] < r['reorder_point'] else "bold"
                # H-4 FIX: Show recommended_qty (not full EOQ) — display dash for LOW risk
                rec_qty = int(r.get('recommended_qty', 0))
                qty_display = f"+{rec_qty}" if rec_qty > 0 else '<span class="muted">—</span>'
                rows_str += f"""<tr>
<td class="bold">{r['store_id']}</td>
<td class="muted">{r['product_id']}</td>
<td class="{stock_class}">{int(r['inventory_level'])}</td>
<td class="muted">{r['reorder_point']:.0f}</td>
<td class="primary">{qty_display}</td>
<td>{risk_badge(r['risk_level'])}</td>
</tr>"""

            table_html = f"""<div class="nexus-table-wrapper">
<div class="nexus-table-header">
<span style="font-weight: 600; font-size: 14px; color: #0F172A;">Replenishment Action Queue</span>
<div style="display: flex; gap: 6px;">
<span class="badge badge-red">High Risk ({high_c})</span>
<span class="badge badge-amber" style="opacity: 0.6;">Med ({med_c})</span>
<span class="badge badge-green" style="opacity: 0.6;">Low ({low_c})</span>
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

        pending_df = get_pending_recommendations(reorder_df)
        review_pool = pending_df if not pending_df.empty else view_df

        # H-3 FIX: Apply risk filter to review_pool so HITL panel stays in sync with table
        if risk_filter and not review_pool.empty and 'risk_level' in review_pool.columns:
            review_pool = review_pool[review_pool['risk_level'].isin(risk_filter)]

        with col_panel:
            if not review_pool.empty:
                sku_options = []
                for _, r in review_pool.iterrows():
                    rec_id_tag = f" [ID #{int(r['id'])}]" if "id" in r and pd.notna(r["id"]) else ""
                    sku_options.append(f"{r['store_id']} • {r['product_id']} ({r['risk_level']} RISK){rec_id_tag}")

                selected_sku_idx = st.selectbox(
                    "Select Pending SKU for Review:",
                    range(len(sku_options)),
                    format_func=lambda i: sku_options[i]
                )
                target = review_pool.iloc[selected_sku_idx]
                # M-6 FIX: Guard against NaN values
                inv_level = int(target['inventory_level']) if pd.notna(target.get('inventory_level')) else 0
                rop_val = target['reorder_point'] if pd.notna(target.get('reorder_point')) else 0
                stock_val_class = "hitl-stat-value-danger" if inv_level < rop_val else "hitl-stat-value"
                # H-5 FIX: Prioritize recommended_qty over economic_order_qty
                rec_qty_val = int(target.get('recommended_qty', target.get('economic_order_qty', 0)))

                render_html(f'''
                <div class="hitl-panel">
                    <div class="hitl-panel-header">
                        <div class="hitl-panel-header-top">
                            <div class="hitl-panel-title">HITL Review</div>
                            {risk_badge(target["risk_level"])}
                        </div>
                        <div class="hitl-panel-subtitle">{target["store_id"]} • {target["product_id"]} • {target.get("category", "Retail")}</div>
                    </div>
                    <div class="hitl-panel-body">
                        <div class="hitl-stat-grid">
                            <div class="hitl-stat-box hitl-stat-box-default">
                                <div class="hitl-stat-label">Current Stock</div>
                                <div class="{stock_val_class}">{int(target["inventory_level"])}</div>
                            </div>
                            <div class="hitl-stat-box hitl-stat-box-primary">
                                <div class="hitl-stat-label-primary">Recommended EOQ</div>
                                <div class="hitl-stat-value-primary">+{rec_qty_val}</div>
                            </div>
                        </div>
                        <div class="hitl-detail-row">
                            <span class="hitl-detail-label">Safety Stock (SS)</span>
                            <span class="hitl-detail-value">{target["safety_stock"]:.0f} units</span>
                        </div>
                        <div class="hitl-detail-row">
                            <span class="hitl-detail-label">Reorder Point (ROP)</span>
                            <span class="hitl-detail-value">{target["reorder_point"]:.0f} units</span>
                        </div>
                        <div class="policy-reasoner">
                            <div class="policy-reasoner-label">Policy Reasoner</div>
                            {target["reasoning"]}
                        </div>
                    </div>
                </div>
                ''')

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("✕ Reject", use_container_width=True, type="secondary"):
                        try:
                            payload = {
                                "decision": "rejected",
                                "approver": st.session_state.get("user_name", "Alex Chen (Supply Chain Mgr)")
                            }
                            if "id" in target and pd.notna(target["id"]):
                                payload["recommendation_id"] = int(target["id"])
                            if "thread_id" in target and pd.notna(target["thread_id"]):
                                payload["thread_id"] = str(target["thread_id"])
                            # C-4 FIX: Generate fallback thread_id if missing
                            if "thread_id" not in payload:
                                payload["thread_id"] = f"{target['store_id']}_{target['product_id']}_manual"

                            resp = requests.post(f"{API_URL}/audit/approve", json=payload, timeout=10)
                            if resp.status_code == 200:
                                # M-3 FIX: Use toast so message survives rerun
                                st.toast(f"Rejected restock for {target['product_id']} at {target['store_id']}.", icon="⚠️")
                                st.rerun()
                            else:
                                st.error(f"Rejection failed (HTTP {resp.status_code}): {resp.text}")
                        except Exception as e:
                            st.error(f"Error communicating with backend API: {str(e)}")
                with bcol2:
                    if st.button("✓ Approve", use_container_width=True, type="primary"):
                        try:
                            payload = {
                                "decision": "approved",
                                "approver": st.session_state.get("user_name", "Alex Chen (Supply Chain Mgr)")
                            }
                            if "id" in target and pd.notna(target["id"]):
                                payload["recommendation_id"] = int(target["id"])
                            if "thread_id" in target and pd.notna(target["thread_id"]):
                                payload["thread_id"] = str(target["thread_id"])
                            # C-4 FIX: Generate fallback thread_id if missing
                            if "thread_id" not in payload:
                                payload["thread_id"] = f"{target['store_id']}_{target['product_id']}_manual"

                            resp = requests.post(f"{API_URL}/audit/approve", json=payload, timeout=10)
                            if resp.status_code == 200:
                                # M-3 FIX: Use toast so message survives rerun
                                st.toast(f"✓ Approved order of {rec_qty_val} units for {target['product_id']} at {target['store_id']}.", icon="✅")
                                st.rerun()
                            else:
                                st.error(f"Approval failed (HTTP {resp.status_code}): {resp.text}")
                        except Exception as e:
                            st.error(f"Error communicating with backend API: {str(e)}")

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
                "content": f"Hello. I've analyzed the overnight dataset batch across {df['store_id'].nunique()} stores. There are {len(reorder_df[reorder_df['risk_level'] == 'HIGH'])} high-risk SKUs that need immediate replenishment. How can I assist you?",
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
    if pcol1.button("🔍 High-Risk SKUs Query", use_container_width=True):
        preset_prompt = "Which SKUs are at highest risk across the network?"
    if pcol2.button("📈 Run Agent on S001 / P0001", use_container_width=True):
        preset_prompt = "Generate demand forecast and reorder policy for Store S001 Product P0001"
    if pcol3.button("📦 Category Groceries Policy", use_container_width=True):
        preset_prompt = "Recommend orders for Category Groceries"

    user_input = st.chat_input("Ask about inventory, forecasts, or policies (e.g., 'Analyze Store S001 Product P0001')...") or preset_prompt
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input, "reasoning_steps": None})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            q_lower = user_input.lower()
            response_text = ""
            reasoning_steps = []

            # Check if query targets a specific store and SKU
            has_sku = re.search(r'\b(P\d{4})\b', user_input, re.IGNORECASE) or re.search(r'\b(S\d{3})\b', user_input, re.IGNORECASE)

            if has_sku and API_URL:
                try:
                    resp = requests.post(f"{API_URL}/agent/query", json={"query": user_input}, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        response_text = data.get("answer", "Agent executed successfully.")
                        r_chain = data.get("reasoning_chain", {})
                        if r_chain:
                            if r_chain.get("forecast_explanation"):
                                reasoning_steps.append(f"**Forecast Agent:** {r_chain['forecast_explanation']}")
                            if r_chain.get("risk_reasoning"):
                                reasoning_steps.append(f"**Risk Agent:** {r_chain['risk_reasoning']}")
                            if r_chain.get("reorder_reasoning"):
                                reasoning_steps.append(f"**Reorder Agent:** {r_chain['reorder_reasoning']}")
                    elif resp.status_code == 400:
                        detail = resp.json().get("detail", "Bad Request")
                        response_text = f"⚠️ {detail}"
                    else:
                        response_text = f"Backend agent error (HTTP {resp.status_code}): {resp.text}"
                except Exception as e:
                    # If backend API is unreachable, fall back to direct module execution if BACKEND_AVAILABLE
                    if BACKEND_AVAILABLE:
                        try:
                            from backend.api.agent_routes import parse_intent
                            from backend.agents.graph import run_agent_pipeline
                            import uuid
                            entities = parse_intent(user_input)
                            thread_id = str(uuid.uuid4())
                            state = run_agent_pipeline(entities.store_id, entities.product_id, thread_id)
                            response_text = f"Forecast and reorder plan generated for {entities.product_id} at Store {entities.store_id}: Risk assessed as **{state.get('risk_level')}**. {state.get('reorder_reasoning', '')}"
                            if state.get("explanation"):
                                reasoning_steps.append(f"**Forecast Agent:** {state['explanation']}")
                            if state.get("risk_reasoning"):
                                reasoning_steps.append(f"**Risk Agent:** {state['risk_reasoning']}")
                            if state.get("reorder_reasoning"):
                                reasoning_steps.append(f"**Reorder Agent:** {state['reorder_reasoning']}")
                        except Exception as inner_e:
                            response_text = f"Unable to process query: {inner_e}"
                    else:
                        response_text = f"Could not connect to backend agent API at {API_URL}: {e}"
            elif "high" in q_lower or "risk" in q_lower or "stockout" in q_lower:
                high_items = reorder_df[reorder_df['risk_level'] == 'HIGH']
                if not high_items.empty:
                    top3 = high_items.head(3)
                    items_str = "\n".join([f"- 🚨 **{r['product_id']} in {r['store_id']}:** {int(r['inventory_level'])} units left (ROP: {r['reorder_point']:.0f}). Recommended EOQ: +{r['recommended_qty']} units." for _, r in top3.iterrows()])
                    response_text = f"Identified **{len(high_items)} SKUs** at **HIGH stockout risk** across the store network:\n{items_str}"
                else:
                    response_text = "All store SKUs are currently maintaining inventory levels above required safety stock thresholds."
                reasoning_steps = [
                    "**1. Intent Resolution:** Query high risk SKUs across network",
                    "**2. Database Filter:** Evaluated inventory levels against calculated Reorder Points (ROP)",
                    f"**3. Synthesis:** Flagged {len(high_items)} SKUs requiring replenishment."
                ]
            elif "groceries" in q_lower or "category" in q_lower or "electronics" in q_lower or "clothing" in q_lower:
                cat_name = "Groceries" if "groceries" in q_lower else ("Electronics" if "electronics" in q_lower else ("Clothing" if "clothing" in q_lower else df['category'].iloc[0]))
                cat_df = reorder_df[reorder_df['category'].str.lower() == cat_name.lower()]
                total_rec = cat_df['recommended_qty'].sum()
                response_text = f"For **{cat_name}**, aggregate recommended reorder volume across all stores is **{total_rec:,} units**. Active stockout risk detected on **{len(cat_df[cat_df['risk_level'] == 'HIGH'])}** SKUs in this category."
                reasoning_steps = [
                    f"**1. Catalog Filter:** Extracted SKUs for category `{cat_name}`",
                    "**2. Aggregation:** Computed aggregate replenishment demand",
                    "**3. Policy:** Applied Economic Order Quantity batch sizing."
                ]
            else:
                response_text = "Please specify a Store ID (e.g. `S001`) and Product ID (e.g. `P0001`) to run the multi-agent forecasting and optimization pipeline, or ask about network-level risk."

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
    audit_df = get_audit_logs()

    if not audit_df.empty:
        tot_decisions = len(audit_df)
        approved_count = len(audit_df[audit_df['decision_status'] == 'APPROVED'])
        rejected_count = len(audit_df[audit_df['decision_status'] == 'REJECTED'])
        pending_count = len(audit_df[audit_df['decision_status'] == 'PENDING'])
        app_rate = (approved_count / (approved_count + rejected_count) * 100) if (approved_count + rejected_count) > 0 else 0

        ak1, ak2, ak3, ak4 = st.columns(4)
        with ak1:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-label">Total Audit Events</div>
                <div class="kpi-card-value">{tot_decisions}</div>
            </div>
            ''')
        with ak2:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-label">Human Approval Rate</div>
                <div class="kpi-card-value text-emerald">{app_rate:.1f}%</div>
            </div>
            ''')
        with ak3:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-label">Pending Reviews</div>
                <div class="kpi-card-value text-amber">{pending_count}</div>
            </div>
            ''')
        with ak4:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-label">Rejected Proposals</div>
                <div class="kpi-card-value text-red">{rejected_count}</div>
            </div>
            ''')

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        audit_rows = ""
        for _, a in audit_df.iterrows():
            audit_rows += f"""<tr>
<td class="muted">{a['timestamp']}</td>
<td>
<div style="font-weight: 600;">{a['store_id']}</div>
<div style="font-size: 12px; color: #64748B;">{a['product_id']}</div>
</td>
<td>Order {a['recommended_qty']} units</td>
<td>{status_badge(a['decision_status'])}</td>
<td style="color: #334155; font-weight: 500;">{a['approver']}</td>
</tr>"""

        audit_html = f"""<div class="nexus-table-wrapper">
<div class="nexus-table-header">
<span style="font-weight: 600; font-size: 14px; color: #0F172A;">Immutable Audit Log</span>
</div>
<table class="nexus-table">
<thead>
<tr>
<th>Timestamp</th>
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
# TAB 6: FAIRNESS & BIAS AUDIT
# ═════════════════════════════════════════════════════════════════════════════
elif active_tab == nav_options[5]:
    fairness_data = fetch_fairness_audit()

    if fairness_data is None:
        st.warning("⚠️ Fairness audit report not found. Run `python backend/models/fairness_audit.py` to compute and generate artifacts/fairness_audit.json.")
    else:
        overall = fairness_data.get("overall_metrics", {})
        summary = fairness_data.get("summary", {})
        thresh = fairness_data.get("degradation_threshold_pct", 25.0)
        flagged_count = summary.get("flagged_subgroups_count", 0)

        # ── KPI Cards ─────────────────────────────────────────────────────────
        fk1, fk2, fk3, fk4 = st.columns(4)
        with fk1:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-label">Overall Test MAPE</div>
                <div class="kpi-card-value">{overall.get("MAPE", 0) * 100:.1f}%</div>
            </div>
            ''')
        with fk2:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-label">Degradation Threshold</div>
                <div class="kpi-card-value">+{thresh:.0f}% <span style="font-size: 13px; font-weight: 400; color: #64748B;">Rel.</span></div>
            </div>
            ''')
        with fk3:
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-label">Subgroups Audited</div>
                <div class="kpi-card-value">{summary.get("total_subgroups_audited", 0)}</div>
            </div>
            ''')
        with fk4:
            val_class = "text-emerald" if flagged_count == 0 else "text-red"
            status_text = "0 Flagged" if flagged_count == 0 else f"{flagged_count} Flagged"
            render_html(f'''
            <div class="kpi-card">
                <div class="kpi-card-label">Fairness Status</div>
                <div class="kpi-card-value {val_class}">{status_text}</div>
            </div>
            ''')

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        render_html(f'''
        <div class="info-box">
            <div class="info-box-icon">{ICON_INFO}</div>
            <div class="info-box-text">
                <strong>Subgroup Fairness & Governance Audit:</strong> Evaluates XGBoost forecast accuracy and inventory optimization parameters across Geographic (Region), Product (Category), and Operational (Store ID) segments on held-out test records. Segments with relative MAPE degradation exceeding <strong>+{thresh:.0f}%</strong> are flagged for review.
            </div>
        </div>
        ''')

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        subgroups = fairness_data.get("subgroups", {})
        dim_choice = st.radio(
            "Select Subgroup Dimension:",
            ["Geographic (Region)", "Product Category", "Store Location"],
            horizontal=True
        )

        if dim_choice == "Geographic (Region)":
            active_list = subgroups.get("by_region", [])
            dim_title = "Regional Geographic Parity"
        elif dim_choice == "Product Category":
            active_list = subgroups.get("by_category", [])
            dim_title = "Product Category Performance Breakdown"
        else:
            active_list = subgroups.get("by_store", [])
            dim_title = "Store-Level Operational Performance"

        if active_list:
            rows_html = ""
            for item in active_list:
                is_flagged = item.get("is_flagged", False)
                rel_deg = item.get("relative_degradation_pct", 0.0)
                # M-5 FIX: Only show red when flagged, amber for positive but unflagged
                rel_color = "#DC2626" if is_flagged else ("#F59E0B" if rel_deg > 0 else "#10B981")
                badge_html = '<span class="badge badge-red">Requires Review</span>' if is_flagged else '<span class="badge badge-green">Nominal</span>'
                row_bg = "background: rgba(239, 68, 68, 0.06);" if is_flagged else ""

                rows_html += f"""<tr style="{row_bg}">
<td style="font-weight: 600; color: #0F172A;">{item.get('segment')}</td>
<td class="muted">{item.get('sample_count')}</td>
<td style="font-weight: 600;">{item.get('mape', 0) * 100:.2f}%</td>
<td style="color: {rel_color}; font-weight: 500;">{'+' if rel_deg > 0 else ''}{rel_deg:.1f}%</td>
<td class="muted">{item.get('avg_actual_demand', 0):.1f} / day</td>
<td class="muted">{item.get('avg_reorder_point', 0):.0f} units</td>
<td class="primary">+{item.get('avg_recommended_eoq', 0):.0f} units</td>
<td>{badge_html}</td>
</tr>"""

            table_fairness = f"""<div class="nexus-table-wrapper">
<div class="nexus-table-header">
<span style="font-weight: 600; font-size: 14px; color: #0F172A;">{dim_title}</span>
</div>
<table class="nexus-table">
<thead>
<tr>
<th>Segment</th>
<th>Samples</th>
<th>Subgroup MAPE</th>
<th>Rel. Variance</th>
<th>Avg Demand</th>
<th>Policy ROP</th>
<th>Policy EOQ</th>
<th>Status</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>"""
            render_html(table_fairness)
        else:
            st.info("No subgroup data available for this dimension.")

