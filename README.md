# Demand Forecasting & Inventory Optimization Agent

An enterprise AI-powered demand forecasting and inventory replenishment intelligence system that predicts unconstrained demand, evaluates dynamic stockout risks, optimizes reorder policies (EOQ, ROP, Safety Stock), and enforces Human-in-the-Loop governance via LangGraph stateful interrupts.

---

## 🏗️ Architecture & Human-in-the-Loop (HITL) Workflow

```
User Query / Scheduled Trigger
        │
        ▼
┌─────────────────┐
│  Forecast Node  │ → XGBoost model → predicted demand + 95% confidence interval + SHAP explanations
└────────┬────────┘
         ▼
┌─────────────────┐
│    Risk Node    │ → Evaluates inventory vs lead time & dynamic safety buffer → HIGH / MED / LOW risk
└────────┬────────┘
         ▼
┌─────────────────┐
│   Reorder Node  │ → Pure-math EOQ / ROP optimization → proposed order qty + policy reasoning
└────────┬────────┘
         ▼
┌─────────────────┐
│  Approval Node  │ ⏸ interrupt() pauses execution with MemorySaver thread checkpoint
└────────┬────────┘
         │
    (Human Review) ── Resumes via Command(resume={"decision": "approved"|"rejected", "approver": "..."})
         │
         ▼
┌─────────────────┐
│ Audit Log Write │ → Resumed graph node records immutable decision in SQLite audit log
└─────────────────┘
```

### Key Technical Guardrails:
1. **"Deterministic Math Decides, LLM Explains":** The LLM never invents numbers. Quantile projections come from XGBoost; safety stock and order batching come from classical inventory equations ($SS = z \cdot \sigma_d \sqrt{L}$, $EOQ = \sqrt{\frac{2DS}{H}}$).
2. **Native LangGraph `interrupt()` Pattern:** Rather than faking HITL via database flags, the execution graph genuinely suspends state at the approval checkpoint using `interrupt()`, assignable via unique `thread_id`s, and resumes cleanly with `Command(resume=...)`.

---

## 📊 Dataset & Feature Engineering

Based on the [Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting) schema with 16 core columns:

| Column | Type | Role |
|---|---|---|
| `Date` | Datetime | Temporal index |
| `Store ID` | String | Retail store entity |
| `Product ID` | String | SKU catalog item |
| `Category` | Categorical | Product category (Groceries, Beverages, etc.) |
| `Region` | Categorical | Geographic market |
| `Inventory Level` | Numeric | On-hand available stock |
| `Units Sold` | Numeric | Historical sales volume (*censored demand signal*) |
| `Units Ordered` | Numeric | Incoming supplier replenishment |
| `Price` | Numeric | Store retail selling price |
| `Discount` | Numeric | Applied promotional discount percentage |
| `Weather Condition` | Categorical | Weather factor (Sunny, Rainy, Stormy, etc.) |
| `Promotion` | Binary Flag | Active marketing campaign |
| `Competitor Pricing` | Numeric | Market competitor benchmark price |
| `Seasonality` | Categorical | Seasonal period (Winter, Spring, Summer, Fall) |
| `Epidemic` | Binary Flag | Surge anomaly indicator |
| **`Demand`** | Numeric | **Model Prediction Target (True Unconstrained Demand)** |

### Target Selection Rationale (`Demand` vs `Units Sold`):
In retail supply chains, `Units Sold` represents **censored demand**—during stockouts (`Inventory Level = 0`), sales drop to zero regardless of customer demand. Training a model on `Units Sold` introduces a severe downward bias. We train on unconstrained `Demand` to capture true customer purchasing intent, retaining `Units Sold` as an informative historical feature.

---

## 🔬 Model Benchmark Comparison

To rigorously validate model selection, we benchmarked our tabular XGBoost model against a classical additive time series model (Facebook Prophet):

| Model | Target | Features / Regressors | MAPE | RMSE | MAE | Inference Speed |
|---|---|---|---|---|---|---|
| **XGBoost (Selected)** | Unconstrained `Demand` | 25 tabular features (lags, rolling stats, price ratio, weather OHE) | **18.6%** | **6.71 units** | **5.10 units** | **< 5ms** |
| **Prophet (Baseline)** | Aggregate Daily `Demand` | Additive Trend + Weekly/Yearly Fourier Seasonality | 27.3% | 409.27 (agg) | 386.90 (agg) | ~250ms |

*XGBoost outperforms Prophet because it natively captures complex cross-feature interactions (e.g. competitor price ratio, promotional discounts, and localized weather shocks).*

---

## 📈 Backtest Financial & Operational Impact

Evaluating the AI Agent policy against a standard Naive replenishment policy (static ROP / fixed reorder quantity) across 50 store-SKU combinations over a 60-day test window:

| Metric | Naive Policy | AI Agent Policy | Improvement |
|---|---|---|---|
| **Total Inventory Cost** | $104,343.38 | **$103,166.14** | **+$1,177.24 (1.1% Net Savings)** |
| **Ordering Cost** | $45,050.00 | **$17,650.00** | **-60.8% fewer purchase orders** |
| **Lost Sales from Stockouts** | $48,275.90 | **$45,452.72** | **-$2,823.18 in saved revenue** |
| **Service Fill Rate** | 99.2% | **99.3%** | **Optimal service level achieved** |

---

## 🛠️ Tech Stack & Dashboard Tabs

- **Backend:** FastAPI, LangGraph (with `MemorySaver` checkpointer & `interrupt()`), SQLite, SQLAlchemy
- **ML / Analytics:** XGBoost Regressor, Prophet, SHAP TreeExplainer, Scikit-learn
- **Frontend Dashboard (Streamlit 6-Tab Interface):**
  1. `📊 Overview`: High-level operational KPIs, stockout risk distribution, catalog summaries.
  2. `🔮 Forecast Explorer`: Multi-horizon demand forecasts with 95% confidence bands and SHAP drivers.
  3. `📦 Reorder Recommendations`: Replenishment queue with EOQ calculations and Human-in-the-Loop review buttons.
  4. `💬 Agent Chat`: Autonomous assistant supporting multi-step natural language queries.
  5. `📋 Audit Log`: Immutable record of all automated recommendations and manager sign-offs.
  6. `📡 Live Monitoring`: Real-time control tower stepping through days, simulating inventory depletion, and triggering instant toast alerts on high-risk stockout crossings.

---

## ⚙️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Retrain models & evaluate benchmarks
python backend/models/train.py

# 3. Run backtest simulation
python backend/models/backtest.py

# 4. Run automated test suite
python -m pytest tests/ -v

# 5. Start FastAPI backend (Port 8000)
uvicorn backend.main:app --port 8000

# 6. Launch Streamlit UI
streamlit run frontend/app.py
```
