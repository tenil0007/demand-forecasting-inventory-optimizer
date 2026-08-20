# Demand Forecasting & Inventory Optimization Agent

An enterprise AI-powered demand forecasting and inventory replenishment intelligence system that predicts unconstrained demand, evaluates dynamic stockout risks, optimizes reorder policies (EOQ, ROP, Safety Stock), and enforces Human-in-the-Loop governance via LangGraph stateful interrupts.

---

## 🏗️ Architecture & Human-in-the-Loop (HITL) Workflow

```
User Query / Scheduled Trigger
        │
        ▼
┌─────────────────┐
│  Forecast Node  │ → XGBoost model → predicted demand + empirical 95% prediction interval + SHAP explanations
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
│  Approval Node  │ ⏸ interrupt() pauses execution with SqliteSaver persistent checkpoint
└────────┬────────┘
         │
    (Human Review) ── Resumes via Command(resume={"decision": "approved"|"rejected", "approver": "..."})
         │
         ▼
┌─────────────────┐
│ Audit Log Write │ → Resumed graph node records decision in persistent SQLite audit trail
└─────────────────┘
```

### Key Technical Guardrails:
1. **"Deterministic Math Decides, LLM Explains":** The LLM never invents numbers. Quantile projections come from recursive XGBoost autoregression; safety stock and order batching come from classical inventory equations ($SS = z \cdot \sigma_d \sqrt{L}$, $EOQ = \sqrt{\frac{2DS}{H}}$).
2. **Native LangGraph `interrupt()` Pattern:** The execution graph genuinely suspends state at the approval checkpoint using `interrupt()`, assignable via unique `thread_id`s, and resumes cleanly across sessions using `SqliteSaver` persistent storage.

---

## 📊 Dataset & Feature Engineering

Based on the retail store dataset (`data/raw/retail_store_inventory.csv`) comprising **76,000 transaction records** across 5 retail stores (`S001`–`S005`), 20 SKUs (`P0001`–`P0020`), 5 categories (`Clothing`, `Electronics`, `Furniture`, `Groceries`, `Toys`), and 4 geographic regions (`East`, `North`, `South`, `West`) spanning 2022 to 2024 with 16 core columns:

| Column | Type | Role |
|---|---|---|
| `Date` | Datetime | Temporal index (2022-01-01 to 2024-01-30) |
| `Store ID` | String | Retail store entity (`S001`–`S005`) |
| `Product ID` | String | SKU catalog item (`P0001`–`P0020`) |
| `Category` | Categorical | Product category (`Clothing`, `Electronics`, `Furniture`, `Groceries`, `Toys`) |
| `Region` | Categorical | Geographic market (`East`, `North`, `South`, `West`) |
| `Inventory Level` | Numeric | On-hand available stock |
| `Units Sold` | Numeric | Historical sales volume (*censored demand signal*) |
| `Units Ordered` | Numeric | Incoming supplier replenishment |
| `Price` | Numeric | Store retail selling price |
| `Discount` | Numeric | Applied promotional discount percentage |
| `Weather Condition` | Categorical | Weather factor (Sunny, Rainy, Snowy, Cloudy, Stormy) |
| `Promotion` | Binary Flag | Active marketing campaign (0/1) |
| `Competitor Pricing` | Numeric | Market competitor benchmark price |
| `Seasonality` | Categorical | Seasonal period (Winter, Spring, Summer, Autumn) |
| `Epidemic` | Binary Flag | Surge anomaly indicator (0/1) |
| **`Demand`** | Numeric | **Model Prediction Target (True Unconstrained Demand)** |

### Target Selection Rationale (`Demand` vs `Units Sold`):
In retail supply chains, `Units Sold` represents **censored demand**—during stockouts (`Inventory Level = 0`), sales drop to zero regardless of customer demand. Training a model on `Units Sold` introduces a severe downward bias. We train on unconstrained `Demand` to capture true customer purchasing intent, retaining `Units Sold` as an informative historical feature.

---

## 🔬 Model Benchmark Comparison

To rigorously validate model selection, we benchmarked our tabular XGBoost model against Facebook Prophet across both fine-grained row-level demand and daily aggregated total network demand:

| Model | Evaluation Granularity | Target | Features / Regressors | MAPE | RMSE | MAE |
|---|---|---|---|---|---|---|
| **XGBoost (Production)** | **Daily Aggregate (Network)** | Aggregate Daily `Demand` | 24 tabular features aggregated to daily sum | **5.74%** | **606.50** | **520.70** |
| **Prophet (Baseline)** | **Daily Aggregate (Network)** | Aggregate Daily `Demand` | Additive Trend + Weekly/Yearly Fourier Seasonality | 20.96% | 2,153.79 | 1,883.31 |
| **XGBoost (Production)** | **Per-Row (Store-SKU-Day)** | SKU-Store `Demand` | 24 tabular features (lags, rolling stats, price ratio, weather OHE) | **36.58%** | **32.69** | **24.67** |

*Methodology Note: When compared at identical daily aggregate granularity, XGBoost achieves a **5.74% MAPE**, outperforming Prophet's **20.96% MAPE** by **72.6% relative error reduction** because XGBoost captures non-linear interactions across competitor pricing, promotions, and localized weather.*

---

## 📈 Backtest Financial & Operational Impact

Evaluating the AI Agent policy against a standard Naive replenishment policy (static ROP / fixed reorder quantity) across all 100 store-SKU combinations over a 60-day causal backtest simulation with lead-time pipeline delay:

| Metric | Naive Policy | AI Agent Policy | Improvement |
|---|---|---|---|
| **Total Inventory Cost** | $34,726,418.50 | **$4,190,059.90** | **+$30,536,358.60 (87.93% Cost Reduction)** |
| **Lost Sales from Stockouts** | $34,680,037.16 | **$4,037,984.01** | **-$30,642,053.15 (88.36% Lost-Sales Reduction)** |
| **Ordering Cost** | $43,900.00 | **$58,700.00** | Proactive replenishment batches |
| **Holding Cost** | $2,481.34 | $93,375.89 | Optimal safety stock investment |
| **Stockout Units** | 513,056 units | **58,449 units** | **-88.61% fewer stockout units** |
| **Service Fill Rate** | 17.32% | **90.58%** | **+73.26% Fill Rate Improvement** |

---

## ⚖️ Responsible AI — Fairness & Subgroup Auditing

To prevent algorithmic bias and disparate inventory availability, the platform runs an automated subgroup fairness audit across geographic, product, and store segments on held-out test data:

- **Audited Dimensions:** Geographic Region (`East`, `North`, `South`, `West`), Product Category (`Clothing`, `Electronics`, `Furniture`, `Groceries`, `Toys`), and Store Location (`S001`–`S005`) — 14 subgroups total.
- **Fairness Criterion:** Evaluates forecast accuracy (MAPE, RMSE, MAE) and replenishment policy parity (Average ROP, EOQ, Safety Stock). Any subgroup experiencing relative MAPE degradation $> 25\%$ compared to the overall test baseline (36.58%) is flagged as `Requires Review`.
- **Audit Findings:**
  - `Toys` Category: 49.26% MAPE (+34.7% relative degradation) → **Flagged for operational review**.
  - All 4 regions (34.7%–37.8% MAPE) and 5 stores (34.7%–39.4% MAPE) perform within nominal fairness bounds.

```bash
# Run fairness & subgroup audit
python backend/models/fairness_audit.py
```

---

## 🛠️ Tech Stack & Dashboard Tabs

- **Backend:** FastAPI, LangGraph (with `MemorySaver` checkpointer & `interrupt()`), SQLite, SQLAlchemy
- **ML / Analytics:** XGBoost Regressor, Prophet, SHAP TreeExplainer, Scikit-learn
- **Frontend Dashboard (Streamlit 6-Tab Interface):**
  1. `📊 Network Pulse`: High-level operational KPIs, stockout risk distribution, catalog summaries.
  2. `🔮 Forecast Explorer`: Multi-horizon demand forecasts with 95% confidence bands and real dynamic SHAP drivers.
  3. `📦 Replenishment`: Replenishment queue with EOQ calculations and Human-in-the-Loop review buttons.
  4. `💬 Agent Chat`: Autonomous assistant supporting multi-step natural language queries and LangGraph traces.
  5. `🛡️ Audit Trail`: Immutable record of all automated recommendations and manager sign-offs.
  6. `⚖️ Fairness & Bias`: Interactive subgroup performance audit across Regions, Categories, and Stores.

---

## ⚙️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Retrain models & evaluate benchmarks
python backend/models/train.py

# 3. Run backtest simulation
python backend/models/backtest.py

# 4. Run subgroup fairness audit
python backend/models/fairness_audit.py

# 5. Run automated test suite
python -m pytest tests/ -v

# 6. Start FastAPI backend (Port 8000)
uvicorn backend.main:app --port 8000

# 7. Launch Streamlit UI
streamlit run frontend/app.py
```

