# Demand Forecasting & Inventory Optimization Agent
## Detailed Build Plan — 4 Days to Evaluation (Aug 21)

**Use Case:** Build a forecasting and inventory optimization solution that predicts demand, identifies stockout risk, and recommends replenishment actions for human approval.
**Dataset:** `atomicd/retail-store-inventory-and-demand-forecasting` (Kaggle) — Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Price, Discount, Weather Condition, Epidemic flag.
**Target:** Top 3 of 24 teams.

---

## 0. Before You Write Any Code (2–3 hrs, Day 1 morning)

### 0.1 Team setup
- [ ] Assign roles: (a) ML/forecasting owner, (b) backend/agent owner, (c) frontend/dashboard owner, (d) docs+deck+demo owner. Overlap is fine on a small team, but every deliverable needs one clear owner.
- [ ] Create GitHub repo now. Add a proper `.gitignore` (Python, Node if used, `.env`, `__pycache__`, `*.csv` if large).
- [ ] Create `AI_USAGE.md` immediately and add an entry every time you use Claude/Copilot/Cursor. Format:
  ```
  ## [Date] [Feature]
  Prompt: "..."
  Tool: Claude / Copilot
  What I kept / changed: ...
  ```
  This file is explicitly graded — don't backfill it from memory on Day 4.
- [ ] Attend the orientation call and connect with your assigned mentor. Get their contact info for the mid-sprint check-in (this is a literal gate criterion).

### 0.2 Repo structure (create this exact skeleton first)

```
retail-demand-agent/
├── README.md
├── AI_USAGE.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/                  # original Kaggle CSV
│   └── processed/            # cleaned/feature-engineered parquet/csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── backend/
│   ├── main.py                # FastAPI app entrypoint
│   ├── config.py
│   ├── models/
│   │   ├── forecast_model.py  # XGBoost training/inference wrapper
│   │   └── train.py           # standalone training script
│   ├── optimization/
│   │   └── inventory_policy.py  # EOQ, ROP, Safety Stock logic
│   ├── agents/
│   │   ├── graph.py           # LangGraph orchestration definition
│   │   ├── forecast_agent.py
│   │   ├── risk_agent.py
│   │   ├── reorder_agent.py
│   │   └── approval_agent.py
│   ├── db/
│   │   ├── models.py          # SQLite tables: audit_log, approvals
│   │   └── database.py
│   └── api/
│       ├── forecast_routes.py
│       ├── reorder_routes.py
│       ├── agent_routes.py
│       └── audit_routes.py
├── frontend/
│   └── app.py                  # Streamlit dashboard
├── tests/
│   └── test_optimization.py
├── artifacts/
│   ├── architecture_diagram.png
│   └── model_metrics.json
└── deployment/
    ├── Dockerfile
    └── docker-compose.yml
```

Commit this empty skeleton first. Every subsequent commit builds on it — this alone satisfies "Git Version History: continuous commits documenting development progression."

---

## DAY 1 — Data, EDA, Architecture (Stage 2 equivalent)

### Step 1: Environment setup (30 min)
```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost prophet shap fastapi uvicorn streamlit langgraph langchain-openai python-dotenv sqlalchemy plotly pytest
pip freeze > requirements.txt
```

### Step 2: Load and inspect data (30 min)
- Download CSV from Kaggle into `data/raw/`.
- In `notebooks/01_eda.ipynb`:
  ```python
  df = pd.read_csv("data/raw/retail_store_inventory.csv")
  print(df.columns)
  print(df.dtypes)
  print(df.isnull().sum())
  print(df['Store ID'].nunique(), df['Product ID'].nunique())
  print(df['Date'].min(), df['Date'].max())
  ```
- **Confirm actual column names now** — don't assume. Adjust all later code to match exactly what `df.columns` returns.

### Step 3: EDA (2–3 hrs)
Produce and save these plots (you'll reuse them in the deck):
- [ ] Units Sold over time, aggregated (overall trend + seasonality)
- [ ] Units Sold by Category / Region (bar charts — which segments matter most)
- [ ] Price & Discount vs Units Sold (does discount actually move demand?)
- [ ] Weather Condition vs Units Sold (boxplot)
- [ ] Epidemic flag periods vs non-epidemic — demand shift comparison (**this is your differentiator finding — highlight it in the deck**)
- [ ] Inventory Level vs Units Sold over time for 2–3 sample SKUs — visually spot where stockouts likely happened (Inventory Level near 0 while Units Sold demand was there)
- [ ] Missing data / outlier check, decide imputation strategy

Write 3–5 bullet "key findings" in the notebook — you'll lift these directly into the pitch deck.

### Step 4: Architecture design (1–2 hrs)
Lock in the design now, don't redesign mid-build:

**Agent chain (adapted from LangGraph multi-agent pattern):**
```
User query / scheduled trigger
        ↓
Forecast Agent   → calls forecast_model.py → returns predicted demand + confidence interval per SKU/store
        ↓
Risk Agent       → compares forecast vs current Inventory Level + lead time → flags stockout/overstock risk
        ↓
Reorder Agent    → runs inventory_policy.py (EOQ/ROP/Safety Stock) → proposes reorder qty + reasoning
        ↓
Approval Agent   → holds for human approve/reject via UI → writes to audit_log on decision
```

**System layers:**
- Frontend: Streamlit (dashboard + chat-style query box)
- Backend: FastAPI (serves forecast, reorder, agent, audit endpoints)
- ML: XGBoost (primary), Prophet (baseline comparison)
- Optimization: pure Python EOQ/ROP module, no external solver needed
- Orchestration: LangGraph (agent graph) + OpenAI/Claude function-calling for the reasoning/explanation layer
- Storage: SQLite (audit log, approvals) — simple, no setup overhead
- Explainability: SHAP on top of the XGBoost model

Draw this as an actual diagram (draw.io, Excalidraw, or even PowerPoint) and save to `artifacts/architecture_diagram.png`. This is a required Stage 2 deliverable — don't skip it.

### Step 5: Define API contracts (1 hr)
Write these down before coding so backend/frontend can work in parallel:

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/forecast/{store_id}/{product_id}` | GET | date range | predicted demand + confidence interval |
| `/reorder/{store_id}/{product_id}` | GET | — | recommended qty, ROP, safety stock, reasoning |
| `/agent/query` | POST | natural language question | agent's chained reasoning + final answer |
| `/audit/log` | GET | filters | list of past decisions (who approved what, when) |
| `/audit/approve` | POST | recommendation_id, decision | writes approval, returns confirmation |

### Day 1 exit checklist
- [ ] Repo skeleton committed
- [ ] EDA notebook done with saved charts + key findings
- [ ] Architecture diagram saved
- [ ] API contract table written into README
- [ ] `AI_USAGE.md` has at least 3 entries already

---

## DAY 2 — Forecasting Model + Optimization Logic (Stage 3 core)

### Step 6: Feature engineering (2 hrs)
In `notebooks/02_feature_engineering.ipynb`, then port to `backend/models/train.py`:
- Date features: day of week, month, is_weekend, is_holiday (derive from date if no holiday column exists)
- Lag features: units_sold lag 1, 7, 14, 28 days (per store-product)
- Rolling stats: 7-day and 28-day rolling mean/std of units sold
- Price/discount features: discount %, price change vs rolling average price
- Weather + Epidemic: one-hot encode Weather Condition, keep Epidemic as binary flag
- Category/Region/Store/Product: encode appropriately (target encoding or one-hot depending on cardinality)
- Train/test split: **time-based split**, not random (e.g., last 30–60 days as test) — critical for time series, mention this explicitly to judges as a deliberate choice

### Step 7: Train and evaluate models (2–3 hrs)
- Train XGBoost regressor on Units Sold as target.
- Train a Prophet baseline per top SKU for comparison (or aggregate level if time-constrained).
- Use **quantile regression or `XGBRegressor` with `objective="reg:quantileerror"`** (or LightGBM quantile) to get prediction intervals, not just point forecasts — this gives you the "confidence" story for Responsible AI scoring.
- Metrics to compute and log to `artifacts/model_metrics.json`:
  - MAPE, RMSE, MAE (XGBoost vs Prophet)
  - Feature importance (built-in + SHAP)
- Save trained model: `joblib.dump(model, "backend/models/xgb_model.pkl")`

### Step 8: SHAP explainability (1 hr)
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
```
- Wrap this in a function `explain_forecast(store_id, product_id, date)` that returns top 3 contributing features in plain language, e.g. "Forecast is high mainly because of last week's demand trend (+40%) and the ongoing discount (+15%)." This plain-language wrapper is what turns SHAP numbers into a judge-visible "explainability" feature.

### Step 9: Inventory optimization module (2 hrs)
`backend/optimization/inventory_policy.py` — implement from first principles (style-referenced from dhruvi002, written fresh):

```python
def safety_stock(demand_std, lead_time_days, service_level_z=1.65):
    return service_level_z * demand_std * (lead_time_days ** 0.5)

def reorder_point(avg_daily_demand, lead_time_days, safety_stock_units):
    return (avg_daily_demand * lead_time_days) + safety_stock_units

def economic_order_quantity(annual_demand, order_cost, holding_cost_per_unit):
    return (2 * annual_demand * order_cost / holding_cost_per_unit) ** 0.5

def stockout_risk_flag(current_inventory, forecasted_demand, reorder_point):
    if current_inventory <= reorder_point:
        return "HIGH"
    elif current_inventory <= reorder_point * 1.2:
        return "MEDIUM"
    return "LOW"
```
- You'll need to **assume reasonable values** for lead time, order cost, and holding cost since they're not in the dataset — state these assumptions explicitly in the README and deck (judges will ask about this — having a clear, honest answer is better than hiding the assumption).

### Step 10: Backtesting / business value simulation (2 hrs) — **key differentiator, don't skip**
`notebooks/03_model_experiments.ipynb`:
- Simulate two policies over your test period, per SKU:
  1. **Naive policy:** reorder a fixed amount when inventory hits a static threshold (what most retailers/other teams effectively do)
  2. **Your agent's policy:** dynamic ROP/EOQ driven by the forecast
- For each policy compute: number of stockout days avoided, estimated lost-sales $ avoided, estimated holding cost difference.
- Produce one clean chart: "Naive Policy vs AI Agent Policy — Stockouts & Cost Comparison" — this is your single most important chart for the business-value criterion.

### Step 11: FastAPI backend (2 hrs)
- Wire up `/forecast` and `/reorder` endpoints using the trained model + optimization module.
- Test every endpoint with `curl` or the FastAPI `/docs` Swagger UI before moving on.
- Commit frequently — this satisfies the Stage 3 gate criterion on Git history.

### Day 2 exit checklist
- [ ] XGBoost model trained, metrics logged, beats/compares fairly to Prophet baseline
- [ ] SHAP explanation function working
- [ ] EOQ/ROP/Safety Stock module implemented and unit-tested
- [ ] Backtest simulation chart produced showing $ value of your approach
- [ ] `/forecast` and `/reorder` endpoints live and tested
- [ ] Mid-sprint mentor check-in done (gate criterion — don't skip)

---

## DAY 3 — Agentic Layer + Guardrails + Dashboard

### Step 12: LangGraph agent orchestration (3 hrs)
`backend/agents/graph.py`:
- Define agent state (a shared dict/object passed through the graph): `{store_id, product_id, forecast, risk_level, recommendation, reasoning, approved}`
- Build the 4-node graph: `forecast_agent → risk_agent → reorder_agent → approval_agent (human-in-the-loop interrupt)`
- Each agent node is a Python function that: (a) calls the relevant backend module, (b) optionally calls the LLM to generate a natural-language explanation of its step.
- Use LangGraph's checkpoint/interrupt feature (or a simple manual pause) so the graph **stops before committing** and waits for human approval via the `/audit/approve` endpoint — this is your human-in-the-loop guardrail, not decorative.

### Step 13: Agent query endpoint + LLM reasoning wrapper (2 hrs)
- `/agent/query` accepts a natural language question ("Which SKUs in Region X are at stockout risk this week?").
- The LLM parses intent → determines which store/products to run the graph on → runs the graph → returns a synthesized natural-language answer citing the actual numbers (not hallucinated — pull real values from the graph state and pass them into the final LLM summarization prompt).
- **Guardrail:** the LLM should never be allowed to invent a reorder number — it only explains numbers that came from your deterministic `inventory_policy.py`. State this explicitly in your architecture doc; it's a strong Responsible AI point ("deterministic code decides, LLM explains" — same principle used in the procurement copilot repo you saw earlier).

### Step 14: Audit logging (1 hr)
`backend/db/models.py` — SQLite table:
```
audit_log(id, timestamp, store_id, product_id, recommended_qty, risk_level,
          reasoning_snapshot, approver, decision, decision_timestamp)
```
- Every recommendation generates a row on creation; every approve/reject updates it.
- Expose `/audit/log` so the dashboard can show a full history table — this alone satisfies 3 rubric lines (audit logging, human approval, process of monitoring).

### Step 15: Streamlit dashboard (3–4 hrs)
`frontend/app.py` — structure into tabs:
1. **Overview tab:** KPI cards (total SKUs at risk, forecast accuracy, estimated $ saved vs naive policy), Units Sold trend chart
2. **Forecast Explorer tab:** pick store/product → show forecast chart with confidence band + SHAP explanation panel
3. **Reorder Recommendations tab:** table of SKUs with risk level, recommended qty, Approve/Reject buttons → hits `/audit/approve`
4. **Agent Chat tab:** free-text query box → calls `/agent/query`, displays reasoning + answer
5. **Audit Log tab:** full history table, filterable by store/date/approver

Design notes:
- Use `st.columns`, KPI metric cards, and Plotly (not default matplotlib) for a modern look — UI/UX is a separately scored criterion.
- Add color-coded risk badges (red/amber/green) — visually communicates risk at a glance, which judges notice immediately in a live demo.

### Day 3 exit checklist
- [ ] LangGraph agent chain working end-to-end for at least one store/product
- [ ] Human approval step actually gates the recommendation (test rejecting one)
- [ ] Audit log populating correctly
- [ ] Streamlit dashboard has all 5 tabs functional
- [ ] Full flow tested: query → forecast → risk → recommendation → approve → audit log entry appears

---

## DAY 4 — Integration, Deployment, Deck, Rehearsal

### Step 16: End-to-end stress test (1–2 hrs)
- Run the complete flow for 5–10 different store/product combinations without restarting the app — check for crashes (this is a literal gate criterion: "primary end-to-end user flow executes successfully without critical runtime crashes").
- Test edge cases: a SKU with very low sales history, a SKU with missing data.
- Fix bugs, don't add new features today.

### Step 17: Deployment (1–2 hrs)
- Containerize with the `Dockerfile` + `docker-compose.yml` you scaffolded.
- Deploy Streamlit to Streamlit Community Cloud (free, fast) or Render; deploy FastAPI backend to Render/Railway if time allows, or run both locally and have a screen-recorded backup demo in case live deployment breaks during evaluation.
- **Always have a local + recorded video fallback** — never depend solely on live internet demo for a hiring evaluation.

### Step 18: Documentation (1–2 hrs)
- Finalize `README.md`: problem statement, architecture diagram, tech stack, setup instructions, key results (metrics + business value numbers), assumptions made (lead time, cost figures), team members.
- Finalize `AI_USAGE.md` — go back through your actual chat history and fill any gaps honestly.
- Clean up notebooks (remove dead cells, add markdown headers).

### Step 19: Presentation deck (2–3 hrs)
Structure (matches the "Presentation and Communication" rubric line):
1. Problem statement (from the Christ use case table, in your own words)
2. Approach & architecture diagram
3. Data & key EDA findings (2–3 slides max)
4. Model results (MAPE/RMSE table, XGBoost vs Prophet)
5. **Agentic architecture** — the 4-agent LangGraph chain diagram, explain why agentic (not just "we used an LLM")
6. Responsible AI — SHAP explainability screenshot, human approval screenshot, audit log screenshot
7. **Business value slide** — the naive-vs-agent backtest chart, with a $ number headline
8. Live demo (or embedded video/GIF as backup)
9. Assumptions & limitations (be upfront — judges respect this)
10. What we'd do next with more time (shows maturity)

### Step 20: Rehearsal (2+ hrs, do this seriously)
- Full run-through, live demo included, at least 3 times.
- Assign who presents which slide.
- Prepare answers for likely questions:
  - "Why XGBoost over deep learning?" → interpretability + strong tabular performance + time constraints, backed by your metrics
  - "Why these lead time/cost assumptions?" → state them clearly, note this would come from real ERP data in production
  - "How is this actually agentic and not just an API pipeline?" → point to the LangGraph state graph, the human-in-the-loop interrupt, and the fact that agents make sequential decisions based on prior agent output, not a fixed script
  - "How do you prevent the LLM from giving wrong numbers?" → deterministic code computes all numbers, LLM only explains — never generates figures itself

### Day 4 exit checklist
- [ ] App runs without crashing through a full demo pass
- [ ] Deployed (or reliable local + video backup ready)
- [ ] README + AI_USAGE.md complete
- [ ] Deck finalized and rehearsed
- [ ] Every team member can explain the architecture and defend design choices unprompted

---

## Reference repos (style/pattern only — do not copy code directly)
- MahdiNavaei/pharmaceutical-supply-chain-agentic-ai — multi-agent LangGraph pattern
- dhruvi002/demand-forecast-inventory-optimizer — XGBoost + EOQ structuring style
- CH-S-K-CHAITANYA/Retail-Sales-Forecasting-Inventory-Optimization — ROP/EOQ + Streamlit reference

## Final reminder
Two things separate top-3 from mid-pack here: (1) a **real** agent chain with a human-approval gate and audit trail, not a dashboard with an LLM sticker on it, and (2) a **quantified business value story** (the naive-vs-agent backtest), not just accuracy metrics. Protect time for both — cut scope elsewhere (e.g., skip Next.js, skip cloud deployment complexity) before cutting either of these.
