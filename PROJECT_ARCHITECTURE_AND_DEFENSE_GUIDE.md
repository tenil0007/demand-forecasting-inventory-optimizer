# ⚡ NexusSupply: Autonomous Demand Forecasting & Inventory Optimization
## Comprehensive Project Architecture, Technical Design & Cognizant Defense Guide

---

## 1. Executive Summary (The STAR Story)

*Use this when evaluators or interviewers ask: **"Tell me about your project."***

- **Situation (S)**: Traditional retail supply chains suffer from disjointed forecasting and manual replenishment workflows. Disconnected spreadsheets and static reorder rules cause frequent stockouts during promotional surges and costly overstocking during demand slumps, eroding profit margins and reducing customer service levels.
- **Task (T)**: Architect and build an enterprise-grade, end-to-end autonomous demand forecasting and inventory optimization platform (**NexusSupply**) that ingests multi-store retail data, accurately predicts demand across variable horizons, optimizes inventory replenishment parameters, and automates reorder workflows using Agentic AI with strict Human-in-the-Loop (HITL) governance.
- **Action (A)**:
  1. Engineered a multi-stage time-series ML pipeline using an **XGBoost Ensemble** trained on the official **76,000-row retail store dataset** with 24 engineered lag, rolling-window, competitor price, and weather features.
  2. Implemented classical Operations Research models for dynamic **Safety Stock (SS)**, **Reorder Point (ROP)**, and **Economic Order Quantity (EOQ)** calibrated for a 95% service level ($Z = 1.65$).
  3. Developed an **Agentic AI Workflow (LangGraph / Ollama Llama 3.1)** that reasons over inventory alerts, plans replenishment actions, and surfaces transparent multi-step reasoning traces.
  4. Embedded **Responsible AI guardrails** with SHAP explainability, 95% confidence intervals, subgroup fairness audits, and an immutable SQLite audit trail requiring human approvals for high-value orders.
  5. Built an enterprise-grade web application with a modern shadcn-inspired **NexusSupply** design system.
- **Result (R)**: Achieved a **5.74% MAPE** on aggregate daily demand (beating the Prophet baseline of 20.96% MAPE by 72.6%), reduced total inventory operational cost by **95.23%** in held-out backtesting ($7.79M down to $371.4k), achieved a **99.85% Fill Rate** (up from 82.25%), slashed lost sales from stockouts by **99.19%**, and created an immutable audit record for compliance.

---

## 2. End-to-End Solution Architecture

```
+-----------------------------------------------------------------------------------+
|                           1. DATA INGESTION & STORAGE                             |
|  - Official Dataset (76,000 Records, 5 Stores, 20 SKUs, 5 Categories, 4 Regions)  |
|  - Feature Store: Lags (1,7,14,28), Rolling Means (7d,28d), Weather, Promo Flags |
|  - SQLite Audit DB: Immutable Human-in-the-Loop Decision Logs                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     2. MACHINE LEARNING FORECASTING ENGINE                        |
|  - Production Model: XGBoost Regressor (5.74% Daily Agg MAPE / 36.58% Row MAPE)   |
|  - Baseline Comparison: Facebook Prophet (20.96% Daily Agg MAPE)                  |
|  - SHAP Explainability Engine: Feature Attribution & Driver Analysis               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                  3. OPERATIONS RESEARCH & INVENTORY OPTIMIZATION                  |
|  - Safety Stock: SS = Z * sigma_d * sqrt(Lead_Time)  [Z = 1.65 for 95% SLA]       |
|  - Reorder Point: ROP = (mean_demand * Lead_Time) + SS                            |
|  - Economic Order Quantity: EOQ = sqrt((2 * Demand * Order_Cost) / Holding_Cost)  |
|  - Risk Classifier: HIGH (< ROP), MEDIUM (< 1.35 * ROP), LOW (Healthy)           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       4. AGENTIC AI & GOVERNANCE LAYER                            |
|  - LangGraph State Machine: Multi-Step Reasoner & Tool Dispatcher                 |
|  - Ollama Local LLM (Llama 3.1) + Deterministic Fallback Engine                   |
|  - Human-in-the-Loop (HITL) Gate: Approve / Reject Decision Tracking              |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        5. NEXUSSUPPLY EXECUTIVE DASHBOARD                         |
|  - Tab 1: Network Pulse (Executive Overview, Risk Donut, Category Velocity)       |
|  - Tab 2: Forecast Explorer (Historical vs Predictions, 95% CI, SHAP Drivers)     |
|  - Tab 3: Replenishment Queue (Action Table & Interactive HITL Review Panel)      |
|  - Tab 4: Agent Chat (Copilot with Step-by-Step LangGraph Reasoning Traces)       |
|  - Tab 5: Audit Trail (Governance Compliance & Decision History)                  |
|  - Tab 6: Fairness & Bias (Subgroup Performance Breakdown & Disparity Audit)      |
+-----------------------------------------------------------------------------------+
```

---

## 3. Detailed Technical Components

### Component 1: Data Pipeline & Feature Engineering
- **Dataset**: Official Retail Store Inventory Dataset (`data/raw/retail_store_inventory.csv`).
  - **Scope**: 76,000 rows spanning 2022-01-01 to 2024-01-30 across 5 retail stores (`S001`–`S005`), 20 SKUs (`P0001`–`P0020`), 5 product categories (`Clothing`, `Electronics`, `Furniture`, `Groceries`, `Toys`), and 4 geographic regions (`East`, `North`, `South`, `West`).
- **Engineered Features**:
  1. *Temporal Features*: Day of week, month, quarter, seasonality index, weekend flag.
  2. *Autoregressive Lags*: $Lag_1, Lag_7, Lag_{14}, Lag_{28}$ (capturing weekly and monthly cyclicality).
  3. *Rolling Window Statistics*: 7-day and 28-day rolling mean and standard deviation ($\sigma_d$).
  4. *Exogenous Drivers*: Promotional discounts, price/competitor ratios, weather conditions (Sunny, Rainy, Snowy, Cloudy, Stormy), epidemic flag.

### Component 2: Machine Learning Forecasting Pipeline
- **Production Architecture**: **XGBoost (Extreme Gradient Boosting) Regressor**.
  - **Objective**: Squared error with tree-based gradient boosting.
  - **Hyperparameters**: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`.
  - **Cross-Validation**: Time-based held-out split (last 30 days held out).
- **Benchmark Comparison**:
  | Model | Evaluation Granularity | MAPE (%) | RMSE | MAE |
  | :--- | :--- | :---: | :---: | :---: |
  | **XGBoost (Production)** | **Daily Aggregate (Network)** | **5.74%** | **606.50** | **520.70** |
  | Prophet (Baseline) | Daily Aggregate (Network) | 20.96% | 2,153.79 | 1,883.31 |
  | **XGBoost (Production)** | **Per-Row (Store-SKU-Day)** | **36.58%** | **32.69** | **24.67** |

### Component 3: Operations Research & Inventory Optimization
Instead of relying solely on raw forecasts, predictions feed directly into mathematical inventory policies:
1. **Safety Stock ($SS$)**:
   $$SS = Z \times \sigma_d \times \sqrt{L}$$
   - $Z = 1.65$ (95% Service Level Confidence).
   - $\sigma_d$ = Standard deviation of daily forecast demand.
   - $L = 7$ days (Supplier Lead Time).
2. **Reorder Point ($ROP$)**:
   $$ROP = (\bar{d} \times L) + SS$$
   - $\bar{d}$ = Average daily forecasted demand over lead time.
3. **Economic Order Quantity ($EOQ$)**:
   $$EOQ = \sqrt{\frac{2 \times D \times S}{H}}$$
   - $D$ = Annualized demand ($\bar{d} \times 365$).
   - $S = \$50.00$ (Fixed ordering cost per purchase order).
   - $H = \text{Price} \times 20\%$ (Annual holding cost rate).
4. **Automated Risk Categorization**:
   - **CRITICAL / HIGH**: $\text{Current Inventory} < ROP$ (Immediate stockout threat).
   - **WARNING / MEDIUM**: $ROP \le \text{Current Inventory} < 1.35 \times ROP$ (Approaching safety buffer).
   - **OPTIMAL / LOW**: $\text{Current Inventory} \ge 1.35 \times ROP$ (Healthy stock).

### Component 4: Agentic AI Workflow & LangGraph State Machine
- **Architecture**: Stateful LangGraph workflow powered by local Ollama (`llama3.1`) with fallback rule-based reasoning.
- **Workflow Steps**:
  1. *Intent Classification*: Resolves store IDs, SKU IDs, or category filters from natural language queries.
  2. *Tool Execution*: Dispatches database queries (`SELECT * FROM inventory WHERE stock < rop`) and forecast model inference.
  3. *Policy Evaluation*: Compares forecast outputs against safety thresholds.
  4. *Multi-Step Reasoning Trace*: Returns a collapsible step-by-step reasoning chain so users understand *why* recommendations were made.

### Component 5: Governance & Responsible AI Guardrails
- **Human-in-the-Loop (HITL)**: High-risk orders generate a proposal card requiring manual human approval or rejection with audit logging.
- **Explainability**: SHAP (SHapley Additive exPlanations) visualizes positive and negative demand drivers.
- **Uncertainty Quantification**: 95% Confidence Intervals provide envelope bounds on forecasts.
- **Immutable Audit Logging**: SQLite database records every decision, timestamp, approver ID, and policy snapshot.

---

## 4. Architectural Decisions & Trade-Offs (PREP Framework)

*Use this when evaluators ask: **"Why did you choose X over Y?"***

### Decision 1: XGBoost vs. Deep Learning (LSTM / DeepAR)
- **Point (P)**: We selected XGBoost Regressor over Recurrent Neural Networks (LSTM/Transformers).
- **Reason (R)**: Tabular retail demand data with mixed numerical and categorical exogenous features (promos, weather, prices) trains faster and avoids overfitting on moderate-sized datasets.
- **Evidence (E)**: XGBoost achieved 5.74% aggregate MAPE (and 36.58% SKU-level MAPE) with ~12ms inference latency compared to LSTM's computational overhead and vulnerability to small-batch noise.
- **Point (P)**: For retail operations, sub-second inference enables real-time SKU reordering at low computational cost.

### Decision 2: Streamlit vs. React/FastAPI Separate Microservices
- **Point (P)**: We built the frontend in Streamlit augmented with a custom CSS design system.
- **Reason (R)**: Streamlit combines native Python data manipulation with instant state synchronization, while custom CSS gave us the exact shadcn/Tailwind aesthetic without separate deployment complexities.
- **Evidence (E)**: Both frontend and backend can run seamlessly in standalone or client-server mode with identical responsive components.

### Decision 3: LangGraph + Local LLM vs. Black-Box Cloud APIs
- **Point (P)**: We structured the agent with LangGraph and local LLM execution.
- **Reason (R)**: Enterprise supply chain data often cannot leave corporate boundaries due to compliance and data privacy regulations.
- **Evidence (E)**: The system functions 100% offline with zero external API fees and guaranteed deterministic fallbacks if the local LLM is uninitialized.

---

## 5. System Resilience & Failure Modes (Architecture -> Flow -> Failure)

| Potential Failure Point | Root Cause | System Resilience & Fallback Strategy |
| :--- | :--- | :--- |
| **Backend API Server Offline** | FastAPI server not running on port 8000. | Frontend gracefully catches `requests.exceptions` and falls back to **in-process ML inference** and **direct SQLite ORM** queries with zero user interruption. |
| **Local LLM (Ollama) Unreachable** | Port 11434 inactive or model not pulled. | Agent Chat triggers grounded **rule-based policy reasoner** that computes exact math from `reorder_df`, returning identical structured reasoning steps. |
| **Missing Historical Data** | Cold-start SKU or store location. | Falls back to Category-level Moving Average velocity with expanding variance bands to ensure safety stock is not underestimated. |
| **Database File Locked** | Concurrent writes on `audit.db`. | Session management uses SQLAlchemy connection pooling with automatic retry and graceful fallback display records. |

---

## 6. Project Ownership & Defense Q&A Preparation

### Q1: "What was your specific role and technical contribution?"
> **Answer**: *"I designed and implemented the entire end-to-end architecture: from engineering the time-series lag features on the 76,000-row retail dataset, training the 5.74% aggregate MAPE XGBoost model, formulating the dynamic Operations Research inventory policies (SS, ROP, EOQ), to developing the LangGraph agent reasoning state machine and building the 6-tab executive dashboard with custom CSS design tokens."*

### Q2: "How is this project different from a standard forecasting model?"
> **Answer**: *"Most projects stop at predicting a number. NexusSupply is an actionable decision system. It translates predictions into optimal operational decisions (EOQ order batches and ROP triggers), flags risk categories, provides SHAP explainability, and enforces Human-in-the-Loop governance with an immutable audit trail."*

### Q3: "How does your system implement Responsible AI?"
> **Answer**: *"We enforce 5 core pillars: (1) **Explainability** via real TreeExplainer SHAP attribution bar charts, (2) **Uncertainty bounds** via 95% confidence envelopes, (3) **Fairness & Subgroup Auditing** (`fairness_audit.py`) monitoring MAPE and inventory policy parity across regions, categories, and stores with a +25% degradation threshold, (4) **Human-in-the-Loop** gates so AI cannot execute high-risk purchase orders without human manager approval, and (5) **Immutability** through database-backed audit logging."*

### Q4: "What trade-offs did you make and what would you improve next?"
> **Answer**: *"Our primary trade-off was prioritizing deterministic low-latency tree models (XGBoost) over complex foundation models. Given more compute and data, I would implement multi-echelon network optimization (warehouse-to-store rebalancing) and real-time Kafka event streaming."*
