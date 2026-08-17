# Demand Forecasting & Inventory Optimization Agent

An AI-powered demand forecasting and inventory optimization solution that predicts demand, identifies stockout risk, and recommends replenishment actions for human approval.

## 🏗️ Architecture

```
User Query / Scheduled Trigger
        │
        ▼
┌─────────────────┐
│ Forecast Agent   │ → XGBoost model → predicted demand + confidence interval
└────────┬────────┘
         ▼
┌─────────────────┐
│ Risk Agent       │ → compares forecast vs inventory + lead time → stockout/overstock risk
└────────┬────────┘
         ▼
┌─────────────────┐
│ Reorder Agent    │ → EOQ/ROP/Safety Stock optimization → proposed reorder qty + reasoning
└────────┬────────┘
         ▼
┌─────────────────┐
│ Approval Agent   │ → human approve/reject via UI → audit log
└─────────────────┘
```

**Key Design Principle:** *"Deterministic code decides, LLM explains."* The LLM (Ollama) generates natural-language summaries of decisions already made by optimization code. It never invents numbers.

## 📊 Dataset

[Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting) — 16 columns including Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Price, Discount, Weather Condition, Promotion, Competitor Pricing, Seasonality, Epidemic, and Demand.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML/Forecasting | XGBoost (primary), Prophet (baseline), SHAP (explainability) |
| Optimization | Pure Python EOQ/ROP/Safety Stock |
| Agent Orchestration | LangGraph (multi-agent state graph) |
| LLM | Ollama (local, free — llama3.1/mistral) |
| Backend | FastAPI |
| Frontend | Streamlit (5-tab dashboard) |
| Storage | SQLite (audit log, approvals) |
| Deployment | Streamlit Community Cloud / Docker |

## 📡 API Contracts

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/forecast/{store_id}/{product_id}` | GET | date range (query params) | predicted demand + confidence interval |
| `/reorder/{store_id}/{product_id}` | GET | — | recommended qty, ROP, safety stock, reasoning |
| `/agent/query` | POST | `{"query": "natural language question"}` | agent's chained reasoning + final answer |
| `/audit/log` | GET | filters (query params) | list of past decisions |
| `/audit/approve` | POST | `{"recommendation_id": int, "decision": "approved/rejected", "approver": "name"}` | confirmation |

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- Ollama (optional — for LLM explanations)

### Installation
```bash
# Clone the repo
git clone <repo-url>
cd retail-demand-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your settings

# Download dataset from Kaggle
# Place CSV in data/raw/retail_store_inventory.csv
```

### Running
```bash
# Start the backend
uvicorn backend.main:app --reload --port 8000

# Start the dashboard (in a new terminal)
streamlit run frontend/app.py
```

### Docker
```bash
docker-compose -f deployment/docker-compose.yml up --build
```

## 📈 Key Results

| Metric | Value |
|---|---|
| **XGBoost MAPE** | **18.6%** |
| **XGBoost RMSE** | **6.71 units** |
| **XGBoost MAE** | **5.10 units** |
| **Stockout days reduced (vs naive policy)** | **25.0% reduction** |
| **Estimated $ saved (10 sample SKUs / 60d)** | **$1,131.91** |
| **Stockout rate reduction** | **8 days → 6 days across top SKUs** |

## 🔧 Assumptions

- **Lead time:** 7 days (configurable) — in production, sourced from ERP data
- **Order cost:** $50 per order — standard logistics assumption
- **Holding cost:** 20% of unit price per year — industry standard
- **Service level:** 95% (z = 1.65) — balances stockout risk vs holding cost

## 👥 Team

- Solo developer

## 📝 AI Usage

See [AI_USAGE.md](AI_USAGE.md) for complete AI tool usage log.
