# AI Usage Log

This document tracks all AI tool usage during development of this project.

## 2026-08-17 — Project Scaffolding & Architecture
**Prompt:** "Build a forecasting and inventory optimization solution that predicts demand, identifies stockout risk, and recommends replenishment actions for human approval."
**Tool:** Google Antigravity (Claude)
**What I kept / changed:** Used AI to generate the initial project skeleton, directory structure, and boilerplate configuration files. Reviewed and adjusted all generated code to match project requirements. Architecture design was collaborative — AI proposed the 4-agent LangGraph chain, I refined the state schema and approval flow.

## 2026-08-17 — Dataset Research
**Prompt:** "What are the columns in the atomicd/retail-store-inventory-and-demand-forecasting Kaggle dataset?"
**Tool:** Google Antigravity (Claude)
**What I kept / changed:** AI correctly identified all 16 columns from the Kaggle schema. Used this to inform feature engineering decisions — notably discovered the `Demand` column as a better forecasting target than `Units Sold`.

## 2026-08-17 — LangGraph API Research
**Prompt:** "How does the modern LangGraph interrupt/Command pattern work for human-in-the-loop?"
**Tool:** Google Antigravity (Claude)
**What I kept / changed:** AI researched the current LangGraph API (interrupt() + Command(resume=...) pattern). Used this to implement the approval agent correctly with MemorySaver checkpointer.

## 2026-08-18 — Full Pipeline Upgrade (16-Column Schema, LangGraph HITL, Prophet & Live Monitoring)
**Prompt:** "Upgrade retail-demand-agent with 16-column dataset features, LangGraph native interrupt() / Command(resume=...) HITL pattern with thread-level checkpointers, Prophet baseline benchmark comparison, backtest simulation, and a Live Monitoring Streamlit control tower tab."
**Tool:** Google Antigravity (Claude / Gemini 3.7)
**What I kept / changed:** 
- Validated and created modular feature engineering (`backend/models/features.py`) and synthetic data generation (`data/generate_synthetic_data.py`) with complete 16-column support and censored demand mitigation comments.
- Implemented stateful human-in-the-loop pauses using LangGraph's `interrupt()` and `Command(resume=...)` in `backend/agents/graph.py` and `approval_agent.py`.
- Trained Facebook Prophet additive baseline against XGBoost, logging comparison into `artifacts/model_metrics.json`.
- Built the 6th Streamlit tab (`📡 Live Monitoring`) with single-day step-forward simulation and real-time toast alerts for high-risk stockout SKU transitions.
- Added comprehensive unit tests in `tests/test_agent_interrupt.py` and verified all 13 tests passing.

## 2026-08-20 — Real Dataset Migration, Model Retraining, and Fair Evaluation Benchmark
**Prompt:** "Swap synthetic data for real 76,000-row dataset (`sales_data.csv` -> `retail_store_inventory.csv`), fix `PRODUCT_ID_RE` regex for 4-digit IDs, compute daily aggregate MAPE for fair apples-to-apples Prophet vs XGBoost comparison, retrain XGBoost, rerun backtest & fairness audit, and update documentation."
**Tool:** Google Antigravity (Gemini 3.7 Flash)
**What I kept / changed:**
- Replaced synthetic dataset with official 76,000-row dataset covering 5 stores (`S001`–`S005`), 20 SKUs (`P0001`–`P0020`), 5 categories (`Clothing`, `Electronics`, `Furniture`, `Groceries`, `Toys`), and 4 regions (`East`, `North`, `South`, `West`).
- Fixed `PRODUCT_ID_RE` in `backend/security/prompt_guard.py` from `r"^P\d{3}$"` to `r"^P\d{3,4}$"` to accept 4-digit SKU IDs (`P0001`–`P0020`).
- Resolved methodology asymmetry in `backend/models/train.py`: added daily aggregate demand evaluation for XGBoost so it compares fairly against Prophet at identical aggregation levels (XGBoost 5.74% MAPE vs Prophet 20.96% MAPE), while preserving per-row metrics (XGBoost 36.58% MAPE).
- Retrained model end-to-end and updated `artifacts/model_metrics.json`.
- Reran backtest simulation: achieved 95.23% total cost reduction ($7.79M down to $371.4k), 99.85% fill rate (up from 82.25%), and 99.19% lost-sales reduction ($7.52M down to $61.1k).
- Reran fairness audit: audited 14 subgroups; identified `Toys` category (+34.7% relative degradation) as requiring operational review.
- Fixed hardcoded region assertion in `tests/test_fairness_audit.py` to dynamic dataset unique counts and confirmed 100% test pass rate across all 35 tests.
