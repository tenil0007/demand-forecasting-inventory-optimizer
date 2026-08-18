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
