# MASTER PROMPT: Enterprise Retail Demand Forecasting & Autonomous Replenishment Dashboard

You are an expert UI/UX designer and Senior Full-Stack Engineer specializing in enterprise SaaS, supply chain analytics, and AI agent interfaces.

Build/optimize a production-grade, highly polished, responsive dashboard for an **Autonomous Demand Forecasting & Inventory Optimization Platform** (Retail Supply Chain).

---

## 1. DESIGN SYSTEM & VISUAL IDENTITY
- **Aesthetic:** Modern Enterprise SaaS / FinTech control tower (clean, data-dense yet breathable, high contrast).
- **Typography:** `Inter`, `-apple-system`, `BlinkMacSystemFont`, clean sans-serif with strong hierarchy.
- **Color Palette:**
  - Backgrounds: Off-white `#F8FAFC`, Pure White `#FFFFFF` card containers with subtle `#E2E8F0` borders and soft drop shadows `0 1px 3px rgba(15,23,42,0.04)`.
  - Primary / Accent: Royal Blue `#2563EB`, Indigo `#4F46E5`.
  - Semantic Statuses:
    - 🔴 **Critical / High Stockout Risk:** Crimson `#EF4444` (Bg: `#FEE2E2`, Border: `#FCA5A5`)
    - 🟡 **Warning / Buffer Zone:** Amber `#F59E0B` (Bg: `#FEF3C7`, Border: `#FCD34D`)
    - 🟢 **Optimal / Healthy Stock:** Emerald `#10B981` (Bg: `#D1FAE5`, Border: `#6EE7B7`)
- **Key Architectural Rule:** *"Deterministic Math Decides, LLM Explains."* The UI must always display exact mathematical values (EOQ, ROP, Safety Stock, Forecast Confidence Intervals) and use AI/LLMs strictly for plain-language reasoning, SHAP explanations, and interactive conversational queries.

---

## 2. DATA SCHEMA & CORE METRICS
The dashboard consumes a 16-column retail telemetry dataset:
- `Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Inventory Level`, `Units Sold` (censored sales), `Units Ordered`, `Price`, `Discount`, `Weather Condition`, `Promotion`, `Competitor Pricing`, `Seasonality`, `Epidemic`, and `Demand` (true unconstrained target).

**Key Optimization Formulas Displayed:**
1. **Safety Stock:** $$SS = 1.65 \times \sigma_d \times \sqrt{L}$$ ($95\%$ service level, $L=7$ days lead time)
2. **Reorder Point:** $$ROP = (\bar{d} \times L) + SS$$
3. **Economic Order Quantity:** $$EOQ = \sqrt{\frac{2 \cdot \text{Annual Demand} \cdot S}{H}}$$ ($S=\$50$ order cost, $H=20\%$ annual holding cost)
4. **Stockout Risk Rule:**
   - `HIGH`: Current Inventory $< ROP$
   - `MEDIUM`: Current Inventory $< ROP \times 1.20$
   - `LOW`: Current Inventory $\ge ROP \times 1.20$

---

## 3. COMPLETE 6-TAB DASHBOARD STRUCTURE

### TAB 1: 📊 Executive Overview & Network Pulse
- **KPI Hero Cards Row (4 Cards):**
  1. `Active Catalog SKUs` (Count across categories)
  2. `Monitored Store Network` (Total active retail stores)
  3. `High Stockout Risk SKUs` (Count + percentage badge with semantic red/green color)
  4. `Network Service Level Compliance` (Target: 95.0%, showing current achievement %)
- **Visual Analytics:**
  - **Risk Distribution Chart:** Donut chart showing breakdown of SKUs by risk (`HIGH`, `MEDIUM`, `LOW`).
  - **Category Velocity Bar Chart:** Horizontal bar chart of demand volume across categories (`Groceries`, `Beverages`, `Snacks`, `Personal Care`, `Household`).
  - **Model Benchmark Card:** Side-by-side comparison table showing **XGBoost (18.6% MAPE)** vs **Prophet Baseline (27.3% MAPE)** justifying model selection.

---

### TAB 2: 🔮 Demand Forecast Explorer & Explainability
- **Filters Toolbar:** Select Store (`S001`-`S005`), Select Product SKU (`P001`-`P010`), Forecast Horizon Slider (`7` to `30` days).
- **Interactive Multi-Step Forecast Chart:**
  - Historic actual demand line vs future predicted demand line.
  - **95% Confidence Interval Ribbon Band** (shaded area between Lower Bound and Upper Bound).
- **Plain-Language Precision Benchmark Box:**
  - *"Model Error Verified at 18.6% MAPE. Predictions accurate within ±5.10 units/day under 95% service-level assurance."*
- **SHAP Feature Importance Driver Chart:**
  - Diverging horizontal bar chart explaining the primary drivers behind the SKU's projection (e.g., `Recent 7-Day Velocity +3.8`, `Active Promo +2.4`, `Competitor Price Gap -1.5`, `Weekend Pattern +2.1`, `Weather Shock -0.9`).
  - Green bars for positive demand drivers, Red bars for negative drivers.

---

### TAB 3: 📦 Autonomous Reorder Recommendations (HITL Review)
- **Replenishment Action Queue:**
  - Filterable by Risk Level (`HIGH`, `MEDIUM`, `LOW`).
  - Data table with columns: `Store`, `Product SKU`, `Category`, `Current Stock`, `Safety Stock`, `Reorder Point (ROP)`, `EOQ Order Qty`, `Stockout Risk Badge`.
- **Human-in-the-Loop (HITL) Action Card:**
  - Select any SKU in the queue to inspect details:
    - Display current buffer vs needed batch size.
    - **Policy Reasoning Box:** *"Current inventory (21 units) is below ROP (228 units). Order 300 units (EOQ) immediately to cover 7-day supplier lead time."*
    - **Action Buttons:**
      - `✓ Approve Replenishment` (Triggers LangGraph resume with `approved` status, logs reviewer name and timestamp).
      - `✕ Reject Recommendation` (Triggers override with `rejected` status).
    - Success/Warning banners upon action completion.

---

### TAB 4: 💬 Autonomous Agent Chat (Conversational Ops)
- **Quick Prompt Chips:**
  - *"Which SKUs are at highest stockout risk?"*
  - *"Forecast demand for Store S001"*
  - *"Recommend replenishment for Category Groceries"*
- **Multi-Turn Chat Feed:**
  - Clean chat bubbles for User and AI Assistant.
  - **Expandable "Reasoning Chain" Drawer** beneath assistant responses detailing:
    1. `Intent Resolution`: Identified target Store/SKU.
    2. `Forecast Engine`: Multi-step XGBoost projection.
    3. `Risk Evaluation`: Inventory vs ROP & safety buffer.
    4. `Inventory Policy`: Mathematical EOQ order calculation.
- **Human-in-the-loop Thread Tracker:** Connects directly with backend `thread_id` checkpointers.

---

### TAB 5: 📋 Governance & Immutable Audit Trail
- **Governance Summary Metrics:**
  - `Total Audit Events`, `Manager Approval Rate %`, `Pending Review Queue`, `Rejected Proposals`.
- **Filters:** Filter by Store ID and Decision Status (`ALL`, `APPROVED`, `REJECTED`, `PENDING`).
- **Audit Log Table:**
  - Columns: `Timestamp (UTC)`, `Store`, `Product SKU`, `Restock Units`, `Risk Flag`, `Decision Status` (with badges: `🟢 APPROVED`, `🔴 REJECTED`, `🟡 PENDING`), `Reviewer/Approver`, `Policy Rationale Snapshot`.

---

### TAB 6: 📡 Live Telemetry & Real-Time Monitoring Control Tower
- **Day-by-Day Simulation Control Bar:**
  - Controls: `▶ Step +1 Day` button, `⏮ Reset Simulation` button, and Date scrubber slider.
  - Displays `Current Simulation Date` and progress indicator through the test window.
- **Dynamic Telemetry KPIs (Updates live as days advance):**
  - `Active Critical Alerts` (SKUs dropping below ROP on this day).
  - `Buffer Warnings` (SKUs approaching buffer zone).
  - `Service Level Compliance %` (Dynamic stockout avoidance rate).
  - `Today's Consumed Demand` (Units sold network-wide).
- **Instant Alerts:**
  - Displays instant alert toasts (`st.toast` / notification banners) whenever any SKU crosses from optimal into `HIGH` risk.
- **Live SKU Telemetry Grid:**
  - Dynamic table showing real-time inventory depletion, updated daily demand velocity, live ROP thresholds, and active status tags (`🔴 HIGH ALERT`, `🟡 BUFFER WARN`, `🟢 OPTIMAL`).

---

## 4. CODE & UI QUALITY REQUIREMENTS
- Include clean error handling, tooltips for technical terms (EOQ, ROP, SHAP, MAPE), and empty state fallbacks.
- Fully responsive across desktop (wide/expanded) and tablet viewport widths.
- Implement the exact tab layouts, KPI cards, Plotly charts, and action buttons described above.