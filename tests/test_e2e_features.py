"""
test_e2e_features.py — Comprehensive end-to-end verification script
Testing all models, agents, database, APIs, optimization, and frontend data feeds.
"""
import sys
import os
import unittest
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import (
    RAW_DATA_PATH, MODEL_PATH, LEAD_TIME_DAYS, ORDER_COST,
    HOLDING_COST_PERCENT, SERVICE_LEVEL_Z
)
from backend.models.forecast_model import ForecastModel
from backend.optimization.inventory_policy import (
    safety_stock, reorder_point, economic_order_quantity,
    stockout_risk_flag, generate_recommendation
)
from backend.db.database import init_db, SessionLocal
from backend.db.models import AuditLog
from backend.agents.graph import create_agent_graph, run_agent_pipeline, resume_agent


class TestFullSystemFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_01_dataset_integrity(self):
        """Verify the raw CSV dataset exists and has valid structure."""
        import pandas as pd
        self.assertTrue(os.path.exists(RAW_DATA_PATH), "Dataset CSV must exist")
        df = pd.read_csv(RAW_DATA_PATH)
        self.assertGreater(len(df), 100, "Dataset should have records")
        print(f"  [PASS] Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")

    def test_02_forecast_model_inference(self):
        """Test XGBoost model loading, recursive prediction, and SHAP explainability."""
        model = ForecastModel()
        model.load(MODEL_PATH)
        
        # Test predict
        res = model.predict_demand(store_id="S001", product_id="P0001", days_ahead=14)
        self.assertIn("dates", res)
        self.assertIn("predicted_demand", res)
        self.assertIn("lower_bound", res)
        self.assertIn("upper_bound", res)
        self.assertEqual(len(res["predicted_demand"]), 14)
        
        # Test explain (plain English)
        explanation = model.explain_forecast(store_id="S001", product_id="P0001")
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 10)

        # Test explain (structured SHAP)
        structured_shap = model.explain_forecast(store_id="S001", product_id="P0001", return_structured=True, top_k=5)
        self.assertIsInstance(structured_shap, list)
        self.assertGreaterEqual(len(structured_shap), 1)
        first_feat = structured_shap[0]
        self.assertIn("feature", first_feat)
        self.assertIn("impact", first_feat)
        self.assertIn("signed_impact", first_feat)
        self.assertIn("direction", first_feat)
        self.assertIn(first_feat["direction"], ["positive", "negative"])
        
        print(f"  [PASS] Forecast Model Inference: 14-day predictions & structured SHAP values generated.")

    def test_03_inventory_optimization_math(self):
        """Test all optimization formulas against verified bounds."""
        ss = safety_stock(demand_std=5.0, lead_time_days=7, service_level_z=1.65)
        self.assertAlmostEqual(ss, 1.65 * 5.0 * (7 ** 0.5), places=2)

        rop = reorder_point(avg_daily_demand=20.0, lead_time_days=7, safety_stock_units=ss)
        self.assertAlmostEqual(rop, (20.0 * 7) + ss, places=2)

        eoq = economic_order_quantity(annual_demand=7300, order_cost=50, holding_cost_per_unit=10)
        self.assertAlmostEqual(eoq, ((2 * 7300 * 50) / 10) ** 0.5, places=2)

        # Risk flag tests
        self.assertEqual(stockout_risk_flag(current_inventory=10, forecasted_demand=30, reorder_point_val=50), "HIGH")
        self.assertEqual(stockout_risk_flag(current_inventory=55, forecasted_demand=30, reorder_point_val=50), "MEDIUM")
        self.assertEqual(stockout_risk_flag(current_inventory=100, forecasted_demand=30, reorder_point_val=50), "LOW")
        print("  [PASS] Inventory Optimization formulas & risk classifications mathematically verified.")

    def test_04_langgraph_agent_pipeline(self):
        """Test LangGraph 4-agent execution and human-in-the-loop approval."""
        graph = create_agent_graph()
        self.assertIsNotNone(graph)

        thread_id = f"test_thread_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        state = run_agent_pipeline(store_id="S001", product_id="P0001", thread_id=thread_id)
        
        self.assertIn("store_id", state)
        self.assertIn("risk_level", state)
        self.assertIn("recommendation", state)
        
        # Test approval resume
        resumed_state = resume_agent(thread_id=thread_id, decision="approved", approver="TestAdmin")
        self.assertIsNotNone(resumed_state)
        print("  [PASS] LangGraph 4-Agent Pipeline executed with HITL pause and resume.")

    def test_05_database_audit_trail(self):
        """Test reading and writing to SQLite Audit Log."""
        db = SessionLocal()
        try:
            entry = AuditLog(
                store_id="S001",
                product_id="P0001",
                recommended_qty=100.0,
                risk_level="HIGH",
                reasoning_snapshot="Automated verification test log",
                decision="approved",
                approver="VerificationSuite",
                decision_timestamp=datetime.utcnow()
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            
            self.assertIsNotNone(entry.id)
            fetched = db.query(AuditLog).filter(AuditLog.id == entry.id).first()
            self.assertEqual(fetched.decision, "approved")
        finally:
            db.close()
        print("  [PASS] Database Layer: Audit trail persistence and querying verified.")

    def test_06_fastapi_endpoints(self):
        """Test FastAPI router definitions and schemas."""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        client = TestClient(app)
        
        # 1. Health check / root
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        
        # 2. Forecast endpoint
        r = client.get("/forecast/S001/P0001")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("forecast", data)
        
        # 3. Reorder endpoint
        r = client.get("/reorder/S001/P0001")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("risk_level", data)
        self.assertIn("recommendation", data)
        
        # 4. Audit log endpoint
        r = client.get("/audit/log")
        self.assertEqual(r.status_code, 200)
        
        # 5. Agent query endpoint
        r = client.post("/agent/query", json={"query": "What is the forecast for store S001 product P0001?"})
        self.assertEqual(r.status_code, 200)
        
        # 6. Negative test: invalid entity ID format returns 400
        r_bad = client.get("/forecast/INVALID_STORE/P0001")
        self.assertEqual(r_bad.status_code, 400)
        
        # 7. Negative test: non-existent SKU returns 404
        r_404 = client.get("/forecast/S001/P9999")
        self.assertEqual(r_404.status_code, 404)
        
        print("  [PASS] FastAPI Endpoints: All positive & negative route responses verified.")


if __name__ == "__main__":
    unittest.main()
