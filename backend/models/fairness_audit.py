"""
fairness_audit.py — Subgroup performance and fairness auditing for demand forecasting.

Performs empirical subgroup evaluation across demographic and operational segments:
- Geographic: Region
- Product: Category
- Operational: Store ID

Evaluates both:
1. Model Forecast Accuracy (MAPE, RMSE, MAE)
2. Inventory Optimization Policy (Average ROP, EOQ, Safety Stock)

Flags any segment with relative MAPE degradation exceeding a configurable threshold
(default: 25% degradation relative to the overall test set MAPE).
"""

import os
import json
import logging
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    BASE_DIR, RAW_DATA_PATH, MODEL_PATH,
    COL_DATE, COL_STORE_ID, COL_PRODUCT_ID, COL_CATEGORY, COL_REGION,
    COL_DEMAND, COL_PRICE, LEAD_TIME_DAYS, ORDER_COST,
    HOLDING_COST_PERCENT, SERVICE_LEVEL_Z
)
from backend.models.features import engineer_features
from backend.optimization.inventory_policy import (
    safety_stock, reorder_point, economic_order_quantity
)

logger = logging.getLogger(__name__)

# Default threshold for flagging relative MAPE degradation (e.g. 0.25 = +25% worse than baseline)
DEFAULT_DEGRADATION_THRESHOLD = 0.25


def compute_subgroup_metrics(
    df_eval: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    y_pred_col: str,
    overall_mape: float,
    threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
    min_sample_size: int = 30
) -> List[Dict[str, Any]]:
    """
    Evaluates forecast accuracy and inventory optimization metrics grouped by a column.
    If a subgroup has fewer than min_sample_size records, marks it as 'insufficient_sample_size'.
    """
    subgroup_results = []
    
    for group_val, group_df in df_eval.groupby(group_col):
        y_true = group_df[y_true_col].values
        y_pred = group_df[y_pred_col].values
        sample_count = len(y_true)
        
        if sample_count == 0:
            continue
            
        sub_mape = float(mean_absolute_percentage_error(y_true, y_pred))
        sub_rmse = float(root_mean_squared_error(y_true, y_pred))
        sub_mae = float(mean_absolute_error(y_true, y_pred))
        
        # Calculate relative degradation compared to overall baseline
        rel_degradation = (sub_mape - overall_mape) / overall_mape if overall_mape > 0 else 0.0
        
        if sample_count < min_sample_size:
            is_flagged = False
            status = "insufficient_sample_size"
            sample_size_warning = True
        else:
            is_flagged = bool(rel_degradation > threshold)
            status = "requires_review" if is_flagged else "nominal"
            sample_size_warning = False
        
        # Optimization policy analysis on the subgroup
        mean_demand = float(np.mean(y_true))
        std_demand = float(np.std(y_true)) if len(y_true) > 1 else mean_demand * 0.25
        avg_price = float(group_df[COL_PRICE].mean()) if COL_PRICE in group_df.columns else 20.0
        
        ss = safety_stock(std_demand, LEAD_TIME_DAYS, SERVICE_LEVEL_Z)
        rop = reorder_point(mean_demand, LEAD_TIME_DAYS, ss)
        annual_demand = max(mean_demand * 365, 1.0)
        holding_unit = max(avg_price * HOLDING_COST_PERCENT, 0.01)
        eoq = economic_order_quantity(annual_demand, ORDER_COST, holding_unit)
        
        subgroup_results.append({
            "segment": str(group_val),
            "sample_count": int(sample_count),
            "mape": round(sub_mape, 4),
            "rmse": round(sub_rmse, 3),
            "mae": round(sub_mae, 3),
            "relative_degradation_pct": round(rel_degradation * 100, 1),
            "is_flagged": is_flagged,
            "status": status,
            "sample_size_warning": sample_size_warning,
            "avg_actual_demand": round(mean_demand, 2),
            "avg_predicted_demand": round(float(np.mean(y_pred)), 2),
            "avg_safety_stock": round(ss, 1),
            "avg_reorder_point": round(rop, 1),
            "avg_recommended_eoq": round(eoq, 1)
        })
        
    # Sort by MAPE descending so worst-performing segments are on top
    subgroup_results.sort(key=lambda x: x["mape"], reverse=True)
    return subgroup_results


def run_fairness_audit(
    raw_data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
    save_artifact: bool = True
) -> Dict[str, Any]:
    """
    Executes end-to-end fairness and subgroup performance audit on held-out test data.
    """
    data_path = raw_data_path or RAW_DATA_PATH
    m_path = model_path or MODEL_PATH
    
    raw_df = pd.read_csv(data_path)
    raw_df[COL_DATE] = pd.to_datetime(raw_df[COL_DATE])
    
    # Feature engineering (time-consistent)
    df, feature_cols, season_encoder = engineer_features(raw_df)
    
    # Time-based split: last 30 days as held-out test set
    max_date = df[COL_DATE].max()
    test_start_date = max_date - pd.Timedelta(days=30)
    
    test_mask = df[COL_DATE] >= test_start_date
    train_mask = df[COL_DATE] < test_start_date
    
    test_df = df.loc[test_mask].copy()
    X_test = test_df[feature_cols]
    y_test = test_df[COL_DEMAND].values
    
    # Load trained model (or train if missing)
    if Path(m_path).exists():
        saved = joblib.load(m_path)
        model = saved["model"]
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, COL_DEMAND])
        
    # Overall predictions & metrics
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)
    test_df["predicted_demand"] = preds
    
    overall_mape = float(mean_absolute_percentage_error(y_test, preds))
    overall_rmse = float(root_mean_squared_error(y_test, preds))
    overall_mae = float(mean_absolute_error(y_test, preds))
    
    # Subgroup evaluations
    by_region = compute_subgroup_metrics(test_df, COL_REGION, COL_DEMAND, "predicted_demand", overall_mape, threshold)
    by_category = compute_subgroup_metrics(test_df, COL_CATEGORY, COL_DEMAND, "predicted_demand", overall_mape, threshold)
    by_store = compute_subgroup_metrics(test_df, COL_STORE_ID, COL_DEMAND, "predicted_demand", overall_mape, threshold)
    
    all_subgroups = by_region + by_category + by_store
    flagged_subgroups = [s["segment"] for s in all_subgroups if s["is_flagged"]]
    
    verdict = "REQUIRES_ATTENTION" if len(flagged_subgroups) > 0 else "NOMINAL_FAIRNESS"
    
    results = {
        "audit_timestamp": datetime.utcnow().isoformat(),
        "degradation_threshold_pct": round(threshold * 100, 1),
        "overall_metrics": {
            "MAPE": round(overall_mape, 4),
            "RMSE": round(overall_rmse, 3),
            "MAE": round(overall_mae, 3),
            "sample_count": int(len(test_df))
        },
        "subgroups": {
            "by_region": by_region,
            "by_category": by_category,
            "by_store": by_store
        },
        "summary": {
            "total_subgroups_audited": len(all_subgroups),
            "flagged_subgroups_count": len(flagged_subgroups),
            "flagged_subgroups": flagged_subgroups,
            "verdict": verdict
        }
    }
    
    if save_artifact:
        artifacts_dir = Path(BASE_DIR) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_path = artifacts_dir / "fairness_audit.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Fairness audit results saved to {out_path}")
        
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_fairness_audit()
    print("\n" + "=" * 60)
    print("FAIRNESS AUDIT REPORT")
    print("=" * 60)
    print(f"Overall Test MAPE: {res['overall_metrics']['MAPE'] * 100:.2f}% | RMSE: {res['overall_metrics']['RMSE']} | Samples: {res['overall_metrics']['sample_count']}")
    print(f"Verdict: {res['summary']['verdict']} (Flagged: {res['summary']['flagged_subgroups']})")
    print("\n--- Subgroups by Region ---")
    for s in res['subgroups']['by_region']:
        flag_str = " [FLAGGED]" if s['is_flagged'] else ""
        print(f"  {s['segment']:<15} | MAPE: {s['mape']*100:.2f}% | Rel: {s['relative_degradation_pct']:+5.1f}% | ROP: {s['avg_reorder_point']} | EOQ: {s['avg_recommended_eoq']}{flag_str}")
    print("\n--- Subgroups by Category ---")
    for s in res['subgroups']['by_category']:
        flag_str = " [FLAGGED]" if s['is_flagged'] else ""
        print(f"  {s['segment']:<15} | MAPE: {s['mape']*100:.2f}% | Rel: {s['relative_degradation_pct']:+5.1f}% | ROP: {s['avg_reorder_point']} | EOQ: {s['avg_recommended_eoq']}{flag_str}")
    print("\n--- Subgroups by Store ID ---")
    for s in res['subgroups']['by_store']:
        flag_str = " [FLAGGED]" if s['is_flagged'] else ""
        print(f"  {s['segment']:<15} | MAPE: {s['mape']*100:.2f}% | Rel: {s['relative_degradation_pct']:+5.1f}% | ROP: {s['avg_reorder_point']} | EOQ: {s['avg_recommended_eoq']}{flag_str}")
