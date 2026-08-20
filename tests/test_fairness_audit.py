"""
test_fairness_audit.py — Unit and integration tests for subgroup fairness auditing.
"""
import pytest
import numpy as np
import pandas as pd

from backend.models.fairness_audit import (
    compute_subgroup_metrics,
    run_fairness_audit,
    DEFAULT_DEGRADATION_THRESHOLD
)
from backend.config import RAW_DATA_PATH


def test_compute_subgroup_metrics_flags_degraded_subgroup():
    """
    Synthetic test: When one subgroup has deliberately high noise/error,
    confirm it gets flagged as 'requires_review' while nominal groups do not.
    """
    n_samples = 100
    
    # Group A: High accuracy (MAPE ~ 5%)
    y_true_a = np.array([50.0] * n_samples)
    y_pred_a = np.array([52.0] * n_samples)
    
    # Group B: High accuracy (MAPE ~ 5%)
    y_true_b = np.array([50.0] * n_samples)
    y_pred_b = np.array([48.0] * n_samples)
    
    # Group C: Degraded accuracy (MAPE ~ 50% - deliberately noisy)
    y_true_c = np.array([50.0] * n_samples)
    y_pred_c = np.array([75.0] * n_samples)
    
    df_synthetic = pd.DataFrame({
        "Region": ["North"] * n_samples + ["South"] * n_samples + ["West"] * n_samples,
        "demand": np.concatenate([y_true_a, y_true_b, y_true_c]),
        "predicted": np.concatenate([y_pred_a, y_pred_b, y_pred_c]),
        "Price": [20.0] * (3 * n_samples)
    })
    
    # Overall MAPE across all 300 samples
    y_all_true = df_synthetic["demand"].values
    y_all_pred = df_synthetic["predicted"].values
    overall_mape = float(np.mean(np.abs((y_all_true - y_all_pred) / y_all_true)))
    # overall_mape is roughly (0.04 + 0.04 + 0.50)/3 = 0.193
    
    results = compute_subgroup_metrics(
        df_eval=df_synthetic,
        group_col="Region",
        y_true_col="demand",
        y_pred_col="predicted",
        overall_mape=overall_mape,
        threshold=0.25
    )
    
    # Map results by region name
    res_by_name = {r["segment"]: r for r in results}
    
    # West (Group C) has MAPE 0.50 vs overall 0.193 (+159% relative degradation) -> MUST be flagged
    assert res_by_name["West"]["is_flagged"] is True
    assert res_by_name["West"]["status"] == "requires_review"
    assert res_by_name["West"]["relative_degradation_pct"] > 25.0
    
    # North and South have MAPE 0.04 vs overall 0.193 (negative relative degradation) -> MUST NOT be flagged
    assert res_by_name["North"]["is_flagged"] is False
    assert res_by_name["North"]["status"] == "nominal"
    assert res_by_name["South"]["is_flagged"] is False
    assert res_by_name["South"]["status"] == "nominal"


def test_compute_subgroup_metrics_no_false_positives_when_uniform():
    """
    Synthetic test: When all subgroups perform with identical accuracy,
    no subgroup should be flagged as degraded (no false positives).
    """
    n_samples = 100
    # All groups have ~10% error
    y_true = np.array([40.0] * n_samples)
    y_pred = np.array([44.0] * n_samples)
    
    df_uniform = pd.DataFrame({
        "Category": ["Apparel"] * n_samples + ["Electronics"] * n_samples + ["Groceries"] * n_samples,
        "demand": np.concatenate([y_true, y_true, y_true]),
        "predicted": np.concatenate([y_pred, y_pred, y_pred]),
        "Price": [15.0] * (3 * n_samples)
    })
    
    overall_mape = float(np.mean(np.abs((df_uniform["demand"] - df_uniform["predicted"]) / df_uniform["demand"])))
    
    results = compute_subgroup_metrics(
        df_eval=df_uniform,
        group_col="Category",
        y_true_col="demand",
        y_pred_col="predicted",
        overall_mape=overall_mape,
        threshold=0.25
    )
    
    for r in results:
        assert r["is_flagged"] is False
        assert r["status"] == "nominal"
        assert abs(r["relative_degradation_pct"]) < 1.0


def test_run_fairness_audit_real_data_execution():
    """
    Integration test: Runs the fairness audit against the real dataset
    and verifies schema completeness and reasonable metric bounds.
    """
    res = run_fairness_audit(save_artifact=False)
    
    assert "audit_timestamp" in res
    assert "degradation_threshold_pct" in res
    assert "overall_metrics" in res
    assert "subgroups" in res
    assert "summary" in res
    
    overall = res["overall_metrics"]
    assert 0.05 < overall["MAPE"] < 0.50
    assert overall["RMSE"] > 0
    assert overall["sample_count"] > 100
    
    subgroups = res["subgroups"]
    assert "by_region" in subgroups
    assert "by_category" in subgroups
    assert "by_store" in subgroups
    
    assert len(subgroups["by_region"]) == 5
    assert len(subgroups["by_category"]) == 5
    assert len(subgroups["by_store"]) == 5
    
    # Check that each subgroup includes optimization parameters (ROP, EOQ, SS)
    for seg in subgroups["by_region"]:
        assert "avg_reorder_point" in seg
        assert "avg_recommended_eoq" in seg
        assert "avg_safety_stock" in seg
        assert seg["avg_reorder_point"] > 0
        assert seg["avg_recommended_eoq"] > 0
