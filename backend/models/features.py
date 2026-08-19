"""
features.py — Feature engineering definitions and utilities for demand forecasting.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import OrdinalEncoder

RAW_COLUMNS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price",
    "Discount", "Weather Condition", "Promotion", "Competitor Pricing",
    "Seasonality", "Epidemic", "Demand"
]

def get_season_from_date(dt) -> str:
    """Map calendar date/month to seasonality category consistent with the dataset."""
    month = dt.month if hasattr(dt, "month") else pd.to_datetime(dt).month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Summer"

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], OrdinalEncoder]:
    """
    Perform feature engineering on raw retail store inventory dataframe without data leakage.
    Returns:
        (transformed_df, feature_column_names, fitted_season_encoder)
    """
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        
    df = df.sort_values(by=["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    
    # Calendar Features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Lag and Rolling Stats per store-product (target: Demand)
    # Using true Demand to learn customer demand dynamics
    group = df.groupby(["Store ID", "Product ID"])["Demand"]
    lag_cols = []
    for lag in [1, 7, 14, 28]:
        col_name = f"demand_lag_{lag}"
        df[col_name] = group.shift(lag)
        lag_cols.append(col_name)
        
    rolling_cols = [
        "demand_rolling_mean_7",
        "demand_rolling_std_7",
        "demand_rolling_mean_28",
        "demand_rolling_std_28"
    ]
    df["demand_rolling_mean_7"] = group.transform(lambda x: x.shift(1).rolling(7).mean())
    df["demand_rolling_std_7"] = group.transform(lambda x: x.shift(1).rolling(7).std())
    df["demand_rolling_mean_28"] = group.transform(lambda x: x.shift(1).rolling(28).mean())
    df["demand_rolling_std_28"] = group.transform(lambda x: x.shift(1).rolling(28).std())
    
    # Price & Competitor Pricing Features
    df["price_discount_ratio"] = df["Discount"] / (df["Price"] + 1e-5)
    df["price_competitor_ratio"] = df["Price"] / (df["Competitor Pricing"] + 1e-5)
    
    # Binary Features (Promotion, Epidemic)
    df["Promotion"] = df["Promotion"].map({"Yes": 1, "No": 0, 1: 1, 0: 0, "1": 1, "0": 0}).fillna(0).astype(int)
    df["Epidemic"] = df["Epidemic"].map({"Yes": 1, "No": 0, 1: 1, 0: 0, "1": 1, "0": 0}).fillna(0).astype(int)
    
    # Weather One-Hot Encoding
    weather_dummies = pd.get_dummies(df["Weather Condition"], prefix="weather")
    df = pd.concat([df, weather_dummies], axis=1)
    
    # Seasonality Encoding
    season_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if "Seasonality" in df.columns:
        df["seasonality_encoded"] = season_encoder.fit_transform(df[["Seasonality"]])
    else:
        df["seasonality_encoded"] = 0
        
    # Drop rows where lag or rolling features are NaN (first 28 rows per store-product)
    # This completely eliminates target/future data leakage from bfill()
    df = df.dropna(subset=lag_cols + rolling_cols).reset_index(drop=True)
    
    exclude_cols = [
        "Date", "Store ID", "Product ID", "Category", "Region",
        "Inventory Level", "Units Sold", "Units Ordered", "Weather Condition",
        "Seasonality", "Demand"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return df, feature_cols, season_encoder
