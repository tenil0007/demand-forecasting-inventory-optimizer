import logging
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from typing import Dict, List, Any, Optional

from backend.config import (
    MODEL_PATH, RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID,
    COL_CATEGORY, COL_REGION, COL_PRICE, COL_DISCOUNT, COL_WEATHER,
    COL_PROMOTION, COL_COMPETITOR_PRICING, COL_SEASONALITY, COL_EPIDEMIC, COL_DEMAND
)
from backend.models.features import get_season_from_date

logger = logging.getLogger(__name__)

class ModelUnavailableError(Exception):
    """Raised when the ML model artifact is missing or incomplete."""
    pass

class ModelInferenceError(Exception):
    """Raised when feature construction or inference fails."""
    pass

class EntityNotFoundError(Exception):
    """Raised when a requested Store ID or Product ID is not in the dataset."""
    pass


class ForecastModel:
    """Wrapper for the XGBoost demand forecasting model."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_cols = None
        self.season_encoder = None
        self.prediction_interval_q95 = None
        self.is_loaded = False
        
    def load(self, model_path: Optional[str] = None) -> bool:
        """Load the trained model and features from disk. Raises ModelUnavailableError if missing or corrupted."""
        target_path = Path(model_path or self.model_path)
        if not target_path.exists():
            self.is_loaded = False
            raise ModelUnavailableError(f"Model artifact not found at {target_path}. Run training pipeline first.")
            
        try:
            data = joblib.load(target_path)
            if not isinstance(data, dict) or "model" not in data or "feature_cols" not in data:
                raise ModelUnavailableError(f"Corrupted model artifact at {target_path}: missing required keys.")
            self.model = data["model"]
            self.feature_cols = data["feature_cols"]
            self.season_encoder = data.get("season_encoder")
            self.prediction_interval_q95 = float(data.get("prediction_interval_q95", 15.0))
            self.is_loaded = True
            logger.info(f"Successfully loaded model from {target_path} (Calibration Q95: +/-{self.prediction_interval_q95:.2f})")
            return True
        except Exception as e:
            self.is_loaded = False
            if isinstance(e, ModelUnavailableError):
                raise
            raise ModelUnavailableError(f"Failed to load model from {target_path}: {e}") from e

    def _encode_seasons(self, season_names: List[str]) -> List[float]:
        """Encode seasonality using fitted season_encoder."""
        if self.season_encoder is not None:
            try:
                return self.season_encoder.transform(pd.DataFrame({'Seasonality': season_names})).ravel().tolist()
            except Exception as e:
                logger.warning(f"Encoder transform warning ({e}), using canonical encoding.")
        mapping = {"Spring": 0.0, "Summer": 1.0, "Autumn": 2.0, "Winter": 3.0}
        return [float(mapping.get(s, 0.0)) for s in season_names]

    def _get_historical_sku_data(self, store_id: str, product_id: str) -> pd.DataFrame:
        """Fetch real sorted historical records for the SKU from RAW_DATA_PATH."""
        raw_path = Path(RAW_DATA_PATH)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw dataset file not found at {raw_path}")
            
        df = pd.read_csv(raw_path)
        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
        
        sku_df = df[(df[COL_STORE_ID] == store_id) & (df[COL_PRODUCT_ID] == product_id)].sort_values(COL_DATE).reset_index(drop=True)
        if sku_df.empty:
            raise EntityNotFoundError(f"Store '{store_id}' and Product '{product_id}' combination not found in dataset.")
        return sku_df

    def predict_demand(self, store_id: str = "S001", product_id: str = "P0001", 
                       days_ahead: int = 14) -> Dict[str, Any]:
        """
        Predict demand for a store/product using true multi-step recursive autoregression.
        """
        if not self.is_loaded:
            self.load()
            
        sku_df = self._get_historical_sku_data(store_id, product_id)
        last_row = sku_df.iloc[-1]
        last_date = last_row[COL_DATE]
        
        # Historical baseline statistics from recent 28-day window
        recent_window = sku_df.tail(28)
        recent_price = float(recent_window[COL_PRICE].median()) if COL_PRICE in recent_window else float(last_row[COL_PRICE])
        recent_discount = float(recent_window[COL_DISCOUNT].median()) if COL_DISCOUNT in recent_window else 0.0
        recent_comp_price = float(recent_window[COL_COMPETITOR_PRICING].median()) if COL_COMPETITOR_PRICING in recent_window else recent_price
        
        # Ratios
        price_disc_ratio = recent_discount / (recent_price + 1e-5)
        price_comp_ratio = recent_price / (recent_comp_price + 1e-5)
        
        # Modes for binary flags and weather
        promo_mode = int(recent_window[COL_PROMOTION].mode().iloc[0]) if COL_PROMOTION in recent_window and not recent_window[COL_PROMOTION].empty else 0
        epidemic_mode = int(recent_window[COL_EPIDEMIC].mode().iloc[0]) if COL_EPIDEMIC in recent_window and not recent_window[COL_EPIDEMIC].empty else 0
        weather_mode = str(recent_window[COL_WEATHER].mode().iloc[0]) if COL_WEATHER in recent_window and not recent_window[COL_WEATHER].empty else "Sunny"
        
        # Historical demand sequence
        history_demands = sku_df[COL_DEMAND].tolist()
        
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
        predictions = []
        lower_bounds = []
        upper_bounds = []
        feature_rows = []
        
        cal_bound = self.prediction_interval_q95 if self.prediction_interval_q95 is not None else 15.0
        
        # Recursive Forecasting Loop
        for d in future_dates:
            # Calendar features
            dow = d.dayofweek
            month = d.month
            quarter = d.quarter
            is_weekend = int(dow in [5, 6])
            
            # Seasonality
            season_name = get_season_from_date(d)
            encoded_season = self._encode_seasons([season_name])[0]
            
            # Lag features from history_demands
            lag_1 = float(history_demands[-1]) if len(history_demands) >= 1 else 20.0
            lag_7 = float(history_demands[-7]) if len(history_demands) >= 7 else lag_1
            lag_14 = float(history_demands[-14]) if len(history_demands) >= 14 else lag_7
            lag_28 = float(history_demands[-28]) if len(history_demands) >= 28 else lag_14
            
            # Rolling window stats
            tail_7 = history_demands[-7:]
            roll_mean_7 = float(np.mean(tail_7)) if tail_7 else lag_1
            roll_std_7 = float(np.std(tail_7)) if len(tail_7) > 1 else 2.0
            
            tail_28 = history_demands[-28:]
            roll_mean_28 = float(np.mean(tail_28)) if tail_28 else lag_1
            roll_std_28 = float(np.std(tail_28)) if len(tail_28) > 1 else 3.0
            
            row = {
                COL_DATE: d,
                COL_PRICE: recent_price,
                COL_DISCOUNT: recent_discount,
                COL_COMPETITOR_PRICING: recent_comp_price,
                'day_of_week': dow,
                'month': month,
                'quarter': quarter,
                'is_weekend': is_weekend,
                'demand_lag_1': lag_1,
                'demand_lag_7': lag_7,
                'demand_lag_14': lag_14,
                'demand_lag_28': lag_28,
                'demand_rolling_mean_7': roll_mean_7,
                'demand_rolling_std_7': roll_std_7,
                'demand_rolling_mean_28': roll_mean_28,
                'demand_rolling_std_28': roll_std_28,
                'price_discount_ratio': price_disc_ratio,
                'price_competitor_ratio': price_comp_ratio,
                COL_PROMOTION: promo_mode,
                COL_EPIDEMIC: epidemic_mode,
                'seasonality_encoded': encoded_season,
                'weather_Cloudy': 1 if weather_mode == "Cloudy" else 0,
                'weather_Rainy': 1 if weather_mode == "Rainy" else 0,
                'weather_Snowy': 1 if weather_mode == "Snowy" else 0,
                'weather_Stormy': 1 if weather_mode == "Stormy" else 0,
                'weather_Sunny': 1 if weather_mode == "Sunny" else 0
            }
            
            # Check feature schema
            missing_cols = [c for c in self.feature_cols if c not in row]
            if missing_cols:
                raise ModelInferenceError(f"Missing required model features during inference: {missing_cols}")
                
            X_step = pd.DataFrame([row])[self.feature_cols]
            pred_val = float(np.clip(self.model.predict(X_step)[0], 0, None))
            
            # Recursive append
            history_demands.append(pred_val)
            predictions.append(round(pred_val, 2))
            lower_bounds.append(round(max(0.0, pred_val - cal_bound), 2))
            upper_bounds.append(round(pred_val + cal_bound, 2))
            feature_rows.append(row)
            
        self._last_feature_df = pd.DataFrame(feature_rows)
        results = {
            "store_id": store_id,
            "product_id": product_id,
            "dates": [str(d.date()) for d in future_dates],
            "predicted_demand": predictions,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
            "prediction_interval_q95": cal_bound,
            "metadata": {
                "forecast_horizon_days": days_ahead,
                "forecasting_strategy": "recursive_autoregression",
                "interval_type": "empirical_prediction_interval_95pct",
                "calibration_margin_units": round(cal_bound, 2),
                "baseline_weather": weather_mode,
                "baseline_promotion": promo_mode
            }
        }
        return results

    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for predict_demand."""
        return self.predict_demand(*args, **kwargs)

    def explain_forecast(
        self,
        df_features: Optional[pd.DataFrame] = None, 
        store_id: str = "S001",
        product_id: str = "P0001",
        return_structured: bool = False,
        top_k: int = 5,
        days_ahead: int = 7
    ) -> Any:
        """
        Returns top SHAP contributing features computed from real model and features.
        """
        if not self.is_loaded:
            self.load()
            
        if df_features is None:
            if not hasattr(self, "_last_feature_df") or self._last_feature_df is None or self._last_feature_df.empty:
                self.predict_demand(store_id, product_id, days_ahead=days_ahead)
            df_features = getattr(self, "_last_feature_df", None)
            
        if df_features is None or df_features.empty:
            if return_structured:
                return []
            return "SHAP feature attribution unavailable: feature matrix empty."
            
        for c in self.feature_cols:
            if c not in df_features.columns:
                raise ModelInferenceError(f"Feature '{c}' missing for SHAP explanation calculation.")

        X = df_features[self.feature_cols]
        
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            mean_signed_shap = shap_values.mean(axis=0)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
            
            structured_data = []
            for idx in top_indices:
                feat_name = self.feature_cols[idx]
                abs_val = round(float(mean_abs_shap[idx]), 2)
                signed_val = round(float(mean_signed_shap[idx]), 2)
                direction = "positive" if signed_val >= 0 else "negative"
                structured_data.append({
                    "feature": feat_name,
                    "impact": abs_val,
                    "signed_impact": signed_val,
                    "direction": direction
                })
                
            if return_structured:
                return structured_data

            explanations = []
            for item in structured_data[:3]:
                feat_name = item["feature"]
                sgn = "+" if item["signed_impact"] >= 0 else ""
                explanations.append(f"{feat_name} ({sgn}{item['signed_impact']:.2f} impact)")
            return f"Primary model drivers: {', '.join(explanations)}."
        except Exception as e:
            logger.error(f"Error during SHAP calculation: {e}")
            if return_structured:
                return []
            return f"SHAP attribution calculation unavailable ({type(e).__name__})."

    def explain_forecast_structured(
        self,
        df_features: Optional[pd.DataFrame] = None,
        store_id: str = "S001",
        product_id: str = "P0001",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Structured SHAP feature explanation helper."""
        return self.explain_forecast(
            df_features=df_features,
            store_id=store_id,
            product_id=product_id,
            return_structured=True,
            top_k=top_k
        )

forecast_model = ForecastModel()
