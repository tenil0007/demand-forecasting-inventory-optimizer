"""
Application configuration loaded from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data Paths ---
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", str(BASE_DIR / "data" / "raw" / "retail_store_inventory.csv"))
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", str(BASE_DIR / "data" / "processed" / "features.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "backend" / "models" / "xgb_model.pkl"))

# --- Database & Persistence ---
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'audit.db'}")
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", str(BASE_DIR / "data" / "checkpoints.db"))

# --- Security & CORS ---
ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",") if origin.strip()
]

# --- Ollama LLM ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# --- Langfuse Observability ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
OBSERVABILITY_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# --- Inventory Optimization Parameters ---
LEAD_TIME_DAYS = int(os.getenv("LEAD_TIME_DAYS", "7"))
ORDER_COST = float(os.getenv("ORDER_COST", "50.0"))
HOLDING_COST_PERCENT = float(os.getenv("HOLDING_COST_PERCENT", "0.20"))
SERVICE_LEVEL_Z = float(os.getenv("SERVICE_LEVEL_Z", "1.65"))

# --- Column Name Constants (from Kaggle dataset) ---
COL_DATE = "Date"
COL_STORE_ID = "Store ID"
COL_PRODUCT_ID = "Product ID"
COL_CATEGORY = "Category"
COL_REGION = "Region"
COL_INVENTORY_LEVEL = "Inventory Level"
COL_UNITS_SOLD = "Units Sold"
COL_UNITS_ORDERED = "Units Ordered"
COL_PRICE = "Price"
COL_DISCOUNT = "Discount"
COL_WEATHER = "Weather Condition"
COL_PROMOTION = "Promotion"
COL_COMPETITOR_PRICING = "Competitor Pricing"
COL_SEASONALITY = "Seasonality"
COL_EPIDEMIC = "Epidemic"
COL_DEMAND = "Demand"
