"""
generate_sample_data.py — Generates a realistic sample dataset matching the Kaggle schema
if data/raw/retail_store_inventory.csv does not already exist.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.config import (
    RAW_DATA_PATH, COL_DATE, COL_STORE_ID, COL_PRODUCT_ID, COL_CATEGORY,
    COL_REGION, COL_INVENTORY_LEVEL, COL_UNITS_SOLD, COL_UNITS_ORDERED,
    COL_PRICE, COL_DISCOUNT, COL_WEATHER, COL_PROMOTION, COL_COMPETITOR_PRICING,
    COL_SEASONALITY, COL_EPIDEMIC, COL_DEMAND
)

def generate_sample_dataset(filepath=RAW_DATA_PATH, num_days=180, num_stores=5, num_products=10):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"Dataset already exists at: {filepath}")
        return

    print(f"Generating synthetic retail store dataset -> {filepath}")
    np.random.seed(42)

    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq='D')
    store_ids = [f"S{str(i).zfill(3)}" for i in range(1, num_stores + 1)]
    product_ids = [f"P{str(i).zfill(3)}" for i in range(1, num_products + 1)]
    
    categories = ["Groceries", "Electronics", "Apparel", "Home & Kitchen", "Health & Beauty"]
    product_category_map = {p: categories[i % len(categories)] for i, p in enumerate(product_ids)}
    
    regions = ["North", "South", "East", "West", "Central"]
    store_region_map = {s: regions[i % len(regions)] for i, s in enumerate(store_ids)}
    
    weather_conditions = ["Sunny", "Rainy", "Cloudy", "Snowy", "Stormy"]
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall", 10: "Fall", 11: "Fall"}

    records = []

    for store in store_ids:
        region = store_region_map[store]
        for product in product_ids:
            category = product_category_map[product]
            base_price = np.random.uniform(10.0, 150.0)
            inventory = np.random.randint(50, 200)

            for d in dates:
                month = d.month
                season = season_map[month]
                weather = np.random.choice(weather_conditions, p=[0.45, 0.25, 0.15, 0.1, 0.05])
                
                # Epidemic flag (simulated spike in some window)
                epidemic = 1 if (num_days - 60 <= (d - dates[0]).days <= num_days - 30) and np.random.rand() > 0.3 else 0
                promotion = 1 if np.random.rand() < 0.2 else 0
                discount = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.25], p=[0.6, 0.15, 0.1, 0.1, 0.05]) if promotion else 0.0
                
                competitor_price = round(base_price * np.random.uniform(0.9, 1.15), 2)
                price = round(base_price * (1.0 - discount), 2)

                # Demand calculation with baseline + seasonal + promo + weather effects
                day_of_week_effect = 1.3 if d.weekday() in [5, 6] else 1.0
                promo_effect = 1.4 if promotion else 1.0
                epidemic_effect = 1.5 if (epidemic and category in ["Groceries", "Health & Beauty"]) else (0.7 if epidemic else 1.0)
                
                base_demand = np.random.poisson(lam=25)
                demand = int(base_demand * day_of_week_effect * promo_effect * epidemic_effect * (1.1 if weather == "Sunny" else 0.9))
                demand = max(1, demand)

                units_sold = min(inventory, demand)
                units_ordered = 0
                
                # Reorder logic simulation
                if inventory - units_sold < 40:
                    units_ordered = np.random.randint(50, 120)
                
                inventory = max(0, inventory - units_sold + (units_ordered if np.random.rand() > 0.7 else 0))

                records.append({
                    COL_DATE: d.strftime('%Y-%m-%d'),
                    COL_STORE_ID: store,
                    COL_PRODUCT_ID: product,
                    COL_CATEGORY: category,
                    COL_REGION: region,
                    COL_INVENTORY_LEVEL: inventory,
                    COL_UNITS_SOLD: units_sold,
                    COL_UNITS_ORDERED: units_ordered,
                    COL_PRICE: price,
                    COL_DISCOUNT: discount,
                    COL_WEATHER: weather,
                    COL_PROMOTION: promotion,
                    COL_COMPETITOR_PRICING: competitor_price,
                    COL_SEASONALITY: season,
                    COL_EPIDEMIC: epidemic,
                    COL_DEMAND: demand
                })

    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f"[OK] Generated {len(df)} records across {num_stores} stores and {num_products} products.")

if __name__ == "__main__":
    generate_sample_dataset()
