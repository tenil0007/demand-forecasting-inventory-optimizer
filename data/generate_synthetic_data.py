"""
generate_synthetic_data.py — Generates synthetic retail store inventory & demand data.

Schema aligns with Kaggle dataset (16 columns):
- Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold,
  Units Ordered, Price, Discount, Weather Condition, Promotion, Competitor Pricing,
  Seasonality, Epidemic, Demand

JUSTIFICATION FOR PREDICTION TARGET (Demand vs. Units Sold):
-----------------------------------------------------------
In retail supply chain management, `Units Sold` represents *censored demand*.
When a store experiences a stockout (Inventory Level = 0), `Units Sold` drops to 0,
even if 50 customers walked into the store attempting to purchase the item.
Training a forecasting model on `Units Sold` creates a severe negative bias: the model
learns that demand is zero during stockouts, leading to lower future orders, causing
further stockouts and lost revenue.
Therefore, `Demand` represents the true underlying consumer demand signal (unconstrained),
which is the correct target for demand forecasting models. `Units Sold` is retained
as an informative historical feature.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import random

def generate_retail_dataset(
    num_stores: int = 5,
    num_products: int = 10,
    start_date: str = "2026-02-19",
    days: int = 180,
    output_path: str = "data/raw/retail_store_inventory.csv"
):
    np.random.seed(42)
    random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    stores = [f"S{i:03d}" for i in range(1, num_stores + 1)]
    products = [f"P{i:03d}" for i in range(1, num_products + 1)]
    
    categories = ["Groceries", "Beverages", "Snacks", "Personal Care", "Household"]
    regions = ["North", "South", "East", "West", "Central"]
    weather_conditions = ["Sunny", "Rainy", "Cloudy", "Snowy", "Stormy"]
    
    prod_meta = {}
    for p in products:
        prod_meta[p] = {
            "category": random.choice(categories),
            "base_price": round(random.uniform(10.0, 100.0), 2),
            "base_demand": random.randint(15, 60)
        }
        
    store_meta = {}
    for s in stores:
        store_meta[s] = {
            "region": random.choice(regions)
        }
        
    rows = []
    
    for s in stores:
        for p in products:
            meta_p = prod_meta[p]
            meta_s = store_meta[s]
            
            # Initial inventory state
            current_inventory = random.randint(80, 150)
            
            for date in dates:
                month = date.month
                # Determine Seasonality from month
                if month in [12, 1, 2]:
                    season = "Winter"
                elif month in [3, 4, 5]:
                    season = "Spring"
                elif month in [6, 7, 8]:
                    season = "Summer"
                else:
                    season = "Fall"
                    
                day_of_week = date.dayofweek
                is_weekend = 1 if day_of_week in [5, 6] else 0
                
                # Promotion & Discount correlation
                has_promo = 1 if random.random() < 0.20 else 0
                discount = round(random.choice([0.05, 0.10, 0.15, 0.20]), 2) if has_promo else (round(random.choice([0.0, 0.05]), 2) if random.random() < 0.15 else 0.0)
                
                # Pricing & Competitor Pricing
                price = round(meta_p["base_price"] * (1.0 - discount), 2)
                # Competitor pricing correlated with base price + noise
                comp_noise = random.uniform(-0.15, 0.15)
                competitor_price = round(meta_p["base_price"] * (1.0 + comp_noise), 2)
                
                weather = random.choice(weather_conditions)
                epidemic = 1 if random.random() < 0.03 else 0
                
                # True underlying Demand calculation (unconstrained)
                demand_factor = 1.0
                if is_weekend:
                    demand_factor += 0.25
                if has_promo:
                    demand_factor += 0.35
                if discount > 0:
                    demand_factor += (discount * 1.5)
                if competitor_price > price:
                    demand_factor += 0.15  # our price is cheaper than competitor
                elif competitor_price < price:
                    demand_factor -= 0.10  # competitor is cheaper
                if weather in ["Rainy", "Stormy", "Snowy"]:
                    demand_factor -= 0.10
                if epidemic:
                    demand_factor += 0.40 if meta_p["category"] in ["Groceries", "Household"] else -0.30
                    
                noise = random.randint(-4, 4)
                unconstrained_demand = max(1, int(round(meta_p["base_demand"] * demand_factor + noise)))
                
                # Units Sold is CENSORED by current available inventory
                units_sold = min(current_inventory, unconstrained_demand)
                
                # Update inventory after sales
                current_inventory -= units_sold
                
                # Simple periodic inventory replenishment simulation
                units_ordered = 0
                if current_inventory < 30:
                    units_ordered = random.randint(60, 120)
                    # Simulated stock arrival
                    current_inventory += units_ordered
                    
                row = {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Store ID": s,
                    "Product ID": p,
                    "Category": meta_p["category"],
                    "Region": meta_s["region"],
                    "Inventory Level": current_inventory,
                    "Units Sold": units_sold,
                    "Units Ordered": units_ordered,
                    "Price": price,
                    "Discount": discount,
                    "Weather Condition": weather,
                    "Promotion": has_promo,
                    "Competitor Pricing": competitor_price,
                    "Seasonality": season,
                    "Epidemic": epidemic,
                    "Demand": unconstrained_demand
                }
                rows.append(row)
                
    df = pd.DataFrame(rows)
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} rows and {len(df.columns)} columns at {output_path}")
    return df

if __name__ == "__main__":
    generate_retail_dataset()
