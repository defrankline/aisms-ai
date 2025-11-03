import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.stock_constants import build_stock_balance_query

Z95 = 1.65  # 95% service level


def optimize_inventory(company_id, warehouse_id):
    """
    Calculate optimal stock levels and inventory health per product.
    """
    # Forecasted demand (30 days)
    forecast_query = text("""
                          SELECT product_id, predicted_quantity
                          FROM sku_demand_forecast
                          WHERE company_id = :company_id
                            AND warehouse_id = :warehouse_id
                            AND forecast_date >= CURRENT_DATE
                            AND forecast_date <= CURRENT_DATE + INTERVAL '30 days'
                          """)

    # Current stock from movements
    stock_query = build_stock_balance_query()
    stock_df = pd.read_sql(stock_query, engine,
                           params={"company_id": company_id, "warehouse_id": warehouse_id})

    forecast_df = pd.read_sql(forecast_query, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})

    if forecast_df.empty or stock_df.empty:
        return []

    results = []
    for pid, group in forecast_df.groupby("product_id"):
        avg_demand = group["predicted_quantity"].mean()
        std_demand = group["predicted_quantity"].std(ddof=0)
        lead_time_days = 7

        safety_stock = Z95 * std_demand * np.sqrt(lead_time_days)
        optimal_stock = (avg_demand * lead_time_days) + safety_stock

        stock_row = stock_df.loc[stock_df["product_id"] == pid]
        current_stock = float(stock_row["current_stock"].values[0]) if not stock_row.empty else 0.0

        # Determine stock status
        if current_stock < safety_stock:
            status = "UNDERSTOCK"
        elif current_stock > (optimal_stock * 1.3):
            status = "OVERSTOCK"
        else:
            status = "OPTIMAL"

        # Health score (100 = perfect)
        score = 100 - abs(optimal_stock - current_stock) / (optimal_stock + 1) * 100
        score = round(max(0, min(score, 100)), 2)

        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(pid),
            "current_stock": current_stock,
            "avg_daily_demand": round(avg_demand, 3),
            "safety_stock": round(safety_stock, 3),
            "optimal_stock_level": round(optimal_stock, 3),
            "stock_status": status,
            "inventory_health_score": score,
            "forecast_horizon_days": 30
        })

    return results
