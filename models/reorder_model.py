import math

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION, Z95, LEAD_TIME_DAYS
from utils.stock_constants import build_stock_balance_query


def generate_reorder_suggestions(
        company_id: int,
        warehouse_id: int,
        service_level_z: float | None = None,
        lead_time_days: int | None = None,
        horizon_days: int = 30
):
    """
    Compute reorder suggestions per product for a given company & warehouse.

    Parameters:
      - service_level_z: z-score for service level (default Z95)
      - lead_time_days:  days of supplier lead time (default LEAD_TIME_DAYS)
      - horizon_days:    how many forecast days to use (default 30)
    """

    z = float(service_level_z if service_level_z is not None else Z95)
    lt = int(lead_time_days if lead_time_days is not None else LEAD_TIME_DAYS)

    # 1) Forecasted demand (next horizon_days) — avg & std per product
    forecast_query = text("""
        SELECT product_id, forecast_date, predicted_quantity
        FROM sku_demand_forecast
        WHERE company_id = :company_id
          AND warehouse_id = :warehouse_id
          AND forecast_date >= CURRENT_DATE
          AND forecast_date <  CURRENT_DATE + (:horizon_days || ' days')::interval
        ORDER BY product_id, forecast_date
    """)

    forecast_df = pd.read_sql(
        forecast_query,
        engine,
        params={
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "horizon_days": horizon_days
        }
    )

    # If no forecasts yet, nothing to do
    if forecast_df.empty:
        return []

    demand_stats = (
        forecast_df.groupby("product_id")["predicted_quantity"]
        .agg(avg_daily_demand="mean", std_daily_demand=lambda s: float(np.std(s, ddof=0)))
        .reset_index()
    )

    # 2) Current stock per product (shared movement logic)
    stock_query = build_stock_balance_query()
    stock_df = pd.read_sql(
        stock_query, engine,
        params={"company_id": company_id, "warehouse_id": warehouse_id}
    ).rename(columns={"current_stock": "stock_in_hand"})

    # Left-join with stock (missing → 0)
    merged = demand_stats.merge(stock_df, how="left", on="product_id")
    merged["stock_in_hand"] = merged["stock_in_hand"].fillna(0.0).astype(float)

    # 3) Compute safety stock, reorder point, suggested qty
    def _calc(row):
        avg_d = float(row["avg_daily_demand"])
        std_d = float(row["std_daily_demand"])
        ss = z * std_d * math.sqrt(lt)
        rp = (avg_d * lt) + ss
        suggested = max(0.0, rp - float(row["stock_in_hand"]))
        return pd.Series({
            "safety_stock": round(ss, 3),
            "reorder_point": round(rp, 3),
            "suggested_qty": round(suggested, 3),
            "avg_daily_demand": round(avg_d, 3)
        })

    out = merged.join(merged.apply(_calc, axis=1))

    # 4) Build results to insert
    results = []
    for _, r in out.iterrows():
        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(r["product_id"]),
            "reorder_point": float(r["reorder_point"]),
            "safety_stock": float(r["safety_stock"]),
            "suggested_qty": float(r["suggested_qty"]),
            "model_version": MODEL_VERSION
        })

    return results
