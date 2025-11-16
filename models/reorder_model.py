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
        horizon_days: int = 30,
):
    """
    Generates reorder suggestions per product for a given company + warehouse.
    Fully clamped, safe, negative-proof.
    """

    z = float(service_level_z if service_level_z is not None else Z95)
    lt = int(lead_time_days if lead_time_days is not None else LEAD_TIME_DAYS)

    # ---------------------------------------------------------
    # 1) Load forecast data (clamp negative predicted quantities)
    # ---------------------------------------------------------
    forecast_query = text("""
                          SELECT product_id, forecast_date, predicted_quantity
                          FROM sku_demand_forecast
                          WHERE company_id = :company_id
                            AND warehouse_id = :warehouse_id
                            AND forecast_date >= CURRENT_DATE
                            AND forecast_date < CURRENT_DATE + (:horizon_days || ' days')::interval
                          ORDER BY product_id, forecast_date;
                          """)

    forecast_df = pd.read_sql(
        forecast_query,
        engine,
        params={
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "horizon_days": horizon_days,
        },
    )

    if forecast_df.empty:
        return []

    # Clamp negatives
    forecast_df["predicted_quantity"] = forecast_df["predicted_quantity"].apply(
        lambda x: max(0.0, float(x))
    )

    # ---------------------------------------------------------
    # 2) Compute demand statistics
    # ---------------------------------------------------------
    demand_stats = (
        forecast_df.groupby("product_id")["predicted_quantity"]
        .agg(
            avg_daily_demand=lambda s: max(0.0, float(s.mean())),
            std_daily_demand=lambda s: max(0.0, float(np.std(s, ddof=0))),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 3) Load stock in hand
    # ---------------------------------------------------------
    stock_df = pd.read_sql(
        build_stock_balance_query(),
        engine,
        params={"company_id": company_id, "warehouse_id": warehouse_id},
    ).rename(columns={"current_stock": "stock_in_hand"})

    stock_df["stock_in_hand"] = stock_df["stock_in_hand"].astype(float)

    # ---------------------------------------------------------
    # 4) Merge forecasts + stock
    # ---------------------------------------------------------
    merged = demand_stats.merge(stock_df, how="left", on="product_id")
    merged["stock_in_hand"] = merged["stock_in_hand"].fillna(0.0)

    # ---------------------------------------------------------
    # 5) Compute reorder metrics
    # ---------------------------------------------------------
    def calc(row):
        avg_d = max(0.0, float(row["avg_daily_demand"]))
        std_d = max(0.0, float(row["std_daily_demand"]))
        stock = max(0.0, float(row["stock_in_hand"]))

        # Safety Stock = z * std * sqrt(LT)
        safety_stock = max(0.0, z * std_d * math.sqrt(lt))

        # Reorder Point = avg_daily_demand * lead_time + safety_stock
        reorder_point = max(0.0, (avg_d * lt) + safety_stock)

        # Suggested qty = max(0, reorder_point - stock)
        suggested_qty = max(0.0, reorder_point - stock)

        return pd.Series({
            "safety_stock": round(safety_stock, 3),
            "reorder_point": round(reorder_point, 3),
            "suggested_qty": round(suggested_qty, 3),
        })

    computed = merged.join(merged.apply(calc, axis=1))

    # ---------------------------------------------------------
    # 6) Build output rows
    # ---------------------------------------------------------
    results = []
    for _, r in computed.iterrows():
        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(r["product_id"]),
            "avg_daily_demand": float(r["avg_daily_demand"]),
            "std_daily_demand": float(r["std_daily_demand"]),
            "reorder_point": float(r["reorder_point"]),
            "safety_stock": float(r["safety_stock"]),
            "suggested_qty": float(r["suggested_qty"]),
            "stock_in_hand": float(r["stock_in_hand"]),
            "model_version": MODEL_VERSION,
        })

    return results
