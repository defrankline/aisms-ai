import math

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION, Z95, LEAD_TIME_DAYS, DEFAULT_FORECAST_DAYS
from utils.stock_constants import build_stock_balance_query


def _health_score(current: float, optimal: float) -> float:
    # 100 at optimal; linearly penalize absolute deviation as a fraction of optimal
    denom = max(optimal, 1.0)
    deviation = abs(current - optimal) / denom
    return float(max(0.0, 100.0 - 100.0 * deviation))


def _status(current: float, safety: float, optimal: float) -> str:
    if current <= 0.0001:
        return "STOCKOUT"
    if current < safety:
        return "LOW"
    if current <= optimal * 1.2:  # within +20% of optimal
        return "HEALTHY"
    return "OVERSTOCK"


def _recent_sales_stats(company_id: int, warehouse_id: int, lookback_days: int = 90):
    """
    Returns DataFrame with columns:
      product_id, avg_daily (float), std_daily (float)
    Computed from last `lookback_days` of daily sales.
    """
    q = text("""
             SELECT sl.product_id,
                    DATE(s.date)     AS ds,
                    SUM(sl.quantity) AS qty
             FROM sales s
                      JOIN sale_lines sl ON sl.sale_id = s.id
             WHERE s.company_id = :company_id
               AND s.warehouse_id = :warehouse_id
               AND s.status IN ('PAID', 'DELIVERED', 'CONFIRMED')
               AND s.date >= CURRENT_DATE - (:lookback || ' days')::interval
             GROUP BY sl.product_id, DATE(s.date)
             """)
    df = pd.read_sql(
        q, engine,
        params={"company_id": company_id, "warehouse_id": warehouse_id, "lookback": lookback_days}
    )
    if df.empty:
        return pd.DataFrame(columns=["product_id", "avg_daily", "std_daily"])
    stats = (
        df.groupby("product_id")["qty"]
        .agg(avg_daily=lambda s: float(np.mean(s)), std_daily=lambda s: float(np.std(s, ddof=0)))
        .reset_index()
    )
    return stats


def _forecast_stats(company_id: int, warehouse_id: int, horizon_days: int):
    """
    Returns DataFrame with columns:
      product_id, avg_daily_fc (float), std_daily_fc (float)
    Computed from `sku_demand_forecast` over horizon_days from today.
    """
    q = text("""
             SELECT product_id,
                    predicted_quantity AS qty
             FROM sku_demand_forecast
             WHERE company_id = :company_id
               AND warehouse_id = :warehouse_id
               AND forecast_date >= CURRENT_DATE
               AND forecast_date < CURRENT_DATE + (:h || ' days')::interval
             """)
    df = pd.read_sql(
        q, engine,
        params={"company_id": company_id, "warehouse_id": warehouse_id, "h": horizon_days}
    )
    if df.empty:
        return pd.DataFrame(columns=["product_id", "avg_daily_fc", "std_daily_fc"])
    stats = (
        df.groupby("product_id")["qty"]
        .agg(avg_daily_fc=lambda s: float(np.mean(s)), std_daily_fc=lambda s: float(np.std(s, ddof=0)))
        .reset_index()
    )
    return stats


def optimize_inventory(
        company_id: int,
        warehouse_id: int,
        service_level_z: float | None = None,
        lead_time_days: int | None = None,
        horizon_days: int = DEFAULT_FORECAST_DAYS,
        lookback_days: int = 90
):
    """
    Produces rows for InventoryOptimization per (company, warehouse, product).
    """

    z = float(service_level_z if service_level_z is not None else Z95)
    lt = int(lead_time_days if lead_time_days is not None else LEAD_TIME_DAYS)
    horizon = int(horizon_days)

    # 1) Current stock (movement-aware)
    stock_q = build_stock_balance_query()
    stock_df = pd.read_sql(
        stock_q, engine,
        params={"company_id": company_id, "warehouse_id": warehouse_id}
    ).rename(columns={"current_stock": "current_stock"})

    # 2) Demand statistics: recent sales + forecast fallback/boost
    recent = _recent_sales_stats(company_id, warehouse_id, lookback_days=lookback_days)
    fc = _forecast_stats(company_id, warehouse_id, horizon_days=horizon)

    # Merge demand signals
    merged = pd.merge(recent, fc, on="product_id", how="outer")
    # Choose avg_daily_demand: prefer forecast if available; else recent; else 0
    merged["avg_daily_demand"] = merged.apply(
        lambda r: (r["avg_daily_fc"] if not pd.isna(r.get("avg_daily_fc", np.nan)) else r.get("avg_daily", 0.0)),
        axis=1
    )
    # Choose std: prefer forecast std if present, else recent std, else small epsilon
    merged["std_daily_demand"] = merged.apply(
        lambda r: (r["std_daily_fc"] if not pd.isna(r.get("std_daily_fc", np.nan)) else r.get("std_daily", 0.0)),
        axis=1
    )
    merged = merged[["product_id", "avg_daily_demand", "std_daily_demand"]].fillna(0.0)

    if stock_df.empty and merged.empty:
        return []

    # Join with stock; if stock missing, treat as 0
    base = pd.merge(merged, stock_df, on="product_id", how="outer").fillna({"current_stock": 0.0})
    base["avg_daily_demand"] = base["avg_daily_demand"].fillna(0.0)
    base["std_daily_demand"] = base["std_daily_demand"].fillna(0.0)

    results = []
    for _, r in base.iterrows():
        pid = int(r["product_id"])
        cur = float(r["current_stock"])
        avg_d = float(r["avg_daily_demand"])
        std_d = float(r["std_daily_demand"])

        # Safety stock and optimal
        safety = z * std_d * math.sqrt(max(lt, 1))
        optimal = avg_d * horizon + safety

        status = _status(cur, safety, max(optimal, 0.0))
        health = _health_score(cur, max(optimal, 0.0))

        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": pid,
            "current_stock": round(cur, 2),
            "avg_daily_demand": round(avg_d, 3),
            "safety_stock": round(safety, 3),
            "optimal_stock_level": round(optimal, 3),
            "stock_status": status,
            "inventory_health_score": round(health, 2),
            "forecast_horizon_days": horizon,
            "model_version": MODEL_VERSION
        })

    return results
