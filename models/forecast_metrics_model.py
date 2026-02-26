import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _drift_level(s):
    if s >= 0.75: return "HIGH"
    if s >= 0.45: return "MEDIUM"
    return "LOW"


def compute_forecast_metrics(company_id: int, warehouse_id: int, lookback_days: int = 180):
    # Forecasts in window
    qf = text("""
              SELECT product_id, forecast_date AS ds, predicted_quantity
              FROM sku_demand_forecast
              WHERE company_id = :c
                AND warehouse_id = :w
                AND model_version = :mv
                AND forecast_date >= CURRENT_DATE - (:lb || ' days')::interval
              """)
    fc = pd.read_sql(qf, engine,
                     params={"c": company_id, "w": warehouse_id, "lb": int(lookback_days), "mv": MODEL_VERSION})
    if fc.empty:
        return []

    # Actual demand proxy = SALE_OUT
    qa = text("""
              SELECT product_id, DATE(date_received) AS ds, SUM(ABS(quantity)) AS actual_qty
              FROM stock_movements
              WHERE approved = true
                AND company_id = :c
                AND warehouse_id = :w
                AND movement_type_id = 2
                AND date_received >= NOW() - (:lb || ' days')::interval
              GROUP BY product_id, DATE(date_received)
              """)
    ac = pd.read_sql(qa, engine, params={"c": company_id, "w": warehouse_id, "lb": int(lookback_days)})
    if ac.empty:
        return []

    df = fc.merge(ac, on=["product_id", "ds"], how="inner")
    if df.empty:
        return []

    df["predicted_quantity"] = pd.to_numeric(df["predicted_quantity"], errors="coerce").fillna(0.0)
    df["actual_qty"] = pd.to_numeric(df["actual_qty"], errors="coerce").fillna(0.0)

    rows = []
    eps = 1e-6

    for pid, g in df.groupby("product_id"):
        y = g["actual_qty"].values.astype(float)
        yhat = g["predicted_quantity"].values.astype(float)

        err = yhat - y
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err) / (y + eps)))
        bias = float(np.mean(err))

        # drift score: compare last 30d error distribution vs previous
        g = g.sort_values("ds")
        n = len(g)
        if n < 20:
            drift = min(1.0, mape)  # fallback
        else:
            cut = max(1, int(n * 0.5))
            e1 = (g.iloc[:cut]["predicted_quantity"] - g.iloc[:cut]["actual_qty"]).abs().mean()
            e2 = (g.iloc[cut:]["predicted_quantity"] - g.iloc[cut:]["actual_qty"]).abs().mean()
            drift = float(min(1.0, abs(e2 - e1) / (e1 + eps)))

        notes = f"MAE={mae:.3f}, MAPE={mape:.3f}, bias={bias:.3f}, drift={drift:.3f}"

        rows.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(pid),
            "lookback_days": int(lookback_days),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "bias": round(bias, 4),
            "drift_score": round(drift, 4),
            "drift_level": _drift_level(drift),
            "notes": notes[:480],
            "model_version": MODEL_VERSION
        })

    return rows
