import math
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _level(p):
    if p >= 0.75: return "HIGH"
    if p >= 0.40: return "MEDIUM"
    return "LOW"


def compute_stockout_risk(company_id: int, warehouse_id: int, horizon_days: int = 14, lookback_days: int = 60):
    # On-hand
    q_onhand = text("""
                    SELECT product_id, COALESCE(SUM(quantity), 0) AS on_hand
                    FROM stock_movements
                    WHERE approved = true
                      AND company_id = :c
                      AND warehouse_id = :w
                    GROUP BY product_id
                    """)
    onh = pd.read_sql(q_onhand, engine, params={"c": company_id, "w": warehouse_id})
    if onh.empty:
        return []

    # Daily sales out (type 2)
    q_sales = text("""
                   SELECT product_id, DATE(date_received) AS ds, SUM(ABS(quantity)) AS qty
                   FROM stock_movements
                   WHERE approved = true
                     AND company_id = :c
                     AND warehouse_id = :w
                     AND movement_type_id = 2
                     AND date_received >= NOW() - (:lb || ' days')::interval
                   GROUP BY product_id, DATE(date_received)
                   """)
    s = pd.read_sql(q_sales, engine, params={"c": company_id, "w": warehouse_id, "lb": int(lookback_days)})
    if s.empty:
        return []

    vel = s.groupby("product_id")["qty"].agg(["mean", "std"]).reset_index().rename(
        columns={"mean": "avg", "std": "std"})
    vel["std"] = vel["std"].fillna(0.0)

    df = onh.merge(vel, on="product_id", how="left").fillna({"avg": 0.0, "std": 0.0})
    rows = []
    today = datetime.utcnow().date()

    for _, r in df.iterrows():
        on_hand = float(r["on_hand"])
        mu = float(r["avg"])
        sd = float(r["std"])
        if mu <= 0:
            continue

        expected = mu * horizon_days
        # Approximate distribution of sum over horizon: Normal(mu*h, sd*sqrt(h))
        sigma = max(1e-6, sd * math.sqrt(horizon_days))
        z = (on_hand - expected) / sigma
        p_stockout = float(1.0 - _normal_cdf(z))
        p_stockout = max(0.0, min(1.0, p_stockout))

        # expected stockout date (rough)
        days_cover = on_hand / max(mu, 1e-6)
        est_date = today + timedelta(days=int(days_cover)) if days_cover < 3650 else None

        recommended = max(0.0, expected - on_hand)

        rows.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(r["product_id"]),
            "horizon_days": int(horizon_days),
            "lookback_days": int(lookback_days),
            "on_hand": round(on_hand, 3),
            "avg_daily_demand": round(mu, 3),
            "std_daily_demand": round(sd, 3),
            "expected_demand": round(expected, 3),
            "stockout_probability": round(p_stockout, 4),
            "expected_stockout_date": est_date,
            "recommended_qty": round(recommended, 3),
            "risk_level": _level(p_stockout),
            "model_version": MODEL_VERSION
        })

    return rows
