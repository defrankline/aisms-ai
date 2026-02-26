import json

import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _risk_level(score01: float) -> str:
    if score01 >= 0.75:
        return "HIGH"
    if score01 >= 0.45:
        return "MEDIUM"
    return "LOW"


def compute_stocktake_variance_risk(company_id: int, warehouse_id: int, lookback_days: int = 180):
    """
    Produces per-product variance risk using stock_movements behavior.

    Returns rows compatible with StocktakeVarianceRisk insert:
      company_id, warehouse_id, product_id, lookback_days,
      risk_score, risk_level, expected_abs_variance, drivers, model_version
    """

    q = text("""
             SELECT sm.product_id,
                    SUM(CASE WHEN sm.movement_type_id = 1 THEN sm.quantity ELSE 0 END)           AS receipt_in,
                    SUM(CASE WHEN sm.movement_type_id = 2 THEN ABS(sm.quantity) ELSE 0 END)      AS sale_out,
                    SUM(CASE WHEN sm.movement_type_id = 3 THEN sm.quantity ELSE 0 END)           AS return_in,
                    SUM(CASE WHEN sm.movement_type_id = 4 THEN sm.quantity ELSE 0 END)           AS adjust_in,
                    SUM(CASE WHEN sm.movement_type_id = 5 THEN ABS(sm.quantity) ELSE 0 END)      AS adjust_out,
                    SUM(CASE WHEN sm.movement_type_id = 6 THEN sm.quantity ELSE 0 END)           AS transfer_in,
                    SUM(CASE WHEN sm.movement_type_id = 7 THEN ABS(sm.quantity) ELSE 0 END)      AS transfer_out,
                    COUNT(*)                                                                     AS movement_count,
                    COUNT(DISTINCT sm.staff_id)                                                  AS staff_count,
                    AVG(CASE WHEN sm.movement_type_id IN (1, 4, 5) THEN sm.unit_cost END)        AS avg_sig_cost,
                    STDDEV_POP(CASE WHEN sm.movement_type_id IN (1, 4, 5) THEN sm.unit_cost END) AS std_sig_cost
             FROM stock_movements sm
             WHERE sm.approved = TRUE
               AND sm.company_id = :company_id
               AND sm.warehouse_id = :warehouse_id
               AND sm.date >= CURRENT_DATE - (:lb || ' days')::interval
             GROUP BY sm.product_id
             """)

    df = pd.read_sql(q, engine, params={
        "company_id": company_id,
        "warehouse_id": warehouse_id,
        "lb": int(lookback_days),
    })

    if df.empty:
        return []

    df = df.fillna(0.0)

    eps = 1e-6
    df["adj_out_ratio"] = df["adjust_out"] / (df["sale_out"] + eps)
    df["return_ratio"] = df["return_in"] / (df["sale_out"] + eps)
    df["transfer_ratio"] = (df["transfer_in"] + df["transfer_out"]) / (df["sale_out"] + eps)
    df["cost_volatility"] = df["std_sig_cost"] / (df["avg_sig_cost"] + eps)

    # Normalize by p95 caps to get stable 0..1 scoring
    def n01(series):
        cap = float(series.quantile(0.95)) if len(series) > 5 else float(series.max())
        cap = max(cap, eps)
        return (series / cap).clip(0, 1)

    s_adj = n01(df["adj_out_ratio"])
    s_ret = n01(df["return_ratio"])
    s_trf = n01(df["transfer_ratio"])
    s_mov = n01(df["movement_count"] / 50.0)
    s_staff = n01(df["staff_count"] / 10.0)
    s_cost = n01(df["cost_volatility"])

    # Weighted risk score (adjust_out dominates)
    df["risk_score"] = (
                0.45 * s_adj + 0.20 * s_ret + 0.15 * s_trf + 0.10 * s_mov + 0.05 * s_staff + 0.05 * s_cost).clip(0, 1)

    rows = []
    for _, r in df.iterrows():
        score = float(r["risk_score"])
        drivers = {
            "adj_out_ratio": round(float(r["adj_out_ratio"]), 4),
            "return_ratio": round(float(r["return_ratio"]), 4),
            "transfer_ratio": round(float(r["transfer_ratio"]), 4),
            "movement_count": int(r["movement_count"]),
            "staff_count": int(r["staff_count"]),
            "cost_volatility": round(float(r["cost_volatility"]), 6),
        }

        rows.append({
            "company_id": int(company_id),
            "warehouse_id": int(warehouse_id),
            "product_id": int(r["product_id"]),
            "lookback_days": int(lookback_days),
            "risk_score": round(score, 4),
            "risk_level": _risk_level(score),
            "expected_abs_variance": None,  # optional later if you build supervised regression
            "drivers": json.dumps(drivers),
            "model_version": MODEL_VERSION
        })

    return rows
