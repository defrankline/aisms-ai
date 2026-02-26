import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _level(score):
    if score >= 0.75: return "HIGH"
    if score >= 0.45: return "MEDIUM"
    return "LOW"


def compute_return_pressure(company_id: int, warehouse_id: int, period_days: int = 60):
    q = text("""
             SELECT product_id,
                    SUM(CASE WHEN movement_type_id = 2 THEN ABS(quantity) ELSE 0 END) AS sale_out_qty,
                    SUM(CASE WHEN movement_type_id = 3 THEN quantity ELSE 0 END)      AS return_in_qty
             FROM stock_movements
             WHERE approved = true
               AND company_id = :c
               AND warehouse_id = :w
               AND date >= CURRENT_DATE - (:pd || ' days')::interval
             GROUP BY product_id
             """)
    df = pd.read_sql(q, engine, params={"c": company_id, "w": warehouse_id, "pd": int(period_days)})
    if df.empty:
        return []

    eps = 1e-6
    df["return_rate"] = df["return_in_qty"] / (df["sale_out_qty"] + eps)

    # trend: compare last half vs first half
    half = max(1, int(period_days / 2))
    q2 = text("""
              SELECT product_id,
                     SUM(CASE WHEN movement_type_id = 2 THEN ABS(quantity) ELSE 0 END) AS sale_out_qty,
                     SUM(CASE WHEN movement_type_id = 3 THEN quantity ELSE 0 END)      AS return_in_qty
              FROM stock_movements
              WHERE approved = true
                AND company_id = :c
                AND warehouse_id = :w
                AND date >= CURRENT_DATE - (:pd || ' days')::interval
                AND date < CURRENT_DATE - (:half || ' days')::interval
              GROUP BY product_id
              """)
    early = pd.read_sql(q2, engine,
                        params={"c": company_id, "w": warehouse_id, "pd": int(period_days), "half": int(half)})
    early["return_rate_early"] = early["return_in_qty"] / (early["sale_out_qty"] + eps)

    q3 = text("""
              SELECT product_id,
                     SUM(CASE WHEN movement_type_id = 2 THEN ABS(quantity) ELSE 0 END) AS sale_out_qty,
                     SUM(CASE WHEN movement_type_id = 3 THEN quantity ELSE 0 END)      AS return_in_qty
              FROM stock_movements
              WHERE approved = true
                AND company_id = :c
                AND warehouse_id = :w
                AND date >= CURRENT_DATE - (:half || ' days')::interval
              GROUP BY product_id
              """)
    late = pd.read_sql(q3, engine, params={"c": company_id, "w": warehouse_id, "half": int(half)})
    late["return_rate_late"] = late["return_in_qty"] / (late["sale_out_qty"] + eps)

    df = df.merge(early[["product_id", "return_rate_early"]], on="product_id", how="left") \
        .merge(late[["product_id", "return_rate_late"]], on="product_id", how="left") \
        .fillna(0.0)

    # normalize score by p95
    cap = float(df["return_rate"].quantile(0.95)) if len(df) > 5 else float(df["return_rate"].max())
    cap = max(cap, eps)
    df["score"] = (df["return_rate"] / cap).clip(0, 1)

    rows = []
    for _, r in df.iterrows():
        trend = "STABLE"
        if float(r["return_rate_late"]) > float(r["return_rate_early"]) * 1.15:
            trend = "UP"
        elif float(r["return_rate_late"]) < float(r["return_rate_early"]) * 0.85:
            trend = "DOWN"

        s = float(r["score"])
        rows.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(r["product_id"]),
            "period_days": int(period_days),
            "sale_out_qty": round(float(r["sale_out_qty"]), 3),
            "return_in_qty": round(float(r["return_in_qty"]), 3),
            "return_rate": round(float(r["return_rate"]), 4),
            "score": round(s, 4),
            "level": _level(s),
            "trend": trend,
            "model_version": MODEL_VERSION
        })

    return rows
