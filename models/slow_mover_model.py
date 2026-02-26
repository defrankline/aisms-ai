from datetime import datetime

import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _level(score):
    if score >= 0.75: return "HIGH"
    if score >= 0.45: return "MEDIUM"
    return "LOW"


def compute_slow_movers(company_id: int, warehouse_id: int, lookback_days: int = 120, min_on_hand: float = 1.0):
    q = text("""
             WITH onhand AS (SELECT product_id, COALESCE(SUM(quantity), 0) AS on_hand
                             FROM stock_movements
                             WHERE approved = true
                               AND company_id = :c
                               AND warehouse_id = :w
                             GROUP BY product_id),
                  sales AS (SELECT product_id,
                                   AVG(qty_out) AS avg_daily_sales,
                                   MAX(ds)      AS last_sale_date
                            FROM (SELECT product_id, DATE(date_received) AS ds, SUM(ABS(quantity)) AS qty_out
                                  FROM stock_movements
                                  WHERE approved = true
                                    AND company_id = :c
                                    AND warehouse_id = :w
                                    AND movement_type_id = 2
                                    AND date_received >= NOW() - (:lb || ' days')::interval
                                  GROUP BY product_id, DATE(date_received)) t
                            GROUP BY product_id)
             SELECT o.product_id,
                    o.on_hand,
                    COALESCE(s.avg_daily_sales, 0) AS avg_daily_sales,
                    s.last_sale_date
             FROM onhand o
                      LEFT JOIN sales s ON s.product_id = o.product_id
             WHERE o.on_hand >= :min_on_hand
             """)
    df = pd.read_sql(q, engine, params={"c": company_id, "w": warehouse_id, "lb": int(lookback_days),
                                        "min_on_hand": float(min_on_hand)})
    if df.empty:
        return []

    today = datetime.utcnow().date()
    rows = []

    for _, r in df.iterrows():
        on_hand = float(r["on_hand"])
        avg_sales = float(r["avg_daily_sales"])
        last_sale = r["last_sale_date"]
        days_since = 9999
        if pd.notna(last_sale):
            days_since = (today - last_sale).days

        days_cover = on_hand / max(avg_sales, 1e-6) if avg_sales > 0 else 9999.0

        # score: older last sale + huge cover + low velocity
        score = 0.0
        score += min(1.0, days_since / 90.0) * 0.45
        score += min(1.0, days_cover / 180.0) * 0.45
        score += (1.0 if avg_sales == 0 else 0.0) * 0.10
        score = max(0.0, min(1.0, score))

        if score >= 0.75:
            action = "DISCOUNT or TRANSFER to higher-velocity store"
        elif score >= 0.45:
            action = "Monitor; consider small markdown / bundle"
        else:
            action = "OK"

        rationale = f"on_hand={on_hand:.2f}, avg_daily_sales={avg_sales:.3f}, days_since_last_sale={days_since}, days_cover={days_cover:.1f}"

        rows.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(r["product_id"]),
            "lookback_days": int(lookback_days),
            "on_hand": round(on_hand, 3),
            "avg_daily_sales": round(avg_sales, 3),
            "days_since_last_sale": int(days_since),
            "days_cover": round(days_cover, 3),
            "slow_mover_score": round(score, 4),
            "risk_level": _level(score),
            "recommended_action": action,
            "rationale": rationale[:480],
            "model_version": MODEL_VERSION
        })

    return rows
