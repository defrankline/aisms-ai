import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _level(pct, vol):
    if abs(pct) >= 0.20 or vol >= 0.25: return "ALERT"
    if abs(pct) >= 0.10 or vol >= 0.15: return "WARN"
    return "INFO"


def compute_receipt_cost_alerts(company_id: int, warehouse_id: int, lookback_days: int = 180,
                                short_window_days: int = 7, long_window_days: int = 30):
    q = text("""
             WITH receipts AS (SELECT product_id, date_received::date AS ds, unit_cost
                               FROM stock_movements
                               WHERE approved = true
                                 AND company_id = :c
                                 AND warehouse_id = :w
                                 AND movement_type_id = 1
                                 AND unit_cost IS NOT NULL
                                 AND date_received >= NOW() - (:lb || ' days')::interval)
             SELECT product_id,
                    AVG(CASE WHEN ds >= CURRENT_DATE - (:sw || ' days')::interval THEN unit_cost END) AS avg_cost_short,
                    AVG(CASE WHEN ds >= CURRENT_DATE - (:lw || ' days')::interval THEN unit_cost END) AS avg_cost_long,
                    STDDEV_POP(unit_cost)                                                             AS std_cost,
                    AVG(unit_cost)                                                                    AS mean_cost
             FROM receipts
             GROUP BY product_id
             HAVING AVG(CASE WHEN ds >= CURRENT_DATE - (:lw || ' days')::interval THEN unit_cost END) IS NOT NULL
             """)
    df = pd.read_sql(q, engine, params={"c": company_id, "w": warehouse_id, "lb": int(lookback_days),
                                        "sw": int(short_window_days), "lw": int(long_window_days)})
    if df.empty:
        return []

    rows = []
    for _, r in df.iterrows():
        short = float(r["avg_cost_short"]) if r["avg_cost_short"] is not None else 0.0
        long = float(r["avg_cost_long"]) if r["avg_cost_long"] is not None else 0.0
        mean = float(r["mean_cost"]) if r["mean_cost"] is not None else 0.0
        std = float(r["std_cost"]) if r["std_cost"] is not None else 0.0

        pct = 0.0
        if long > 0:
            pct = (short - long) / long

        vol = (std / mean) if mean > 0 else 0.0
        level = _level(pct, vol)

        msg = f"Receipt unit_cost short={short:.4f}, long={long:.4f}, change={pct * 100:.1f}%, volatility={vol * 100:.1f}%"

        rows.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(r["product_id"]),
            "lookback_days": int(lookback_days),
            "short_window_days": int(short_window_days),
            "long_window_days": int(long_window_days),
            "avg_cost_short": round(short, 4),
            "avg_cost_long": round(long, 4),
            "cost_change_pct": round(pct, 4),
            "volatility_lookback": round(vol, 4),
            "level": level,
            "message": msg[:480],
            "model_version": MODEL_VERSION
        })

    return rows
