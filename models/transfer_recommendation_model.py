import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def recommend_transfers(company_id: int, days_window: int = 60, horizon_days: int = 14, min_suggest_qty: float = 1.0):
    """
    Recommend transfers between warehouses based on:
      - On-hand (SUM(quantity))
      - Demand velocity from SALE_OUT movements (movement_type_id=2)
      - Days cover = on_hand / avg_daily_sale_out

    Returns rows compatible with TransferRecommendation insert.
    """

    # On-hand per product+warehouse
    q_onhand = text("""
                    SELECT warehouse_id, product_id, COALESCE(SUM(quantity), 0) AS on_hand
                    FROM stock_movements
                    WHERE approved = TRUE
                      AND company_id = :company_id
                    GROUP BY warehouse_id, product_id
                    HAVING COALESCE(SUM(quantity), 0) > 0
                    """)
    onh = pd.read_sql(q_onhand, engine, params={"company_id": company_id})
    if onh.empty:
        return []

    onh["on_hand"] = pd.to_numeric(onh["on_hand"], errors="coerce").fillna(0.0)

    # Avg daily demand from SALE_OUT in last N days
    q_sales_daily = text("""
                         SELECT warehouse_id,
                                product_id,
                                DATE(date_received) AS ds,
                                SUM(ABS(quantity))  AS qty_out
                         FROM stock_movements
                         WHERE approved = TRUE
                           AND company_id = :company_id
                           AND movement_type_id = 2
                           AND date_received >= NOW() - (:dw || ' days')::interval
                         GROUP BY warehouse_id, product_id, DATE(date_received)
                         """)
    daily = pd.read_sql(q_sales_daily, engine, params={"company_id": company_id, "dw": int(days_window)})
    if daily.empty:
        return []

    daily["qty_out"] = pd.to_numeric(daily["qty_out"], errors="coerce").fillna(0.0)

    vel = daily.groupby(["warehouse_id", "product_id"])["qty_out"].mean().reset_index().rename(
        columns={"qty_out": "avg_daily_out"})
    vel["avg_daily_out"] = vel["avg_daily_out"].fillna(0.0)

    df = onh.merge(vel, on=["warehouse_id", "product_id"], how="left").fillna({"avg_daily_out": 0.0})

    # Days cover (if no demand signal => treat as very high cover)
    df["days_cover"] = df.apply(
        lambda r: float(r["on_hand"]) / float(r["avg_daily_out"]) if float(r["avg_daily_out"]) > 0 else 9999.0,
        axis=1
    )

    min_qty = float(min_suggest_qty)
    horizon = float(horizon_days)
    results = []

    for pid, g in df.groupby("product_id"):
        if len(g) < 2:
            continue

        # Need at least one warehouse with demand
        if (g["avg_daily_out"] > 0).sum() == 0:
            continue

        g = g.sort_values("days_cover", ascending=True)
        low = g.iloc[0]  # lowest cover
        high = g.iloc[-1]  # highest cover

        # Only recommend if low is below horizon, and high has meaningful excess
        if float(low["days_cover"]) >= horizon:
            continue
        if float(high["days_cover"]) <= horizon * 1.5:
            continue

        # shortage at low warehouse to reach horizon cover
        need = max(0.0, (horizon * float(low["avg_daily_out"])) - float(low["on_hand"]))

        # excess at high warehouse after keeping horizon cover
        keep = horizon * float(high["avg_daily_out"])
        excess = max(0.0, float(high["on_hand"]) - keep)

        suggested = min(need, excess)
        if suggested < min_qty:
            continue

        # confidence
        conf = 60.0
        if float(low["avg_daily_out"]) > 0 and float(high["avg_daily_out"]) > 0:
            conf = min(95.0, 60.0 + 10.0 * min(3.0, float(high["days_cover"]) / max(1.0, horizon)))
        else:
            conf = 55.0

        results.append({
            "company_id": int(company_id),
            "product_id": int(pid),
            "from_warehouse_id": int(high["warehouse_id"]),
            "to_warehouse_id": int(low["warehouse_id"]),
            "suggested_qty": round(float(suggested), 3),
            "from_on_hand": round(float(high["on_hand"]), 3),
            "to_on_hand": round(float(low["on_hand"]), 3),
            "from_days_cover": round(float(high["days_cover"]), 3),
            "to_days_cover": round(float(low["days_cover"]), 3),
            "rationale": f"Target {horizon_days}d cover. Low-cover store needs {need:.2f}, high-cover store has excess {excess:.2f}.",
            "confidence": round(float(conf), 2),
            "model_version": MODEL_VERSION
        })

    return results
