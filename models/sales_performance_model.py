from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _calc_growth(cur: float, prev: float) -> float:
    if prev == 0:
        return 100.0 if cur > 0 else 0.0
    return ((cur - prev) / prev) * 100.0


def _trend_from_growth(g: float) -> str:
    if g > 5:
        return "UP"
    if g < -5:
        return "DOWN"
    return "FLAT"


def score_salespersons(company_id: int, warehouse_id: int | None, start_date: str, end_date: str):
    """
    Scores sales persons for a period and compares to previous equal period.
    """
    wh_filter = "AND s.warehouse_id = :warehouse_id" if warehouse_id is not None else ""

    q = text(f"""
        SELECT s.sales_person_id,
               s.id AS sale_id,
               DATE(s.date) AS day,
               SUM(sl.quantity * sl.unit_price) AS amount
        FROM sales s
        JOIN sale_lines sl ON sl.sale_id = s.id
        WHERE s.company_id = :company_id
          AND s.status IN ('PAID','DELIVERED')
          AND s.date BETWEEN :start_date AND :end_date
          {wh_filter}
          AND s.sales_person_id IS NOT NULL
        GROUP BY s.sales_person_id, s.id, DATE(s.date)
    """)
    params = {"company_id": company_id, "start_date": start_date, "end_date": end_date}
    if warehouse_id is not None:
        params["warehouse_id"] = warehouse_id

    cur_df = pd.read_sql(q, engine, params=params)
    if cur_df.empty:
        return []

    # previous window
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end_dt - start_dt).days + 1
    prev_start = (start_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    prev_end = (start_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    params_prev = {"company_id": company_id, "start_date": prev_start, "end_date": prev_end}
    if warehouse_id is not None:
        params_prev["warehouse_id"] = warehouse_id
    prev_df = pd.read_sql(q, engine, params=params_prev)

    # Aggregate current window
    cur_agg = (
        cur_df.groupby("sales_person_id")
        .agg(total_sales=("amount", "sum"),
             total_orders=("sale_id", "nunique"))
        .reset_index()
    )
    cur_agg["avg_order_value"] = cur_agg["total_sales"] / cur_agg["total_orders"]

    # Aggregate previous window
    if prev_df.empty:
        prev_agg = pd.DataFrame({"sales_person_id": cur_agg["sales_person_id"], "total_sales_prev": 0.0})
    else:
        prev_agg = (
            prev_df.groupby("sales_person_id")
            .agg(total_sales_prev=("amount", "sum"))
            .reset_index()
        )

    merged = cur_agg.merge(prev_agg, on="sales_person_id", how="left").fillna({"total_sales_prev": 0.0})
    merged["growth_rate"] = merged.apply(lambda r: _calc_growth(float(r["total_sales"]), float(r["total_sales_prev"])),
                                         axis=1)
    merged["performance_trend"] = merged["growth_rate"].apply(_trend_from_growth)

    # Composite performance score (0..100)
    # normalize by maxima; growth contributes positively but capped
    sales_norm = (merged["total_sales"] / merged["total_sales"].max()).fillna(0)
    aov_norm = (merged["avg_order_value"] / merged["avg_order_value"].max()).fillna(0)
    growth_norm = (merged["growth_rate"].clip(lower=-50, upper=50) + 50) / 100.0

    merged["performance_score"] = (0.55 * sales_norm + 0.20 * aov_norm + 0.25 * growth_norm) * 100.0
    merged["performance_score"] = merged["performance_score"].round(2)

    results = []
    for _, r in merged.iterrows():
        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id if warehouse_id is not None else 0,
            "salesperson_id": int(r["sales_person_id"]),
            "total_sales": float(r["total_sales"]),
            "total_orders": int(r["total_orders"]),
            "avg_order_value": float(r["avg_order_value"]),
            "growth_rate": float(r["growth_rate"]),
            "performance_trend": str(r["performance_trend"]),
            "performance_score": float(r["performance_score"]),
            "model_version": MODEL_VERSION
        })
    return results
