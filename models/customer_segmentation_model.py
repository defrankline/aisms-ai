import re
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION

# only allow safe SQL identifiers like: customer_id, client_id, member_id
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str) -> str:
    if not identifier or not _SAFE_IDENTIFIER.match(identifier):
        raise ValueError(f"Invalid customer identifier column: {identifier}")
    return identifier


def calculate_customer_segments(
        company_id: int,
        customer_column: str,
        days_window: int = 365,
        warehouse_id: int | None = None
):
    """
    Build RFM + CLV segmentation for a company (optionally filtered by warehouse).
    You MUST provide the name of the column in `sales` that represents the customer,
    e.g. 'customer_id'. We do NOT assume which column is your customer.
    """

    cust_col = _validate_identifier(customer_column)

    # Build dynamic filter for warehouse when provided
    wh_filter = "AND s.warehouse_id = :warehouse_id" if warehouse_id is not None else ""

    # Get last N days of order totals per sale + customer
    # We aggregate sale_lines to get monetary value per sale (order).
    query = text(f"""
        SELECT
            s.company_id,
            s.id AS sale_id,
            s.date::date AS sale_date,
            s.{cust_col} AS customer_id,
            SUM(sl.quantity * sl.unit_price) AS order_total
        FROM sales s
        JOIN sale_lines sl ON s.id = sl.sale_id
        WHERE s.company_id = :company_id
          AND s.status IN ('PAID','DELIVERED')
          {wh_filter}
          AND s.date >= CURRENT_DATE - (:days_window || ' days')::interval
          AND s.{cust_col} IS NOT NULL
        GROUP BY s.company_id, s.id, s.date, s.{cust_col}
        ORDER BY sale_date
    """)

    params = {
        "company_id": company_id,
        "days_window": days_window
    }
    if warehouse_id is not None:
        params["warehouse_id"] = warehouse_id

    df = pd.read_sql(query, engine, params=params)

    if df.empty:
        return []

    # RFM:
    today = pd.Timestamp(datetime.utcnow().date())

    # Aggregate per customer
    agg = df.groupby("customer_id").agg(
        last_purchase=("sale_date", "max"),
        frequency=("sale_id", "nunique"),
        monetary_value=("order_total", "sum")
    ).reset_index()

    agg["recency_days"] = (today - pd.to_datetime(agg["last_purchase"])).dt.days.astype(int)

    # Quantile scores 1..5 (higher is better)
    # Recency is inverted: lower days => better score
    # Use rank(method='first') to avoid binning ties issues
    try:
        agg["r_score"] = pd.qcut(agg["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    except ValueError:
        # not enough unique values — fallback to median split
        median_r = agg["recency_days"].median()
        agg["r_score"] = np.where(agg["recency_days"] <= median_r, 5, 3)

    try:
        agg["f_score"] = pd.qcut(agg["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:
        median_f = agg["frequency"].median()
        agg["f_score"] = np.where(agg["frequency"] <= median_f, 2, 4)

    try:
        agg["m_score"] = pd.qcut(agg["monetary_value"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:
        median_m = agg["monetary_value"].median()
        agg["m_score"] = np.where(agg["monetary_value"] <= median_m, 2, 4)

    # CLV score (0–100) as weighted average of R, F, M
    agg["clv_score"] = ((agg["r_score"] + agg["f_score"] + agg["m_score"]) / 15.0) * 100.0
    agg["clv_score"] = agg["clv_score"].clip(lower=0, upper=100).round(2)

    # Segment labeling
    def _segment(row):
        score = row["clv_score"]
        if score >= 80:
            return "Champions"
        elif score >= 60:
            return "Loyal"
        elif score >= 40:
            return "At Risk"
        elif score >= 20:
            return "Churn Risk"
        else:
            return "New"

    agg["segment"] = agg.apply(_segment, axis=1)

    # Build results for insertion
    results = []
    for _, row in agg.iterrows():
        results.append({
            "company_id": company_id,
            "customer_id": int(row["customer_id"]),
            "recency_days": int(row["recency_days"]),
            "frequency": int(row["frequency"]),
            "monetary_value": float(row["monetary_value"]),
            "clv_score": float(row["clv_score"]),
            "segment": row["segment"],
            "model_version": MODEL_VERSION
        })

    return results
