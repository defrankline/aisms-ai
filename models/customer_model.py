from datetime import datetime

import pandas as pd
from sqlalchemy import text

from db.connection import engine


def calculate_customer_segments(company_id, days_window=365):
    """
    Compute Recency, Frequency, Monetary (RFM) and segment customers.
    """
    query = text("""
                 SELECT s.company_id,
                        s.id                             AS sale_id,
                        s.date::date                     AS sale_date,
                        s.cashier_id                     AS customer_id,
                        SUM(sl.quantity * sl.unit_price) AS total_value
                 FROM sales s
                          JOIN sale_lines sl ON s.id = sl.sale_id
                 WHERE s.company_id = :company_id
                   AND s.status IN ('PAID', 'DELIVERED')
                   AND s.date >= CURRENT_DATE - INTERVAL ':days_window days'
                 GROUP BY s.company_id, s.id, s.date, s.cashier_id
                 """)

    df = pd.read_sql(query, engine, params={
        "company_id": company_id,
        "days_window": days_window
    })

    if df.empty:
        return []

    today = pd.Timestamp(datetime.utcnow().date())

    # Aggregate per customer
    agg = df.groupby("customer_id").agg({
        "sale_date": lambda x: (today - pd.to_datetime(x.max())).days,
        "sale_id": "count",
        "total_value": "sum"
    }).reset_index()

    agg.rename(columns={
        "sale_date": "recency_days",
        "sale_id": "frequency",
        "total_value": "monetary_value"
    }, inplace=True)

    # Normalize and compute CLV score (0–100)
    agg["r_score"] = pd.qcut(agg["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    agg["f_score"] = pd.qcut(agg["frequency"].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    agg["m_score"] = pd.qcut(agg["monetary_value"].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    agg["clv_score"] = (agg["r_score"] + agg["f_score"] + agg["m_score"]) / 3 * 20  # 0–100 scale

    # Segment labeling
    def classify(row):
        if row["clv_score"] >= 80:
            return "Champions"
        elif row["clv_score"] >= 60:
            return "Loyal"
        elif row["clv_score"] >= 40:
            return "At Risk"
        elif row["clv_score"] >= 20:
            return "Churn Risk"
        else:
            return "New Customers"

    agg["segment"] = agg.apply(classify, axis=1)

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
            "segment": row["segment"]
        })

    return results
