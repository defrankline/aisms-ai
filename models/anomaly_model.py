import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine


def detect_sales_anomalies(company_id, warehouse_id):
    """
    Detect unusual sales totals for a specific warehouse within a company.
    """
    query = text("""
                 SELECT s.id                             AS sale_id,
                        s.company_id,
                        s.warehouse_id,
                        SUM(sl.quantity * sl.unit_price) AS total_amount
                 FROM sales s
                          JOIN sale_lines sl ON s.id = sl.sale_id
                 WHERE s.company_id = :company_id
                   AND s.warehouse_id = :warehouse_id
                   AND s.status IN ('PAID', 'DELIVERED')
                 GROUP BY s.id, s.company_id, s.warehouse_id
                 """)

    df = pd.read_sql(query, engine, params={
        "company_id": company_id,
        "warehouse_id": warehouse_id
    })

    if df.empty:
        return []

    mean = df['total_amount'].mean()
    std = df['total_amount'].std()
    if std == 0:
        return []

    df['zscore'] = (df['total_amount'] - mean) / std
    df['abs_zscore'] = df['zscore'].abs()
    df['level'] = pd.cut(
        df['abs_zscore'],
        bins=[0, 2, 3, np.inf],
        labels=['INFO', 'WARN', 'ALERT']
    )

    anomalies = df[df['abs_zscore'] >= 2].copy()
    results = []
    for _, row in anomalies.iterrows():
        results.append({
            "sale_id": int(row["sale_id"]),
            "company_id": int(row["company_id"]),
            "warehouse_id": int(row["warehouse_id"]),
            "policy_code": "ZSCORE",
            "score": float(row["abs_zscore"]),
            "level": row["level"]
        })
    return results
