import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine


def calculate_sales_performance(company_id, warehouse_id):
    """
    Analyze each salesperson's sales and compute performance metrics.
    """
    query = text("""
                 SELECT s.company_id,
                        s.warehouse_id,
                        s.sales_person_id                AS salesperson_id,
                        DATE_TRUNC('month', s.date)      AS month,
                        SUM(sl.quantity * sl.unit_price) AS total_value,
                        COUNT(DISTINCT s.id)             AS orders
                 FROM sales s
                          JOIN sale_lines sl ON s.id = sl.sale_id
                 WHERE s.company_id = :company_id
                   AND s.warehouse_id = :warehouse_id
                   AND s.status IN ('PAID', 'DELIVERED')
                   AND s.sales_person_id IS NOT NULL
                   AND s.date >= CURRENT_DATE - INTERVAL '6 months'
                 GROUP BY s.company_id, s.warehouse_id, s.sales_person_id, DATE_TRUNC('month', s.date)
                 ORDER BY month;
                 """)

    df = pd.read_sql(query, engine, params={
        "company_id": company_id,
        "warehouse_id": warehouse_id
    })

    if df.empty:
        return []

    results = []
    for sp_id, group in df.groupby("salesperson_id"):
        group = group.sort_values("month")
        total_sales = group["total_value"].sum()
        total_orders = group["orders"].sum()
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0

        # Growth rate over last two months
        if len(group) >= 2:
            growth_rate = ((group["total_value"].iloc[-1] - group["total_value"].iloc[-2]) /
                           max(group["total_value"].iloc[-2], 1)) * 100
        else:
            growth_rate = 0

        # Trend detection (simple slope)
        months = np.arange(len(group))
        sales_values = group["total_value"].values
        slope, _ = np.polyfit(months, sales_values, 1)
        trend = "Rising" if slope > 0 else "Falling" if slope < 0 else "Stable"

        # Performance score (weighted formula)
        perf_score = (
                             (0.5 * (total_sales / total_sales.max() if total_sales.max() != 0 else 0)) +
                             (0.3 * (growth_rate / 100)) +
                             (0.2 * (avg_order_value / avg_order_value.max() if avg_order_value.max() != 0 else 0))
                     ) * 100
        perf_score = round(max(0, min(perf_score, 100)), 2)

        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "salesperson_id": int(sp_id),
            "total_sales": float(total_sales),
            "total_orders": int(total_orders),
            "avg_order_value": float(avg_order_value),
            "growth_rate": float(round(growth_rate, 2)),
            "performance_trend": trend,
            "performance_score": float(perf_score)
        })

    return results
