import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine


def generate_pricing_recommendations(company_id, warehouse_id):
    """
    Analyze recent sales trends to suggest optimal product prices.
    """
    query = text("""
                 SELECT sl.product_id,
                        sl.unit_price,
                        SUM(sl.quantity)   AS qty_sold,
                        AVG(sl.unit_price) AS avg_price
                 FROM sales s
                          JOIN sale_lines sl ON s.id = sl.sale_id
                 WHERE s.company_id = :company_id
                   AND s.warehouse_id = :warehouse_id
                   AND s.status IN ('PAID', 'DELIVERED', 'CONFIRMED')
                   AND s.date >= CURRENT_DATE - INTERVAL '60 days'
                 GROUP BY sl.product_id, sl.unit_price
                 ORDER BY sl.product_id, sl.unit_price;
                 """)

    df = pd.read_sql(query, engine,
                     params={"company_id": company_id, "warehouse_id": warehouse_id})
    if df.empty:
        return []

    results = []
    for pid, group in df.groupby("product_id"):
        if len(group) < 3:
            continue

        # Fit a simple demand curve (log-log regression)
        group = group[group["qty_sold"] > 0]
        X = np.log(group["unit_price"])
        y = np.log(group["qty_sold"])
        slope, intercept = np.polyfit(X, y, 1)
        elasticity = slope  # typically negative

        avg_price = group["unit_price"].mean()
        current_price = avg_price
        # Basic recommendation: if elasticity < -1, demand is elastic -> lower price slightly
        # if elasticity > -0.5, demand is inelastic -> can increase price slightly
        if elasticity < -1:
            suggested_price = current_price * 0.97
        elif elasticity > -0.5:
            suggested_price = current_price * 1.03
        else:
            suggested_price = current_price

        price_change_pct = ((suggested_price - current_price) / current_price) * 100
        expected_demand_change = (-elasticity) * (abs(price_change_pct) / 100) * 100
        confidence = min(100, 100 - abs(elasticity) * 20)

        rationale = (
            "High price sensitivity, reduce slightly"
            if elasticity < -1
            else "Low price sensitivity, increase slightly"
            if elasticity > -0.5
            else "Stable demand, maintain current price"
        )

        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(pid),
            "current_price": round(current_price, 2),
            "suggested_price": round(suggested_price, 2),
            "price_change_pct": round(price_change_pct, 2),
            "expected_demand_change": round(expected_demand_change, 2),
            "confidence_level": round(confidence, 2),
            "rationale": rationale
        })

    return results
