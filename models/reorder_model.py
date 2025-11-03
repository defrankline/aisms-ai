import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine

# z-score for 95 % service level
Z95 = 1.65


def generate_reorder_suggestions(company_id, warehouse_id):
    """
    Combine forecast + stock data to compute reorder points.
    """
    # Forecasted daily demand
    forecast_query = text("""
                          SELECT product_id, forecast_date, predicted_quantity
                          FROM sku_demand_forecast
                          WHERE company_id = :company_id
                            AND warehouse_id = :warehouse_id
                            AND forecast_date >= CURRENT_DATE
                          ORDER BY forecast_date
                          """)

    # Current stock in hand (sum of all approved movements)
    stock_query = text("""
                       SELECT product_id,
                              SUM(
                                      CASE
                                          WHEN smt.code IN ('PURCHASE', 'TRANSFER_IN', 'ADJUSTMENT_PLUS')
                                              THEN sm.quantity
                                          WHEN smt.code IN ('SALE', 'TRANSFER_OUT', 'ADJUSTMENT_MINUS')
                                              THEN -sm.quantity
                                          ELSE 0
                                          END
                              ) AS stock_in_hand
                       FROM stock_movements sm
                                JOIN stock_movement_types smt ON sm.movement_type_id = smt.id
                       WHERE sm.company_id = :company_id
                         AND sm.warehouse_id = :warehouse_id
                         AND sm.approved = true
                       GROUP BY product_id
                       """)

    forecast_df = pd.read_sql(forecast_query, engine,
                              params={"company_id": company_id, "warehouse_id": warehouse_id})
    stock_df = pd.read_sql(stock_query, engine,
                           params={"company_id": company_id, "warehouse_id": warehouse_id})

    if forecast_df.empty or stock_df.empty:
        return []

    suggestions = []
    # Compute demand statistics per product
    for pid, group in forecast_df.groupby("product_id"):
        avg_demand = group["predicted_quantity"].mean()
        std_demand = group["predicted_quantity"].std(ddof=0)

        lead_time_days = 7  # you can later make this per-supplier
        safety_stock = Z95 * std_demand * np.sqrt(lead_time_days)
        reorder_point = (avg_demand * lead_time_days) + safety_stock

        stock_row = stock_df.loc[stock_df["product_id"] == pid]
        current_stock = float(stock_row["stock_in_hand"].values[0]) if not stock_row.empty else 0.0
        suggested_qty = max(0.0, reorder_point - current_stock)

        suggestions.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(pid),
            "reorder_point": float(reorder_point),
            "safety_stock": float(safety_stock),
            "suggested_qty": float(suggested_qty)
        })
    return suggestions
