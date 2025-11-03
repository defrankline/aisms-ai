from datetime import datetime

import numpy as np
import pandas as pd
from prophet import Prophet
from sqlalchemy import text

from db.connection import engine


def calculate_profitability(company_id, warehouse_id):
    """
    Compute monthly revenue, COGS, expenses, and net profit.
    Forecast next 3 months using Prophet.
    """
    # --- Revenue (Sales) ---
    sales_query = text("""
                       SELECT DATE_TRUNC('month', s.date)::date AS month,
                              SUM(sl.quantity * sl.unit_price)  AS revenue
                       FROM sales s
                                JOIN sale_lines sl ON s.id = sl.sale_id
                       WHERE s.company_id = :company_id
                         AND s.warehouse_id = :warehouse_id
                         AND s.status IN ('PAID', 'DELIVERED')
                       GROUP BY DATE_TRUNC('month', s.date)
                       ORDER BY month;
                       """)

    # --- COGS (Purchases) ---
    cogs_query = text("""
                      SELECT DATE_TRUNC('month', p.date)::date        AS month,
                             SUM(pl.quantity_received * pl.unit_cost) AS cogs
                      FROM purchases p
                               JOIN purchase_lines pl ON p.id = pl.purchase_id
                      WHERE p.company_id = :company_id
                        AND p.warehouse_id = :warehouse_id
                        AND p.approved = true
                      GROUP BY DATE_TRUNC('month', p.date)
                      ORDER BY month;
                      """)

    # --- Expenses ---
    expenses_query = text("""
                          SELECT DATE_TRUNC('month', e.date)::date AS month,
                                 SUM(el.amount)                    AS expenses
                          FROM expenses e
                                   JOIN expense_lines el ON e.id = el.expense_id
                          WHERE e.company_id = :company_id
                            AND e.store_id = :warehouse_id
                            AND e.approved = true
                          GROUP BY DATE_TRUNC('month', e.date)
                          ORDER BY month;
                          """)

    sales_df = pd.read_sql(sales_query, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})
    cogs_df = pd.read_sql(cogs_query, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})
    exp_df = pd.read_sql(expenses_query, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})

    if sales_df.empty and cogs_df.empty and exp_df.empty:
        return []

    df = pd.DataFrame({"month": pd.date_range(start="2024-01-01", end=datetime.utcnow().date(), freq="MS")})
    df = df.merge(sales_df, on="month", how="left")
    df = df.merge(cogs_df, on="month", how="left")
    df = df.merge(exp_df, on="month", how="left")

    df.fillna(0, inplace=True)
    df["net_profit"] = df["revenue"] - df["cogs"] - df["expenses"]
    df["profit_margin"] = np.where(df["revenue"] > 0, (df["net_profit"] / df["revenue"]) * 100, 0)

    # Trend detection
    df["trend"] = np.where(df["net_profit"].diff() > 0, "Rising",
                           np.where(df["net_profit"].diff() < 0, "Falling", "Stable"))

    # Forecast next 3 months using Prophet
    prophet_df = df[["month", "net_profit"]].rename(columns={"month": "ds", "net_profit": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=3, freq="MS")
    forecast = model.predict(future)
    forecast_future = forecast.tail(3)[["ds", "yhat"]].rename(columns={"ds": "month", "yhat": "forecast_profit"})

    df = df.merge(forecast_future, on="month", how="outer")

    # Build results
    results = []
    for _, row in df.iterrows():
        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "month": row["month"].date(),
            "total_revenue": float(row["revenue"]),
            "total_cogs": float(row["cogs"]),
            "total_expenses": float(row["expenses"]),
            "net_profit": float(row["net_profit"]),
            "profit_margin": float(row["profit_margin"]),
            "trend": str(row["trend"]),
            "forecast_profit": float(row["forecast_profit"]) if not pd.isna(row["forecast_profit"]) else None
        })

    return results
