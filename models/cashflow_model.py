import numpy as np
import pandas as pd
from prophet import Prophet
from sqlalchemy import text

from db.connection import engine


def compute_cashflow(company_id, warehouse_id):
    """
    Monthly cash inflows/outflows and forecasted liquidity.
    """

    # --- Inflows (Sales Payments) ---
    inflow_query = text("""
                        SELECT DATE_TRUNC('month', s.date)::date AS month,
                               SUM(spm.amount)                   AS inflows
                        FROM sales s
                                 JOIN sale_payment_methods spm ON s.id = spm.sale_id
                        WHERE s.company_id = :company_id
                          AND s.warehouse_id = :warehouse_id
                          AND s.status IN ('PAID', 'DELIVERED')
                        GROUP BY DATE_TRUNC('month', s.date)
                        ORDER BY month;
                        """)

    # --- Outflows (Purchases + Expenses) ---
    outflow_query = text("""
                         SELECT DATE_TRUNC('month', month)::date AS month, SUM(total) AS outflows
                         FROM (SELECT DATE_TRUNC('month', p.date)::date        AS month,
                                      SUM(pl.quantity_received * pl.unit_cost) AS total
                               FROM purchases p
                                        JOIN purchase_lines pl ON p.id = pl.purchase_id
                               WHERE p.company_id = :company_id
                                 AND p.warehouse_id = :warehouse_id
                                 AND p.approved = true
                               GROUP BY DATE_TRUNC('month', p.date)
                               UNION ALL
                               SELECT DATE_TRUNC('month', e.date)::date AS month,
                                      SUM(el.amount)                    AS total
                               FROM expenses e
                                        JOIN expense_lines el ON e.id = el.expense_id
                               WHERE e.company_id = :company_id
                                 AND e.store_id = :warehouse_id
                                 AND e.approved = true
                               GROUP BY DATE_TRUNC('month', e.date)) t
                         GROUP BY DATE_TRUNC('month', month)
                         ORDER BY month;
                         """)

    inflow_df = pd.read_sql(inflow_query, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})
    outflow_df = pd.read_sql(outflow_query, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})

    if inflow_df.empty and outflow_df.empty:
        return []

    df = pd.merge(inflow_df, outflow_df, on="month", how="outer").fillna(0)
    df["net_cashflow"] = df["inflows"] - df["outflows"]

    # Compute rolling cash balance
    df["cash_balance"] = df["net_cashflow"].cumsum()

    # Simple Prophet forecast
    prophet_df = df[["month", "net_cashflow"]].rename(columns={"month": "ds", "net_cashflow": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=3, freq="MS")
    forecast = model.predict(future)
    future_df = forecast.tail(3)[["ds", "yhat"]].rename(columns={"ds": "month", "yhat": "forecasted_next_balance"})

    df = df.merge(future_df, on="month", how="outer")

    # Health & Risk scoring
    df["cash_health_score"] = np.clip(100 - (df["outflows"] / (df["inflows"] + 1)) * 100, 0, 100)
    df["risk_level"] = pd.cut(
        df["cash_health_score"], bins=[-1, 40, 70, 100], labels=["CRITICAL", "MODERATE", "HEALTHY"]
    )

    # Format results
    results = []
    for _, r in df.iterrows():
        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "month": r["month"].date(),
            "cash_inflows": float(r["inflows"]),
            "cash_outflows": float(r["outflows"]),
            "net_cashflow": float(r["net_cashflow"]),
            "cash_balance": float(r["cash_balance"]),
            "cash_health_score": float(round(r["cash_health_score"], 2)),
            "risk_level": str(r["risk_level"]),
            "forecasted_next_balance": float(r["forecasted_next_balance"]) if not pd.isna(
                r["forecasted_next_balance"]) else None
        })
    return results
