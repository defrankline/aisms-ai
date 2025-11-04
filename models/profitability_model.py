import numpy as np
import pandas as pd
from prophet import Prophet
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    coef = np.polyfit(x, values, 1)
    return float(coef[0])


def compute_monthly_profitability(company_id: int, warehouse_id: int):
    """
    Monthly P&L and one-month-ahead forecast of net profit.
    """

    # Revenue per month
    q_rev = text("""
                 SELECT DATE_TRUNC('month', s.date)::date AS month,
                        SUM(sl.quantity * sl.unit_price)  AS revenue
                 FROM sales s
                          JOIN sale_lines sl ON sl.sale_id = s.id
                 WHERE s.company_id = :company_id
                   AND s.warehouse_id = :warehouse_id
                   AND s.status IN ('PAID', 'DELIVERED', 'CONFIRMED')
                 GROUP BY DATE_TRUNC('month', s.date)
                 ORDER BY month
                 """)

    # COGS proxy per month (sum of received cost this month)
    q_cogs = text("""
                  SELECT DATE_TRUNC('month', p.date)::date        AS month,
                         SUM(pl.quantity_received * pl.unit_cost) AS cogs
                  FROM purchases p
                           JOIN purchase_lines pl ON pl.purchase_id = p.id
                  WHERE p.company_id = :company_id
                    AND p.warehouse_id = :warehouse_id
                    AND p.approved = true
                  GROUP BY DATE_TRUNC('month', p.date)
                  ORDER BY month
                  """)

    # Expenses per month
    q_exp = text("""
                 SELECT DATE_TRUNC('month', e.date)::date AS month,
                        SUM(el.amount)                    AS expenses
                 FROM expenses e
                          JOIN expense_lines el ON el.expense_id = e.id
                 WHERE e.company_id = :company_id
                   AND e.store_id = :warehouse_id
                   AND e.approved = true
                 GROUP BY DATE_TRUNC('month', e.date)
                 ORDER BY month
                 """)

    rev = pd.read_sql(q_rev, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})
    cogs = pd.read_sql(q_cogs, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})
    exp = pd.read_sql(q_exp, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})

    if rev.empty and cogs.empty and exp.empty:
        return []

    df = rev.merge(cogs, on="month", how="outer").merge(exp, on="month", how="outer").fillna(0)
    df = df.sort_values("month")
    df["net_profit"] = df["revenue"] - df["cogs"] - df["expenses"]
    df["profit_margin"] = (df["net_profit"] / df["revenue"].replace(0, np.nan)).fillna(0)

    # Trend label on last 6 months net_profit
    last_n = df.tail(6)
    slope = _slope(last_n["net_profit"].tolist())
    trend = "UP" if slope > 0 else ("DOWN" if slope < 0 else "FLAT")

    # Prophet forecast next month net_profit
    prophet_df = df[["month", "net_profit"]].rename(columns={"month": "ds", "net_profit": "y"})
    model = Prophet()
    if len(prophet_df) >= 2:
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=1, freq="MS")
        yhat = float(model.predict(future).tail(1)["yhat"].iloc[0])
    else:
        yhat = float(df["net_profit"].iloc[-1] if not df.empty else 0.0)

    # Prepare rows for each month in df
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "month": pd.to_datetime(r["month"]).date(),
            "total_revenue": float(r["revenue"]),
            "total_cogs": float(r["cogs"]),
            "total_expenses": float(r["expenses"]),
            "net_profit": float(r["net_profit"]),
            "profit_margin": float(r["profit_margin"]),
            "trend": trend,  # same label for current context
            "forecast_profit": round(yhat, 2),
            "model_version": MODEL_VERSION
        })
    return rows
