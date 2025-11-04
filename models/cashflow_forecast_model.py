import pandas as pd
from prophet import Prophet
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _score_cash(balance: float, std: float) -> float:
    """
    Normalize cash safety: higher balance relative to std is safer.
    """
    if std <= 0:
        return 90.0 if balance > 0 else 30.0
    ratio = balance / std
    if ratio > 4:
        return 95
    if ratio > 2:
        return 80
    if ratio > 1:
        return 65
    if ratio > 0:
        return 50
    return 20


def _risk_level(score: float) -> str:
    if score >= 75:
        return "LOW"
    if score >= 50:
        return "MEDIUM"
    return "HIGH"


def compute_cashflow_forecast(company_id: int, warehouse_id: int):
    """
    Computes monthly cashflow + forecasts next month balance.
    """

    # ---------------------------- Inflows (Sales) ----------------------------
    q_in = text("""
                SELECT DATE_TRUNC('month', s.date)::date AS month,
                       SUM(spm.amount)                   AS inflow
                FROM sales s
                         JOIN sale_payment_methods spm ON spm.sale_id = s.id
                WHERE s.company_id = :company_id
                  AND s.warehouse_id = :warehouse_id
                  AND s.status IN ('PAID', 'DELIVERED', 'CONFIRMED')
                GROUP BY DATE_TRUNC('month', s.date)
                ORDER BY month
                """)

    # ---------------------------- Purchase Outflows -------------------------
    q_po = text("""
                SELECT DATE_TRUNC('month', p.date)::date        AS month,
                       SUM(pl.quantity_received * pl.unit_cost) AS outflow
                FROM purchases p
                         JOIN purchase_lines pl ON pl.purchase_id = p.id
                WHERE p.company_id = :company_id
                  AND p.warehouse_id = :warehouse_id
                  AND p.approved = true
                GROUP BY DATE_TRUNC('month', p.date)
                ORDER BY month
                """)

    # ---------------------------- Expense Outflows --------------------------
    q_exp = text("""
                 SELECT DATE_TRUNC('month', e.date)::date AS month,
                        SUM(el.amount)                    AS outflow
                 FROM expenses e
                          JOIN expense_lines el ON el.expense_id = e.id
                 WHERE e.company_id = :company_id
                   AND e.store_id = :warehouse_id
                   AND e.approved = true
                 GROUP BY DATE_TRUNC('month', e.date)
                 ORDER BY month
                 """)

    inflow_df = pd.read_sql(q_in, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})
    purch_df = pd.read_sql(q_po, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})
    exp_df = pd.read_sql(q_exp, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})

    if inflow_df.empty and purch_df.empty and exp_df.empty:
        return []

    df = inflow_df.merge(purch_df, on="month", how="outer", suffixes=("_in", "_po"))
    df = df.merge(exp_df, on="month", how="outer", suffixes=("", "_exp"))

    df["inflow"] = df["inflow"].fillna(0.0)
    df["outflow_po"] = df["outflow_po"].fillna(0.0)
    df["outflow"] = df["outflow"].fillna(0.0)

    df = df.rename(columns={"outflow_po": "purchases", "outflow": "expenses"})
    df["outflow_total"] = df["purchases"] + df["expenses"]
    df["net_cashflow"] = df["inflow"] - df["outflow_total"]

    # Rolling cash balance
    df = df.sort_values("month")
    df["cash_balance"] = df["net_cashflow"].cumsum()

    # Cash safety score
    std_cash = float(df["cash_balance"].std()) if len(df) > 1 else 1.0
    df["cash_health_score"] = df["cash_balance"].apply(lambda x: _score_cash(x, std_cash))
    df["risk_level"] = df["cash_health_score"].apply(_risk_level)

    # Prophet forecast for next month balance
    prophet_df = df[["month", "cash_balance"]].rename(columns={"month": "ds", "cash_balance": "y"})
    if len(prophet_df) >= 2:
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=1, freq="MS")
        yhat = float(m.predict(future).tail(1)["yhat"].iloc[0])
    else:
        yhat = float(df["cash_balance"].iloc[-1])

    # build results
    results = []
    for _, r in df.iterrows():
        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "month": pd.to_datetime(r["month"]).date(),
            "cash_inflows": float(r["inflow"]),
            "cash_outflows": float(r["outflow_total"]),
            "net_cashflow": float(r["net_cashflow"]),
            "cash_balance": float(r["cash_balance"]),
            "cash_health_score": float(r["cash_health_score"]),
            "risk_level": r["risk_level"],
            "forecasted_next_balance": round(float(yhat), 2),
            "model_version": MODEL_VERSION
        })

    return results
