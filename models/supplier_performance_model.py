from datetime import date

import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def score_suppliers(company_id: int, start: date, end: date):
    query = text("""
                 SELECT p.supplier_id        AS supplier_id,
                        s.name               AS supplier_name,
                        p.date               AS order_date,
                        p.date_received      AS date_received,
                        p.rejected           AS rejected,
                        pl.quantity_ordered  AS quantity_ordered,
                        pl.quantity_received AS quantity_received,
                        pl.unit_cost         AS unit_cost
                 FROM purchases p
                          JOIN purchase_lines pl ON pl.purchase_id = p.id
                          JOIN suppliers s ON p.supplier_id = s.id
                 WHERE p.company_id = :company_id
                   AND p.date BETWEEN :start AND :end
                   AND p.approved = TRUE
                 """)

    df = pd.read_sql(
        query,
        engine,
        params={"company_id": company_id, "start": start, "end": end}
    )

    # No rows
    if df.empty:
        return []

    # --- FIX DATES ---
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    # replace missing date_received with order_date
    df["date_received"] = df["date_received"].fillna(df["order_date"])

    # Days to deliver
    df["days_to_deliver"] = (df["date_received"] - df["order_date"]).dt.days.fillna(0).astype(int)

    # fix numeric
    df["quantity_ordered"] = df["quantity_ordered"].fillna(0)
    df["quantity_received"] = df["quantity_received"].fillna(0)
    df["unit_cost"] = df["unit_cost"].astype(float)

    # accuracy flag: received = ordered
    df["accuracy_flag"] = (
            (df["quantity_ordered"] - df["quantity_received"]).abs() <= 0.01
    ).astype(int)

    output = []

    # --- GROUP PER SUPPLIER ---
    for sup_id, grp in df.groupby("supplier_id"):

        # on-time delivery
        avg_lt = grp["days_to_deliver"].mean() if not grp.empty else 0
        grp["on_time_flag"] = (grp["days_to_deliver"] <= avg_lt).astype(int)
        on_time_rate = grp["on_time_flag"].mean() * 100

        # accuracy
        accuracy_rate = grp["accuracy_flag"].mean() * 100

        # rejection
        rejection_rate = grp["rejected"].astype(int).mean() * 100

        # cost stability = 100(1 - CV)
        if grp["unit_cost"].mean() == 0:
            cost_stability = 100.0
        else:
            cv = grp["unit_cost"].std(ddof=0) / grp["unit_cost"].mean()
            cost_stability = max(0.0, 100 * (1 - cv))

        # final score
        overall_score = (
                0.40 * on_time_rate +
                0.30 * accuracy_rate +
                0.10 * (100 - rejection_rate) +
                0.20 * cost_stability
        )

        output.append({
            "supplier_id": int(sup_id),
            "company_id": company_id,
            "period_start": start,
            "period_end": end,
            "on_time_rate": round(on_time_rate, 2),
            "accuracy_rate": round(accuracy_rate, 2),
            "rejection_rate": round(rejection_rate, 2),
            "cost_stability": round(cost_stability, 2),
            "overall_score": round(overall_score, 2),
            "model_version": MODEL_VERSION
        })

    return output
