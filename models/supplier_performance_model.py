from datetime import date

import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def score_suppliers(company_id: int, start: date, end: date):
    """
    Score suppliers for a company over a given timeframe [start, end].
    Returns dict rows to insert into DB.
    """

    query = text("""
        SELECT
            p.supplier_id,
            p.date AS order_date,
            p.date_received,
            p.rejected,
            pl.quantity_ordered,
            pl.quantity_received,
            pl.unit_cost
        FROM purchases p
        JOIN purchase_lines pl ON pl.purchase_id = p.id
        WHERE p.company_id = :company_id
          AND p.date BETWEEN :start AND :end
          AND p.approved = true
    """)

    df = pd.read_sql(
        query,
        engine,
        params={
            "company_id": company_id,
            "start": start,
            "end": end
        }
    )

    if df.empty:
        return []

    # -------------------------------------------------------
    # Transform
    # -------------------------------------------------------
    df["days_to_deliver"] = (df["date_received"] - df["order_date"]).dt.days

    # accuracy
    df["accuracy_flag"] = (
            (df["quantity_ordered"] - df["quantity_received"]).abs() <= 0.01
    ).astype(int)

    # group per supplier
    out = []
    for supplier_id, grp in df.groupby("supplier_id"):

        # 1) ON-TIME RATE
        # If days_to_deliver <= lead_time (avg), considered on-time
        lt_avg = grp["days_to_deliver"].mean() if not grp["days_to_deliver"].empty else 0
        grp["on_time_flag"] = (grp["days_to_deliver"] <= lt_avg).astype(int)
        on_time_rate = grp["on_time_flag"].mean() * 100

        # 2) ACCURACY RATE
        accuracy_rate = grp["accuracy_flag"].mean() * 100

        # 3) REJECTION RATE
        rejection_rate = (grp["rejected"].astype(int).mean()) * 100

        # 4) COST STABILITY — inverse of variation
        # Higher CV → more unstable → lower score
        if grp["unit_cost"].mean() == 0:
            cost_stability = 100.0
        else:
            cv = grp["unit_cost"].std(ddof=0) / grp["unit_cost"].mean()
            cost_stability = max(0.0, 100 * (1 - cv))

        # 5) Overall score weighted
        overall = (
                0.40 * on_time_rate +
                0.30 * accuracy_rate +
                0.10 * (100 - rejection_rate) +
                0.20 * cost_stability
        )

        out.append({
            "supplier_id": int(supplier_id),
            "company_id": company_id,
            "period_start": start,
            "period_end": end,
            "on_time_rate": round(on_time_rate, 2),
            "accuracy_rate": round(accuracy_rate, 2),
            "rejection_rate": round(rejection_rate, 2),
            "cost_stability": round(cost_stability, 2),
            "overall_score": round(overall, 2),
            "model_version": MODEL_VERSION
        })

    return out
