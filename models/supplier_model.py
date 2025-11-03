import pandas as pd
from sqlalchemy import text

from db.connection import engine


def calculate_supplier_performance(company_id, start_date, end_date):
    """
    Compute supplier performance scores for a given company & period.
    """
    query = text("""
                 SELECT p.company_id,
                        p.supplier_id,
                        p.date AS order_date,
                        p.date_received,
                        p.rejected,
                        p.approved,
                        p.received,
                        pl.quantity_ordered,
                        pl.quantity_received,
                        pl.unit_cost
                 FROM purchases p
                          JOIN purchase_lines pl ON p.id = pl.purchase_id
                 WHERE p.company_id = :company_id
                   AND p.date BETWEEN :start_date AND :end_date
                   AND p.supplier_id IS NOT NULL
                 """)

    df = pd.read_sql(query, engine, params={
        "company_id": company_id,
        "start_date": start_date,
        "end_date": end_date
    })

    if df.empty:
        return []

    results = []
    # Compute metrics per supplier
    for sid, group in df.groupby("supplier_id"):
        total_orders = len(group)
        if total_orders == 0:
            continue

        # On-time rate
        group["days_diff"] = (pd.to_datetime(group["date_received"]) - pd.to_datetime(group["order_date"])).dt.days
        on_time_rate = (group["days_diff"] <= 7).mean() * 100  # within 7 days

        # Accuracy rate
        group["accuracy"] = (group["quantity_received"] / group["quantity_ordered"]).clip(upper=1)
        accuracy_rate = group["accuracy"].mean() * 100

        # Rejection rate
        rejection_rate = group["rejected"].astype(bool).mean() * 100

        # Cost stability (inverse of cost variance)
        if group["unit_cost"].std(ddof=0) == 0:
            cost_stability = 100
        else:
            coef_var = (group["unit_cost"].std(ddof=0) / group["unit_cost"].mean()) * 100
            cost_stability = max(0, 100 - coef_var)

        # Weighted overall score
        overall = (0.35 * on_time_rate +
                   0.35 * accuracy_rate +
                   0.15 * (100 - rejection_rate) +
                   0.15 * cost_stability)

        results.append({
            "supplier_id": int(sid),
            "company_id": int(company_id),
            "period_start": start_date,
            "period_end": end_date,
            "on_time_rate": round(on_time_rate, 2),
            "accuracy_rate": round(accuracy_rate, 2),
            "rejection_rate": round(rejection_rate, 2),
            "cost_stability": round(cost_stability, 2),
            "overall_score": round(overall, 2)
        })

    return results
