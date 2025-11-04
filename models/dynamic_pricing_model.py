import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _safe_pct_change(new, old):
    if old == 0:
        return 100.0 if new > 0 else 0.0
    return ((new - old) / old) * 100.0


def calc_price_elasticity(df: pd.DataFrame) -> float:
    """Elasticity = d(log Q) / d(log P)."""
    df = df.sort_values("day")
    if len(df) < 2:
        return -1.0
    df = df[df["quantity"] > 0]
    if df.empty:
        return -1.0
    df["log_price"] = np.log(df["unit_price"])
    df["log_qty"] = np.log(df["quantity"])
    x = df["log_price"].to_numpy()
    y = df["log_qty"].to_numpy()
    if np.std(x) == 0:
        return -1.0
    cov = np.cov(x, y)
    slope = cov[0, 1] / cov[0, 0]
    return float(slope)


def recommend_prices(company_id: int, warehouse_id: int):
    """
    Builds DynamicPricingRecommendation rows matching your model exactly.
    """
    # 1) Historical sales (price/qty)
    q_sales = text("""
                   SELECT sl.product_id,
                          DATE(s.date) AS day,
                          sl.unit_price,
                          sl.quantity
                   FROM sales s
                            JOIN sale_lines sl ON sl.sale_id = s.id
                   WHERE s.company_id = :company_id
                     AND s.warehouse_id = :warehouse_id
                     AND s.status IN ('PAID', 'DELIVERED', 'CONFIRMED')
                     AND sl.quantity > 0
                     AND sl.unit_price > 0
                   ORDER BY sl.product_id, day
                   """)
    sales_df = pd.read_sql(q_sales, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})
    if sales_df.empty:
        return []

    # 2) Latest cost (optional signal)
    q_cost = text("""
                  SELECT pl.product_id, AVG(pl.unit_cost) AS avg_unit_cost
                  FROM purchases p
                           JOIN purchase_lines pl ON pl.purchase_id = p.id
                  WHERE p.company_id = :company_id
                    AND p.warehouse_id = :warehouse_id
                  GROUP BY pl.product_id
                  """)
    cost_df = pd.read_sql(q_cost, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})

    # 3) Near-term demand forecast (next 30d)
    q_fc = text("""
                SELECT product_id, AVG(predicted_quantity) AS avg_forecast
                FROM sku_demand_forecast
                WHERE company_id = :company_id
                  AND warehouse_id = :warehouse_id
                  AND forecast_date >= CURRENT_DATE
                  AND forecast_date < CURRENT_DATE + INTERVAL '30 days'
                GROUP BY product_id
                """)
    fc_df = pd.read_sql(q_fc, engine, params={"company_id": company_id, "warehouse_id": warehouse_id})

    results = []
    for pid, grp in sales_df.groupby("product_id"):
        grp = grp.sort_values("day")
        current_price = float(grp["unit_price"].iloc[-1])
        # Simple recent demand proxy
        recent_qty = float(grp["quantity"].tail(7).mean()) if len(grp) >= 7 else float(grp["quantity"].mean())

        # cost
        avg_cost = 0.0
        c = cost_df[cost_df["product_id"] == pid]
        if not c.empty:
            avg_cost = float(c["avg_unit_cost"].iloc[0])

        # forecast demand
        avg_forecast = recent_qty
        f = fc_df[fc_df["product_id"] == pid]
        if not f.empty:
            avg_forecast = float(f["avg_forecast"].iloc[0])

        # elasticity
        elasticity = calc_price_elasticity(grp)  # usually negative

        # baseline: cost + 20% margin floor
        floor_price = max(avg_cost * 1.20, avg_cost * 1.05)
        # elasticity adjustment: if highly elastic (e.g., -1.5), keep prices closer to floor
        elasticity_factor = 1.0 + max(-0.3, min(0.3, -0.1 * elasticity))
        # demand pressure: if forecast >> recent, we can push price slightly
        demand_factor = 1.0 + max(-0.1, min(0.1, (avg_forecast - recent_qty) / (recent_qty + 1e-6) * 0.05))

        suggested = max(floor_price, current_price * elasticity_factor * demand_factor)

        price_change_pct = _safe_pct_change(suggested, current_price)
        expected_demand_change = -elasticity * (price_change_pct / 100.0) * 100.0  # %ΔQ ≈ -e * %ΔP
        # confidence: more data points → higher, else lower
        n = len(grp)
        confidence_level = float(max(30.0, min(95.0, 30 + n * 2)))

        rationale_bits = []
        if avg_cost > 0:
            rationale_bits.append("Cost floor")
        if elasticity < -0.8:
            rationale_bits.append("High elasticity")
        elif elasticity > -0.2:
            rationale_bits.append("Low elasticity")
        if avg_forecast > recent_qty * 1.1:
            rationale_bits.append("Rising demand")
        elif avg_forecast < recent_qty * 0.9:
            rationale_bits.append("Weak demand")
        rationale = ", ".join(rationale_bits) or "Balanced factors"

        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "product_id": int(pid),
            "current_price": round(current_price, 2),
            "suggested_price": round(float(suggested), 2),
            "price_change_pct": round(float(price_change_pct), 2),
            "expected_demand_change": round(float(expected_demand_change), 2),
            "confidence_level": round(confidence_level, 2),
            "rationale": rationale,
            "model_version": MODEL_VERSION
        })

    return results
