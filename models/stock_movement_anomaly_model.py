import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION


def _level(severity: float) -> str:
    if severity > 4:
        return "ALERT"
    if severity > 2:
        return "WARN"
    return "INFO"


def _build_reason(row, mu, sigma) -> str:
    reasons = []
    qty = float(row["quantity"])
    abs_qty = float(row["abs_qty"])
    mt = int(row["movement_type_id"])
    pid = int(row["product_id"])

    if bool(row["is_z_anomaly"]):
        reasons.append(
            f"Z-score anomaly for product {pid} type {mt}: abs_qty={abs_qty:.3f} vs mean {mu:.3f} (σ={sigma:.3f})."
        )

    if bool(row["is_iso_anomaly"]):
        reasons.append(
            f"Isolation Forest flagged this movement as unusual (iso_score={float(row['iso_score']):.3f})."
        )

    if row.get("source"):
        reasons.append(f"source='{row['source']}'.")

    if not reasons:
        return "Marked as anomaly, but no specific reason computed."

    return " ".join(reasons)[:480]


def detect_stock_movement_anomalies(company_id: int, warehouse_id: int, days_window: int = 90, limit: int = 200):
    """
    Detect anomalies per stock movement record.

    Returns rows compatible with StockMovementAnomalyEvent insert:
      company_id, warehouse_id, product_id, stock_movement_id,
      movement_date, movement_type_id, quantity, unit_cost, source,
      policy_code, score, level, reason, model_version
    """

    q = text("""
             SELECT sm.id                   AS stock_movement_id,
                    sm.company_id           AS company_id,
                    sm.warehouse_id         AS warehouse_id,
                    sm.product_id           AS product_id,
                    sm.movement_type_id     AS movement_type_id,
                    sm.date::date           AS movement_date,
                    sm.date_received        AS movement_ts,
                    sm.quantity             AS quantity,
                    sm.unit_cost            AS unit_cost,
                    COALESCE(sm.source, '') AS source
             FROM stock_movements sm
             WHERE sm.approved = TRUE
               AND sm.company_id = :company_id
               AND sm.warehouse_id = :warehouse_id
               AND sm.date >= CURRENT_DATE - (:days_window || ' days')::interval
             ORDER BY sm.date_received DESC, sm.id DESC
             """)

    df = pd.read_sql(q, engine, params={
        "company_id": company_id,
        "warehouse_id": warehouse_id,
        "days_window": int(days_window),
    })

    if df.empty:
        return []

    # Normalize
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0).astype(float)
    df["abs_qty"] = df["quantity"].abs()
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")

    # Group baselines per (product_id, movement_type_id)
    g = df.groupby(["product_id", "movement_type_id"])["abs_qty"]
    stats = g.agg(mu="mean", sigma=lambda s: float(np.std(s.values, ddof=0)) if len(s) > 1 else 0.0).reset_index()
    df = df.merge(stats, on=["product_id", "movement_type_id"], how="left")

    df["sigma"] = df["sigma"].replace(0.0, 1e-9)
    df["z_score"] = (df["abs_qty"] - df["mu"]) / df["sigma"]
    df["is_z_anomaly"] = df["z_score"].abs() > 3.0

    # IsolationForest (multivariate: abs_qty, unit_cost)
    # Fill unit_cost nulls with 0 (so ISO can run)
    df["unit_cost_f"] = df["unit_cost"].fillna(0.0).astype(float)

    if len(df) > 25:
        iso = IsolationForest(contamination=0.05, random_state=42)
        X = df[["abs_qty", "unit_cost_f"]].to_numpy(dtype=float)
        df["iso_label"] = iso.fit_predict(X)
        df["iso_score"] = iso.decision_function(X)  # higher = more normal
        df["is_iso_anomaly"] = df["iso_label"] == -1
    else:
        df["iso_score"] = 0.5
        df["is_iso_anomaly"] = False

    df["is_anomaly"] = df["is_z_anomaly"] | df["is_iso_anomaly"]

    # Severity (match your sales anomaly pattern)
    # iso_score is "normality"; convert to anomaly contribution as (1 - iso_score_scaled).
    # decision_function is often around [-0.5..0.5], so scale to 0..1 using min/max.
    iso_min = float(df["iso_score"].min())
    iso_max = float(df["iso_score"].max())
    denom = max(1e-9, iso_max - iso_min)
    df["iso_score_01"] = (df["iso_score"] - iso_min) / denom  # 0..1, higher=more normal
    df["severity"] = df["z_score"].abs() + (1.0 - df["iso_score_01"])

    df["level"] = df["severity"].apply(_level)

    # Build result rows (top N by severity)
    anomalies = df[df["is_anomaly"]].sort_values("severity", ascending=False).head(int(limit))

    results = []
    for _, row in anomalies.iterrows():
        reason = _build_reason(row, float(row["mu"]), float(row["sigma"]))
        results.append({
            "company_id": int(company_id),
            "warehouse_id": int(warehouse_id),
            "product_id": int(row["product_id"]),
            "stock_movement_id": int(row["stock_movement_id"]),
            "movement_date": row["movement_date"],
            "movement_type_id": int(row["movement_type_id"]),
            "quantity": float(row["quantity"]),
            "unit_cost": float(row["unit_cost"]) if pd.notna(row["unit_cost"]) else None,
            "source": str(row["source"]),
            "policy_code": "Z+ISO",
            "score": float(round(float(row["severity"]), 4)),
            "level": str(row["level"]),
            "reason": reason,
            "model_version": MODEL_VERSION,
        })

    return results
