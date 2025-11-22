import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import text

from db.connection import engine


# ----------------------------------------------------------
# HUMAN-READABLE REASON BUILDER
# ----------------------------------------------------------
def build_reason(row, mu, sigma):
    amount = row["total_amount"]
    sale_date = row["sale_date"]

    reasons = []

    # Z-score issues
    if row["is_z_anomaly"]:
        reasons.append(
            f"Z-score anomaly on {sale_date}: {amount:,.0f} vs mean {mu:,.0f} (σ={sigma:,.0f})."
        )

    # Isolation forest issues
    if row["is_iso_anomaly"]:
        reasons.append(
            f"Isolation Forest flagged this sale as unusual (iso_score={row['iso_score']:.3f})."
        )

    # Severity
    if row["severity"] > 5:
        reasons.append("Severity extremely high based on combined metrics.")
    elif row["severity"] > 3:
        reasons.append("Severity moderately high.")

    if not reasons:
        return "Marked as anomaly, but no specific reason computed."

    return " ".join(reasons)


# ----------------------------------------------------------
# MAIN ANOMALY DETECTION FUNCTION
# ----------------------------------------------------------
def detect_sales_anomalies(company_id: int, warehouse_id: int):
    """
    Detect anomalies per sale.
    Each row contains:
      sale_id, reference_number, sale_date, warehouse_name,
      total_amount, policy_code, score, level, reason
    """

    query = text("""
                 SELECT s.id                             AS sale_id,
                        s.reference_number,
                        DATE(s.date)                     AS sale_date,
                        w.name                           AS warehouse_name,
                        SUM(sl.quantity * sl.unit_price) AS total_amount
                 FROM sales s
                          JOIN sale_lines sl ON s.id = sl.sale_id
                          JOIN warehouses w ON w.id = s.warehouse_id
                 WHERE s.company_id = :company_id
                   AND s.warehouse_id = :warehouse_id
                   AND s.status IN ('PAID', 'DELIVERED', 'CONFIRMED')
                 GROUP BY s.id, s.reference_number, s.date, w.name
                 ORDER BY s.date
                 """)

    df = pd.read_sql(
        query,
        engine,
        params={"company_id": company_id, "warehouse_id": warehouse_id}
    )

    if df.empty:
        return []

    df["total_amount"] = df["total_amount"].astype(float)

    # ------------------------------------------------------
    # Z-SCORE
    # ------------------------------------------------------
    mu = df["total_amount"].mean()
    sigma = df["total_amount"].std(ddof=0)

    df["z_score"] = 0 if sigma == 0 else (df["total_amount"] - mu) / sigma
    df["is_z_anomaly"] = df["z_score"].abs() > 3.0  # threshold

    # ------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------
    if len(df) > 10:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["iso_label"] = iso.fit_predict(df[["total_amount"]])
        df["iso_score"] = iso.decision_function(df[["total_amount"]])
        df["is_iso_anomaly"] = df["iso_label"] == -1
    else:
        df["iso_score"] = 0.5  # neutral score
        df["is_iso_anomaly"] = False

    # ------------------------------------------------------
    # Combined detection
    # ------------------------------------------------------
    df["is_anomaly"] = df["is_z_anomaly"] | df["is_iso_anomaly"]

    # severity = |z| + (1 - iso_score)
    df["severity"] = df["z_score"].abs() + (1 - df["iso_score"])

    # Level mapping
    def level_of(row):
        if not row["is_anomaly"]:
            return "INFO"
        if row["severity"] > 4:
            return "ALERT"
        return "WARN"

    df["level"] = df.apply(level_of, axis=1)

    # ------------------------------------------------------
    # Build results
    # ------------------------------------------------------
    results = []
    for _, row in df.iterrows():
        if not row["is_anomaly"]:
            continue

        reason = build_reason(row, mu, sigma)

        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "sale_id": int(row["sale_id"]),
            "reference_number": row["reference_number"],
            "sale_date": row["sale_date"],
            "warehouse_name": row["warehouse_name"],
            "total_amount": float(row["total_amount"]),
            "policy_code": "Z+ISO",  # consistent naming
            "score": float(round(row["severity"], 4)),
            "level": row["level"],
            "reason": reason
        })

    return results
