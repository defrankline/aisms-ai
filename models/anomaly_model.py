import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import text

from db.connection import engine


def detect_sales_anomalies(company_id: int, warehouse_id: int):
    """
    Detect anomalous days from aggregated daily sales using:
      * Z-score
      * Isolation Forest
    Returns standardized dict rows for insertion.
    """

    query = text("""
                 SELECT DATE(s.date)                     AS day,
                        SUM(sl.quantity * sl.unit_price) AS total_amount
                 FROM sales s
                          JOIN sale_lines sl ON s.id = sl.sale_id
                 WHERE s.company_id = :company_id
                   AND s.warehouse_id = :warehouse_id
                   AND s.status IN ('PAID', 'DELIVERED', 'CONFIRMED')
                 GROUP BY DATE(s.date)
                 ORDER BY day
                 """)

    df = pd.read_sql(
        query,
        engine,
        params={"company_id": company_id, "warehouse_id": warehouse_id}
    )

    if df.empty:
        return []

    df = df.sort_values("day")
    df["total_amount"] = df["total_amount"].astype(float)

    # ------------------------------------------------------------
    # ✅ 1) Z-SCORE
    # ------------------------------------------------------------
    mu = df["total_amount"].mean()
    sigma = df["total_amount"].std(ddof=0)

    if sigma == 0:
        df["z_score"] = 0
    else:
        df["z_score"] = (df["total_amount"] - mu) / sigma

    # threshold
    Z_LIMIT = 3.0
    df["is_z_anomaly"] = df["z_score"].abs() > Z_LIMIT

    # ------------------------------------------------------------
    # ✅ 2) Isolation Forest
    # ------------------------------------------------------------
    if len(df) > 10:  # at least 10 samples
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["iso_label"] = iso.fit_predict(df[["total_amount"]])
        df["iso_score"] = iso.decision_function(df[["total_amount"]])
        df["is_iso_anomaly"] = df["iso_label"] == -1
    else:
        df["iso_score"] = 0
        df["is_iso_anomaly"] = False

    # ------------------------------------------------------------
    # ✅ Combine results
    # An anomaly occurs if either method says so
    # ------------------------------------------------------------
    df["is_anomaly"] = df["is_z_anomaly"] | df["is_iso_anomaly"]

    # Severity score = |z| + (1 - iso_score)
    df["severity"] = df["z_score"].abs() + (1 - df["iso_score"])

    # Assign level
    def label_level(row):
        if not row["is_anomaly"]:
            return "INFO"
        if row["severity"] > 4:
            return "ALERT"
        return "WARN"

    df["level"] = df.apply(label_level, axis=1)

    # ------------------------------------------------------------
    # ✅ Build result JSON
    # ------------------------------------------------------------
    results = []
    for _, row in df.iterrows():
        if not row["is_anomaly"]:
            continue

        results.append({
            "company_id": company_id,
            "warehouse_id": warehouse_id,
            "sale_id": 0,  # aggregated anomaly (no single sale)
            "policy_code": "Z+ISO",
            "score": float(round(row["severity"], 4)),
            "level": row["level"],
        })

    return results
