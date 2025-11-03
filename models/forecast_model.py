# models/forecast_model.py

import os

import joblib
import pandas as pd
from prophet import Prophet
from sqlalchemy import text

from db.connection import engine

MODELS_DIR = "models/saved"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_predict(company_id, warehouse_id, days_ahead=30):
    """
    Train or load a Prophet model per company_id + warehouse_id.
    """
    query = text("""
                 SELECT DATE(s.date)     AS ds,
                        SUM(sl.quantity) AS y
                 FROM sales s
                          JOIN sale_lines sl ON s.id = sl.sale_id
                 WHERE s.company_id = :company_id
                   AND s.warehouse_id = :warehouse_id
                   AND s.status IN ('PAID', 'DELIVERED')
                 GROUP BY DATE(s.date)
                 ORDER BY ds;
                 """)

    df = pd.read_sql(query, engine, params={
        "company_id": company_id,
        "warehouse_id": warehouse_id
    })

    if df.empty:
        print(f"[AI] No sales data for company {company_id}, warehouse {warehouse_id}")
        return []

    # 🔹 Model file now includes both company and warehouse
    model_path = os.path.join(
        MODELS_DIR,
        f"forecast_company_{company_id}_warehouse_{warehouse_id}.pkl"
    )

    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"[AI] Loaded model for company {company_id}, warehouse {warehouse_id}")
        except Exception as e:
            print(f"[AI] Failed to load model: {e}")

    if model is None:
        model = Prophet()
        model.fit(df)
        joblib.dump(model, model_path)
        print(f"[AI] Trained and saved model for company {company_id}, warehouse {warehouse_id}")

    # 🔹 Predict forward
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    results = []
    for _, row in forecast.tail(days_ahead).iterrows():
        results.append({
            "forecast_date": str(row["ds"].date()),
            "predicted_quantity": float(row["yhat"])
        })

    return results
