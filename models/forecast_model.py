import os

import joblib
import pandas as pd
from prophet import Prophet
from sqlalchemy import text

from db.connection import engine
from utils.ai_config import MODEL_VERSION, DEFAULT_FORECAST_DAYS, MIN_TS_POINTS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(MODELS_DIR, exist_ok=True)


def _model_path(company_id: int, warehouse_id: int, product_id: int) -> str:
    return os.path.join(
        MODELS_DIR,
        f"forecast_company_{company_id}_warehouse_{warehouse_id}_product_{product_id}.pkl"
    )


def _load_or_train(df: pd.DataFrame, model_path: str) -> Prophet:
    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            model = None
    if model is None:
        model = Prophet()
        model.fit(df)
        joblib.dump(model, model_path)
    return model


def train_and_predict(company_id: int, warehouse_id: int, days_ahead: int = DEFAULT_FORECAST_DAYS,
                      product_id: int | None = None):
    """
    Trains (or loads) Prophet models per product for a given company+warehouse.
    If product_id is provided, restrict to that SKU; otherwise forecast all SKUs with enough data.
    Returns list of dict rows to insert into sku_demand_forecast.
    """

    # Pull daily sales per SKU
    base_query = """
        SELECT
            sl.product_id,
            DATE(s.date) AS ds,
            SUM(sl.quantity) AS y
        FROM sales s
        JOIN sale_lines sl ON s.id = sl.sale_id
        WHERE s.company_id = :company_id
          AND s.warehouse_id = :warehouse_id
          AND s.status IN ('PAID','DELIVERED')
        {product_filter}
        GROUP BY sl.product_id, DATE(s.date)
        ORDER BY sl.product_id, ds
    """

    product_filter = ""
    params = {"company_id": company_id, "warehouse_id": warehouse_id}
    if product_id is not None:
        product_filter = "AND sl.product_id = :product_id"
        params["product_id"] = product_id

    query = text(base_query.format(product_filter=product_filter))
    df = pd.read_sql(query, engine, params=params)

    if df.empty:
        return []

    results = []
    for pid, grp in df.groupby("product_id"):
        # Prophet needs columns ds (date) and y (value)
        grp = grp[["ds", "y"]].sort_values("ds")

        # Require enough points
        if len(grp) < MIN_TS_POINTS:
            continue

        path = _model_path(company_id, warehouse_id, int(pid))
        model = _load_or_train(grp, path)

        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        # take only the future horizon (last N)
        tail = forecast.tail(days_ahead)[["ds", "yhat"]]

        for _, row in tail.iterrows():
            results.append({
                "company_id": company_id,
                "warehouse_id": warehouse_id,
                "product_id": int(pid),
                "forecast_date": row["ds"].date(),
                "predicted_quantity": float(row["yhat"]),
                "model_version": MODEL_VERSION
            })

    return results
