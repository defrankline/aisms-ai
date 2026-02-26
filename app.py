import logging
import math
from datetime import datetime
from decimal import Decimal

import numpy as np
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
from models.stock_movement_anomaly_model import detect_stock_movement_anomalies
from models.stocktake_variance_model import compute_stocktake_variance_risk
from models.transfer_recommendation_model import recommend_transfers
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from db.connection import SessionLocal
from db.connection import engine
from db.models import Base, SkuDemandForecast, SalesAnomalyEvent, ReorderSuggestion, SupplierPerformanceScore, \
    CustomerSegment, DynamicPricingRecommendation, SalesPerformanceScore, ProfitabilityForecast, InventoryOptimization, \
    CashflowForecast
from db.models import (
    StockMovementAnomalyEvent, StocktakeVarianceRisk, TransferRecommendation,
    StockoutRisk, SlowMoverRisk, ReturnPressureIndex, ReceiptCostAlert, ForecastModelMetric
)
from models.anomaly_model import detect_sales_anomalies
from models.cashflow_model import compute_cashflow
from models.customer_segmentation_model import calculate_customer_segments
from models.dynamic_pricing_model import recommend_prices
from models.forecast_metrics_model import compute_forecast_metrics
from models.forecast_model import train_and_predict
from models.inventory_optimization_model import optimize_inventory
from models.profitability_model import compute_monthly_profitability
from models.receipt_cost_alerts_model import compute_receipt_cost_alerts
from models.reorder_model import generate_reorder_suggestions
from models.return_pressure_model import compute_return_pressure
from models.sales_performance_model import score_salespersons
from models.slow_mover_model import compute_slow_movers
from models.stockout_risk_model import compute_stockout_risk
from models.supplier_performance_model import score_suppliers
from utils.ai_config import DEFAULT_FORECAST_DAYS, MODEL_VERSION

# ✅ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AismsAI")

app = Flask(__name__)
CORS(app)

# ✅ Database initialization and health check
try:
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("✅ Database connection established successfully.")
except SQLAlchemyError as e:
    logger.error(f"❌ Database connection failed: {e}")
except Exception as ex:
    logger.error(f"❌ Unexpected error during startup: {ex}")
else:
    logger.info("✅ All AI service tables ensured in database.")

logger.info("🚀 AISMS AI Flask service initialized successfully.")


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error("🔥 Unhandled Exception", exc_info=e)
    return {"error": str(e)}, 500


@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/v1/forecast", methods=["POST"])
def forecast():
    """
    Body:
    {
      "company_id": 1,
      "warehouse_id": 5,
      "days": 30,             # optional (default 30)
      "product_id": 12345     # optional; if missing, forecasts all SKUs in the warehouse
    }
    """
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    days = int(data.get("days", 30))
    product_id = data.get("product_id")

    if not company_id or not warehouse_id:
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    predictions = train_and_predict(company_id, warehouse_id, days, product_id)

    if not predictions:
        return jsonify({"status": "ok", "count": 0, "message": "No trainable SKUs found (insufficient history)."}), 200

    db = SessionLocal()
    try:
        stmt = insert(SkuDemandForecast).values(predictions)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                SkuDemandForecast.company_id,
                SkuDemandForecast.product_id,
                SkuDemandForecast.warehouse_id,
                SkuDemandForecast.forecast_date,
                SkuDemandForecast.model_version
            ],
            set_={
                "predicted_quantity": stmt.excluded.predicted_quantity,
                "generated_at": datetime.utcnow()
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(predictions)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/forecast/query", methods=["POST"])
def forecast_query():
    """
    Retrieve existing forecasts from sku_demand_forecast.

    Body:
    {
      "company_id": 1,
      "warehouse_id": 1,
      "product_id": 123,         # optional
      "start_date": "2025-10-01",# optional (YYYY-MM-DD)
      "end_date": "2025-11-30",  # optional (YYYY-MM-DD)
      "model_version": "v1.0"    # optional, defaults to MODEL_VERSION
    }
    """

    data = request.get_json() or {}

    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    product_id = data.get("product_id")
    start_date_str = data.get("start_date")
    end_date_str = data.get("end_date")
    model_version = data.get("model_version", MODEL_VERSION)

    if not company_id or not warehouse_id:
        return jsonify({
            "status": "error",
            "message": "company_id and warehouse_id are required"
        }), 400

    # Parse dates if provided
    start_date = None
    end_date = None
    try:
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str).date()
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str).date()
    except ValueError:
        return jsonify({
            "status": "error",
            "message": "start_date/end_date must be in YYYY-MM-DD format"
        }), 400

    db = SessionLocal()
    try:
        q = db.query(SkuDemandForecast).filter(
            SkuDemandForecast.company_id == company_id,
            SkuDemandForecast.warehouse_id == warehouse_id,
            SkuDemandForecast.model_version == model_version
        )

        if product_id is not None:
            q = q.filter(SkuDemandForecast.product_id == product_id)

        if start_date is not None:
            q = q.filter(SkuDemandForecast.forecast_date >= start_date)

        if end_date is not None:
            q = q.filter(SkuDemandForecast.forecast_date <= end_date)

        q = q.order_by(
            SkuDemandForecast.product_id,
            SkuDemandForecast.forecast_date
        )

        rows = q.all()

        items = []
        for r in rows:
            items.append({
                "company_id": r.company_id,
                "warehouse_id": r.warehouse_id,
                "product_id": r.product_id,
                "forecast_date": r.forecast_date.isoformat(),
                "predicted_quantity": float(r.predicted_quantity),
                "model_version": r.model_version,
            })

        return jsonify({
            "status": "success",
            "count": len(items),
            "items": items
        }), 200

    except Exception as e:
        db.rollback()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    finally:
        db.close()


@app.route("/api/v1/anomaly/sales", methods=["POST"])
def detect_anomaly_sales():
    """
    Body:
    {
      "company_id": 1,
      "warehouse_id": 5
    }
    """

    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not company_id or not warehouse_id:
        return jsonify({"status": "error",
                        "message": "company_id & warehouse_id required"}), 400

    results = detect_sales_anomalies(company_id, warehouse_id)

    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No anomalies found"}), 200

    db = SessionLocal()
    try:
        stmt = insert(SalesAnomalyEvent).values(results)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                SalesAnomalyEvent.sale_id,
                SalesAnomalyEvent.policy_code
            ],
            set_={
                "score": stmt.excluded.score,
                "level": stmt.excluded.level,
                "created_at": datetime.utcnow(),
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(results)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/anomaly/sales/<int:company_id>/<int:warehouse_id>",
           methods=["GET"])
def get_anomaly_sales(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(SalesAnomalyEvent)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(SalesAnomalyEvent.created_at.desc())
            .all()
        )
        return jsonify([
            {
                "sale_id": r.sale_id,
                "policy_code": r.policy_code,
                "score": float(r.score),
                "level": r.level,
                "created_at": r.created_at.isoformat()
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/reorders", methods=["POST"])
def compute_reorders():
    """
    Body:
    {
      "company_id": 1,
      "warehouse_id": 5,
      "service_level_z": 1.65,  // optional
      "lead_time_days": 7,      // optional
      "horizon_days": 30        // optional
    }
    """
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not company_id or not warehouse_id:
        return jsonify({
            "status": "error",
            "message": "company_id and warehouse_id are required"
        }), 400

    rows = generate_reorder_suggestions(
        company_id=company_id,
        warehouse_id=warehouse_id,
        service_level_z=data.get("service_level_z"),
        lead_time_days=data.get("lead_time_days"),
        horizon_days=int(data.get("horizon_days", 30))
    )

    if not rows:
        return jsonify({
            "status": "ok",
            "count": 0,
            "message": "No reorder suggestions (no forecast/stock data found)."
        }), 200

    db = SessionLocal()
    try:
        stmt = insert(ReorderSuggestion).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                ReorderSuggestion.company_id,
                ReorderSuggestion.warehouse_id,
                ReorderSuggestion.product_id,
                ReorderSuggestion.model_version
            ],
            set_={
                "reorder_point": stmt.excluded.reorder_point,
                "safety_stock": stmt.excluded.safety_stock,
                "suggested_qty": stmt.excluded.suggested_qty,
                "avg_daily_demand": stmt.excluded.avg_daily_demand,
                "generated_at": datetime.utcnow()
            }
        )

        db.execute(stmt)
        db.commit()

        return jsonify({"status": "success", "count": len(rows)}), 200

    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        db.close()


@app.route("/api/v1/reorders/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_reorders(company_id, warehouse_id):
    """
    Returns last saved reorder suggestions for a company + warehouse.
    """
    db = SessionLocal()
    try:
        q = (
            db.query(ReorderSuggestion)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(ReorderSuggestion.suggested_qty.desc())
        )

        rows = q.all()

        return jsonify([
            {
                "company_id": r.company_id,
                "warehouse_id": r.warehouse_id,
                "product_id": r.product_id,
                "reorder_point": float(r.reorder_point),
                "safety_stock": float(r.safety_stock),
                "suggested_qty": float(r.suggested_qty),
                "avg_daily_demand": float(r.avg_daily_demand) if r.avg_daily_demand is not None else None,
                "model_version": r.model_version,
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for r in rows
        ])

    finally:
        db.close()


@app.route("/api/v1/suppliers/score", methods=["POST"])
def compute_supplier_scores():
    data = request.get_json() or {}
    company_id = data.get("company_id")

    period_start = data.get("period_start")
    period_end = data.get("period_end")

    if not (company_id and period_start and period_end):
        return jsonify({"status": "error",
                        "message": "company_id, period_start, period_end required"}), 400

    start = datetime.strptime(period_start, "%Y-%m-%d").date()
    end = datetime.strptime(period_end, "%Y-%m-%d").date()

    results = score_suppliers(company_id, start, end)

    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No purchase activity in period"}), 200

    # -------------------------------
    # 🔥 FIX: convert numpy → Python
    # -------------------------------
    def convert_np(obj):
        if isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
            return float(obj)
        return obj

    clean_results = []
    for row in results:
        clean_results.append({k: convert_np(v) for k, v in row.items()})

    db = SessionLocal()
    try:
        stmt = insert(SupplierPerformanceScore).values(clean_results)

        stmt = stmt.on_conflict_do_update(
            index_elements=[
                SupplierPerformanceScore.supplier_id,
                SupplierPerformanceScore.company_id,
                SupplierPerformanceScore.period_start,
                SupplierPerformanceScore.period_end,
                SupplierPerformanceScore.model_version
            ],
            set_={
                "on_time_rate": stmt.excluded.on_time_rate,
                "accuracy_rate": stmt.excluded.accuracy_rate,
                "rejection_rate": stmt.excluded.rejection_rate,
                "cost_stability": stmt.excluded.cost_stability,
                "overall_score": stmt.excluded.overall_score,
                "generated_at": datetime.utcnow()
            }
        )

        db.execute(stmt)
        db.commit()

        return jsonify({"status": "success", "count": len(clean_results)})

    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        db.close()


@app.route("/api/v1/suppliers/score/<int:company_id>", methods=["GET"])
def get_supplier_scores(company_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(SupplierPerformanceScore)
            .filter_by(company_id=company_id)
            .order_by(SupplierPerformanceScore.overall_score.desc())
            .all()
        )
        return jsonify([
            {
                "supplier_id": r.supplier_id,
                "period_start": r.period_start.strftime("%Y-%m-%d"),
                "period_end": r.period_end.strftime("%Y-%m-%d"),
                "on_time_rate": float(r.on_time_rate),
                "accuracy_rate": float(r.accuracy_rate),
                "rejection_rate": float(r.rejection_rate),
                "cost_stability": float(r.cost_stability),
                "overall_score": float(r.overall_score),
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/pricing/recommend", methods=["POST"])
def pricing_recommend():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = recommend_prices(company_id, warehouse_id)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No valid sales/forecast data"}), 200

    db = SessionLocal()
    try:
        stmt = insert(DynamicPricingRecommendation).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                DynamicPricingRecommendation.company_id,
                DynamicPricingRecommendation.warehouse_id,
                DynamicPricingRecommendation.product_id,
                DynamicPricingRecommendation.model_version
            ],
            set_={
                "current_price": stmt.excluded.current_price,
                "suggested_price": stmt.excluded.suggested_price,
                "price_change_pct": stmt.excluded.price_change_pct,
                "expected_demand_change": stmt.excluded.expected_demand_change,
                "confidence_level": stmt.excluded.confidence_level,
                "rationale": stmt.excluded.rationale,
                "generated_at": datetime.utcnow()
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/pricing/recommend/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def pricing_recommend_get(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(DynamicPricingRecommendation)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(DynamicPricingRecommendation.price_change_pct.desc())
            .all()
        )
        return jsonify([
            {
                "product_id": r.product_id,
                "current_price": float(r.current_price),
                "suggested_price": float(r.suggested_price),
                "price_change_pct": float(r.price_change_pct),
                "expected_demand_change": float(r.expected_demand_change),
                "confidence_level": float(r.confidence_level),
                "rationale": r.rationale,
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/customers/segments", methods=["POST"])
def compute_customer_segments():
    """
    Body:
    {
      "company_id": 1,
      "customer_column": "customer_id",   // REQUIRED name of column in sales that identifies the customer
      "days_window": 365,                 // optional (default 365)
      "warehouse_id": 5                   // optional: restrict to a warehouse
    }
    """
    data = request.get_json() or {}
    company_id = data.get("company_id")
    customer_column = data.get("customer_column")
    days_window = int(data.get("days_window", 365))
    warehouse_id = data.get("warehouse_id")

    if not company_id or not customer_column:
        return jsonify({"status": "error",
                        "message": "company_id and customer_column are required"}), 400

    rows = calculate_customer_segments(
        company_id=company_id,
        customer_column=customer_column,
        days_window=days_window,
        warehouse_id=warehouse_id
    )

    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No eligible sales found."}), 200

    db = SessionLocal()
    try:
        stmt = insert(CustomerSegment).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                CustomerSegment.company_id,
                CustomerSegment.customer_id,
                CustomerSegment.model_version
            ],
            set_={
                "recency_days": stmt.excluded.recency_days,
                "frequency": stmt.excluded.frequency,
                "monetary_value": stmt.excluded.monetary_value,
                "clv_score": stmt.excluded.clv_score,
                "segment": stmt.excluded.segment,
                "generated_at": datetime.utcnow()
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/customers/segments/<int:company_id>", methods=["GET"])
def get_customer_segments(company_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(CustomerSegment)
            .filter_by(company_id=company_id)
            .order_by(CustomerSegment.clv_score.desc())
            .all()
        )
        return jsonify([
            {
                "customer_id": r.customer_id,
                "recency_days": int(r.recency_days),
                "frequency": int(r.frequency),
                "monetary_value": float(r.monetary_value),
                "clv_score": float(r.clv_score),
                "segment": r.segment,
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/salespersons/score", methods=["POST"])
def salespersons_score():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")  # optional
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    if not (company_id and start_date and end_date):
        return jsonify({"status": "error", "message": "company_id, start_date, end_date required"}), 400

    rows = score_salespersons(company_id, warehouse_id, start_date, end_date)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No data"}), 200

    db = SessionLocal()
    try:
        stmt = insert(SalesPerformanceScore).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_sales_performance_company_wh_person_model",
            set_={
                "total_sales": stmt.excluded.total_sales,
                "total_orders": stmt.excluded.total_orders,
                "avg_order_value": stmt.excluded.avg_order_value,
                "growth_rate": stmt.excluded.growth_rate,
                "performance_trend": stmt.excluded.performance_trend,
                "performance_score": stmt.excluded.performance_score,
                "generated_at": datetime.utcnow()
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/salespersons/score/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def salespersons_score_get(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(SalesPerformanceScore)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(SalesPerformanceScore.performance_score.desc())
            .all()
        )
        return jsonify([
            {
                "salesperson_id": r.salesperson_id,
                "total_sales": float(r.total_sales),
                "total_orders": int(r.total_orders),
                "avg_order_value": float(r.avg_order_value),
                "growth_rate": float(r.growth_rate),
                "performance_trend": r.performance_trend,
                "performance_score": float(r.performance_score),
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/profitability/forecast", methods=["POST"])
def profitability_forecast():
    """
    Body:
    {
      "company_id": 1,
      "warehouse_id": 5
    }
    """
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = compute_monthly_profitability(company_id, warehouse_id)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No data"}), 200

    db = SessionLocal()
    try:
        stmt = insert(ProfitabilityForecast).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                ProfitabilityForecast.company_id,
                ProfitabilityForecast.warehouse_id,
                ProfitabilityForecast.month,
                ProfitabilityForecast.model_version
            ],
            set_={
                "total_revenue": stmt.excluded.total_revenue,
                "total_cogs": stmt.excluded.total_cogs,
                "total_expenses": stmt.excluded.total_expenses,
                "net_profit": stmt.excluded.net_profit,
                "profit_margin": stmt.excluded.profit_margin,
                "trend": stmt.excluded.trend,
                "forecast_profit": stmt.excluded.forecast_profit,
                "generated_at": datetime.utcnow()
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/profitability/forecast/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def profitability_forecast_get(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(ProfitabilityForecast)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(ProfitabilityForecast.month.asc())
            .all()
        )
        return jsonify([
            {
                "month": r.month.strftime("%Y-%m"),
                "total_revenue": float(r.total_revenue),
                "total_cogs": float(r.total_cogs),
                "total_expenses": float(r.total_expenses),
                "net_profit": float(r.net_profit),
                "profit_margin": float(r.profit_margin),
                "trend": r.trend,
                "forecast_profit": float(r.forecast_profit) if r.forecast_profit is not None else None,
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/inventory/optimize", methods=["POST"])
def inventory_optimize():
    """
    Body:
    {
      "company_id": 1,
      "warehouse_id": 5,
      "service_level_z": 1.65,      // optional (default Z95)
      "lead_time_days": 7,          // optional (default LEAD_TIME_DAYS)
      "horizon_days": 30,           // optional (default DEFAULT_FORECAST_DAYS)
      "lookback_days": 90           // optional (default 90)
    }
    """
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = optimize_inventory(
        company_id=company_id,
        warehouse_id=warehouse_id,
        service_level_z=data.get("service_level_z"),
        lead_time_days=data.get("lead_time_days"),
        horizon_days=int(data.get("horizon_days", DEFAULT_FORECAST_DAYS)),
        lookback_days=int(data.get("lookback_days", 90))
    )

    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No data (no stock or demand found)"}), 200

    db = SessionLocal()
    try:
        stmt = insert(InventoryOptimization).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                InventoryOptimization.company_id,
                InventoryOptimization.warehouse_id,
                InventoryOptimization.product_id,
                InventoryOptimization.model_version
            ],
            set_={
                "current_stock": stmt.excluded.current_stock,
                "avg_daily_demand": stmt.excluded.avg_daily_demand,
                "safety_stock": stmt.excluded.safety_stock,
                "optimal_stock_level": stmt.excluded.optimal_stock_level,
                "stock_status": stmt.excluded.stock_status,
                "inventory_health_score": stmt.excluded.inventory_health_score,
                "forecast_horizon_days": stmt.excluded.forecast_horizon_days,
                "generated_at": datetime.utcnow()
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/inventory/optimize/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def inventory_optimize_get(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(InventoryOptimization)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(InventoryOptimization.inventory_health_score.asc())
            .all()
        )
        return jsonify([
            {
                "product_id": r.product_id,
                "current_stock": float(r.current_stock),
                "avg_daily_demand": float(r.avg_daily_demand),
                "safety_stock": float(r.safety_stock),
                "optimal_stock_level": float(r.optimal_stock_level),
                "stock_status": r.stock_status,
                "inventory_health_score": float(r.inventory_health_score),
                "forecast_horizon_days": int(r.forecast_horizon_days),
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


def to_float(value, default=0.0):
    """
    Safely cast values to float.
    Handles None, Decimal, numpy.nan, strings, etc.
    """
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


@app.route("/api/v1/cashflow/forecast", methods=["POST"])
def cashflow_forecast():
    """
    {
      "company_id": 1,
      "warehouse_id": 5
    }
    """
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = compute_cashflow(company_id, warehouse_id)
    for r in rows:
        r["cash_inflows"] = to_float(r.get("cash_inflows"))
        r["cash_outflows"] = to_float(r.get("cash_outflows"))
        r["net_cashflow"] = to_float(r.get("net_cashflow"))
        r["cash_balance"] = to_float(r.get("cash_balance"))
        r["cash_health_score"] = to_float(r.get("cash_health_score"))
        r["forecasted_next_balance"] = to_float(r.get("forecasted_next_balance"))

    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No data"}), 200

    db = SessionLocal()
    try:
        stmt = insert(CashflowForecast).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                CashflowForecast.company_id,
                CashflowForecast.warehouse_id,
                CashflowForecast.month,
                CashflowForecast.model_version
            ],
            set_={
                "cash_inflows": stmt.excluded.cash_inflows,
                "cash_outflows": stmt.excluded.cash_outflows,
                "net_cashflow": stmt.excluded.net_cashflow,
                "cash_balance": stmt.excluded.cash_balance,
                "cash_health_score": stmt.excluded.cash_health_score,
                "risk_level": stmt.excluded.risk_level,
                "forecasted_next_balance": stmt.excluded.forecasted_next_balance,
                "generated_at": datetime.utcnow()
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/cashflow/forecast/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def cashflow_forecast_get(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(CashflowForecast)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(CashflowForecast.month.asc())
            .all()
        )
        return jsonify([
            {
                "month": r.month.strftime("%Y-%m"),
                "cash_inflows": float(r.cash_inflows),
                "cash_outflows": float(r.cash_outflows),
                "net_cashflow": float(r.net_cashflow),
                "cash_balance": float(r.cash_balance),
                "cash_health_score": float(r.cash_health_score),
                "risk_level": r.risk_level,
                "forecasted_next_balance": float(r.forecasted_next_balance) if r.forecasted_next_balance else None,
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


# -------------------- 11) Stock Movement Anomaly --------------------
@app.route("/api/v1/anomaly/stock-movements/train", methods=["POST"])
def train_stock_movement_anomaly():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    days_window = int(data.get("days_window", 90))
    limit = int(data.get("limit", 200))
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = detect_stock_movement_anomalies(int(company_id), int(warehouse_id), days_window, limit)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No anomalies"}), 200

    db = SessionLocal()
    try:
        stmt = insert(StockMovementAnomalyEvent).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_stock_mv_policy_model",
            set_={"score": stmt.excluded.score, "level": stmt.excluded.level, "reason": stmt.excluded.reason,
                  "created_at": datetime.utcnow()}
        )
        db.execute(stmt);
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback();
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/anomaly/stock-movements/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_stock_movement_anomaly(company_id, warehouse_id):
    limit = int(request.args.get("limit", 200))
    db = SessionLocal()
    try:
        rows = (db.query(StockMovementAnomalyEvent)
                .filter_by(company_id=company_id, warehouse_id=warehouse_id)
                .order_by(StockMovementAnomalyEvent.created_at.desc())
                .limit(limit).all())
        return jsonify([{
            "stock_movement_id": r.stock_movement_id,
            "product_id": r.product_id,
            "movement_date": r.movement_date.isoformat(),
            "movement_type_id": r.movement_type_id,
            "quantity": float(r.quantity),
            "unit_cost": float(r.unit_cost) if r.unit_cost is not None else None,
            "policy_code": r.policy_code,
            "score": float(r.score),
            "level": r.level,
            "reason": r.reason,
            "created_at": r.created_at.isoformat()
        } for r in rows])
    finally:
        db.close()


# -------------------- 12) Stocktake Variance Risk --------------------
@app.route("/api/v1/stocktake/variance-risk/train", methods=["POST"])
def train_stocktake_variance_risk():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    lookback_days = int(data.get("lookback_days", 180))
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = compute_stocktake_variance_risk(int(company_id), int(warehouse_id), lookback_days)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No data"}), 200

    db = SessionLocal()
    try:
        stmt = insert(StocktakeVarianceRisk).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_stocktake_risk_company_wh_prod_model",
            set_={"lookback_days": stmt.excluded.lookback_days, "risk_score": stmt.excluded.risk_score,
                  "risk_level": stmt.excluded.risk_level, "drivers": stmt.excluded.drivers,
                  "generated_at": datetime.utcnow()}
        )
        db.execute(stmt);
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback();
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/stocktake/variance-risk/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_stocktake_variance_risk(company_id, warehouse_id):
    limit = int(request.args.get("limit", 500))
    db = SessionLocal()
    try:
        rows = (db.query(StocktakeVarianceRisk)
                .filter_by(company_id=company_id, warehouse_id=warehouse_id)
                .order_by(StocktakeVarianceRisk.risk_score.desc())
                .limit(limit).all())
        return jsonify([{
            "product_id": r.product_id,
            "risk_score": float(r.risk_score),
            "risk_level": r.risk_level,
            "drivers": r.drivers,
            "generated_at": r.generated_at.isoformat()
        } for r in rows])
    finally:
        db.close()


# -------------------- 13) Transfer Recommendations --------------------
@app.route("/api/v1/transfers/recommend/train", methods=["POST"])
def train_transfer_recommendations():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    days_window = int(data.get("days_window", 60))
    horizon_days = int(data.get("horizon_days", 14))
    min_suggest_qty = float(data.get("min_suggest_qty", 1))
    if not company_id:
        return jsonify({"status": "error", "message": "company_id required"}), 400

    rows = recommend_transfers(int(company_id), days_window, horizon_days, min_suggest_qty)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No recommendations"}), 200

    db = SessionLocal()
    try:
        stmt = insert(TransferRecommendation).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_transfer_rec_company_prod_from_to_model",
            set_={"suggested_qty": stmt.excluded.suggested_qty, "from_on_hand": stmt.excluded.from_on_hand,
                  "to_on_hand": stmt.excluded.to_on_hand, "from_days_cover": stmt.excluded.from_days_cover,
                  "to_days_cover": stmt.excluded.to_days_cover, "rationale": stmt.excluded.rationale,
                  "confidence": stmt.excluded.confidence, "generated_at": datetime.utcnow()}
        )
        db.execute(stmt);
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback();
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/transfers/recommend/<int:company_id>", methods=["GET"])
def get_transfer_recommendations(company_id):
    limit = int(request.args.get("limit", 500))
    db = SessionLocal()
    try:
        rows = (db.query(TransferRecommendation)
                .filter_by(company_id=company_id)
                .order_by(TransferRecommendation.confidence.desc())
                .limit(limit).all())
        return jsonify([{
            "product_id": r.product_id,
            "from_warehouse_id": r.from_warehouse_id,
            "to_warehouse_id": r.to_warehouse_id,
            "suggested_qty": float(r.suggested_qty),
            "from_on_hand": float(r.from_on_hand),
            "to_on_hand": float(r.to_on_hand),
            "from_days_cover": float(r.from_days_cover) if r.from_days_cover is not None else None,
            "to_days_cover": float(r.to_days_cover) if r.to_days_cover is not None else None,
            "confidence": float(r.confidence),
            "rationale": r.rationale,
            "generated_at": r.generated_at.isoformat()
        } for r in rows])
    finally:
        db.close()


# -------------------- 14) Stockout Risk --------------------
@app.route("/api/v1/inventory/stockout-risk/train", methods=["POST"])
def train_stockout_risk():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    horizon_days = int(data.get("horizon_days", 14))
    lookback_days = int(data.get("lookback_days", 60))
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = compute_stockout_risk(int(company_id), int(warehouse_id), horizon_days, lookback_days)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No data"}), 200

    db = SessionLocal()
    try:
        stmt = insert(StockoutRisk).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_stockout_company_wh_prod_horizon_model",
            set_={"lookback_days": stmt.excluded.lookback_days, "on_hand": stmt.excluded.on_hand,
                  "avg_daily_demand": stmt.excluded.avg_daily_demand,
                  "std_daily_demand": stmt.excluded.std_daily_demand,
                  "expected_demand": stmt.excluded.expected_demand,
                  "stockout_probability": stmt.excluded.stockout_probability,
                  "expected_stockout_date": stmt.excluded.expected_stockout_date,
                  "recommended_qty": stmt.excluded.recommended_qty,
                  "risk_level": stmt.excluded.risk_level, "generated_at": datetime.utcnow()}
        )
        db.execute(stmt);
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback();
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/inventory/stockout-risk/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_stockout_risk(company_id, warehouse_id):
    horizon_days = int(request.args.get("horizon_days", 14))
    limit = int(request.args.get("limit", 500))
    db = SessionLocal()
    try:
        rows = (db.query(StockoutRisk)
                .filter_by(company_id=company_id, warehouse_id=warehouse_id, horizon_days=horizon_days)
                .order_by(StockoutRisk.stockout_probability.desc())
                .limit(limit).all())
        return jsonify([{
            "product_id": r.product_id,
            "on_hand": float(r.on_hand),
            "avg_daily_demand": float(r.avg_daily_demand),
            "expected_demand": float(r.expected_demand),
            "stockout_probability": float(r.stockout_probability),
            "expected_stockout_date": r.expected_stockout_date.isoformat() if r.expected_stockout_date else None,
            "recommended_qty": float(r.recommended_qty),
            "risk_level": r.risk_level,
            "generated_at": r.generated_at.isoformat()
        } for r in rows])
    finally:
        db.close()


# -------------------- 15) Slow Movers --------------------
@app.route("/api/v1/inventory/slow-movers/train", methods=["POST"])
def train_slow_movers():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    lookback_days = int(data.get("lookback_days", 120))
    min_on_hand = float(data.get("min_on_hand", 1))
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = compute_slow_movers(int(company_id), int(warehouse_id), lookback_days, min_on_hand)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No slow movers"}), 200

    db = SessionLocal()
    try:
        stmt = insert(SlowMoverRisk).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_slow_mover_company_wh_prod_model",
            set_={"lookback_days": stmt.excluded.lookback_days, "on_hand": stmt.excluded.on_hand,
                  "avg_daily_sales": stmt.excluded.avg_daily_sales,
                  "days_since_last_sale": stmt.excluded.days_since_last_sale,
                  "days_cover": stmt.excluded.days_cover, "slow_mover_score": stmt.excluded.slow_mover_score,
                  "risk_level": stmt.excluded.risk_level, "recommended_action": stmt.excluded.recommended_action,
                  "rationale": stmt.excluded.rationale, "generated_at": datetime.utcnow()}
        )
        db.execute(stmt);
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback();
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/inventory/slow-movers/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_slow_movers(company_id, warehouse_id):
    limit = int(request.args.get("limit", 500))
    db = SessionLocal()
    try:
        rows = (db.query(SlowMoverRisk)
                .filter_by(company_id=company_id, warehouse_id=warehouse_id)
                .order_by(SlowMoverRisk.slow_mover_score.desc())
                .limit(limit).all())
        return jsonify([{
            "product_id": r.product_id,
            "on_hand": float(r.on_hand),
            "avg_daily_sales": float(r.avg_daily_sales),
            "days_since_last_sale": int(r.days_since_last_sale),
            "days_cover": float(r.days_cover),
            "slow_mover_score": float(r.slow_mover_score),
            "risk_level": r.risk_level,
            "recommended_action": r.recommended_action,
            "rationale": r.rationale,
            "generated_at": r.generated_at.isoformat()
        } for r in rows])
    finally:
        db.close()


# -------------------- 16) Return Pressure --------------------
@app.route("/api/v1/returns/pressure/train", methods=["POST"])
def train_return_pressure():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    period_days = int(data.get("period_days", 60))
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = compute_return_pressure(int(company_id), int(warehouse_id), period_days)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No data"}), 200

    db = SessionLocal()
    try:
        stmt = insert(ReturnPressureIndex).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_return_pressure_company_wh_prod_period_model",
            set_={"sale_out_qty": stmt.excluded.sale_out_qty, "return_in_qty": stmt.excluded.return_in_qty,
                  "return_rate": stmt.excluded.return_rate, "score": stmt.excluded.score,
                  "level": stmt.excluded.level, "trend": stmt.excluded.trend,
                  "generated_at": datetime.utcnow()}
        )
        db.execute(stmt);
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback();
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/returns/pressure/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_return_pressure(company_id, warehouse_id):
    period_days = int(request.args.get("period_days", 60))
    limit = int(request.args.get("limit", 500))
    db = SessionLocal()
    try:
        rows = (db.query(ReturnPressureIndex)
                .filter_by(company_id=company_id, warehouse_id=warehouse_id, period_days=period_days)
                .order_by(ReturnPressureIndex.score.desc())
                .limit(limit).all())
        return jsonify([{
            "product_id": r.product_id,
            "sale_out_qty": float(r.sale_out_qty),
            "return_in_qty": float(r.return_in_qty),
            "return_rate": float(r.return_rate),
            "score": float(r.score),
            "level": r.level,
            "trend": r.trend,
            "generated_at": r.generated_at.isoformat()
        } for r in rows])
    finally:
        db.close()


# -------------------- 17) Receipt Cost Alerts --------------------
@app.route("/api/v1/cost/receipt-alerts/train", methods=["POST"])
def train_receipt_cost_alerts():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    lookback_days = int(data.get("lookback_days", 180))
    short_window_days = int(data.get("short_window_days", 7))
    long_window_days = int(data.get("long_window_days", 30))
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = compute_receipt_cost_alerts(int(company_id), int(warehouse_id), lookback_days, short_window_days,
                                       long_window_days)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No alerts"}), 200

    db = SessionLocal()
    try:
        stmt = insert(ReceiptCostAlert).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_receipt_cost_company_wh_prod_model",
            set_={"avg_cost_short": stmt.excluded.avg_cost_short, "avg_cost_long": stmt.excluded.avg_cost_long,
                  "cost_change_pct": stmt.excluded.cost_change_pct,
                  "volatility_lookback": stmt.excluded.volatility_lookback,
                  "level": stmt.excluded.level, "message": stmt.excluded.message,
                  "generated_at": datetime.utcnow()}
        )
        db.execute(stmt);
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback();
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/cost/receipt-alerts/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_receipt_cost_alerts(company_id, warehouse_id):
    limit = int(request.args.get("limit", 500))
    db = SessionLocal()
    try:
        rows = (db.query(ReceiptCostAlert)
                .filter_by(company_id=company_id, warehouse_id=warehouse_id)
                .order_by(ReceiptCostAlert.generated_at.desc())
                .limit(limit).all())
        return jsonify([{
            "product_id": r.product_id,
            "avg_cost_short": float(r.avg_cost_short),
            "avg_cost_long": float(r.avg_cost_long),
            "cost_change_pct": float(r.cost_change_pct),
            "volatility_lookback": float(r.volatility_lookback),
            "level": r.level,
            "message": r.message,
            "generated_at": r.generated_at.isoformat()
        } for r in rows])
    finally:
        db.close()


# -------------------- 18) Forecast Metrics & Drift --------------------
@app.route("/api/v1/models/forecast-metrics/train", methods=["POST"])
def train_forecast_metrics():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    lookback_days = int(data.get("lookback_days", 180))
    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id & warehouse_id required"}), 400

    rows = compute_forecast_metrics(int(company_id), int(warehouse_id), lookback_days)
    if not rows:
        return jsonify({"status": "ok", "count": 0, "message": "No comparable data"}), 200

    db = SessionLocal()
    try:
        stmt = insert(ForecastModelMetric).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_forecast_metrics_company_wh_prod_model",
            set_={"lookback_days": stmt.excluded.lookback_days, "mae": stmt.excluded.mae, "mape": stmt.excluded.mape,
                  "bias": stmt.excluded.bias, "drift_score": stmt.excluded.drift_score,
                  "drift_level": stmt.excluded.drift_level,
                  "notes": stmt.excluded.notes, "generated_at": datetime.utcnow()}
        )
        db.execute(stmt);
        db.commit()
        return jsonify({"status": "success", "count": len(rows)})
    except Exception as e:
        db.rollback();
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/models/forecast-metrics/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_forecast_metrics(company_id, warehouse_id):
    limit = int(request.args.get("limit", 500))
    db = SessionLocal()
    try:
        rows = (db.query(ForecastModelMetric)
                .filter_by(company_id=company_id, warehouse_id=warehouse_id)
                .order_by(ForecastModelMetric.drift_score.desc())
                .limit(limit).all())
        return jsonify([{
            "product_id": r.product_id,
            "mae": float(r.mae),
            "mape": float(r.mape),
            "bias": float(r.bias),
            "drift_score": float(r.drift_score),
            "drift_level": r.drift_level,
            "notes": r.notes,
            "generated_at": r.generated_at.isoformat()
        } for r in rows])
    finally:
        db.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app.run(host="0.0.0.0", port=5001)
