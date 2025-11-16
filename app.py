import logging
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from db.connection import engine, SessionLocal
from db.models import Base, SkuDemandForecast, SalesAnomalyEvent, ReorderSuggestion, SupplierPerformanceScore, \
    CustomerSegment, DynamicPricingRecommendation, SalesPerformanceScore, ProfitabilityForecast, InventoryOptimization, \
    CashflowForecast
from models.anomaly_model import detect_sales_anomalies
from models.cashflow_forecast_model import compute_cashflow_forecast
from models.customer_segmentation_model import calculate_customer_segments
from models.dynamic_pricing_model import recommend_prices
from models.forecast_model import train_and_predict
from models.inventory_optimization_model import optimize_inventory
from models.profitability_model import compute_monthly_profitability
from models.reorder_model import generate_reorder_suggestions
from models.sales_performance_model import score_salespersons
from models.supplier_performance_model import score_suppliers
from utils.ai_config import DEFAULT_FORECAST_DAYS

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

logger.info("🚀 AismsAI Flask service initialized successfully.")


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
      "service_level_z": 1.65,   # optional; default Z95
      "lead_time_days": 7,       # optional; default LEAD_TIME_DAYS
      "horizon_days": 30         # optional; forecast horizon window to summarize demand variability
    }
    """
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    service_level_z = data.get("service_level_z")  # optional
    lead_time_days = data.get("lead_time_days")  # optional
    horizon_days = int(data.get("horizon_days", 30))

    if not company_id or not warehouse_id:
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    results = generate_reorder_suggestions(
        company_id=company_id,
        warehouse_id=warehouse_id,
        service_level_z=service_level_z,
        lead_time_days=lead_time_days,
        horizon_days=horizon_days
    )

    if not results:
        return jsonify(
            {"status": "ok", "count": 0, "message": "No forecasts available; generate /api/v1/forecast first."}), 200

    db = SessionLocal()
    try:
        stmt = insert(ReorderSuggestion).values(results)
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
                "generated_at": datetime.utcnow()
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


@app.route("/api/v1/reorders/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_reorders(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(ReorderSuggestion)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(ReorderSuggestion.suggested_qty.desc())
            .all()
        )
        return jsonify([
            {
                "product_id": r.product_id,
                "reorder_point": float(r.reorder_point),
                "safety_stock": float(r.safety_stock),
                "suggested_qty": float(r.suggested_qty),
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/suppliers/score", methods=["POST"])
def compute_supplier_scores():
    """
    Body:
    {
      "company_id": 1,
      "period_start": "2025-01-01",
      "period_end": "2025-01-31"
    }
    """
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

    db = SessionLocal()
    try:
        stmt = insert(SupplierPerformanceScore).values(results)
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
        return jsonify({"status": "success", "count": len(results)})
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
            index_elements=[
                SalesPerformanceScore.company_id,
                SalesPerformanceScore.warehouse_id,
                SalesPerformanceScore.salesperson_id,
                SalesPerformanceScore.model_version
            ],
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

    rows = compute_cashflow_forecast(company_id, warehouse_id)
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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app.run(host="0.0.0.0", port=5001)
