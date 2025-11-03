import uuid
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy.dialects.postgresql import insert

from db.connection import engine, SessionLocal
from db.models import Base, SkuDemandForecast, SalesAnomalyEvent, ReorderSuggestion, SupplierPerformanceScore, \
    DynamicPricingRecommendation, CustomerSegment, SalesPerformanceScore, ProfitabilityForecast, InventoryOptimization, \
    CashflowForecast
from models.anomaly_model import detect_sales_anomalies
from models.cashflow_model import compute_cashflow
from models.customer_model import calculate_customer_segments
from models.forecast_model import train_and_predict
from models.inventory_model import optimize_inventory
from models.pricing_model import generate_pricing_recommendations
from models.profitability_model import calculate_profitability
from models.sales_performance_model import calculate_sales_performance
from models.supplier_model import calculate_supplier_performance

app = Flask(__name__)
CORS(app)

Base.metadata.create_all(bind=engine)


# 🔹 Automatically clean up sessions after each request
@app.teardown_appcontext
def shutdown_session(exception=None):
    SessionLocal.remove()


@app.route("/api/v1/forecast", methods=["POST"])
def forecast():
    data = request.get_json()
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")
    product_id = data.get("product_id")
    days = data.get("days", 30)

    if not company_id or not warehouse_id:
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    predictions = train_and_predict(company_id, warehouse_id, days)

    db = SessionLocal()
    try:
        rows = []
        for p in predictions:
            rows.append({
                "id": str(uuid.uuid4()),
                "product_id": product_id or 0,
                "warehouse_id": warehouse_id,
                "forecast_date": p["forecast_date"],
                "predicted_quantity": p["predicted_quantity"],
                "model_version": "v1.0",
                "company_id": company_id,
                "generated_at": datetime.utcnow(),
            })

        if rows:
            stmt = insert(SkuDemandForecast).values(rows)
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

        return jsonify({"status": "success", "count": len(rows)})
    finally:
        db.close()


@app.route("/api/v1/forecast/<int:company_id>", methods=["GET"])
def get_forecasts(company_id):
    db = SessionLocal()
    try:
        results = db.query(SkuDemandForecast).filter_by(company_id=company_id).order_by(
            SkuDemandForecast.forecast_date).all()
        return jsonify([{
            "product_id": f.product_id,
            "warehouse_id": f.warehouse_id,
            "forecast_date": f.forecast_date.strftime("%Y-%m-%d"),
            "predicted_quantity": float(f.predicted_quantity),
        } for f in results])
    finally:
        db.close()


@app.route("/api/v1/anomalies", methods=["POST"])
def detect_anomalies():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not company_id or not warehouse_id:
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    anomalies = detect_sales_anomalies(company_id, warehouse_id)
    if not anomalies:
        return jsonify({"status": "ok", "count": 0, "message": "No anomalies found."})

    db = SessionLocal()
    try:
        stmt = insert(SalesAnomalyEvent).values(anomalies)
        stmt = stmt.on_conflict_do_update(
            index_elements=[SalesAnomalyEvent.sale_id, SalesAnomalyEvent.policy_code],
            set_={
                "score": stmt.excluded.score,
                "level": stmt.excluded.level,
                "warehouse_id": stmt.excluded.warehouse_id
            }
        )
        db.execute(stmt)
        db.commit()
        return jsonify({"status": "success", "count": len(anomalies)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/anomalies/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_anomalies(company_id, warehouse_id):
    db = SessionLocal()
    try:
        results = (
            db.query(SalesAnomalyEvent)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(SalesAnomalyEvent.score.desc())
            .all()
        )
        return jsonify([
            {
                "sale_id": a.sale_id,
                "warehouse_id": a.warehouse_id,
                "policy_code": a.policy_code,
                "score": float(a.score),
                "level": a.level,
                "created_at": a.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for a in results
        ])
    finally:
        db.close()


@app.route("/api/v1/reorders", methods=["POST"])
def compute_reorders():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not company_id or not warehouse_id:
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    results = generate_reorder_suggestions(company_id, warehouse_id)
    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No suggestions generated."})

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


@app.route("/api/v1/supplier-scores", methods=["POST"])
def compute_supplier_scores():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not (company_id and start_date and end_date):
        return jsonify({"status": "error", "message": "company_id, start_date, end_date are required"}), 400

    scores = calculate_supplier_performance(company_id, start_date, end_date)
    if not scores:
        return jsonify({"status": "ok", "count": 0, "message": "No data for given period."})

    db = SessionLocal()
    try:
        stmt = insert(SupplierPerformanceScore).values(scores)
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
        return jsonify({"status": "success", "count": len(scores)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/supplier-scores/<int:company_id>", methods=["GET"])
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
                "on_time_rate": float(r.on_time_rate),
                "accuracy_rate": float(r.accuracy_rate),
                "rejection_rate": float(r.rejection_rate),
                "cost_stability": float(r.cost_stability),
                "overall_score": float(r.overall_score),
                "period_start": r.period_start.strftime("%Y-%m-%d"),
                "period_end": r.period_end.strftime("%Y-%m-%d"),
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/pricing", methods=["POST"])
def compute_pricing():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not company_id or not warehouse_id:
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    results = generate_pricing_recommendations(company_id, warehouse_id)
    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No pricing recommendations generated."})

    db = SessionLocal()
    try:
        stmt = insert(DynamicPricingRecommendation).values(results)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                DynamicPricingRecommendation.company_id,
                DynamicPricingRecommendation.warehouse_id,
                DynamicPricingRecommendation.product_id,
                DynamicPricingRecommendation.model_version
            ],
            set_={
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
        return jsonify({"status": "success", "count": len(results)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/pricing/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_pricing(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(DynamicPricingRecommendation)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(DynamicPricingRecommendation.confidence_level.desc())
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
    data = request.get_json() or {}
    company_id = data.get("company_id")

    if not company_id:
        return jsonify({"status": "error", "message": "company_id is required"}), 400

    results = calculate_customer_segments(company_id)
    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No customers found."})

    db = SessionLocal()
    try:
        stmt = insert(CustomerSegment).values(results)
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
        return jsonify({"status": "success", "count": len(results)})
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
                "recency_days": r.recency_days,
                "frequency": r.frequency,
                "monetary_value": float(r.monetary_value),
                "clv_score": float(r.clv_score),
                "segment": r.segment,
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/sales-performance", methods=["POST"])
def compute_sales_performance():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    results = calculate_sales_performance(company_id, warehouse_id)
    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No sales performance data found."})

    db = SessionLocal()
    try:
        stmt = insert(SalesPerformanceScore).values(results)
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
        return jsonify({"status": "success", "count": len(results)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/sales-performance/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_sales_performance(company_id, warehouse_id):
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


@app.route("/api/v1/profitability", methods=["POST"])
def compute_profitability():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not company_id or not warehouse_id:
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    results = calculate_profitability(company_id, warehouse_id)
    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No profitability data found."})

    db = SessionLocal()
    try:
        stmt = insert(ProfitabilityForecast).values(results)
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
        return jsonify({"status": "success", "count": len(results)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/profitability/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_profitability(company_id, warehouse_id):
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
                "forecast_profit": float(r.forecast_profit) if r.forecast_profit else None,
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/inventory/optimize", methods=["POST"])
def compute_inventory_optimization():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not company_id or not warehouse_id:
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    results = optimize_inventory(company_id, warehouse_id)
    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No inventory data found."})

    db = SessionLocal()
    try:
        stmt = insert(InventoryOptimization).values(results)
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


@app.route("/api/v1/inventory/optimize/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_inventory_optimization(company_id, warehouse_id):
    db = SessionLocal()
    try:
        rows = (
            db.query(InventoryOptimization)
            .filter_by(company_id=company_id, warehouse_id=warehouse_id)
            .order_by(InventoryOptimization.inventory_health_score.desc())
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
                "generated_at": r.generated_at.strftime("%Y-%m-%d %H:%M:%S")
            } for r in rows
        ])
    finally:
        db.close()


@app.route("/api/v1/cashflow", methods=["POST"])
def compute_cashflow_route():
    data = request.get_json() or {}
    company_id = data.get("company_id")
    warehouse_id = data.get("warehouse_id")

    if not (company_id and warehouse_id):
        return jsonify({"status": "error", "message": "company_id and warehouse_id are required"}), 400

    results = compute_cashflow(company_id, warehouse_id)
    if not results:
        return jsonify({"status": "ok", "count": 0, "message": "No cashflow data found."})

    db = SessionLocal()
    try:
        stmt = insert(CashflowForecast).values(results)
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
        return jsonify({"status": "success", "count": len(results)})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@app.route("/api/v1/cashflow/<int:company_id>/<int:warehouse_id>", methods=["GET"])
def get_cashflow(company_id, warehouse_id):
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


@app.route("/api/v1/health")
def health():
    return jsonify({"status": "ok"})
