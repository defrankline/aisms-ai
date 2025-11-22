import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Numeric, Date, TIMESTAMP, BigInteger, UniqueConstraint
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# 1) Demand Forecast
class SkuDemandForecast(Base):
    __tablename__ = "sku_demand_forecast"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(BigInteger, nullable=False)
    warehouse_id = Column(BigInteger, nullable=False)
    product_id = Column(BigInteger, nullable=False)
    forecast_date = Column(Date, nullable=False)
    predicted_quantity = Column(Numeric(12, 3), nullable=False)
    model_version = Column(String(50), default="v1.0")
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint(
            "company_id", "product_id", "warehouse_id", "forecast_date", "model_version",
            name="uq_forecast_company_product_wh_date_version"
        ),
    )


# 2) Anomaly Detection
class SalesAnomalyEvent(Base):
    __tablename__ = "sales_anomaly_event"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identifiers
    sale_id = Column(BigInteger, nullable=False)
    company_id = Column(BigInteger, nullable=False)
    warehouse_id = Column(BigInteger, nullable=False)

    # Extra data for UI
    sale_date = Column(Date, nullable=False)
    reference_number = Column(String(255), nullable=False)
    warehouse_name = Column(String(255), nullable=False)
    total_amount = Column(Numeric(32, 2), nullable=False)

    reason = Column(String(500), nullable=False)

    # ML policy metadata
    policy_code = Column(String(50), nullable=False)
    score = Column(Numeric(10, 4), nullable=False)
    level = Column(String(20), nullable=False)

    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("sale_id", "policy_code", name="uq_sale_policy"),
    )


# 3) Reorder Suggestions
class ReorderSuggestion(Base):
    __tablename__ = "reorder_suggestions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(BigInteger, nullable=False)
    warehouse_id = Column(BigInteger, nullable=False)
    product_id = Column(BigInteger, nullable=False)

    avg_daily_demand = Column(Numeric(14, 3), nullable=False)
    std_daily_demand = Column(Numeric(14, 3), nullable=False)

    reorder_point = Column(Numeric(14, 3), nullable=False)
    safety_stock = Column(Numeric(14, 3), nullable=False)
    suggested_qty = Column(Numeric(14, 3), nullable=False)
    stock_in_hand = Column(Numeric(14, 3), nullable=False)

    model_version = Column(String(50), nullable=False)
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "company_id", "warehouse_id", "product_id", "model_version",
            name="uq_reorder_unique"
        ),
    )


# 4) Supplier Performance
class SupplierPerformanceScore(Base):
    __tablename__ = "supplier_performance_scores"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    supplier_id = Column(BigInteger, nullable=False)
    company_id = Column(BigInteger, nullable=False)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    on_time_rate = Column(Numeric(5, 2), nullable=False)
    accuracy_rate = Column(Numeric(5, 2), nullable=False)
    rejection_rate = Column(Numeric(5, 2), nullable=False)
    cost_stability = Column(Numeric(5, 2), nullable=False)
    overall_score = Column(Numeric(5, 2), nullable=False)
    model_version = Column(String(50), default="v1.0")
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint(
            "supplier_id", "company_id", "period_start", "period_end", "model_version",
            name="uq_supplier_company_period_model"
        ),
    )


# 5) Dynamic Pricing
class DynamicPricingRecommendation(Base):
    __tablename__ = "dynamic_pricing_recommendations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(BigInteger, nullable=False)
    warehouse_id = Column(BigInteger, nullable=False)
    product_id = Column(BigInteger, nullable=False)
    current_price = Column(Numeric(12, 2), nullable=False)
    suggested_price = Column(Numeric(12, 2), nullable=False)
    price_change_pct = Column(Numeric(5, 2), nullable=False)
    expected_demand_change = Column(Numeric(5, 2), nullable=False)
    confidence_level = Column(Numeric(5, 2), nullable=False)
    rationale = Column(String(255))
    model_version = Column(String(50), default="v1.0")
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint(
            "company_id", "warehouse_id", "product_id", "model_version",
            name="uq_dynamic_pricing_company_wh_prod_model"
        ),
    )


# 6) Customer Segmentation
class CustomerSegment(Base):
    __tablename__ = "customer_segments"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(BigInteger, nullable=False)
    customer_id = Column(BigInteger, nullable=False)
    recency_days = Column(BigInteger, nullable=False)
    frequency = Column(BigInteger, nullable=False)
    monetary_value = Column(Numeric(12, 2), nullable=False)
    clv_score = Column(Numeric(6, 2), nullable=False)
    segment = Column(String(50), nullable=False)
    model_version = Column(String(50), default="v1.0")
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint(
            "company_id", "customer_id", "model_version",
            name="uq_customer_company_model"
        ),
    )


# 7) Sales Performance
class SalesPerformanceScore(Base):
    __tablename__ = "sales_performance_scores"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(BigInteger, nullable=False)
    warehouse_id = Column(BigInteger, nullable=False)
    salesperson_id = Column(BigInteger, nullable=False)
    total_sales = Column(Numeric(14, 2), nullable=False)
    total_orders = Column(BigInteger, nullable=False)
    avg_order_value = Column(Numeric(12, 2), nullable=False)
    growth_rate = Column(Numeric(6, 2), nullable=False)
    performance_trend = Column(String(20), nullable=False)
    performance_score = Column(Numeric(6, 2), nullable=False)
    model_version = Column(String(50), default="v1.0")
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint(
            "company_id", "warehouse_id", "salesperson_id", "model_version",
            name="uq_sales_performance_company_wh_person_model"
        ),
    )


# 8) Profitability
class ProfitabilityForecast(Base):
    __tablename__ = "profitability_forecast"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    company_id = Column(BigInteger, nullable=False)
    warehouse_id = Column(BigInteger, nullable=False)

    # Newly added (correct)
    warehouse_name = Column(String(255), nullable=False)

    month = Column(Date, nullable=False)

    total_revenue = Column(Numeric(14, 2), nullable=False)
    total_cogs = Column(Numeric(14, 2), nullable=False)
    total_expenses = Column(Numeric(14, 2), nullable=False)

    net_profit = Column(Numeric(14, 2), nullable=False)
    profit_margin = Column(Numeric(6, 2), nullable=False)

    trend = Column(String(20), nullable=False)
    forecast_profit = Column(Numeric(14, 2))

    model_version = Column(String(50), default="v1.0")
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "warehouse_id",
            "month",
            "model_version",
            name="uq_profit_company_wh_month_model"
        ),
    )


# 9) Inventory Optimization
class InventoryOptimization(Base):
    __tablename__ = "inventory_optimization"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(BigInteger, nullable=False)
    warehouse_id = Column(BigInteger, nullable=False)
    product_id = Column(BigInteger, nullable=False)
    current_stock = Column(Numeric(14, 2), nullable=False)
    avg_daily_demand = Column(Numeric(12, 3), nullable=False)
    safety_stock = Column(Numeric(12, 3), nullable=False)
    optimal_stock_level = Column(Numeric(12, 3), nullable=False)
    stock_status = Column(String(20), nullable=False)
    inventory_health_score = Column(Numeric(6, 2), nullable=False)
    forecast_horizon_days = Column(BigInteger, default=30)
    model_version = Column(String(50), default="v1.0")
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint(
            "company_id", "warehouse_id", "product_id", "model_version",
            name="uq_inventory_opt_company_wh_prod_model"
        ),
    )


# 10) Cashflow
class CashflowForecast(Base):
    __tablename__ = "cashflow_forecast"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(BigInteger, nullable=False)
    warehouse_id = Column(BigInteger, nullable=False)
    month = Column(Date, nullable=False)
    cash_inflows = Column(Numeric(14, 2), nullable=False)
    cash_outflows = Column(Numeric(14, 2), nullable=False)
    net_cashflow = Column(Numeric(14, 2), nullable=False)
    cash_balance = Column(Numeric(14, 2), nullable=False)
    cash_health_score = Column(Numeric(6, 2), nullable=False)
    risk_level = Column(String(20), nullable=False)
    forecasted_next_balance = Column(Numeric(14, 2))
    model_version = Column(String(50), default="v1.0")
    generated_at = Column(TIMESTAMP, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint(
            "company_id", "warehouse_id", "month", "model_version",
            name="uq_cashflow_company_wh_month_model"
        ),
    )
