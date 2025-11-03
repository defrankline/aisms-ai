from sqlalchemy import text

INBOUND_CODES = ('RECEIPT_IN', 'RETURN_IN', 'ADJUSTMENT_IN', 'TRANSFER_IN')
OUTBOUND_CODES = ('SALE_OUT', 'ADJUSTMENT_OUT', 'TRANSFER_OUT')

SQL_STOCK_SUM_CASE = f"""
    SUM(
        CASE
            WHEN smt.code IN {INBOUND_CODES} THEN sm.quantity
            WHEN smt.code IN {OUTBOUND_CODES} THEN -sm.quantity
            ELSE 0
        END
    ) AS current_stock
"""


def build_stock_balance_query():
    return text(f"""
        SELECT product_id,
               {SQL_STOCK_SUM_CASE}
        FROM stock_movements sm
        JOIN stock_movement_types smt ON sm.movement_type_id = smt.id
        WHERE sm.company_id = :company_id
          AND sm.warehouse_id = :warehouse_id
          AND sm.approved = true
        GROUP BY product_id
    """)
