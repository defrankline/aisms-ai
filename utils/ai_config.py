MODEL_VERSION = "v1.0"
DEFAULT_FORECAST_DAYS = 30
Z95 = 1.65  # 95% service level for safety stock
MIN_TS_POINTS = 3  # minimum daily points to train Prophet
LEAD_TIME_DAYS = 7  # default supplier lead time in days (override via API if needed)
