import json
import os
from typing import Any, Dict

CONFIG_FILE = "forecast_config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "prophet": {
        "changepoint_prior_scale": 0.03,
        "changepoint_range": 0.5,
        "seasonality_prior_scale": 0.15,
        "holidays_prior_scale": 0.1,
        "yearly_seasonality": True,
        "yearly_fourier_order": 5,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "use_holidays": True,
        "use_iq_value_scaled": False,
    },
    "random_forest": {
        "n_estimators": 400,
        "max_depth": 6,
        "use_holidays": True,
        "use_iq_value_scaled": False,
    },
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4,
        "use_holidays": True,
        "use_iq_value_scaled": False,
    },
    "var": {
        "lags": 2,
        "use_holidays": True,
        "use_iq_value_scaled": False,
    },
    "sarimax": {
        "order": [1, 1, 1],
        "seasonal_order": [1, 1, 1, 12],
        "use_holidays": True,
        "use_iq_value_scaled": False,
    },
    "general": {
        "use_seasonality": True,
    },
}


def _ensure_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing sections/keys with defaults."""
    merged = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    for k, v in (config or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
                return _ensure_config(json.load(fh))
        except Exception:
            pass
    save_config(DEFAULT_CONFIG)
    return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(config: Dict[str, Any]) -> None:
    cfg = _ensure_config(config)
    with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def reset_to_default() -> Dict[str, Any]:
    save_config(DEFAULT_CONFIG)
    return json.loads(json.dumps(DEFAULT_CONFIG))
