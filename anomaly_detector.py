"""
anomaly_detector.py
===================
Isolation Forest anomaly detection module.

Loads the three pre-trained model artefacts once at import time and exposes
a single public function:  score_snapshot(sensor_stats_dict) -> dict

Training details (from data-science team):
- Trained on healthy machine data only  (file_type = 'max')
- 3 sensors merged into one 61-feature row per snapshot
- StandardScaler fitted on the same data
- IsolationForest: 200 trees, contamination = 0.01

IMPORTANT: The pkl files must be loaded with joblib, NOT pickle.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load artefacts once at module import
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_isolation_forest = None
_standard_scaler  = None
_feature_columns  = None   # ordered list[str] of 61 column names
_model_ready      = False

def _load_models():
    """Load all three pkl artefacts.  Called once at module import."""
    global _isolation_forest, _standard_scaler, _feature_columns, _model_ready
    try:
        import joblib
        iso_path = os.path.join(_MODEL_DIR, "isolation_forest.pkl")
        sc_path  = os.path.join(_MODEL_DIR, "standard_scaler.pkl")
        fc_path  = os.path.join(_MODEL_DIR, "feature_columns.pkl")

        if not all(os.path.exists(p) for p in [iso_path, sc_path, fc_path]):
            logger.warning("⚠️  Model artefacts not found – anomaly detection disabled")
            return

        _isolation_forest = joblib.load(iso_path)
        _standard_scaler  = joblib.load(sc_path)
        raw_cols = joblib.load(fc_path)

        # feature_columns.pkl is saved as a numpy array of strings – normalise
        if hasattr(raw_cols, "tolist"):
            _feature_columns = raw_cols.tolist()
        else:
            _feature_columns = list(raw_cols)

        _model_ready = True
        logger.info(
            "✅ Anomaly detection models loaded  "
            f"(IsolationForest n_estimators={_isolation_forest.n_estimators}, "
            f"contamination={_isolation_forest.contamination}, "
            f"features={len(_feature_columns)})"
        )
    except Exception as exc:
        logger.error(f"❌ Failed to load anomaly detection models: {exc}", exc_info=True)

_load_models()


# ---------------------------------------------------------------------------
# Feature assembly helpers
# ---------------------------------------------------------------------------

# Map from model column suffix → key inside each per-sensor stats dict
_SUFFIX_TO_KEY = {
    "x_min":              "x_min",
    "x_max":              "x_max",
    "mean":               "mean",
    "standard_deviation": "standard_deviation",
    "range":              "range",
    "skewness":           "skewness",
    "kurtosis":           "kurtosis",
    "rms":                "rms",
    "peak":               "peak",
    "crest_factor":       "crest_factor",
    "load_factor":        "load_factor",
    # frequency1..5
    "frequency1": "frequency1",
    "frequency2": "frequency2",
    "frequency3": "frequency3",
    "frequency4": "frequency4",
    "frequency5": "frequency5",
    # amplitude1..5
    "amplitude1": "amplitude1",
    "amplitude2": "amplitude2",
    "amplitude3": "amplitude3",
    "amplitude4": "amplitude4",
    "amplitude5": "amplitude5",
}

def _extract_value(sensor_dict: dict, suffix: str) -> float:
    """Return the float value for a given feature suffix from a sensor dict."""
    key = _SUFFIX_TO_KEY.get(suffix, suffix)
    val = sensor_dict.get(key)
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return 0.0


def _build_feature_vector(sensor_stats: dict) -> np.ndarray:
    """
    Build a (1, 61) numpy array in the exact column order stored in
    feature_columns.pkl.

    sensor_stats must look like:
        {
          'acceleration': { 'x_min': …, 'x_max': …, 'mean': …, … },
          'current':      { … },
          'audio':        { … }
        }
    """
    row = []
    for col_name in _feature_columns:
        # col_name has the form  "sensor_suffix"  e.g. "acceleration_x_min"
        # Split on first underscore that corresponds to a known sensor prefix
        placed = False
        for sensor in ("acceleration", "audio", "current"):
            prefix = sensor + "_"
            if col_name.startswith(prefix):
                suffix = col_name[len(prefix):]
                val = _extract_value(sensor_stats.get(sensor, {}), suffix)
                row.append(val)
                placed = True
                break
        if not placed:
            row.append(0.0)

    return np.array(row, dtype=float).reshape(1, -1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_snapshot(sensor_stats: dict) -> dict:
    """
    Run anomaly detection on a single multi-sensor snapshot.

    Parameters
    ----------
    sensor_stats : dict
        {
            'acceleration': { 'x_min': …, 'x_max': …, 'mean': …,
                              'standard_deviation': …, 'range': …,
                              'skewness': …, 'kurtosis': …,
                              'rms': …, 'peak': …, 'crest_factor': …,
                              'load_factor': …,
                              'frequency1': …, …, 'frequency5': …,
                              'amplitude1': …, …, 'amplitude5': … },
            'current':      { … },
            'audio':        { … }
        }

    Returns
    -------
    dict
        anomaly_score  – float in (-inf, 0]; closer to -1 means more anomalous
        is_anomaly     – bool; True when the model predicts an anomaly
        confidence     – float 0–100; higher = more confident it is anomalous
        status         – 'normal' | 'anomaly' | 'model_not_ready' | 'error'
        n_sensors_used – int; how many sensors had non-zero data
    """
    if not _model_ready:
        return {
            "anomaly_score": None,
            "is_anomaly":    False,
            "confidence":    0.0,
            "status":        "model_not_ready",
            "n_sensors_used": 0,
        }

    try:
        # Count how many sensors provided real data
        n_sensors_used = sum(
            1 for s in ("acceleration", "current", "audio")
            if any(v not in (None, 0, 0.0) for v in (sensor_stats.get(s) or {}).values())
        )

        X = _build_feature_vector(sensor_stats)

        # Scale → predict
        X_scaled = _standard_scaler.transform(X)
        prediction = int(_isolation_forest.predict(X_scaled)[0])     # 1 or -1
        raw_score  = float(_isolation_forest.decision_function(X_scaled)[0])

        # decision_function returns scores roughly in [-0.5, 0.5]
        # Normalise to a 0–100 "anomaly confidence" where 100 = definitely anomalous
        # threshold is model offset_ (≈ -0.54 for this model)
        threshold = float(_isolation_forest.offset_)
        # Clamp raw_score to a sensible window for display
        clamped = max(threshold * 2, min(0.0, raw_score))
        # Scale so that threshold → 50 and threshold*2 → 100
        if threshold != 0:
            confidence = min(100.0, max(0.0, (clamped / threshold) * 100))
        else:
            confidence = 0.0

        is_anomaly = (prediction == -1)
        status = "anomaly" if is_anomaly else "normal"

        return {
            "anomaly_score":  round(raw_score, 6),
            "is_anomaly":     is_anomaly,
            "confidence":     round(confidence, 1),
            "status":         status,
            "n_sensors_used": n_sensors_used,
        }

    except Exception as exc:
        logger.error(f"❌ Anomaly scoring failed: {exc}", exc_info=True)
        return {
            "anomaly_score": None,
            "is_anomaly":    False,
            "confidence":    0.0,
            "status":        "error",
            "n_sensors_used": 0,
        }


def is_ready() -> bool:
    """Return True if models were loaded successfully."""
    return _model_ready
