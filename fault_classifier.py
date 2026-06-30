"""
fault_classifier.py
===================
XGBoost machine learning module for predicting specific fault types.

Loads the pre-trained XGBoost classifier, scaler, features, and label encoder
once at module import, and exposes the predict_fault(sensor_stats) function.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load artefacts once at module import
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "CLASSIFIERS")

_classifier = None
_scaler = None
_feature_columns = None   # ordered list[str] of feature names
_label_encoder = None
_model_ready = False

def _load_models():
    """Load all four XGBoost classifier artefacts. Called once at module import."""
    global _classifier, _scaler, _feature_columns, _label_encoder, _model_ready
    try:
        import joblib
        clf_path = os.path.join(_MODEL_DIR, "xgboost_fault_classifier.pkl")
        sc_path  = os.path.join(_MODEL_DIR, "xg_boost_feature_scaler.pkl")
        fc_path  = os.path.join(_MODEL_DIR, "xg_boost_feature_columns.pkl")
        le_path  = os.path.join(_MODEL_DIR, "xg_boost_label_encoder.pkl")

        if not all(os.path.exists(p) for p in [clf_path, sc_path, fc_path, le_path]):
            logger.warning("⚠️  XGBoost classifier artefacts not found – fault classification disabled")
            return

        _classifier = joblib.load(clf_path)
        _scaler     = joblib.load(sc_path)
        
        # Load feature columns
        raw_cols = joblib.load(fc_path)
        if hasattr(raw_cols, "tolist"):
            _feature_columns = raw_cols.tolist()
        else:
            _feature_columns = list(raw_cols)

        # Load label encoder
        _label_encoder = joblib.load(le_path)

        _model_ready = True
        logger.info(
            "✅ XGBoost fault classifier loaded successfully "
            f"(features={len(_feature_columns)}, classes={len(_label_encoder.classes_)})"
        )
    except Exception as exc:
        logger.error(f"❌ Failed to load fault classifier models: {exc}", exc_info=True)

# Run loader
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
    Build a feature vector in the exact column order expected by the classifier.
    """
    row = []
    for col_name in _feature_columns:
        # col_name has the form "sensor_suffix" e.g. "acceleration_mean"
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

def predict_fault(sensor_stats: dict) -> dict:
    """
    Predict specific fault type from a multi-sensor stats snapshot.

    Parameters
    ----------
    sensor_stats : dict
        {
            'acceleration': { 'mean': …, 'standard_deviation': …, … },
            'current':      { … },
            'audio':        { … }
        }

    Returns
    -------
    dict
        predicted_fault     – str; the classified fault name
        confidence          – float 0–100; classifier confidence percentage
        distribution        – list[dict]; list of dicts with {"fault": name, "probability": percent}
    """
    if not _model_ready:
        return {
            "predicted_fault": "Model Not Ready",
            "confidence": 0.0,
            "distribution": []
        }

    try:
        X = _build_feature_vector(sensor_stats)
        X_scaled = _scaler.transform(X)

        # Run prediction
        pred_idx = int(_classifier.predict(X_scaled)[0])
        pred_label = _label_encoder.inverse_transform([pred_idx])[0]

        # Calculate prediction probabilities if available
        confidence = 0.0
        dist = []
        if hasattr(_classifier, "predict_proba"):
            probs = _classifier.predict_proba(X_scaled)[0]
            confidence = float(np.max(probs)) * 100.0
            
            for idx, prob in enumerate(probs):
                class_label = _label_encoder.inverse_transform([idx])[0]
                dist.append({
                    "fault": str(class_label),
                    "probability": round(float(prob) * 100.0, 1)
                })
            # Sort descending by probability
            dist.sort(key=lambda x: x["probability"], reverse=True)
        else:
            confidence = 100.0
            dist = [{"fault": str(pred_label), "probability": 100.0}]

        return {
            "predicted_fault": str(pred_label),
            "confidence": round(confidence, 1),
            "distribution": dist
        }

    except Exception as exc:
        logger.error(f"❌ Fault classification prediction failed: {exc}", exc_info=True)
        return {
            "predicted_fault": "Error during prediction",
            "confidence": 0.0,
            "distribution": []
        }


def is_ready() -> bool:
    """Return True if XGBoost models were loaded successfully."""
    return _model_ready
