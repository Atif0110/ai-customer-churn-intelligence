from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from backend.core.validators import validate_inputs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class ChurnPrediction:
    """Every piece of information the platform derives about a single customer."""
    churn_probability: float       # 0.0–1.0
    risk_level: str                # "Low" | "Medium" | "High"
    drivers: List[str]             # plain-English reasons
    confidence_score: float        # 0.0–1.0, how certain the model is
    percentile: float              # 0–100, relative to the population
    interpretation: str            # one-liner for the UI header

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model training (happens once at import time)
# ---------------------------------------------------------------------------

# Representative SaaS customer samples:
#   columns = [usage_hours/mo, support_tickets/mo, tenure_months]
#   label   = 1 (churned)  /  0 (retained)
#
# Designed to create a smooth probability gradient across the feature space:
#   High usage + low tickets + long tenure  → retained (Low risk)
#   Middle usage + medium tickets + medium tenure → mix (Medium risk zone)
#   Low usage + high tickets + short tenure → churned (High risk)
_X = np.array([
    # --- Clearly retained (Low risk) ---
    [95,  1, 58],
    [88,  2, 52],
    [82,  1, 48],
    [78,  2, 44],
    [72,  3, 40],
    [65,  3, 36],
    [60,  4, 32],
    # --- Borderline / retained (Medium-Low) ---
    [55,  4, 30],
    [52,  5, 26],
    [48,  5, 24],
    # --- Borderline / churned (Medium-High) ---
    [42,  7, 20],
    [38,  7, 18],
    [35,  8, 16],
    # --- Clearly churned (High risk) ---
    [25, 10, 12],
    [18, 11,  8],
    [12, 13,  5],
    [ 8, 14,  3],
    [ 3, 15,  1],
], dtype=np.float32)

_y = np.array([
    0, 0, 0, 0, 0, 0, 0,   # retained
    0, 0, 0,               # borderline retained
    1, 1, 1,               # borderline churned
    1, 1, 1, 1, 1,         # churned
], dtype=np.int32)

try:
    _model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        class_weight="balanced",
    )
    _model.fit(_X, _y)
    logger.info("✅ Churn prediction model trained and ready")
except Exception as exc:
    logger.critical("❌ Could not train churn model: %s", exc, exc_info=True)
    raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _identify_drivers(v1: float, v2: float, v3: float) -> List[str]:
    """Return plain-English churn signals based on industry benchmarks."""
    drivers: List[str] = []
    if v1 < 20:
        drivers.append("Low product engagement")
    if v2 > 7:
        drivers.append("High support load")
    if v3 < 18:
        drivers.append("Early-stage customer")
    return drivers or ["Stable customer profile"]


def _classify_risk(prob: float) -> str:
    if prob >= 0.50:
        return "High"
    if prob >= 0.03:
        return "Medium"
    return "Low"


def _confidence(v1: float, v2: float, v3: float) -> float:
    """
    Estimate model confidence.
    Confidence decreases the further a customer sits from the training
    distribution's centre.  Not a calibrated probability — just a useful
    heuristic for surfacing "this is an unusual profile" in the UI.
    """
    centre = [50.0, 5.0, 24.0]
    scale  = [50.0, 7.5, 24.0]
    distances = [abs(v - c) / s for v, c, s in zip([v1, v2, v3], centre, scale)]
    avg_dist = float(np.mean(distances))
    return round(max(0.0, 1.0 - avg_dist * 0.30), 3)


def _percentile(prob: float) -> float:
    """Map churn probability to a 0–100 risk percentile."""
    return round(min(100.0, max(0.0, prob * 100.0)), 1)


def _interpretation(prob: float, risk: str, confidence: float) -> str:
    """One-liner for the UI."""
    if prob >= 0.85:
        headline = "🚨 Critical — act within 24 hours"
    elif prob >= 0.65:
        headline = "🔴 High risk — prioritise this customer now"
    elif prob >= 0.45:
        headline = "🟡 Medium risk — schedule a proactive check-in"
    elif prob >= 0.20:
        headline = "🟢 Low risk — stable, keep an eye on trends"
    else:
        headline = "✅ Very healthy — expand the relationship"

    conf_note = (
        "high-confidence prediction"
        if confidence > 0.80
        else "moderate-confidence prediction"
        if confidence > 0.50
        else "low-confidence prediction (unusual profile)"
    )
    return f"{headline}  •  {conf_note}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(values: List[float]) -> Dict[str, Any]:
    """
    Predict churn for a single customer.

    Args:
        values: [usage_hours, support_tickets, tenure_months]

    Returns:
        Dict representation of ChurnPrediction — safe to serialise to JSON.

    Raises:
        ValueError: human-readable validation message.
    """
    # --- validate ---
    v1, v2, v3 = float(values[0]), float(values[1]), float(values[2])
    ok, msg = validate_inputs(v1, v2, v3)
    if not ok:
        raise ValueError(msg)

    logger.info("📊 Predicting churn: usage=%.1f  tickets=%.1f  tenure=%.1f", v1, v2, v3)

    # --- score ---
    arr  = np.array([[v1, v2, v3]], dtype=np.float32)
    prob = float(np.clip(_model.predict_proba(arr)[0][1], 0.0, 1.0))

    # --- enrich ---
    drivers     = _identify_drivers(v1, v2, v3)
    risk        = _classify_risk(prob)
    conf        = _confidence(v1, v2, v3)
    pctile      = _percentile(prob)
    interp      = _interpretation(prob, risk, conf)

    result = ChurnPrediction(
        churn_probability=round(prob, 4),
        risk_level=risk,
        drivers=drivers,
        confidence_score=conf,
        percentile=pctile,
        interpretation=interp,
    )

    logger.info(
        "✅ Prediction → prob=%.2f%%  risk=%s  confidence=%.2f",
        prob * 100, risk, conf,
    )
    return result.to_dict()
