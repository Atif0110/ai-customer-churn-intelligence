# backend/core/validators.py
"""
Input validation with context-aware, human-friendly error messages.
These messages are surfaced directly to the user — they're written for
people, not for machines.
"""

from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# What each field means and what's considered normal
_FIELD_META = {
    "v1": {
        "name": "Monthly Usage Hours",
        "min": 0,
        "max": 100,
        "typical": "10–80",
        "description": "How many hours per month the customer actively uses the product.",
    },
    "v2": {
        "name": "Support Tickets",
        "min": 0,
        "max": 15,
        "typical": "0–8",
        "description": "Number of support tickets raised per month.",
    },
    "v3": {
        "name": "Customer Tenure",
        "min": 0,
        "max": 60,
        "typical": "6–48",
        "description": "How many months the customer has been with you.",
    },
}


def validate_inputs(v1: float, v2: float, v3: float) -> Tuple[bool, Optional[str]]:
    """
    Validate all three customer metrics.

    Returns:
        (True, None) when all inputs are valid.
        (False, helpful_message) when something is off.
    """
    for key, value in [("v1", v1), ("v2", v2), ("v3", v3)]:
        ok, msg = _validate_one(key, value)
        if not ok:
            return False, msg
    return True, None


def _validate_one(key: str, value) -> Tuple[bool, Optional[str]]:
    """Validate a single field and return a friendly message on failure."""
    meta = _FIELD_META[key]

    # Must be numeric
    if not isinstance(value, (int, float)):
        return False, (
            f"❌ {meta['name']} must be a number — got '{value}' instead.\n"
            f"   {meta['description']}"
        )

    # Must be finite
    import math
    if math.isnan(value) or math.isinf(value):
        return False, f"❌ {meta['name']} must be a real number, not {value}."

    # Must be in range
    lo, hi = meta["min"], meta["max"]
    if not (lo <= value <= hi):
        return False, (
            f"❌ {meta['name']} is outside the expected range.\n\n"
            f"   You entered  : {value}\n"
            f"   Valid range  : {lo} – {hi}\n"
            f"   Typical range: {meta['typical']}\n\n"
            f"   {meta['description']}"
        )

    return True, None


def get_profile_insights(v1: float, v2: float, v3: float) -> List[str]:
    """
    Translate raw numbers into plain-English observations about the
    customer profile.  Used in logs and as contextual hints in the UI.
    """
    insights: List[str] = []

    # Usage
    if v1 < 5:
        insights.append("🔴 Near-zero usage — customer has essentially stopped engaging")
    elif v1 < 20:
        insights.append("🟡 Low usage — disengagement is setting in")
    elif v1 < 40:
        insights.append("🟡 Below-average usage — room to grow engagement")
    elif v1 > 80:
        insights.append("🟢 Power user — heavily invested in the product")

    # Support load
    if v2 > 10:
        insights.append("⚠️ Very high support volume — customer is experiencing friction")
    elif v2 > 7:
        insights.append("⚠️ Above-average support load — worth investigating why")
    elif v2 == 0:
        insights.append("✅ Zero support tickets — fully self-sufficient")

    # Tenure
    if v3 < 3:
        insights.append("🆕 Brand-new customer — still in onboarding, high churn window")
    elif v3 < 12:
        insights.append("📅 Early-stage customer — evaluation period, needs attention")
    elif v3 > 36:
        insights.append("⭐ Long-term customer — high lifetime value, worth protecting")

    return insights or ["📊 Balanced profile — no extreme signals detected"]
