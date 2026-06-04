from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR   = Path("data")
_HIST_FILE  = _DATA_DIR / "history.json"
_BAK_FILE   = _DATA_DIR / "history.json.bak"
_MAX_RECORDS = 10_000


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw() -> List[Dict[str, Any]]:
    """Return the current history list, or [] on any read error."""
    _ensure_dir()
    if not _HIST_FILE.exists():
        return []
    try:
        with _HIST_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        logger.warning("⚠️  history.json was not a list — resetting")
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("❌ Could not read history: %s", exc)
        return []


def _save_raw(records: List[Dict[str, Any]]) -> bool:
    """
    Write records atomically.
    Returns True on success, False on failure.
    """
    _ensure_dir()
    try:
        # Write to a temp file first
        fd, tmp_path = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False)
        except Exception:
            os.unlink(tmp_path)
            raise

        # Rotate: current → .bak, tmp → current
        if _HIST_FILE.exists():
            _HIST_FILE.replace(_BAK_FILE)
        Path(tmp_path).replace(_HIST_FILE)

        logger.debug("💾 Saved %d history records", len(records))
        return True

    except Exception as exc:
        logger.error("❌ Failed to save history: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_record(values: List[float], result: Dict[str, Any]) -> bool:
    """
    Append one analysis result to the history file.

    Args:
        values: [usage_hours, support_tickets, tenure_months]
        result: ds_service.predict() output dict

    Returns:
        True if the record was persisted, False otherwise.
    """
    records = _load_raw()

    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "usage_hours":      float(values[0]),
            "support_tickets":  float(values[1]),
            "tenure_months":    float(values[2]),
        },
        "output": {
            "churn_probability": float(result.get("churn_probability", 0)),
            "risk_level":        result.get("risk_level", "Unknown"),
            "drivers":           result.get("drivers", []),
            "confidence":        float(result.get("confidence_score", 0)),
        },
    }
    records.append(record)

    # Trim oldest entries if we exceed the cap
    if len(records) > _MAX_RECORDS:
        records = records[-_MAX_RECORDS:]

    saved = _save_raw(records)
    if saved:
        logger.info("✅ History saved  (total: %d)", len(records))
    return saved


def get_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return history records, newest first.

    Args:
        limit: If given, return only this many records.
    """
    records = list(reversed(_load_raw()))
    if limit:
        records = records[:limit]
    logger.info("📖 Retrieved %d history records", len(records))
    return records


def get_statistics() -> Dict[str, Any]:
    """Summarise history for the dashboard."""
    records = _load_raw()
    if not records:
        return {"total_analyses": 0, "message": "No analyses recorded yet."}

    probs = [r["output"]["churn_probability"] for r in records if "output" in r]
    risks = [r["output"]["risk_level"]        for r in records if "output" in r]

    return {
        "total_analyses":            len(records),
        "average_churn_probability": round(sum(probs) / len(probs), 3) if probs else 0,
        "min_churn":                 round(min(probs), 3) if probs else 0,
        "max_churn":                 round(max(probs), 3) if probs else 0,
        "high_risk_count":           risks.count("High"),
        "medium_risk_count":         risks.count("Medium"),
        "low_risk_count":            risks.count("Low"),
        "latest_timestamp":          records[-1].get("timestamp"),
    }


def clear_history() -> bool:
    """Erase all history (useful for tests or a manual reset)."""
    ok = _save_raw([])
    if ok:
        logger.warning("⚠️  History cleared")
    return ok
