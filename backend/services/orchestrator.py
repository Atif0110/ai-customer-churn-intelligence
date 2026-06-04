from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def run_analysis(values: List[float]) -> Dict[str, Any]:
    """
    Run a complete churn analysis pipeline.

    Args:
        values: [usage_hours, support_tickets, tenure_months]

    Returns:
        {
            "ds_output":   <ChurnPrediction dict>,
            "explanation": <str>,
        }

    Raises:
        ValueError:  validation failure (bad inputs)
        Exception:   unexpected ML or LLM failure
    """
    t_start = time.time()
    logger.info("═" * 60)
    logger.info("🚀 Analysis started: %s", values)

    # ── 1. ML prediction ────────────────────────────────────────────
    from backend.services.ds_service import predict

    t0        = time.time()
    ds_output = predict(values)
    logger.info("✅ ML done in %.0f ms  →  risk=%s  prob=%.1f%%",
                (time.time() - t0) * 1000,
                ds_output["risk_level"],
                ds_output["churn_probability"] * 100)

    # ── 2. LLM explanation ──────────────────────────────────────────
    from backend.services.genai_service import generate_explanation

    t0          = time.time()
    explanation = generate_explanation(ds_output)
    logger.info("✅ Explanation done in %.0f ms", (time.time() - t0) * 1000)

    # ── 3. Persist to history (non-fatal) ───────────────────────────
    try:
        from backend.services.history_service import save_record
        save_record(values, ds_output)
    except Exception as exc:
        logger.warning("⚠️  History save failed (non-fatal): %s", exc)

    total_ms = (time.time() - t_start) * 1000
    logger.info("🏁 Analysis complete in %.0f ms total", total_ms)
    logger.info("═" * 60)

    return {"ds_output": ds_output, "explanation": explanation}
