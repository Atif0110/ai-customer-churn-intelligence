"""
LLM explanation service.

Tries the configured LLM provider (Groq / OpenAI / Gemini) with
exponential-backoff retries.  If every attempt fails, it silently falls
back to a deterministic, rule-based explanation so the user always gets
a useful response — never a blank screen or a raw exception.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from backend.core.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(ds: Dict[str, Any]) -> str:
    drivers_text = "\n".join(f"  • {d}" for d in ds.get("drivers", []))
    prob = ds.get("churn_probability", 0)
    risk = ds.get("risk_level", "Unknown")
    conf = ds.get("confidence_score", 0)

    return f"""You are a senior customer-success strategist briefing your team.

CUSTOMER SNAPSHOT
─────────────────
Churn probability : {prob*100:.1f}%
Risk tier         : {risk}
Model confidence  : {conf*100:.0f}%
Key risk signals  :
{drivers_text}

YOUR BRIEF — respond in exactly this format, no extras:

**Why they might leave** (2 sentences, customer's perspective)
[your text here]

**Two actions for this week** (2 concrete bullet points)
• [action 1]
• [action 2]

**Business impact if they churn** (1 sentence, revenue/strategic framing)
[your text here]

TONE: direct, empathetic, business-friendly.  No jargon.  No markdown
headings other than the bold labels above.  Urgency must match the risk tier.
"""


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_DRIVER_CONTEXT = {
    "Low product engagement": (
        "The customer is barely touching the product — they haven't yet connected "
        "its value to their day-to-day work.  Without intervention, out-of-sight "
        "quickly becomes out-of-mind when renewal time arrives."
    ),
    "High support load": (
        "Frequent support tickets are a distress signal.  Something is causing "
        "repeated friction — whether that's a confusing UI, missing features, or "
        "an integration issue — and the customer is absorbing that pain each time."
    ),
    "Early-stage customer": (
        "New customers are still deciding whether your product fits their workflow.  "
        "The first 12–18 months are the highest-churn window; a single bad experience "
        "can tip the scales toward cancellation before they ever reach the 'loyal' stage."
    ),
    "Stable customer profile": (
        "No major red flags here.  The customer is engaged and largely self-sufficient.  "
        "The opportunity is to deepen the relationship and expand usage."
    ),
}

def _fallback_explanation(ds: Dict[str, Any]) -> str:
    prob    = ds.get("churn_probability", 0.0)
    risk    = ds.get("risk_level", "Unknown")
    drivers = ds.get("drivers", [])

    if prob >= 0.85:
        urgency_line = "🚨 **Immediate action required** — this customer could leave within weeks."
    elif prob >= 0.65:
        urgency_line = "🔴 **Prioritise this week** — the risk is real and growing."
    elif prob >= 0.35:
        urgency_line = "🟡 **Schedule a check-in soon** — things aren't critical yet, but they could be."
    else:
        urgency_line = "🟢 **Keep doing what's working** — this customer is in good shape."

    driver_block = ""
    for d in drivers:
        ctx = _DRIVER_CONTEXT.get(d, f"{d} is contributing to churn risk.")
        driver_block += f"\n**{d}**\n{ctx}\n"

    return f"""
{urgency_line}

**Why they might leave**
{driver_block.strip()}

**Two actions for this week**
• Schedule a 30-minute success call — ask open questions about what's working and what isn't.  Document everything for the product team.
• Create one concrete "quick win" — resolve their top support issue or demo a feature that directly addresses their reported pain.

**Business impact if they churn**
Customers at {risk.lower()} risk represent an active revenue threat; losing them costs not just the contract value but also the referrals and expansion revenue they would have generated.

---
*Insight generated via rule-based analysis (AI service temporarily unavailable).*
""".strip()


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class GenAIService:
    """
    Wraps the LLM provider with retry logic and a fallback strategy.
    Always returns a non-empty, human-readable explanation.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._llm: Optional[Any] = None
        self._init_llm()

    def _init_llm(self) -> None:
        """Lazy-import so the app doesn't crash on missing env vars at startup."""
        try:
            from backend.llm.factory import get_llm
            self._llm = get_llm()
            logger.info("✅ LLM provider initialised")
        except Exception as exc:
            logger.warning("⚠️  LLM init failed — will use fallback: %s", exc)
            self._llm = None

    # ------------------------------------------------------------------
    def generate_explanation(self, ds_output: Dict[str, Any]) -> str:
        """
        Generate a retention strategy explanation.

        Always returns a string — never raises.
        """
        if self._llm is None:
            logger.info("LLM unavailable at startup — using fallback")
            return _fallback_explanation(ds_output)

        try:
            return self._call_with_retry(ds_output)
        except LLMServiceError as exc:
            logger.warning("LLM exhausted retries: %s  →  using fallback", exc)
            return _fallback_explanation(ds_output)
        except Exception as exc:
            logger.error("Unexpected LLM error: %s  →  using fallback", exc, exc_info=True)
            return _fallback_explanation(ds_output)

    # ------------------------------------------------------------------
    def _call_with_retry(self, ds_output: Dict[str, Any]) -> str:
        prompt      = _build_prompt(ds_output)
        last_error  = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("🔄 LLM attempt %d/%d …", attempt, self.max_retries)
                t0     = time.time()
                result = self._llm.generate(prompt)
                elapsed = (time.time() - t0) * 1000

                if result and len(result.strip()) > 30:
                    logger.info("✅ LLM succeeded in %.0f ms (attempt %d)", elapsed, attempt)
                    return result.strip()

                last_error = "Response was empty or too short"
                logger.warning("⚠️  Attempt %d: %s", attempt, last_error)

            except Exception as exc:
                last_error = str(exc)
                logger.warning("⚠️  LLM attempt %d failed: %s", attempt, exc)

            if attempt < self.max_retries:
                wait = 2 ** (attempt - 1)          # 1 s, 2 s, …
                logger.info("⏳ Retrying in %d s …", wait)
                time.sleep(wait)

        raise LLMServiceError(
            f"All {self.max_retries} LLM attempts failed.  Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Module-level singleton + thin wrapper used by orchestrator
# ---------------------------------------------------------------------------

_service = GenAIService()


def generate_explanation(ds_output: Dict[str, Any]) -> str:
    """Public entry point — keeps backward compatibility."""
    return _service.generate_explanation(ds_output)
