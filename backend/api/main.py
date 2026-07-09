from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Pydantic response models ─────────────────────────────────────────────────

class DSOutput(BaseModel):
    churn_probability: float
    risk_level: str
    drivers: List[str]
    confidence_score: float
    percentile: float
    interpretation: str


class AnalysisResponse(BaseModel):
    ds_output: DSOutput
    explanation: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimulationResponse(BaseModel):
    before: float = Field(..., ge=0, le=1)
    after:  float = Field(..., ge=0, le=1)
    impact: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    components: Dict[str, str] = Field(default_factory=dict)


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Customer Churn Intelligence API",
    version="2.0.0",
    description="Predict, explain, and act on customer churn risk.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lifecycle ────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup() -> None:
    logger.info("=" * 60)
    logger.info("🚀 Churn Intelligence API v2.0.0 starting …")
    logger.info("=" * 60)


@app.on_event("shutdown")
def _shutdown() -> None:
    logger.info("🛑 API shutting down gracefully")


# ── Utility ──────────────────────────────────────────────────────────────────

def _validate(v1: float, v2: float, v3: float) -> None:
    """
    Validate the three input metrics and raise HTTPException 400 with a
    human-friendly message if anything is out of range.
    """
    from backend.core.validators import validate_inputs
    ok, msg = validate_inputs(v1, v2, v3)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root() -> Dict[str, Any]:
    return {
        "service":   "AI Customer Churn Intelligence API",
        "version":   "2.0.0",
        "status":    "operational",
        "endpoints": {"/health": "GET", "/analyze": "POST", "/simulate": "POST"},
        "timestamp": datetime.now().isoformat(),
    }


@app.api_route(
    "/health",
    methods=["GET", "HEAD"],
    response_model=HealthResponse,
    tags=["Monitoring"],
)

def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        timestamp=datetime.now().isoformat(),
        components={
            "api":        "✅ operational",
            "ml_model":   "✅ loaded",
            "llm_service":"✅ ready",
        },
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze(
    v1: float = 50.0,
    v2: float = 5.0,
    v3: float = 25.0,
) -> AnalysisResponse:
    """
    Full churn analysis for a single customer.

    - **v1**: Monthly usage hours (0–100)
    - **v2**: Support tickets per month (0–15)
    - **v3**: Customer tenure in months (0–60)
    """
    t0 = time.time()

    _validate(v1, v2, v3)

    from backend.core.validators import get_profile_insights
    insights = get_profile_insights(v1, v2, v3)
    logger.info("📋 Profile insights: %s", insights)

    try:
        from backend.services.orchestrator import run_analysis
        result = run_analysis([v1, v2, v3])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("❌ Analysis error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed.  Please check your inputs and try again.  ({exc})",
        )

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return AnalysisResponse(
        ds_output=DSOutput(**result["ds_output"]),
        explanation=result["explanation"],
        metadata={
            "processing_time_ms": elapsed_ms,
            "api_version":        "2.0.0",
            "profile_insights":   insights,
            "timestamp":          datetime.now().isoformat(),
        },
    )


@app.post("/simulate", response_model=SimulationResponse, tags=["Analysis"])
def simulate(
    v1: float     = 50.0,
    v2: float     = 5.0,
    v3: float     = 25.0,
    change: float = 0.30,
) -> SimulationResponse:
    """
    What-if simulation: how does churn risk change if usage increases by `change`?

    - **change**: fractional change to apply to v1 (e.g. 0.30 = +30 %)
    """
    t0 = time.time()

    _validate(v1, v2, v3)

    if not (-0.5 <= change <= 1.0):
        raise HTTPException(
            status_code=400,
            detail=f"Change ({change}) is outside [-0.5, 1.0].  Use a fraction, e.g. 0.30 for +30 %.",
        )

    adjusted_v1 = v1 * (1 + change)
    if not (0 <= adjusted_v1 <= 100):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Applying a {change*100:.0f}% change would push usage to {adjusted_v1:.1f}, "
                "which is outside [0–100].  Reduce the change amount."
            ),
        )

    try:
        from backend.services.ds_service import predict

        base = predict([v1, v2, v3])
        new  = predict([adjusted_v1, v2, v3])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("❌ Simulation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

    before_prob = float(base["churn_probability"])
    after_prob  = float(new["churn_probability"])
    impact      = round(after_prob - before_prob, 4)

    elapsed_ms  = round((time.time() - t0) * 1000, 1)
    improvement = round(abs(impact) / before_prob * 100, 1) if before_prob > 0 else 0.0

    return SimulationResponse(
        before=before_prob,
        after=after_prob,
        impact=impact,
        metadata={
            "processing_time_ms":    elapsed_ms,
            "change_applied":        f"{change*100:.1f}%",
            "adjusted_usage_hours":  round(adjusted_v1, 1),
            "improvement_percent":   improvement,
            "direction":             "improved ✅" if impact < 0 else "worsened ❌",
            "api_version":           "2.0.0",
        },
    )


# ── Exception handlers ───────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP %d at %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error":      exc.detail,
            "status_code": exc.status_code,
            "timestamp":   datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception at %s: %s", request.url.path, exc, exc_info=True)
    debug = os.getenv("DEBUG", "false").lower() == "true"
    return JSONResponse(
        status_code=500,
        content={
            "error":     "An unexpected error occurred.  Please try again.",
            "detail":    str(exc) if debug else "Contact support if this persists.",
            "timestamp": datetime.now().isoformat(),
        },
    )


# ── Dev entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port  = int(os.environ.get("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=port, reload=debug)
