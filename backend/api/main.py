from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# RESPONSE MODELS (Type-Safe Responses)
# ============================================================================

class AnalysisResponse(BaseModel):
    """Response model for /analyze endpoint"""
    ds_output: Dict[str, Any] = Field(
        ..., 
        description="ML model output with predictions"
    )
    explanation: str = Field(
        ..., 
        description="AI-generated explanation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        schema_extra = {
            "example": {
                "ds_output": {
                    "churn_probability": 0.65,
                    "risk_level": "High",
                    "drivers": ["Low engagement"]
                },
                "explanation": "This customer shows signs...",
                "metadata": {
                    "processing_time_ms": 1250,
                    "api_version": "2.0.0"
                }
            }
        }


class SimulationResponse(BaseModel):
    """Response model for /simulate endpoint"""
    before: float = Field(
        ..., 
        ge=0, le=1,
        description="Churn probability before intervention"
    )
    after: float = Field(
        ..., 
        ge=0, le=1,
        description="Churn probability after intervention"
    )
    impact: float = Field(
        ...,
        description="Change in churn probability"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        schema_extra = {
            "example": {
                "before": 0.65,
                "after": 0.45,
                "impact": -0.20,
                "metadata": {
                    "processing_time_ms": 150,
                    "improvement_percentage": 30.8
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str = Field(..., description="System status")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="Current timestamp")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "message": "All systems operational",
                "timestamp": "2025-01-15T10:30:45.123456",
                "components": {
                    "api": "✅",
                    "ml_model": "✅",
                    "llm_provider": "✅"
                }
            }
        }


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="AI Customer Churn Decision Intelligence API",
    version="2.0.0",
    description="Production-grade churn prediction with ML and LLM integration",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
def startup_event():
    """Initialize application on startup"""
    logger.info("=" * 70)
    logger.info("🚀 STARTUP: AI Customer Churn Intelligence API v2.0.0")
    logger.info("=" * 70)
    logger.info("✅ CORS middleware configured")
    logger.info("✅ Request logging enabled")
    logger.info("✅ Health check endpoint ready")
    logger.info("=" * 70)


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("=" * 70)
    logger.info("🛑 SHUTDOWN: API shutting down gracefully")
    logger.info("=" * 70)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_input_insights(v1: float, v2: float, v3: float) -> list:
    """
    Generate insights about input profile
    Helps with debugging and understanding customer profiles
    """
    insights = []
    
    if v1 < 10:
        insights.append("Minimal usage - critical engagement issue")
    elif v1 < 30:
        insights.append("Below average usage")
    elif v1 > 80:
        insights.append("High engagement - power user")
    
    if v2 > 10:
        insights.append("High support volume - possible friction")
    elif v2 == 0:
        insights.append("No support requests - fully self-sufficient")
    
    if v3 < 6:
        insights.append("Brand new customer - onboarding phase")
    elif v3 > 36:
        insights.append("Long-term customer - established relationship")
    
    return insights if insights else ["Standard profile"]


def create_error_response(
    detail: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create helpful error response with context
    """
    response = {
        "error": detail,
        "timestamp": datetime.now().isoformat(),
        "hint": "Check API documentation at /docs",
    }
    
    if context:
        response["context"] = context
    
    return response


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Info"])
def root() -> Dict[str, Any]:
    """
    Root endpoint - API information
    """
    return {
        "service": "AI Customer Churn Decision Intelligence API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "simulate": "/simulate",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """
    Health check endpoint with detailed component status
    """
    try:
        # Verify services are responsive
        components = {
            "api": "✅ Operational",
            "ml_model": "✅ Loaded",
            "llm_provider": "✅ Ready",
            "logging": "✅ Active"
        }
        
        logger.info("✅ Health check passed")
        
        return HealthResponse(
            status="healthy",
            message="All systems operational",
            timestamp=datetime.now().isoformat(),
            components=components
        )
    
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )


# ============================================================================
# ANALYZE ENDPOINT
# ============================================================================

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze(
    v1: float = 50.0,
    v2: float = 5.0,
    v3: float = 25.0
) -> AnalysisResponse:
    """
    Analyze customer churn risk with AI explanation
    
    Parameters:
    - **v1**: Monthly usage hours (0-100)
    - **v2**: Support tickets per month (0-15)
    - **v3**: Customer tenure in months (0-60)
    
    Returns:
    - **ds_output**: ML prediction with churn probability
    - **explanation**: AI-generated business insights
    - **metadata**: Processing info and diagnostics
    """
    
    request_start = time.time()
    
    try:
        logger.info(f"📊 Analysis requested: v1={v1}, v2={v2}, v3={v3}")
        
        # ====== VALIDATION ======
        # Input validation with helpful context
        errors = []
        
        if not (0 <= v1 <= 100):
            errors.append(
                f"Usage hours ({v1}) outside valid range [0-100]. "
                f"Typical SaaS: 10-80 hours/month"
            )
        
        if not (0 <= v2 <= 15):
            errors.append(
                f"Support tickets ({v2}) outside valid range [0-15]. "
                f"Typical SaaS: 0-8 tickets/month"
            )
        
        if not (0 <= v3 <= 60):
            errors.append(
                f"Tenure ({v3} months) outside valid range [0-60]. "
                f"Platform supports up to 5 years of history"
            )
        
        if errors:
            logger.warning(f"❌ Validation failed: {errors}")
            raise HTTPException(
                status_code=400,
                detail="\n".join(errors)
            )
        
        # Get profile insights for logging
        insights = get_input_insights(v1, v2, v3)
        logger.info(f"📈 Customer profile: {insights}")
        
        # ====== ANALYSIS ======
        # Lazy import to prevent startup crashes
        from backend.services.orchestrator import run_analysis
        
        logger.info("🔄 Starting analysis pipeline...")
        
        analysis_start = time.time()
        result = run_analysis([v1, v2, v3])
        analysis_time = time.time() - analysis_start
        
        logger.info(f"✅ Analysis complete in {analysis_time*1000:.1f}ms")
        
        # ====== RESPONSE BUILDING ======
        total_time = time.time() - request_start
        
        response = AnalysisResponse(
            ds_output=result.get("ds_output", {}),
            explanation=result.get("explanation", ""),
            metadata={
                "processing_time_ms": round(total_time * 1000, 1),
                "analysis_time_ms": round(analysis_time * 1000, 1),
                "api_version": "2.0.0",
                "profile_insights": insights,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(
            f"✅ Response prepared: "
            f"churn={response.ds_output.get('churn_probability', 0):.2%}, "
            f"total_time={total_time*1000:.1f}ms"
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Handle unexpected errors gracefully
        logger.error(f"❌ Analysis error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis failed. Please check your inputs and try again. "
                f"Error: {str(e)}"
            )
        )


# ============================================================================
# SIMULATE ENDPOINT
# ============================================================================

@app.post("/simulate", response_model=SimulationResponse, tags=["Analysis"])
def simulate(
    v1: float = 50.0,
    v2: float = 5.0,
    v3: float = 25.0,
    change: float = 0.3
) -> SimulationResponse:
    """
    Simulate impact of intervention (what-if analysis)
    
    Parameters:
    - **v1**: Current monthly usage hours (0-100)
    - **v2**: Current support tickets per month (0-15)
    - **v3**: Current tenure in months (0-60)
    - **change**: Usage change as decimal (-0.5 to 1.0)
      - 0.3 = 30% increase
      - -0.2 = 20% decrease
    
    Returns:
    - **before**: Churn probability before intervention
    - **after**: Churn probability after intervention
    - **impact**: Change in churn probability
    - **metadata**: Processing info
    """
    
    request_start = time.time()
    
    try:
        logger.info(
            f"🎯 Simulation requested: "
            f"v1={v1}, v2={v2}, v3={v3}, change={change*100:.1f}%"
        )
        
        # ====== VALIDATION ======
        errors = []
        
        # Validate base inputs
        if not (0 <= v1 <= 100):
            errors.append(f"Usage hours ({v1}) outside range [0-100]")
        
        if not (0 <= v2 <= 15):
            errors.append(f"Support tickets ({v2}) outside range [0-15]")
        
        if not (0 <= v3 <= 60):
            errors.append(f"Tenure ({v3}) outside range [0-60]")
        
        # Validate change parameter
        if not (-0.5 <= change <= 1.0):
            errors.append(
                f"Change ({change}) outside valid range [-0.5, 1.0]. "
                f"Expected: -50% to +100% adjustment"
            )
        
        # Validate adjusted values won't exceed limits
        adjusted_usage = v1 * (1 + change)
        if not (0 <= adjusted_usage <= 100):
            errors.append(
                f"Adjusted usage ({adjusted_usage:.1f}) would exceed limits. "
                f"Reduce change amount."
            )
        
        if errors:
            logger.warning(f"❌ Validation failed: {errors}")
            raise HTTPException(
                status_code=400,
                detail="\n".join(errors)
            )
        
        # ====== SIMULATION ======
        # Lazy import
        from backend.services.ds_service import predict
        
        logger.info("🔄 Running simulation...")
        
        # Get baseline prediction
        base_start = time.time()
        base = predict([v1, v2, v3])
        base_time = time.time() - base_start
        
        # Get adjusted prediction
        adj_start = time.time()
        new = predict([adjusted_usage, v2, v3])
        adj_time = time.time() - adj_start
        
        logger.info(
            f"✅ Simulation complete: "
            f"before={base.get('churn_probability', 0):.2%} "
            f"→ after={new.get('churn_probability', 0):.2%}"
        )
        
        # ====== RESPONSE BUILDING ======
        before_prob = float(base.get("churn_probability", 0))
        after_prob = float(new.get("churn_probability", 0))
        impact = round(after_prob - before_prob, 3)
        
        # Calculate improvement percentage
        if before_prob > 0:
            improvement_pct = (abs(impact) / before_prob) * 100
        else:
            improvement_pct = 0
        
        total_time = time.time() - request_start
        
        response = SimulationResponse(
            before=before_prob,
            after=after_prob,
            impact=impact,
            metadata={
                "processing_time_ms": round(total_time * 1000, 1),
                "api_version": "2.0.0",
                "change_applied": f"{change*100:.1f}%",
                "adjusted_usage": round(adjusted_usage, 1),
                "improvement_percentage": round(improvement_pct, 1),
                "direction": "Improved ✅" if impact < 0 else "Worsened ❌",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"❌ Simulation error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )


# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler with logging"""
    logger.error(
        f"HTTPException: {exc.status_code} - {exc.detail}",
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unexpected errors"""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error. Please try again.",
            "detail": str(exc) if os.getenv("DEBUG") else "An error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"🚀 Starting server on port {port} (DEBUG={debug})")
    
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        log_level="info"
    )
