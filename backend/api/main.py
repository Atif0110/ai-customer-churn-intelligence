from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- App --------------------
app = FastAPI(
    title="AI Customer Churn Decision Intelligence API",
    version="1.0.2",
    docs_url="/docs",
    redoc_url="/redoc"
)

# -------------------- Middleware --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Startup --------------------
@app.on_event("startup")
def startup_event():
    logger.info("🚀 FastAPI app started successfully")

# -------------------- Root --------------------
@app.get("/")
def root():
    return {
        "message": "AI Customer Churn Intelligence API is running",
        "version": "1.0.2",
        "docs": "/docs"
    }

# -------------------- Health (FIXED) --------------------
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "message": "Backend is healthy"}

# -------------------- Analyze --------------------
@app.post("/analyze")
def analyze(v1: float, v2: float, v3: float):
    try:
        # Lazy import (prevents startup crash)
        from backend.services.orchestrator import run_analysis

        if not (0 <= v1 <= 1000 and 0 <= v2 <= 100 and 0 <= v3 <= 120):
            raise HTTPException(
                status_code=400,
                detail="Invalid values. Expected: 0≤v1≤1000, 0≤v2≤100, 0≤v3≤120"
            )

        logger.info(f"Analyzing: v1={v1}, v2={v2}, v3={v3}")

        result = run_analysis([v1, v2, v3])

        logger.info("Analysis complete")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# -------------------- Simulation --------------------
@app.post("/simulate")
def simulate(v1: float, v2: float, v3: float, change: float):
    try:
        # Lazy import (prevents startup crash)
        from backend.services.ds_service import predict

        if not (0 <= v1 <= 1000 and 0 <= v2 <= 100 and 0 <= v3 <= 120):
            raise HTTPException(
                status_code=400,
                detail="Invalid values. Expected: 0≤v1≤1000, 0≤v2≤100, 0≤v3≤120"
            )

        if not (-0.5 <= change <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="Invalid change value. Expected: -0.5 ≤ change ≤ 1.0"
            )

        logger.info(f"Simulating: v1={v1}, v2={v2}, v3={v3}, change={change}")

        base = predict([v1, v2, v3])
        new = predict([v1 * (1 + change), v2, v3])

        response = {
            "before": float(base.get("churn_probability", 0)),
            "after": float(new.get("churn_probability", 0)),
            "impact": round(
                float(new.get("churn_probability", 0)) -
                float(base.get("churn_probability", 0)), 3
            ),
        }

        logger.info("Simulation complete")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

# -------------------- Global Exception Handler --------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# -------------------- Local Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
