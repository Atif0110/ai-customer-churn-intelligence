from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.services.orchestrator import run_analysis
from backend.services.ds_service import predict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Customer Churn Decision Intelligence API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "AI Customer Churn Intelligence API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    try:
        logger.info("Health check requested")
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "message": "Backend is healthy"}
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/analyze")
def analyze(v1: float, v2: float, v3: float):
    try:
        if not (0 <= v1 <= 1000 and 0 <= v2 <= 100 and 0 <= v3 <= 120):
            raise HTTPException(
                status_code=400,
                detail="Invalid values. Expected: 0≤v1≤1000, 0≤v2≤100, 0≤v3≤120"
            )
        
        logger.info(f"Analyzing: v1={v1}, v2={v2}, v3={v3}")
        result = run_analysis([v1, v2, v3])
        logger.info(f"Analysis complete: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
def simulate(v1: float, v2: float, v3: float, change: float):
    try:
        if not (0 <= v1 <= 1000 and 0 <= v2 <= 100 and 0 <= v3 <= 120):
            raise HTTPException(
                status_code=400,
                detail="Invalid values. Expected: 0≤v1≤1000, 0≤v2≤100, 0≤v3≤120"
            )
        
        if not (-0.5 <= change <= 1.0):
            raise HTTPException(
                status_code=400,
                detail="Invalid change value. Expected: -0.5 ≤ change ≤ 1.0 (-50% to +100%)"
            )
        
        logger.info(f"Simulating: v1={v1}, v2={v2}, v3={v3}, change={change}")
        
        base = predict([v1, v2, v3])
        new = predict([v1 * (1 + change), v2, v3])
        
        response = {
            "before": float(base["churn_probability"]),
            "after": float(new["churn_probability"]),
            "impact": round(
                float(new["churn_probability"]) - float(base["churn_probability"]), 3
            ),
        }
        
        logger.info(f"Simulation result: {response}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
