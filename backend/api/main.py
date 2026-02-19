from fastapi import FastAPI
from fastapi.responses import JSONResponse
from backend.services.orchestrator import run_analysis
from backend.services.ds_service import predict

app = FastAPI(
    title="AI Customer Churn Decision Intelligence API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "AI Customer Churn API is running"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return JSONResponse(content={"status": "ok"})

@app.post("/analyze")
def analyze(v1: float, v2: float, v3: float):
    return run_analysis([v1, v2, v3])


@app.post("/simulate")
def simulate(v1: float, v2: float, v3: float, change: float):

    base = predict([v1, v2, v3])
    new = predict([v1 * (1 + change), v2, v3])

    return {
        "before": base["churn_probability"],
        "after": new["churn_probability"],
        "impact": round(
            new["churn_probability"] - base["churn_probability"], 3
        ),
    }
