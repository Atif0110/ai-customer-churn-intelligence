from backend.services.ds_service import predict
from backend.services.genai_service import generate_explanation
from backend.services.history_service import save_record


def run_analysis(values):

    ds = predict(values)

    save_record(values, ds)

    explanation = generate_explanation(ds)

    return {
        "ds_output": ds,
        "explanation": explanation,
    }
