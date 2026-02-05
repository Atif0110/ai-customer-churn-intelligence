import json
import os
from datetime import datetime

FILE = "data/history.json"


def _ensure_file():

    # Create folder if missing
    os.makedirs("data", exist_ok=True)

    # Create file if missing
    if not os.path.exists(FILE):
        with open(FILE, "w") as f:
            json.dump([], f)


def save_record(values, result):

    _ensure_file()

    # Load safely
    try:
        with open(FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []

    # Append new record
    data.append({
        "time": datetime.now().isoformat(),
        "inputs": values,
        "churn_probability": result["churn_probability"],
        "risk_level": result["risk_level"],
    })

    # Save back
    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_history():

    _ensure_file()

    try:
        with open(FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []
