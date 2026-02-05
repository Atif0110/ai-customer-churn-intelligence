import numpy as np
from sklearn.linear_model import LogisticRegression

# Fake churn-like dataset
X = np.array([
    [5, 1, 24],
    [10, 0, 36],
    [2, 5, 6],
    [8, 0, 48],
    [1, 7, 3],
    [6, 1, 30],
])

# 1 = churn, 0 = stay
y = np.array([0, 0, 1, 0, 1, 0])

model = LogisticRegression()
model.fit(X, y)


def predict(values):

    arr = np.array(values).reshape(1, -1)

    prob = model.predict_proba(arr)[0][1]

    # ---- Feature Drivers ----
    v1, v2, v3 = values
    drivers = []

    if v1 < 5:
        drivers.append("Low product usage")

    if v2 > 3:
        drivers.append("High support tickets")

    if v3 < 12:
        drivers.append("Short customer tenure")

    if not drivers:
        drivers.append("Stable usage behavior")

    return {
        "churn_probability": float(prob),
        "risk_level": (
            "High" if prob > 0.6
            else "Medium" if prob > 0.3
            else "Low"
        ),
        "drivers": drivers,
    }
