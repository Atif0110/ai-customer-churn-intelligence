import streamlit as st
import requests
import matplotlib.pyplot as plt

API = st.secrets["API_URL"]

st.title("📉 AI Customer Churn Intelligence Platform")

# Health check
try:
    r = requests.get(f"{API}/health", timeout=3)
    msg = ("✅ Backend Connected"
           if r.status_code == 200
           else "⚠️ Backend issue")
    st.success(msg)
except requests.RequestException:
    st.error("❌ Backend Offline")

# Inputs
st.header("🔍 Customer Profile")

v1 = st.number_input("Monthly Usage Hours", 0.0, 100.0, 5.0)
v2 = st.number_input("Support Tickets This Month", 0.0, 20.0, 1.0)
v3 = st.number_input("Months as Customer", 0.0, 120.0, 12.0)

# ---- Prediction ----
if st.button("🚀 Predict Churn Risk"):

    res = requests.post(
        f"{API}/analyze",
        params={"v1": v1, "v2": v2, "v3": v3},
    ).json()

    ds = res["ds_output"]

    st.subheader("📊 Churn Prediction")

    st.metric(
        "Churn Probability",
        f"{ds['churn_probability']*100:.1f}%"
    )

    st.metric("Risk Level", ds["risk_level"])

    # ---- Drivers ----
    st.subheader("🔍 Key Drivers")

    for d in ds["drivers"]:
        st.write("•", d)

    # ---- Chart ----
    fig, ax = plt.subplots()
    ax.bar(["Churn Risk"], [ds["churn_probability"]])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    st.subheader("🤖 AI Retention Advice")
    st.write(res["explanation"])

# ---- Simulation ----
st.header("🧪 What-If Simulation")

change = st.slider("Increase Usage (%)", 0, 100, 10) / 100

if st.button("Simulate Impact"):

    res = requests.post(
        f"{API}/simulate",
        params={
            "v1": v1,
            "v2": v2,
            "v3": v3,
            "change": change,
        },
    ).json()

    st.write(
        f"Churn changes from "
        f"**{res['before']*100:.1f}%** "
        f"to **{res['after']*100:.1f}%**"
    )

