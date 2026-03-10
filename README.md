# 📉 AI Customer Churn Intelligence Platform

### ML + Generative AI System for Churn Prediction, Explanation, and Retention Strategy

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat\&logo=fastapi\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat\&logo=streamlit\&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat\&logo=scikit-learn\&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.1-8B-blueviolet?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An end-to-end **Decision Intelligence platform** that combines **Machine Learning + Generative AI** to predict customer churn, explain risk drivers, and recommend targeted retention strategies.

> **Beyond prediction — this platform turns churn scores into business actions.**

---

# ⭐ Key Features

• ML-based churn prediction using **Logistic Regression**
• Rule-based **driver detection** for interpretable churn signals
• **LLM-generated retention strategies** using Llama 3.1
• **Scenario simulation** to estimate the impact of interventions
• **FastAPI backend** with modular service architecture
• **Interactive Streamlit dashboard** for analysts
• **LLM provider abstraction** (Groq / OpenAI / Gemini ready)

---

# ⚡ Quick Start

Clone and run the project in a few commands.

```bash
git clone https://github.com/your-username/ai-customer-churn-intelligence.git
cd ai-customer-churn-intelligence

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Start backend:

```bash
uvicorn backend.api.main:app --reload
```

Start frontend:

```bash
streamlit run frontend/app.py
```

Backend → http://127.0.0.1:8000
Frontend → http://localhost:8501

---

# 🎬 Demo

### Customer Risk Dashboard

*(Add screenshot here)*

### Churn Risk Analysis

*(Add screenshot here)*

### Intervention Simulation

*(Add screenshot here)*

Tip: record a **10-second GIF** of the app using Loom or ScreenToGif and embed it here.

---

# 📋 Table of Contents

* Overview
* Architecture
* Tech Stack
* Project Structure
* How It Works
* API Reference
* Installation
* Future Work
* License

---

# 🚀 Overview

Customer churn is one of the most expensive problems for subscription businesses.

Many ML systems stop at a **prediction score**.
Business teams actually need answers to three questions:

| Question                  | Platform Output                     |
| ------------------------- | ----------------------------------- |
| **Who is at risk?**       | Churn probability score + risk tier |
| **Why are they at risk?** | Identified churn drivers            |
| **What should we do?**    | AI-generated retention strategy     |
| **What if we act?**       | Simulation of intervention impact   |

This project demonstrates a **Decision Intelligence workflow** used by modern data teams to move from **prediction → explanation → action**.

---

# 🏗 Architecture

```
Streamlit Frontend
        │
        │ HTTP
        ▼
FastAPI Backend
        │
        ▼
Orchestrator Service
        │
 ┌──────────────┬───────────────┐
 │ Data Science │  GenAI Layer  │
 │   Service    │  (LLM)        │
 └──────────────┴───────────────┘
        │
        ▼
History Logging Service
        │
        ▼
LLM Provider Factory
(Groq / OpenAI / Gemini)
```

Design principles:

• Modular services
• Provider-agnostic LLM layer
• Clear separation of concerns
• Replaceable components

---

# 🛠 Tech Stack

| Layer         | Technology         |
| ------------- | ------------------ |
| Frontend      | Streamlit          |
| Backend       | FastAPI            |
| ML Model      | scikit-learn       |
| LLM           | Llama 3.1 via Groq |
| Visualization | Matplotlib         |
| Config        | python-dotenv      |
| Language      | Python 3.10+       |

---

# 📁 Project Structure

```
ai-customer-churn-intelligence/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── services/
│   │   ├── ds_service.py
│   │   ├── genai_service.py
│   │   ├── orchestrator.py
│   │   └── history_service.py
│   │
│   └── llm/
│       ├── base.py
│       ├── factory.py
│       └── providers/
│           └── groq_provider.py
│
├── frontend/
│   └── app.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# 🧠 How It Works

## 1️⃣ Data Science Layer

A **Logistic Regression model** predicts churn probability using:

| Feature         | Meaning               |
| --------------- | --------------------- |
| Usage Hours     | Product engagement    |
| Support Tickets | Customer support load |
| Tenure          | Time as a customer    |

Risk tiers:

🔴 High → >60%
🟡 Medium → 30–60%
🟢 Low → <30%

Driver detection identifies signals such as:

• Low product usage
• High support volume
• Short customer tenure

---

## 2️⃣ GenAI Layer

The model output is passed to an LLM that acts as a **customer retention strategist**.

The AI generates:

1. Explanation of churn risk
2. Recommended retention actions
3. Business impact summary

Temperature is kept low to ensure **consistent, structured responses**.

---

## 3️⃣ Scenario Simulation

The `/simulate` endpoint models intervention impact.

Example:

```
Before: 78% churn risk
After intervention: 51%

Impact: −27%
```

This allows teams to **test strategies before executing them.**

---

# 📡 API Reference

Base URL

```
http://127.0.0.1:8000
```

| Method | Endpoint  | Description             |
| ------ | --------- | ----------------------- |
| GET    | /         | Root                    |
| GET    | /health   | Backend status          |
| POST   | /analyze  | Full churn analysis     |
| POST   | /simulate | Intervention simulation |

Example:

```
POST /analyze?v1=2&v2=5&v3=4
```

Response:

```json
{
  "ds_output": {
    "churn_probability": 0.87,
    "risk_level": "High",
    "drivers": ["Low usage", "High support load"]
  },
  "explanation": "..."
}
```

---

# 🔭 Future Work

• Replace synthetic data with a real churn dataset
• Add SHAP model explainability
• Support additional LLM providers
• Add batch scoring endpoint
• Docker deployment
• Authentication layer

---

# 📄 License

MIT License

---

# 👨‍💻 Author

**Atif**

Machine Learning • Data Systems • Decision Intelligence

GitHub
https://github.com/your-username

LinkedIn
https://linkedin.com/in/your-handle

Open to **Data Analyst / Data Scientist / ML Engineer roles**.
