# 📉 AI Customer Churn Decision Intelligence Platform

An end-to-end **Decision Intelligence** system that combines **Data Science + Generative AI** to predict customer churn, explain risk drivers, and recommend retention actions.

This project demonstrates how **Machine Learning and LLMs can work together to support real business decisions — not just predictions.**

---

# 🚀 Overview

Customer churn is a major challenge for subscription and SaaS businesses.  
Predicting churn is useful — but **acting on it is what creates real value.**

This platform goes beyond prediction by providing:

- ✅ Churn probability scoring  
- ✅ Risk segmentation (Low / Medium / High)  
- ✅ Key driver identification  
- ✅ AI-generated retention advice  
- ✅ What-if scenario simulation  
- ✅ Full-stack deployment (API + UI)

It simulates a real **Decision Intelligence workflow** used in modern companies.

---

# 🧠 How It Works

## 1️⃣ Data Science Layer

A **Logistic Regression model** predicts churn using:

- Monthly usage hours  
- Support tickets  
- Customer tenure  

### Outputs
- Churn probability  
- Risk level classification  

---

## 2️⃣ Explainability Layer

Rule-based logic identifies key churn drivers:

- High support volume  
- Low product usage  
- Short tenure  

This makes predictions **interpretable for business users.**

---

## 3️⃣ GenAI Layer

A Large Language Model (Groq / OpenAI / Gemini compatible) generates:

- Churn reasoning  
- Retention strategies  
- Business impact analysis  

This turns raw predictions into **actionable insights.**

---

## 4️⃣ Decision Intelligence Layer

Scenario simulation allows users to test:

- "What if usage increases?"  
- "What if engagement improves?"  

This supports **proactive decision-making.**

---

# 🏗 Architecture

Streamlit UI
↓
FastAPI Backend
↓
Orchestrator Service
├── DS Model (scikit-learn)
├── Explainability Logic
└── LLM Provider Layer


- Modular design  
- Provider-agnostic LLM integration  
- Easy to extend and maintain  

### Supported LLM Providers

- Groq  
- OpenAI  
- Gemini  
- Local models  

---

# 📁 Project Structure

backend/
│
├── api/ # FastAPI endpoints
├── services/ # DS + GenAI logic
├── llm/ # LLM provider abstraction
├── ml/ # Model logic
└── data/ # History storage

frontend/
└── app.py # Streamlit UI

---

# ⚙️ Installation & Setup

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Configure Environment Variables

Create a .env file:

GROQ_API_KEY=your_key_here
LLM_PROVIDER=groq

▶️ Running the Project
Start Backend
uvicorn backend.api.main:app --reload


Backend runs at:

http://127.0.0.1:8000

Start Frontend
streamlit run frontend/app.py


UI opens at:

http://localhost:8501