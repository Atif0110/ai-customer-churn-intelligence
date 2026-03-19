<div align="center">
 
# 🧠 AI Customer Churn Decision Intelligence Platform
 
**Predict. Explain. Retain.**
 
An end-to-end **Decision Intelligence** system that combines **Data Science + Generative AI** to predict customer churn, explain risk drivers, and recommend retention actions.
 
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI%20%7C%20Gemini-8B5CF6?style=for-the-badge&logo=openai&logoColor=white)](#supported-llm-providers)
 
</div>
 
---
 
## 💡 Why This Project?
 
> Predicting churn is useful — **acting on it is what creates real value.**
 
Most churn models stop at a probability score. This platform goes further by turning raw predictions into **business-ready decisions** using Generative AI — simulating a real **Decision Intelligence workflow** used in modern SaaS companies.
 
---
 
## ✨ Key Features
 
| Feature | Description |
|---|---|
| 🎯 **Churn Probability Scoring** | ML-powered prediction using customer behavioral signals |
| 🚦 **Risk Segmentation** | Automatic Low / Medium / High classification |
| 🔍 **Driver Identification** | Explains *why* a customer is at risk |
| 🤖 **AI Retention Advice** | LLM-generated, personalized retention strategies |
| 🔮 **What-If Simulation** | Scenario modeling to guide proactive decisions |
| 🏗️ **Full-Stack Deployment** | Production-ready FastAPI backend + Streamlit UI |
 
---
 
## 🏗️ Architecture
 
```
┌─────────────────────────────────────┐
│           Streamlit UI              │  ← Interactive Dashboard
└──────────────┬──────────────────────┘
               │ HTTP
┌──────────────▼──────────────────────┐
│          FastAPI Backend            │  ← REST API Layer
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Orchestrator Service         │
│                                     │
│  ┌──────────┐  ┌─────────────────┐  │
│  │ DS Model │  │ Explainability  │  │
│  │(sklearn) │  │     Logic       │  │
│  └──────────┘  └─────────────────┘  │
│                                     │
│  ┌─────────────────────────────┐    │
│  │     LLM Provider Layer      │    │
│  │  Groq │ OpenAI │ Gemini │..  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```
 
---
 
## 🧠 How It Works
 
### 1️⃣ Data Science Layer
A **Logistic Regression model** predicts churn probability using key behavioral signals:
 
- 📊 Monthly usage hours
- 🎫 Support ticket volume
- 📅 Customer tenure
 
### 2️⃣ Explainability Layer
Rule-based logic identifies and surfaces the **key churn drivers** in plain language:
 
- ⚠️ High support volume → friction signal
- 📉 Low product usage → disengagement signal
- 🕐 Short tenure → early-exit risk
 
> This makes predictions **interpretable for business users**, not just data scientists.
 
### 3️⃣ GenAI Layer
A Large Language Model generates contextualized, human-readable output:
 
- 📝 Churn reasoning narrative
- 💼 Tailored retention strategies
- 📈 Business impact analysis
 
### 4️⃣ Decision Intelligence Layer
Scenario simulation allows users to ask **"what if?" questions**:
 
- *"What if this customer increases usage by 20%?"*
- *"What if we resolve their open tickets?"*
 
This supports **proactive, data-driven decision-making** at the account level.
 
---
 
## 📁 Project Structure
 
```
├── backend/
│   ├── api/              # FastAPI route definitions
│   ├── services/         # DS + GenAI orchestration logic
│   ├── llm/              # LLM provider abstraction layer
│   ├── ml/               # Model training & inference
│   └── data/             # Prediction history storage
│
├── frontend/
│   └── app.py            # Streamlit UI application
│
├── .env                  # Environment variables (not committed)
└── requirements.txt      # Python dependencies
```
 
---
 
## ⚙️ Setup & Installation
 
### Prerequisites
- Python 3.9+
- An API key from a supported LLM provider
 
### 1️⃣ Clone the Repository
 
```bash
git clone https://github.com/your-username/churn-decision-intelligence.git
cd churn-decision-intelligence
```
 
### 2️⃣ Create a Virtual Environment
 
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# macOS / Linux
source venv/bin/activate
```
 
### 3️⃣ Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4️⃣ Configure Environment Variables
 
Create a `.env` file in the project root:
 
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
```
 
> **Supported providers:** `groq`, `openai`, `gemini`, `local`
 
---
 
## ▶️ Running the Application
 
### Start the Backend (FastAPI)
 
```bash
uvicorn backend.api.main:app --reload
```
 
The API will be available at: **`http://127.0.0.1:8000`**
 
> 📖 Interactive API docs: `http://127.0.0.1:8000/docs`
 
### Start the Frontend (Streamlit)
 
```bash
streamlit run frontend/app.py
```
 
The UI will open at: **`http://localhost:8501`**
 
---
 
## 🤖 Supported LLM Providers
 
| Provider | Speed | Notes |
|---|---|---|
| **Groq** | ⚡ Fastest | Recommended for development |
| **OpenAI** | 🔥 Fast | GPT-4o / GPT-3.5 Turbo |
| **Gemini** | 🚀 Fast | Google Gemini Pro |
| **Local** | 🛠️ Variable | Ollama or custom endpoints |
 
---
 
## 🗺️ Roadmap
 
- [ ] XGBoost / ensemble model upgrade
- [ ] SHAP-based feature importance
- [ ] Multi-tenant customer dashboard
- [ ] Email alert integration
- [ ] Historical trend visualization
 
---
 
## 🤝 Contributing
 
Contributions are welcome! Please open an issue or submit a pull request.
 
---
 
<div align="center">
 
Built to demonstrate how **Machine Learning and LLMs can work together** to support real business decisions — not just predictions.
 
⭐ Star this repo if you find it useful!
 
</div>
