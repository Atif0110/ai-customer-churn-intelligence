<div align="center">

# 🧠 AI Customer Churn Decision Intelligence Platform

**Predict • Explain • Retain**

An end-to-end **Decision Intelligence** system that combines **Data Science + Generative AI** to predict customer churn, explain risk drivers, and recommend retention actions.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI%20%7C%20Gemini-8B5CF6?style=for-the-badge)](https://groq.com/)

**🔗 Live Demo:** [Streamlit App](https://ai-customer-churn-intelligence.streamlit.app/)  
**💻 GitHub:** [Repository](https://github.com/Atif0110/ai-customer-churn-intelligence)  
**🚀 API:** [FastAPI Backend](https://ai-customer-churn-intelligence.onrender.com)

</div>

---

## 💡 Why This Project?

> Predicting churn is useful — **acting on it is what creates real value.**

Most churn models stop at a probability score. This platform goes further by turning raw predictions into **business-ready decisions** using Generative AI — simulating a real **Decision Intelligence workflow** used in modern SaaS companies.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Churn Prediction** | ML-powered predictions using behavioral signals |
| 🚦 **Risk Segmentation** | Automatic Low / Medium / High classification |
| 🔍 **Driver Identification** | Explains *why* a customer is at risk |
| 🤖 **AI Retention Advice** | LLM-generated, personalized strategies |
| 🔮 **What-If Simulation** | Scenario modeling for proactive decisions |
| 📊 **Batch Processing** | Analyze 100+ customers in seconds |
| 🏗️ **Production-Ready** | FastAPI backend + Streamlit UI |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      Streamlit UI (Frontend)         │  ← Interactive Dashboard
│  - Single Customer Analysis          │
│  - Batch Upload Processing           │
│  - Real-time Visualizations          │
└──────────────┬──────────────────────┘
               │ HTTP/REST
┌──────────────▼──────────────────────┐
│    FastAPI Backend (Production)      │
│  - /analyze endpoint                 │
│  - /simulate endpoint                │
│  - /health check                     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Orchestrator Service            │
│                                      │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ DS Model │  │ Driver Detection │ │
│  │(sklearn) │  │     Logic        │ │
│  └──────────┘  └──────────────────┘ │
│                                      │
│  ┌──────────────────────────────┐   │
│  │   LLM Provider Layer         │   │
│  │  Groq | OpenAI | Gemini      │   │
│  └──────────────────────────────┘   │
│                                      │
│  ┌──────────────────────────────┐   │
│  │   History & Analytics        │   │
│  │   Storage Layer              │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

---

## 🧠 How It Works

### **1️⃣ Data Science Layer**
A **Logistic Regression model** predicts churn probability using behavioral signals:

```python
Features:
├─ 📊 Monthly usage hours (0-100)
├─ 🎫 Support ticket volume (0-15)
└─ 📅 Customer tenure (0-60 months)

Training Data: 15 realistic customer profiles
Accuracy: ~95% on test data
```

### **2️⃣ Explainability Layer**
Rule-based logic identifies key **churn drivers**:

```
Rules:
├─ Low usage (< 20 hours)     → Disengagement signal
├─ High tickets (> 7/month)   → Friction/frustration signal
└─ Short tenure (< 18 months) → Early-exit risk
```

### **3️⃣ GenAI Layer**
Large Language Model generates **contextualized explanations**:

```
Input: Churn probability + Risk drivers
       ↓
LLM (Groq Llama 3.1 8B)
       ↓
Output:
├─ Why this customer might churn
├─ Two retention actions
└─ Business impact analysis
```

### **4️⃣ Decision Intelligence Layer**
Scenario simulation for **proactive decision-making**:

```
Question: "What if we increase their usage by 30%?"
          ↓
Model predicts new churn probability
          ↓
Shows impact: 70% → 50% churn (-20%)
```

---

## 📁 Project Structure

```
ai-customer-churn-intelligence/
│
├── frontend/
│   ├── app.py                 # Streamlit UI (single + batch analysis)
│   └── requirements.txt
│
├── backend/
│   ├── api/
│   │   └── main.py           # FastAPI routes (/analyze, /simulate, /health)
│   │
│   ├── services/
│   │   ├── orchestrator.py    # Workflow orchestration
│   │   ├── ds_service.py      # ML model + predictions
│   │   ├── genai_service.py   # LLM integration
│   │   └── history_service.py # Prediction history
│   │
│   ├── llm/
│   │   ├── base.py            # LLM abstraction
│   │   ├── factory.py         # Provider factory
│   │   └── providers/
│   │       └── groq_provider.py
│   │
│   └── requirements.txt
│
├── data/
│   └── history.json           # Prediction history (auto-generated)
│
├── .env                        # Environment variables (not committed)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### **Prerequisites**
- Python 3.9+
- Groq API key (free tier available)
- Git

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/Atif0110/ai-customer-churn-intelligence.git
cd ai-customer-churn-intelligence
```

### **2️⃣ Create Virtual Environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4️⃣ Configure Environment Variables**

Create `.env` file in project root:

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
```

Get free API key: [Groq Console](https://console.groq.com/)

### **5️⃣ Run Locally**

**Backend (FastAPI):**
```bash
uvicorn backend.api.main:app --reload --port 8000
```
📖 API Docs: http://127.0.0.1:8000/docs

**Frontend (Streamlit):**
```bash
streamlit run frontend/app.py
```
🎨 UI: http://localhost:8501

---

## 🚀 Deployment

### **Backend (Render)**

```bash
# Already deployed to:
https://ai-customer-churn-intelligence.onrender.com

# Redeploy on code changes:
1. Push to GitHub main branch
2. Render auto-builds and deploys
3. Check dashboard for status
```

### **Frontend (Streamlit Cloud)**

```bash
# Already deployed to:
https://ai-customer-churn-intelligence.streamlit.app/

# Configure secrets in Streamlit Cloud:
Settings → Secrets
Add: API_URL = https://ai-customer-churn-intelligence.onrender.com
```

---

## 📖 API Documentation

### **1. Health Check**

```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "message": "Backend is healthy"
}
```

### **2. Analyze Customer**

```bash
POST /analyze?v1=50&v2=5&v3=25
```

**Parameters:**
- `v1` (float): Monthly usage hours [0-100]
- `v2` (float): Support tickets [0-15]
- `v3` (float): Tenure months [0-60]

**Response:**
```json
{
  "ds_output": {
    "churn_probability": 0.445,
    "risk_level": "Medium",
    "drivers": ["Early-stage customer"]
  },
  "explanation": "This customer shows early adoption characteristics..."
}
```

### **3. Simulate Intervention**

```bash
POST /simulate?v1=50&v2=5&v3=25&change=0.3
```

**Parameters:**
- `v1, v2, v3`: Customer metrics (same as analyze)
- `change` (float): Usage increase [-0.5 to 1.0] (e.g., 0.3 = +30%)

**Response:**
```json
{
  "before": 0.445,
  "after": 0.315,
  "impact": -0.13
}
```

---

## 🎯 Using the App

### **Mode 1: Single Customer Analysis**

1. **Set sliders:**
   - Usage Hours (0-100)
   - Support Tickets (0-15)
   - Tenure (0-60 months)

2. **Click "Analyze"**
   - See churn probability
   - View risk classification
   - Read AI explanation

3. **Test Scenarios (What-If):**
   - Move simulation slider
   - Click "Simulate Impact"
   - See how interventions affect churn

### **Mode 2: Batch Upload**

1. **Prepare CSV** with columns:
   ```
   v1,v2,v3
   95,1,58
   50,5,25
   15,10,8
   ```

2. **Upload file**
   - Select CSV
   - Wait for processing (30 customers: ~30 seconds)

3. **View Results:**
   - Summary metrics (High/Medium/Low risk counts)
   - Distribution chart
   - Full results table
   - Download predictions

---

## 📊 Sample Results

**Input:** 30 diverse customers

**Output:**
```
🔴 High Risk:    21 customers (70-100% churn)
🟡 Medium Risk:   0 customers (30-70% churn)
🟢 Low Risk:      9 customers (0-30% churn)
📈 Avg Churn:     69.6%
```

**Example Predictions:**
```
Customer 1:  v1=95, v2=1,  v3=58 → 8.5%  🟢 LOW
Customer 14: v1=3,  v2=15, v3=1  → 98.2% 🔴 HIGH
Customer 19: v1=50, v2=5,  v3=25 → 45.1% 🟡 MEDIUM
```

---

## 🤖 Supported LLM Providers

| Provider | Speed | Cost | Setup |
|----------|-------|------|-------|
| **Groq** ⭐ | ⚡ Fastest | Free tier | API key |
| **OpenAI** | 🔥 Fast | Paid | API key |
| **Gemini** | 🚀 Fast | Free tier | API key |
| **Local** | 🛠️ Variable | Free | Ollama |

**Current:** Groq (Llama 3.1 8B Instant)

---

## 📈 Model Performance

```
Training Data:     15 customer profiles (realistic)
Test Accuracy:     ~95%
Prediction Speed:  < 5ms per customer
Memory Usage:      < 1MB
Scalability:       Handles 1000s of predictions
```

**Test Cases (All Passing ✅):**
- Loyal customers (95+ usage, 1-2 tickets, 40+ months) → 5-15% churn
- Medium customers (40-60 usage, 4-6 tickets) → 30-50% churn
- At-risk customers (10-30 usage, 8-12 tickets) → 60-80% churn
- Critical customers (< 10 usage, 13+ tickets, < 6 months) → 85-99% churn

---

## 🔒 Security & Privacy

✅ **No user data stored** (except analysis history)
✅ **No API key exposure** (environment variables)
✅ **CORS enabled** (only trusted origins)
✅ **Input validation** (0-100, 0-15, 0-60 ranges)
✅ **Error handling** (safe error messages)
✅ **HTTPS** (production deployment)

---

## 🐛 Troubleshooting

### **Backend Offline**
```bash
# Check health
curl https://api-url/health

# Redeploy on Render
git push origin main
```

### **CSV Upload Error**
```
Make sure CSV has columns: v1, v2, v3
Or: usage_hours, support_tickets, tenure_months
```

### **Slider Scrolls to Top**
```
Hard refresh: Ctrl+Shift+R
Clear cache: Settings → Clear Cache
```

### **LLM Not Generating Text**
```
Check API key in Render environment variables
Verify GROQ_API_KEY is set correctly
```

---

## 📚 For Interview Discussions

### **Technical Highlights:**
- **ML Model:** Logistic Regression on realistic customer data
- **Feature Engineering:** Usage, tickets, tenure → churn probability
- **LLM Integration:** Groq API for real-time explanations
- **What-If Analysis:** Scenario modeling for decision support
- **Full Stack:** FastAPI + Streamlit + Render + Streamlit Cloud

### **Business Impact:**
- Identify churn risk BEFORE customers leave
- Personalized retention actions via AI
- Scenario testing to optimize interventions
- Batch processing for large customer bases

### **Decision Intelligence Concept:**
```
Prediction (What will happen?)
      ↓
Explanation (Why will it happen?)
      ↓
Recommendation (What should we do?)
      ↓
Simulation (What if we do this?)
      ↓
ACTION (Actually do it!)
```

---

## 🗺️ Roadmap

- [ ] XGBoost/ensemble model for higher accuracy
- [ ] SHAP-based feature importance explanations
- [ ] Multi-tenant dashboard
- [ ] Email alert integration
- [ ] Historical trend visualization
- [ ] A/B testing framework for retention tactics

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**Mohd Atif**
- 🎓 B.S. Data Science & Applications, IIT Madras (2025)
- 📍 Lucknow, India
- 🔗 [LinkedIn](https://linkedin.com/in/mohd-atif01)
- 💻 [GitHub](https://github.com/Atif0110)

---

<div align="center">

**⭐ If this project helped you, please star it on GitHub!**

Built to demonstrate how **Machine Learning and LLMs work together** to create real business value — not just predictions.

**Last Updated:** March 2025

</div>
