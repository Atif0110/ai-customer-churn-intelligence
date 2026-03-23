import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 32px; font-weight: 800; }
    [data-testid="stMetricLabel"] { font-size: 13px; }
    .risk-high { color: #ef4444; font-weight: 700; }
    .risk-medium { color: #f59e0b; font-weight: 700; }
    .risk-low { color: #10b981; font-weight: 700; }
    .section-divider { margin: 30px 0; border-top: 2px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

API = st.secrets.get("API_URL", "https://ai-customer-churn-intelligence.onrender.com")

# Initialize session state variables
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "sim_slider_value" not in st.session_state:
    st.session_state.sim_slider_value = 10

def check_backend():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def get_risk_color(probability: float) -> str:
    if probability >= 0.6:
        return "#ef4444"
    elif probability >= 0.3:
        return "#f59e0b"
    return "#10b981"

def get_risk_label(probability: float) -> str:
    if probability >= 0.6:
        return "🔴 HIGH RISK"
    elif probability >= 0.3:
        return "🟡 MEDIUM RISK"
    return "🟢 LOW RISK"

def create_gauge_chart(prob: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Churn Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": get_risk_color(prob)},
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.2)"}
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 4},
                "thickness": 0.8,
                "value": 60
            }
        },
        number={"suffix": "%", "font": {"size": 32, "color": get_risk_color(prob)}}
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=80, b=20))
    return fig

def create_driver_chart(drivers: list):
    if not drivers:
        return None
    
    importance = [0.45, 0.30, 0.20][:len(drivers)]
    colors = ["#ef4444", "#f59e0b", "#3b82f6"][:len(drivers)]
    
    fig = go.Figure(data=[
        go.Bar(
            y=drivers,
            x=importance,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.0%}" for v in importance],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Churn Risk Drivers",
        xaxis_title="Impact",
        height=300,
        margin=dict(l=200, r=80, t=50, b=20),
        showlegend=False,
    )
    return fig

def create_comparison_chart(before: float, after: float):
    fig = go.Figure(data=[
        go.Bar(
            x=["Current Risk", "After Change"],
            y=[before * 100, after * 100],
            marker_color=[get_risk_color(before), get_risk_color(after)],
            text=[f"{before*100:.1f}%", f"{after*100:.1f}%"],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Impact of Usage Increase",
        yaxis_title="Churn Probability (%)",
        height=350,
        showlegend=False,
    )
    return fig

def analyze_customer(v1: float, v2: float, v3: float):
    try:
        if not (0 <= v1 <= 100 and 0 <= v2 <= 15 and 0 <= v3 <= 60):
            st.error("❌ Invalid values. Check slider ranges.")
            return None
        
        with st.spinner("🔄 Analyzing customer..."):
            res = requests.post(
                f"{API}/analyze",
                params={"v1": v1, "v2": v2, "v3": v3},
                timeout=30
            )
        
        if res.status_code == 400:
            st.error(f"❌ Invalid input: {res.json().get('detail', 'Unknown error')}")
            return None
        elif res.status_code != 200:
            st.error(f"❌ Backend error ({res.status_code}). Please try again.")
            return None
        
        return res.json()
    
    except requests.Timeout:
        st.error("⏱️ Request timeout. Backend might be loading. Please try again.")
        return None
    except requests.ConnectionError:
        st.error("🔌 Connection error. Check if backend is online.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def simulate_intervention(v1: float, v2: float, v3: float, change_percent: float):
    try:
        change = change_percent / 100.0
        
        with st.spinner("🔄 Running simulation..."):
            res = requests.post(
                f"{API}/simulate",
                params={"v1": v1, "v2": v2, "v3": v3, "change": change},
                timeout=30
            )
        
        if res.status_code == 400:
            st.error(f"❌ Invalid input: {res.json().get('detail', 'Unknown error')}")
            return None
        elif res.status_code != 200:
            st.error(f"❌ Backend error ({res.status_code}). Please try again.")
            return None
        
        return res.json()
    
    except requests.Timeout:
        st.error("⏱️ Simulation timeout. Please try again.")
        return None
    except requests.ConnectionError:
        st.error("🔌 Connection error. Check if backend is online.")
        return None
    except Exception as e:
        st.error(f"❌ Simulation error: {str(e)}")
        return None

def render_header():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("# 📊 Churn Intelligence Platform")
        st.markdown("*AI-Powered Prediction & Retention Strategy*")
    with col3:
        backend_status = check_backend()
        if backend_status:
            st.success("Backend Connected", icon="✅")
        else:
            st.error("Backend Offline", icon="❌")

def single_analysis_page():
    st.markdown("## 👤 Single Customer Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        v1 = st.slider("Monthly Usage Hours", 0.0, 100.0, 50.0, step=5.0, key="usage_slider")
    with col2:
        v2 = st.slider("Support Tickets", 0.0, 15.0, 5.0, step=1.0, key="tickets_slider")
    with col3:
        v3 = st.slider("Tenure (Months)", 0.0, 60.0, 24.0, step=1.0, key="tenure_slider")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    if st.button("🔍 Analyze", use_container_width=True, key="analyze_btn"):
        result = analyze_customer(v1, v2, v3)
        if result:
            st.session_state.analysis_result = {
                "data": result,
                "v1": v1,
                "v2": v2,
                "v3": v3
            }
    
    # Display analysis results if available
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result["data"]
        stored_v1 = st.session_state.analysis_result["v1"]
        stored_v2 = st.session_state.analysis_result["v2"]
        stored_v3 = st.session_state.analysis_result["v3"]
        
        ds = result.get("ds_output", {})
        prob = ds.get("churn_probability", 0)
        risk = ds.get("risk_level", "Unknown")
        drivers = ds.get("drivers", [])
        explanation = result.get("explanation", "")
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
        with metric_col2:
            st.markdown(f"<div style='text-align: center; padding-top: 10px;'><p style='font-size: 12px; margin-bottom: 5px;'>RISK LEVEL</p><p class='risk-{'high' if prob >= 0.6 else 'medium' if prob >= 0.3 else 'low'}'>{get_risk_label(prob)}</p></div>", unsafe_allow_html=True)
        with metric_col3:
            st.metric("Drivers Identified", len(drivers))
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns([1.2, 1])
        with chart_col1:
            st.plotly_chart(create_gauge_chart(prob), use_container_width=True, key="gauge_chart")
        with chart_col2:
            if drivers:
                st.plotly_chart(create_driver_chart(drivers), use_container_width=True, key="driver_chart")
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### 🤖 AI-Generated Retention Strategy")
        st.info(explanation)
        
        # ============ WHAT-IF SIMULATION - FIXED PROPERLY ============
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### 🎯 What-If Simulation")
        
        sim_col1, sim_col2 = st.columns([1, 1])
        
        with sim_col1:
            # Slider with persistent session state - NO PAGE RELOAD
            slider_value = st.slider(
                "Increase Usage by (%)",
                0, 100,
                st.session_state.sim_slider_value,
                step=5,
                key="simulation_slider_final"
            )
            st.session_state.sim_slider_value = slider_value
        
        with sim_col2:
            simulate_btn = st.button(
                "📊 Simulate Impact",
                use_container_width=True,
                key="simulate_btn_final"
            )
        
        # Show simulation results
        if simulate_btn:
            sim = simulate_intervention(stored_v1, stored_v2, stored_v3, st.session_state.sim_slider_value)
            
            if sim:
                before_prob = sim.get("before", prob)
                after_prob = sim.get("after", prob)
                impact = sim.get("impact", 0)
                
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                
                sim_metric1, sim_metric2, sim_metric3 = st.columns(3)
                with sim_metric1:
                    st.metric("New Probability", f"{after_prob*100:.1f}%")
                with sim_metric2:
                    st.metric("Risk Change", f"{impact*100:.1f}%")
                with sim_metric3:
                    st.metric("Improvement", f"{abs(impact)*100:.1f}%")
                
                st.plotly_chart(
                    create_comparison_chart(before_prob, after_prob),
                    use_container_width=True,
                    key="comparison_final"
                )

def batch_upload_page():
    st.markdown("## 📁 Batch Customer Analysis")
    st.info("📋 Upload CSV with columns: v1, v2, v3 (or usage_hours, support_tickets, tenure_months)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            columns_map = {
                'v1': 'v1', 'usage_hours': 'v1', 'usage': 'v1',
                'v2': 'v2', 'support_tickets': 'v2', 'tickets': 'v2',
                'v3': 'v3', 'tenure_months': 'v3', 'tenure': 'v3'
            }
            
            df_renamed = df.rename(columns={old: new for old, new in columns_map.items() if old in df.columns})
            
            if 'v1' not in df_renamed.columns or 'v2' not in df_renamed.columns or 'v3' not in df_renamed.columns:
                st.error("❌ CSV must have: v1, v2, v3 (or usage_hours, support_tickets, tenure_months)")
                return
            
            st.info(f"Processing {len(df_renamed)} customers...")
            progress = st.progress(0)
            results = []
            
            for idx, row in df_renamed.iterrows():
                result = analyze_customer(
                    float(row.get("v1", 0)),
                    float(row.get("v2", 0)),
                    float(row.get("v3", 0))
                )
                
                if result:
                    ds = result.get("ds_output", {})
                    results.append({
                        "ID": idx + 1,
                        "V1": round(row.get("v1", 0), 2),
                        "V2": round(row.get("v2", 0), 2),
                        "V3": round(row.get("v3", 0), 2),
                        "Churn %": round(ds.get("churn_probability", 0) * 100, 1),
                        "Risk": ds.get("risk_level", "Unknown")
                    })
                
                progress.progress((idx + 1) / len(df_renamed))
            
            if results:
                results_df = pd.DataFrame(results)
                
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                st.markdown("### 📊 Summary")
                
                high = len(results_df[results_df["Churn %"] >= 60])
                medium = len(results_df[(results_df["Churn %"] >= 30) & (results_df["Churn %"] < 60)])
                low = len(results_df[results_df["Churn %"] < 30])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔴 High Risk", high)
                with col2:
                    st.metric("🟡 Medium Risk", medium)
                with col3:
                    st.metric("🟢 Low Risk", low)
                with col4:
                    st.metric("📈 Avg Churn", f"{results_df['Churn %'].mean():.1f}%")
                
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                
                fig = px.histogram(
                    results_df,
                    x="Churn %",
                    nbins=15,
                    title="Churn Distribution",
                    color_discrete_sequence=["#3b82f6"]
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                st.dataframe(results_df.sort_values("Churn %", ascending=False), use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"churn_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")

def main():
    render_header()
    
    with st.sidebar:
        st.markdown("### 📌 Navigation")
        page = st.radio("Select Mode", ["Single Analysis", "Batch Upload"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        **AI Customer Churn Intelligence**
        
        - ML Churn Prediction
        - LLM Retention Strategy
        - What-If Simulation
        - Batch Processing
        """)
    
    st.markdown("---")
    
    if page == "Single Analysis":
        single_analysis_page()
    else:
        batch_upload_page()
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #808080; font-size: 11px;'>© 2025 Churn Intelligence | FastAPI + Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
