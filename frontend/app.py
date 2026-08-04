from __future__ import annotations

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --ink-900:#0b1220; --ink-700:#1e293b; --ink-500:#64748b; --ink-300:#cbd5e1;
        --surface:#ffffff; --surface-muted:#f8fafc; --border:#e6e9f0;
        --accent:#4f46e5; --accent-soft:#eef2ff;
        --danger:#dc2626; --warn:#ea580c; --caution:#ca8a04; --ok:#16a34a;
        --shadow-sm: 0 1px 2px rgba(15,23,42,.04), 0 1px 1px rgba(15,23,42,.03);
        --shadow-md: 0 4px 10px rgba(15,23,42,.06), 0 2px 4px rgba(15,23,42,.04);
        --shadow-lg: 0 12px 28px rgba(15,23,42,.10), 0 4px 10px rgba(15,23,42,.05);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--ink-700);
    }
    code, .stCode, [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

    .stApp { background: var(--surface-muted); }
    .block-container { padding-top: 2rem; max-width: 1180px; }

    /* ── Header band ── */
    .header-band {
        background: linear-gradient(120deg, #0b1220 0%, #1e2a4a 55%, #4f46e5 130%);
        color: white;
        padding: 40px 44px;
        border-radius: 18px;
        margin-bottom: 32px;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    .header-band::after {
        content: "";
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,.12), transparent 70%);
    }
    .header-band .eyebrow {
        display:inline-block; font-size:11px; font-weight:700; letter-spacing:.12em;
        text-transform:uppercase; color:#a5b4fc; margin-bottom:10px;
    }
    .header-band h1 { margin:0 0 8px 0; font-size:30px; font-weight:800; letter-spacing:-.01em; }
    .header-band p  { margin:0; opacity:.78; font-size:15px; font-weight:400; max-width:560px; }

    /* ── Force readable text inside bordered "cards" only (this is where the
         original white-on-white bug occurred). Scoping it here — rather than
         to the whole app — avoids fighting the header band's own white-text
         rule with a specificity war. ── */
    [data-testid="stVerticalBlockBorderWrapper"] :is(p, span, li, label, h1, h2, h3, h4, h5),
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"],
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"],
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"],
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] * {
        color: var(--ink-700) !important;
    }
    .header-band, .header-band * { color: white !important; }
    .header-band .eyebrow { color: #a5b4fc !important; }
    .driver-pill { color: var(--accent) !important; }

    /* ── Cards: style Streamlit's native st.container(border=True) ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        box-shadow: var(--shadow-sm);
        margin-bottom: 20px;
        transition: box-shadow .15s ease;
        background: var(--surface) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow: var(--shadow-md); }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px 12px;
        box-shadow: var(--shadow-sm);
    }
    [data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; color: var(--ink-900); }
    [data-testid="stMetricLabel"] { font-size: 11px; color: var(--ink-500); text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }

    /* ── Risk callouts ── */
    .risk-critical, .risk-high, .risk-medium, .risk-low {
        border-radius: 14px; padding: 20px 24px; box-shadow: var(--shadow-sm);
        border-left: none; position: relative; overflow: hidden;
    }
    .risk-critical { background:#fef2f2; }
    .risk-high     { background:#fff7ed; }
    .risk-medium   { background:#fefce8; }
    .risk-low      { background:#f0fdf4; }
    .risk-critical::before, .risk-high::before, .risk-medium::before, .risk-low::before {
        content:""; position:absolute; left:0; top:0; bottom:0; width:5px;
    }
    .risk-critical::before { background: var(--danger); }
    .risk-high::before     { background: var(--warn); }
    .risk-medium::before   { background: var(--caution); }
    .risk-low::before      { background: var(--ok); }

    /* ── Driver pills ── */
    .driver-pill {
        display: inline-block;
        background: var(--accent-soft);
        border: 1px solid #d8ddfd;
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 3px 5px 3px 0;
        color: var(--accent);
    }

    /* ── Divider ── */
    .divider { margin: 32px 0; border-top: 1px solid var(--border); }

    /* ── Action card ── */
    .action-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        box-shadow: var(--shadow-sm);
    }
    .action-card strong { color: var(--ink-900); }

    /* ── Buttons ── */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
        transition: all .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border: none;
        color: white !important;
    }
    .stButton > button[kind="primary"] p { color: white !important; }
    .stButton > button[kind="primary"]:hover { background:#4338ca; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--ink-900);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stRadio label {
        border-radius: 8px; padding: 6px 10px; transition: background .12s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,.06); }

    /* ── Tables ── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; box-shadow: var(--shadow-sm); }

    /* ── Fade-in on results ── */
    .risk-critical, .risk-high, .risk-medium, .risk-low {
        animation: fadeUp .35s ease both;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ── Config ───────────────────────────────────────────────────────────────────
try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = "https://ai-customer-churn-intelligence.onrender.com"

TIMEOUT = 30   # seconds


# ── Session state ─────────────────────────────────────────────────────────────
class _S(str, Enum):
    IDLE    = "idle"
    LOADING = "loading"
    ERROR   = "error"
    SUCCESS = "success"


def _init_state() -> None:
    defaults = {
        "result":       None,
        "state":        _S.IDLE,
        "error_msg":    None,
        "last_run":     None,
        "analyses_done": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Backend communication ─────────────────────────────────────────────────────
def _backend_ok() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _analyze(v1: float, v2: float, v3: float) -> Dict[str, Any]:
    r = requests.post(
        f"{API_URL}/analyze",
        params={"v1": v1, "v2": v2, "v3": v3},
        timeout=TIMEOUT,
    )
    if r.status_code == 200:
        return r.json()
    err = r.json().get("error", r.json().get("detail", "Backend error"))
    raise ValueError(err) if r.status_code == 400 else Exception(err)


def _simulate(v1: float, v2: float, v3: float, change: float) -> Dict[str, Any]:
    r = requests.post(
        f"{API_URL}/simulate",
        params={"v1": v1, "v2": v2, "v3": v3, "change": change / 100},
        timeout=TIMEOUT,
    )
    if r.status_code == 200:
        return r.json()
    err = r.json().get("error", r.json().get("detail", "Simulation failed"))
    raise Exception(err)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_dot(prob: float) -> str:
    if prob >= 0.80: return "●"
    if prob >= 0.60: return "●"
    if prob >= 0.40: return "●"
    return "●"


def _risk_color(prob: float) -> str:
    if prob >= 0.80: return "#dc2626"
    if prob >= 0.60: return "#ea580c"
    if prob >= 0.40: return "#ca8a04"
    return "#16a34a"


def _risk_css_class(prob: float) -> str:
    if prob >= 0.80: return "risk-critical"
    if prob >= 0.60: return "risk-high"
    if prob >= 0.40: return "risk-medium"
    return "risk-low"


def _customer_story(prob: float, risk: str) -> str:
    """Plain-English headline about what the score means for this customer."""
    if prob >= 0.85:
        return (
            "This customer is in serious trouble. The warning signs are stacking up — "
            "low engagement, friction with support, or a very short runway. "
            "Without a targeted intervention in the next few days, they're likely gone."
        )
    if prob >= 0.65:
        return (
            "Red flags are showing. This customer hasn't reached the point of no return, "
            "but the trajectory is moving in the wrong direction. A focused outreach "
            "this week could change the outcome."
        )
    if prob >= 0.40:
        return (
            "There are a few early signals worth watching. Nothing alarming yet, "
            "but this customer would benefit from a proactive check-in to make sure "
            "they're getting value from the product."
        )
    if prob >= 0.20:
        return (
            "This customer looks stable. Engagement is solid and there are no major "
            "friction signals. Keep up the good work and look for ways to deepen "
            "the relationship."
        )
    return (
        "This is a healthy, engaged customer. They're getting real value from the "
        "product. Consider whether there's an opportunity to expand their usage "
        "or turn them into an advocate."
    )


# ── Plotly helpers ────────────────────────────────────────────────────────────
def _gauge(prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": _risk_color(prob), "family": "JetBrains Mono"}},
        title={"text": "Churn Risk Score", "font": {"size": 14, "color": "#64748b"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar":  {"color": _risk_color(prob), "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(22,163,74,.12)"},
                {"range": [30, 60], "color": "rgba(202,138,4,.12)"},
                {"range": [60, 80], "color": "rgba(234,88,12,.12)"},
                {"range": [80,100], "color": "rgba(220,38,38,.12)"},
            ],
            "threshold": {
                "line": {"color": "#dc2626", "width": 2},
                "thickness": 0.75,
                "value": 60,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin={"t": 30, "b": 0, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans"},
    )
    return fig


def _before_after_chart(before: float, after: float, change_pct: float) -> go.Figure:
    labels = ["Before intervention", "After intervention"]
    values = [round(before * 100, 1), round(after * 100, 1)]
    colors = [_risk_color(before), _risk_color(after)]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        width=0.4,
    ))
    fig.update_layout(
        title=f"Simulated impact of +{change_pct:.0f}% usage increase",
        yaxis={"title": "Churn Probability (%)", "range": [0, max(values) * 1.3]},
        xaxis={"title": ""},
        height=320,
        margin={"t": 50, "b": 40, "l": 50, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font={"family": "Plus Jakarta Sans"},
    )
    return fig


# ── Page sections ─────────────────────────────────────────────────────────────
def _render_header() -> None:
    st.markdown("""
    <div class="header-band">
        <span class="eyebrow">Churn Intelligence Platform</span>
        <h1>AI Customer Churn Intelligence</h1>
        <p>Predict churn risk, understand the drivers behind it, and get an AI-generated
        retention strategy — before you pick up the phone.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_inputs() -> tuple[float, float, float]:
    """Render the customer input sliders and return (v1, v2, v3)."""
    with st.container(border=True):
        st.markdown("### Customer Profile")
        st.caption("Adjust the sliders to match this customer's metrics.")

        c1, c2, c3 = st.columns(3)
        with c1:
            v1 = st.slider(
                "Monthly Usage Hours",
                min_value=0, max_value=100, value=50,
                help="How many hours per month does this customer actively use the product?",
            )
            st.caption("0 = no engagement · 100 = power user")
        with c2:
            v2 = st.slider(
                "Support Tickets / Month",
                min_value=0, max_value=15, value=5,
                help="Average number of support tickets raised each month.",
            )
            st.caption("0 = self-sufficient · 15 = constant friction")
        with c3:
            v3 = st.slider(
                "Tenure (months)",
                min_value=0, max_value=60, value=24,
                help="How many months has this customer been with you?",
            )
            st.caption("0 = brand new · 60 = 5-year loyal customer")

    return float(v1), float(v2), float(v3)


def _render_result(result: Dict[str, Any], v1: float, v2: float, v3: float) -> None:
    """Render the full analysis result panel."""
    ds   = result.get("ds_output", {})
    expl = result.get("explanation", "")
    prob = float(ds.get("churn_probability", 0))
    risk = ds.get("risk_level", "Unknown")
    drivers   = ds.get("drivers", [])
    interp    = ds.get("interpretation", "")
    conf      = float(ds.get("confidence_score", 0))
    pctile    = float(ds.get("percentile", 0))

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### Analysis Results")

    # ── Customer story headline ──
    css_class = _risk_css_class(prob)
    story     = _customer_story(prob, risk)
    st.markdown(
        f'<div class="{css_class}">'
        f'<strong style="color:{_risk_color(prob)}">{_risk_dot(prob)}</strong> '
        f'<strong>{risk} Risk — {prob*100:.1f}% Churn Probability</strong>'
        f'<br><br>{story}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacer

    # ── Metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Churn Probability", f"{prob*100:.1f}%")
    m2.metric("Risk Level",        risk)
    m3.metric("Model Confidence",  f"{conf*100:.0f}%")
    m4.metric("Risk Percentile",   f"Top {pctile:.0f}%")

    # ── Gauge + interpretation ──
    with st.container(border=True):
        g_col, i_col = st.columns([1, 1])
        with g_col:
            st.plotly_chart(_gauge(prob), use_container_width=True, key="gauge_chart")
        with i_col:
            st.markdown("#### What this score means")
            st.markdown(f"**{interp}**")
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("**Risk signals detected:**")
            driver_html = " ".join(
                f'<span class="driver-pill">{d}</span>' for d in drivers
            )
            st.markdown(driver_html, unsafe_allow_html=True)

    # ── AI Explanation ──
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### AI Retention Strategy")
    with st.expander("Read the full analysis and action plan", expanded=True):
        st.markdown(expl)

    # ── What-if simulation ──
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### What-If Simulation")
        st.caption("Estimate how much churn risk would drop if this customer increased their usage.")

        sim_change = st.slider(
            "Simulate usage increase (%)",
            min_value=5, max_value=100, value=30, step=5,
            key="sim_slider",
        )

        if st.button("Run Simulation", key="sim_btn", type="primary"):
            with st.spinner("Running simulation …"):
                try:
                    sim = _simulate(v1, v2, v3, float(sim_change))
                    before = float(sim["before"])
                    after  = float(sim["after"])
                    impact = float(sim["impact"])
                    impr   = abs(sim.get("metadata", {}).get("improvement_percent", abs(impact * 100)))

                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Risk Before", f"{before*100:.1f}%")
                    sc2.metric("Risk After",  f"{after*100:.1f}%",  delta=f"{impact*100:+.1f}%")
                    sc3.metric("Reduction",   f"{impr:.1f}%")

                    st.plotly_chart(
                        _before_after_chart(before, after, float(sim_change)),
                        use_container_width=True,
                        key="sim_chart",
                    )

                    if impact < -0.15:
                        st.success(
                            f"**Strong impact.** Raising usage by {sim_change}% could cut "
                            f"churn risk by {abs(impact)*100:.1f} percentage points."
                        )
                    elif impact < 0:
                        st.info(
                            "**Moderate impact.** Usage improvement helps, but other factors "
                            "may have more leverage — check the driver list above."
                        )
                    else:
                        st.warning(
                            "**Limited impact.** Usage alone may not move the needle here. "
                            "Focus on addressing the specific risk drivers listed."
                        )

                except Exception as exc:
                    st.error(f"Simulation failed: {exc}")


def _render_single_analysis_page() -> None:
    """Main single-customer analysis page."""
    v1, v2, v3 = _render_inputs()

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("Analyse This Customer", type="primary", use_container_width=True)

    if run_clicked:
        st.session_state["state"] = _S.LOADING
        with st.spinner("Thinking … this usually takes 5–10 seconds …"):
            try:
                t0 = time.time()
                result = _analyze(v1, v2, v3)
                elapsed = time.time() - t0

                st.session_state["result"]        = result
                st.session_state["state"]         = _S.SUCCESS
                st.session_state["error_msg"]     = None
                st.session_state["last_run"]      = datetime.now()
                st.session_state["analyses_done"] += 1

                st.success(f"Analysis complete in {elapsed:.1f} s")

            except ValueError as exc:
                st.session_state["state"]     = _S.ERROR
                st.session_state["error_msg"] = str(exc)
            except ConnectionError:
                st.session_state["state"]     = _S.ERROR
                st.session_state["error_msg"] = (
                    "Cannot reach the backend API. "
                    "It may still be waking up — wait 30 seconds and try again."
                )
            except TimeoutError:
                st.session_state["state"]     = _S.ERROR
                st.session_state["error_msg"] = (
                    "The request timed out. The backend is probably waking up on "
                    "Render's free tier (takes ~30 s). Please try again in a moment."
                )
            except Exception as exc:
                st.session_state["state"]     = _S.ERROR
                st.session_state["error_msg"] = str(exc)

    # ── Show error if any ──
    if st.session_state["state"] == _S.ERROR:
        err = st.session_state.get("error_msg", "Unknown error")
        st.error(err)

    # ── Show result ──
    if st.session_state["state"] == _S.SUCCESS and st.session_state["result"]:
        _render_result(st.session_state["result"], v1, v2, v3)


def _render_batch_page() -> None:
    """Batch CSV upload page."""
    with st.container(border=True):
        st.markdown("## Batch Analysis")
        st.markdown("Upload a CSV to analyse multiple customers in one go.")

        st.info(
            "**CSV format:** columns must be named `v1`, `v2`, `v3` "
            "(or `usage_hours`, `support_tickets`, `tenure_months`)."
        )

        sample_csv = "usage_hours,support_tickets,tenure_months\n75,2,36\n15,10,8\n50,5,24"
        st.download_button("Download sample CSV", sample_csv, "sample.csv", "text/csv")

        uploaded = st.file_uploader("Choose your CSV file", type="csv")
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    # Normalise column names
    col_map = {
        "usage_hours": "v1", "usage": "v1",
        "support_tickets": "v2", "tickets": "v2",
        "tenure_months": "v3", "tenure": "v3",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    missing = {"v1", "v2", "v3"} - set(df.columns)
    if missing:
        st.error(f"CSV is missing columns: {', '.join(missing)}")
        return

    st.success(f"Loaded {len(df)} customers")

    progress = st.progress(0)
    status   = st.empty()
    results  = []

    for idx, row in df.iterrows():
        try:
            res = _analyze(float(row["v1"]), float(row["v2"]), float(row["v3"]))
            ds  = res.get("ds_output", {})
            results.append({
                "Customer #":    idx + 1,
                "Usage hrs":     round(float(row["v1"]), 1),
                "Support tkts":  int(round(float(row["v2"]))),
                "Tenure (mo)":   int(round(float(row["v3"]))),
                "Churn %":       round(float(ds.get("churn_probability", 0)) * 100, 1),
                "Risk":          ds.get("risk_level", "Unknown"),
                "Interpretation": ds.get("interpretation", ""),
            })
        except Exception as exc:
            status.warning(f"Row {idx+1} failed: {exc}")

        progress.progress((idx + 1) / len(df))
        status.text(f"Processing customer {idx+1} of {len(df)} …")
        time.sleep(0.25)   # gentle rate limiting

    progress.empty()
    status.empty()

    if not results:
        st.error("No valid records could be processed.")
        return

    rdf = pd.DataFrame(results)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### Summary")

    high   = len(rdf[rdf["Churn %"] >= 60])
    medium = len(rdf[(rdf["Churn %"] >= 30) & (rdf["Churn %"] < 60)])
    low    = len(rdf[rdf["Churn %"] < 30])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("High Risk",   high)
    s2.metric("Medium Risk", medium)
    s3.metric("Low Risk",    low)
    s4.metric("Avg Churn",   f"{rdf['Churn %'].mean():.1f}%")

    with st.container(border=True):
        fig = px.histogram(
            rdf, x="Churn %", nbins=20,
            title="Churn Probability Distribution",
            color_discrete_sequence=["#4f46e5"],
        )
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"family": "Plus Jakarta Sans"})
        st.plotly_chart(fig, use_container_width=True, key="batch_hist")

    st.dataframe(
        rdf.sort_values("Churn %", ascending=False),
        use_container_width=True, height=400,
    )

    st.download_button(
        "Download results (CSV)",
        rdf.to_csv(index=False),
        f"churn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
    )


def _render_sidebar(backend_online: bool, v1: float = 0, v2: float = 0, v3: float = 0) -> str:
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Select mode",
            ["Single Analysis", "Batch Upload"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Backend status
        if backend_online:
            st.success("Backend: online")
        else:
            st.warning(
                "Backend offline.\n\n"
                "If you're using Render's free tier, wait ~30 s for it to wake up, "
                "then try again."
            )

        # Live profile insights
        if v1 or v2 or v3:
            from backend.core.validators import get_profile_insights
            try:
                insights = get_profile_insights(v1, v2, v3)
                st.markdown("---")
                st.markdown("**Profile insights:**")
                for tip in insights:
                    st.markdown(f"- {tip}")
            except Exception:
                pass

        st.markdown("---")
        st.markdown("""
**How It Works**
1. Enter customer metrics
2. Click Analyse
3. Read the AI strategy
4. Run what-if simulations

**Risk signals to watch:**
- Usage < 20 h/mo → disengagement
- Tickets > 7/mo → friction
- Tenure < 18 mo → evaluation window
""")

        st.markdown("---")
        analyses = st.session_state.get("analyses_done", 0)
        if analyses > 0:
            st.metric("Analyses this session", analyses)

    return page


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_state()

    # Check backend health once per session (not on every rerun)
    if "backend_ok" not in st.session_state:
        st.session_state["backend_ok"] = _backend_ok()
    backend_ok: bool = st.session_state["backend_ok"]

    # We need v1/v2/v3 for the sidebar insights — read from session or use defaults
    result = st.session_state.get("result")
    last_v1 = float(result["ds_output"].get("churn_probability", 0)) if result else 0.0

    page = _render_sidebar(backend_ok)

    _render_header()

    if page == "Single Analysis":
        _render_single_analysis_page()
    else:
        _render_batch_page()

    st.markdown("""
    <div class="divider"></div>
    <div style='text-align:center;color:#94a3b8;font-size:12px;padding-bottom:20px;'>
        Churn Intelligence Platform · FastAPI + Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
