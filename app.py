import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title="Energy Client EDA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  h1 { color: #f0f2f6; font-size: 2rem !important; }
  h2 { color: #c9d1e0; }
  h3 { color: #a0aabb; font-size: 1rem !important; font-weight: 500; }
  .metric-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-val { font-size: 1.8rem; font-weight: 700; color: #4fc3f7; }
  .metric-lbl { font-size: 0.75rem; color: #7a8499; text-transform: uppercase; letter-spacing: .05em; }
  .section-title {
    font-size: 1.05rem; font-weight: 600; color: #c9d1e0;
    border-left: 3px solid #4fc3f7; padding-left: .6rem;
    margin-bottom: 0.8rem; margin-top: 0.5rem;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    background: #1e2130; border-radius: 8px 8px 0 0;
    padding: .4rem 1rem; color: #7a8499; font-size: .88rem;
  }
  .stTabs [aria-selected="true"] { background: #2d3250 !important; color: #4fc3f7 !important; }
  img { border-radius: 8px; }
  .stSelectbox label { color: #c9d1e0 !important; }
  .stDataFrame { background: #1e2130; }
</style>
""", unsafe_allow_html=True)

# ── Paths ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
PLOT_DIR = os.path.join(BASE, "results", "eda_plots")
DATA_DIR = os.path.join(BASE, "Data")
EXCLUDE = {"weather.csv"}


@st.cache_data(show_spinner="Loading client data…")
def load_clients():
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and f not in EXCLUDE
    )
    out = {}
    for f in files:
        cid = f.replace(".csv", "")
        df = pd.read_csv(
            os.path.join(DATA_DIR, f),
            parse_dates=["datetime"],
            usecols=["datetime", "consumption", "status", "generation"],
        )
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        out[cid] = df
    return out


@st.cache_data(show_spinner="Computing summary…")
def compute_summary(clients):
    rows = []
    for cid, df in clients.items():
        total = len(df)
        valid = df["consumption"].notna().sum()
        rows.append({
            "Client": cid,
            "Records": total,
            "Completeness %": round(valid / total * 100, 1),
            "Missing %": round((total - valid) / total * 100, 1),
            "Mean kW": round(df["consumption"].mean(), 3),
            "Std kW": round(df["consumption"].std(), 3),
            "Max kW": round(df["consumption"].max(), 3),
            "Zero %": round((df["consumption"] == 0).sum() / total * 100, 1),
            "Start": df["datetime"].min().date(),
            "End": df["datetime"].max().date(),
        })
    return pd.DataFrame(rows).set_index("Client")


def show_plot(fname, caption=None, use_container_width=True):
    path = os.path.join(PLOT_DIR, fname)
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=use_container_width)
    else:
        st.warning(f"Plot not found: {fname}")


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Energy Clients — EDA Dashboard")
st.markdown("Exploratory data analysis of **56 smart-meter clients** (hourly, 2023–2025)")
st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Overview",
    "📊 Consumption",
    "🕐 Temporal",
    "🔍 Outliers",
    "🌡️ Weather",
    "🔗 Cross-Client",
    "🤖 DL Suitability",
    "🔎 Client Explorer",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – Dataset Overview
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    clients = load_clients()
    summary = compute_summary(clients)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(clients), "Clients"),
        (c2, f"{summary['Records'].sum():,}", "Total Records"),
        (c3, f"{summary['Completeness %'].mean():.1f}%", "Avg Completeness"),
        (c4, f"{summary['Mean kW'].mean():.2f}", "Fleet Mean kW"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<p class="section-title">Data Completeness per Client</p>', unsafe_allow_html=True)
    show_plot("01_completeness.png")

    st.markdown('<p class="section-title">Missing Values per Client</p>', unsafe_allow_html=True)
    show_plot("02_missing_values.png")

    st.markdown('<p class="section-title">Recording Timeline per Client</p>', unsafe_allow_html=True)
    show_plot("03_timeline.png")

    st.markdown('<p class="section-title">Status Code Distribution (Measured / Estimated / Adjusted)</p>', unsafe_allow_html=True)
    show_plot("04_status_distribution.png")

    with st.expander("📄 Full summary table"):
        st.dataframe(summary.style.background_gradient(cmap="Blues", subset=["Completeness %"]), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – Consumption Statistics
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-title">Consumption Distribution — Box Plots</p>', unsafe_allow_html=True)
    show_plot("05_boxplots_consumption.png")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Mean ± Std per Client</p>', unsafe_allow_html=True)
        show_plot("06_mean_std_consumption.png")
    with col2:
        st.markdown('<p class="section-title">Coefficient of Variation</p>', unsafe_allow_html=True)
        show_plot("08_coefficient_variation.png")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="section-title">Skewness & Kurtosis</p>', unsafe_allow_html=True)
        show_plot("07_skewness_kurtosis.png")
    with col4:
        st.markdown('<p class="section-title">Zero Consumption Rate</p>', unsafe_allow_html=True)
        show_plot("09_zero_consumption.png")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 – Temporal Patterns
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-title">Daily Time Series — Representative Clients</p>', unsafe_allow_html=True)
    show_plot("10_timeseries_daily.png")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Hourly Consumption Profile</p>', unsafe_allow_html=True)
        show_plot("11_hourly_profile.png")
    with col2:
        st.markdown('<p class="section-title">Day-of-Week Pattern</p>', unsafe_allow_html=True)
        show_plot("12_dow_pattern.png")

    st.markdown('<p class="section-title">Monthly Pattern</p>', unsafe_allow_html=True)
    show_plot("13_monthly_pattern.png")

    st.markdown('<p class="section-title">Hour × Month Heatmap</p>', unsafe_allow_html=True)
    show_plot("14_heatmap_hour_month.png")

    st.markdown('<p class="section-title">Seasonal Decomposition (Trend / Seasonal / Residual)</p>', unsafe_allow_html=True)
    show_plot("15_seasonal_decomposition.png")

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 – Outlier Analysis
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-title">Outlier Rate per Client (IQR & Z-Score methods)</p>', unsafe_allow_html=True)
    show_plot("16_outliers.png")

    st.markdown('<p class="section-title">Consumption vs Time — Outliers Highlighted</p>', unsafe_allow_html=True)
    show_plot("17_outlier_scatter.png")

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 – Weather & Correlation
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-title">Weather Variables Overview</p>', unsafe_allow_html=True)
    show_plot("18_weather_overview.png")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Consumption–Weather Correlation Heatmap</p>', unsafe_allow_html=True)
        show_plot("19_weather_corr_heatmap.png")
    with col2:
        st.markdown('<p class="section-title">Temperature vs Consumption</p>', unsafe_allow_html=True)
        show_plot("20_temp_vs_consumption.png")

    st.markdown('<p class="section-title">ACF / PACF — Autocorrelation Structure</p>', unsafe_allow_html=True)
    show_plot("21_acf_pacf.png")

# ────────────────────────────────────────────────────────────────────────────
# TAB 6 – Cross-Client Analysis
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<p class="section-title">Cross-Client Consumption Correlation Matrix</p>', unsafe_allow_html=True)
    show_plot("23_cross_client_corr.png")
    st.info(
        "Highly correlated clients share similar usage patterns — useful for federated "
        "learning grouping strategies. Low correlation pairs are strong candidates for diverse local models."
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 7 – DL Suitability
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<p class="section-title">Deep-Learning Suitability Score per Client</p>', unsafe_allow_html=True)
    st.caption("Score ≥ 70 → good  |  45–70 → moderate  |  < 45 → poor")
    show_plot("24_dl_score.png")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Radar Chart — Top-15 DL Clients</p>', unsafe_allow_html=True)
        show_plot("25_radar_top_clients.png")
    with col2:
        st.markdown('<p class="section-title">Data Quality Heatmap</p>', unsafe_allow_html=True)
        show_plot("26_quality_heatmap.png")

# ────────────────────────────────────────────────────────────────────────────
# TAB 8 – Interactive Client Explorer
# ────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    clients = load_clients()

    st.markdown("### Select a client to explore")
    client_ids = sorted(clients.keys())
    selected = st.selectbox("Client ID", client_ids, label_visibility="collapsed")

    df = clients[selected].copy()
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    total = len(df)
    valid = df["consumption"].notna().sum()
    for col, val, lbl in [
        (c1, f"{total:,}", "Records"),
        (c2, f"{valid/total*100:.1f}%", "Completeness"),
        (c3, f"{df['consumption'].mean():.3f}", "Mean kW"),
        (c4, f"{df['consumption'].max():.3f}", "Peak kW"),
        (c5, f"{(df['consumption']==0).sum()/total*100:.1f}%", "Zero Rate"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Time series
    daily = df.groupby("date")["consumption"].mean().reset_index()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=daily["date"], y=daily["consumption"],
        mode="lines", name="Daily avg",
        line=dict(color="#4fc3f7", width=1.5),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.08)",
    ))
    fig_ts.update_layout(
        title=f"Daily Average Consumption — Client {selected}",
        xaxis_title="Date", yaxis_title="Consumption (kW)",
        template="plotly_dark", height=320,
        margin=dict(t=40, b=30, l=50, r=20),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Hourly profile
        hourly = df.groupby("hour")["consumption"].mean().reset_index()
        fig_h = go.Figure(go.Bar(
            x=hourly["hour"], y=hourly["consumption"],
            marker_color="#4fc3f7", marker_line_width=0,
        ))
        fig_h.update_layout(
            title="Avg Hourly Profile",
            xaxis_title="Hour of Day", yaxis_title="kW",
            template="plotly_dark", height=280,
            margin=dict(t=40, b=30, l=50, r=20),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_r:
        # Day-of-week
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = df.groupby("dow")["consumption"].mean().reset_index()
        dow["day"] = dow["dow"].map(lambda x: dow_labels[x])
        fig_d = go.Figure(go.Bar(
            x=dow["day"], y=dow["consumption"],
            marker_color="#7e57c2", marker_line_width=0,
        ))
        fig_d.update_layout(
            title="Avg by Day of Week",
            xaxis_title="", yaxis_title="kW",
            template="plotly_dark", height=280,
            margin=dict(t=40, b=30, l=50, r=20),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # Monthly pattern
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby("month")["consumption"].mean().reset_index()
    monthly["month_name"] = monthly["month"].map(lambda x: month_labels[x - 1])
    fig_m = go.Figure(go.Bar(
        x=monthly["month_name"], y=monthly["consumption"],
        marker_color="#26a69a", marker_line_width=0,
    ))
    fig_m.update_layout(
        title="Avg Monthly Consumption",
        xaxis_title="", yaxis_title="kW",
        template="plotly_dark", height=260,
        margin=dict(t=40, b=30, l=50, r=20),
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # Status breakdown
    st.markdown('<p class="section-title">Status Code Breakdown</p>', unsafe_allow_html=True)
    status_map = {3: "Estimated", 4: "Measured", 6: "Adjusted"}
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    status_counts["label"] = status_counts["status"].map(lambda x: status_map.get(x, str(x)))
    fig_s = go.Figure(go.Pie(
        labels=status_counts["label"],
        values=status_counts["count"],
        hole=0.45,
        marker_colors=["#26a69a", "#4fc3f7", "#ff7043"],
    ))
    fig_s.update_layout(
        template="plotly_dark", height=280,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=True,
    )
    st.plotly_chart(fig_s, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#4a5568;font-size:.78rem'>"
    "Energy Federated Learning · EDA Dashboard · 56 clients · 2023–2025"
    "</center>",
    unsafe_allow_html=True,
)
