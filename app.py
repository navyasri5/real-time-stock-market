"""
app.py  -  StockSaathi  |  Indian Market Intelligence
Coinbase-inspired dark navy UI. No emojis. Educational focus for beginners.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import random, sys, os

sys.path.insert(0, os.path.dirname(__file__))

from utils.stock_data import (
    fetch_stock_data, get_current_snapshot, get_top_gainers_losers,
    get_index_data, NIFTY50_SYMBOLS, COMPANY_NAMES, SECTOR_MAP
)
from utils.ml_model import (
    train_model, predict_next_day, get_shap_explanation, get_lime_explanation
)
from utils.portfolio import (
    add_holding, remove_holding, get_portfolio_summary, get_portfolio_total
)
from utils.education import (
    GLOSSARY, BEGINNER_TIPS, HOW_TO_READ_PREDICTION, SECTOR_EXPLAINERS
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSaathi - Indian Market Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN TOKENS ─────────────────────────────────────────────────────────────
BG         = "#0B0F1A"
CARD       = "#131929"
CARD2      = "#1A2238"
BORDER     = "#243050"
NAVY       = "#1652F0"
BLUE_LIGHT = "#4A7CF0"
GREEN      = "#00C853"
RED        = "#FF3D3D"
GOLD       = "#F0B429"
TEXT       = "#E8EDF5"
MUTED      = "#7A8BA8"
WHITE      = "#FFFFFF"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    background-color: #0B0F1A;
    color: #E8EDF5;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: #0B0F1A;
}
.main .block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1526 0%, #0B1020 100%);
    border-right: 1px solid #243050;
}
[data-testid="stSidebar"] * { color: #E8EDF5 !important; }
[data-testid="stSidebar"] hr { border-color: #243050 !important; margin: 1rem 0; }

/* Brand */
.brand-wrap {
    padding: 0.5rem 0 1.2rem 0;
    border-bottom: 1px solid #243050;
    margin-bottom: 1.4rem;
}
.brand-logo {
    display: inline-block;
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #1652F0 0%, #4A7CF0 100%);
    border-radius: 10px;
    text-align: center;
    line-height: 38px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.5rem;
}
.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    line-height: 1;
}
.brand-tag {
    font-size: 0.65rem;
    color: #8FA8C8;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Nav */
[data-testid="stSidebar"] .stRadio label {
    display: block;
    padding: 0.55rem 0.9rem !important;
    border-radius: 8px;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #C8D8F0 !important;
    cursor: pointer;
    margin-bottom: 0.1rem;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #1A2238;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div {
    background: #1652F0 !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #131929 0%, #0E1B38 60%, #091428 100%);
    border: 1px solid #243050;
    border-radius: 16px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, #1652F022 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 0.72rem;
    color: #F0B429;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.35rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.15;
    margin-bottom: 0.8rem;
    letter-spacing: -0.025em;
    text-shadow: 0 2px 20px rgba(255,255,255,0.06);
}
.hero-title span {
    color: #F0B429;
    text-shadow: 0 0 40px rgba(240,180,41,0.4);
}
.hero-sub {
    font-size: 0.92rem;
    color: #7A8BA8;
    line-height: 1.65;
    max-width: 560px;
}

/* Section */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.35rem;
    margin-top: 0.5rem;
    letter-spacing: -0.01em;
    text-shadow: 0 0 30px rgba(255,255,255,0.08);
}
.section-sub {
    font-size: 0.83rem;
    color: #8FA8C8;
    margin-bottom: 1.3rem;
    line-height: 1.55;
}

/* Cards */
.card {
    background: #131929;
    border: 1px solid #243050;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.8rem;
}
.metric-card {
    background: #131929;
    border: 1px solid #243050;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1652F0 0%, #4A7CF0 100%);
    border-radius: 12px 12px 0 0;
}
.metric-label {
    font-size: 0.70rem;
    color: #9AB0CC;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.metric-change-up   { font-size: 0.8rem; color: #00C853; font-weight: 600; margin-top: 0.25rem; }
.metric-change-down { font-size: 0.8rem; color: #FF3D3D; font-weight: 600; margin-top: 0.25rem; }

/* Ticker */
.ticker-strip { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.6rem; }
.ticker-pill {
    background: #131929;
    border: 1px solid #243050;
    border-radius: 50px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}
.ticker-name  { color: #7A8BA8; font-weight: 500; }
.ticker-price { color: #FFFFFF; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
.ticker-up    { color: #00C853; font-weight: 600; font-size: 0.75rem; }
.ticker-down  { color: #FF3D3D; font-weight: 600; font-size: 0.75rem; }

/* Prediction */
.pred-up {
    background: linear-gradient(135deg, #001A0F 0%, #002918 100%);
    border: 1px solid #004D25;
    border-left: 4px solid #00C853;
    border-radius: 12px;
    padding: 1.6rem;
    text-align: center;
}
.pred-down {
    background: linear-gradient(135deg, #1A0000 0%, #2A0808 100%);
    border: 1px solid #4D0000;
    border-left: 4px solid #FF3D3D;
    border-radius: 12px;
    padding: 1.6rem;
    text-align: center;
}
.pred-label { font-size: 0.65rem; color: #7A8BA8; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 600; }
.pred-direction-up   { font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 700; color: #00C853; letter-spacing: -0.02em; }
.pred-direction-down { font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 700; color: #FF3D3D; letter-spacing: -0.02em; }
.pred-confidence { font-size: 0.88rem; color: #F0B429; margin-top: 0.4rem; font-weight: 600; }

/* Tip */
.tip-box {
    background: #0F1E38;
    border: 1px solid #243050;
    border-left: 3px solid #F0B429;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.7rem 0;
    font-size: 0.86rem;
    color: #C0D0E0;
    line-height: 1.65;
}

/* Divider */
.divider { border: none; border-top: 1px solid #243050; margin: 1.5rem 0; }

/* Streamlit overrides */
[data-testid="stDataFrame"] {
    border: 1px solid #243050;
    border-radius: 10px;
    background: #131929;
}
.stButton button {
    background: linear-gradient(135deg, #1652F0 0%, #4A7CF0 100%);
    color: #FFFFFF;
    border: none;
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 8px;
    padding: 0.5rem 1.6rem;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 12px #1652F044;
}
.stButton button:hover {
    background: linear-gradient(135deg, #4A7CF0 0%, #1652F0 100%);
    box-shadow: 0 4px 20px #1652F066;
}
.stSelectbox > div > div,
.stNumberInput input,
.stTextInput input {
    background: #1A2238 !important;
    border: 1px solid #243050 !important;
    border-radius: 8px !important;
    color: #E8EDF5 !important;
}
.stExpander {
    background: #131929 !important;
    border: 1px solid #243050 !important;
    border-radius: 10px !important;
}
.stExpander summary { color: #E8EDF5 !important; font-weight: 500; }
.stAlert {
    background: #0F1E38;
    border: 1px solid #243050;
    color: #E8EDF5;
    border-radius: 8px;
}
.glossary-def { font-size: 0.87rem; color: #7A8BA8; line-height: 1.7; }
.disclaimer {
    font-size: 0.70rem;
    color: #3A4A60;
    text-align: center;
    margin-top: 2.5rem;
    border-top: 1px solid #243050;
    padding-top: 1rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-logo">S</div><br>
        <div class="brand-name">StockSaathi</div>
        <div class="brand-tag">Indian Market Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Market Overview", "Stock Analysis & Prediction",
         "Top Gainers / Losers", "Portfolio Tracker", "Learn the Basics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#9AB0CC;line-height:1.7;padding:0.2rem 0.4rem;">
        Data: Yahoo Finance (yfinance)<br>
        Exchange: NSE / BSE<br>
        <span style="color:#3A4A60;">Not financial advice.</span>
    </div>
    <div style="margin-top:1.4rem;padding:0.7rem 0.8rem;background:#0D1526;
                border:1px solid #243050;border-radius:8px;text-align:center;">
        <div style="font-size:0.60rem;color:#3A4A60;letter-spacing:0.12em;
                    text-transform:uppercase;margin-bottom:0.3rem;">Developed by</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
                    font-weight:700;color:#F0B429;letter-spacing:0.04em;">Navya</div>
    </div>
    """, unsafe_allow_html=True)


# ── CHART HELPERS ─────────────────────────────────────────────────────────────
def color_change(val):
    if isinstance(val, (int, float)):
        return f"color: {GREEN if val >= 0 else RED}; font-weight:600;"
    return ""

def chart_layout(fig, height=420, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#E8EDF5", family="Space Grotesk", size=13)) if title else None,
        paper_bgcolor="#131929", plot_bgcolor="#1A2238",
        font=dict(color="#7A8BA8", family="Manrope", size=11),
        xaxis=dict(gridcolor="#243050", zeroline=False, showgrid=True,
                   tickfont=dict(color="#7A8BA8"), linecolor="#243050"),
        yaxis=dict(gridcolor="#243050", zeroline=False, showgrid=True,
                   tickfont=dict(color="#7A8BA8"), linecolor="#243050"),
        legend=dict(bgcolor="#131929", bordercolor="#243050", font=dict(color="#E8EDF5")),
        margin=dict(l=12, r=12, t=40 if title else 12, b=12),
        height=height,
        xaxis_rangeslider_visible=False,
    )
    return fig

def plotly_candlestick(df, title=""):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name="Price"
    ))
    if "EMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"],
            line=dict(color=GOLD, width=1.4), name="EMA 20"))
    if "EMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"],
            line=dict(color=BLUE_LIGHT, width=1.4), name="EMA 50"))
    return chart_layout(fig, 430, title)

def plotly_volume(df):
    returns = df["Daily_Return"].tolist() if "Daily_Return" in df.columns else [0]*len(df)
    colors  = ["rgba(0,200,83,0.7)" if r >= 0 else "rgba(255,61,61,0.7)" for r in returns]
    fig = go.Figure(go.Bar(
        x=df.index, y=df["Volume"],
        marker=dict(color=colors),
        name="Volume"
    ))
    return chart_layout(fig, 175, "Volume")

def plotly_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color="#F0B429", width=1.8), name="RSI",
        fill="tozeroy", fillcolor="rgba(240,180,41,0.07)"))
    fig.add_hline(y=70, line_dash="dot", line_color="#FF3D3D", opacity=0.7,
                  annotation_text="Overbought", annotation_font_color="#FF3D3D")
    fig.add_hline(y=30, line_dash="dot", line_color="#00C853", opacity=0.7,
                  annotation_text="Oversold", annotation_font_color="#00C853")
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(22,82,240,0.04)", line_width=0)
    fig.update_yaxes(range=[0, 100])
    return chart_layout(fig, 220, "RSI (14)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Market Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Live Indian Stock Market</div>
        <div class="hero-title">Track the Market.<br><span>Understand Your Investments.</span></div>
        <div class="hero-sub">
            Real-time data across NIFTY 50 and BSE. Built for first-time investors
            who want clarity, not jargon. All prices in Indian Rupees.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching index data..."):
        indices = get_index_data()

    pills = '<div class="ticker-strip">'
    for name, data in indices.items():
        chg = data["change_pct"]
        chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
        cls = "ticker-up" if chg >= 0 else "ticker-down"
        pills += f"""<div class="ticker-pill">
            <span class="ticker-name">{name}</span>
            <span class="ticker-price">{data['price']:,.2f}</span>
            <span class="{cls}">{chg_str}</span>
        </div>"""
    pills += "</div>"
    st.markdown(pills, unsafe_allow_html=True)

    cols = st.columns(len(indices))
    for i, (name, data) in enumerate(indices.items()):
        chg = data["change_pct"]
        chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
        cls = "metric-change-up" if chg >= 0 else "metric-change-down"
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{data['price']:,.2f}</div>
                <div class="{cls}">{chg_str} today</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">NIFTY 50 Live Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All 50 index constituents ranked by today\'s performance.</div>', unsafe_allow_html=True)

    with st.spinner("Loading market data..."):
        snapshot = get_current_snapshot()

    if not snapshot.empty:
        snapshot = snapshot.dropna(subset=["Price","Change_Pct"])
        display = snapshot[["Company","Sector","Price","Change_Pct","Volume"]].copy()
        display.columns = ["Company","Sector","Price (Rs)","Change %","Volume"]
        styled = (display.style
            .map(color_change, subset=["Change %"])
            .format({"Price (Rs)":"{:,.2f}","Change %":"{:+.2f}%","Volume":"{:,}"})
            .set_properties(**{"background-color":"#131929","color":"#E8EDF5","border":"1px solid #243050","font-size":"0.84rem"}))
        st.dataframe(styled, use_container_width=True, height=460)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sector Performance Today</div>', unsafe_allow_html=True)

        sector_avg = snapshot.groupby("Sector")["Change_Pct"].mean().reset_index()
        sector_avg.columns = ["Sector","Avg Change %"]
        sector_avg = sector_avg.dropna(subset=["Avg Change %"])
        sector_avg = sector_avg[sector_avg["Avg Change %"].apply(lambda x: isinstance(x, (int,float)) and x==x)]
        sector_avg = sector_avg.sort_values("Avg Change %", ascending=False)

        bar_colors = ["#00C853" if v >= 0 else "#FF3D3D" for v in sector_avg["Avg Change %"]]
        bar_texts   = [f"{v:+.2f}%" for v in sector_avg["Avg Change %"]]

        fig_s = go.Figure(go.Bar(
            x=sector_avg["Sector"],
            y=sector_avg["Avg Change %"],
            marker=dict(color=bar_colors, opacity=0.88),
            text=bar_texts,
            textposition="outside",
            textfont=dict(color="#9AB0CC", size=11),
            width=0.55,
        ))
        fig_s.update_layout(
            paper_bgcolor="#131929",
            plot_bgcolor="#1A2238",
            font=dict(color="#9AB0CC", family="Manrope", size=11),
            xaxis=dict(gridcolor="#243050", tickfont=dict(color="#9AB0CC", size=10),
                       linecolor="#243050", zeroline=False),
            yaxis=dict(gridcolor="#243050", tickfont=dict(color="#9AB0CC"),
                       linecolor="#243050", zeroline=True, zerolinecolor="#243050",
                       ticksuffix="%"),
            margin=dict(l=12, r=12, t=30, b=12),
            height=320,
            showlegend=False,
            bargap=0.3,
        )
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown(f"""
        <div class="tip-box">
            <strong style="color:{GOLD};">For Beginners:</strong>
            Green sectors are rising today; red sectors are falling.
            Spreading your money across multiple sectors reduces risk because
            when one sector falls, others often hold steady or rise.
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Could not load stock data. Check your internet connection.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STOCK ANALYSIS & PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Stock Analysis & Prediction":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">AI-Powered Analysis</div>
        <div class="hero-title">Deep Dive into Any<br><span>NIFTY 50 Stock</span></div>
        <div class="hero-sub">
            Select a company to view price charts, technical indicators, and an
            AI prediction with full explanation of the reasoning behind it.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        sym_map = {v: k for k, v in COMPANY_NAMES.items()}
        selected_co = st.selectbox("Select Company", sorted(COMPANY_NAMES.values()))
        sym = sym_map[selected_co]
    with c2:
        period = st.selectbox("Period", ["3mo","6mo","1y","2y"], index=1)

    with st.spinner(f"Loading {selected_co}..."):
        df = fetch_stock_data(sym, period=period)

    if df.empty:
        st.error("Could not fetch data. Try another stock.")
        st.stop()

    curr  = df["Close"].iloc[-1]
    prev  = df["Close"].iloc[-2]
    chg   = ((curr - prev) / prev) * 100
    rsi   = df["RSI"].iloc[-1] if "RSI" in df.columns else 0
    rsi_label = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
    rsi_color = RED if rsi > 70 else (GREEN if rsi < 30 else GOLD)
    chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
    chg_cls = "metric-change-up" if chg >= 0 else "metric-change-down"

    m1,m2,m3,m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">Rs {curr:,.2f}</div>
            <div class="{chg_cls}">{chg_str} vs prev close</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value" style="color:{rsi_color};">{rsi:.1f}</div>
            <div style="font-size:0.75rem;color:{rsi_color};margin-top:0.2rem;">{rsi_label}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Period High</div>
            <div class="metric-value">Rs {df['High'].max():,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Period Low</div>
            <div class="metric-value">Rs {df['Low'].min():,.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.plotly_chart(plotly_candlestick(df, f"{selected_co} — Price & EMA"), use_container_width=True)
    st.plotly_chart(plotly_volume(df), use_container_width=True)
    st.plotly_chart(plotly_rsi(df), use_container_width=True)

    sec = SECTOR_MAP.get(sym, "")
    if sec in SECTOR_EXPLAINERS:
        st.markdown(f"""<div class="tip-box">
            <strong style="color:{GOLD};">About {sec}:</strong> {SECTOR_EXPLAINERS[sec]}
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A Gradient Boosting model trained on technical indicators predicts next-day price direction. SHAP and LIME explain the reasoning.</div>', unsafe_allow_html=True)

    if st.button("Train Model and Generate Prediction"):
        with st.spinner("Training model on historical data..."):
            artifacts = train_model(df)

        if "error" in artifacts:
            st.error(artifacts["error"])
        else:
            pred = predict_next_day(artifacts, df)
            acc  = artifacts["accuracy"]
            box  = "pred-up" if pred["direction"] == "UP" else "pred-down"
            dcls = "pred-direction-up" if pred["direction"] == "UP" else "pred-direction-down"

            p1,p2,p3 = st.columns(3)
            with p1:
                st.markdown(f"""<div class="{box}">
                    <div class="pred-label">Predicted Direction</div>
                    <div class="{dcls}">{pred['direction']}</div>
                    <div class="pred-confidence">Confidence: {pred['confidence']}%</div>
                </div>""", unsafe_allow_html=True)
            with p2:
                st.markdown(f"""<div class="metric-card" style="text-align:center;">
                    <div class="metric-label">Model Accuracy</div>
                    <div class="metric-value">{acc*100:.1f}%</div>
                    <div style="font-size:0.72rem;color:{MUTED};margin-top:0.2rem;">On held-out historical data</div>
                </div>""", unsafe_allow_html=True)
            with p3:
                st.markdown(f"""<div class="metric-card" style="text-align:center;">
                    <div class="metric-label">Probability UP / DOWN</div>
                    <div class="metric-value">{pred['prob_up']}% / {pred['prob_down']}%</div>
                    <div style="font-size:0.72rem;color:{MUTED};margin-top:0.2rem;">Based on latest indicators</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="tip-box" style="margin-top:1rem;white-space:pre-line;">{HOW_TO_READ_PREDICTION}</div>', unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">SHAP — Overall Feature Importance</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Which technical indicators drive the model\'s decisions the most?</div>', unsafe_allow_html=True)

            with st.spinner("Computing SHAP values..."):
                try:
                    shap_r = get_shap_explanation(artifacts)
                    imp = shap_r["importance"]
                    fig_shap, ax = plt.subplots(figsize=(9, 5))
                    ax.barh(imp.index[::-1], imp.values[::-1], color=NAVY, alpha=0.85)
                    ax.set_xlabel("Mean |SHAP Value|", fontsize=10, color="#7A8BA8")
                    ax.set_title("SHAP Feature Importance", fontsize=12,
                                 fontweight="bold", color=WHITE, pad=10)
                    ax.spines[["top","right","left","bottom"]].set_visible(False)
                    fig_shap.patch.set_facecolor("#131929")
                    ax.set_facecolor("#1A2238")
                    ax.tick_params(colors="#7A8BA8", labelsize=9)
                    ax.xaxis.label.set_color(MUTED)
                    plt.tight_layout()
                    st.pyplot(fig_shap, use_container_width=True)
                    plt.close("all")
                    top3 = ", ".join([f"<strong>{k}</strong>" for k in imp.head(3).index])
                    st.markdown(f'<div class="tip-box">The three strongest indicators for this model are: {top3}. Higher values mean stronger influence on every prediction.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"SHAP issue: {e}")

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">LIME — Why This Specific Prediction?</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-sub">What pushed the model toward {pred["direction"]} for this stock right now?</div>', unsafe_allow_html=True)

            with st.spinner("Computing LIME explanation..."):
                try:
                    lime_r = get_lime_explanation(artifacts, df)
                    lime_list = lime_r["lime_list"]
                    labels_l = [x[0] for x in lime_list]
                    vals_l   = [x[1] for x in lime_list]
                    clr_lime = [GREEN+"CC" if v > 0 else RED+"CC" for v in vals_l]

                    fig_lime, ax = plt.subplots(figsize=(9, 5))
                    ax.barh(range(len(labels_l)), vals_l, color=clr_lime, alpha=0.9)
                    ax.set_yticks(range(len(labels_l)))
                    ax.set_yticklabels(labels_l, fontsize=8.5, color="#7A8BA8")
                    ax.axvline(0, color="#243050", linewidth=1)
                    ax.set_xlabel("LIME Weight (positive supports UP)", fontsize=10, color="#7A8BA8")
                    ax.set_title("LIME — Why This Prediction Was Made", fontsize=12,
                                 fontweight="bold", color=WHITE, pad=10)
                    ax.spines[["top","right","left","bottom"]].set_visible(False)
                    fig_lime.patch.set_facecolor("#131929")
                    ax.set_facecolor("#1A2238")
                    ax.tick_params(colors="#7A8BA8", labelsize=9)
                    ax.xaxis.label.set_color(MUTED)
                    plt.tight_layout()
                    st.pyplot(fig_lime, use_container_width=True)
                    plt.close("all")

                    sup = [l for l,v in lime_list if v > 0][:3]
                    opp = [l for l,v in lime_list if v < 0][:3]
                    if sup:
                        st.markdown(f'<div class="tip-box">Supporting {pred["direction"]}: {", ".join([f"<strong>{x}</strong>" for x in sup])}.</div>', unsafe_allow_html=True)
                    if opp:
                        st.markdown(f'<div class="tip-box">Opposing the prediction: {", ".join([f"<strong>{x}</strong>" for x in opp])}.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"LIME issue: {e}")

    st.markdown(f'<div class="disclaimer">AI predictions are based on historical patterns and carry no guarantee. This is not financial advice. Consult a SEBI-registered advisor.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TOP GAINERS / LOSERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Top Gainers / Losers":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Daily Movers</div>
        <div class="hero-title">Top Gainers<br><span>and Losers Today</span></div>
        <div class="hero-sub">
            Stocks with the largest price moves in the NIFTY 50 today.
            Understanding why stocks move is the first step to smart investing.
        </div>
    </div>
    """, unsafe_allow_html=True)

    n = st.slider("Number of stocks to show", 5, 20, 10)
    with st.spinner("Fetching market data..."):
        gainers, losers = get_top_gainers_losers(n)

    cg, cl = st.columns(2)
    with cg:
        st.markdown(f'<div style="font-size:0.72rem;color:{GREEN};letter-spacing:0.14em;text-transform:uppercase;font-weight:700;margin-bottom:0.6rem;">Top Gainers</div>', unsafe_allow_html=True)
        if not gainers.empty:
            dg = gainers[["Company","Sector","Price","Change_Pct"]].copy()
            dg.columns = ["Company","Sector","Price (Rs)","Change %"]
            st.dataframe(
                dg.style.map(color_change, subset=["Change %"])
                  .format({"Price (Rs)":"{:,.2f}","Change %":"{:+.2f}%"})
                  .set_properties(**{"background-color":"#131929","color":"#E8EDF5","font-size":"0.83rem"}),
                use_container_width=True, hide_index=True)
            g_clean = gainers.dropna(subset=["Change_Pct","Company"])
            fig_g = go.Figure(go.Bar(
                x=g_clean["Change_Pct"], y=g_clean["Company"], orientation="h",
                marker=dict(color="#00C853", opacity=0.85),
                text=[f"+{v:.2f}%" for v in g_clean["Change_Pct"]],
                textposition="outside", textfont=dict(color="#9AB0CC", size=10)
            ))
            fig_g = chart_layout(fig_g, 340)
            fig_g.update_xaxes(ticksuffix="%")
            st.plotly_chart(fig_g, use_container_width=True)

    with cl:
        st.markdown(f'<div style="font-size:0.72rem;color:{RED};letter-spacing:0.14em;text-transform:uppercase;font-weight:700;margin-bottom:0.6rem;">Top Losers</div>', unsafe_allow_html=True)
        if not losers.empty:
            dl = losers[["Company","Sector","Price","Change_Pct"]].copy()
            dl.columns = ["Company","Sector","Price (Rs)","Change %"]
            st.dataframe(
                dl.style.map(color_change, subset=["Change %"])
                  .format({"Price (Rs)":"{:,.2f}","Change %":"{:+.2f}%"})
                  .set_properties(**{"background-color":"#131929","color":"#E8EDF5","font-size":"0.83rem"}),
                use_container_width=True, hide_index=True)
            l_clean = losers.dropna(subset=["Change_Pct","Company"])
            fig_l = go.Figure(go.Bar(
                x=l_clean["Change_Pct"], y=l_clean["Company"], orientation="h",
                marker=dict(color="#FF3D3D", opacity=0.85),
                text=[f"{v:.2f}%" for v in l_clean["Change_Pct"]],
                textposition="outside", textfont=dict(color="#9AB0CC", size=10)
            ))
            fig_l = chart_layout(fig_l, 340)
            fig_l.update_xaxes(ticksuffix="%")
            st.plotly_chart(fig_l, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f"""<div class="tip-box">
        <strong style="color:{GOLD};">What does this mean for you?</strong><br>
        A stock rising 5% in one day may be overbought — the price may correct soon.
        A stock falling sharply may be a buying opportunity or a warning sign of deeper trouble.
        Never decide based on a single day. Always ask: why is it moving?
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Tracker":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Your Holdings</div>
        <div class="hero-title">Portfolio<br><span>Tracker</span></div>
        <div class="hero-sub">
            Add your stock holdings and track live profit and loss.
            All data stays on your machine. Nothing is sent anywhere.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Add a New Holding"):
        sym_map = {v: k for k, v in COMPANY_NAMES.items()}
        a1,a2,a3 = st.columns([2,1,1])
        with a1:
            add_co = st.selectbox("Company", sorted(COMPANY_NAMES.values()), key="p_co")
        with a2:
            add_qty = st.number_input("Qty (shares)", min_value=1, value=10, step=1)
        with a3:
            add_px = st.number_input("Buy Price (Rs)", min_value=0.01, value=100.0, step=0.5)
        if st.button("Add to Portfolio"):
            add_holding(sym_map[add_co], add_co, add_qty, add_px)
            st.success(f"Added {add_qty} shares of {add_co} at Rs {add_px:.2f}.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.spinner("Loading portfolio..."):
        pf = get_portfolio_summary()

    if pf.empty:
        st.markdown(f"""<div class="card" style="text-align:center;padding:2.5rem;">
            <div style="color:{MUTED};font-size:0.95rem;">
                Your portfolio is empty.<br>Add your first holding above to start tracking.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        totals = get_portfolio_total(pf)
        sign = "+" if totals["pnl"] >= 0 else ""
        pcls = "metric-change-up" if totals["pnl"] >= 0 else "metric-change-down"

        t1,t2,t3,t4 = st.columns(4)
        with t1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Invested</div><div class="metric-value">Rs {totals["invested"]:,.0f}</div></div>', unsafe_allow_html=True)
        with t2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Current Value</div><div class="metric-value">Rs {totals["current"]:,.0f}</div></div>', unsafe_allow_html=True)
        with t3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total P&L</div><div class="metric-value {pcls}">Rs {sign}{totals["pnl"]:,.0f}</div></div>', unsafe_allow_html=True)
        with t4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Return %</div><div class="metric-value {pcls}">{sign}{totals["pnl_pct"]:.2f}%</div></div>', unsafe_allow_html=True)

        styled_pf = (pf.style
            .map(color_change, subset=["P&L (Rs)","P&L %"])
            .format({"Buy Price":"Rs {:,.2f}","Current Price":"Rs {:,.2f}",
                     "Invested (Rs)":"Rs {:,.2f}","Current Value (Rs)":"Rs {:,.2f}",
                     "P&L (Rs)":"Rs {:+,.2f}","P&L %":"{:+.2f}%"})
            .set_properties(**{"background-color":"#131929","color":"#E8EDF5","font-size":"0.83rem"}))
        st.dataframe(styled_pf, use_container_width=True, hide_index=True)

        fig_pie = go.Figure(go.Pie(
            labels=pf["Company"], values=pf["Current Value (Rs)"],
            hole=0.6,
            marker=dict(
                colors=[NAVY,BLUE_LIGHT,GOLD,GREEN,"#7B68EE","#20B2AA","#FF8C00","#DA70D6"],
                line=dict(color=BG, width=2)),
            textinfo="label+percent",
            textfont=dict(size=9, color="#E8EDF5")
        ))
        fig_pie.update_layout(
            paper_bgcolor="#131929", font=dict(color="#7A8BA8", family="Manrope"),
            legend=dict(bgcolor="#131929", bordercolor="#243050", font=dict(color="#E8EDF5")),
            height=340, margin=dict(l=10,r=10,t=20,b=10), showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("Remove a Holding"):
            held = {COMPANY_NAMES.get(s,s): s for s in pf["Symbol"].values}
            rm_co = st.selectbox("Select to remove", list(held.keys()))
            if st.button("Remove"):
                remove_holding(held[rm_co])
                st.success(f"Removed {rm_co}.")

        st.markdown(f"""<div class="tip-box">
            P&L shown is unrealised — it only becomes real when you sell.
            Focus on the strength of the company's business rather than reacting
            to daily price fluctuations.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LEARN THE BASICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Learn the Basics":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Education Centre</div>
        <div class="hero-title">New to the<br><span>Stock Market?</span></div>
        <div class="hero-sub">
            Plain-language explanations of every term used in this app.
            Start here if you are just beginning your investing journey.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tip = random.choice(BEGINNER_TIPS)
    st.markdown(f"""<div class="card" style="border-left:3px solid {GOLD};margin-bottom:1.6rem;">
        <div style="font-size:0.66rem;color:{GOLD};letter-spacing:0.14em;text-transform:uppercase;font-weight:700;margin-bottom:0.45rem;">Insight of the Day</div>
        <div style="font-size:0.90rem;color:{TEXT};line-height:1.65;">{tip}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Glossary of Key Terms</div>', unsafe_allow_html=True)
    search = st.text_input("Search", placeholder="Try: RSI, MACD, Dividend, Bull Market...")
    filtered = {k:v for k,v in GLOSSARY.items()
                if search.lower() in k.lower() or search.lower() in v.lower()} if search else GLOSSARY

    for term, defn in filtered.items():
        with st.expander(term):
            st.markdown(f'<div class="glossary-def">{defn}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">10 Rules for the First-Time Investor</div>', unsafe_allow_html=True)
    for i, t in enumerate(BEGINNER_TIPS, 1):
        st.markdown(f"""<div class="tip-box">
            <span style="color:{GOLD};font-weight:700;font-family:'Space Grotesk',sans-serif;">{i}.</span> {t}
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How the AI Prediction Works</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="tip-box" style="white-space:pre-line;">{HOW_TO_READ_PREDICTION}</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="disclaimer">
        StockSaathi is an educational tool. Nothing here constitutes financial advice.
        Past market patterns do not guarantee future results.
        Consult a SEBI-registered investment advisor before making any investment decision.
    </div>""", unsafe_allow_html=True)
