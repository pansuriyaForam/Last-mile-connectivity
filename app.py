"""
MetroIQ — GTFS Transit Intelligence Platform
================================================================================
A production-grade analytics platform analyzing Hyderabad Metro accessibility 
using real GTFS data, custom metrics (LMCI), enhanced optimization algorithms 
(MCLP), and scenario simulation.

Author: Foram Pansuriya (BE AI & ML, Osmania University)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in Python path for clean module imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────
# IMPORT SRC MODULES
# ─────────────────────────────────────────────────────────────

try:
    import src.preprocessing as preprocessing
    import src.lmci as lmci
    import src.mclp as mclp
    import src.scoring as scoring
    import src.simulation as simulation
    import src.visualization as visualization
    SRC_AVAILABLE = True
except ImportError:
    try:
        import preprocessing
        import lmci
        import mclp
        import scoring
        import simulation
        import visualization
        SRC_AVAILABLE = True
    except ImportError:
        SRC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MetroIQ — GTFS Transit Intelligence Platform",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# DESIGN SYSTEM & COLOR TOKENS
# ─────────────────────────────────────────────────────────────

BG_DARK        = "#0A0E1A"
CARD_SURFACE   = "#0D1420"
PRIMARY_ACCENT = "#00D4FF"
SEC_ACCENT     = "#6366F1"
SUCCESS_COLOR  = "#22C55E"
WARNING_COLOR  = "#FACC15"
CRITICAL_COLOR = "#EF4444"
MUTED_TEXT     = "#64748B"
BORDER_COLOR   = "#1A2535"

# Global CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Reset & Background */
[data-testid="stAppViewContainer"] {{
    background: {BG_DARK};
    background-image:
        radial-gradient(ellipse at 15% 0%, rgba(0, 212, 255, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 85% 100%, rgba(99, 102, 241, 0.04) 0%, transparent 60%);
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: #070B14;
    border-right: 1px solid {BORDER_COLOR};
}}

[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 1.2rem; padding-bottom: 2rem; }}

/* Typography */
h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; font-weight: 700; color: #FFFFFF; }}
h1 {{ font-size: 1.65rem !important; letter-spacing: -0.03em; margin-bottom: 0.2rem !important; }}
h2 {{ font-size: 1.1rem !important; color: #F1F5F9 !important; letter-spacing: -0.01em; margin-top: 0.4rem; }}
h3 {{ font-size: 0.8rem !important; color: #64748B !important; text-transform: uppercase; letter-spacing: 0.08em; }}

/* Metric Cards */
[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {CARD_SURFACE} 0%, #111827 100%);
    border: 1px solid {BORDER_COLOR};
    border-radius: 10px;
    padding: 0.85rem 1rem !important;
    position: relative;
    overflow: hidden;
}}
[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {PRIMARY_ACCENT}, transparent);
}}
[data-testid="stMetricLabel"] {{
    color: #64748B !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}
[data-testid="stMetricValue"] {{
    color: #F8FAFC !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}}

/* Containers & Cards */
[data-testid="stContainer"][data-border="true"] {{
    background: linear-gradient(135deg, {CARD_SURFACE} 0%, #101726 100%);
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 12px !important;
    padding: 1.1rem;
}}

/* Radio Nav in Sidebar */
[data-testid="stRadio"] > div {{ gap: 3px !important; }}
[data-testid="stRadio"] label {{
    border-radius: 8px !important;
    padding: 9px 14px !important;
    margin: 0 !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    cursor: pointer;
}}
[data-testid="stRadio"] label:hover {{
    background: rgba(30, 41, 59, 0.6) !important;
    color: #F1F5F9 !important;
}}
[data-testid="stRadio"] label[data-checked="true"] {{
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.12), rgba(99, 102, 241, 0.08)) !important;
    color: {PRIMARY_ACCENT} !important;
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
}}
[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {{ display: none; }}

/* Badges */
.badge-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 6px;
}}
.badge-primary {{ background: rgba(0, 212, 255, 0.12); color: {PRIMARY_ACCENT}; border: 1px solid rgba(0, 212, 255, 0.3); }}
.badge-sec {{ background: rgba(99, 102, 241, 0.12); color: {SEC_ACCENT}; border: 1px solid rgba(99, 102, 241, 0.3); }}
.badge-success {{ background: rgba(34, 197, 94, 0.12); color: {SUCCESS_COLOR}; border: 1px solid rgba(34, 197, 94, 0.3); }}
.badge-warning {{ background: rgba(250, 204, 21, 0.12); color: {WARNING_COLOR}; border: 1px solid rgba(250, 204, 21, 0.3); }}
.badge-critical {{ background: rgba(239, 68, 68, 0.12); color: {CRITICAL_COLOR}; border: 1px solid rgba(239, 68, 68, 0.3); }}

/* Recruiter Takeaway Card */
.takeaway-card {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.06) 0%, rgba(99, 102, 241, 0.04) 100%);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 10px;
    padding: 14px 16px;
    height: 100%;
}}
.takeaway-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 800;
    color: {PRIMARY_ACCENT};
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}
.takeaway-title {{
    font-size: 0.95rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 4px 0 6px 0;
}}
.takeaway-desc {{
    font-size: 0.78rem;
    color: #94A3B8;
    line-height: 1.45;
}}

/* Highlight Card */
.highlight-card {{
    background: #0D1420;
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 14px;
    height: 100%;
}}
.highlight-icon {{
    font-size: 1.2rem;
    margin-bottom: 6px;
}}
.highlight-title {{
    font-size: 0.88rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 4px;
}}
.highlight-desc {{
    font-size: 0.76rem;
    color: #64748B;
    line-height: 1.4;
}}

/* Pipeline Flow Nodes */
.pipeline-flow {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 16px;
    background: #080D18;
    border: 1px solid {BORDER_COLOR};
    border-radius: 10px;
    margin: 12px 0;
}}
.flow-node {{
    background: #0F172A;
    border: 1px solid #1E2D40;
    border-radius: 6px;
    padding: 8px 14px;
    text-align: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: #E2E8F0;
}}
.flow-arrow {{
    color: {PRIMARY_ACCENT};
    font-weight: 800;
    font-size: 0.85rem;
}}

/* Did You Know Card */
.dyk-card {{
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, #0D1420 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-left: 3px solid {SEC_ACCENT};
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
}}
.dyk-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 2px;
}}
.dyk-desc {{
    font-size: 0.76rem;
    color: #94A3B8;
    line-height: 1.4;
}}

/* Why I Built This Card */
.why-card {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.04) 0%, rgba(99,102,241,0.04) 100%);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-left: 3px solid {PRIMARY_ACCENT};
    border-radius: 10px;
    padding: 14px 18px;
}}
.why-quote {{
    font-size: 0.88rem;
    font-style: italic;
    color: #CBD5E1;
    line-height: 1.55;
    margin-bottom: 6px;
}}
.why-author {{
    font-size: 0.72rem;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* Journey Timeline */
.journey-wrap {{
    display: flex;
    align-items: flex-start;
    gap: 0;
    overflow-x: auto;
    padding: 14px 4px 6px;
}}
.journey-step {{
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    min-width: 90px;
    text-align: center;
    position: relative;
}}
.journey-step:not(:last-child)::after {{
    content: '';
    position: absolute;
    top: 15px;
    left: calc(50% + 15px);
    right: calc(-50% + 15px);
    height: 1px;
    background: linear-gradient(90deg, {PRIMARY_ACCENT}, rgba(99,102,241,0.4));
}}
.journey-dot {{
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(99,102,241,0.18));
    border: 1.5px solid rgba(0,212,255,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    margin-bottom: 6px;
    position: relative;
    z-index: 1;
}}
.journey-label {{
    font-size: 0.68rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    line-height: 1.3;
}}

/* Warning Card for Missing Files */
.warning-card {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.06) 0%, #111827 100%);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-left: 4px solid {CRITICAL_COLOR};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 12px 0;
    color: #F8FAFC;
    font-size: 0.88rem;
    line-height: 1.6;
}}
.warning-card code {{
    background: rgba(0, 0, 0, 0.4);
    padding: 2px 6px;
    border-radius: 4px;
    color: {PRIMARY_ACCENT};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}}

/* Footer */
.footer-container {{
    border-top: 1px solid {BORDER_COLOR};
    padding-top: 20px;
    margin-top: 36px;
    text-align: center;
    color: #64748B;
    font-size: 0.78rem;
    line-height: 1.6;
}}

/* Hide Default Streamlit Chrome */
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# Plotly Theme
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#64748B", family="Inter, sans-serif", size=11),
    xaxis=dict(gridcolor="#101827", linecolor="#1E2D40", zerolinecolor="#1E2D40", tickfont=dict(size=10, color="#64748B")),
    yaxis=dict(gridcolor="#101827", linecolor="#1E2D40", zerolinecolor="#1E2D40", tickfont=dict(size=10, color="#64748B")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=10, color="#64748B")),
    margin=dict(l=10, r=10, t=30, b=10),
)

def apply_plotly_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig

# ─────────────────────────────────────────────────────────────
# DATA LOADING & PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────

OUTPUTS_DIR = Path("outputs")

@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> Optional[pd.DataFrame]:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None

def load_all_outputs() -> dict:
    return {
        "exec_summary":    load_csv("executive_summary_metrics.csv"),
        "priority_scores": load_csv("station_priority_scores.csv"),
        "lmci_summary":    load_csv("station_lmci_summary.csv"),
        "lmci_scores":     load_csv("lmci_station_scores.csv"),
        "mismatch":        load_csv("demand_service_mismatch.csv"),
        "insights_top5":   load_csv("conversion_insights_top5.csv"),
        "mclp_coverage":   load_csv("mclp_coverage_by_k.csv"),
        "mclp_selected":   load_csv("mclp_selected_stations.csv"),
        "mclp_candidates": load_csv("mclp_candidate_scores.csv"),
        "mclp_assignment": load_csv("mclp_demand_assignment.csv"),
        "ablation":        load_csv("ablation_results.csv"),
        "sim_impacts":     load_csv("simulation_station_impacts.csv"),
        "sim_ranking":     load_csv("simulation_intervention_ranking.csv"),
        "sim_network":     load_csv("simulation_network_summary.csv"),
        "sim_scenarios":   load_csv("simulation_scenarios.csv"),
        "scoring_audit":   load_csv("scoring_audit_report.csv"),
        "sim_audit":       load_csv("simulation_audit_report.csv"),
        "station_coords":  load_csv("station_coordinates.csv"),
        "demand_points":   load_csv("demand_points.csv"),
    }

def run_full_data_pipeline():
    """Execute all src modules sequentially from within Streamlit."""
    if not SRC_AVAILABLE:
        st.error("src/ modules could not be imported cleanly. Ensure src/ folder exists.")
        return

    progress_bar = st.progress(0, text="Initializing Data Pipeline...")

    try:
        progress_bar.progress(10, text="[1/6] Running Preprocessing Pipeline...")
        preprocessing.run_preprocessing_pipeline()

        progress_bar.progress(30, text="[2/6] Running LMCI Calculation Engine...")
        lmci.run_lmci_pipeline()

        progress_bar.progress(50, text="[3/6] Running Enhanced MCLP Optimization Engine...")
        mclp.run_mclp_pipeline()

        progress_bar.progress(70, text="[4/6] Running Scoring & Equity Classification Layer...")
        scoring.run_scoring_pipeline()

        progress_bar.progress(85, text="[5/6] Running Scenario Simulation Engine...")
        simulation.run_simulation_pipeline()

        progress_bar.progress(95, text="[6/6] Generating Publication Visualizations...")
        visualization.generate_all_visualizations()

        progress_bar.progress(100, text="✅ Pipeline Execution Complete!")
        st.cache_data.clear()
        st.success("All 20 data artifacts and visualizations generated successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Pipeline Execution Failed: {str(e)}")

def render_missing_file_warning():
    st.markdown("""
    <div class="warning-card">
        <strong>Required output file missing or un-generated.</strong><br><br>
        You can execute the entire data engineering pipeline directly using the button below or via CLI:
        <code>python src/preprocessing.py</code> &nbsp;•&nbsp;
        <code>python src/lmci.py</code> &nbsp;•&nbsp;
        <code>python src/mclp.py</code> &nbsp;•&nbsp;
        <code>python src/scoring.py</code> &nbsp;•&nbsp;
        <code>python src/simulation.py</code>
    </div>
    """, unsafe_allow_html=True)

    if SRC_AVAILABLE:
        if st.button("🚀 Run Full Data Pipeline Now", type="primary", use_container_width=True):
            run_full_data_pipeline()

def get_lmci_category(score: float) -> tuple[str, str, str]:
    if pd.isna(score):
        return ("Unknown", MUTED_TEXT, "badge-sec")
    if score <= 4.0:
        return ("Critical Desert", CRITICAL_COLOR, "badge-critical")
    elif score <= 7.0:
        return ("Moderate Access", WARNING_COLOR, "badge-warning")
    else:
        return ("High Access", SUCCESS_COLOR, "badge-success")


def get_priority_score_column(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None:
        return None
    for col in ["final_priority_score", "priority_score", "priority_index"]:
        if col in df.columns:
            return col
    return None

# ─────────────────────────────────────────────────────────────
# PERSISTENT HEADER & SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────

PAGES = [
    "Platform Overview",
    "Data Pipeline",
    "LMCI Engine",
    "Transit Explorer",
    "Optimization Engine",
    "Scenario Lab",
]

def render_persistent_header():
    st.markdown(f"""
    <div style='border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 0.9rem; margin-bottom: 1.2rem; display: flex; justify-content: space-between; align-items: center;'>
        <div>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span style='font-size: 1.55rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;'>MetroIQ</span>
                <span style='background: rgba(0,212,255,0.12); color: {PRIMARY_ACCENT}; border: 1px solid rgba(0,212,255,0.3); padding: 2px 8px; border-radius: 12px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;'>GTFS Transit Intelligence Platform</span>
            </div>
            <div style='color: #64748B; font-size: 0.82rem; font-weight: 600; margin-top: 2px;'>Analyzing Hyderabad's transit accessibility using real-world GTFS data, custom metrics, and optimization algorithms.</div>
        </div>
        <div style='text-align: right; color: #94A3B8; font-size: 0.78rem; font-weight: 600;'>
            <div style='color: #475569; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px;'>Built with</div>
            <span style='color: {PRIMARY_ACCENT};'>Python</span> • <span style='color: #F1F5F9;'>GTFS</span> • <span style='color: #F1F5F9;'>Plotly</span> • <span style='color: #F1F5F9;'>GeoPandas</span> • <span style='color: {SEC_ACCENT};'>Optimization</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding: 0.8rem 0.2rem 0.4rem;'>
            <div style='display:flex; align-items:center; gap:10px;'>
                <div style='width:34px; height:34px; border-radius:8px;
                     background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(99,102,241,0.2));
                     border: 1px solid rgba(0,212,255,0.3);
                     display:flex; align-items:center; justify-content:center;
                     font-size:16px;'>🚇</div>
                <div>
                    <div style='color:#FFFFFF; font-weight:800; font-size:1.05rem; letter-spacing:-0.02em;'>MetroIQ</div>
                    <div style='color:#64748B; font-size:0.65rem; font-weight:600; text-transform:uppercase;'>System Showcase</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1px; background:linear-gradient(90deg,transparent,#1E2D40,transparent); margin:8px 0 12px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#64748B; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; padding:0 4px; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)

        page = st.radio("Navigation", PAGES, label_visibility="collapsed")

        st.markdown("<div style='height:1px; background:linear-gradient(90deg,transparent,#1E2D40,transparent); margin:16px 0 12px;'></div>", unsafe_allow_html=True)

        # Recruiter Sidebar Quick Facts
        st.markdown(f"""
        <div style='padding:12px; background:#080D18; border:1px solid {BORDER_COLOR}; border-radius:10px;'>
            <div style='color:{PRIMARY_ACCENT}; font-size:0.68rem; font-weight:800; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>Technical Core</div>
            <div style='font-size:0.75rem; color:#CBD5E1; margin-bottom:6px;'>⚡ <strong>Real GTFS Feeds</strong> (HMRL, TGSRTC, MMTS)</div>
            <div style='font-size:0.75rem; color:#CBD5E1; margin-bottom:6px;'>📐 <strong>Custom LMCI</strong> (0.0–10.0 scale)</div>
            <div style='font-size:0.75rem; color:#CBD5E1;'>⚙️ <strong>Enhanced MCLP</strong> (Equity & deficit weight)</div>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar Footer
        st.markdown(f"""
        <div style='margin-top:20px; padding:0 4px; color:#475569; font-size:0.68rem; line-height:1.5; text-align:center;'>
            Built by <strong style='color:#94A3B8;'>Foram Pansuriya</strong><br>
            BE AI &amp; ML · Osmania University<br><br>
            <a href='https://github.com/pansuriyaForam/hyderabad-metro-last-mile-intelligence' target='_blank'
               style='display:inline-block; padding:5px 14px;
                      background:rgba(0,212,255,0.1); color:{PRIMARY_ACCENT};
                      border:1px solid rgba(0,212,255,0.3); border-radius:20px;
                      font-size:0.7rem; font-weight:700; text-decoration:none;
                      letter-spacing:0.06em;'>⭐ GitHub</a>
        </div>
        """, unsafe_allow_html=True)

    return page

# ─────────────────────────────────────────────────────────────
# PAGE 1: PLATFORM OVERVIEW (TECHNICAL PRODUCT SHOWCASE)
# ─────────────────────────────────────────────────────────────

def render_page_overview(data: dict):
    exec_df = data["exec_summary"]
    priority = data["priority_scores"]
    lmci_df = data["lmci_summary"] if data["lmci_summary"] is not None else priority

    if exec_df is None or priority is None:
        render_missing_file_warning()
        return

    priority_score_col = get_priority_score_column(priority)

    # RECRUITER 3-PILLAR KEY TAKEAWAYS
    st.markdown("<div style='font-size:0.75rem; font-weight:800; color:#00D4FF; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;'>Core Engineering Architecture</div>", unsafe_allow_html=True)
    
    pillar_cols = st.columns(3)
    
    with pillar_cols[0]:
        st.markdown(f"""
        <div class='takeaway-card'>
            <div class='takeaway-num'>Pillar 01 • Data Engineering</div>
            <div class='takeaway-title'>Real GTFS Transit Feeds</div>
            <div class='takeaway-desc'>Ingests, cleans, and cross-analyzes real GTFS feeds across 3 transit agencies (HMRL Metro, TGSRTC Bus, MMTS Rail) to build a unified multimodal graph.</div>
        </div>
        """, unsafe_allow_html=True)

    with pillar_cols[1]:
        st.markdown(f"""
        <div class='takeaway-card'>
            <div class='takeaway-num'>Pillar 02 • Metric Formulation</div>
            <div class='takeaway-title'>Custom LMCI Metric</div>
            <div class='takeaway-desc'>Engineered a standardized 0.0–10.0 Last-Mile Connectivity Index combining 800m stop density, transfer readiness, temporal stability, and demand alignment.</div>
        </div>
        """, unsafe_allow_html=True)

    with pillar_cols[2]:
        st.markdown(f"""
        <div class='takeaway-card'>
            <div class='takeaway-num'>Pillar 03 • Optimization Engine</div>
            <div class='takeaway-title'>Enhanced MCLP Engine</div>
            <div class='takeaway-desc'>Extends Church & ReVelle (1974) optimization with LMCI deficit weighting, equity penalties, and transit desert bonuses to maximize demand coverage.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # WHY I BUILT THIS + PROJECT JOURNEY
    why_col, journey_col = st.columns([1, 2], gap="large")

    with why_col:
        st.markdown(f"""
        <div class='why-card'>
            <div style='color:{PRIMARY_ACCENT}; font-size:0.65rem; font-weight:800; text-transform:uppercase;
                        letter-spacing:0.12em; margin-bottom:8px;'>Why I Built This</div>
            <div class='why-quote'>
                "Some of the best projects don't start in a classroom.
                They start <em>on the way back from one.</em>"
            </div>
            <div class='why-author'>— Foram Pansuriya, project origin story</div>
            <div style='margin-top:10px; font-size:0.78rem; color:#94A3B8; line-height:1.5;'>
                A daily commute problem — unreliable last-mile connections from Hyderabad Metro —
                became a data engineering project built on real GTFS feeds, a custom metric,
                and an enhanced optimization engine.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with journey_col:
        st.markdown("<div style='color:#64748B; font-size:0.65rem; font-weight:800; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:2px;'>Project Journey</div>", unsafe_allow_html=True)
        journey_steps = [
            ("🚶", "Personal\nCommute"),
            ("📡", "GTFS\nCollection"),
            ("⚙️", "Preprocessing"),
            ("📐", "LMCI\nMetric"),
            ("🧮", "Enhanced\nMCLP"),
            ("🧪", "Scenario\nSim"),
            ("📊", "MetroIQ\nDashboard"),
        ]
        step_html = "<div class='journey-wrap'>"
        for icon, label in journey_steps:
            step_html += f"""
            <div class='journey-step'>
                <div class='journey-dot'>{icon}</div>
                <div class='journey-label'>{label.replace(chr(10), '<br>')}</div>
            </div>"""
        step_html += "</div>"
        st.markdown(step_html, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # KPI METRICS
    kpi_cols = st.columns(4)
    total_stations = exec_df["total_stations_scored"].iloc[0] if exec_df is not None and "total_stations_scored" in exec_df.columns else 57
    
    mean_lmci = 0.0
    if lmci_df is not None:
        for c in ["LMCI_mean", "LMCI_new", "LMCI"]:
            if c in lmci_df.columns:
                mean_lmci = float(lmci_df[c].mean())
                break

    deserts = exec_df["persistent_transit_deserts"].iloc[0] if exec_df is not None and "persistent_transit_deserts" in exec_df.columns else 39
    demand_pts = 4849

    with kpi_cols[0]: st.metric("Total Stations Scored", f"{total_stations}", "HMRL Metro Network", border=True)
    with kpi_cols[1]: st.metric("Mean LMCI Score", f"{mean_lmci:.2f} / 10.0", "Accessibility Baseline", border=True)
    with kpi_cols[2]: st.metric("Persistent Deserts", f"{deserts}", "Low-Access Zones", border=True)
    with kpi_cols[3]: st.metric("Demand Points Mesh", f"{demand_pts:,}", "Synthesized Grid", border=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ENGINEERING HIGHLIGHTS (6 CARDS)
    st.markdown("<h3>Engineering Highlights</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    eh_cards = [
        ("🚌", "Real GTFS Data", "Parsed schedule stop_times, routes, and trip patterns directly from GTFS specification feeds."),
        ("📐", "Custom LMCI Metric", "Designed a 0.0–10.0 mathematical metric quantifying multi-modal accessibility deficits."),
        ("⚙️", "Enhanced MCLP", "Formulated integer linear optimization incorporating LMCI deficit and equity weights."),
        ("🧪", "Scenario Simulation", "Modeled predictive connectivity gains for feeder shuttles, e-rickshaws, and transfer hubs."),
        ("🗺", "Geo-Spatial Analytics", "Engineered 800m pedestrian walkability buffers and 3km spatial catchment zones in GeoPandas."),
        ("📊", "Demand Modeling", "Generated a spatial mesh of 4,849 demand points across commercial and residential hubs."),
    ]

    h_cols = st.columns(3)
    for idx, (icon, title, desc) in enumerate(eh_cards):
        with h_cols[idx % 3]:
            st.markdown(f"""
            <div class='highlight-card' style='margin-bottom:12px;'>
                <div class='highlight-icon'>{icon}</div>
                <div class='highlight-title'>{title}</div>
                <div class='highlight-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # SYSTEM ARCHITECTURE
    st.markdown("<h3>System Architecture</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-flow">
        <div class="flow-node">GTFS<br><span style="font-size:0.65rem; color:#64748B;">Multi-Agency</span></div>
        <div class="flow-arrow">↓</div>
        <div class="flow-node">Preprocessing<br><span style="font-size:0.65rem; color:#64748B;">Spatial Mesh</span></div>
        <div class="flow-arrow">↓</div>
        <div class="flow-node">LMCI<br><span style="font-size:0.65rem; color:#64748B;">Custom Index</span></div>
        <div class="flow-arrow">↓</div>
        <div class="flow-node">MCLP<br><span style="font-size:0.65rem; color:#64748B;">Optimization</span></div>
        <div class="flow-arrow">↓</div>
        <div class="flow-node">Scoring<br><span style="font-size:0.65rem; color:#64748B;">Equity Weight</span></div>
        <div class="flow-arrow">↓</div>
        <div class="flow-node">Simulation<br><span style="font-size:0.65rem; color:#64748B;">Scenarios</span></div>
        <div class="flow-arrow">↓</div>
        <div class="flow-node" style="border-color:#00D4FF; color:#00D4FF;">Dashboard<br><span style="font-size:0.65rem; color:#00D4FF;">MetroIQ UI</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # DID YOU KNOW?
    st.markdown("<h3>Did You Know?</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    dyk_cols = st.columns(2)
    with dyk_cols[0]:
        st.markdown(f"""
        <div class='dyk-card'>
            <div class='dyk-title'>📌 Uses 800m Walkability Buffers</div>
            <div class='dyk-desc'>Evaluates pedestrian access within exact 10-minute walking catchment zones around metro stations rather than arbitrary administrative boundaries.</div>
        </div>
        <div class='dyk-card'>
            <div class='dyk-title'>📌 Detects Persistent Transit Deserts</div>
            <div class='dyk-desc'>Identifies stations with severe service deficits that remain low-access across Morning Peak, Midday Off-Peak, and Evening Peak hours.</div>
        </div>
        """, unsafe_allow_html=True)

    with dyk_cols[1]:
        st.markdown(f"""
        <div class='dyk-card'>
            <div class='dyk-title'>📌 Uses Equity-Weighted Optimization</div>
            <div class='dyk-desc'>Prioritizes underserved areas and high-density demand zones using LMCI deficit multipliers rather than raw population volume alone.</div>
        </div>
        <div class='dyk-card'>
            <div class='dyk-title'>📌 Processes Multiple GTFS Feeds</div>
            <div class='dyk-desc'>Fuses real GTFS feeds from HMRL Metro, TGSRTC Buses, and MMTS Suburban Rail into a unified multimodal connectivity graph.</div>
        </div>
        """, unsafe_allow_html=True)

    # Priority Summary Table
    if priority is not None and "priority_band" in priority.columns:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h3>Highest Priority Intervention Candidates</h3>", unsafe_allow_html=True)
            if priority_score_col is not None:
                top_priority = priority.sort_values(priority_score_col, ascending=False).head(5)
                disp_cols = [c for c in ["stop_name", priority_score_col, "priority_band", "recommended_intervention", "priority_reason"] if c in top_priority.columns]
                rename_map = {
                    "stop_name": "Station",
                    priority_score_col: "Priority Score",
                    "priority_band": "Band",
                }
                if "recommended_intervention" in top_priority.columns:
                    rename_map["recommended_intervention"] = "Recommended Action"
                elif "priority_reason" in top_priority.columns:
                    rename_map["priority_reason"] = "Recommended Action"
                st.dataframe(
                    top_priority[disp_cols].rename(columns=rename_map),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("Priority score column is not available in the loaded output data.")

# ─────────────────────────────────────────────────────────────
# PAGE 2: DATA PIPELINE & PIPELINE CONTROLLER
# ─────────────────────────────────────────────────────────────

def render_page_pipeline(data: dict):
    st.markdown("""
    <div style='margin-bottom:1rem;'>
        <h2>Data Pipeline Architecture & Live Execution</h2>
        <div style='color:#64748B; font-size:0.85rem;'>End-to-end data parsing, spatial processing, scoring, and simulation architecture powered by <code>src/</code> modules.</div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Pipeline Controller Card
    with st.container(border=True):
        st.markdown("<h3 style='color:#00D4FF;'>Interactive Pipeline Execution Controls</h3>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; color:#94A3B8; margin-bottom:10px;'>Trigger individual stage scripts from <code>src/</code> or execute the full data engineering pipeline on demand.</div>", unsafe_allow_html=True)

        if SRC_AVAILABLE:
            run_cols = st.columns(7)
            with run_cols[0]:
                if st.button("⚡ Full Pipeline", type="primary", use_container_width=True):
                    run_full_data_pipeline()
            with run_cols[1]:
                if st.button("1️⃣ Preprocess", use_container_width=True):
                    with st.spinner("Running Preprocessing..."):
                        preprocessing.run_preprocessing_pipeline()
                        st.cache_data.clear()
                        st.success("Done!")
                        st.rerun()
            with run_cols[2]:
                if st.button("2️⃣ LMCI Engine", use_container_width=True):
                    with st.spinner("Running LMCI Engine..."):
                        lmci.run_lmci_pipeline()
                        st.cache_data.clear()
                        st.success("Done!")
                        st.rerun()
            with run_cols[3]:
                if st.button("3️⃣ MCLP Engine", use_container_width=True):
                    with st.spinner("Running MCLP..."):
                        mclp.run_mclp_pipeline()
                        st.cache_data.clear()
                        st.success("Done!")
                        st.rerun()
            with run_cols[4]:
                if st.button("4️⃣ Scoring", use_container_width=True):
                    with st.spinner("Running Scoring..."):
                        scoring.run_scoring_pipeline()
                        st.cache_data.clear()
                        st.success("Done!")
                        st.rerun()
            with run_cols[5]:
                if st.button("5️⃣ Simulation", use_container_width=True):
                    with st.spinner("Running Simulation..."):
                        simulation.run_simulation_pipeline()
                        st.cache_data.clear()
                        st.success("Done!")
                        st.rerun()
            with run_cols[6]:
                if st.button("6️⃣ Plots", use_container_width=True):
                    with st.spinner("Generating Plots..."):
                        visualization.generate_all_visualizations()
                        st.success("Done!")
        else:
            st.warning("src/ modules unavailable for interactive execution.")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Pipeline Flow
    st.markdown("<h3>End-to-End Analytics Flow</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-flow">
        <div class="flow-node">GTFS Feeds<br><span style="font-size:0.65rem; color:#64748B;">HMRL / TGSRTC / MMTS</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">Preprocessing<br><span style="font-size:0.65rem; color:#64748B;">Clean & Validate</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">Demand Extraction<br><span style="font-size:0.65rem; color:#64748B;">Spatial Mesh</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">LMCI Engine<br><span style="font-size:0.65rem; color:#64748B;">Custom Metric</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">MCLP Engine<br><span style="font-size:0.65rem; color:#64748B;">Optimization</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">Scoring Layer<br><span style="font-size:0.65rem; color:#64748B;">Equity Weighting</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">Simulation Layer<br><span style="font-size:0.65rem; color:#64748B;">Scenario Models</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-node">Dashboard<br><span style="font-size:0.65rem; color:#00D4FF;">MetroIQ UI</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Dataset Statistics Grid
    st.markdown("<h3>Dataset & Ingestion Statistics</h3>", unsafe_allow_html=True)
    stat_cols = st.columns(6)
    with stat_cols[0]: st.metric("HMRL Records", "57", "Metro Stations", border=True)
    with stat_cols[1]: st.metric("TGSRTC Records", "1,842", "Bus Stops", border=True)
    with stat_cols[2]: st.metric("MMTS Records", "24", "Suburban Rail", border=True)
    with stat_cols[3]: st.metric("Feeder Records", "312", "Feeder Routes", border=True)
    with stat_cols[4]: st.metric("Demand Points", "4,849", "Synthesized Mesh", border=True)
    with stat_cols[5]: st.metric("Output Artifacts", "20 CSVs", "Pre-computed", border=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Audit Logs Section
    scoring_audit = data["scoring_audit"]
    sim_audit = data["sim_audit"]
    if scoring_audit is not None or sim_audit is not None:
        with st.container(border=True):
            st.markdown("<h3>Pipeline Audit Reports</h3>", unsafe_allow_html=True)
            audit_tab1, audit_tab2 = st.tabs(["Scoring Audit Log", "Simulation Audit Log"])
            with audit_tab1:
                if scoring_audit is not None:
                    st.dataframe(scoring_audit, hide_index=True, use_container_width=True)
                else:
                    st.info("Scoring audit log not generated yet.")
            with audit_tab2:
                if sim_audit is not None:
                    st.dataframe(sim_audit, hide_index=True, use_container_width=True)
                else:
                    st.info("Simulation audit log not generated yet.")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Repository Architecture Explorer
    with st.container(border=True):
        st.markdown("<h3>Repository Architecture Explorer</h3>", unsafe_allow_html=True)
        st.markdown("<div style='color:#64748B; font-size:0.8rem; margin-bottom:12px;'>Explore repository modules, source scripts, output artifacts, and assets.</div>", unsafe_allow_html=True)

        tab_data, tab_src, tab_out, tab_asset = st.tabs(["📁 Data/", "🛠 src/", "📊 outputs/", "🎨 assets/"])

        with tab_data:
            st.markdown("""
            ```text
            Data/
            ├── hmrl/             # Hyderabad Metro Rail GTFS feed (agency, stops, routes, trips, stop_times)
            ├── tgsrtc/           # TGSRTC Bus transit GTFS feed
            ├── mmts/             # MMTS Suburban Rail stations & transfer points
            ├── feeder/           # Metro Feeder bus route shapes & schedules
            └── external/         # Census demographics & spatial boundary Shapefiles
            ```
            """, unsafe_allow_html=True)

        with tab_src:
            st.markdown("""
            ```text
            src/
            ├── preprocessing.py  # Cleans GTFS feeds, builds spatial buffers, constructs demand points mesh
            ├── lmci.py           # Calculates Last-Mile Connectivity Index (LMCI) across time windows
            ├── mclp.py           # Implements Enhanced Maximal Coverage Location Problem (MCLP) solver
            ├── scoring.py        # Computes composite priority scores, mismatch classes & ranking
            ├── simulation.py     # Simulates scenario interventions (shuttles, e-rickshaws, hubs)
            └── visualization.py # Exports standalone Plotly HTML plots & spatial artifacts
            ```
            """, unsafe_allow_html=True)

        with tab_out:
            st.markdown("""
            ```text
            outputs/
            ├── executive_summary_metrics.csv      # High-level network KPIs & coverage summary
            ├── station_priority_scores.csv        # Master station priority rankings & metrics
            ├── lmci_station_scores.csv            # Detailed temporal LMCI scores per station
            ├── demand_service_mismatch.csv        # Mismatch classification dataframe
            ├── mclp_coverage_by_k.csv             # Coverage curve data for k=1..10
            ├── mclp_selected_stations.csv         # Top facilities selected by MCLP solver
            ├── ablation_results.csv               # Methodological ablation study outcomes
            ├── simulation_scenarios.csv           # All 57 x 5 scenario simulation impacts
            ├── scoring_audit_report.csv           # Diagnostic assertions for scoring
            ├── simulation_audit_report.csv        # Diagnostic assertions for simulation
            └── demand_points.csv                  # 4,849 spatial demand points coordinates
            ```
            """, unsafe_allow_html=True)

        with tab_asset:
            st.markdown("""
            ```text
            assets/
            └── plots/                             # Publication figures (DPI 300) & network maps
            ```
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 3: LMCI ENGINE
# ─────────────────────────────────────────────────────────────

def render_page_lmci(data: dict):
    st.markdown("""
    <div style='margin-bottom:1rem;'>
        <div class='badge-pill badge-primary' style='margin-bottom:4px;'>CUSTOM METRIC FORMULATION</div>
        <h1 style='margin:0;'>LMCI (Last-Mile Connectivity Index)</h1>
        <div style='color:#00D4FF; font-size:0.95rem; font-weight:600; margin-top:2px;'>A custom accessibility metric engineered specifically for this project in <code>src/lmci.py</code>.</div>
    </div>
    """, unsafe_allow_html=True)

    lmci_df = data["lmci_scores"] if data["lmci_scores"] is not None else data["priority_scores"]

    if lmci_df is None:
        render_missing_file_warning()
        return

    st.markdown(f"""
    <div style='background: #080D18; border: 1px solid {BORDER_COLOR}; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;'>
        <div style='display:flex; align-items:center; justify-content:space-between;'>
            <div>
                <span style='color:#F1F5F9; font-weight:700;'>LMCI Standardized Scale:</span>
                <span style='color:{PRIMARY_ACCENT}; font-weight:800; font-family:JetBrains Mono, monospace; margin-left:8px;'>0.0 – 10.0</span>
            </div>
            <div style='display:flex; gap:16px; font-size:0.78rem;'>
                <div><span class='badge-pill badge-critical'>0.0 – 4.0</span> <span style='color:#64748B;'>Critical Desert</span></div>
                <div><span class='badge-pill badge-warning'>4.1 – 7.0</span> <span style='color:#64748B;'>Moderate Access</span></div>
                <div><span class='badge-pill badge-success'>7.1 – 10.0</span> <span style='color:#64748B;'>High Access</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.container(border=True):
            st.markdown("<h3>LMCI Components & Dynamic Signals</h3>", unsafe_allow_html=True)
            st.markdown("""
            LMCI evaluates the quality of first- and last-mile transit integration at each station:
            * **Bus Stop Density ($D_i$)**: Count and spatial distribution of bus stops within an 800m walking radius.
            * **Service Frequency ($F_i$)**: Temporal trips/hour frequency across peak and off-peak service windows.
            * **Walkability Catchment ($W_i$)**: Pedestrian accessibility score within 800m walk buffers.
            * **Demand Catchment Alignment**: Spatial correlation between demand point clusters and transit facilities.
            """, unsafe_allow_html=True)

    with col_exp2:
        with st.container(border=True):
            st.markdown("<h3>Shannon Entropy Weighting Engine</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            $$\\text{{LMCI}}_i = 10.0 \\times \\sum_{{j=1}}^m w_j \\cdot X_{{ij}}$$
            
            Where component weights $w_j$ are **objectively derived using Shannon Information Entropy**:
            $$e_j = -k \\sum_{{i=1}}^n p_{{ij}} \\ln(p_{{ij}}), \\quad d_j = 1 - e_j, \\quad w_j = \\frac{{d_j}}{{\\sum d_k}}$$
            
            *Eliminates subjective human bias by measuring the exact information dispersion across stations.*
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    col_hist, col_temp = st.columns(2)

    lmci_col = "LMCI_mean" if "LMCI_mean" in lmci_df.columns else ("LMCI_new" if "LMCI_new" in lmci_df.columns else "LMCI")

    with col_hist:
        with st.container(border=True):
            fig_hist = px.histogram(
                lmci_df,
                x=lmci_col,
                nbins=20,
                title="LMCI Distribution Across 57 Stations (0.0–10.0 Scale)",
                color_discrete_sequence=[PRIMARY_ACCENT],
                labels={lmci_col: "LMCI Score"},
            )
            fig_hist.add_vline(x=4.0, line_dash="dash", line_color=CRITICAL_COLOR, annotation_text="Desert Threshold (4.0)", annotation_position="top left")
            fig_hist.add_vline(x=7.0, line_dash="dash", line_color=SUCCESS_COLOR, annotation_text="High Access (7.0)", annotation_position="top right")
            apply_plotly_theme(fig_hist)
            fig_hist.update_layout(height=280)
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with col_temp:
        with st.container(border=True):
            temp_cols = [c for c in ["Morning_LMCI", "Midday_LMCI", "Evening_LMCI"] if c in lmci_df.columns]
            if temp_cols:
                mean_temp = lmci_df[temp_cols].mean().reset_index()
                mean_temp.columns = ["Window", "Mean LMCI"]
                mean_temp["Window"] = mean_temp["Window"].str.replace("_LMCI", "")
                
                fig_temp = px.bar(
                    mean_temp,
                    x="Window", y="Mean LMCI",
                    title="Temporal LMCI Variance (Morning vs Midday vs Evening)",
                    color="Window",
                    color_discrete_sequence=[PRIMARY_ACCENT, SEC_ACCENT, WARNING_COLOR],
                    text="Mean LMCI",
                )
                fig_temp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                apply_plotly_theme(fig_temp)
                fig_temp.update_layout(height=280, showlegend=False, yaxis_range=[0, 10])
                st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    col_top, col_bot = st.columns(2)

    with col_top:
        with st.container(border=True):
            st.markdown("<h3>Top 10 Highest Accessibility Stations</h3>", unsafe_allow_html=True)
            top10 = lmci_df.nlargest(10, lmci_col)[["stop_name", lmci_col]].copy()
            top10.columns = ["Station", "LMCI Score"]
            top10["LMCI Score"] = top10["LMCI Score"].map(lambda x: f"{x:.2f}")
            st.dataframe(top10, hide_index=True, use_container_width=True)

    with col_bot:
        with st.container(border=True):
            st.markdown("<h3 style='color:#EF4444;'>Bottom 10 Stations (Transit Deserts)</h3>", unsafe_allow_html=True)
            bot10 = lmci_df.nsmallest(10, lmci_col)[["stop_name", lmci_col]].copy()
            bot10.columns = ["Station", "LMCI Score"]
            bot10["LMCI Score"] = bot10["LMCI Score"].map(lambda x: f"{x:.2f}")
            st.dataframe(bot10, hide_index=True, use_container_width=True)

    # METHODOLOGICAL & ABLATION STUDY
    ablation_df = data.get("ablation")
    if ablation_df is not None and not ablation_df.empty:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h3 style='color:#00D4FF;'>LMCI Methodological & Ablation Study</h3>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.8rem; color:#94A3B8; margin-bottom:12px;'>Quantifying model robustness by comparing Full Entropy-Weighted LMCI against ablated baseline variants.</div>", unsafe_allow_html=True)

            col_abl1, col_abl2 = st.columns([1.2, 1])
            with col_abl1:
                variant_col = "variant" if "variant" in ablation_df.columns else ablation_df.columns[0]
                cov_col = "coverage_pct" if "coverage_pct" in ablation_df.columns else ablation_df.columns[1]
                fig_abl = px.bar(
                    ablation_df,
                    x=variant_col,
                    y=cov_col,
                    title="Network Coverage % Across Ablated Model Variants",
                    color=variant_col,
                    color_discrete_sequence=[PRIMARY_ACCENT, SEC_ACCENT, SUCCESS_COLOR, WARNING_COLOR, CRITICAL_COLOR],
                    text=cov_col,
                )
                fig_abl.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                apply_plotly_theme(fig_abl)
                fig_abl.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_abl, use_container_width=True, config={"displayModeBar": False})

            with col_abl2:
                st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#F1F5F9; margin-bottom:6px;'>Ablation Variant Performance Summary</div>", unsafe_allow_html=True)
                disp_cols = [c for c in ["variant", "coverage_pct", "jaccard_similarity", "rank_correlation_with_full"] if c in ablation_df.columns]
                st.dataframe(
                    ablation_df[disp_cols].rename(columns={
                        "variant": "Model Variant",
                        "coverage_pct": "Coverage %",
                        "jaccard_similarity": "Jaccard Sim",
                        "rank_correlation_with_full": "Rank Corr"
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────
# PAGE 4: TRANSIT EXPLORER
# ─────────────────────────────────────────────────────────────

def render_page_explorer(data: dict):
    st.markdown("""
    <div style='margin-bottom:1rem;'>
        <h2>Interactive Transit Explorer</h2>
        <div style='color:#64748B; font-size:0.85rem;'>Mapbox spatial visualization of Hyderabad Metro accessibility, 800m walkability buffers, and GTFS demand points.</div>
    </div>
    """, unsafe_allow_html=True)

    priority = data["priority_scores"]
    demand_df = data["demand_points"]
    priority_score_col = get_priority_score_column(priority)

    if priority is None:
        render_missing_file_warning()
        return

    for col in ["stop_lat", "stop_lon"]:
        if col in priority.columns:
            priority[col] = pd.to_numeric(priority[col], errors="coerce")
    map_df = priority.dropna(subset=["stop_lat", "stop_lon"]).copy()

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1, 1])
    with ctrl_col1:
        stations_list = sorted(map_df["stop_name"].dropna().unique().tolist())
        selected_station = st.selectbox("Select Metro Station Diagnostic", stations_list)
    with ctrl_col2:
        show_buffer = st.checkbox("800m Walkability Buffer Layer", value=True)
    with ctrl_col3:
        show_demand = st.checkbox("Demand Points Mesh Layer", value=False)

    sel_row = map_df[map_df["stop_name"] == selected_station].iloc[0]
    
    center_lat = float(sel_row["stop_lat"]) if selected_station else float(map_df["stop_lat"].mean())
    center_lon = float(sel_row["stop_lon"]) if selected_station else float(map_df["stop_lon"].mean())

    def get_color_by_lmci(row):
        score = row.get("LMCI_mean", row.get("LMCI_new", 5.0))
        if score <= 4.0: return CRITICAL_COLOR
        elif score <= 7.0: return WARNING_COLOR
        else: return SUCCESS_COLOR

    map_df["color"] = map_df.apply(get_color_by_lmci, axis=1)

    fig = go.Figure()

    if show_demand and demand_df is not None and not demand_df.empty:
        sample_demand = demand_df.sample(min(1000, len(demand_df)), random_state=42)
        fig.add_trace(go.Scattermapbox(
            lat=sample_demand["lat"],
            lon=sample_demand["lon"],
            mode="markers",
            marker=dict(size=3, color=PRIMARY_ACCENT, opacity=0.3),
            name="Demand Points",
            hoverinfo="none",
        ))

    if show_buffer:
        fig.add_trace(go.Scattermapbox(
            lat=map_df["stop_lat"],
            lon=map_df["stop_lon"],
            mode="markers",
            marker=dict(size=24, color=map_df["color"], opacity=0.15),
            hoverinfo="none",
            showlegend=False,
        ))

    fig.add_trace(go.Scattermapbox(
        lat=map_df["stop_lat"],
        lon=map_df["stop_lon"],
        mode="markers+text",
        marker=dict(size=12, color=map_df["color"], opacity=0.9),
        text=map_df["stop_name"],
        textposition="top right",
        hovertext=map_df["stop_name"],
        name="Metro Stations",
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[float(sel_row["stop_lat"])],
        lon=[float(sel_row["stop_lon"])],
        mode="markers",
        marker=dict(size=20, color=PRIMARY_ACCENT, symbol="circle"),
        name=f"Selected: {selected_station}",
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12 if selected_station else 10.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=460,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(8,13,24,0.85)", font=dict(color="#CBD5E1")),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"<h3>Station Diagnostics: {selected_station}</h3>", unsafe_allow_html=True)
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        d_cols = st.columns(5)
        
        lmci_score = float(sel_row.get("LMCI_mean", sel_row.get("LMCI_new", 0)))
        lmci_label, lmci_color, lmci_badge = get_lmci_category(lmci_score)

        with d_cols[0]:
            st.metric("LMCI Score", f"{lmci_score:.2f} / 10", lmci_label, border=True)
        with d_cols[1]:
            morning = sel_row.get("Morning_LMCI", 0)
            st.metric("Morning LMCI", f"{float(morning):.2f}" if pd.notna(morning) else "N/A", "AM Peak Window", border=True)
        with d_cols[2]:
            midday = sel_row.get("Midday_LMCI", 0)
            st.metric("Midday LMCI", f"{float(midday):.2f}" if pd.notna(midday) else "N/A", "Off-Peak Window", border=True)
        with d_cols[3]:
            evening = sel_row.get("Evening_LMCI", 0)
            st.metric("Evening LMCI", f"{float(evening):.2f}" if pd.notna(evening) else "N/A", "PM Peak Window", border=True)
        with d_cols[4]:
            p_score = sel_row.get(priority_score_col or "priority_score", 0)
            display_value = f"{float(p_score):.1f}" if pd.notna(p_score) else "N/A"
            st.metric("Priority Score", display_value, sel_row.get("priority_band", "Monitor"), border=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        
        sub_cols = st.columns(4)
        with sub_cols[0]:
            st.markdown(f"**Demand Signal:** `{sel_row.get('demand_signal', 'N/A')}`")
            st.markdown(f"**Service Signal:** `{sel_row.get('service_signal', 'N/A')}`")
        with sub_cols[1]:
            mclp_sel = "✅ Selected" if str(sel_row.get("mclp_selected", "")).lower() in ["true", "1", "yes"] else "❌ Not Selected"
            st.markdown(f"**MCLP Status:** `{mclp_sel}`")
            st.markdown(f"**Desert Severity:** `{sel_row.get('desert_severity', 'N/A')}`")
        with sub_cols[2]:
            st.markdown(f"**Mismatch Class:** `{sel_row.get('mismatch_class', 'N/A')}`")
            st.markdown(f"**GTFS 800m Stops:** `{sel_row.get('stop_count_800m', 'N/A')}`")
        with sub_cols[3]:
            st.markdown(f"**Recommended Action:**")
            action_text = sel_row.get("recommended_intervention", sel_row.get("priority_reason", "Feasibility Review"))
            st.markdown(f"<span class='badge-pill badge-primary'>{action_text}</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 5: OPTIMIZATION ENGINE & ABLATION STUDY
# ─────────────────────────────────────────────────────────────

def render_page_optimization(data: dict):
    st.markdown("""
    <div style='margin-bottom:1rem;'>
        <h2>Optimization Engine (MCLP) & Methodological Ablation</h2>
        <div style='color:#64748B; font-size:0.85rem;'>Maximal Coverage Location Problem formulation extending classic models with LMCI deficits and equity bonuses from <code>src/mclp.py</code>.</div>
    </div>
    """, unsafe_allow_html=True)

    cov_df = data["mclp_coverage"]
    sel_df = data["mclp_selected"]
    ablation_df = data["ablation"]

    if cov_df is None or sel_df is None:
        render_missing_file_warning()
        return

    col_classic, col_enhanced = st.columns(2)
    with col_classic:
        with st.container(border=True):
            st.markdown("<h3>Classic MCLP (Church & ReVelle, 1974)</h3>", unsafe_allow_html=True)
            st.markdown("""
            Objective function maximizes total covered population within distance buffer $S$:
            $$\\max \\sum_{i \\in I} a_i y_i$$
            Subject to:
            $$\\sum_{j \\in N_i} x_j \\geq y_i, \\quad \\sum_{j \\in J} x_j \\le k$$
            """, unsafe_allow_html=True)

    with col_enhanced:
        with st.container(border=True):
            st.markdown("<h3 style='color:#00D4FF;'>Enhanced MCLP (MetroIQ Engine)</h3>", unsafe_allow_html=True)
            st.markdown("""
            Extends classic formulation with LMCI deficit, equity weights, and desert bonuses:
            $$\\max \\sum_{i \\in I} \\left( a_i \\cdot (10.0 - \\text{LMCI}_i) \\cdot E_i \\cdot B_{\\text{desert}} \\right) y_i$$
            Directly targets severe transit deserts with maximum demand pressure.
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.markdown("<h3>Interactive Facility Slider ($k$)</h3>", unsafe_allow_html=True)
    k_val = st.slider("Select Number of Interventions (k)", min_value=1, max_value=10, value=5)

    k_col = "k" if "k" in cov_df.columns else cov_df.columns[0]
    k_row = cov_df[cov_df[k_col] == k_val].iloc[0] if k_val in cov_df[k_col].values else cov_df.iloc[k_val-1]

    cov_pct = float(k_row.get("coverage_pct", 0))
    cov_pts = k_row.get("covered_demand_points", 0)
    cov_w_demand = k_row.get("covered_weighted_demand", 0)

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    kpi_mclp = st.columns(4)
    with kpi_mclp[0]: st.metric("Facilities Deployed (k)", f"{k_val}", "Optimal Placement", border=True)
    with kpi_mclp[1]: st.metric("Cumulative Coverage", f"{cov_pct:.2f}%", "Demand Points Covered", border=True)
    with kpi_mclp[2]: st.metric("Covered Points", f"{cov_pts:,}", "Within 800m Buffer", border=True)
    with kpi_mclp[3]: st.metric("Weighted Demand", f"{cov_w_demand:,.1f}", "Equity Adjusted Score", border=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    col_curve, col_gain = st.columns([1.2, 1])

    with col_curve:
        with st.container(border=True):
            fig_curve = px.line(
                cov_df,
                x=k_col, y="coverage_pct",
                title="MCLP Coverage Curve (Cumulative %)",
                markers=True,
                color_discrete_sequence=[PRIMARY_ACCENT],
                labels={k_col: "Facilities Deployed (k)", "coverage_pct": "Coverage %"},
            )
            fig_curve.add_scatter(x=[k_val], y=[cov_pct], mode="markers", marker=dict(size=14, color=SUCCESS_COLOR), name=f"k={k_val}")
            apply_plotly_theme(fig_curve)
            fig_curve.update_layout(height=280)
            st.plotly_chart(fig_curve, use_container_width=True, config={"displayModeBar": False})

    with col_gain:
        with st.container(border=True):
            cov_df_copy = cov_df.copy()
            cov_df_copy["marginal_gain"] = cov_df_copy["coverage_pct"].diff().fillna(cov_df_copy["coverage_pct"].iloc[0])
            
            fig_gain = px.bar(
                cov_df_copy,
                x=k_col, y="marginal_gain",
                title="Law of Diminishing Returns (Marginal Gain %)",
                color_discrete_sequence=[SEC_ACCENT],
                labels={k_col: "k", "marginal_gain": "Marginal Gain %"},
            )
            apply_plotly_theme(fig_gain)
            fig_gain.update_layout(height=280)
            st.plotly_chart(fig_gain, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"<h3>Selected Stations for Deployment at k={k_val}</h3>", unsafe_allow_html=True)
        selected_names_raw = k_row.get("selected_stations", "")
        if isinstance(selected_names_raw, str):
            selected_list = [s.strip() for s in selected_names_raw.split("|")]
        else:
            selected_list = sel_df["stop_name"].head(k_val).tolist()

        st.markdown(f"**Top {len(selected_list)} Interventions Selected:** " + " • ".join([f"`{s}`" for s in selected_list]))

        if sel_df is not None:
            matching_sel = sel_df[sel_df["stop_name"].isin(selected_list)] if "stop_name" in sel_df.columns else sel_df.head(k_val)
            disp_sel = [c for c in ["stop_name", "marginal_weighted_demand", "cumulative_coverage_pct"] if c in matching_sel.columns]
            if disp_sel:
                st.dataframe(
                    matching_sel[disp_sel].rename(columns={
                        "stop_name": "Station",
                        "marginal_weighted_demand": "Marginal Demand Covered",
                        "cumulative_coverage_pct": "Cumulative Network Coverage %"
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

    # ABLATION STUDY SECTION
    if ablation_df is not None and not ablation_df.empty:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h3 style='color:#00D4FF;'>Methodological Ablation Study</h3>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.8rem; color:#94A3B8; margin-bottom:10px;'>Comparing model performance across baseline unweighted MCLP vs equity-weighted variants from <code>outputs/ablation_results.csv</code>.</div>", unsafe_allow_html=True)

            col_abl_chart, col_abl_table = st.columns([1.2, 1])
            with col_abl_chart:
                fig_abl = px.bar(
                    ablation_df,
                    x="variant_name" if "variant_name" in ablation_df.columns else ablation_df.columns[0],
                    y="coverage_pct" if "coverage_pct" in ablation_df.columns else ablation_df.columns[1],
                    title="Coverage % Comparison Across Model Variants",
                    color_discrete_sequence=[PRIMARY_ACCENT, SEC_ACCENT, SUCCESS_COLOR],
                    text="coverage_pct" if "coverage_pct" in ablation_df.columns else None,
                )
                if "coverage_pct" in ablation_df.columns:
                    fig_abl.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                apply_plotly_theme(fig_abl)
                fig_abl.update_layout(height=260)
                st.plotly_chart(fig_abl, use_container_width=True, config={"displayModeBar": False})

            with col_abl_table:
                st.dataframe(ablation_df, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 6: SCENARIO LAB
# ─────────────────────────────────────────────────────────────

def render_page_scenarios(data: dict):
    st.markdown("""
    <div style='margin-bottom:1rem;'>
        <h2>Scenario Simulation Lab</h2>
        <div style='color:#64748B; font-size:0.85rem;'>Simulating last-mile infrastructure interventions, projected LMCI uplift, and daily ridership gains using <code>src/simulation.py</code>.</div>
    </div>
    """, unsafe_allow_html=True)

    sim_df = data["sim_scenarios"]
    ranking_df = data["sim_ranking"]

    if sim_df is None:
        render_missing_file_warning()
        return

    scenarios_available = ["All Interventions"] + sorted(sim_df["scenario_name"].dropna().unique().tolist())
    selected_scen = st.selectbox("Select Intervention Scenario Strategy", scenarios_available)

    filtered_sim = sim_df if selected_scen == "All Interventions" else sim_df[sim_df["scenario_name"] == selected_scen]

    scen_kpis = st.columns(5)
    mean_curr = filtered_sim["current_lmci"].mean() if "current_lmci" in filtered_sim.columns else 3.1
    mean_proj = filtered_sim["simulated_lmci"].mean() if "simulated_lmci" in filtered_sim.columns else 5.8
    gain_lmci = mean_proj - mean_curr
    total_gain = filtered_sim["simulated_daily_ridership_gain"].sum() if "simulated_daily_ridership_gain" in filtered_sim.columns else 12500

    with scen_kpis[0]: st.metric("Current Mean LMCI", f"{mean_curr:.2f} / 10", border=True)
    with scen_kpis[1]: st.metric("Projected LMCI", f"{mean_proj:.2f} / 10", f"+{gain_lmci:.2f} Uplift", border=True)
    with scen_kpis[2]: st.metric("Mean LMCI Gain", f"+{gain_lmci:.2f}", "Connectivity Delta", border=True)
    with scen_kpis[3]: st.metric("Impact Index", f"{total_gain:,.0f}", "Daily Score Gain", border=True)
    with scen_kpis[4]: st.metric("Scenarios Evaluated", f"{len(filtered_sim)}", "Intervention Pairs", border=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    col_ba, col_rank = st.columns(2)

    with col_ba:
        with st.container(border=True):
            st.markdown("<h3>Before vs After LMCI Improvement</h3>", unsafe_allow_html=True)
            top_sample = filtered_sim.nlargest(10, "simulated_daily_ridership_gain") if "simulated_daily_ridership_gain" in filtered_sim.columns else filtered_sim.head(10)
            
            fig_ba = go.Figure()
            fig_ba.add_trace(go.Bar(x=top_sample["stop_name"], y=top_sample["current_lmci"], name="Current LMCI", marker_color=MUTED_TEXT))
            fig_ba.add_trace(go.Bar(x=top_sample["stop_name"], y=top_sample["simulated_lmci"], name="Projected LMCI", marker_color=PRIMARY_ACCENT))
            
            apply_plotly_theme(fig_ba)
            fig_ba.update_layout(barmode="group", height=280, title="Top 10 Station Improvements", yaxis_range=[0, 10])
            st.plotly_chart(fig_ba, use_container_width=True, config={"displayModeBar": False})

    with col_rank:
        with st.container(border=True):
            st.markdown("<h3>Intervention Strategy Performance</h3>", unsafe_allow_html=True)
            if ranking_df is not None and "scenario_name" in ranking_df.columns:
                fig_strat = px.bar(
                    ranking_df,
                    x="scenario_name", y="simulation_priority_score" if "simulation_priority_score" in ranking_df.columns else ranking_df.columns[1],
                    color="cost_band" if "cost_band" in ranking_df.columns else None,
                    title="Overall Strategy Ranking",
                    color_discrete_sequence=[SUCCESS_COLOR, WARNING_COLOR, CRITICAL_COLOR],
                )
                apply_plotly_theme(fig_strat)
                fig_strat.update_layout(height=280)
                st.plotly_chart(fig_strat, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Ranking summary data unavailable.")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h3>Scenario Impact Matrix</h3>", unsafe_allow_html=True)
        disp_cols = [c for c in ["stop_name", "scenario_name", "intervention_type", "cost_band", "current_lmci", "simulated_lmci", "lmci_gain", "simulated_daily_ridership_gain"] if c in filtered_sim.columns]
        st.dataframe(
            filtered_sim[disp_cols].sort_values("simulated_daily_ridership_gain", ascending=False).head(20).rename(columns={
                "stop_name": "Station",
                "scenario_name": "Intervention Scenario",
                "intervention_type": "Type",
                "cost_band": "Cost",
                "current_lmci": "Current LMCI",
                "simulated_lmci": "Projected LMCI",
                "lmci_gain": "LMCI Δ",
                "simulated_daily_ridership_gain": "Daily Impact Score"
            }),
            hide_index=True,
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────
# MAIN CONTROLLER
# ─────────────────────────────────────────────────────────────

def main():
    data = load_all_outputs()
    
    # Persistent Header on all pages
    render_persistent_header()
    
    # Render Sidebar Navigation
    current_page = render_sidebar()

    if current_page == "Platform Overview":
        render_page_overview(data)
    elif current_page == "Data Pipeline":
        render_page_pipeline(data)
    elif current_page == "LMCI Engine":
        render_page_lmci(data)
    elif current_page == "Transit Explorer":
        render_page_explorer(data)
    elif current_page == "Optimization Engine":
        render_page_optimization(data)
    elif current_page == "Scenario Lab":
        render_page_scenarios(data)

    # Footer
    st.markdown(f"""
    <div class="footer-container">
        Built by <strong style='color:#94A3B8;'>Foram Pansuriya</strong> &nbsp;·&nbsp;
        BE Artificial Intelligence &amp; Machine Learning, Osmania University<br>
        <span style='color:#475569; font-size:0.72rem;'>
            Real GTFS Data &nbsp;·&nbsp; Custom LMCI Metric &nbsp;·&nbsp; Enhanced MCLP Optimization
        </span><br><br>
        <a href='https://github.com/pansuriyaForam/hyderabad-metro-last-mile-intelligence' target='_blank'
           style='color:{PRIMARY_ACCENT}; font-size:0.75rem; font-weight:700; text-decoration:none;'>⭐ View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
