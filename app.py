import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our data processor
from data_processor import BankingDataProcessor

# Page configuration
st.set_page_config(
    page_title="Systemic Risk Analysis Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    .feature-card h3 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.4rem;
    }
    .feature-card p {
        color: rgba(255,255,255,0.9);
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .feature-card ul {
        color: rgba(255,255,255,0.9);
        margin: 0;
        padding-left: 1.5rem;
    }
    .feature-card li {
        margin-bottom: 0.3rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 2px solid #e1e5e9;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.1);
    }
    .metric-card h3 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #2c3e50;
        margin: 0.5rem 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-card p {
        color: #666;
        margin: 0;
        font-size: 0.9rem;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    .stats-box {
        background-color: #f8fbff;
        border: 2px solid #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stats-box h4 {
        color: #1976d2;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .stats-box p {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .stats-box ul {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .stats-box li {
        margin-bottom: 0.3rem;
    }
    .warning-box {
        background-color: #fffbf0;
        border: 2px solid #ffeaa7;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-box h4 {
        color: #d68910;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .warning-box ol {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .warning-box li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .success-box {
        background-color: #f0fff4;
        border: 2px solid #c6f6d5;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-box h4 {
        color: #38a169;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .success-box p {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .success-box ol {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .success-box li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Improve overall readability */
    .stMarkdown {
        color: #333;
    }
    
    /* Better button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Better text contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    
    /* Improve list readability */
    ul, ol {
        color: #555;
    }
    
    /* Better link colors */
    a {
        color: #1f77b4;
    }
    a:hover {
        color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("## 🧭 Navigation")
if st.sidebar.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
    st.switch_page("dashboard.py")
if st.sidebar.button("📚 Methodology", key="nav_methodology", use_container_width=True):
    st.switch_page("methodology.py")
if st.sidebar.button("⚠️ Early Warning", key="nav_ml", use_container_width=True):
    st.switch_page("machinelearning.py")

# Header
st.markdown('<h1 class="main-header">🏦 Systemic Risk Analysis Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced banking systemic risk analysis using Extreme Value Theory and Machine Learning</p>', unsafe_allow_html=True)

# Main overview section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Real-Time Dashboard</h3>
        <p>Interactive analysis of 28 global banks with user-configurable selection and real-time data from Yahoo Finance.</p>
        <ul>
        <li>Live systemic risk metrics</li>
        <li>Regional comparisons</li>
        <li>Time series analysis</li>
        <li>Risk alerts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📚 EVT Methodology</h3>
        <p>Comprehensive guide to Extreme Value Theory with interactive demonstrations and real data examples.</p>
        <ul>
        <li>Value-at-Risk (VaR)</li>
        <li>Hill Estimator</li>
        <li>Tail Dependence</li>
        <li>Systemic Beta</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>⚠️ Early Warning System</h3>
        <p>Machine learning-based crisis prediction with 8-10 weeks advance warning using 22 risk features.</p>
        <ul>
        <li>Risk prediction</li>
        <li>Scenario testing</li>
        <li>Feature analysis</li>
        <li>Real data integration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Quick start section
st.markdown("## 🚀 Quick Start")
st.markdown("Choose your analysis path:")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Launch Dashboard", key="quick_dashboard", use_container_width=True):
        st.switch_page("dashboard.py")

with col2:
    if st.button("📚 Learn Methodology", key="quick_methodology", use_container_width=True):
        st.switch_page("methodology.py")

with col3:
    if st.button("⚠️ Test Early Warning", key="quick_ml", use_container_width=True):
        st.switch_page("machinelearning.py")

# System overview
st.markdown("## 📈 System Overview")

# Available banks summary
processor = BankingDataProcessor()
banks_by_region = processor.get_banks_by_region()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🏦 Total Banks</h3>
        <h2>28</h2>
        <p>Global Systemically Important Banks</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    americas_count = len(banks_by_region.get('Americas', []))
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌎 Americas</h3>
        <h2>{americas_count}</h2>
        <p>Banks</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    europe_count = len(banks_by_region.get('Europe', []))
    st.markdown(f"""
    <div class="metric-card">
        <h3>🇪🇺 Europe</h3>
        <h2>{europe_count}</h2>
        <p>Banks</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    asia_count = len(banks_by_region.get('Asia/Pacific', []))
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌏 Asia/Pacific</h3>
        <h2>{asia_count}</h2>
        <p>Banks</p>
    </div>
    """, unsafe_allow_html=True)

# Key metrics explanation
st.markdown("## 📊 Key Risk Metrics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="stats-box">
        <h4>⚖️ Systemic Beta (βT)</h4>
        <p><strong>What it measures:</strong> A bank's contribution to systemic risk</p>
        <p><strong>Formula:</strong> βT = (τ^(1/ξ)) × (VaR_bank / VaR_market)</p>
        <ul>
        <li>βT > 2.0: High systemic risk</li>
        <li>1.5 < βT ≤ 2.0: Medium risk</li>
        <li>βT ≤ 1.5: Low risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats-box">
        <h4>📊 Value-at-Risk (VaR)</h4>
        <p><strong>What it measures:</strong> Maximum potential loss at specified confidence level</p>
        <p><strong>Available levels:</strong> 95% and 99% confidence</p>
        <ul>
        <li>Higher VaR = Higher risk</li>
        <li>Used for capital requirements</li>
        <li>Regulatory standard</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-box">
        <h4>🔍 Hill Estimator (ξ)</h4>
        <p><strong>What it measures:</strong> Tail index of return distributions</p>
        <p><strong>Interpretation:</strong> Higher values indicate heavier tails</p>
        <ul>
        <li>ξ > 0.5: Very heavy tails</li>
        <li>0.3 < ξ ≤ 0.5: Heavy tails</li>
        <li>ξ ≤ 0.3: Moderate tails</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats-box">
        <h4>🔗 Tail Dependence (τ)</h4>
        <p><strong>What it measures:</strong> Probability of joint extreme events</p>
        <p><strong>Formula:</strong> τ = P(Y > threshold | X > threshold)</p>
        <ul>
        <li>τ = 1: Perfect dependence</li>
        <li>τ = 0: No dependence</li>
        <li>τ > 0.5: High contagion risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Methodology overview
st.markdown("## 🔬 Methodology Overview")

st.markdown("""
<div class="success-box">
<h4>🎯 Extreme Value Theory (EVT) Framework</h4>
<p>Our system uses advanced statistical methods to measure systemic risk in banking institutions:</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="warning-box">
    <h4>📋 Data Processing</h4>
    <ol>
    <li><strong>Data Collection:</strong> Daily prices from Yahoo Finance</li>
    <li><strong>Resampling:</strong> Weekly returns (Friday close)</li>
    <li><strong>Rolling Windows:</strong> 52-week (1 year) analysis</li>
    <li><strong>Regional Mapping:</strong> Bank-to-index associations</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
    <h4>🔧 Risk Calculation</h4>
    <ol>
    <li><strong>VaR Estimation:</strong> Percentile-based risk measure</li>
    <li><strong>Hill Estimator:</strong> Adaptive threshold selection</li>
    <li><strong>Tail Dependence:</strong> Extreme event correlation</li>
    <li><strong>Systemic Beta:</strong> Combined risk measure</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Available banks list
st.markdown("## 🏦 Available Banks")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🌎 Americas**")
    for bank in banks_by_region.get('Americas', []):
        st.markdown(f"• {bank}")

with col2:
    st.markdown("**🇪🇺 Europe**")
    for bank in banks_by_region.get('Europe', []):
        st.markdown(f"• {bank}")

with col3:
    st.markdown("**🌏 Asia/Pacific**")
    for bank in banks_by_region.get('Asia/Pacific', []):
        st.markdown(f"• {bank}")

# Use cases
st.markdown("## 🎯 Use Cases")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>🏛️ Regulatory Compliance</h4>
        <p>Monitor systemic risk for regulatory reporting and stress testing requirements.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>💰 Risk Management</h4>
        <p>Identify high-risk banks and potential contagion effects for portfolio decisions.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>📚 Academic Research</h4>
        <p>Study systemic risk patterns and validate theoretical models with real data.</p>
    </div>
    """, unsafe_allow_html=True)

# Technical details
st.markdown("## 🔧 Technical Details")

st.markdown("""
<div class="stats-box">
<h4>📊 Data Specifications</h4>
<ul>
<li><strong>Data Source:</strong> Yahoo Finance via yfinance library</li>
<li><strong>Frequency:</strong> Weekly returns (Friday close prices)</li>
<li><strong>Window Size:</strong> 52 weeks (1 year) rolling window</li>
<li><strong>Regional Indices:</strong> S&P 500, FTSE 100, DAX, Nikkei 225, etc.</li>
<li><strong>Confidence Levels:</strong> 95% and 99%</li>
<li><strong>Lead Time:</strong> 8-10 weeks (Early Warning System)</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Getting started guide
st.markdown("## 🚀 Getting Started Guide")

st.markdown("""
<div class="success-box">
<h4>📋 Step-by-Step Guide</h4>
<ol>
<li><strong>Choose Your Analysis:</strong> Select from Dashboard, Methodology, or Early Warning</li>
<li><strong>Configure Parameters:</strong> Select banks, date ranges, and confidence levels</li>
<li><strong>Load Data:</strong> Download and process real banking data</li>
<li><strong>Analyze Results:</strong> Interpret risk metrics and visualizations</li>
<li><strong>Take Action:</strong> Use insights for decision-making</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
<h3>🏦 Systemic Risk Analysis Platform</h3>
<p><strong>Built with:</strong> Streamlit, Python, Extreme Value Theory, and Machine Learning</p>
<p><strong>Data Source:</strong> Yahoo Finance | <strong>Methodology:</strong> Advanced EVT Framework</p>
<p>This platform provides comprehensive systemic risk analysis for global banking institutions.</p>
</div>
""", unsafe_allow_html=True)
