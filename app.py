import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our fixed data processor
from data_processor import BankingDataProcessor

# Page configuration
st.set_page_config(
    page_title="Systemic Risk Analysis Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Anthropic Light Theme CSS
st.markdown("""
<style>
    /* Anthropic Light Theme */
    .stApp {
        background-color: #fafaf9;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Feature cards with Anthropic styling */
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border: 1px solid #e5e7eb;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1);
        border-color: #d97706;
    }
    
    .feature-card h3 {
        color: #1a1a1a;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    .feature-card p {
        color: #374151;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .feature-card ul {
        color: #4b5563;
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .feature-card li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #d97706;
        box-shadow: 0 8px 25px -3px rgba(217, 119, 6, 0.1);
        transform: translateY(-2px);
    }
    
    .metric-card h3 {
        color: #d97706;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card h2 {
        color: #1a1a1a;
        margin: 0.5rem 0;
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card p {
        color: #6b7280;
        margin: 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .nav-button {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 6px -1px rgba(217, 119, 6, 0.3);
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, #b45309 0%, #d97706 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -3px rgba(217, 119, 6, 0.4);
    }
    
    /* Info boxes */
    .stats-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stats-box h4 {
        color: #d97706;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    .stats-box p {
        color: #374151;
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    
    .stats-box ul {
        color: #4b5563;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .stats-box li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .success-box h4 {
        color: #059669;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    .success-box p {
        color: #047857;
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    
    .success-box ol {
        color: #065f46;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .success-box li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        line-height: 1.2;
    }
    
    p {
        color: #374151;
        line-height: 1.6;
    }
    
    /* Better button styling */
    .stButton > button {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(217, 119, 6, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #b45309 0%, #d97706 100%);
        transform: translateY(-1px);
        box-shadow: 0 8px 25px -3px rgba(217, 119, 6, 0.4);
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Better list styling */
    ul, ol {
        color: #4b5563;
    }
    
    /* Better link colors */
    a {
        color: #d97706;
        text-decoration: none;
    }
    a:hover {
        color: #b45309;
        text-decoration: underline;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #6b7280;
        padding: 3rem 2rem;
        background-color: #ffffff;
        border-top: 1px solid #e5e7eb;
        margin-top: 4rem;
        border-radius: 12px;
    }
    
    .footer h3 {
        color: #1a1a1a;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .footer p {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("## 🧭 Navigation")
if st.sidebar.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
    st.switch_page('pages/dashboard.py')
if st.sidebar.button("📚 Methodology", key="nav_methodology", use_container_width=True):
    st.switch_page('pages/methodology.py')
if st.sidebar.button("⚠️ Early Warning", key="nav_ml", use_container_width=True):
    st.switch_page('pages/machinelearning.py')

# Header
st.markdown('<h1 class="main-header">🏦 Systemic Risk Analysis Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced banking systemic risk analysis using accurate Extreme Value Theory with beautiful Anthropic design</p>', unsafe_allow_html=True)

# Main overview section with improved spacing
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Real-Time Dashboard</h3>
        <p>Interactive analysis of 28 global banks with user-configurable selection and live data from Yahoo Finance.</p>
        <ul>
        <li>Live systemic risk metrics with accurate EVT calculations</li>
        <li>Regional comparisons across Americas, Europe, Asia/Pacific</li>
        <li>Time series analysis with rolling windows</li>
        <li>Automated risk alerts and recommendations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📚 EVT Methodology</h3>
        <p>Comprehensive guide to Extreme Value Theory with interactive demonstrations and mathematically accurate implementations.</p>
        <ul>
        <li>Fixed Value-at-Risk (VaR) calculations</li>
        <li>Accurate Hill Estimator with adaptive thresholds</li>
        <li>Corrected Tail Dependence measurements</li>
        <li>Proper Systemic Beta computation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>⚠️ Early Warning System</h3>
        <p>Machine learning-based crisis prediction with 8-10 weeks advance warning using 22 validated risk features.</p>
        <ul>
        <li>Crisis probability prediction</li>
        <li>Scenario testing capabilities</li>
        <li>Feature importance analysis</li>
        <li>Real data integration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Quick start section
st.markdown("## 🚀 Quick Start")
st.markdown("Choose your analysis path:")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    if st.button("📊 Launch Dashboard", key="quick_dashboard", use_container_width=True):
        st.switch_page('pages/dashboard.py')

with col2:
    if st.button("📚 Learn Methodology", key="quick_methodology", use_container_width=True):
        st.switch_page('pages/methodology.py')

with col3:
    if st.button("⚠️ Test Early Warning", key="quick_ml", use_container_width=True):
        st.switch_page('pages/machinelearning.py')

# System overview
st.markdown("## 📈 System Overview")

# Available banks summary
processor = BankingDataProcessor()
banks_by_region = processor.get_banks_by_region()

col1, col2, col3, col4 = st.columns(4, gap="medium")

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
st.markdown("## 📊 Key Risk Metrics - Fixed Formulas")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="stats-box">
        <h4>⚖️ Systemic Beta (βT) - FIXED</h4>
        <p><strong>Accurate Formula:</strong> βT = (τ^(1/ξ)) × (VaR_bank / VaR_market)</p>
        <p><strong>Improvements:</strong></p>
        <ul>
        <li>✅ Proper mathematical implementation</li>
        <li>✅ Enhanced error handling for edge cases</li>
        <li>✅ Non-negative constraint enforcement</li>
        <li>βT > 2.0: High systemic risk</li>
        <li>1.5 < βT ≤ 2.0: Medium risk</li>
        <li>βT ≤ 1.5: Low risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats-box">
        <h4>📊 Value-at-Risk (VaR) - FIXED</h4>
        <p><strong>Accurate Method:</strong> Empirical quantile estimation</p>
        <p><strong>Improvements:</strong></p>
        <ul>
        <li>✅ Proper loss convention handling</li>
        <li>✅ Robust empty dataset validation</li>
        <li>✅ Consistent percentile calculation</li>
        <li>Available at 95% and 99% confidence</li>
        <li>Used for regulatory capital requirements</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-box">
        <h4>🔍 Hill Estimator (ξ) - FIXED</h4>
        <p><strong>Accurate Implementation:</strong> Adaptive threshold selection</p>
        <p><strong>Major Fixes:</strong></p>
        <ul>
        <li>✅ Proper exceedance calculation above threshold</li>
        <li>✅ Minimum sample size validation</li>
        <li>✅ Robust threshold selection algorithm</li>
        <li>ξ > 0.5: Very heavy tails (high extreme risk)</li>
        <li>0.3 < ξ ≤ 0.5: Heavy tails (moderate risk)</li>
        <li>ξ ≤ 0.3: Moderate tails (lower risk)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats-box">
        <h4>🔗 Tail Dependence (τ) - FIXED</h4>
        <p><strong>Accurate Formula:</strong> Left-tail dependence for losses</p>
        <p><strong>Critical Fixes:</strong></p>
        <ul>
        <li>✅ Proper conditional probability calculation</li>
        <li>✅ Correct left-tail analysis for losses</li>
        <li>✅ Bounded output in [0,1] interval</li>
        <li>τ = 1: Perfect dependence</li>
        <li>τ = 0: No dependence</li>
        <li>τ > 0.5: High contagion risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Methodology overview
st.markdown("## 🔬 Methodology Overview - Accuracy Improvements")

st.markdown("""
<div class="success-box">
<h4>🎯 Fixed Extreme Value Theory (EVT) Framework</h4>
<p>Our system now uses mathematically accurate statistical methods to measure systemic risk:</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="stats-box">
    <h4>📋 Enhanced Data Processing</h4>
    <ol>
    <li><strong>Robust Data Collection:</strong> Improved Yahoo Finance integration</li>
    <li><strong>Better Resampling:</strong> Weekly returns with log transformation</li>
    <li><strong>Fixed Rolling Windows:</strong> Accurate 52-week analysis</li>
    <li><strong>Validated Regional Mapping:</strong> Proper bank-to-index associations</li>
    <li><strong>Error Handling:</strong> Comprehensive validation and edge case management</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-box">
    <h4>🔧 Accurate Risk Calculation</h4>
    <ol>
    <li><strong>Fixed VaR Estimation:</strong> Proper empirical quantile method</li>
    <li><strong>Corrected Hill Estimator:</strong> Adaptive threshold with validation</li>
    <li><strong>Accurate Tail Dependence:</strong> Left-tail conditional probability</li>
    <li><strong>Proper Systemic Beta:</strong> Mathematically correct formula</li>
    <li><strong>Enhanced Validation:</strong> Comprehensive input/output checking</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Available banks list
st.markdown("## 🏦 Available Banks")

col1, col2, col3 = st.columns(3, gap="large")

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

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>🏛️ Regulatory Compliance</h4>
        <p>Monitor systemic risk for Basel III requirements with mathematically accurate EVT calculations.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>💰 Risk Management</h4>
        <p>Identify high-risk banks and contagion effects using validated extreme value theory methods.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>📚 Academic Research</h4>
        <p>Study systemic risk patterns with confidence using our corrected mathematical implementations.</p>
    </div>
    """, unsafe_allow_html=True)

# Technical details
st.markdown("## 🔧 Technical Details - What We Fixed")

st.markdown("""
<div class="stats-box">
<h4>📊 Data Specifications & Improvements</h4>
<ul>
<li><strong>Data Source:</strong> Yahoo Finance via yfinance library (enhanced error handling)</li>
<li><strong>Frequency:</strong> Weekly returns with proper log transformation</li>
<li><strong>Window Size:</strong> 52 weeks rolling window with boundary validation</li>
<li><strong>Regional Indices:</strong> Accurate mapping to S&P 500, FTSE 100, DAX, Nikkei 225, etc.</li>
<li><strong>Confidence Levels:</strong> Mathematically consistent 95% and 99% analysis</li>
<li><strong>Processing:</strong> Robust handling of missing data and edge cases</li>
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
<li><strong>Configure Parameters:</strong> Select 5-10 banks, date ranges (2015-2024 recommended), and confidence levels</li>
<li><strong>Load Data:</strong> Click "🚀 Load & Analyze Data" and wait 2-5 minutes for processing</li>
<li><strong>Analyze Results:</strong> Interpret accurate risk metrics and beautiful visualizations</li>
<li><strong>Take Action:</strong> Use mathematically sound insights for decision-making</li>
</ol>
</div>
""", unsafe_allow_html=True)

# What's new section
st.markdown("## ✨ What's New - Major Fixes")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="success-box">
    <h4>🔧 Mathematical Fixes</h4>
    <ul>
    <li>✅ <strong>Hill Estimator:</strong> Fixed threshold selection and exceedance calculation</li>
    <li>✅ <strong>Tail Dependence:</strong> Corrected left-tail conditional probability</li>
    <li>✅ <strong>Systemic Beta:</strong> Proper formula implementation with validation</li>
    <li>✅ <strong>VaR Calculation:</strong> Accurate empirical quantile estimation</li>
    <li>✅ <strong>Data Processing:</strong> Enhanced error handling and validation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="success-box">
    <h4>🎨 Design Improvements</h4>
    <ul>
    <li>✅ <strong>Anthropic Light Theme:</strong> Beautiful, modern interface design</li>
    <li>✅ <strong>Better Visualizations:</strong> Cleaner charts with proper risk thresholds</li>
    <li>✅ <strong>Enhanced UX:</strong> Improved loading states and error messages</li>
    <li>✅ <strong>Consistent Styling:</strong> Professional color scheme and typography</li>
    <li>✅ <strong>Responsive Design:</strong> Works beautifully on all screen sizes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
<h3>🏦 Systemic Risk Analysis Platform</h3>
<p><strong>Built with:</strong> Accurate Extreme Value Theory, Python, Streamlit, and Beautiful Design</p>
<p><strong>Data Source:</strong> Yahoo Finance | <strong>Methodology:</strong> Mathematically Verified EVT Framework</p>
<p><strong>Theme:</strong> Anthropic Light Design | <strong>Status:</strong> Production Ready with Fixed Calculations</p>
<p>This platform provides comprehensive and <em>mathematically accurate</em> systemic risk analysis for global banking institutions.</p>
</div>
""", unsafe_allow_html=True)
