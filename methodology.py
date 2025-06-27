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
    page_title="EVT Methodology Explainer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .concept-box {
        background-color: #f8f9fa;
        border: 2px solid #2e7d32;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .concept-box h3 {
        color: #2e7d32;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .concept-box p {
        color: #333;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .concept-box ul {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .concept-box li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    .formula-box {
        background-color: #f0f8ff;
        border: 2px solid #1976d2;
        padding: 1.2rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        margin: 0.8rem 0;
        color: #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .step-box {
        background-color: #fffbf0;
        border-left: 4px solid #ff9800;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .step-box h4 {
        color: #d68910;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .step-box p {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .step-box ol {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .step-box li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    .warning-box {
        background-color: #fff5f5;
        border: 2px solid #f56565;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-box h4 {
        color: #c53030;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .warning-box p {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .warning-box ul {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .warning-box li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    .success-box {
        background-color: #f0fff4;
        border: 2px solid #48bb78;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-box h4 {
        color: #38a169;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .success-box p {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .success-box ul {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .success-box li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.25rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
    }
    .nav-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
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
    
    /* Improve metric displays */
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Better sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("## 🧭 Navigation")
if st.sidebar.button("🏠 Home", key="nav_home"):
    st.switch_page("app.py")
if st.sidebar.button("📊 Dashboard", key="nav_dashboard"):
    st.switch_page("dashboard.py")
if st.sidebar.button("⚠️ Early Warning", key="nav_ml"):
    st.switch_page("machinelearning.py")

# Generate sample banking data
@st.cache_data
def generate_sample_data():
    """Generate sample banking return data for demonstrations"""
    np.random.seed(42)
    
    # Generate 1000 daily returns
    n_days = 1000
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Simulate bank returns with fat tails (t-distribution)
    bank_returns = stats.t.rvs(df=3, scale=0.02, size=n_days)
    
    # Simulate market index returns
    market_returns = stats.t.rvs(df=4, scale=0.015, size=n_days)
    
    # Add some correlation
    correlation = 0.6
    bank_returns = correlation * market_returns + np.sqrt(1 - correlation**2) * bank_returns
    
    # Convert to losses (negative returns)
    bank_losses = -bank_returns
    market_losses = -market_returns
    
    return pd.DataFrame({
        'Date': dates,
        'Bank_Returns': bank_returns,
        'Market_Returns': market_returns,
        'Bank_Losses': bank_losses,
        'Market_Losses': market_losses
    })

# EVT calculation functions (using the accurate methodology)
def calculate_var(returns, alpha=0.95):
    """Calculate Value at Risk"""
    return -np.percentile(returns, (1-alpha)*100)

def hill_estimator(x, threshold_quantile=0.99, min_excesses=5):
    """
    Estimate the tail index via Hill, but only at the
    largest threshold_quantile such that at least min_excesses are in the tail.
    """
    # sort unique quantiles from, say, 90% up to desired level
    candidate_q = np.linspace(0.90, threshold_quantile, 50)
    for q in reversed(candidate_q):
        p = 1 - q
        u = np.quantile(x, p)
        losses = -x[x < u]
        losses = losses[losses > 0]
        if len(losses) >= min_excesses:
            min_loss = losses.min()
            return np.mean(np.log(losses / min_loss))
    # if even 90% gives too few points, fall back or return NaN
    return np.nan

def tail_dependence(x, y, u=0.95):
    """Calculate tail dependence coefficient"""
    qx, qy = np.quantile(x, u), np.quantile(y, u)
    mask = x < qx   # left‐tail dependence (for losses)
    return np.sum(y[mask] < qy) / np.sum(mask) if np.sum(mask)>0 else np.nan

def systemic_beta(x, y, u=0.95):
    """Calculate systemic beta"""
    VaR_x = calculate_var(x, alpha=u)
    VaR_y = calculate_var(y, alpha=u)
    xi_y  = hill_estimator(y, threshold_quantile=u)
    tau   = tail_dependence(x, y, u=u)
    if xi_y is None or xi_y==0 or np.isnan(tau):
        return np.nan
    return (tau ** (1.0/xi_y)) * (VaR_x / VaR_y)

# Load sample data
sample_data = generate_sample_data()

# Header
st.markdown('<h1 class="main-header">📚 Extreme Value Theory (EVT) Methodology</h1>', unsafe_allow_html=True)
st.markdown("**Interactive guide to understanding systemic risk measurement using Extreme Value Theory**")

# Navigation tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Introduction", 
    "📊 Value-at-Risk", 
    "🔍 Hill Estimator", 
    "🔗 Tail Dependence", 
    "⚖️ Systemic Beta",
    "🏦 Real Data Example"
])

with tab1:
    st.header("What is Extreme Value Theory?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>🎯 Core Concept</h3>
        <p><strong>Extreme Value Theory (EVT)</strong> is a statistical framework that focuses on the behavior of extreme events - the "tail" of probability distributions. In banking, we care about extreme losses that could cause systemic crises.</p>
        
        <p><strong>Why EVT for Banking Risk?</strong></p>
        <ul>
        <li>🏦 Bank failures are rare but catastrophic events</li>
        <li>📈 Normal distributions underestimate tail risks</li>
        <li>🔗 EVT captures interconnectedness during stress</li>
        <li>⚡ Focus on what matters: extreme losses</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Visual comparison: Normal vs Fat-tailed distribution
        st.subheader("Normal vs Fat-Tailed Distributions")
        
        x = np.linspace(-6, 6, 1000)
        normal_dist = stats.norm.pdf(x, 0, 1)
        t_dist = stats.t.pdf(x, df=3)  # Fat-tailed
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(
            x=x, y=normal_dist, 
            name='Normal Distribution',
            line=dict(color='blue', width=2)
        ))
        fig_dist.add_trace(go.Scatter(
            x=x, y=t_dist,
            name='Fat-Tailed Distribution (Banking Returns)',
            line=dict(color='red', width=2)
        ))
        
        # Highlight tail areas
        tail_mask = x <= -2
        fig_dist.add_trace(go.Scatter(
            x=x[tail_mask], y=normal_dist[tail_mask],
            fill='tonexty', fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='blue', width=0),
            showlegend=False
        ))
        
        fig_dist.add_trace(go.Scatter(
            x=x[tail_mask], y=t_dist[tail_mask],
            fill='tonexty', fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='red', width=0),
            showlegend=False
        ))
        
        fig_dist.update_layout(
            title="Distribution Comparison",
            xaxis_title="Returns",
            yaxis_title="Probability Density",
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Key Insight</h4>
        <p>The fat-tailed distribution shows much higher probability of extreme losses compared to the normal distribution. 
        This is why traditional risk models fail during financial crises - they underestimate the likelihood of extreme events.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
        <h4>✅ EVT Advantages</h4>
        <ul>
        <li>Accurate tail modeling</li>
        <li>Captures extreme dependencies</li>
        <li>Robust to distribution assumptions</li>
        <li>Focuses on systemic risk</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-box">
        <h4>📋 Our Methodology</h4>
        <ol>
        <li>Weekly return calculation</li>
        <li>Rolling window analysis</li>
        <li>Hill estimator for tail index</li>
        <li>Tail dependence measurement</li>
        <li>Systemic beta computation</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("Value-at-Risk (VaR)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>📊 What is VaR?</h3>
        <p><strong>Value-at-Risk (VaR)</strong> measures the maximum potential loss over a given time horizon at a specified confidence level.</p>
        
        <div class="formula-box">
        VaR<sub>α</sub> = -F<sup>-1</sup>(1-α)
        </div>
        
        <p>Where:</p>
        <ul>
        <li>α = confidence level (e.g., 95% or 99%)</li>
        <li>F<sup>-1</sup> = inverse cumulative distribution function</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive VaR calculation
        st.subheader("Interactive VaR Calculator")
        
        confidence_level = st.slider(
            "Confidence Level (%)", 
            min_value=90, max_value=99, value=95, step=1
        )
        
        # Calculate VaR for sample data
        var_value = calculate_var(sample_data['Bank_Returns'], alpha=confidence_level/100)
        
        st.metric(
            f"VaR ({confidence_level}%)", 
            f"{var_value:.4f}",
            help="Maximum expected loss at the specified confidence level"
        )
    
    with col2:
        # VaR visualization
        st.subheader("VaR Visualization")
        
        returns = sample_data['Bank_Returns']
        var_95 = calculate_var(returns, alpha=0.95)
        var_99 = calculate_var(returns, alpha=0.99)
        
        fig_var = go.Figure()
        
        # Histogram of returns
        fig_var.add_trace(go.Histogram(
            x=returns, 
            nbinsx=50,
            name='Return Distribution',
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        # Add VaR lines
        fig_var.add_vline(
            x=-var_95, 
            line_dash="dash", 
            line_color="orange",
            annotation_text=f"VaR 95%: {-var_95:.4f}"
        )
        
        fig_var.add_vline(
            x=-var_99, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"VaR 99%: {-var_99:.4f}"
        )
        
        fig_var.update_layout(
            title="Return Distribution with VaR Levels",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_var, use_container_width=True)
        
        st.markdown("""
        <div class="step-box">
        <h4>🔍 Interpretation</h4>
        <p>• <strong>VaR 95%</strong>: 95% chance that losses won't exceed this level</p>
        <p>• <strong>VaR 99%</strong>: 99% chance that losses won't exceed this level</p>
        <p>• Higher confidence = higher VaR = more conservative estimate</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.header("Hill Estimator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>🔍 What is the Hill Estimator?</h3>
        <p>The <strong>Hill Estimator</strong> estimates the tail index (ξ) of heavy-tailed distributions, which characterizes how "heavy" the tails are.</p>
        
        <div class="formula-box">
        ξ̂ = (1/k) Σ<sub>i=1</sub><sup>k</sup> log(X<sub>(i)</sub>/u)
        </div>
        
        <p>Where:</p>
        <ul>
        <li>k = number of exceedances above threshold u</li>
        <li>X<sub>(i)</sub> = i-th largest observation</li>
        <li>u = threshold value</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Hill estimator
        st.subheader("Interactive Hill Estimator")
        
        threshold_q = st.slider(
            "Threshold Quantile", 
            min_value=0.90, max_value=0.99, value=0.95, step=0.01
        )
        
        hill_value = hill_estimator(sample_data['Bank_Returns'], threshold_quantile=threshold_q)
        
        if not np.isnan(hill_value):
            st.metric(
                f"Hill Estimator (q={threshold_q})", 
                f"{hill_value:.4f}",
                help="Tail index estimate - higher values indicate heavier tails"
            )
            
            # Interpretation
            if hill_value > 0.5:
                interpretation = "Very heavy tails (high extreme risk)"
                color = "red"
            elif hill_value > 0.3:
                interpretation = "Heavy tails (moderate extreme risk)"
                color = "orange"
            else:
                interpretation = "Moderate tails (lower extreme risk)"
                color = "green"
            
            st.markdown(f"""
            <div class="step-box">
            <h4 style="color: {color}">📊 Interpretation</h4>
            <p>{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient data for Hill estimator at this threshold")
    
    with col2:
        # Hill estimator visualization
        st.subheader("Hill Estimator Analysis")
        
        # Calculate Hill estimator for different thresholds
        thresholds = np.linspace(0.90, 0.99, 20)
        hill_values = []
        
        for q in thresholds:
            hill_val = hill_estimator(sample_data['Bank_Returns'], threshold_quantile=q)
            hill_values.append(hill_val)
        
        fig_hill = go.Figure()
        fig_hill.add_trace(go.Scatter(
            x=thresholds, 
            y=hill_values,
            mode='lines+markers',
            name='Hill Estimator',
            line=dict(color='purple', width=2)
        ))
        
        fig_hill.update_layout(
            title="Hill Estimator vs Threshold",
            xaxis_title="Threshold Quantile",
            yaxis_title="Hill Estimator (ξ)",
            height=400
        )
        
        st.plotly_chart(fig_hill, use_container_width=True)
        
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Important Notes</h4>
        <ul>
        <li>Hill estimator is sensitive to threshold choice</li>
        <li>Too low threshold: bias from non-tail observations</li>
        <li>Too high threshold: high variance from few observations</li>
        <li>We use adaptive threshold selection in our implementation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.header("Tail Dependence")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>🔗 What is Tail Dependence?</h3>
        <p><strong>Tail Dependence</strong> measures the probability that one variable exceeds its threshold given that another variable exceeds its threshold.</p>
        
        <div class="formula-box">
        τ = P(Y > F<sub>Y</sub><sup>-1</sup>(u) | X > F<sub>X</sub><sup>-1</sup>(u))
        </div>
        
        <p>Where:</p>
        <ul>
        <li>τ = tail dependence coefficient</li>
        <li>u = threshold quantile (e.g., 0.95)</li>
        <li>F<sup>-1</sup> = inverse cumulative distribution function</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive tail dependence
        st.subheader("Interactive Tail Dependence")
        
        threshold_u = st.slider(
            "Threshold (u)", 
            min_value=0.90, max_value=0.99, value=0.95, step=0.01
        )
        
        tail_dep = tail_dependence(
            sample_data['Bank_Returns'], 
            sample_data['Market_Returns'], 
            u=threshold_u
        )
        
        if not np.isnan(tail_dep):
            st.metric(
                f"Tail Dependence (u={threshold_u})", 
                f"{tail_dep:.4f}",
                help="Probability of joint extreme events"
            )
            
            # Interpretation
            if tail_dep > 0.7:
                interpretation = "Very high systemic risk"
                color = "red"
            elif tail_dep > 0.5:
                interpretation = "High systemic risk"
                color = "orange"
            elif tail_dep > 0.3:
                interpretation = "Moderate systemic risk"
                color = "yellow"
            else:
                interpretation = "Low systemic risk"
                color = "green"
            
            st.markdown(f"""
            <div class="step-box">
            <h4 style="color: {color}">📊 Interpretation</h4>
            <p>{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Tail dependence visualization
        st.subheader("Tail Dependence Analysis")
        
        # Scatter plot with tail dependence regions
        fig_tail = go.Figure()
        
        # Main scatter plot
        fig_tail.add_trace(go.Scatter(
            x=sample_data['Bank_Returns'],
            y=sample_data['Market_Returns'],
            mode='markers',
            marker=dict(
                size=4,
                color='lightblue',
                opacity=0.6
            ),
            name='All Observations'
        ))
        
        # Highlight tail region
        threshold_bank = np.quantile(sample_data['Bank_Returns'], 0.95)
        threshold_market = np.quantile(sample_data['Market_Returns'], 0.95)
        
        tail_mask = (sample_data['Bank_Returns'] < threshold_bank) & (sample_data['Market_Returns'] < threshold_market)
        
        fig_tail.add_trace(go.Scatter(
            x=sample_data['Bank_Returns'][tail_mask],
            y=sample_data['Market_Returns'][tail_mask],
            mode='markers',
            marker=dict(
                size=6,
                color='red',
                opacity=0.8
            ),
            name='Tail Region'
        ))
        
        # Add threshold lines
        fig_tail.add_hline(y=threshold_market, line_dash="dash", line_color="red")
        fig_tail.add_vline(x=threshold_bank, line_dash="dash", line_color="red")
        
        fig_tail.update_layout(
            title="Bank vs Market Returns with Tail Dependence",
            xaxis_title="Bank Returns",
            yaxis_title="Market Returns",
            height=400
        )
        
        st.plotly_chart(fig_tail, use_container_width=True)
        
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Systemic Risk Implications</h4>
        <ul>
        <li>High tail dependence = high contagion risk</li>
        <li>Banks fail together during crises</li>
        <li>Traditional correlation underestimates this risk</li>
        <li>Critical for systemic risk assessment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

with tab5:
    st.header("Systemic Beta")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>⚖️ What is Systemic Beta?</h3>
        <p><strong>Systemic Beta (βT)</strong> measures a bank's contribution to systemic risk, combining individual risk (VaR) with systemic interconnectedness (tail dependence).</p>
        
        <div class="formula-box">
        βT = (τ<sup>1/ξ</sup>) × (VaR<sub>bank</sub> / VaR<sub>market</sub>)
        </div>
        
        <p>Where:</p>
        <ul>
        <li>τ = tail dependence coefficient</li>
        <li>ξ = Hill estimator (tail index)</li>
        <li>VaR = Value-at-Risk</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive systemic beta calculation
        st.subheader("Interactive Systemic Beta Calculator")
        
        confidence_beta = st.slider(
            "Confidence Level for Beta (%)", 
            min_value=90, max_value=99, value=95, step=1
        )
        
        beta_value = systemic_beta(
            sample_data['Bank_Returns'], 
            sample_data['Market_Returns'], 
            u=confidence_beta/100
        )
        
        if not np.isnan(beta_value):
            st.metric(
                f"Systemic Beta ({confidence_beta}%)", 
                f"{beta_value:.4f}",
                help="Bank's contribution to systemic risk"
            )
            
            # Risk assessment
            if beta_value > 2.0:
                risk_level = "🔴 High Risk"
                risk_desc = "Significant systemic risk contribution"
                color = "red"
            elif beta_value > 1.5:
                risk_level = "🟡 Medium Risk"
                risk_desc = "Moderate systemic risk contribution"
                color = "orange"
            else:
                risk_level = "🟢 Low Risk"
                risk_desc = "Low systemic risk contribution"
                color = "green"
            
            st.markdown(f"""
            <div class="step-box">
            <h4 style="color: {color}">{risk_level}</h4>
            <p>{risk_desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Systemic beta components breakdown
        st.subheader("Systemic Beta Components")
        
        # Calculate components
        var_bank = calculate_var(sample_data['Bank_Returns'], alpha=0.95)
        var_market = calculate_var(sample_data['Market_Returns'], alpha=0.95)
        hill_market = hill_estimator(sample_data['Market_Returns'], threshold_quantile=0.95)
        tail_dep = tail_dependence(sample_data['Bank_Returns'], sample_data['Market_Returns'], u=0.95)
        
        # Create component breakdown
        components_data = {
            'Component': ['VaR Bank', 'VaR Market', 'Hill Estimator', 'Tail Dependence', 'Systemic Beta'],
            'Value': [var_bank, var_market, hill_market, tail_dep, beta_value if not np.isnan(beta_value) else 0]
        }
        
        fig_components = go.Figure(data=[
            go.Bar(
                x=components_data['Component'],
                y=components_data['Value'],
                marker_color=['blue', 'green', 'purple', 'orange', 'red']
            )
        ])
        
        fig_components.update_layout(
            title="Systemic Beta Components",
            xaxis_title="Component",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig_components, use_container_width=True)
        
        st.markdown("""
        <div class="success-box">
        <h4>✅ Key Advantages</h4>
        <ul>
        <li>Combines individual and systemic risk</li>
        <li>Accounts for extreme event dependencies</li>
        <li>Provides interpretable risk measure</li>
        <li>Enables bank ranking by systemic importance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

with tab6:
    st.header("Real Data Example")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🏦 Real Banking Data Analysis</h3>
    <p>Let's see how our methodology works with real banking data from the 28 global banks in our system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Bank selection for demonstration
    processor = BankingDataProcessor()
    banks_by_region = processor.get_banks_by_region()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Banks for Analysis")
        
        # Quick selection of a few banks
        demo_banks = st.multiselect(
            "Choose banks to analyze:",
            options=processor.get_available_banks(),
            default=['JPMorgan Chase', 'HSBC', 'Deutsche Bank'],
            max_selections=5
        )
        
        if st.button("Analyze Selected Banks", type="primary"):
            if demo_banks:
                with st.spinner("Downloading and analyzing data..."):
                    try:
                        # Process data for selected banks
                        demo_processor = process_banking_data(
                            demo_banks, 
                            start_date='2020-01-01', 
                            end_date='2024-12-31'
                        )
                        
                        # Get latest metrics
                        latest_metrics = demo_processor.get_latest_metrics(0.95)
                        
                        st.success("Analysis completed!")
                        
                        # Display results
                        st.subheader("Latest Systemic Risk Metrics")
                        
                        # Create a nice table
                        display_data = latest_metrics[['Bank', 'Region', 'Beta_T', 'VaR_95', 'Tau_95']].copy()
                        display_data = display_data.round(4)
                        
                        # Color code by risk level
                        def color_risk(val):
                            if val > 2.0:
                                return 'background-color: #ffebee'
                            elif val > 1.5:
                                return 'background-color: #fff3e0'
                            else:
                                return 'background-color: #e8f5e8'
                        
                        st.dataframe(
                            display_data.style.applymap(color_risk, subset=['Beta_T']),
                            use_container_width=True
                        )
                        
                        # Risk summary
                        high_risk = len(latest_metrics[latest_metrics['Beta_T'] > 2.0])
                        medium_risk = len(latest_metrics[
                            (latest_metrics['Beta_T'] > 1.5) & 
                            (latest_metrics['Beta_T'] <= 2.0)
                        ])
                        low_risk = len(latest_metrics[latest_metrics['Beta_T'] <= 1.5])
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("🔴 High Risk", high_risk)
                        with col_b:
                            st.metric("🟡 Medium Risk", medium_risk)
                        with col_c:
                            st.metric("🟢 Low Risk", low_risk)
                        
                    except Exception as e:
                        st.error(f"Error analyzing data: {str(e)}")
                        st.info("This might be due to network issues or data availability.")
            else:
                st.warning("Please select at least one bank.")
    
    with col2:
        st.subheader("Available Banks by Region")
        
        for region, banks in banks_by_region.items():
            with st.expander(f"{region} ({len(banks)} banks)"):
                for bank in banks:
                    st.write(f"• {bank}")
        
        st.markdown("""
        <div class="step-box">
        <h4>📋 How to Use Real Data</h4>
        <ol>
        <li>Select banks from the dropdown</li>
        <li>Click "Analyze Selected Banks"</li>
        <li>View systemic risk metrics</li>
        <li>Interpret risk levels</li>
        <li>Compare across banks</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Important Notes</h4>
    <ul>
    <li>Real data analysis requires internet connection</li>
    <li>Data download may take several minutes</li>
    <li>Some banks may have limited historical data</li>
    <li>Results are based on weekly rolling windows</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<p><strong>Extreme Value Theory Methodology</strong> | Built with Streamlit and Python</p>
<p>This methodology provides a robust framework for measuring systemic risk in banking institutions.</p>
</div>
""", unsafe_allow_html=True)
