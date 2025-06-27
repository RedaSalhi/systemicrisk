import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    }
    .concept-box {
        background-color: #f8f9fa;
        border: 2px solid #2e7d32;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #e3f2fd;
        border: 1px solid #1976d2;
        padding: 1rem;
        border-radius: 0.25rem;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
    }
    .step-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

# EVT calculation functions
def calculate_var(returns, alpha=0.95):
    """Calculate Value at Risk"""
    return -np.percentile(returns, (1-alpha)*100)

def hill_estimator(losses, k=None):
    """Calculate Hill estimator for tail index"""
    if k is None:
        k = int(len(losses) * 0.05)  # Top 5% of losses
    
    losses_sorted = np.sort(losses)[::-1]  # Sort in descending order
    if k >= len(losses_sorted) or k < 2:
        return np.nan
    
    threshold = losses_sorted[k]
    excesses = losses_sorted[:k]
    
    if threshold <= 0:
        return np.nan
    
    hill_index = np.mean(np.log(excesses / threshold))
    return hill_index

def tail_dependence_coefficient(x_losses, y_losses, threshold_quantile=0.95):
    """Calculate tail dependence coefficient"""
    threshold_x = np.quantile(x_losses, threshold_quantile)
    threshold_y = np.quantile(y_losses, threshold_quantile)
    
    # Joint exceedances
    joint_exceedances = np.sum((x_losses > threshold_x) & (y_losses > threshold_y))
    x_exceedances = np.sum(x_losses > threshold_x)
    
    if x_exceedances == 0:
        return 0
    
    return joint_exceedances / x_exceedances

def systemic_beta(bank_losses, market_losses, alpha=0.95):
    """Calculate systemic beta"""
    var_bank = calculate_var(-bank_losses, alpha)
    var_market = calculate_var(-market_losses, alpha)
    
    hill_market = hill_estimator(market_losses)
    tail_dep = tail_dependence_coefficient(bank_losses, market_losses, alpha)
    
    if hill_market <= 0 or np.isnan(hill_market) or var_market <= 0:
        return np.nan
    
    return (tail_dep ** (1/hill_market)) * (var_bank / var_market)

# Load sample data
sample_data = generate_sample_data()

# Header
st.markdown('<h1 class="main-header">📚 Extreme Value Theory (EVT) Methodology</h1>', unsafe_allow_html=True)
st.markdown("**Interactive guide to understanding systemic risk measurement using Extreme Value Theory**")

# Navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Introduction", 
    "📊 Value-at-Risk", 
    "🔍 Hill Estimator", 
    "🔗 Tail Dependence", 
    "⚖️ Systemic Beta"
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
            x=x[tail_mask], y=t_dist[tail_mask],
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Extreme Loss Region',
            showlegend=False
        ))
        
        fig_dist.update_layout(
            title="Why EVT Matters: Fat Tails in Financial Data",
            xaxis_title="Standard Deviations from Mean",
            yaxis_title="Probability Density",
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("""
        **Key Insight**: The red area shows how fat-tailed distributions have much higher probability 
        of extreme events compared to normal distributions. This is why banks need EVT!
        """)
    
    with col2:
        st.markdown("""
        <div class="step-box">
        <h4>🗺️ EVT Framework Steps</h4>
        <ol>
        <li><strong>Value-at-Risk (VaR)</strong><br>Quantify potential losses</li>
        <li><strong>Hill Estimator</strong><br>Measure tail thickness</li>
        <li><strong>Tail Dependence</strong><br>Capture co-movement in extremes</li>
        <li><strong>Systemic Beta</strong><br>Combine into risk measure</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Real-World Impact</h4>
        <p>The 2008 financial crisis showed that banks' risk models (based on normal distributions) dramatically underestimated tail risks. EVT-based models would have provided better warnings.</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("Value-at-Risk (VaR)")
    
    st.markdown("""
    <div class="concept-box">
    <h3>📊 What is Value-at-Risk?</h3>
    <p><strong>Value-at-Risk (VaR)</strong> answers the question: "What is the maximum loss we can expect with X% confidence over a given time period?"</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Interactive VaR Calculator")
        
        # VaR parameters
        confidence_level = st.slider(
            "Confidence Level (%)", 
            min_value=90, max_value=99, value=95, step=1,
            help="Higher confidence = more conservative risk estimate"
        )
        
        time_horizon = st.selectbox(
            "Time Horizon",
            options=["1 Day", "1 Week", "1 Month"],
            help="Period over which VaR is calculated"
        )
        
        # Calculate VaR for sample data
        alpha = confidence_level / 100
        bank_var = calculate_var(sample_data['Bank_Returns'], alpha)
        market_var = calculate_var(sample_data['Market_Returns'], alpha)
        
        st.markdown(f"""
        <div class="formula-box">
        <strong>VaR Calculation:</strong><br>
        VaR({confidence_level}%) = -Percentile({100-confidence_level}%) of returns<br><br>
        <strong>Results:</strong><br>
        📊 Bank VaR ({confidence_level}%): {bank_var:.3f} ({bank_var*100:.1f}%)<br>
        📈 Market VaR ({confidence_level}%): {market_var:.3f} ({market_var*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)
        
        # VaR interpretation
        if bank_var > market_var:
            st.warning(f"⚠️ Bank is riskier than market (VaR {bank_var/market_var:.1f}x higher)")
        else:
            st.success(f"✅ Bank is less risky than market (VaR {market_var/bank_var:.1f}x lower)")
    
    with col2:
        st.subheader("VaR Visualization")
        
        # Create histogram with VaR line
        fig_var = go.Figure()
        
        # Histogram of returns
        fig_var.add_trace(go.Histogram(
            x=sample_data['Bank_Returns'],
            nbinsx=50,
            name='Bank Returns Distribution',
            opacity=0.7,
            histnorm='probability density'
        ))
        
        # VaR line
        fig_var.add_vline(
            x=-bank_var,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"VaR ({confidence_level}%): {bank_var:.3f}"
        )
        
        # Shade the tail area
        x_fill = np.linspace(-0.1, -bank_var, 100)
        y_fill = [0.1] * len(x_fill)  # Approximate density for visualization
        
        fig_var.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Tail Risk ({100-confidence_level}%)',
            showlegend=False
        ))
        
        fig_var.update_layout(
            title=f"Bank Returns Distribution with {confidence_level}% VaR",
            xaxis_title="Daily Returns",
            yaxis_title="Density",
            height=400
        )
        
        st.plotly_chart(fig_var, use_container_width=True)
    
    # VaR comparison across confidence levels
    st.subheader("VaR Sensitivity Analysis")
    
    confidence_levels = [90, 95, 97.5, 99, 99.5]
    var_values = [calculate_var(sample_data['Bank_Returns'], cl/100) for cl in confidence_levels]
    
    var_df = pd.DataFrame({
        'Confidence_Level': confidence_levels,
        'VaR': var_values,
        'VaR_Percentage': [v*100 for v in var_values]
    })
    
    fig_var_sens = px.bar(
        var_df,
        x='Confidence_Level',
        y='VaR_Percentage',
        title="VaR at Different Confidence Levels",
        labels={'VaR_Percentage': 'VaR (%)', 'Confidence_Level': 'Confidence Level (%)'}
    )
    
    st.plotly_chart(fig_var_sens, use_container_width=True)
    
    st.markdown("""
    **Key Observations:**
    - Higher confidence levels → Higher VaR (more conservative)
    - VaR increases rapidly at extreme confidence levels (99%+)
    - Regulators often use 99% VaR for stress testing
    """)

with tab3:
    st.header("Hill Estimator")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🔍 What is the Hill Estimator?</h3>
    <p>The <strong>Hill estimator</strong> measures how "thick" the tail of a distribution is. It tells us how likely extreme events are compared to a normal distribution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hill Estimator Calculation")
        
        # Parameters for Hill estimator
        tail_fraction = st.slider(
            "Tail Fraction (%)", 
            min_value=1, max_value=20, value=5, step=1,
            help="Percentage of most extreme observations to use"
        )
        
        k = int(len(sample_data) * tail_fraction / 100)
        
        # Calculate Hill estimator
        bank_hill = hill_estimator(sample_data['Bank_Losses'], k)
        market_hill = hill_estimator(sample_data['Market_Losses'], k)
        
        st.markdown(f"""
        <div class="formula-box">
        <strong>Hill Estimator Formula:</strong><br>
        ξ̂ = (1/k) × Σ log(Xi / u)<br>
        where u is the threshold (top {tail_fraction}% of losses)<br><br>
        <strong>Results:</strong><br>
        🏦 Bank Hill Index: {bank_hill:.3f}<br>
        📈 Market Hill Index: {market_hill:.3f}<br><br>
        <strong>Interpretation:</strong><br>
        • ξ > 0.5: Very heavy tails (high extreme risk)<br>
        • ξ ≈ 0.25-0.5: Moderate heavy tails<br>
        • ξ < 0.25: Light tails (low extreme risk)
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretation
        if bank_hill > 0.5:
            st.error("🔴 Very heavy tails - High extreme risk!")
        elif bank_hill > 0.25:
            st.warning("🟡 Moderate heavy tails - Elevated risk")
        else:
            st.success("🟢 Light tails - Lower extreme risk")
    
    with col2:
        st.subheader("Tail Behavior Visualization")
        
        # Show the extreme losses used in Hill calculation
        bank_losses_sorted = np.sort(sample_data['Bank_Losses'])[::-1]
        threshold = bank_losses_sorted[k-1]
        extreme_losses = bank_losses_sorted[:k]
        
        fig_hill = go.Figure()
        
        # All losses
        fig_hill.add_trace(go.Histogram(
            x=sample_data['Bank_Losses'],
            nbinsx=50,
            name='All Losses',
            opacity=0.6,
            histnorm='probability density'
        ))
        
        # Extreme losses
        fig_hill.add_trace(go.Histogram(
            x=extreme_losses,
            nbinsx=20,
            name=f'Top {tail_fraction}% Extreme Losses',
            opacity=0.8,
            histnorm='probability density'
        ))
        
        # Threshold line
        fig_hill.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold:.3f}"
        )
        
        fig_hill.update_layout(
            title="Distribution of Losses with Hill Estimator Threshold",
            xaxis_title="Losses",
            yaxis_title="Density",
            height=400
        )
        
        st.plotly_chart(fig_hill, use_container_width=True)
    
    # Hill estimator sensitivity
    st.subheader("Hill Estimator Sensitivity")
    
    fractions = range(1, 21)
    hill_values = []
    
    for frac in fractions:
        k_temp = int(len(sample_data) * frac / 100)
        if k_temp >= 2:
            hill_val = hill_estimator(sample_data['Bank_Losses'], k_temp)
            hill_values.append(hill_val)
        else:
            hill_values.append(np.nan)
    
    hill_df = pd.DataFrame({
        'Tail_Fraction': fractions,
        'Hill_Index': hill_values
    })
    
    fig_hill_sens = px.line(
        hill_df,
        x='Tail_Fraction',
        y='Hill_Index',
        title="Hill Estimator vs Tail Fraction",
        markers=True
    )
    fig_hill_sens.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Heavy Tail Threshold")
    fig_hill_sens.add_hline(y=0.25, line_dash="dash", line_color="orange", annotation_text="Moderate Tail Threshold")
    
    st.plotly_chart(fig_hill_sens, use_container_width=True)

with tab4:
    st.header("Tail Dependence")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🔗 What is Tail Dependence?</h3>
    <p><strong>Tail dependence</strong> measures how likely it is that two assets (e.g., a bank and the market) will both experience extreme losses simultaneously. This captures contagion risk.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tail Dependence Calculator")
        
        # Parameters
        tail_threshold = st.slider(
            "Tail Threshold (%)",
            min_value=90, max_value=99, value=95, step=1,
            help="Defines what constitutes an 'extreme' loss"
        )
        
        # Calculate tail dependence
        alpha_tail = tail_threshold / 100
        tail_dep = tail_dependence_coefficient(
            sample_data['Bank_Losses'], 
            sample_data['Market_Losses'], 
            alpha_tail
        )
        
        # Get the actual extreme events
        bank_threshold = np.quantile(sample_data['Bank_Losses'], alpha_tail)
        market_threshold = np.quantile(sample_data['Market_Losses'], alpha_tail)
        
        bank_extremes = sample_data['Bank_Losses'] > bank_threshold
        market_extremes = sample_data['Market_Losses'] > market_threshold
        joint_extremes = bank_extremes & market_extremes
        
        n_bank_extremes = bank_extremes.sum()
        n_joint_extremes = joint_extremes.sum()
        
        st.markdown(f"""
        <div class="formula-box">
        <strong>Tail Dependence Formula:</strong><br>
        τ = P(Y > F⁻¹(u) | X > F⁻¹(u)) as u → 1<br><br>
        <strong>Empirical Calculation:</strong><br>
        τ = (Joint extremes) / (Bank extremes)<br>
        τ = {n_joint_extremes} / {n_bank_extremes} = {tail_dep:.3f}<br><br>
        <strong>Interpretation:</strong><br>
        • τ = 1: Perfect tail dependence<br>
        • τ = 0: No tail dependence<br>
        • τ > 0.5: Strong contagion risk
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretation
        if tail_dep > 0.7:
            st.error("🔴 Very high contagion risk!")
        elif tail_dep > 0.5:
            st.warning("🟡 Moderate contagion risk")
        elif tail_dep > 0.3:
            st.info("🔵 Some tail dependence")
        else:
            st.success("🟢 Low contagion risk")
    
    with col2:
        st.subheader("Tail Dependence Visualization")
        
        # Scatter plot of losses with extreme quadrant highlighted
        fig_tail = go.Figure()
        
        # All observations
        fig_tail.add_trace(go.Scatter(
            x=sample_data['Bank_Losses'],
            y=sample_data['Market_Losses'],
            mode='markers',
            name='All Observations',
            marker=dict(color='lightblue', size=4, opacity=0.6)
        ))
        
        # Joint extreme events
        joint_extreme_data = sample_data[joint_extremes]
        if len(joint_extreme_data) > 0:
            fig_tail.add_trace(go.Scatter(
                x=joint_extreme_data['Bank_Losses'],
                y=joint_extreme_data['Market_Losses'],
                mode='markers',
                name='Joint Extremes',
                marker=dict(color='red', size=8)
            ))
        
        # Threshold lines
        fig_tail.add_vline(x=bank_threshold, line_dash="dash", line_color="red")
        fig_tail.add_hline(y=market_threshold, line_dash="dash", line_color="red")
        
        # Shade extreme quadrant
        max_loss = max(sample_data['Bank_Losses'].max(), sample_data['Market_Losses'].max())
        fig_tail.add_shape(
            type="rect",
            x0=bank_threshold, y0=market_threshold,
            x1=max_loss*1.1, y1=max_loss*1.1,
            fillcolor="rgba(255,0,0,0.1)",
            line_width=0
        )
        
        fig_tail.update_layout(
            title="Tail Dependence: Joint Extreme Events",
            xaxis_title="Bank Losses",
            yaxis_title="Market Losses",
            height=400
        )
        
        st.plotly_chart(fig_tail, use_container_width=True)
    
    # Tail dependence across thresholds
    st.subheader("Tail Dependence Sensitivity")
    
    thresholds = np.arange(90, 99.5, 0.5)
    tail_deps = []
    
    for thresh in thresholds:
        alpha = thresh / 100
        td = tail_dependence_coefficient(
            sample_data['Bank_Losses'], 
            sample_data['Market_Losses'], 
            alpha
        )
        tail_deps.append(td)
    
    tail_dep_df = pd.DataFrame({
        'Threshold': thresholds,
        'Tail_Dependence': tail_deps
    })
    
    fig_tail_sens = px.line(
        tail_dep_df,
        x='Threshold',
        y='Tail_Dependence',
        title="Tail Dependence vs Threshold",
        markers=True
    )
    fig_tail_sens.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Moderate Risk")
    fig_tail_sens.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
    
    st.plotly_chart(fig_tail_sens, use_container_width=True)

with tab5:
    st.header("Systemic Beta (βT)")
    
    st.markdown("""
    <div class="concept-box">
    <h3>⚖️ What is Systemic Beta?</h3>
    <p><strong>Systemic Beta (βT)</strong> combines all EVT components into a single measure of a bank's contribution to systemic risk. It generalizes the traditional market beta by focusing on tail behavior.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Systemic Beta Calculation")
        
        # Parameters for systemic beta
        confidence_systemic = st.slider(
            "VaR Confidence Level (%)",
            min_value=90, max_value=99, value=95, step=1
        )
        
        alpha_systemic = confidence_systemic / 100
        
        # Calculate components
        var_bank_sys = calculate_var(sample_data['Bank_Returns'], alpha_systemic)
        var_market_sys = calculate_var(sample_data['Market_Returns'], alpha_systemic)
        hill_market_sys = hill_estimator(sample_data['Market_Losses'])
        tail_dep_sys = tail_dependence_coefficient(
            sample_data['Bank_Losses'], 
            sample_data['Market_Losses'], 
            alpha_systemic
        )
        
        # Calculate systemic beta
        if hill_market_sys > 0 and not np.isnan(hill_market_sys) and var_market_sys > 0:
            beta_t = (tail_dep_sys ** (1/hill_market_sys)) * (var_bank_sys / var_market_sys)
        else:
            beta_t = np.nan
        
        st.markdown(f"""
        <div class="formula-box">
        <strong>Systemic Beta Formula:</strong><br>
        βT = τ^(1/ξ_market) × (VaR_bank / VaR_market)<br><br>
        <strong>Components:</strong><br>
        • Tail Dependence (τ): {tail_dep_sys:.3f}<br>
        • Market Hill Index (ξ): {hill_market_sys:.3f}<br>
        • Bank VaR: {var_bank_sys:.3f}<br>
        • Market VaR: {var_market_sys:.3f}<br><br>
        <strong>Systemic Beta (βT): {beta_t:.3f}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretation
        st.subheader("βT Interpretation")
        
        if np.isnan(beta_t):
            st.error("❌ Cannot calculate βT (insufficient data)")
        elif beta_t > 2.0:
            st.error("🔴 High Systemic Risk (βT > 2.0)")
            st.markdown("Bank poses significant threat to financial stability")
        elif beta_t > 1.0:
            st.warning("🟡 Above-Average Systemic Risk (βT > 1.0)")
            st.markdown("Bank contributes more risk than market average")
        elif beta_t > 0.5:
            st.info("🔵 Below-Average Systemic Risk")
            st.markdown("Bank contributes less risk than market average")
        else:
            st.success("🟢 Low Systemic Risk (βT < 0.5)")
            st.markdown("Bank poses minimal systemic threat")
    
    with col2:
        st.subheader("βT Components Breakdown")
        
        # Components pie chart
        if not np.isnan(beta_t) and beta_t > 0:
            # Decompose the beta calculation for visualization
            tail_component = tail_dep_sys ** (1/hill_market_sys) if hill_market_sys > 0 else 0
            var_ratio = var_bank_sys / var_market_sys if var_market_sys > 0 else 0
            
            components_df = pd.DataFrame({
                'Component': ['Tail Dependence Effect', 'VaR Ratio Effect'],
                'Value': [tail_component, var_ratio],
                'Description': [
                    f'τ^(1/ξ) = {tail_component:.3f}',
                    f'VaR ratio = {var_ratio:.3f}'
                ]
            })
            
            fig_components = px.bar(
                components_df,
                x='Component',
                y='Value',
                title="Systemic Beta Components",
                color='Component'
            )
            
            st.plotly_chart(fig_components, use_container_width=True)
        
        # Show the interpretation table
        st.subheader("βT Risk Categories")
        
        interpretation_df = pd.DataFrame({
            'βT Range': ['< 0.5', '0.5 - 1.0', '1.0 - 2.0', '> 2.0'],
            'Risk Level': ['Low', 'Moderate', 'High', 'Very High'],
            'Interpretation': [
                'Minimal systemic impact',
                'Below-average contribution',
                'Above-average risk',
                'Significant systemic threat'
            ],
            'Regulatory Action': [
                'Standard oversight',
                'Regular monitoring',
                'Enhanced supervision',
                'Immediate intervention'
            ]
        })
        
        st.dataframe(interpretation_df, use_container_width=True)
    
    # Historical evolution simulation
    st.subheader("Systemic Beta Evolution Over Time")
    
    # Simulate rolling window calculation
    window_size = 252  # 1 year of daily data
    dates_rolling = []
    beta_t_rolling = []
    
    for i in range(window_size, len(sample_data), 10):  # Every 10 days
        window_data = sample_data.iloc[i-window_size:i]
        
        # Calculate rolling systemic beta
        var_bank_roll = calculate_var(window_data['Bank_Returns'], 0.95)
        var_market_roll = calculate_var(window_data['Market_Returns'], 0.95)
        hill_market_roll = hill_estimator(window_data['Market_Losses'])
        tail_dep_roll = tail_dependence_coefficient(
            window_data['Bank_Losses'], 
            window_data['Market_Losses'], 
            0.95
        )
        
        if hill_market_roll > 0 and not np.isnan(hill_market_roll) and var_market_roll > 0:
            beta_t_roll = (tail_dep_roll ** (1/hill_market_roll)) * (var_bank_roll / var_market_roll)
        else:
            beta_t_roll = np.nan
        
        dates_rolling.append(window_data['Date'].iloc[-1])
        beta_t_rolling.append(beta_t_roll)
    
    rolling_df = pd.DataFrame({
        'Date': dates_rolling,
        'Systemic_Beta': beta_t_rolling
    })
    
    fig_evolution = go.Figure()
    fig_evolution.add_trace(go.Scatter(
        x=rolling_df['Date'],
        y=rolling_df['Systemic_Beta'],
        mode='lines+markers',
        name='Systemic Beta (βT)',
        line=dict(color='blue', width=2)
    ))
    
    # Add risk threshold lines
    fig_evolution.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="Risk Threshold")
    fig_evolution.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="High Risk")
    
    fig_evolution.update_layout(
        title="Rolling Systemic Beta Over Time (252-day window)",
        xaxis_title="Date",
        yaxis_title="Systemic Beta (βT)",
        height=400
    )
    
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    # Summary insights
    st.subheader("Key EVT Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 VaR**
        - Quantifies potential losses
        - Higher confidence = more conservative
        - Foundation for risk measurement
        """)
    
    with col2:
        st.markdown("""
        **🔍 Hill Estimator**
        - Measures tail thickness
        - Higher values = more extreme risk
        - Critical for understanding loss severity
        """)
    
    with col3:
        st.markdown("""
        **🔗 Tail Dependence**
        - Captures contagion risk
        - Shows co-movement in extremes
        - Key for systemic risk assessment
        """)

# Footer
st.divider()
st.markdown("""
**Educational Note**: This interactive tool demonstrates the EVT methodology used in systemic risk research. 
The sample data is simulated for educational purposes.

**Key Takeaways**:
- EVT focuses on extreme events that matter most for financial stability
- Systemic Beta (βT) combines multiple risk dimensions into a single measure
- Values above 1.0 indicate above-average systemic risk contribution
- Regular monitoring of these metrics can provide early warning of systemic stress

**Source**: Based on "Systemic Risk in Global Banking Institutions" - R. Salhi, Queen's University Belfast
""")
