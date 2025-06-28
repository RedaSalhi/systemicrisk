import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import genpareto, norm
import sys
import os

# Remove the problematic import - methodology page is self-contained
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Methodology",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better equation display
st.markdown("""
<style>
    .equation-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Computer Modern', serif;
    }
    .methodology-section {
        background-color: #FAFAFA;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    .code-example {
        background-color: #1E293B;
        color: #F1F5F9;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📚 Methodology & Mathematical Framework")
    st.markdown("**Theoretical foundation of the EVT-based systemic risk measurement**")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Overview", 
        "📊 EVT Fundamentals", 
        "🔗 Systemic Beta", 
        "💻 Implementation", 
        "📖 References"
    ])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_evt_fundamentals()
    
    with tab3:
        show_systemic_beta()
    
    with tab4:
        show_implementation()
    
    with tab5:
        show_references()

def show_overview():
    """Overview of the methodology"""
    
    st.markdown("## 🎯 Framework Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This application implements a comprehensive systemic risk measurement framework 
        based on **Extreme Value Theory (EVT)** for analyzing Global Systemically Important Banks (G-SIBs).
        
        ### 🔬 Key Components
        
        1. **Value-at-Risk (VaR)** estimation using empirical quantiles
        2. **Hill estimator** for tail index calculation
        3. **Tail dependence** analysis between banks and regional indices
        4. **Systemic Beta (βT)** as proposed by van Oordt & Zhou (2018)
        5. **Spillover-aware stress testing** incorporating systemic linkages
        6. **Machine Learning early-warning system** for crisis prediction
        
        ### 🎯 Research Objectives
        
        - Measure individual bank tail risk using rolling VaR
        - Quantify systemic importance through tail dependence
        - Calculate systemic beta incorporating extreme value properties
        - Develop stress testing scenarios accounting for spillover effects
        - Build predictive models for early crisis detection
        """)
    
    with col2:
        st.markdown('<div class="methodology-section">', unsafe_allow_html=True)
        st.markdown("""
        ### 📊 Data Specification
        
        **Universe**: 28 G-SIBs
        
        **Period**: 2011-2024
        
        **Frequency**: Weekly
        
        **Window**: 52 weeks rolling
        
        **Indices**: Regional market indices for systemic beta calculation
        
        **Crisis Periods**: 
        - Eurozone Crisis (2011-2012)
        - China Correction (2015-2016)  
        - COVID-19 (2020)
        - Ukraine War (2022)
        - Banking Stress 2023
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Workflow diagram
    st.markdown("### 🔄 Analytical Workflow")
    
    workflow_steps = [
        "1. **Data Collection**: Download weekly closing prices for 28 G-SIBs and regional indices",
        "2. **Return Calculation**: Compute weekly log-returns: $r_t = \\ln(P_t/P_{t-1})$",
        "3. **Rolling Windows**: Apply 52-week rolling windows for all calculations",
        "4. **VaR Estimation**: Calculate empirical quantiles at 95% and 99% confidence levels",
        "5. **Hill Estimation**: Compute tail index using Hill estimator with dynamic thresholds",
        "6. **Tail Dependence**: Measure left-tail dependence between banks and regional indices",
        "7. **Systemic Beta**: Calculate βT using van Oordt & Zhou (2018) formula",
        "8. **Stress Testing**: Perform spillover-aware scenario analysis",
        "9. **ML Prediction**: Engineer features and train early-warning models"
    ]
    
    for step in workflow_steps:
        st.markdown(step)

def show_evt_fundamentals():
    """EVT fundamentals and equations"""
    
    st.markdown("## 📊 Extreme Value Theory Fundamentals")
    
    # Value-at-Risk
    st.markdown("### 1. Value-at-Risk (VaR)")
    
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.markdown("""
    **Definition**: The Value-at-Risk at confidence level α is defined as:
    
    $$VaR_\\alpha = -F^{-1}(1-\\alpha)$$
    
    where $F^{-1}$ is the inverse of the return distribution function.
    
    **Empirical Implementation**:
    $$VaR_\\alpha = -\\text{quantile}(r_t, 1-\\alpha)$$
    
    For α = 0.95: $VaR_{0.95} = -\\text{quantile}(r_t, 0.05)$
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Hill Estimator
    st.markdown("### 2. Hill Estimator")
    
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.markdown("""
    **Purpose**: Estimate the tail index ξ for the extreme value distribution.
    
    **Formula**: For excesses over threshold u:
    $$\\hat{\\xi}_H = \\frac{1}{n} \\sum_{i=1}^{n} \\ln\\left(\\frac{X_i}{u}\\right)$$
    
    where $X_i$ are the n largest losses exceeding threshold u.
    
    **Dynamic Threshold Selection**: 
    - Start from high quantile (e.g., 99%)
    - Reduce until minimum number of excesses is reached
    - Ensures statistical reliability while capturing tail behavior
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tail Dependence
    st.markdown("### 3. Tail Dependence")
    
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.markdown("""
    **Definition**: Measures the probability of joint extreme events.
    
    **Left-tail dependence coefficient**:
    $$\\tau_L = \\lim_{u \\to 0^+} P(Y \\leq F_Y^{-1}(u) | X \\leq F_X^{-1}(u))$$
    
    **Empirical Estimation**:
    $$\\hat{\\tau}_L = \\frac{\\sum_{i=1}^T \\mathbf{1}_{\\{X_i \\leq q_X, Y_i \\leq q_Y\\}}}{\\sum_{i=1}^T \\mathbf{1}_{\\{X_i \\leq q_X\\}}}$$
    
    where $q_X$ and $q_Y$ are the u-quantiles of X and Y respectively.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive example
    st.markdown("### 📈 Interactive EVT Example")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Parameter Controls")
        n_samples = st.slider("Sample Size", 500, 2000, 1000)
        tail_param = st.slider("Tail Parameter (ξ)", 0.1, 0.5, 0.2, 0.05)
        threshold_pct = st.slider("Threshold (%)", 90, 99, 95)
    
    with col2:
        # Generate sample data
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, n_samples)
        
        # Add some extreme events
        extreme_prob = 0.05
        n_extreme = int(n_samples * extreme_prob)
        extreme_indices = np.random.choice(n_samples, n_extreme, replace=False)
        returns[extreme_indices] += np.random.exponential(0.05, n_extreme) * np.random.choice([-1, 1], n_extreme)
        
        # Calculate VaR
        var_95 = -np.percentile(returns, 5)
        var_99 = -np.percentile(returns, 1)
        
        # Plot distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(returns, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax.axvline(-var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95% = {var_95:.4f}')
        ax.axvline(-var_99, color='red', linestyle='--', linewidth=2, label=f'VaR 99% = {var_99:.4f}')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.set_title('Return Distribution with VaR Estimates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def show_systemic_beta():
    """Systemic beta methodology"""
    
    st.markdown("## 🔗 Systemic Beta (βT) Framework")
    
    # Main formula
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.markdown("""
    ## Van Oordt & Zhou (2018) Systemic Beta
    
    The systemic beta captures a bank's contribution to systemic risk by combining:
    - **Tail dependence** with the financial system
    - **Relative tail thickness** compared to the system
    - **Individual tail risk** magnitude
    
    $$\\beta_T = \\tau^{1/\\xi_y} \\cdot \\frac{VaR_x}{VaR_y}$$
    
    **Where**:
    - $\\tau$: Tail dependence coefficient between bank and system
    - $\\xi_y$: Hill estimator (tail index) of the system
    - $VaR_x$: Value-at-Risk of the individual bank
    - $VaR_y$: Value-at-Risk of the system (regional index)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Component explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="methodology-section">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 Component Interpretation
        
        **1. Tail Dependence (τ)**
        - Measures correlation in extreme downturns
        - τ = 0: No tail dependence
        - τ = 1: Perfect tail dependence
        - Higher τ → more systemic importance
        
        **2. Tail Index Adjustment (1/ξ_y)**
        - Accounts for system's tail thickness
        - Higher ξ → thicker tails → more extreme events
        - Normalizes for different tail characteristics
        
        **3. Relative VaR Ratio**
        - Compares individual vs. system tail risk
        - Captures relative magnitude of extreme losses
        - Higher ratio → higher individual contribution
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="methodology-section">', unsafe_allow_html=True)
        st.markdown("""
        ### 📊 Interpretation Guidelines
        
        **βT < 1.0**: Low systemic importance
        - Bank contributes less than proportionally to systemic risk
        - Relatively isolated from system-wide events
        
        **1.0 ≤ βT < 2.0**: Moderate systemic importance  
        - Bank moves in line with system during stress
        - Standard level of interconnectedness
        
        **βT ≥ 2.0**: High systemic importance
        - Bank amplifies systemic risk
        - Potential source of contagion
        - Requires enhanced supervision
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Mathematical properties
    st.markdown("### 🔢 Mathematical Properties")
    
    properties = [
        "**Asymptotic Consistency**: βT converges to true systemic importance as sample size increases",
        "**Scale Invariance**: Results unchanged by scaling of return series",
        "**Tail Focus**: Emphasizes extreme events rather than normal market conditions",
        "**Relative Measure**: Compares bank risk to system benchmark",
        "**Non-Negativity**: βT ≥ 0 by construction",
        "**Crisis Sensitivity**: Higher values during financial stress periods"
    ]
    
    for prop in properties:
        st.markdown(f"- {prop}")
    
    # Interactive systemic beta calculator
    st.markdown("### 🧮 Interactive Systemic Beta Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Input Parameters")
        tau_input = st.slider("Tail Dependence (τ)", 0.1, 0.9, 0.5, 0.05)
        xi_input = st.slider("System Tail Index (ξ)", 0.1, 0.8, 0.3, 0.05)
        var_bank = st.slider("Bank VaR", 0.02, 0.15, 0.08, 0.01)
        var_system = st.slider("System VaR", 0.02, 0.10, 0.05, 0.01)
    
    with col2:
        # Calculate systemic beta
        beta_t = (tau_input ** (1/xi_input)) * (var_bank / var_system)
        
        st.markdown("#### Calculated Systemic Beta")
        st.metric("βT", f"{beta_t:.3f}")
        
        # Risk classification
        if beta_t < 1.0:
            risk_level = "🟢 Low Risk"
            risk_color = "green"
        elif beta_t < 2.0:
            risk_level = "🟡 Moderate Risk"
            risk_color = "orange"
        else:
            risk_level = "🔴 High Risk"
            risk_color = "red"
        
        st.markdown(f"**Risk Level**: {risk_level}")
        
        st.markdown("#### Component Contributions")
        tail_component = tau_input ** (1/xi_input)
        var_component = var_bank / var_system
        
        st.metric("Tail Component", f"{tail_component:.3f}")
        st.metric("VaR Ratio", f"{var_component:.3f}")
    
    with col3:
        # Sensitivity analysis
        st.markdown("#### Sensitivity Analysis")
        
        tau_range = np.linspace(0.1, 0.9, 20)
        beta_sensitivity = [(tau ** (1/xi_input)) * (var_bank / var_system) for tau in tau_range]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(tau_range, beta_sensitivity, linewidth=2, color='blue')
        ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk')
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='High Risk')
        ax.axvline(x=tau_input, color='green', linestyle=':', alpha=0.7, label='Current τ')
        ax.set_xlabel('Tail Dependence (τ)')
        ax.set_ylabel('Systemic Beta (βT)')
        ax.set_title('Sensitivity to Tail Dependence')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

def show_implementation():
    """Implementation details and code examples"""
    
    st.markdown("## 💻 Implementation Details")
    
    # Data processing pipeline
    st.markdown("### 1. Data Processing Pipeline")
    
    st.markdown('<div class="code-example">', unsafe_allow_html=True)
    st.code("""
# Data download and preprocessing
import yfinance as yf
import pandas as pd
import numpy as np

def download_data(start_date='2011-01-01', end_date='2024-12-31'):
    # Download bank stock prices
    bank_tickers = ['JPM', 'BAC', 'C', 'WFC', 'GS', ...]  # 28 G-SIBs
    raw_banks = yf.download(bank_tickers, start=start_date, end=end_date)['Close']
    
    # Download regional indices
    index_tickers = ['^GSPC', '^FTSE', '^N225', ...]
    raw_indices = yf.download(index_tickers, start=start_date, end=end_date)['Close']
    
    return raw_banks, raw_indices

def prepare_returns(price_data):
    # Convert to weekly frequency (Friday close)
    weekly_prices = price_data.resample('W-FRI').last().ffill()
    
    # Calculate log returns
    returns = np.log(weekly_prices / weekly_prices.shift(1)).dropna()
    
    return returns
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # EVT functions
    st.markdown("### 2. EVT Function Implementations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["VaR", "Hill Estimator", "Tail Dependence", "Systemic Beta"])
    
    with tab1:
        st.markdown("#### Value-at-Risk Implementation")
        st.code("""
def rolling_var(returns, alpha=0.95):
    \"\"\"
    Calculate Value-at-Risk using empirical quantiles
    
    Parameters:
    - returns: array of return observations
    - alpha: confidence level (0.95 for 95% VaR)
    
    Returns:
    - VaR: Value-at-Risk (positive number)
    \"\"\"
    return -np.percentile(returns, 100*(1-alpha))

# Example usage
weekly_returns = np.array([-0.02, 0.01, -0.05, 0.03, -0.08, ...])
var_95 = rolling_var(weekly_returns, alpha=0.95)
print(f"95% VaR: {var_95:.4f}")
        """, language="python")
    
    with tab2:
        st.markdown("#### Hill Estimator Implementation")
        st.code("""
def hill_estimator(returns, threshold_quantile=0.95, min_excesses=5):
    \"\"\"
    Calculate Hill estimator for tail index
    
    Parameters:
    - returns: array of return observations
    - threshold_quantile: quantile for threshold selection
    - min_excesses: minimum number of exceedances required
    
    Returns:
    - xi: Hill estimator (tail index)
    \"\"\"
    # Try different thresholds from 90% to desired quantile
    candidate_quantiles = np.linspace(0.90, threshold_quantile, 50)
    
    for q in reversed(candidate_quantiles):
        # Calculate threshold
        u = np.quantile(returns, 1-q)
        
        # Extract losses (negative returns)
        losses = -returns[returns < u]
        losses = losses[losses > 0]  # Ensure positive losses
        
        if len(losses) >= min_excesses:
            # Calculate Hill estimator
            min_loss = losses.min()
            xi = np.mean(np.log(losses / min_loss))
            return xi
    
    return np.nan  # Insufficient data

# Example usage
xi_hat = hill_estimator(weekly_returns, threshold_quantile=0.95)
print(f"Hill estimator: {xi_hat:.4f}")
        """, language="python")
    
    with tab3:
        st.markdown("#### Tail Dependence Implementation")
        st.code("""
def tail_dependence(x, y, u=0.95):
    \"\"\"
    Calculate left-tail dependence coefficient
    
    Parameters:
    - x, y: return series for bank and system
    - u: quantile level for tail definition
    
    Returns:
    - tau: tail dependence coefficient
    \"\"\"
    # Calculate quantiles
    qx = np.quantile(x, 1-u)  # Left tail threshold for x
    qy = np.quantile(y, 1-u)  # Left tail threshold for y
    
    # Identify left tail observations for x
    mask_x = x < qx
    
    if np.sum(mask_x) == 0:
        return np.nan
    
    # Calculate conditional probability
    tau = np.sum(y[mask_x] < qy) / np.sum(mask_x)
    
    return tau

# Example usage
bank_returns = np.array([...])  # Bank return series
system_returns = np.array([...])  # System return series

tau = tail_dependence(bank_returns, system_returns, u=0.95)
print(f"Tail dependence: {tau:.4f}")
        """, language="python")
    
    with tab4:
        st.markdown("#### Systemic Beta Implementation")
        st.code("""
def systemic_beta(bank_returns, system_returns, u=0.95):
    \"\"\"
    Calculate systemic beta following van Oordt & Zhou (2018)
    
    Parameters:
    - bank_returns: individual bank return series
    - system_returns: system/index return series  
    - u: quantile level for calculations
    
    Returns:
    - beta_T: systemic beta
    \"\"\"
    # Calculate components
    var_bank = rolling_var(bank_returns, alpha=u)
    var_system = rolling_var(system_returns, alpha=u)
    xi_system = hill_estimator(system_returns, threshold_quantile=u)
    tau = tail_dependence(bank_returns, system_returns, u=u)
    
    # Check for valid inputs
    if (xi_system is None or xi_system == 0 or 
        np.isnan(tau) or var_system == 0):
        return np.nan
    
    # Calculate systemic beta
    beta_T = (tau ** (1.0/xi_system)) * (var_bank / var_system)
    
    return beta_T

# Example usage in rolling window
def compute_rolling_systemic_beta(bank_data, system_data, window=52):
    results = []
    
    for i in range(window, len(bank_data)):
        # Extract rolling window
        bank_window = bank_data[i-window:i]
        system_window = system_data[i-window:i]
        
        # Calculate systemic beta
        beta = systemic_beta(bank_window, system_window)
        results.append(beta)
    
    return np.array(results)
        """, language="python")
    
    # Rolling window implementation
    st.markdown("### 3. Rolling Window Framework")
    
    st.markdown('<div class="methodology-section">', unsafe_allow_html=True)
    st.markdown("""
    #### Rolling Window Rationale
    
    **Window Size**: 52 weeks (1 year)
    - Captures seasonal patterns in banking
    - Sufficient observations for EVT estimation
    - Balance between stability and responsiveness
    
    **Update Frequency**: Weekly
    - Aligns with data frequency
    - Provides timely risk updates
    - Smooth evolution of risk metrics
    
    **Estimation Procedure**:
    1. For each date t, use data from [t-51, t]
    2. Calculate all EVT metrics on this window
    3. Store results with date t
    4. Advance to t+1 and repeat
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.code("""
def compute_rolling_metrics(combined_data, bank_names, window_size=52):
    \"\"\"
    Compute rolling systemic risk metrics for all banks
    \"\"\"
    results = []
    dates = combined_data.index[window_size:]  # Start after first window
    
    for date in dates:
        # Get rolling window ending at current date
        window_data = combined_data.loc[:date].tail(window_size)
        
        for bank in bank_names:
            # Get bank and corresponding regional index
            bank_returns = window_data[bank].values
            index_name = get_regional_index(bank)  # Map bank to index
            index_returns = window_data[index_name].values
            
            # Calculate all metrics
            results.append({
                'Date': date,
                'Bank': bank,
                'VaR_95': rolling_var(bank_returns, alpha=0.95),
                'Hill_95': hill_estimator(bank_returns, threshold_quantile=0.95),
                'Tau_95': tail_dependence(bank_returns, index_returns, u=0.95),
                'Beta_T': systemic_beta(bank_returns, index_returns, u=0.95)
            })
    
    return pd.DataFrame(results).set_index(['Date', 'Bank'])
    """, language="python")

def show_references():
    """References and citations"""
    
    st.markdown("## 📖 References & Further Reading")
    
    st.markdown("### 📚 Primary References")
    
    references = [
        {
            "title": "Systemic tail risk",
            "authors": "van Oordt, M. R. C., & Zhou, C.",
            "journal": "Journal of Financial and Quantitative Analysis",
            "year": "2016",
            "volume": "51(2)",
            "pages": "685-705",
            "doi": "10.1017/S0022109016000193"
        },
        {
            "title": "Estimating systematic risk in the international banking sector with extreme value theory",
            "authors": "van Oordt, M. R. C., & Zhou, C.",
            "journal": "Journal of Empirical Finance",
            "year": "2018", 
            "volume": "47",
            "pages": "1-13",
            "doi": "10.1016/j.jempfin.2018.02.004"
        },
        {
            "title": "An introduction to statistical modeling of extreme values",
            "authors": "Coles, S.",
            "journal": "Springer Series in Statistics",
            "year": "2001",
            "publisher": "Springer-Verlag London"
        },
        {
            "title": "Extreme value theory for risk managers",
            "authors": "McNeil, A. J.",
            "journal": "Internal Modelling and CAD II",
            "year": "1999",
            "publisher": "Risk Books"
        }
    ]
    
    for i, ref in enumerate(references, 1):
        st.markdown(f"**[{i}]** {ref['authors']} ({ref['year']}). *{ref['title']}*. {ref['journal']}")
        if 'volume' in ref:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Volume {ref['volume']}, pages {ref['pages']}")
        if 'doi' in ref:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;DOI: {ref['doi']}")
        st.markdown("")
    
    st.markdown("### 🔗 Methodological References")
    
    additional_refs = [
        "Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution. *Annals of Statistics*, 3(5), 1163-1174.",
        "Pickands, J. (1975). Statistical inference using extreme order statistics. *Annals of Statistics*, 3(1), 119-131.",
        "Joe, H. (1997). *Multivariate models and dependence concepts*. Chapman & Hall/CRC.",
        "Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). *Modelling extremal events*. Springer-Verlag.",
        "Adrian, T., & Brunnermeier, M. K. (2016). CoVaR. *American Economic Review*, 106(7), 1705-1741."
    ]
    
    for ref in additional_refs:
        st.markdown(f"- {ref}")
    
    st.markdown("### 📊 Data Sources")
    
    st.markdown("""
    - **Yahoo Finance**: Stock price data for G-SIBs and regional indices
    - **Financial Stability Board**: G-SIB designation and classification
    - **Basel Committee on Banking Supervision**: Regulatory frameworks
    - **FRED Economic Data**: Macroeconomic control variables
    """)
    
    st.markdown("### 🛠️ Software & Packages")
    
    software_list = [
        "**Python 3.8+**: Core programming language",
        "**Streamlit**: Web application framework", 
        "**yfinance**: Yahoo Finance data API",
        "**pandas**: Data manipulation and analysis",
        "**numpy**: Numerical computing",
        "**scipy**: Scientific computing and statistics",
        "**scikit-learn**: Machine learning algorithms",
        "**xgboost**: Gradient boosting framework",
        "**matplotlib/seaborn**: Data visualization",
        "**plotly**: Interactive visualizations"
    ]
    
    for software in software_list:
        st.markdown(f"- {software}")
    
    st.markdown("---")
    
    st.markdown("### 📞 Contact Information")
    
    st.markdown("""
    For questions about the methodology or implementation:
    
    - **Research Paper**: "Systemic Risk in Global Banking Institutions" by R. Salhi
    - **Implementation**: This Streamlit application
    - **Technical Issues**: Check GitHub repository for updates and bug reports
    """)
    
    st.markdown('<div class="methodology-section">', unsafe_allow_html=True)
    st.markdown("""
    ### ⚖️ Disclaimer
    
    This application is for research and educational purposes only. The systemic risk metrics 
    and predictions should not be used for investment decisions or regulatory compliance 
    without proper validation and expert review. Past performance does not guarantee future results.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
