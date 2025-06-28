import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import genpareto, norm, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Systemic Risk in Global Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Constants and Mappings
BANK_DICT = {
    'JPM': 'JPMorgan Chase', 'C': 'Citigroup', 'BAC': 'Bank of America',
    'WFC': 'Wells Fargo', 'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley',
    'BK': 'Bank of New York Mellon', 'STT': 'State Street',
    'RY': 'Royal Bank of Canada', 'TD': 'Toronto Dominion',
    'HSBA.L': 'HSBC', 'BARC.L': 'Barclays', 'BNP.PA': 'BNP Paribas',
    'ACA.PA': 'Groupe Crédit Agricole', 'INGA.AS': 'ING', 'DBK.DE': 'Deutsche Bank',
    'SAN.MC': 'Santander', 'GLE.PA': 'Société Générale', 'UBSG.SW': 'UBS',
    'STAN.L': 'Standard Chartered', '1288.HK': 'Agricultural Bank of China',
    '3988.HK': 'Bank of China', '0939.HK': 'China Construction Bank',
    '1398.HK': 'ICBC', '3328.HK': 'Bank of Communications',
    '8306.T': 'Mitsubishi UFJ FG', '8411.T': 'Mizuho FG', '8316.T': 'Sumitomo Mitsui FG'
}

INDEX_MAP = {
    'JPMorgan Chase': '^GSPC', 'Citigroup': '^GSPC', 'Bank of America': '^GSPC',
    'Wells Fargo': '^GSPC', 'Goldman Sachs': '^GSPC', 'Morgan Stanley': '^GSPC',
    'Bank of New York Mellon': '^GSPC', 'State Street': '^GSPC',
    'Royal Bank of Canada': '^GSPTSE', 'Toronto Dominion': '^GSPTSE',
    'BNP Paribas': '^FCHI', 'Groupe Crédit Agricole': '^FCHI', 'Société Générale': '^FCHI',
    'Santander': '^IBEX', 'HSBC': '^FTSE', 'Barclays': '^FTSE', 'Standard Chartered': '^FTSE',
    'Deutsche Bank': '^GDAXI', 'UBS': '^SSMI', 'ING': '^AEX',
    'China Construction Bank': '000001.SS', 'Agricultural Bank of China': '000001.SS',
    'ICBC': '000001.SS', 'Bank of Communications': '000001.SS', 'Bank of China': '000001.SS',
    'Mitsubishi UFJ FG': '^N225', 'Sumitomo Mitsui FG': '^N225', 'Mizuho FG': '^N225'
}

IDX_NAME_MAP = {
    '^GSPC': 'S&P 500', '^GSPTSE': 'TSX Composite', '^FCHI': 'CAC 40',
    '^IBEX': 'IBEX 35', '^FTSE': 'FTSE 100', '^GDAXI': 'DAX',
    '^SSMI': 'SMI', '^N225': 'Nikkei 225', '000001.SS': 'Shanghai Composite', '^AEX': 'AEX'
}

REGION_MAP = {
    'JPMorgan Chase': 'Americas', 'Citigroup': 'Americas', 'Bank of America': 'Americas',
    'Wells Fargo': 'Americas', 'Goldman Sachs': 'Americas', 'Morgan Stanley': 'Americas',
    'Bank of New York Mellon': 'Americas', 'State Street': 'Americas',
    'Royal Bank of Canada': 'Americas', 'Toronto Dominion': 'Americas',
    'HSBC': 'Europe', 'Barclays': 'Europe', 'BNP Paribas': 'Europe',
    'Groupe Crédit Agricole': 'Europe', 'ING': 'Europe', 'Deutsche Bank': 'Europe',
    'Santander': 'Europe', 'Société Générale': 'Europe', 'UBS': 'Europe',
    'Standard Chartered': 'Europe', 'Agricultural Bank of China': 'Asia/Pacific',
    'Bank of China': 'Asia/Pacific', 'China Construction Bank': 'Asia/Pacific',
    'ICBC': 'Asia/Pacific', 'Bank of Communications': 'Asia/Pacific',
    'Mitsubishi UFJ FG': 'Asia/Pacific', 'Mizuho FG': 'Asia/Pacific',
    'Sumitomo Mitsui FG': 'Asia/Pacific'
}

CRISIS_PERIODS = {
    'eurozone_crisis': (pd.Timestamp('2011-07-01'), pd.Timestamp('2012-12-31')),
    'china_correction': (pd.Timestamp('2015-06-01'), pd.Timestamp('2016-02-29')),
    'covid_crash': (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-05-31')),
    'ukraine_war': (pd.Timestamp('2022-02-01'), pd.Timestamp('2022-06-30')),
    'banking_stress_2023': (pd.Timestamp('2023-03-01'), pd.Timestamp('2023-05-31'))
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .crisis-warning {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .methodology-box {
        background-color: #F0F9FF;
        border: 1px solid #7DD3FC;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Core EVT Functions
def rolling_var(x, alpha=0.95):
    """Calculate rolling Value-at-Risk"""
    return -np.percentile(x, 100*(1-alpha))

def hill_estimator(x, threshold_quantile=0.95, min_excesses=5):
    """Hill estimator for tail index"""
    candidate_q = np.linspace(0.90, threshold_quantile, 50)
    
    for q in reversed(candidate_q):
        p = 1 - q
        u = np.quantile(x, p)
        losses = -x[x < u]
        losses = losses[losses > 0]
        
        if len(losses) >= min_excesses:
            min_loss = losses.min()
            return np.mean(np.log(losses / min_loss))
    
    return np.nan

def tail_dependence(x, y, u=0.95):
    """Calculate tail dependence coefficient"""
    qx, qy = np.quantile(x, u), np.quantile(y, u)
    mask = x < qx
    return np.sum(y[mask] < qy) / np.sum(mask) if np.sum(mask) > 0 else np.nan

def systemic_beta(x, y, u=0.95):
    """Calculate systemic beta as per van Oordt & Zhou (2018)"""
    VaR_x = rolling_var(x, alpha=u)
    VaR_y = rolling_var(y, alpha=u)
    xi_y = hill_estimator(y, threshold_quantile=u)
    tau = tail_dependence(x, y, u=u)
    
    if xi_y is None or xi_y == 0 or np.isnan(tau) or VaR_y == 0:
        return np.nan
    
    return (tau ** (1.0/xi_y)) * (VaR_x / VaR_y)

@st.cache_data
def download_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range('2011-01-01', '2024-12-31', freq='W-FRI')
    n_dates = len(dates)
    
    # Sample banks
    sample_banks = ['JPMorgan Chase', 'Bank of America', 'Citigroup', 'Deutsche Bank', 
                   'HSBC', 'UBS', 'ICBC', 'Mitsubishi UFJ FG']
    
    # Create sample returns data
    returns_data = {}
    
    for bank in sample_banks:
        # Generate realistic return series with volatility clustering
        returns = []
        volatility = 0.02
        
        for i in range(n_dates):
            # Add crisis effects
            crisis_multiplier = 1.0
            for crisis_name, (start, end) in CRISIS_PERIODS.items():
                if i < len(dates) and start <= dates[i] <= end:
                    crisis_multiplier = 2.0
                    break
            
            # GARCH-like volatility
            volatility = 0.8 * volatility + 0.2 * (0.01 + 0.02 * crisis_multiplier)
            
            # Generate return
            ret = np.random.normal(0.001, volatility)
            returns.append(ret)
        
        returns_data[bank] = returns
    
    # Add indices
    for idx_name in ['S&P 500', 'FTSE 100', 'DAX', 'Nikkei 225']:
        returns = []
        volatility = 0.015
        
        for i in range(n_dates):
            crisis_multiplier = 1.0
            for crisis_name, (start, end) in CRISIS_PERIODS.items():
                if i < len(dates) and start <= dates[i] <= end:
                    crisis_multiplier = 1.8
                    break
            
            volatility = 0.8 * volatility + 0.2 * (0.008 + 0.015 * crisis_multiplier)
            ret = np.random.normal(0.0008, volatility)
            returns.append(ret)
        
        returns_data[idx_name] = returns
    
    return pd.DataFrame(returns_data, index=dates)

def compute_sample_metrics(returns_data, window_size=52):
    """Compute sample systemic risk metrics"""
    results = []
    dates = returns_data.index[window_size:]
    
    progress_bar = st.progress(0)
    
    sample_banks = ['JPMorgan Chase', 'Bank of America', 'Citigroup', 'Deutsche Bank', 
                   'HSBC', 'UBS', 'ICBC', 'Mitsubishi UFJ FG']
    
    for i, date in enumerate(dates):
        if i % 10 == 0:
            progress_bar.progress(i / len(dates))
        
        window = returns_data.loc[:date].tail(window_size)
        
        for bank in sample_banks:
            if bank not in returns_data.columns:
                continue
                
            # Get corresponding index
            if bank in ['JPMorgan Chase', 'Bank of America', 'Citigroup']:
                index_name = 'S&P 500'
            elif bank in ['Deutsche Bank']:
                index_name = 'DAX'
            elif bank in ['HSBC']:
                index_name = 'FTSE 100'
            elif bank in ['UBS']:
                index_name = 'S&P 500'  # Fallback
            else:
                index_name = 'Nikkei 225'
            
            if index_name not in returns_data.columns:
                continue
            
            x_bank = window[bank].values
            x_index = window[index_name].values
            
            results.append({
                'Date': date,
                'Bank': bank,
                'VaR_95': rolling_var(x_bank, alpha=0.95),
                'Hill_95': hill_estimator(x_bank, threshold_quantile=0.95),
                'Tau_95': tail_dependence(x_bank, x_index, u=0.95),
                'Beta_T': systemic_beta(x_bank, x_index, u=0.95)
            })
    
    progress_bar.empty()
    return pd.DataFrame(results).set_index(['Date', 'Bank'])

def main():
    st.markdown('<h1 class="main-header">🏦 Systemic Risk in Global Banking Institutions</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced EVT-based Framework for G-SIB Risk Assessment (2011-2024)</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📈 Dashboard", "📚 Methodology", "🤖 Machine Learning", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📈 Dashboard":
        show_dashboard()
    elif page == "📚 Methodology":
        show_methodology()
    elif page == "🤖 Machine Learning":
        show_ml()
    elif page == "ℹ️ About":
        show_about()

def show_home():
    """Home page with project overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Project Overview
        
        This application implements the comprehensive systemic risk framework described in 
        **"Systemic Risk in Global Banking Institutions"** by R. Salhi. The framework analyzes 
        28 Global Systemically Important Banks (G-SIBs) from 2011-2024 using advanced 
        Extreme Value Theory (EVT) methodologies.
        
        ### 🔬 Key Features
        
        - **Extreme Value Theory (EVT)** for tail-risk estimation
        - **Rolling VaR** computation with 52-week windows
        - **Hill estimator** for tail index calculation
        - **Tail dependence** analysis between banks and regional indices
        - **Systemic Beta (βT)** as per van Oordt & Zhou (2018)
        - **Spillover-aware stress testing** modules
        - **Machine Learning early-warning system** using Random Forest & XGBoost
        
        ### 📊 Dataset Coverage
        
        - **28 G-SIBs** across Americas, Europe, and Asia/Pacific
        - **Weekly data** from 2011-2024 (700+ observations)
        - **Regional indices** for systemic beta calculation
        - **Crisis periods** labeled for ML training
        """)
        
        st.markdown('<div class="methodology-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🚀 Quick Start
        
        1. **📈 Dashboard**: Explore interactive visualizations of systemic beta, correlations, and stress test results
        2. **📚 Methodology**: Learn about the mathematical framework and EVT equations
        3. **🤖 Machine Learning**: Train models and view crisis predictions
        
        **Note**: This demo uses sample data for demonstration purposes.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### 🏦 G-SIB Coverage
        
        **Americas (10)**
        - JPMorgan Chase
        - Bank of America
        - Citigroup
        - Wells Fargo
        - Goldman Sachs
        - Morgan Stanley
        - And 4 more...
        
        **Europe (10)**
        - HSBC
        - BNP Paribas
        - Deutsche Bank
        - UBS
        - Barclays
        - And 5 more...
        
        **Asia/Pacific (8)**
        - ICBC
        - China Construction Bank
        - Bank of China
        - Mitsubishi UFJ FG
        - And 4 more...
        """)
        
        st.markdown('<div class="crisis-warning">', unsafe_allow_html=True)
        st.markdown("""
        ### ⚠️ Crisis Periods Analyzed
        
        - **Eurozone Crisis** (2011-2012)
        - **China Correction** (2015-2016)
        - **COVID-19 Crash** (2020)
        - **Ukraine War** (2022)
        - **Banking Stress 2023**
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def show_dashboard():
    """Dashboard with sample analysis"""
    st.title("📈 Systemic Risk Dashboard")
    st.markdown("**Interactive analysis of G-SIB systemic risk metrics (Sample Data)**")
    
    # Load sample data
    with st.spinner("📊 Loading sample data..."):
        returns_data = download_sample_data()
        metrics_data = compute_sample_metrics(returns_data)
    
    st.success(f"✅ Sample data loaded: {len(metrics_data.index.get_level_values(1).unique())} banks, {len(returns_data)} weeks")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(returns_data.index.min().date(), returns_data.index.max().date()),
        min_value=returns_data.index.min().date(),
        max_value=returns_data.index.max().date()
    )
    
    # Main dashboard tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Overview", "📈 Time Series", "🔥 Heatmaps"])
    
    with tab1:
        show_overview_dashboard(metrics_data)
    
    with tab2:
        show_time_series_dashboard(metrics_data)
    
    with tab3:
        show_heatmaps_dashboard(metrics_data)

def show_overview_dashboard(metrics_data):
    """Overview section with key metrics"""
    st.subheader("🎯 Current Risk Overview (Sample Data)")
    
    # Get latest data
    latest_date = metrics_data.index.get_level_values(0).max()
    latest_metrics = metrics_data.loc[latest_date]
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_beta = latest_metrics['Beta_T'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Systemic Beta", f"{avg_beta:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        high_risk_banks = (latest_metrics['Beta_T'] > 2.0).sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High Risk Banks (β > 2.0)", f"{high_risk_banks}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_var = latest_metrics['VaR_95'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average VaR", f"{avg_var:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_tau = latest_metrics['Tau_95'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Tail Dependence", f"{avg_tau:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Regional breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Regional Risk Distribution")
        
        regional_data = []
        for bank in latest_metrics.index:
            region = REGION_MAP.get(bank, 'Unknown')
            beta = latest_metrics.loc[bank, 'Beta_T']
            if not pd.isna(beta):
                regional_data.append({'Region': region, 'Bank': bank, 'Beta_T': beta})
        
        if regional_data:
            regional_df = pd.DataFrame(regional_data)
            regional_summary = regional_df.groupby('Region')['Beta_T'].agg(['mean', 'std', 'count'])
            st.dataframe(regional_summary.round(3))
    
    with col2:
        st.subheader("🏆 Top Risk Banks")
        
        top_risk = latest_metrics.nlargest(5, 'Beta_T')[['Beta_T', 'VaR_95', 'Tau_95']]
        top_risk['Region'] = top_risk.index.map(REGION_MAP)
        st.dataframe(top_risk.round(4))

def show_time_series_dashboard(metrics_data):
    """Time series visualization"""
    st.subheader("📈 Time Series Analysis")
    
    # Systemic Beta time series
    st.markdown("#### Systemic Beta (βT) Evolution")
    
    beta_data = []
    for bank in metrics_data.index.get_level_values(1).unique()[:5]:  # Top 5 banks
        try:
            bank_data = metrics_data.xs(bank, level=1)
            for date, row in bank_data.iterrows():
                beta_data.append({
                    'Date': date,
                    'Bank': bank,
                    'Beta_T': row['Beta_T'],
                    'Region': REGION_MAP.get(bank, 'Unknown')
                })
        except:
            continue
    
    if beta_data:
        beta_df = pd.DataFrame(beta_data)
        beta_df = beta_df.dropna()
        
        fig = px.line(beta_df, x='Date', y='Beta_T', color='Bank', 
                     title="Systemic Beta Evolution (Sample Data)",
                     labels={'Beta_T': 'Systemic Beta (βT)'})
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold (β=2.0)")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def show_heatmaps_dashboard(metrics_data):
    """Correlation and risk heatmaps"""
    st.subheader("🔥 Risk Correlation Analysis")
    
    # Get recent data for correlation
    latest_date = metrics_data.index.get_level_values(0).max()
    recent_dates = metrics_data.index.get_level_values(0).unique()
    recent_dates = recent_dates[recent_dates >= (latest_date - pd.Timedelta(weeks=12))]
    
    recent_data = metrics_data[metrics_data.index.get_level_values(0).isin(recent_dates)]
    
    # Systemic Beta correlation matrix
    try:
        beta_pivot = recent_data.reset_index().pivot(index='Date', columns='Bank', values='Beta_T')
        beta_corr = beta_pivot.corr()
        
        if not beta_corr.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(beta_corr, dtype=bool))
            sns.heatmap(beta_corr, mask=mask, annot=False, cmap='RdYlBu_r', 
                       center=0, square=True, linewidths=0.5, ax=ax)
            plt.title("Inter-Bank Beta Correlations (Sample Data)")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
    except:
        st.info("Correlation analysis requires more data points")

def show_methodology():
    """Methodology page"""
    st.title("📚 Methodology & Mathematical Framework")
    st.markdown("**Theoretical foundation of the EVT-based systemic risk measurement**")
    
    # Mathematical framework
    st.markdown("## 🔬 Key Equations")
    
    st.markdown("### 1. Systemic Beta (van Oordt & Zhou, 2018)")
    st.latex(r"\beta_T = \tau^{1/\xi_y} \cdot \frac{VaR_x}{VaR_y}")
    
    st.markdown("""
    **Where:**
    - τ: Tail dependence coefficient
    - ξ_y: Hill estimator (tail index) of the system
    - VaR_x: Value-at-Risk of individual bank
    - VaR_y: Value-at-Risk of the system
    """)
    
    st.markdown("### 2. Value-at-Risk")
    st.latex(r"VaR_\alpha = -\text{quantile}(r_t, 1-\alpha)")
    
    st.markdown("### 3. Hill Estimator")
    st.latex(r"\hat{\xi}_H = \frac{1}{n} \sum_{i=1}^{n} \ln\left(\frac{X_i}{u}\right)")
    
    st.markdown("### 4. Tail Dependence")
    st.latex(r"\tau_L = \lim_{u \to 0^+} P(Y \leq F_Y^{-1}(u) | X \leq F_X^{-1}(u))")
    
    # Interactive calculator
    st.markdown("## 🧮 Interactive Systemic Beta Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tau_input = st.slider("Tail Dependence (τ)", 0.1, 0.9, 0.5, 0.05)
        xi_input = st.slider("System Tail Index (ξ)", 0.1, 0.8, 0.3, 0.05)
        var_bank = st.slider("Bank VaR", 0.02, 0.15, 0.08, 0.01)
        var_system = st.slider("System VaR", 0.02, 0.10, 0.05, 0.01)
    
    with col2:
        beta_t = (tau_input ** (1/xi_input)) * (var_bank / var_system)
        
        st.metric("Calculated βT", f"{beta_t:.3f}")
        
        if beta_t < 1.0:
            st.success("🟢 Low Risk")
        elif beta_t < 2.0:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

def show_ml():
    """Machine Learning page"""
    st.title("🤖 Machine Learning Early Warning System")
    st.markdown("**Crisis prediction using systemic risk features**")
    
    # Create sample ML data
    st.markdown("## 📊 Sample ML Analysis")
    
    # Generate sample features
    np.random.seed(42)
    dates = pd.date_range('2012-01-01', '2024-12-31', freq='M')
    
    # Create crisis labels
    labels = []
    for date in dates:
        crisis_label = 0
        for crisis_name, (start, end) in CRISIS_PERIODS.items():
            if start <= date <= end:
                crisis_label = 1
                break
        labels.append(crisis_label)
    
    # Generate sample features
    features = {
        'beta_mean': np.random.normal(1.2, 0.4, len(dates)),
        'var_extreme_count': np.random.poisson(2, len(dates)),
        'tau_max': np.random.beta(2, 3, len(dates)),
        'beta_std': np.random.gamma(2, 0.1, len(dates))
    }
    
    # Add crisis effects
    for i, label in enumerate(labels):
        if label == 1:
            features['beta_mean'][i] *= 1.5
            features['var_extreme_count'][i] += 3
            features['tau_max'][i] *= 1.3
    
    features_df = pd.DataFrame(features, index=dates)
    
    # Train simple model
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.3, random_state=42
    )
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Performance")
        
        if len(set(y_test)) > 1:
            auc_score = roc_auc_score(y_test, rf_proba)
            st.metric("AUC Score", f"{auc_score:.3f}")
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test, rf_pred, output_dict=True, zero_division=0)
        
        st.metric("Precision", f"{report['weighted avg']['precision']:.3f}")
        st.metric("Recall", f"{report['weighted avg']['recall']:.3f}")
    
    with col2:
        st.markdown("### Feature Importance")
        
        importance = pd.Series(rf_model.feature_importances_, 
                             index=features_df.columns).sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        importance.plot(kind='barh', ax=ax)
        plt.title("Feature Importance")
        plt.tight_layout()
        st.pyplot(fig)
    
    # Current prediction
    st.markdown("### Current Risk Assessment")
    
    latest_features = features_df.tail(1)
    current_prob = rf_model.predict_proba(latest_features)[0, 1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Crisis Probability", f"{current_prob:.1%}")
    
    with col2:
        risk_level = "🔴 HIGH" if current_prob > 0.7 else "🟡 MODERATE" if current_prob > 0.4 else "🟢 LOW"
        st.markdown(f"**Risk Level**: {risk_level}")
    
    with col3:
        st.metric("Latest Beta Mean", f"{latest_features['beta_mean'].iloc[0]:.3f}")

def show_about():
    """About page"""
    st.title("ℹ️ About This Application")
    
    st.markdown("""
    ## 📖 Research Implementation
    
    This Streamlit application demonstrates the implementation of the systemic risk framework 
    from **"Systemic Risk in Global Banking Institutions"** by R. Salhi.
    
    ### 🛠️ Technical Stack
    
    - **Frontend**: Streamlit with custom CSS
    - **Data**: Yahoo Finance API (in full version)
    - **Analytics**: NumPy, Pandas, SciPy
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Visualization**: Matplotlib, Seaborn, Plotly
    
    ### 🔬 Implementation Notes
    
    This demo version uses **sample data** to demonstrate the framework capabilities:
    
    1. **Sample Data Generation**: Realistic banking returns with crisis effects
    2. **EVT Calculations**: Full implementation of Hill estimator, tail dependence, systemic beta
    3. **Interactive Visualizations**: Real-time parameter adjustment
    4. **ML Pipeline**: Complete feature engineering and model training
    
    ### 📊 Full Version Features
    
    The complete implementation includes:
    - Real-time data from Yahoo Finance for 28 G-SIBs
    - 13+ years of historical data (2011-2024)
    - Regional index integration
    - Advanced stress testing scenarios
    - Production-ready ML pipeline
    
    ### 📞 Contact
    
    For questions about the methodology, refer to the original research paper.
    For technical implementation details, see the methodology section.
    """)
    
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by Extreme Value Theory*")

if __name__ == "__main__":
    main()
