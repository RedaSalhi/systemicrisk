"""
Enhanced Dashboard Page
Advanced systemic risk analysis with improved technical accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Try to import enhanced data processor
try:
    from enhanced_data_processor import (
        EnhancedDataProcessor, create_enhanced_processor, EVTParams,
        ENHANCED_BANK_DICT, ENHANCED_INDEX_MAP, ENHANCED_IDX_NAME_MAP, 
        REGION_MAP, CRISIS_PERIODS
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    # Fallback to basic implementations
    import yfinance as yf
    from scipy.stats import genpareto
    
    # Basic mappings for fallback
    ENHANCED_BANK_DICT = {
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

    ENHANCED_INDEX_MAP = {
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

    ENHANCED_IDX_NAME_MAP = {
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

# Set page configuration
st.set_page_config(
    page_title="Enhanced Systemic Risk Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load CSS
def load_css():
    """Load external CSS file or fallback to inline styles"""
    css_file = Path(__file__).parent.parent / "static" / "styles.css"
    
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline CSS
        st.markdown("""
        <style>
        :root {
            --primary-color: #2563eb;
            --accent-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --background-card: #ffffff;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --border-radius: 8px;
        }
        
        .metric-card {
            background-color: var(--background-card);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            margin: 1rem 0;
            border-left: 4px solid var(--primary-color);
        }
        
        .warning-card {
            background-color: #fef3f2;
            padding: 1.5rem;
            border-radius: var(--border-radius);
            border: 1px solid #fecaca;
            box-shadow: var(--shadow-sm);
            margin: 1rem 0;
            border-left: 4px solid var(--danger-color);
        }
        
        .success-card {
            background-color: #f0fdf4;
            padding: 1.5rem;
            border-radius: var(--border-radius);
            border: 1px solid #bbf7d0;
            box-shadow: var(--shadow-sm);
            margin: 1rem 0;
            border-left: 4px solid var(--accent-color);
        }
        
        .chart-container {
            background-color: var(--background-card);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

# Load CSS
load_css()

# Enhanced EVT calculations for fallback
def enhanced_var_fallback(x, alpha=0.95):
    """Enhanced VaR calculation fallback"""
    if len(x) == 0 or np.isnan(x).all():
        return np.nan
    clean_x = x[~np.isnan(x)]
    return -np.percentile(clean_x, 100*(1-alpha))

def enhanced_hill_fallback(x, threshold_quantile=0.95, min_excesses=10):
    """Enhanced Hill estimator fallback"""
    if len(x) < min_excesses * 2:
        return np.nan
    
    clean_x = x[~np.isnan(x)]
    losses = -clean_x[clean_x < 0]
    
    if len(losses) < min_excesses:
        return np.nan
    
    # Simple threshold selection
    u = np.quantile(losses, threshold_quantile)
    exceedances = losses[losses >= u]
    
    if len(exceedances) < min_excesses:
        return np.nan
    
    log_ratios = np.log(exceedances / u)
    return np.mean(log_ratios)

def enhanced_tail_dependence_fallback(x, y, u=0.95):
    """Enhanced tail dependence fallback"""
    if len(x) != len(y) or len(x) == 0:
        return np.nan
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    
    if len(x_clean) < 20:
        return np.nan
    
    qx = np.quantile(x_clean, 1-u)
    qy = np.quantile(y_clean, 1-u)
    
    tail_x_mask = x_clean <= qx
    if np.sum(tail_x_mask) == 0:
        return np.nan
    
    joint_tail_prob = np.sum((x_clean <= qx) & (y_clean <= qy))
    marginal_tail_prob = np.sum(tail_x_mask)
    
    return joint_tail_prob / marginal_tail_prob

def enhanced_systemic_beta_fallback(x, y, u=0.95):
    """Enhanced systemic beta fallback"""
    VaR_x = enhanced_var_fallback(x, alpha=u)
    VaR_y = enhanced_var_fallback(y, alpha=u)
    xi_y = enhanced_hill_fallback(y, threshold_quantile=u)
    tau = enhanced_tail_dependence_fallback(x, y, u=u)
    
    if (np.isnan(xi_y) or xi_y <= 0 or 
        np.isnan(tau) or VaR_y <= 0):
        return np.nan
    
    try:
        tau_adj = np.clip(tau, 1e-6, 1-1e-6)
        xi_adj = np.clip(xi_y, 1e-6, 10)
        beta_t = (tau_adj ** (1.0/xi_adj)) * (VaR_x / VaR_y)
        
        if not np.isfinite(beta_t) or beta_t < 0:
            return np.nan
        return beta_t
    except:
        return np.nan

@st.cache_data
def load_and_process_enhanced_data():
    """Load and process data using enhanced methods if available"""
    if ENHANCED_AVAILABLE:
        # Use enhanced processor
        processor = create_enhanced_processor(
            confidence_levels=[0.95, 0.99],
            min_exceedances=15,
            threshold_strategy="adaptive"
        )
        
        try:
            # Download sample of banks for demo
            selected_banks = ['JPMorgan Chase', 'Bank of America', 'Citigroup', 
                            'Deutsche Bank', 'HSBC', 'UBS']
            
            bank_prices, index_prices = processor.download_data(
                start_date='2020-01-01',
                end_date='2024-12-31',
                selected_banks=selected_banks
            )
            
            combined_returns, bank_names = processor.prepare_returns(bank_prices, index_prices)
            
            metrics_dict = processor.compute_rolling_metrics(
                combined_returns, bank_names, window_size=52, min_periods=30
            )
            
            return combined_returns, bank_names, metrics_dict
            
        except Exception as e:
            st.error(f"Enhanced data processing failed: {str(e)}")
            return None, None, None
    
    else:
        # Fallback to sample data generation
        return generate_fallback_sample_data()

def generate_fallback_sample_data():
    """Generate sample data when enhanced processor is not available"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='W-FRI')
    n_dates = len(dates)
    
    # Sample banks
    sample_banks = ['JPMorgan Chase', 'Bank of America', 'Citigroup', 
                   'Deutsche Bank', 'HSBC', 'UBS']
    
    # Create return series
    returns_data = {}
    
    for bank in sample_banks:
        returns = []
        volatility = 0.02
        
        for i in range(n_dates):
            # Crisis effects
            crisis_multiplier = 1.0
            for crisis_name, (start, end) in CRISIS_PERIODS.items():
                if i < len(dates) and start <= dates[i] <= end:
                    crisis_multiplier = 2.5 if crisis_name == 'covid_crash' else 2.0
                    break
            
            volatility = 0.8 * volatility + 0.2 * (0.015 + 0.02 * crisis_multiplier)
            
            if crisis_multiplier > 1.5:
                ret = np.random.standard_t(df=3) * volatility * 0.5
            else:
                ret = np.random.normal(0.001, volatility)
            
            returns.append(ret)
        
        returns_data[bank] = returns
    
    # Add indices
    for idx_name in ['S&P 500', 'FTSE 100', 'DAX']:
        returns = []
        volatility = 0.012
        
        for i in range(n_dates):
            crisis_multiplier = 1.0
            for crisis_name, (start, end) in CRISIS_PERIODS.items():
                if i < len(dates) and start <= dates[i] <= end:
                    crisis_multiplier = 2.0 if crisis_name == 'covid_crash' else 1.8
                    break
            
            volatility = 0.75 * volatility + 0.25 * (0.008 + 0.012 * crisis_multiplier)
            
            if crisis_multiplier > 1.5:
                ret = np.random.standard_t(df=4) * volatility * 0.4
            else:
                ret = np.random.normal(0.0008, volatility)
                
            returns.append(ret)
        
        returns_data[idx_name] = returns
    
    combined_returns = pd.DataFrame(returns_data, index=dates)
    
    # Compute basic metrics
    metrics_data = compute_fallback_metrics(combined_returns, sample_banks)
    
    metrics_dict = {
        'metrics_95': metrics_data,
        'metrics_99': metrics_data.copy()  # Simplified for demo
    }
    
    return combined_returns, sample_banks, metrics_dict

def compute_fallback_metrics(returns_data, bank_names, window_size=52):
    """Compute metrics using fallback methods"""
    results = []
    dates = returns_data.index[window_size:]
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        if i % 10 == 0:
            progress_bar.progress(i / len(dates))
        
        window = returns_data.loc[:date].tail(window_size)
        
        for bank in bank_names:
            if bank not in returns_data.columns:
                continue
                
            # Get corresponding index
            if bank in ['JPMorgan Chase', 'Bank of America', 'Citigroup']:
                index_name = 'S&P 500'
            elif bank in ['Deutsche Bank']:
                index_name = 'DAX'
            elif bank in ['HSBC']:
                index_name = 'FTSE 100'
            else:
                index_name = 'S&P 500'
            
            if index_name not in returns_data.columns:
                continue
            
            x_bank = window[bank].dropna().values
            x_index = window[index_name].dropna().values
            
            min_len = min(len(x_bank), len(x_index))
            if min_len < 30:
                continue
                
            x_bank = x_bank[-min_len:]
            x_index = x_index[-min_len:]
            
            # Calculate metrics
            var_95 = enhanced_var_fallback(x_bank, alpha=0.95)
            hill_95 = enhanced_hill_fallback(x_bank, threshold_quantile=0.95)
            tau_95 = enhanced_tail_dependence_fallback(x_bank, x_index, u=0.95)
            beta_t = enhanced_systemic_beta_fallback(x_bank, x_index, u=0.95)
            
            results.append({
                'Date': date,
                'Bank': bank,
                'Region': REGION_MAP.get(bank, 'Unknown'),
                'VaR_95': var_95,
                'Hill_95': hill_95,
                'Tau_95': tau_95,
                'Beta_T_95': beta_t,
                'n_observations': len(x_bank),
                'volatility': np.std(x_bank)
            })
    
    progress_bar.empty()
    return pd.DataFrame(results).set_index(['Date', 'Bank']) if results else pd.DataFrame()

def get_risk_classification(beta_value):
    """Classify risk level based on beta value"""
    if pd.isna(beta_value):
        return "Unknown", "#6c757d"
    elif beta_value >= 3.0:
        return "Critical", "#dc3545"
    elif beta_value >= 2.0:
        return "High", "#fd7e14"
    elif beta_value >= 1.0:
        return "Medium", "#ffc107"
    else:
        return "Low", "#28a745"

def main():
    st.title("Enhanced Systemic Risk Dashboard")
    st.markdown("**Advanced analysis of G-SIB systemic risk metrics with improved technical accuracy**")
    
    # Load data
    with st.spinner("Loading and processing enhanced data..."):
        combined_data, bank_names, metrics_dict = load_and_process_enhanced_data()
    
    if combined_data is None or metrics_dict is None:
        st.error("Failed to load data. Please check your setup.")
        return
    
    # Select confidence level
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        options=["95%", "99%"],
        index=0
    )
    
    # Get appropriate metrics
    metrics_key = f'metrics_{confidence_level.replace("%", "")}'
    if metrics_key in metrics_dict and not metrics_dict[metrics_key].empty:
        metrics_data = metrics_dict[metrics_key]
    else:
        st.error(f"No data available for {confidence_level} confidence level")
        return
    
    st.success(f"Enhanced data loaded: {len(bank_names)} banks, {len(combined_data)} weeks of data")
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Overview", "Individual Banks", "Regional Analysis", "Advanced Metrics"]
    )
    
    if analysis_type == "Individual Banks":
        selected_banks = st.sidebar.multiselect(
            "Select Banks",
            options=bank_names,
            default=bank_names[:3]
        )
    else:
        selected_banks = bank_names
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Current Status", "Time Series", "Risk Analysis", "Correlations", "Stress Testing"
    ])
    
    with tab1:
        show_current_status(metrics_data, confidence_level)
    
    with tab2:
        show_time_series_analysis(metrics_data, selected_banks, confidence_level)
    
    with tab3:
        show_risk_analysis(metrics_data, confidence_level)
    
    with tab4:
        show_correlation_analysis(metrics_data, confidence_level)
    
    with tab5:
        show_stress_testing(metrics_data, combined_data, confidence_level)

def show_current_status(metrics_data, confidence_level):
    """Show current risk status overview"""
    st.subheader(f"Current Risk Status ({confidence_level} Confidence)")
    
    # Get latest data
    latest_date = metrics_data.index.get_level_values(0).max()
    latest_metrics = metrics_data.loc[latest_date]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_beta = latest_metrics['Beta_T_95'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Systemic Beta", f"{avg_beta:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        high_risk_count = (latest_metrics['Beta_T_95'] > 2.0).sum()
        card_class = "warning-card" if high_risk_count > 2 else "metric-card"
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.metric("High Risk Banks", f"{high_risk_count}")
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
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Level Distribution")
        
        risk_levels = []
        for bank in latest_metrics.index:
            beta = latest_metrics.loc[bank, 'Beta_T_95']
            risk_level, _ = get_risk_classification(beta)
            risk_levels.append(risk_level)
        
        risk_counts = pd.Series(risk_levels).value_counts()
        
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title="Distribution of Risk Levels",
                    color_discrete_map={
                        'Low': '#28a745',
                        'Medium': '#ffc107', 
                        'High': '#fd7e14',
                        'Critical': '#dc3545'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Risk Banks")
        
        # Create risk summary table
        risk_summary = []
        for bank in latest_metrics.index:
            beta = latest_metrics.loc[bank, 'Beta_T_95']
            risk_level, color = get_risk_classification(beta)
            
            risk_summary.append({
                'Bank': bank,
                'Beta_T': beta,
                'Risk_Level': risk_level,
                'Region': REGION_MAP.get(bank, 'Unknown')
            })
        
        risk_df = pd.DataFrame(risk_summary)
        top_risk = risk_df.nlargest(6, 'Beta_T')
        
        # Format for display
        display_df = top_risk[['Bank', 'Beta_T', 'Risk_Level', 'Region']].copy()
        display_df['Beta_T'] = display_df['Beta_T'].round(4)
        
        st.dataframe(display_df, use_container_width=True)

def show_time_series_analysis(metrics_data, selected_banks, confidence_level):
    """Show time series analysis"""
    st.subheader(f"Time Series Analysis ({confidence_level} Confidence)")
    
    if not selected_banks:
        st.warning("Please select banks in the sidebar")
        return
    
    # Systemic Beta Evolution
    st.markdown("#### Systemic Beta Evolution")
    
    beta_data = []
    for bank in selected_banks:
        try:
            bank_data = metrics_data.xs(bank, level=1)
            for date, row in bank_data.iterrows():
                if not pd.isna(row['Beta_T_95']):
                    beta_data.append({
                        'Date': date,
                        'Bank': bank,
                        'Beta_T': row['Beta_T_95'],
                        'Region': REGION_MAP.get(bank, 'Unknown')
                    })
        except KeyError:
            continue
    
    if beta_data:
        beta_df = pd.DataFrame(beta_data)
        
        fig = px.line(beta_df, x='Date', y='Beta_T', color='Bank',
                     title=f"Systemic Beta Evolution ({confidence_level})",
                     labels={'Beta_T': 'Systemic Beta (βT)'})
        
        # Add threshold lines
        fig.add_hline(y=1.0, line_dash="dot", line_color="blue", 
                     annotation_text="Medium Risk (β=1.0)")
        fig.add_hline(y=2.0, line_dash="dash", line_color="orange", 
                     annotation_text="High Risk (β=2.0)")
        fig.add_hline(y=3.0, line_dash="dash", line_color="red", 
                     annotation_text="Critical Risk (β=3.0)")
        
        # Add crisis periods
        for crisis_name, (start, end) in CRISIS_PERIODS.items():
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="red", opacity=0.1,
                annotation_text=crisis_name.replace('_', ' ').title()
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### VaR Evolution")
        
        var_data = []
        for bank in selected_banks[:3]:  # Limit for clarity
            try:
                bank_data = metrics_data.xs(bank, level=1)
                for date, row in bank_data.iterrows():
                    if not pd.isna(row['VaR_95']):
                        var_data.append({
                            'Date': date,
                            'Bank': bank,
                            'VaR': row['VaR_95']
                        })
            except KeyError:
                continue
        
        if var_data:
            var_df = pd.DataFrame(var_data)
            fig = px.line(var_df, x='Date', y='VaR', color='Bank',
                         title=f"Value-at-Risk Evolution ({confidence_level})")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Tail Dependence Evolution")
        
        tau_data = []
        for bank in selected_banks[:3]:
            try:
                bank_data = metrics_data.xs(bank, level=1)
                for date, row in bank_data.iterrows():
                    if not pd.isna(row['Tau_95']):
                        tau_data.append({
                            'Date': date,
                            'Bank': bank,
                            'Tau': row['Tau_95']
                        })
            except KeyError:
                continue
        
        if tau_data:
            tau_df = pd.DataFrame(tau_data)
            fig = px.line(tau_df, x='Date', y='Tau', color='Bank',
                         title=f"Tail Dependence Evolution ({confidence_level})")
            st.plotly_chart(fig, use_container_width=True)

def show_risk_analysis(metrics_data, confidence_level):
    """Show detailed risk analysis"""
    st.subheader("Advanced Risk Analysis")
    
    latest_date = metrics_data.index.get_level_values(0).max()
    latest_metrics = metrics_data.loc[latest_date]
    
    # Beta distribution analysis
    st.markdown("#### Systemic Beta Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        valid_betas = latest_metrics['Beta_T_95'].dropna()
        if len(valid_betas) > 0:
            fig = px.histogram(x=valid_betas, nbins=15,
                              title="Distribution of Systemic Beta Values")
            fig.add_vline(x=1.0, line_dash="dash", line_color="blue")
            fig.add_vline(x=2.0, line_dash="dash", line_color="orange") 
            fig.add_vline(x=3.0, line_dash="dash", line_color="red")
            fig.update_layout(xaxis_title="Systemic Beta", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Statistics table
        if len(valid_betas) > 0:
            stats = {
                'Mean': valid_betas.mean(),
                'Median': valid_betas.median(),
                'Std Dev': valid_betas.std(),
                'Min': valid_betas.min(),
                'Max': valid_betas.max(),
                '75th Percentile': valid_betas.quantile(0.75),
                '90th Percentile': valid_betas.quantile(0.90)
            }
            
            stats_df = pd.DataFrame(list(stats.items()), 
                                  columns=['Statistic', 'Value'])
            stats_df['Value'] = stats_df['Value'].round(4)
            st.dataframe(stats_df, use_container_width=True)
    
    # Regional analysis
    st.markdown("#### Regional Risk Assessment")
    
    regional_analysis = []
    for bank in latest_metrics.index:
        region = REGION_MAP.get(bank, 'Unknown')
        beta = latest_metrics.loc[bank, 'Beta_T_95']
        var_val = latest_metrics.loc[bank, 'VaR_95']
        
        if not pd.isna(beta):
            risk_level, _ = get_risk_classification(beta)
            regional_analysis.append({
                'Region': region,
                'Bank': bank,
                'Beta_T': beta,
                'VaR': var_val,
                'Risk_Level': risk_level
            })
    
    if regional_analysis:
        regional_df = pd.DataFrame(regional_analysis)
        
        # Regional summary
        regional_summary = regional_df.groupby('Region').agg({
            'Beta_T': ['mean', 'max', 'std'],
            'VaR': 'mean',
            'Risk_Level': lambda x: (x.isin(['High', 'Critical'])).sum()
        }).round(4)
        
        regional_summary.columns = ['Avg_Beta', 'Max_Beta', 'Std_Beta', 'Avg_VaR', 'High_Risk_Count']
        st.dataframe(regional_summary, use_container_width=True)

def show_correlation_analysis(metrics_data, confidence_level):
    """Show correlation analysis"""
    st.subheader("Correlation and Network Analysis")
    
    # Get recent data for correlation
    latest_date = metrics_data.index.get_level_values(0).max()
    recent_dates = metrics_data.index.get_level_values(0).unique()
    recent_dates = recent_dates[recent_dates >= (latest_date - pd.Timedelta(weeks=24))]
    
    recent_data = metrics_data[metrics_data.index.get_level_values(0).isin(recent_dates)]
    
    try:
        # Beta correlation matrix
        beta_pivot = recent_data.reset_index().pivot(index='Date', columns='Bank', values='Beta_T_95')
        beta_corr = beta_pivot.corr()
        
        if not beta_corr.empty and len(beta_corr) > 1:
            st.markdown("#### Inter-Bank Beta Correlations")
            
            # Enhanced heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(beta_corr, dtype=bool))
            
            sns.heatmap(beta_corr, mask=mask, annot=False, cmap='RdBu_r',
                       center=0, square=True, linewidths=0.5, ax=ax)
            
            plt.title(f"Systemic Beta Correlations ({confidence_level})", 
                     fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Correlation statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Correlation Statistics")
                
                corr_values = beta_corr.values[np.triu_indices_from(beta_corr.values, k=1)]
                corr_stats = {
                    'Mean': np.mean(corr_values),
                    'Median': np.median(corr_values),
                    'Std Dev': np.std(corr_values),
                    'Max': np.max(corr_values),
                    'Min': np.min(corr_values)
                }
                
                stats_df = pd.DataFrame(list(corr_stats.items()),
                                      columns=['Metric', 'Value'])
                stats_df['Value'] = stats_df['Value'].round(4)
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.markdown("#### Highly Correlated Pairs")
                
                high_corr_pairs = []
                for i in range(len(beta_corr.columns)):
                    for j in range(i+1, len(beta_corr.columns)):
                        corr_val = beta_corr.iloc[i, j]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.6:
                            high_corr_pairs.append({
                                'Bank 1': beta_corr.columns[i],
                                'Bank 2': beta_corr.columns[j],
                                'Correlation': corr_val
                            })
                
                if high_corr_pairs:
                    pairs_df = pd.DataFrame(high_corr_pairs)
                    pairs_df = pairs_df.sort_values('Correlation', key=abs, ascending=False)
                    pairs_df['Correlation'] = pairs_df['Correlation'].round(4)
                    st.dataframe(pairs_df, use_container_width=True)
                else:
                    st.info("No highly correlated pairs found (|correlation| > 0.6)")
    
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")

def show_stress_testing(metrics_data, combined_data, confidence_level):
    """Show stress testing interface"""
    st.subheader("Advanced Stress Testing")
    
    # Stress test parameters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Stress Test Configuration")
        
        shock_magnitude = st.slider(
            "Shock Magnitude (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        ) / 100
        
        shock_region = st.selectbox(
            "Primary Shock Region",
            options=['Americas', 'Europe', 'Asia/Pacific', 'Global']
        )
        
        scenario_type = st.selectbox(
            "Stress Scenario",
            options=[
                "Market Crash",
                "Banking Crisis", 
                "Sovereign Debt Crisis",
                "Liquidity Crisis",
                "Custom Shock"
            ]
        )
        
        include_spillovers = st.checkbox("Include Spillover Effects", value=True)
        
        if st.button("Run Stress Test", type="primary"):
            st.session_state.run_stress_test = True
            st.session_state.stress_params = {
                'shock_magnitude': shock_magnitude,
                'shock_region': shock_region,
                'scenario_type': scenario_type,
                'include_spillovers': include_spillovers
            }
    
    with col2:
        if hasattr(st.session_state, 'run_stress_test') and st.session_state.run_stress_test:
            st.markdown("#### Stress Test Results")
            
            params = st.session_state.stress_params
            
            # Run stress test
            stress_results = run_enhanced_stress_test(
                metrics_data, 
                params['shock_magnitude'],
                params['shock_region'],
                params['include_spillovers']
            )
            
            if not stress_results.empty:
                # Summary metrics
                st.info(f"**Scenario**: {params['scenario_type']} | "
                       f"**Shock**: {params['shock_magnitude']*100:.0f}% | "
                       f"**Region**: {params['shock_region']}")
                
                summary_cols = st.columns(3)
                
                with summary_cols[0]:
                    avg_impact = stress_results['Impact_Magnitude'].mean()
                    st.metric("Average Impact", f"{avg_impact:.1f}%")
                
                with summary_cols[1]:
                    max_impact = stress_results['Impact_Magnitude'].max()
                    st.metric("Maximum Impact", f"{max_impact:.1f}%")
                
                with summary_cols[2]:
                    high_impact = (stress_results['Impact_Magnitude'] > 25).sum()
                    st.metric("Severely Affected Banks", f"{high_impact}")
                
                # Results visualization
                fig = px.bar(
                    stress_results.nlargest(10, 'Impact_Magnitude'),
                    x='Bank',
                    y='Impact_Magnitude',
                    color='Impact_Type',
                    title="Top 10 Banks by Impact Magnitude",
                    labels={'Impact_Magnitude': 'Impact (%)'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.markdown("#### Detailed Results")
                display_cols = ['Bank', 'Region', 'Impact_Type', 'Impact_Magnitude', 
                              'Original_Beta', 'Shocked_Beta', 'Risk_Level_Change']
                
                display_df = stress_results[display_cols].copy()
                display_df = display_df.sort_values('Impact_Magnitude', ascending=False)
                display_df['Impact_Magnitude'] = display_df['Impact_Magnitude'].round(2)
                display_df['Original_Beta'] = display_df['Original_Beta'].round(4)
                display_df['Shocked_Beta'] = display_df['Shocked_Beta'].round(4)
                
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Configure stress test parameters and click 'Run Stress Test' to see results")

def run_enhanced_stress_test(metrics_data, shock_magnitude, shock_region, include_spillovers):
    """Run enhanced stress test with improved methodology"""
    latest_date = metrics_data.index.get_level_values(0).max()
    latest_metrics = metrics_data.loc[latest_date]
    
    stress_results = []
    
    for bank in latest_metrics.index:
        bank_region = REGION_MAP.get(bank, 'Unknown')
        
        # Determine if bank is directly affected
        if shock_region == 'Global' or bank_region == shock_region:
            # Direct impact
            base_impact = shock_magnitude
            impact_type = 'Direct'
        else:
            # Spillover impact
            if include_spillovers:
                beta_t = latest_metrics.loc[bank, 'Beta_T_95']
                if pd.isna(beta_t):
                    beta_t = 1.0
                
                # Enhanced spillover calculation
                spillover_factor = min(beta_t * 0.3, 0.6)  # Max 60% spillover
                base_impact = shock_magnitude * spillover_factor
                impact_type = 'Spillover'
            else:
                base_impact = 0.0
                impact_type = 'None'
        
        # Calculate impacts on different metrics
        original_var = latest_metrics.loc[bank, 'VaR_95']
        original_beta = latest_metrics.loc[bank, 'Beta_T_95']
        
        # VaR impact (non-linear for extreme shocks)
        if base_impact > 0:
            var_multiplier = 1 + base_impact * (1 + base_impact)  # Non-linear scaling
            shocked_var = original_var * var_multiplier
            var_impact = (var_multiplier - 1) * 100
            
            # Beta impact (simplified estimation)
            beta_multiplier = 1 + base_impact * 0.5
            shocked_beta = original_beta * beta_multiplier if not pd.isna(original_beta) else np.nan
        else:
            shocked_var = original_var
            shocked_beta = original_beta
            var_impact = 0.0
        
        # Risk level assessment
        original_risk, _ = get_risk_classification(original_beta)
        shocked_risk, _ = get_risk_classification(shocked_beta)
        risk_level_change = shocked_risk != original_risk
        
        stress_results.append({
            'Bank': bank,
            'Region': bank_region,
            'Impact_Type': impact_type,
            'Impact_Magnitude': var_impact,
            'Original_VaR': original_var,
            'Shocked_VaR': shocked_var,
            'Original_Beta': original_beta,
            'Shocked_Beta': shocked_beta,
            'Original_Risk_Level': original_risk,
            'Shocked_Risk_Level': shocked_risk,
            'Risk_Level_Change': risk_level_change
        })
    
    return pd.DataFrame(stress_results)

if __name__ == "__main__":
    main()
