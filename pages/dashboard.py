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

# Add parent directory to path to import data_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import (
    download_data, prepare_returns, compute_rolling_metrics, 
    stress_test_spillover, REGION_MAP, INDEX_MAP, IDX_NAME_MAP
)

st.set_page_config(
    page_title="Systemic Risk Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #22C55E;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process all data with caching"""
    with st.spinner("📊 Downloading bank and index data..."):
        banks_data, indices_data = download_data()
    
    with st.spinner("🔄 Computing weekly returns..."):
        combined_data, bank_names = prepare_returns(banks_data, indices_data)
    
    with st.spinner("📈 Computing rolling systemic risk metrics..."):
        metrics_95, metrics_99 = compute_rolling_metrics(combined_data, bank_names)
    
    return combined_data, bank_names, metrics_95, metrics_99

def main():
    st.title("📈 Systemic Risk Dashboard")
    st.markdown("**Interactive analysis of G-SIB systemic risk metrics (2011-2024)**")
    
    # Load data
    try:
        combined_data, bank_names, metrics_95, metrics_99 = load_and_process_data()
        st.success(f"✅ Data loaded: {len(bank_names)} banks, {len(combined_data)} weeks")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(combined_data.index.min().date(), combined_data.index.max().date()),
        min_value=combined_data.index.min().date(),
        max_value=combined_data.index.max().date()
    )
    
    # Convert to timestamp for filtering
    if len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered_95 = metrics_95[(metrics_95.index.get_level_values(0) >= start_date) & 
                                 (metrics_95.index.get_level_values(0) <= end_date)]
        filtered_99 = metrics_99[(metrics_99.index.get_level_values(0) >= start_date) & 
                                 (metrics_99.index.get_level_values(0) <= end_date)]
    else:
        filtered_95, filtered_99 = metrics_95, metrics_99
    
    # Bank/Region selector
    analysis_type = st.sidebar.radio(
        "📊 Analysis Type",
        ["Individual Banks", "Regional Analysis", "Stress Testing"]
    )
    
    if analysis_type == "Individual Banks":
        selected_banks = st.sidebar.multiselect(
            "🏦 Select Banks",
            options=bank_names,
            default=bank_names[:5]
        )
    else:
        selected_banks = bank_names
    
    # VaR confidence level
    confidence_level = st.sidebar.selectbox(
        "📊 VaR Confidence Level",
        options=["95%", "99%"],
        index=0
    )
    
    metrics_data = filtered_95 if confidence_level == "95%" else filtered_99
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Overview", "📈 Time Series", "🔥 Heatmaps", "⚠️ Stress Testing"])
    
    with tab1:
        show_overview(metrics_data, selected_banks, confidence_level)
    
    with tab2:
        show_time_series(metrics_data, selected_banks, confidence_level)
    
    with tab3:
        show_heatmaps(metrics_data, confidence_level)
    
    with tab4:
        show_stress_testing(metrics_data, combined_data, bank_names)

def show_overview(metrics_data, selected_banks, confidence_level):
    """Overview section with key metrics"""
    
    st.subheader(f"🎯 Current Risk Overview ({confidence_level} VaR)")
    
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
        st.markdown('<div class="warning-card">' if high_risk_banks > 5 else '<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High Risk Banks (β > 2.0)", f"{high_risk_banks}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_var = latest_metrics['VaR_95' if confidence_level == "95%" else 'VaR_99'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average VaR", f"{avg_var:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_tau = latest_metrics['Tau_95' if confidence_level == "95%" else 'Tau_99'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Tail Dependence", f"{avg_tau:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
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
        
        regional_df = pd.DataFrame(regional_data)
        
        if not regional_df.empty:
            regional_summary = regional_df.groupby('Region')['Beta_T'].agg(['mean', 'std', 'count'])
            st.dataframe(regional_summary.round(3))
    
    with col2:
        st.subheader("🏆 Top 10 Highest Risk Banks")
        
        top_risk = latest_metrics.nlargest(10, 'Beta_T')[['Beta_T', 'VaR_95' if confidence_level == "95%" else 'VaR_99']]
        top_risk['Region'] = top_risk.index.map(REGION_MAP)
        st.dataframe(top_risk.round(4))

def show_time_series(metrics_data, selected_banks, confidence_level):
    """Time series visualization"""
    
    st.subheader(f"📈 Time Series Analysis ({confidence_level} VaR)")
    
    # Systemic Beta time series
    st.markdown("#### Systemic Beta (βT) Evolution")
    
    beta_data = []
    for bank in selected_banks:
        bank_data = metrics_data.xs(bank, level=1)
        for date, row in bank_data.iterrows():
            beta_data.append({
                'Date': date,
                'Bank': bank,
                'Beta_T': row['Beta_T'],
                'Region': REGION_MAP.get(bank, 'Unknown')
            })
    
    beta_df = pd.DataFrame(beta_data)
    
    if not beta_df.empty:
        fig = px.line(beta_df, x='Date', y='Beta_T', color='Bank', 
                     title=f"Systemic Beta Evolution ({confidence_level} VaR)",
                     labels={'Beta_T': 'Systemic Beta (βT)'})
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold (β=2.0)")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # VaR and Tail Dependence
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Value-at-Risk Evolution")
        var_col = 'VaR_95' if confidence_level == "95%" else 'VaR_99'
        
        var_data = []
        for bank in selected_banks[:5]:  # Limit to 5 banks for clarity
            bank_data = metrics_data.xs(bank, level=1)
            for date, row in bank_data.iterrows():
                var_data.append({
                    'Date': date,
                    'Bank': bank,
                    'VaR': row[var_col]
                })
        
        var_df = pd.DataFrame(var_data)
        if not var_df.empty:
            fig = px.line(var_df, x='Date', y='VaR', color='Bank',
                         title=f"Value-at-Risk ({confidence_level})")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Tail Dependence Evolution")
        tau_col = 'Tau_95' if confidence_level == "95%" else 'Tau_99'
        
        tau_data = []
        for bank in selected_banks[:5]:  # Limit to 5 banks for clarity
            bank_data = metrics_data.xs(bank, level=1)
            for date, row in bank_data.iterrows():
                tau_data.append({
                    'Date': date,
                    'Bank': bank,
                    'Tau': row[tau_col]
                })
        
        tau_df = pd.DataFrame(tau_data)
        if not tau_df.empty:
            fig = px.line(tau_df, x='Date', y='Tau', color='Bank',
                         title=f"Tail Dependence ({confidence_level})")
            st.plotly_chart(fig, use_container_width=True)

def show_heatmaps(metrics_data, confidence_level):
    """Correlation and risk heatmaps"""
    
    st.subheader(f"🔥 Risk Correlation Analysis ({confidence_level} VaR)")
    
    # Get latest 12 weeks of data for correlation
    latest_date = metrics_data.index.get_level_values(0).max()
    recent_dates = metrics_data.index.get_level_values(0).unique()
    recent_dates = recent_dates[recent_dates >= (latest_date - pd.Timedelta(weeks=12))]
    
    recent_data = metrics_data[metrics_data.index.get_level_values(0).isin(recent_dates)]
    
    # Systemic Beta correlation matrix
    beta_pivot = recent_data.reset_index().pivot(index='Date', columns='Bank', values='Beta_T')
    beta_corr = beta_pivot.corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Inter-Bank Beta Correlations")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(beta_corr, dtype=bool))
        sns.heatmap(beta_corr, mask=mask, annot=False, cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5, ax=ax)
        plt.title(f"Systemic Beta Correlations ({confidence_level} VaR)")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### Regional Beta Correlations")
        
        # Calculate regional averages
        regional_betas = {}
        for region in ['Americas', 'Europe', 'Asia/Pacific']:
            region_banks = [bank for bank, r in REGION_MAP.items() if r == region]
            region_banks = [bank for bank in region_banks if bank in beta_pivot.columns]
            if region_banks:
                regional_betas[region] = beta_pivot[region_banks].mean(axis=1)
        
        regional_df = pd.DataFrame(regional_betas)
        regional_corr = regional_df.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(regional_corr, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, linewidths=2, ax=ax, cbar_kws={"shrink": .8})
        plt.title(f"Regional Beta Correlations ({confidence_level} VaR)")
        st.pyplot(fig)
    
    # Risk level heatmap
    st.markdown("#### Current Risk Level Matrix")
    
    latest_metrics = metrics_data.loc[latest_date]
    risk_matrix = []
    
    for bank in latest_metrics.index:
        region = REGION_MAP.get(bank, 'Unknown')
        beta = latest_metrics.loc[bank, 'Beta_T']
        var_val = latest_metrics.loc[bank, 'VaR_95' if confidence_level == "95%" else 'VaR_99']
        tau = latest_metrics.loc[bank, 'Tau_95' if confidence_level == "95%" else 'Tau_99']
        
        # Risk scoring
        beta_risk = 'High' if beta > 2.0 else 'Medium' if beta > 1.0 else 'Low'
        var_risk = 'High' if var_val < -0.08 else 'Medium' if var_val < -0.04 else 'Low'
        tau_risk = 'High' if tau > 0.7 else 'Medium' if tau > 0.4 else 'Low'
        
        risk_matrix.append({
            'Bank': bank,
            'Region': region,
            'Beta Risk': beta_risk,
            'VaR Risk': var_risk,
            'Tau Risk': tau_risk,
            'Beta Value': beta,
            'VaR Value': var_val,
            'Tau Value': tau
        })
    
    risk_df = pd.DataFrame(risk_matrix)
    
    # Create a sortable table with color coding
    st.dataframe(
        risk_df[['Bank', 'Region', 'Beta Risk', 'VaR Risk', 'Tau Risk', 'Beta Value', 'VaR Value', 'Tau Value']].round(4),
        use_container_width=True
    )

def show_stress_testing(metrics_data, combined_data, bank_names):
    """Stress testing interface"""
    
    st.subheader("⚠️ Spillover-Aware Stress Testing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Stress Test Parameters")
        
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
        
        scenario = st.selectbox(
            "Stress Scenario",
            options=[
                "Regional Banking Crisis",
                "Global Financial Crisis", 
                "Sovereign Debt Crisis",
                "Market Crash",
                "Custom Shock"
            ]
        )
        
        if st.button("🚀 Run Stress Test"):
            # Determine affected banks
            if shock_region == 'Global':
                affected_banks = bank_names
            else:
                affected_banks = [bank for bank, region in REGION_MAP.items() 
                                if region == shock_region and bank in bank_names]
            
            # Run stress test
            stress_results = stress_test_spillover(
                metrics_data, 
                shock_magnitude=shock_magnitude,
                affected_banks=affected_banks
            )
            
            # Store in session state
            st.session_state.stress_results = stress_results
            st.session_state.stress_params = {
                'magnitude': shock_magnitude * 100,
                'region': shock_region,
                'scenario': scenario
            }
    
    with col2:
        if 'stress_results' in st.session_state:
            st.markdown("#### Stress Test Results")
            
            results = st.session_state.stress_results
            params = st.session_state.stress_params
            
            st.info(f"**Scenario**: {params['scenario']} | **Shock**: {params['magnitude']:.0f}% | **Region**: {params['region']}")
            
            # Summary metrics
            total_impact = results['Impact_Magnitude'].mean()
            max_impact = results['Impact_Magnitude'].max()
            high_impact_banks = (results['Impact_Magnitude'] > 15).sum()
            
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Average Impact", f"{total_impact:.1f}%")
            with col2b:
                st.metric("Maximum Impact", f"{max_impact:.1f}%")
            with col2c:
                st.metric("High Impact Banks", f"{high_impact_banks}")
            
            # Results visualization
            fig = px.bar(
                results.nlargest(15, 'Impact_Magnitude'), 
                x='Bank', 
                y='Impact_Magnitude',
                color='Impact_Type',
                title="Top 15 Banks by Impact Magnitude",
                labels={'Impact_Magnitude': 'Impact (%)'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.markdown("#### Detailed Results")
            display_cols = ['Bank', 'Region', 'Impact_Type', 'Impact_Magnitude', 'Systemic_Beta', 'Original_VaR', 'Shocked_VaR']
            st.dataframe(
                results[display_cols].round(4).sort_values('Impact_Magnitude', ascending=False),
                use_container_width=True
            )
        else:
            st.info("👆 Configure parameters and run a stress test to see results")

if __name__ == "__main__":
    main()
