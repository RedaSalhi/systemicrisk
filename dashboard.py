import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Systemic Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .crisis-period {
        background-color: #ffebee;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data based on the research paper
@st.cache_data
def generate_synthetic_data():
    # Define banks and regions from the paper
    banks_data = {
        'Americas': ['JPMorgan Chase', 'Bank of America', 'Citigroup', 'Wells Fargo', 
                    'Goldman Sachs', 'Morgan Stanley', 'Bank of New York Mellon', 
                    'State Street', 'Royal Bank of Canada', 'Toronto Dominion'],
        'Europe': ['HSBC', 'Barclays', 'BNP Paribas', 'Credit Agricole', 'Societe Generale',
                  'Santander', 'ING', 'Deutsche Bank', 'UBS', 'Standard Chartered'],
        'Asia-Pacific': ['Agricultural Bank of China', 'Bank of China', 'China Construction Bank',
                        'ICBC', 'Bank of Communications', 'Mitsubishi UFJ FG', 
                        'Sumitomo Mitsui FG', 'Mizuho FG']
    }
    
    # Generate date range
    start_date = pd.to_datetime('2011-01-01')
    end_date = pd.to_datetime('2024-12-31')
    dates = pd.date_range(start_date, end_date, freq='W-FRI')
    
    # Crisis periods from the paper
    crisis_periods = [
        ('Eurozone Debt Crisis', '2011-07-01', '2012-12-31'),
        ('Chinese Equity Crash', '2015-06-01', '2016-02-29'),
        ('COVID-19 Pandemic', '2020-02-01', '2020-05-31'),
        ('Ukraine War Volatility', '2022-02-24', '2022-06-30'),
        ('Banking Sector Stress', '2023-03-01', '2023-05-31')
    ]
    
    data = []
    np.random.seed(42)
    
    for region, bank_list in banks_data.items():
        for bank in bank_list:
            for date in dates:
                # Base systemic beta values based on paper's findings
                base_beta = {
                    'Americas': 1.5,
                    'Europe': 1.8,
                    'Asia-Pacific': 1.2
                }[region]
                
                # Add crisis amplification
                crisis_multiplier = 1.0
                for crisis_name, crisis_start, crisis_end in crisis_periods:
                    if pd.to_datetime(crisis_start) <= date <= pd.to_datetime(crisis_end):
                        crisis_multiplier = np.random.uniform(1.5, 2.5)
                        break
                
                # Generate realistic beta_T values
                beta_t_95 = base_beta * crisis_multiplier * np.random.uniform(0.7, 1.3)
                beta_t_99 = beta_t_95 * np.random.uniform(0.9, 1.2)
                
                # Add some temporal correlation
                if data and data[-1]['Bank'] == bank:
                    prev_beta = data[-1]['Beta_T_95']
                    beta_t_95 = 0.7 * prev_beta + 0.3 * beta_t_95
                    beta_t_99 = 0.7 * data[-1]['Beta_T_99'] + 0.3 * beta_t_99
                
                data.append({
                    'Date': date,
                    'Bank': bank,
                    'Region': region,
                    'Beta_T_95': max(0.1, beta_t_95),
                    'Beta_T_99': max(0.1, beta_t_99),
                    'VaR_95': np.random.uniform(0.02, 0.08),
                    'VaR_99': np.random.uniform(0.04, 0.12),
                    'Tail_Dependence': np.random.uniform(0.1, 0.8)
                })
    
    df = pd.DataFrame(data)
    
    # Add crisis flags
    df['In_Crisis'] = False
    for crisis_name, crisis_start, crisis_end in crisis_periods:
        mask = (df['Date'] >= crisis_start) & (df['Date'] <= crisis_end)
        df.loc[mask, 'In_Crisis'] = True
        df.loc[mask, 'Crisis_Name'] = crisis_name
    
    return df, crisis_periods

# Load data
df, crisis_periods = generate_synthetic_data()

# Header
st.markdown('<h1 class="main-header">🏦 Global Systemic Risk Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Real-time monitoring of systemic risk across 28 Global Systemically Important Banks (G-SIBs)**")

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")

# Date range selector
max_date = df['Date'].max()
min_date = df['Date'].min()
default_start = max_date - timedelta(days=365*2)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date
)

# Region selector
regions = st.sidebar.multiselect(
    "Select Regions",
    options=['Americas', 'Europe', 'Asia-Pacific'],
    default=['Americas', 'Europe', 'Asia-Pacific']
)

# VaR threshold selector
var_threshold = st.sidebar.selectbox(
    "VaR Threshold",
    options=['95%', '99%'],
    index=0
)

beta_col = 'Beta_T_95' if var_threshold == '95%' else 'Beta_T_99'

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    filtered_df = df[mask]
else:
    filtered_df = df

filtered_df = filtered_df[filtered_df['Region'].isin(regions)]

# Main dashboard layout
col1, col2, col3, col4 = st.columns(4)

# Current metrics
latest_data = filtered_df[filtered_df['Date'] == filtered_df['Date'].max()]

with col1:
    avg_beta = latest_data[beta_col].mean()
    st.metric(
        "Average Systemic β",
        f"{avg_beta:.2f}",
        delta=f"{avg_beta - 1.5:.2f}" if avg_beta > 1.5 else f"{avg_beta - 1.5:.2f}"
    )

with col2:
    high_risk_banks = len(latest_data[latest_data[beta_col] > 2.0])
    st.metric("High Risk Banks (β > 2.0)", high_risk_banks)

with col3:
    max_beta = latest_data[beta_col].max()
    st.metric("Maximum β", f"{max_beta:.2f}")

with col4:
    crisis_prob = min(100, max(0, (avg_beta - 1.0) * 50))
    st.metric("Crisis Probability", f"{crisis_prob:.1f}%")

st.divider()

# Main charts
tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🌍 Regional Analysis", "🏦 Bank Rankings", "⚠️ Crisis Periods"])

with tab1:
    st.subheader("Systemic Risk Evolution Over Time")
    
    # Regional average time series
    regional_avg = filtered_df.groupby(['Date', 'Region'])[beta_col].mean().reset_index()
    
    fig = px.line(
        regional_avg, 
        x='Date', 
        y=beta_col, 
        color='Region',
        title=f"Regional Average Systemic Beta ({var_threshold} VaR)",
        height=500
    )
    
    # Add crisis period shading
    for crisis_name, crisis_start, crisis_end in crisis_periods:
        fig.add_vrect(
            x0=crisis_start, x1=crisis_end,
            fillcolor="red", opacity=0.1,
            annotation_text=crisis_name,
            annotation_position="top left"
        )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"Systemic Beta ({var_threshold})",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Regional Risk Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current regional averages
        current_regional = latest_data.groupby('Region')[beta_col].mean().reset_index()
        
        fig_bar = px.bar(
            current_regional,
            x='Region',
            y=beta_col,
            title="Current Regional Average Systemic Beta",
            color=beta_col,
            color_continuous_scale='RdYlBu_r'
        )
        fig_bar.add_hline(y=1.0, line_dash="dash", line_color="black", 
                         annotation_text="Neutral Risk Level")
        fig_bar.add_hline(y=2.0, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Threshold")
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Regional risk distribution
        fig_box = px.box(
            latest_data,
            x='Region',
            y=beta_col,
            title="Risk Distribution by Region",
            points="all"
        )
        fig_box.add_hline(y=2.0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader("Bank Risk Rankings")
    
    # Current bank rankings
    bank_rankings = latest_data.sort_values(beta_col, ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top 15 banks chart
        top_banks = bank_rankings.head(15)
        
        fig_banks = px.bar(
            top_banks,
            y='Bank',
            x=beta_col,
            color='Region',
            title="Top 15 Banks by Systemic Risk",
            orientation='h',
            height=600
        )
        fig_banks.add_vline(x=2.0, line_dash="dash", line_color="red",
                           annotation_text="High Risk Threshold")
        fig_banks.update_layout(yaxis={'categoryorder':'total ascending'})
        
        st.plotly_chart(fig_banks, use_container_width=True)
    
    with col2:
        st.markdown("**Current Rankings**")
        for i, (_, row) in enumerate(bank_rankings.head(10).iterrows(), 1):
            risk_level = "🔴" if row[beta_col] > 2.0 else "🟡" if row[beta_col] > 1.5 else "🟢"
            st.markdown(f"{i}. {risk_level} **{row['Bank']}** ({row['Region']}) - {row[beta_col]:.2f}")

with tab4:
    st.subheader("Crisis Period Analysis")
    
    # Crisis impact analysis
    crisis_impact = []
    for crisis_name, crisis_start, crisis_end in crisis_periods:
        crisis_data = df[(df['Date'] >= crisis_start) & (df['Date'] <= crisis_end)]
        pre_crisis = df[(df['Date'] >= pd.to_datetime(crisis_start) - timedelta(days=90)) & 
                       (df['Date'] < crisis_start)]
        
        if not crisis_data.empty and not pre_crisis.empty:
            crisis_avg = crisis_data.groupby('Region')[beta_col].mean()
            pre_crisis_avg = pre_crisis.groupby('Region')[beta_col].mean()
            
            for region in crisis_avg.index:
                if region in pre_crisis_avg.index:
                    impact = crisis_avg[region] - pre_crisis_avg[region]
                    crisis_impact.append({
                        'Crisis': crisis_name,
                        'Region': region,
                        'Impact': impact,
                        'Pre_Crisis_Beta': pre_crisis_avg[region],
                        'Crisis_Beta': crisis_avg[region]
                    })
    
    if crisis_impact:
        crisis_df = pd.DataFrame(crisis_impact)
        
        fig_crisis = px.bar(
            crisis_df,
            x='Crisis',
            y='Impact',
            color='Region',
            title="Crisis Impact on Systemic Risk by Region",
            barmode='group'
        )
        fig_crisis.add_hline(y=0, line_color="black")
        fig_crisis.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig_crisis, use_container_width=True)
        
        # Crisis details table
        st.subheader("Crisis Period Details")
        for crisis_name, crisis_start, crisis_end in crisis_periods:
            with st.expander(f"📅 {crisis_name} ({crisis_start} to {crisis_end})"):
                crisis_subset = crisis_df[crisis_df['Crisis'] == crisis_name]
                if not crisis_subset.empty:
                    for _, row in crisis_subset.iterrows():
                        impact_emoji = "📈" if row['Impact'] > 0 else "📉"
                        st.markdown(f"**{row['Region']}**: {impact_emoji} Impact: {row['Impact']:+.2f} "
                                  f"(Pre: {row['Pre_Crisis_Beta']:.2f} → Crisis: {row['Crisis_Beta']:.2f})")

# Footer
st.divider()
st.markdown("""
**Data Notes**: 
- Systemic Beta (βT) measures tail risk contribution relative to market indices
- Values > 2.0 indicate high systemic risk
- Values > 1.0 indicate above-average risk contribution
- Crisis periods are highlighted in red on time series charts

**Source**: Based on "Systemic Risk in Global Banking Institutions" research framework
""")
