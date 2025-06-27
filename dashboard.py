import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our data processor
from data_processor import BankingDataProcessor, process_banking_data

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
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: #ffffff;
        border: 2px solid #e1e5e9;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.1);
    }
    .region-americas { 
        color: #2e7d32; 
        font-weight: 600;
    }
    .region-europe { 
        color: #1976d2; 
        font-weight: 600;
    }
    .region-asia { 
        color: #f57c00; 
        font-weight: 600;
    }
    .high-risk { 
        color: #d32f2f; 
        font-weight: 600;
        background-color: #fff5f5;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
    }
    .medium-risk { 
        color: #f57c00; 
        font-weight: 600;
        background-color: #fffbf0;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
    }
    .low-risk { 
        color: #388e3c; 
        font-weight: 600;
        background-color: #f0fff4;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
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
    
    /* Improve slider styling */
    .stSlider {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Better selectbox styling */
    .stSelectbox {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Better multiselect styling */
    .stMultiSelect {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Better date input styling */
    .stDateInput {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Improve table styling */
    .dataframe {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Better tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem 0.5rem 0 0;
        color: #555;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
        border-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processor' not in st.session_state:
    st.session_state.processor = None

# Header
st.markdown('<h1 class="main-header">📊 Systemic Risk Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Real-time banking systemic risk analysis using Extreme Value Theory**")

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Date range selection
    st.subheader("📅 Date Range")
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime('2010-01-01').date(),
        min_value=pd.to_datetime('2000-01-01').date(),
        max_value=pd.to_datetime('2024-12-31').date()
    )
    
    end_date = st.date_input(
        "End Date",
        value=pd.to_datetime('2024-12-31').date(),
        min_value=pd.to_datetime('2000-01-01').date(),
        max_value=pd.to_datetime('2024-12-31').date()
    )
    
    # Bank selection
    st.subheader("🏦 Bank Selection")
    
    # Initialize processor to get available banks
    if st.session_state.processor is None:
        temp_processor = BankingDataProcessor()
        available_banks = temp_processor.get_available_banks()
        banks_by_region = temp_processor.get_banks_by_region()
    else:
        available_banks = st.session_state.processor.get_available_banks()
        banks_by_region = st.session_state.processor.get_banks_by_region()
    
    # Bank selection by region
    selected_banks = []
    
    # Americas
    st.markdown("**🌎 Americas**")
    americas_banks = banks_by_region.get('Americas', [])
    americas_selected = st.multiselect(
        "Select American banks:",
        americas_banks,
        default=americas_banks[:3] if len(americas_banks) >= 3 else americas_banks,
        key="americas"
    )
    selected_banks.extend(americas_selected)
    
    # Europe
    st.markdown("**🇪🇺 Europe**")
    europe_banks = banks_by_region.get('Europe', [])
    europe_selected = st.multiselect(
        "Select European banks:",
        europe_banks,
        default=europe_banks[:3] if len(europe_banks) >= 3 else europe_banks,
        key="europe"
    )
    selected_banks.extend(europe_selected)
    
    # Asia/Pacific
    st.markdown("**🌏 Asia/Pacific**")
    asia_banks = banks_by_region.get('Asia/Pacific', [])
    asia_selected = st.multiselect(
        "Select Asia/Pacific banks:",
        asia_banks,
        default=asia_banks[:3] if len(asia_banks) >= 3 else asia_banks,
        key="asia"
    )
    selected_banks.extend(asia_selected)
    
    # Confidence level
    st.subheader("📈 Confidence Level")
    confidence_level = st.selectbox(
        "Select confidence level:",
        [0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%"
    )
    
    # Load data button
    st.subheader("🔄 Data Loading")
    if st.button("Load Data", type="primary"):
        if selected_banks:
            with st.spinner("Downloading and processing data..."):
                try:
                    st.session_state.processor = process_banking_data(
                        selected_banks, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    st.session_state.data_loaded = True
                    st.success(f"Data loaded successfully for {len(selected_banks)} banks!")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        else:
            st.warning("Please select at least one bank.")

# Main content
if st.session_state.data_loaded and st.session_state.processor is not None:
    processor = st.session_state.processor
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🏦 Bank Analysis", 
        "📈 Time Series", 
        "🌍 Regional Analysis",
        "⚠️ Risk Alerts"
    ])
    
    with tab1:
        st.header("📊 Systemic Risk Overview")
        
        # Get latest metrics
        latest_metrics = processor.get_latest_metrics(confidence_level)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_beta = latest_metrics['Beta_T'].mean()
            st.metric(
                "Average Systemic Beta", 
                f"{avg_beta:.3f}",
                delta=f"{avg_beta - 1:.3f} vs baseline"
            )
        
        with col2:
            max_beta = latest_metrics['Beta_T'].max()
            st.metric(
                "Maximum Systemic Beta", 
                f"{max_beta:.3f}",
                delta="Highest risk bank"
            )
        
        with col3:
            avg_var = latest_metrics[f'VaR_{int(confidence_level*100)}'].mean()
            st.metric(
                f"Average VaR ({int(confidence_level*100)}%)", 
                f"{avg_var:.3f}",
                delta="Portfolio risk"
            )
        
        with col4:
            avg_tail_dep = latest_metrics[f'Tau_{int(confidence_level*100)}'].mean()
            st.metric(
                f"Average Tail Dependence ({int(confidence_level*100)}%)", 
                f"{avg_tail_dep:.3f}",
                delta="Systemic correlation"
            )
        
        # Latest metrics table
        st.subheader("Latest Risk Metrics")
        
        # Color code by region
        def color_region(val):
            if 'Americas' in str(val):
                return 'background-color: #e8f5e8'
            elif 'Europe' in str(val):
                return 'background-color: #e3f2fd'
            elif 'Asia' in str(val):
                return 'background-color: #fff3e0'
            return ''
        
        display_metrics = latest_metrics[['Bank', 'Region', 'Beta_T', f'VaR_{int(confidence_level*100)}', 
                                        f'Hill_{int(confidence_level*100)}', f'Tau_{int(confidence_level*100)}']].copy()
        display_metrics = display_metrics.round(4)
        
        st.dataframe(
            display_metrics.style.applymap(color_region, subset=['Region']),
            use_container_width=True
        )
    
    with tab2:
        st.header("🏦 Individual Bank Analysis")
        
        # Bank selector
        available_banks = processor.get_available_banks()
        selected_bank = st.selectbox("Select a bank:", available_banks)
        
        if selected_bank:
            # Get time series for selected bank
            beta_series = processor.get_bank_time_series(selected_bank, 'Beta_T', confidence_level)
            var_series = processor.get_bank_time_series(selected_bank, f'VaR_{int(confidence_level*100)}', confidence_level)
            tail_dep_series = processor.get_bank_time_series(selected_bank, f'Tau_{int(confidence_level*100)}', confidence_level)
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Systemic Beta', f'VaR ({int(confidence_level*100)}%)', f'Tail Dependence ({int(confidence_level*100)}%)'),
                vertical_spacing=0.1
            )
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=beta_series.index, y=beta_series.values, name='Systemic Beta', line=dict(color='red')),
                row=1, col=1
            )
            fig.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=var_series.index, y=var_series.values, name='VaR', line=dict(color='blue')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=tail_dep_series.index, y=tail_dep_series.values, name='Tail Dependence', line=dict(color='green')),
                row=3, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Bank statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Statistics")
                stats_data = {
                    'Metric': ['Current Beta', 'Max Beta', 'Min Beta', 'Beta Volatility'],
                    'Value': [
                        f"{beta_series.iloc[-1]:.3f}",
                        f"{beta_series.max():.3f}",
                        f"{beta_series.min():.3f}",
                        f"{beta_series.std():.3f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            with col2:
                st.subheader("Risk Assessment")
                current_beta = beta_series.iloc[-1]
                if current_beta > 2.0:
                    risk_level = "🔴 High Risk"
                    risk_color = "high-risk"
                elif current_beta > 1.5:
                    risk_level = "🟡 Medium Risk"
                    risk_color = "medium-risk"
                else:
                    risk_level = "🟢 Low Risk"
                    risk_color = "low-risk"
                
                st.markdown(f"<div class='metric-card'><h3 class='{risk_color}'>{risk_level}</h3></div>", unsafe_allow_html=True)
    
    with tab3:
        st.header("📈 Time Series Analysis")
        
        # Metric selector
        metric_options = ['Beta_T', f'VaR_{int(confidence_level*100)}', f'Hill_{int(confidence_level*100)}', f'Tau_{int(confidence_level*100)}']
        selected_metric = st.selectbox("Select metric:", metric_options)
        
        # Get all banks' time series for selected metric
        all_series = {}
        for bank in processor.get_available_banks():
            try:
                series = processor.get_bank_time_series(bank, selected_metric, confidence_level)
                all_series[bank] = series
            except:
                continue
        
        if all_series:
            # Create time series plot
            fig = go.Figure()
            
            for bank, series in all_series.items():
                region = processor.region_map.get(bank, 'Unknown')
                if 'Americas' in region:
                    color = '#2e7d32'
                elif 'Europe' in region:
                    color = '#1976d2'
                else:
                    color = '#f57c00'
                
                fig.add_trace(go.Scatter(
                    x=series.index, 
                    y=series.values, 
                    name=bank,
                    line=dict(color=color),
                    hovertemplate=f'{bank}<br>Date: %{{x}}<br>{selected_metric}: %{{y:.3f}}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f"{selected_metric} Over Time",
                xaxis_title="Date",
                yaxis_title=selected_metric,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🌍 Regional Analysis")
        
        # Get summary statistics by region
        summary_stats = processor.get_summary_statistics(confidence_level)
        
        if not summary_stats.empty:
            # Regional comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Regional Systemic Beta Comparison")
                
                # Extract mean Beta_T by region
                beta_by_region = summary_stats[('Beta_T', 'mean')].reset_index()
                beta_by_region.columns = ['Region', 'Mean Beta']
                
                fig = px.bar(
                    beta_by_region, 
                    x='Region', 
                    y='Mean Beta',
                    color='Region',
                    color_discrete_map={
                        'Americas': '#2e7d32',
                        'Europe': '#1976d2',
                        'Asia/Pacific': '#f57c00'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Regional Risk Statistics")
                st.dataframe(summary_stats.round(4), use_container_width=True)
    
    with tab5:
        st.header("⚠️ Risk Alerts")
        
        # Get latest metrics
        latest_metrics = processor.get_latest_metrics(confidence_level)
        
        # Define risk thresholds
        high_risk_threshold = 2.0
        medium_risk_threshold = 1.5
        
        # Identify high-risk banks
        high_risk_banks = latest_metrics[latest_metrics['Beta_T'] > high_risk_threshold]
        medium_risk_banks = latest_metrics[
            (latest_metrics['Beta_T'] > medium_risk_threshold) & 
            (latest_metrics['Beta_T'] <= high_risk_threshold)
        ]
        
        # Display alerts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 High Risk Banks")
            if not high_risk_banks.empty:
                for _, bank in high_risk_banks.iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 class="high-risk">{bank['Bank']}</h4>
                        <p>Systemic Beta: <strong>{bank['Beta_T']:.3f}</strong></p>
                        <p>Region: {bank['Region']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No high-risk banks detected!")
        
        with col2:
            st.subheader("🟡 Medium Risk Banks")
            if not medium_risk_banks.empty:
                for _, bank in medium_risk_banks.iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 class="medium-risk">{bank['Bank']}</h4>
                        <p>Systemic Beta: <strong>{bank['Beta_T']:.3f}</strong></p>
                        <p>Region: {bank['Region']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No medium-risk banks detected!")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to the Systemic Risk Dashboard</h2>
        <p>This dashboard provides real-time analysis of banking systemic risk using Extreme Value Theory.</p>
        <br>
        <h3>📋 How to get started:</h3>
        <ol style="text-align: left; max-width: 600px; margin: 0 auto;">
            <li>Select your desired date range in the sidebar</li>
            <li>Choose banks from different regions (Americas, Europe, Asia/Pacific)</li>
            <li>Select your preferred confidence level (95% or 99%)</li>
            <li>Click "Load Data" to download and process the data</li>
            <li>Explore the different tabs for comprehensive risk analysis</li>
        </ol>
        <br>
        <p><strong>Note:</strong> Data loading may take a few minutes depending on the number of selected banks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show available banks info
    temp_processor = BankingDataProcessor()
    banks_by_region = temp_processor.get_banks_by_region()
    
    st.subheader("🏦 Available Banks by Region")
    
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
