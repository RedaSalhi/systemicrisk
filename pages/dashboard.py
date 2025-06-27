import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our fixed data processor
from data_processor import BankingDataProcessor, process_banking_data

# Page configuration
st.set_page_config(
    page_title="Systemic Risk Dashboard",
    page_icon="📊",
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
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #d97706;
        box-shadow: 0 4px 12px rgba(217, 119, 6, 0.15);
        transform: translateY(-1px);
    }
    
    /* Region colors matching Anthropic brand */
    .region-americas { 
        color: #059669; 
        font-weight: 600;
        background-color: #ecfdf5;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.875rem;
    }
    
    .region-europe { 
        color: #0284c7; 
        font-weight: 600;
        background-color: #e0f2fe;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.875rem;
    }
    
    .region-asia { 
        color: #dc2626; 
        font-weight: 600;
        background-color: #fef2f2;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.875rem;
    }
    
    /* Risk level indicators */
    .high-risk { 
        color: #dc2626; 
        font-weight: 600;
        background-color: #fef2f2;
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        border: 1px solid #fecaca;
    }
    
    .medium-risk { 
        color: #d97706; 
        font-weight: 600;
        background-color: #fffbeb;
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        border: 1px solid #fed7aa;
    }
    
    .low-risk { 
        color: #059669; 
        font-weight: 600;
        background-color: #ecfdf5;
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        border: 1px solid #a7f3d0;
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
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(217, 119, 6, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #b45309 0%, #d97706 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(217, 119, 6, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Metric displays */
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric > div > div:first-child {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .stMetric > div > div:nth-child(2) {
        color: #1a1a1a;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 8px;
    }
    
    .stMultiSelect > div > div {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 8px;
    }
    
    .stDateInput > div > div {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 8px;
    }
    
    /* Table styling */
    .dataframe {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe th {
        background-color: #f9fafb;
        color: #374151;
        font-weight: 600;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .dataframe td {
        color: #1a1a1a;
        border-bottom: 1px solid #f3f4f6;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f9fafb;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        color: #6b7280;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ffffff;
        color: #374151;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #d97706;
        border-color: #d97706;
        box-shadow: 0 1px 3px rgba(217, 119, 6, 0.1);
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #ecfdf5;
        border-color: #a7f3d0;
        color: #065f46;
    }
    
    /* Error message */
    .stError {
        background-color: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
    }
    
    /* Warning message */
    .stWarning {
        background-color: #fffbeb;
        border-color: #fed7aa;
        color: #92400e;
    }
    
    /* Info message */
    .stInfo {
        background-color: #eff6ff;
        border-color: #bfdbfe;
        color: #1e40af;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #d97706 transparent transparent transparent;
    }
    
    /* Plotly chart improvements */
    .js-plotly-plot {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
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
st.markdown('<p class="subtitle">Advanced banking systemic risk analysis using Extreme Value Theory</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Date range selection
    st.subheader("📅 Date Range")
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime('2015-01-01').date(),
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
        default=asia_banks[:2] if len(asia_banks) >= 2 else asia_banks,
        key="asia"
    )
    selected_banks.extend(asia_selected)
    
    # Confidence level
    st.subheader("📈 Analysis Parameters")
    confidence_level = st.selectbox(
        "Confidence level:",
        [0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%"
    )
    
    window_size = st.slider(
        "Rolling window size (weeks):",
        min_value=26, max_value=104, value=52, step=1,
        help="Number of weeks for rolling window analysis"
    )
    
    # Load data button
    st.subheader("🔄 Data Processing")
    if st.button("🚀 Load & Analyze Data", type="primary"):
        if selected_banks:
            with st.spinner("Downloading and processing data..."):
                try:
                    st.session_state.processor = process_banking_data(
                        selected_banks, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded successfully for {len(selected_banks)} banks!")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    st.info("💡 Try reducing the number of banks or adjusting the date range.")
        else:
            st.warning("⚠️ Please select at least one bank.")

# Main content
if st.session_state.data_loaded and st.session_state.processor is not None:
    processor = st.session_state.processor
    
    # Check if we have results
    try:
        latest_metrics = processor.get_latest_metrics(confidence_level)
        if latest_metrics.empty:
            st.error("❌ No data available for the selected parameters. Try different banks or date range.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error retrieving metrics: {str(e)}")
        st.stop()
    
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
        
        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_beta = latest_metrics['Beta_T'].mean()
            st.metric(
                "Average Systemic Beta", 
                f"{avg_beta:.3f}",
                delta=f"{avg_beta - 1:.3f}" if avg_beta > 1 else f"{avg_beta - 1:.3f}",
                help="Average systemic risk across all banks"
            )
        
        with col2:
            max_beta = latest_metrics['Beta_T'].max()
            max_bank = latest_metrics.loc[latest_metrics['Beta_T'].idxmax(), 'Bank']
            st.metric(
                "Highest Systemic Beta", 
                f"{max_beta:.3f}",
                delta=f"{max_bank}",
                help="Bank with highest systemic risk contribution"
            )
        
        with col3:
            var_col = f'VaR_{int(confidence_level*100)}'
            avg_var = latest_metrics[var_col].mean()
            st.metric(
                f"Average VaR ({int(confidence_level*100)}%)", 
                f"{avg_var:.4f}",
                help="Average Value-at-Risk across banks"
            )
        
        with col4:
            tau_col = f'Tau_{int(confidence_level*100)}'
            avg_tau = latest_metrics[tau_col].mean()
            st.metric(
                f"Average Tail Dependence", 
                f"{avg_tau:.3f}",
                help="Average systemic interconnectedness"
            )
        
        st.divider()
        
        # Risk distribution
        st.subheader("🎯 Risk Distribution")
        
        high_risk_count = len(latest_metrics[latest_metrics['Beta_T'] > 2.0])
        medium_risk_count = len(latest_metrics[
            (latest_metrics['Beta_T'] > 1.5) & (latest_metrics['Beta_T'] <= 2.0)
        ])
        low_risk_count = len(latest_metrics[latest_metrics['Beta_T'] <= 1.5])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 High Risk Banks", high_risk_count, help="Beta > 2.0")
        with col2:
            st.metric("🟡 Medium Risk Banks", medium_risk_count, help="1.5 < Beta ≤ 2.0")
        with col3:
            st.metric("🟢 Low Risk Banks", low_risk_count, help="Beta ≤ 1.5")
        
        # Latest metrics table
        st.subheader("📋 Latest Risk Metrics")
        
        # Prepare display data
        display_metrics = latest_metrics[['Bank', 'Region', 'Beta_T', var_col, 
                                        f'Hill_{int(confidence_level*100)}', tau_col]].copy()
        display_metrics = display_metrics.round(4)
        
        # Add risk level column
        def get_risk_level(beta):
            if beta > 2.0:
                return "🔴 High"
            elif beta > 1.5:
                return "🟡 Medium"
            else:
                return "🟢 Low"
        
        display_metrics['Risk Level'] = display_metrics['Beta_T'].apply(get_risk_level)
        
        # Color code by region
        def color_region(val):
            if 'Americas' in str(val):
                return 'background-color: #ecfdf5'
            elif 'Europe' in str(val):
                return 'background-color: #e0f2fe'
            elif 'Asia' in str(val):
                return 'background-color: #fef2f2'
            return ''
        
        st.dataframe(
            display_metrics.style.applymap(color_region, subset=['Region']),
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        st.header("🏦 Individual Bank Analysis")
        
        # Bank selector
        bank_names = latest_metrics['Bank'].unique()
        selected_bank = st.selectbox("Select a bank for detailed analysis:", bank_names)
        
        if selected_bank:
            # Get bank data
            bank_data = latest_metrics[latest_metrics['Bank'] == selected_bank].iloc[0]
            
            # Bank overview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📊 {selected_bank}")
                
                # Key metrics
                beta_val = bank_data['Beta_T']
                var_val = bank_data[var_col]
                tau_val = bank_data[tau_col]
                hill_val = bank_data[f'Hill_{int(confidence_level*100)}']
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Systemic Beta", f"{beta_val:.4f}")
                with metric_col2:
                    st.metric(f"VaR ({int(confidence_level*100)}%)", f"{var_val:.4f}")
                with metric_col3:
                    st.metric("Tail Dependence", f"{tau_val:.4f}")
                
                # Risk assessment
                if beta_val > 2.0:
                    risk_html = '<span class="high-risk">🔴 High Risk</span>'
                    risk_desc = "This bank poses significant systemic risk and requires enhanced monitoring."
                elif beta_val > 1.5:
                    risk_html = '<span class="medium-risk">🟡 Medium Risk</span>'
                    risk_desc = "This bank poses moderate systemic risk and should be monitored closely."
                else:
                    risk_html = '<span class="low-risk">🟢 Low Risk</span>'
                    risk_desc = "This bank poses low systemic risk under current conditions."
                
                st.markdown(f"**Risk Level:** {risk_html}", unsafe_allow_html=True)
                st.write(risk_desc)
            
            with col2:
                # Region info
                region = bank_data['Region']
                st.subheader("🌍 Regional Information")
                
                if region == 'Americas':
                    st.markdown('<span class="region-americas">🌎 Americas</span>', unsafe_allow_html=True)
                elif region == 'Europe':
                    st.markdown('<span class="region-europe">🇪🇺 Europe</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="region-asia">🌏 Asia/Pacific</span>', unsafe_allow_html=True)
                
                # Additional metrics
                st.metric("Hill Estimator", f"{hill_val:.4f}", help="Tail index estimate")
            
            # Time series analysis
            try:
                st.subheader("📈 Historical Trends")
                
                # Get time series data
                beta_series = processor.get_bank_time_series(selected_bank, 'Beta_T', confidence_level)
                var_series = processor.get_bank_time_series(selected_bank, var_col, confidence_level)
                tau_series = processor.get_bank_time_series(selected_bank, tau_col, confidence_level)
                
                # Create subplot
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Systemic Beta', f'VaR ({int(confidence_level*100)}%)', 'Tail Dependence'),
                    vertical_spacing=0.08,
                    shared_xaxes=True
                )
                
                # Add traces
                fig.add_trace(
                    go.Scatter(
                        x=beta_series.index, 
                        y=beta_series.values, 
                        name='Systemic Beta',
                        line=dict(color='#dc2626', width=2),
                        hovertemplate='Date: %{x}<br>Beta: %{y:.4f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=var_series.index, 
                        y=var_series.values, 
                        name='VaR',
                        line=dict(color='#0284c7', width=2),
                        hovertemplate='Date: %{x}<br>VaR: %{y:.4f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=tau_series.index, 
                        y=tau_series.values, 
                        name='Tail Dependence',
                        line=dict(color='#059669', width=2),
                        hovertemplate='Date: %{x}<br>Tau: %{y:.4f}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                # Add risk threshold lines
                fig.add_hline(y=2.0, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)
                fig.add_hline(y=1.5, line_dash="dash", line_color="orange", opacity=0.7, row=1, col=1)
                
                fig.update_layout(
                    height=600,
                    showlegend=False,
                    title_text=f"Risk Metrics Timeline - {selected_bank}",
                    plot_bgcolor='white'
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"⚠️ Could not load historical data: {str(e)}")
    
    with tab3:
        st.header("📈 Time Series Analysis")
        
        # Metric selector
        metric_options = {
            'Beta_T': 'Systemic Beta',
            var_col: f'VaR ({int(confidence_level*100)}%)',
            f'Hill_{int(confidence_level*100)}': f'Hill Estimator ({int(confidence_level*100)}%)',
            tau_col: f'Tail Dependence ({int(confidence_level*100)}%)'
        }
        
        selected_metric = st.selectbox(
            "Select metric to analyze:",
            list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )
        
        # Get all banks' time series for selected metric
        all_series = {}
        for bank in bank_names:
            try:
                series = processor.get_bank_time_series(bank, selected_metric, confidence_level)
                if not series.empty:
                    all_series[bank] = series
            except:
                continue
        
        if all_series:
            # Create time series plot
            fig = go.Figure()
            
            # Color mapping for regions
            region_colors = {
                'Americas': '#059669',
                'Europe': '#0284c7', 
                'Asia/Pacific': '#dc2626'
            }
            
            for bank, series in all_series.items():
                bank_region = latest_metrics[latest_metrics['Bank'] == bank]['Region'].iloc[0]
                color = region_colors.get(bank_region, '#6b7280')
                
                fig.add_trace(go.Scatter(
                    x=series.index, 
                    y=series.values, 
                    name=bank,
                    line=dict(color=color, width=2),
                    hovertemplate=f'{bank}<br>Date: %{{x}}<br>{metric_options[selected_metric]}: %{{y:.4f}}<extra></extra>'
                ))
            
            # Add risk threshold lines for Beta
            if selected_metric == 'Beta_T':
                fig.add_hline(y=2.0, line_dash="dash", line_color="red", opacity=0.7, 
                            annotation_text="High Risk Threshold")
                fig.add_hline(y=1.5, line_dash="dash", line_color="orange", opacity=0.7,
                            annotation_text="Medium Risk Threshold")
            
            fig.update_layout(
                title=f"{metric_options[selected_metric]} Over Time",
                xaxis_title="Date",
                yaxis_title=metric_options[selected_metric],
                height=500,
                plot_bgcolor='white'
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            
            # Calculate statistics
            stats_data = []
            for bank, series in all_series.items():
                bank_region = latest_metrics[latest_metrics['Bank'] == bank]['Region'].iloc[0]
                stats_data.append({
                    'Bank': bank,
                    'Region': bank_region,
                    'Current': series.iloc[-1],
                    'Mean': series.mean(),
                    'Std': series.std(),
                    'Min': series.min(),
                    'Max': series.max()
                })
            
            stats_df = pd.DataFrame(stats_data).round(4)
            st.dataframe(
                stats_df.style.applymap(color_region, subset=['Region']),
                use_container_width=True,
                hide_index=True
            )
    
    with tab4:
        st.header("🌍 Regional Analysis")
        
        # Regional summary
        try:
            summary_stats = processor.get_summary_statistics(confidence_level)
            
            if not summary_stats.empty:
                st.subheader("📊 Regional Comparison")
                
                # Regional Beta comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart of regional averages
                    beta_by_region = summary_stats[('Beta_T', 'mean')].reset_index()
                    beta_by_region.columns = ['Region', 'Mean_Beta']
                    
                    fig_regional = px.bar(
                        beta_by_region, 
                        x='Region', 
                        y='Mean_Beta',
                        title="Average Systemic Beta by Region",
                        color='Region',
                        color_discrete_map={
                            'Americas': '#059669',
                            'Europe': '#0284c7',
                            'Asia/Pacific': '#dc2626'
                        }
                    )
                    
                    fig_regional.update_layout(
                        plot_bgcolor='white',
                        showlegend=False
                    )
                    fig_regional.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                    fig_regional.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                    
                    st.plotly_chart(fig_regional, use_container_width=True)
                
                with col2:
                    # Risk distribution by region
                    risk_by_region = []
                    for region in ['Americas', 'Europe', 'Asia/Pacific']:
                        region_banks = latest_metrics[latest_metrics['Region'] == region]
                        if not region_banks.empty:
                            high = len(region_banks[region_banks['Beta_T'] > 2.0])
                            medium = len(region_banks[
                                (region_banks['Beta_T'] > 1.5) & (region_banks['Beta_T'] <= 2.0)
                            ])
                            low = len(region_banks[region_banks['Beta_T'] <= 1.5])
                            
                            risk_by_region.extend([
                                {'Region': region, 'Risk Level': 'High', 'Count': high},
                                {'Region': region, 'Risk Level': 'Medium', 'Count': medium},
                                {'Region': region, 'Risk Level': 'Low', 'Count': low}
                            ])
                    
                    if risk_by_region:
                        risk_df = pd.DataFrame(risk_by_region)
                        
                        fig_risk = px.bar(
                            risk_df,
                            x='Region',
                            y='Count',
                            color='Risk Level',
                            title="Risk Distribution by Region",
                            color_discrete_map={
                                'High': '#dc2626',
                                'Medium': '#d97706',
                                'Low': '#059669'
                            }
                        )
                        
                        fig_risk.update_layout(plot_bgcolor='white')
                        fig_risk.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                        fig_risk.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                        
                        st.plotly_chart(fig_risk, use_container_width=True)
                
                # Detailed regional statistics
                st.subheader("📋 Detailed Regional Statistics")
                st.dataframe(summary_stats.round(4), use_container_width=True)
                
        except Exception as e:
            st.warning(f"⚠️ Could not generate regional analysis: {str(e)}")
    
    with tab5:
        st.header("⚠️ Risk Alerts")
        
        # Risk categorization
        high_risk_banks = latest_metrics[latest_metrics['Beta_T'] > 2.0]
        medium_risk_banks = latest_metrics[
            (latest_metrics['Beta_T'] > 1.5) & (latest_metrics['Beta_T'] <= 2.0)
        ]
        low_risk_banks = latest_metrics[latest_metrics['Beta_T'] <= 1.5]
        
        # Display alerts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 High Risk Banks")
            if not high_risk_banks.empty:
                for _, bank in high_risk_banks.iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #dc2626; margin-bottom: 0.5rem;">🔴 {bank['Bank']}</h4>
                        <p><strong>Systemic Beta:</strong> {bank['Beta_T']:.4f}</p>
                        <p><strong>Region:</strong> {bank['Region']}</p>
                        <p><strong>VaR:</strong> {bank[var_col]:.4f}</p>
                        <p style="color: #dc2626; font-weight: 600;">⚠️ Requires immediate attention</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No high-risk banks detected!")
        
        with col2:
            st.subheader("🟡 Medium Risk Banks")
            if not medium_risk_banks.empty:
                for _, bank in medium_risk_banks.iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #d97706; margin-bottom: 0.5rem;">🟡 {bank['Bank']}</h4>
                        <p><strong>Systemic Beta:</strong> {bank['Beta_T']:.4f}</p>
                        <p><strong>Region:</strong> {bank['Region']}</p>
                        <p><strong>VaR:</strong> {bank[var_col]:.4f}</p>
                        <p style="color: #d97706; font-weight: 600;">⚠️ Enhanced monitoring recommended</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ No medium-risk banks detected!")
        
        # System-wide alerts
        st.subheader("🌐 System-wide Risk Assessment")
        
        avg_beta = latest_metrics['Beta_T'].mean()
        max_beta = latest_metrics['Beta_T'].max()
        high_risk_pct = len(high_risk_banks) / len(latest_metrics) * 100
        
        # System risk level
        if avg_beta > 1.8 or high_risk_pct > 20:
            system_risk = "🔴 High System Risk"
            system_color = "#dc2626"
            system_desc = "The banking system shows elevated systemic risk. Consider implementing enhanced supervision measures."
        elif avg_beta > 1.4 or high_risk_pct > 10:
            system_risk = "🟡 Medium System Risk" 
            system_color = "#d97706"
            system_desc = "The banking system shows moderate systemic risk. Monitor closely and prepare contingency plans."
        else:
            system_risk = "🟢 Low System Risk"
            system_color = "#059669"
            system_desc = "The banking system shows low systemic risk. Continue regular monitoring."
        
        st.markdown(f"""
        <div class="metric-card" style="border-color: {system_color};">
            <h3 style="color: {system_color}; margin-bottom: 1rem;">{system_risk}</h3>
            <p>{system_desc}</p>
            <br>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <strong>Average Beta:</strong> {avg_beta:.3f}<br>
                    <strong>Maximum Beta:</strong> {max_beta:.3f}<br>
                    <strong>High Risk Banks:</strong> {high_risk_pct:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if not high_risk_banks.empty:
            st.markdown("""
            **Immediate Actions for High-Risk Banks:**
            - 🔍 Enhance supervisory oversight and monitoring frequency
            - 📊 Conduct detailed stress testing and scenario analysis
            - 💰 Review capital adequacy and liquidity buffers
            - 🔗 Assess interconnectedness with other financial institutions
            - 📋 Implement enhanced risk management requirements
            """)
        
        if not medium_risk_banks.empty:
            st.markdown("""
            **Preventive Measures for Medium-Risk Banks:**
            - 📈 Increase reporting frequency and data quality requirements
            - 🎯 Focus on early warning indicators and trend analysis
            - 🔄 Review and update risk management frameworks
            - 🤝 Enhance coordination with other supervisory authorities
            """)
        
        st.markdown("""
        **System-wide Measures:**
        - 🌐 Monitor cross-border exposures and contagion channels
        - 📊 Regular assessment of systemic risk evolution
        - 🔧 Maintain and update crisis management frameworks
        - 📚 Continue research on emerging systemic risks
        """)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h2 style="color: #1a1a1a; margin-bottom: 1rem;">Welcome to the Systemic Risk Dashboard</h2>
        <p style="color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem;">
            This dashboard provides real-time analysis of banking systemic risk using advanced Extreme Value Theory.
        </p>
        
        <div style="background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; 
                    padding: 2rem; margin: 2rem 0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #d97706; margin-bottom: 1.5rem;">📋 How to Get Started</h3>
            <div style="text-align: left; max-width: 600px; margin: 0 auto;">
                <div style="margin-bottom: 1rem;">
                    <strong>1. 📅 Set Date Range:</strong> Choose your analysis period in the sidebar
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>2. 🏦 Select Banks:</strong> Choose banks from Americas, Europe, and Asia/Pacific
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>3. ⚙️ Configure Parameters:</strong> Set confidence level and window size
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>4. 🚀 Load Data:</strong> Click "Load & Analyze Data" to process
                </div>
                <div>
                    <strong>5. 📊 Explore Results:</strong> Navigate through the different analysis tabs
                </div>
            </div>
        </div>
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
    
    # Feature highlights
    st.subheader("✨ Key Features")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #d97706;">📊 Advanced Analytics</h4>
            <p>Extreme Value Theory-based risk measurement with accurate Hill estimators and tail dependence analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #0284c7;">🔄 Real-time Data</h4>
            <p>Live data from Yahoo Finance with rolling window analysis and up-to-date risk metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #059669;">⚠️ Risk Alerts</h4>
            <p>Automated risk classification and early warning system with actionable recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p style="margin-bottom: 0.5rem;"><strong>Systemic Risk Analysis Platform</strong></p>
    <p style="font-size: 0.9rem;">Built with advanced Extreme Value Theory • Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
