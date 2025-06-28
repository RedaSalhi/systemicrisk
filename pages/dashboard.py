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

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -3px rgba(0, 0, 0, 0.1);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-color: #ef4444;
        color: #991b1b;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-color: #f59e0b;
        color: #92400e;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-color: #10b981;
        color: #065f46;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px -3px rgba(102, 126, 234, 0.4);
    }
    
    .success-message {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        color: #065f46;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        color: #991b1b;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        color: #1e40af;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'loading' not in st.session_state:
        st.session_state.loading = False

init_session_state()

# Header
st.markdown('<h1 class="main-header">📊 Systemic Risk Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.2rem; margin-bottom: 2rem;">Real-time banking systemic risk analysis using Extreme Value Theory</p>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Initialize temporary processor for bank selection
    temp_processor = BankingDataProcessor()
    banks_by_region = temp_processor.get_banks_by_region()
    
    # Date range selection
    st.subheader("📅 Analysis Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime('2020-01-01').date(),
            min_value=pd.to_datetime('2015-01-01').date(),
            max_value=pd.to_datetime('2024-12-31').date()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=pd.to_datetime('2024-12-31').date(),
            min_value=pd.to_datetime('2015-01-01').date(),
            max_value=pd.to_datetime('2024-12-31').date()
        )
    
    # Bank selection by region
    st.subheader("🏦 Bank Selection")
    selected_banks = []
    
    # Quick selection options
    st.markdown("**Quick Select:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select Top US Banks", use_container_width=True):
            st.session_state.selected_americas = ['JPMorgan Chase', 'Bank of America', 'Wells Fargo', 'Citigroup']
    with col2:
        if st.button("Select Major European", use_container_width=True):
            st.session_state.selected_europe = ['HSBC Holdings', 'Deutsche Bank', 'UBS Group', 'BNP Paribas']
    
    # Americas selection
    st.markdown("**🌎 Americas**")
    americas_banks = banks_by_region.get('Americas', [])
    default_americas = getattr(st.session_state, 'selected_americas', americas_banks[:3])
    americas_selected = st.multiselect(
        "American banks:",
        americas_banks,
        default=default_americas,
        key="americas"
    )
    selected_banks.extend(americas_selected)
    
    # Europe selection
    st.markdown("**🇪🇺 Europe**")
    europe_banks = banks_by_region.get('Europe', [])
    default_europe = getattr(st.session_state, 'selected_europe', europe_banks[:3])
    europe_selected = st.multiselect(
        "European banks:",
        europe_banks,
        default=default_europe,
        key="europe"
    )
    selected_banks.extend(europe_selected)
    
    # Asia/Pacific selection
    st.markdown("**🌏 Asia/Pacific**")
    asia_banks = banks_by_region.get('Asia/Pacific', [])
    default_asia = getattr(st.session_state, 'selected_asia', asia_banks[:2])
    asia_selected = st.multiselect(
        "Asia/Pacific banks:",
        asia_banks,
        default=default_asia,
        key="asia"
    )
    selected_banks.extend(asia_selected)
    
    # Analysis parameters
    st.subheader("📈 Parameters")
    confidence_level = st.selectbox(
        "Confidence Level:",
        [0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%",
        help="Confidence level for risk calculations"
    )
    
    window_size = st.slider(
        "Rolling Window (weeks):",
        min_value=26, max_value=104, value=52,
        help="Number of weeks for rolling analysis"
    )
    
    # Data loading section
    st.subheader("🚀 Data Processing")
    
    # Show selection summary
    if selected_banks:
        st.success(f"✅ Selected {len(selected_banks)} banks")
        with st.expander("View selected banks"):
            for bank in selected_banks:
                st.write(f"• {bank}")
    else:
        st.warning("⚠️ Please select at least one bank")
    
    # Load data button
    load_button = st.button(
        "🚀 Load & Analyze Data",
        type="primary",
        disabled=not selected_banks or st.session_state.loading,
        use_container_width=True
    )
    
    if load_button and selected_banks:
        st.session_state.loading = True
        st.rerun()

# Main content area
if st.session_state.loading:
    st.markdown("## 🔄 Loading Data...")
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing data processor...")
            progress_bar.progress(10)
            
            status_text.text("📥 Downloading bank data...")
            progress_bar.progress(30)
            
            processor = process_banking_data(
                selected_banks,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Data loaded successfully!")
            
            # Store in session state
            st.session_state.processor = processor
            st.session_state.data_loaded = True
            st.session_state.loading = False
            
            st.success("🎉 Analysis complete! Navigate through the tabs below to explore results.")
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.session_state.loading = False
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("💡 Try selecting different banks or adjusting the date range.")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write("**Selected banks:**", selected_banks)
                st.write("**Date range:**", f"{start_date} to {end_date}")
                st.write("**Error details:**", str(e))

elif st.session_state.data_loaded and st.session_state.processor is not None:
    processor = st.session_state.processor
    
    try:
        # Get latest metrics
        latest_metrics = processor.get_latest_metrics(confidence_level)
        
        if latest_metrics.empty:
            st.error("❌ No metrics available. Please try different parameters.")
            st.stop()
        
        # Create main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🏦 Bank Analysis", 
            "📈 Time Series",
            "🌍 Regional Analysis",
            "⚠️ Risk Alerts"
        ])
        
        with tab1:
            st.header("📊 System Overview")
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_beta = latest_metrics['Beta_T'].mean()
                st.metric(
                    "Average Systemic Beta",
                    f"{avg_beta:.3f}",
                    delta=f"{avg_beta - 1:.3f}",
                    help="Average systemic risk across all banks"
                )
            
            with col2:
                max_beta = latest_metrics['Beta_T'].max()
                max_bank = latest_metrics.loc[latest_metrics['Beta_T'].idxmax(), 'Bank']
                st.metric(
                    "Highest Beta",
                    f"{max_beta:.3f}",
                    delta=max_bank,
                    help="Bank with highest systemic risk"
                )
            
            with col3:
                var_col = f'VaR_{int(confidence_level*100)}'
                avg_var = latest_metrics[var_col].mean()
                st.metric(
                    f"Avg VaR ({int(confidence_level*100)}%)",
                    f"{avg_var:.4f}",
                    help="Average Value-at-Risk"
                )
            
            with col4:
                tau_col = f'Tau_{int(confidence_level*100)}'
                avg_tau = latest_metrics[tau_col].mean()
                st.metric(
                    "Avg Tail Dependence",
                    f"{avg_tau:.3f}",
                    help="Average systemic interconnectedness"
                )
            
            # Risk distribution
            st.subheader("🎯 Risk Distribution")
            
            high_risk = latest_metrics[latest_metrics['Beta_T'] > 2.0]
            medium_risk = latest_metrics[(latest_metrics['Beta_T'] > 1.5) & (latest_metrics['Beta_T'] <= 2.0)]
            low_risk = latest_metrics[latest_metrics['Beta_T'] <= 1.5]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card risk-high">
                    <h3>🔴 High Risk</h3>
                    <h2>{len(high_risk)}</h2>
                    <p>Beta > 2.0</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card risk-medium">
                    <h3>🟡 Medium Risk</h3>
                    <h2>{len(medium_risk)}</h2>
                    <p>1.5 < Beta ≤ 2.0</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card risk-low">
                    <h3>🟢 Low Risk</h3>
                    <h2>{len(low_risk)}</h2>
                    <p>Beta ≤ 1.5</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk visualization
            st.subheader("📊 Risk Visualization")
            
            # Create risk distribution chart
            fig = px.scatter(
                latest_metrics,
                x=var_col,
                y='Beta_T',
                color='Region',
                size=tau_col,
                hover_name='Bank',
                title=f"Risk Map: VaR vs Systemic Beta ({int(confidence_level*100)}% confidence)",
                color_discrete_map={
                    'Americas': '#10b981',
                    'Europe': '#3b82f6',
                    'Asia/Pacific': '#ef4444'
                }
            )
            
            # Add risk threshold lines
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="High Risk")
            fig.add_hline(y=1.5, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
            
            fig.update_layout(
                height=500,
                xaxis_title=f"Value at Risk ({int(confidence_level*100)}%)",
                yaxis_title="Systemic Beta"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest metrics table
            st.subheader("📋 Latest Risk Metrics")
            
            display_df = latest_metrics[['Bank', 'Region', 'Beta_T', var_col, tau_col]].copy()
            display_df = display_df.round(4)
            
            # Add risk level
            def get_risk_level(beta):
                if beta > 2.0:
                    return "🔴 High"
                elif beta > 1.5:
                    return "🟡 Medium"
                else:
                    return "🟢 Low"
            
            display_df['Risk Level'] = display_df['Beta_T'].apply(get_risk_level)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.header("🏦 Individual Bank Analysis")
            
            # Bank selector
            bank_names = latest_metrics['Bank'].unique()
            selected_bank = st.selectbox(
                "Select a bank for detailed analysis:",
                bank_names,
                key="bank_selector"
            )
            
            if selected_bank:
                bank_data = latest_metrics[latest_metrics['Bank'] == selected_bank].iloc[0]
                
                # Bank overview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"📊 {selected_bank}")
                    
                    # Current metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        beta_val = bank_data['Beta_T']
                        st.metric("Systemic Beta", f"{beta_val:.4f}")
                    with metric_col2:
                        var_val = bank_data[var_col]
                        st.metric(f"VaR ({int(confidence_level*100)}%)", f"{var_val:.4f}")
                    with metric_col3:
                        tau_val = bank_data[tau_col]
                        st.metric("Tail Dependence", f"{tau_val:.4f}")
                    
                    # Risk assessment
                    if beta_val > 2.0:
                        risk_class = "risk-high"
                        risk_text = "🔴 High Risk"
                        risk_desc = "This bank poses significant systemic risk and requires enhanced monitoring."
                    elif beta_val > 1.5:
                        risk_class = "risk-medium"
                        risk_text = "🟡 Medium Risk"
                        risk_desc = "This bank poses moderate systemic risk and should be monitored closely."
                    else:
                        risk_class = "risk-low"
                        risk_text = "🟢 Low Risk"
                        risk_desc = "This bank poses low systemic risk under current conditions."
                    
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>{risk_text}</h3>
                        <p>{risk_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Regional info
                    region = bank_data['Region']
                    st.subheader("🌍 Regional Info")
                    st.write(f"**Region:** {region}")
                    
                    hill_col = f'Hill_{int(confidence_level*100)}'
                    hill_val = bank_data[hill_col]
                    st.metric("Hill Estimator", f"{hill_val:.4f}")
                
                # Time series analysis
                try:
                    st.subheader("📈 Historical Analysis")
                    
                    # Get time series
                    beta_series = processor.get_bank_time_series(selected_bank, 'Beta_T', confidence_level)
                    var_series = processor.get_bank_time_series(selected_bank, var_col, confidence_level)
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(
                            f'Systemic Beta - {selected_bank}',
                            f'Value at Risk ({int(confidence_level*100)}%) - {selected_bank}'
                        ),
                        vertical_spacing=0.1
                    )
                    
                    # Add beta trace
                    fig.add_trace(
                        go.Scatter(
                            x=beta_series.index,
                            y=beta_series.values,
                            name='Systemic Beta',
                            line=dict(color='#ef4444', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add VaR trace
                    fig.add_trace(
                        go.Scatter(
                            x=var_series.index,
                            y=var_series.values,
                            name='VaR',
                            line=dict(color='#3b82f6', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    # Add threshold lines
                    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=1, col=1)
                    fig.add_hline(y=1.5, line_dash="dash", line_color="orange", row=1, col=1)
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not load historical data: {str(e)}")
        
        with tab3:
            st.header("📈 Time Series Analysis")
            
            # Metric selector
            metric_options = {
                'Beta_T': 'Systemic Beta',
                var_col: f'VaR ({int(confidence_level*100)}%)',
                tau_col: f'Tail Dependence ({int(confidence_level*100)}%)'
            }
            
            selected_metric = st.selectbox(
                "Select metric:",
                list(metric_options.keys()),
                format_func=lambda x: metric_options[x]
            )
            
            # Get time series for all banks
            try:
                all_data = processor.get_all_metrics(confidence_level)
                
                # Create pivot table
                pivot_data = all_data.pivot(index='Date', columns='Bank', values=selected_metric)
                
                # Plot time series
                fig = go.Figure()
                
                # Color mapping
                region_colors = {
                    'Americas': '#10b981',
                    'Europe': '#3b82f6', 
                    'Asia/Pacific': '#ef4444'
                }
                
                for bank in pivot_data.columns:
                    bank_region = latest_metrics[latest_metrics['Bank'] == bank]['Region'].iloc[0]
                    color = region_colors.get(bank_region, '#6b7280')
                    
                    fig.add_trace(go.Scatter(
                        x=pivot_data.index,
                        y=pivot_data[bank],
                        name=bank,
                        line=dict(color=color, width=2),
                        hovertemplate=f'{bank}<br>%{{x}}<br>{metric_options[selected_metric]}: %{{y:.4f}}<extra></extra>'
                    ))
                
                # Add threshold lines for Beta
                if selected_metric == 'Beta_T':
                    fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="High Risk")
                    fig.add_hline(y=1.5, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
                
                fig.update_layout(
                    title=f"{metric_options[selected_metric]} Time Series",
                    xaxis_title="Date",
                    yaxis_title=metric_options[selected_metric],
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics table
                st.subheader("📊 Summary Statistics")
                
                stats_data = []
                for bank in pivot_data.columns:
                    series = pivot_data[bank].dropna()
                    if len(series) > 0:
                        bank_region = latest_metrics[latest_metrics['Bank'] == bank]['Region'].iloc[0]
                        stats_data.append({
                            'Bank': bank,
                            'Region': bank_region,
                            'Current': series.iloc[-1] if len(series) > 0 else np.nan,
                            'Mean': series.mean(),
                            'Std': series.std(),
                            'Min': series.min(),
                            'Max': series.max()
                        })
                
                stats_df = pd.DataFrame(stats_data).round(4)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error creating time series: {str(e)}")
        
        with tab4:
            st.header("🌍 Regional Analysis")
            
            try:
                # Regional comparison
                regional_summary = latest_metrics.groupby('Region').agg({
                    'Beta_T': ['mean', 'std', 'count'],
                    var_col: ['mean', 'std'],
                    tau_col: ['mean', 'std']
                }).round(4)
                
                # Regional beta comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    regional_beta = latest_metrics.groupby('Region')['Beta_T'].mean().reset_index()
                    
                    fig = px.bar(
                        regional_beta,
                        x='Region',
                        y='Beta_T',
                        title="Average Systemic Beta by Region",
                        color='Region',
                        color_discrete_map={
                            'Americas': '#10b981',
                            'Europe': '#3b82f6',
                            'Asia/Pacific': '#ef4444'
                        }
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk distribution by region
                    risk_dist = []
                    for region in latest_metrics['Region'].unique():
                        region_data = latest_metrics[latest_metrics['Region'] == region]
                        high = len(region_data[region_data['Beta_T'] > 2.0])
                        medium = len(region_data[(region_data['Beta_T'] > 1.5) & (region_data['Beta_T'] <= 2.0)])
                        low = len(region_data[region_data['Beta_T'] <= 1.5])
                        
                        risk_dist.extend([
                            {'Region': region, 'Risk Level': 'High', 'Count': high},
                            {'Region': region, 'Risk Level': 'Medium', 'Count': medium},
                            {'Region': region, 'Risk Level': 'Low', 'Count': low}
                        ])
                    
                    risk_df = pd.DataFrame(risk_dist)
                    
                    fig = px.bar(
                        risk_df,
                        x='Region',
                        y='Count',
                        color='Risk Level',
                        title="Risk Distribution by Region",
                        color_discrete_map={
                            'High': '#ef4444',
                            'Medium': '#f59e0b',
                            'Low': '#10b981'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed regional table
                st.subheader("📋 Regional Statistics")
                st.dataframe(regional_summary, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error in regional analysis: {str(e)}")
        
        with tab5:
            st.header("⚠️ Risk Alerts & Recommendations")
            
            # Current risk assessment
            high_risk_banks = latest_metrics[latest_metrics['Beta_T'] > 2.0]
            medium_risk_banks = latest_metrics[(latest_metrics['Beta_T'] > 1.5) & (latest_metrics['Beta_T'] <= 2.0)]
            
            # System-wide assessment
            avg_beta = latest_metrics['Beta_T'].mean()
            max_beta = latest_metrics['Beta_T'].max()
            high_risk_pct = len(high_risk_banks) / len(latest_metrics) * 100
            
            # System risk level
            if avg_beta > 1.8 or high_risk_pct > 20:
                system_risk_class = "risk-high"
                system_risk_text = "🔴 HIGH SYSTEM RISK"
                system_desc = "The banking system shows elevated systemic risk. Immediate action required."
            elif avg_beta > 1.4 or high_risk_pct > 10:
                system_risk_class = "risk-medium"
                system_risk_text = "🟡 MODERATE SYSTEM RISK"
                system_desc = "The banking system shows moderate risk. Enhanced monitoring recommended."
            else:
                system_risk_class = "risk-low"
                system_risk_text = "🟢 LOW SYSTEM RISK"
                system_desc = "The banking system shows low systemic risk. Continue regular monitoring."
            
            st.markdown(f"""
            <div class="metric-card {system_risk_class}">
                <h2>{system_risk_text}</h2>
                <p>{system_desc}</p>
                <br>
                <p><strong>Average Beta:</strong> {avg_beta:.3f}</p>
                <p><strong>Maximum Beta:</strong> {max_beta:.3f}</p>
                <p><strong>High Risk Banks:</strong> {high_risk_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual alerts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔴 High Risk Banks")
                if not high_risk_banks.empty:
                    for _, bank in high_risk_banks.iterrows():
                        st.markdown(f"""
                        <div class="metric-card risk-high">
                            <h4>{bank['Bank']}</h4>
                            <p><strong>Beta:</strong> {bank['Beta_T']:.4f}</p>
                            <p><strong>Region:</strong> {bank['Region']}</p>
                            <p><strong>VaR:</strong> {bank[var_col]:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No high-risk banks detected!")
            
            with col2:
                st.subheader("🟡 Medium Risk Banks")
                if not medium_risk_banks.empty:
                    for _, bank in medium_risk_banks.iterrows():
                        st.markdown(f"""
                        <div class="metric-card risk-medium">
                            <h4>{bank['Bank']}</h4>
                            <p><strong>Beta:</strong> {bank['Beta_T']:.4f}</p>
                            <p><strong>Region:</strong> {bank['Region']}</p>
                            <p><strong>VaR:</strong> {bank[var_col]:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No medium-risk banks detected!")
            
            # Recommendations
            st.subheader("💡 Actionable Recommendations")
            
            if not high_risk_banks.empty:
                st.markdown("""
                <div class="info-box">
                <h4>🔴 High-Risk Bank Actions:</h4>
                <ul>
                <li>🔍 Implement enhanced supervisory oversight</li>
                <li>📊 Conduct immediate stress testing</li>
                <li>💰 Review capital adequacy ratios</li>
                <li>🔗 Assess interconnectedness with other institutions</li>
                <li>📋 Increase reporting frequency</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            if not medium_risk_banks.empty:
                st.markdown("""
                <div class="info-box">
                <h4>🟡 Medium-Risk Bank Actions:</h4>
                <ul>
                <li>📈 Increase monitoring frequency</li>
                <li>🎯 Focus on early warning indicators</li>
                <li>🔄 Review risk management frameworks</li>
                <li>🤝 Enhance coordination with supervisors</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <h4>🌐 System-wide Measures:</h4>
            <ul>
            <li>🌐 Monitor cross-border exposures</li>
            <li>📊 Regular systemic risk assessments</li>
            <li>🔧 Maintain crisis management frameworks</li>
            <li>📚 Continue research on emerging risks</li>
            <li>💼 Coordinate with international regulators</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error displaying results: {str(e)}")
        st.info("💡 Try reloading the data or selecting different parameters.")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div class="info-box">
            <h2>🎯 Welcome to the Systemic Risk Dashboard</h2>
            <p style="font-size: 1.1rem;">Advanced banking systemic risk analysis using Extreme Value Theory with real-time data.</p>
        </div>
        
        <div class="metric-card" style="margin: 2rem 0;">
            <h3>📋 Quick Start Guide</h3>
            <div style="text-align: left; max-width: 600px; margin: 0 auto;">
                <p><strong>1. 📅 Set Analysis Period:</strong> Choose your date range (recommended: 2020-2024)</p>
                <p><strong>2. 🏦 Select Banks:</strong> Choose 5-15 banks from different regions</p>
                <p><strong>3. ⚙️ Configure Parameters:</strong> Set confidence level and window size</p>
                <p><strong>4. 🚀 Load Data:</strong> Click the load button to start analysis</p>
                <p><strong>5. 📊 Explore Results:</strong> Navigate through tabs to view insights</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Available banks preview
    st.subheader("🏦 Available Banks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌎 Americas**")
        for bank in banks_by_region.get('Americas', [])[:5]:
            st.write(f"• {bank}")
        if len(banks_by_region.get('Americas', [])) > 5:
            st.write(f"... and {len(banks_by_region.get('Americas', [])) - 5} more")
    
    with col2:
        st.markdown("**🇪🇺 Europe**")
        for bank in banks_by_region.get('Europe', [])[:5]:
            st.write(f"• {bank}")
        if len(banks_by_region.get('Europe', [])) > 5:
            st.write(f"... and {len(banks_by_region.get('Europe', [])) - 5} more")
    
    with col3:
        st.markdown("**🌏 Asia/Pacific**")
        for bank in banks_by_region.get('Asia/Pacific', [])[:5]:
            st.write(f"• {bank}")
        if len(banks_by_region.get('Asia/Pacific', [])) > 5:
            st.write(f"... and {len(banks_by_region.get('Asia/Pacific', [])) - 5} more")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p><strong>Systemic Risk Analysis Platform</strong> | Built with accurate Extreme Value Theory</p>
    <p>Real-time data from Yahoo Finance • Advanced mathematical implementations • Professional analysis tools</p>
</div>
""", unsafe_allow_html=True)
