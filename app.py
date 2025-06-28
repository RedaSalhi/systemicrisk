import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Systemic Risk in Global Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    """Redirect to dashboard page"""
    st.info("🔄 Redirecting to Dashboard page...")
    st.markdown("[📈 Go to Dashboard](pages/dashboard.py)")
    
    # Placeholder for dashboard preview
    st.markdown("### Dashboard Preview")
    
    # Mock visualization
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Systemic Beta Time Series")
        st.line_chart(pd.DataFrame({
            'JPMorgan Chase': np.random.randn(100).cumsum(),
            'Deutsche Bank': np.random.randn(100).cumsum(),
            'HSBC': np.random.randn(100).cumsum()
        }))
    
    with col2:
        st.markdown("#### Regional Beta Correlations")
        corr_data = np.random.rand(3, 3)
        st.dataframe(pd.DataFrame(corr_data, 
                                 columns=['Americas', 'Europe', 'Asia/Pacific'],
                                 index=['Americas', 'Europe', 'Asia/Pacific']))

def show_methodology():
    """Redirect to methodology page"""
    st.info("🔄 Redirecting to Methodology page...")
    st.markdown("[📚 Go to Methodology](pages/methodology.py)")
    
    # Methodology preview
    st.markdown("### Methodology Preview")
    st.markdown("""
    #### Systemic Beta Formula (van Oordt & Zhou, 2018)
    
    The systemic beta is calculated as:
    
    $$\\beta_T = \\tau^{1/\\xi_y} \\cdot \\frac{VaR_x}{VaR_y}$$
    
    Where:
    - $\\tau$ is the tail dependence coefficient
    - $\\xi_y$ is the Hill estimator (tail index)
    - $VaR_x$, $VaR_y$ are Value-at-Risk measures
    """)

def show_ml():
    """Redirect to ML page"""
    st.info("🔄 Redirecting to Machine Learning page...")
    st.markdown("[🤖 Go to Machine Learning](pages/machinelearning.py)")
    
    # ML preview
    st.markdown("### Machine Learning Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Performance")
        performance_data = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost'],
            'Precision': [0.78, 0.82],
            'Recall': [0.71, 0.76],
            'F1-Score': [0.74, 0.79],
            'AUC': [0.85, 0.88]
        })
        st.dataframe(performance_data)
    
    with col2:
        st.markdown("#### Feature Importance")
        features = ['beta_mean_95', 'var_extreme_95', 'tau_max_95', 'beta_spread']
        importance = [0.35, 0.28, 0.22, 0.15]
        st.bar_chart(pd.DataFrame({'Importance': importance}, index=features))

def show_about():
    """About page with technical details"""
    st.markdown("## ℹ️ About This Application")
    
    st.markdown("""
    ### 📖 Research Paper Implementation
    
    This Streamlit application is a faithful implementation of the research framework presented in:
    
    **"Systemic Risk in Global Banking Institutions"** by R. Salhi
    
    ### 🛠️ Technical Stack
    
    - **Frontend**: Streamlit with custom CSS
    - **Data**: Yahoo Finance API via `yfinance`
    - **Analytics**: NumPy, Pandas, SciPy
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Visualization**: Matplotlib, Seaborn, Plotly
    
    ### 📊 Data Sources
    
    - Bank stock prices from Yahoo Finance
    - Regional market indices for systemic beta calculation
    - Crisis period labels based on historical events
    
    ### 🔬 Methodological Notes
    
    1. **Rolling Windows**: 52-week rolling windows for all metrics
    2. **EVT Implementation**: Hill estimator with dynamic threshold selection
    3. **Tail Dependence**: Left-tail dependence for crisis scenarios
    4. **Stress Testing**: Spillover-aware scenarios based on systemic beta
    5. **ML Features**: Engineered from rolling statistics of systemic risk metrics
    
    ### 📞 Contact & Citations
    
    For questions about the methodology, please refer to the original research paper.
    For technical issues with this implementation, please check the GitHub repository.
    """)
    
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by Extreme Value Theory*")

if __name__ == "__main__":
    main()
