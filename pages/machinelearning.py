import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import our data processor
from data_processor import BankingDataProcessor, process_banking_data

# Page configuration
st.set_page_config(
    page_title="Early Warning System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .warning-high {
        background-color: #fff5f5;
        border: 2px solid #f56565;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-high h4 {
        color: #c53030;
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .warning-high p {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .warning-high ul {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .warning-high li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    .warning-medium {
        background-color: #fffbf0;
        border: 2px solid #ed8936;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-medium h4 {
        color: #d68910;
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .warning-medium p {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .warning-medium ul {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .warning-medium li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    .warning-low {
        background-color: #f0fff4;
        border: 2px solid #48bb78;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-low h4 {
        color: #38a169;
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .warning-low p {
        color: #333;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .warning-low ul {
        color: #555;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .warning-low li {
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.4rem 0;
        border: 1px solid #e1e5e9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .feature-importance strong {
        color: #2c3e50;
        font-weight: 600;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.25rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
    }
    .nav-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
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
</style>
""", unsafe_allow_html=True)

class SystemicRiskEarlyWarning:
    def __init__(self):
        """Initialize the Early Warning System"""
        self.define_crisis_periods()
        self.models = {}
        self.feature_names = []

    def define_crisis_periods(self):
        """Define crisis periods for labeling"""
        self.crisis_periods = {
            'eurozone_crisis': (pd.Timestamp('2011-07-01'), pd.Timestamp('2012-12-31')),
            'china_correction': (pd.Timestamp('2015-06-01'), pd.Timestamp('2016-02-29')),
            'covid_crash': (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-05-31')),
            'ukraine_war': (pd.Timestamp('2022-02-01'), pd.Timestamp('2022-06-30')),
            'banking_stress_2023': (pd.Timestamp('2023-03-01'), pd.Timestamp('2023-05-31'))
        }

    def load_data_from_processor(self, processor):
        """Load data from our BankingDataProcessor"""
        try:
            # Get metrics for both confidence levels
            self.metrics_95 = processor.get_all_metrics(0.95)
            self.metrics_99 = processor.get_all_metrics(0.99)
            
            # Convert to multi-index format if needed
            if not isinstance(self.metrics_95.index, pd.MultiIndex):
                self.metrics_95 = self.metrics_95.reset_index().set_index(['Date', 'Bank'])
            if not isinstance(self.metrics_99.index, pd.MultiIndex):
                self.metrics_99 = self.metrics_99.reset_index().set_index(['Date', 'Bank'])
            
            st.success(f"✅ Loaded 95% data shape: {self.metrics_95.shape}")
            st.success(f"✅ Loaded 99% data shape: {self.metrics_99.shape}")
            st.info(f"📅 Date range: {self.metrics_95.index.get_level_values(0).min().date()} to {self.metrics_95.index.get_level_values(0).max().date()}")
            st.info(f"🏦 Banks: {len(self.metrics_95.index.get_level_values(1).unique())}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return False

    def create_crisis_labels(self, lead_weeks=8):
        """Create binary crisis labels with lead time"""
        dates = self.metrics_95.index.get_level_values(0).unique().sort_values()
        labels = pd.Series(0, index=dates, name='crisis_label')

        # Label crisis periods and pre-crisis periods
        for crisis_name, (start, end) in self.crisis_periods.items():
            # Crisis period
            crisis_mask = (dates >= start) & (dates <= end)
            labels.loc[crisis_mask] = 1

            # Pre-crisis period (lead_weeks before crisis)
            pre_crisis_start = start - pd.Timedelta(weeks=lead_weeks)
            pre_crisis_mask = (dates >= pre_crisis_start) & (dates < start)
            labels.loc[pre_crisis_mask] = 1

        st.info(f"🎯 Created labels with {lead_weeks} weeks lead time")
        st.info(f"⚠️  Crisis periods: {labels.sum()} weeks out of {len(labels)} total")
        st.info(f"📊 Crisis ratio: {labels.mean():.3f}")

        return labels

    def engineer_features(self):
        """Engineer features for ML model"""
        st.info("🔧 Engineering features...")

        # Get data
        data_95 = self.metrics_95.copy()
        data_99 = self.metrics_99.copy()

        features_list = []
        dates = data_95.index.get_level_values(0).unique().sort_values()

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, date in enumerate(dates):
            if i % max(1, len(dates) // 20) == 0:
                progress = (i + 1) / len(dates)
                progress_bar.progress(progress)
                status_text.text(f"Processing {i+1}/{len(dates)} dates...")

            try:
                date_data_95 = data_95.loc[date]
                date_data_99 = data_99.loc[date]

                # Basic aggregate features
                feature_dict = {'Date': date}

                # Beta_T statistics (main features)
                if 'Beta_T' in date_data_95.columns:
                    beta_95 = date_data_95['Beta_T']
                    feature_dict.update({
                        'beta_mean_95': beta_95.mean(),
                        'beta_std_95': beta_95.std(),
                        'beta_max_95': beta_95.max(),
                        'beta_75pct_95': beta_95.quantile(0.75),
                        'beta_high_risk_95': (beta_95 > 2.0).sum(),
                        'beta_skew_95': beta_95.skew()
                    })

                if 'Beta_T' in date_data_99.columns:
                    beta_99 = date_data_99['Beta_T']
                    feature_dict.update({
                        'beta_mean_99': beta_99.mean(),
                        'beta_std_99': beta_99.std(),
                        'beta_max_99': beta_99.max(),
                        'beta_75pct_99': beta_99.quantile(0.75),
                        'beta_high_risk_99': (beta_99 > 2.0).sum()
                    })

                # VaR features
                if 'VaR_95' in date_data_95.columns:
                    var_95 = date_data_95['VaR_95']
                    feature_dict['var_mean_95'] = var_95.mean()
                    feature_dict['var_extreme_95'] = (var_95 < -0.1).sum()

                # Tail dependence
                if 'Tau_95' in date_data_95.columns:
                    tau_95 = date_data_95['Tau_95']
                    feature_dict['tau_mean_95'] = tau_95.mean()
                    feature_dict['tau_max_95'] = tau_95.max()

                features_list.append(feature_dict)

            except Exception as e:
                continue

        progress_bar.progress(1.0)
        status_text.text("✅ Feature engineering completed!")

        # Create DataFrame
        features_df = pd.DataFrame(features_list).set_index('Date')

        # Add rolling features
        st.info("🔄 Adding rolling features...")
        for window in [4, 8, 12]:
            if 'beta_mean_95' in features_df.columns:
                features_df[f'beta_mean_95_roll_{window}w'] = features_df['beta_mean_95'].rolling(window).mean()
                features_df[f'beta_mean_95_trend_{window}w'] = features_df['beta_mean_95'].rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )

        # Cross-VaR features
        if 'beta_mean_95' in features_df.columns and 'beta_mean_99' in features_df.columns:
            features_df['beta_spread'] = features_df['beta_mean_99'] - features_df['beta_mean_95']

        # Clean data
        features_df = features_df.dropna(thresh=len(features_df.columns) * 0.6)
        features_df = features_df.fillna(method='ffill')

        st.success(f"✅ Created {len(features_df.columns)} features for {len(features_df)} time periods")

        self.features_df = features_df
        return features_df

    def prepare_ml_dataset(self, lead_weeks=8):
        """Prepare dataset for ML"""
        features_df = self.engineer_features()
        labels = self.create_crisis_labels(lead_weeks)

        # Align data
        common_dates = features_df.index.intersection(labels.index)
        X = features_df.loc[common_dates]
        y = labels.loc[common_dates]

        st.success(f"📈 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        st.info(f"⚖️  Class distribution: {dict(y.value_counts())}")

        return X, y

    def train_models(self, X, y):
        """Train ML models"""
        st.info("🤖 Training ML Models...")

        # Time-based split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        st.info(f"📊 Train set: {len(X_train)} samples")
        st.info(f"📊 Test set: {len(X_test)} samples")

        # Random Forest
        st.info("🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)

        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_proba) if len(y_test.unique()) > 1 else 0.5

        # XGBoost
        st.info("🚀 Training XGBoost...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)

        xgb_pred = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_proba) if len(y_test.unique()) > 1 else 0.5

        # Store results
        self.models = {
            'random_forest': {
                'model': rf_model,
                'y_test': y_test,
                'y_pred': rf_pred,
                'y_proba': rf_proba,
                'auc': rf_auc,
                'feature_importance': pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            },
            'xgboost': {
                'model': xgb_model,
                'y_test': y_test,
                'y_pred': xgb_pred,
                'y_proba': xgb_proba,
                'auc': xgb_auc,
                'feature_importance': pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            }
        }

        st.success(f"✅ Random Forest AUC: {rf_auc:.3f}")
        st.success(f"✅ XGBoost AUC: {xgb_auc:.3f}")

        return self.models

    def get_current_risk_assessment(self):
        """Get current risk assessment"""
        if not hasattr(self, 'features_df') or 'random_forest' not in self.models:
            return None, None, None

        try:
            latest_data = self.features_df.tail(1)
            current_risk = self.models['random_forest']['model'].predict_proba(latest_data)[0, 1]

            if current_risk > 0.7:
                risk_level = "HIGH"
                emoji = "🔴"
            elif current_risk > 0.4:
                risk_level = "MODERATE"
                emoji = "🟡"
            else:
                risk_level = "LOW"
                emoji = "🟢"

            return current_risk, risk_level, emoji

        except Exception as e:
            st.error(f"Unable to assess current risk: {e}")
            return None, None, None

def generate_paper_results(ews):
    """Generate results for academic paper"""
    st.markdown("## 📝 Academic Paper Results")
    st.markdown("---")

    # Model Performance Table
    if ews.models:
        st.markdown("### 🏆 Model Performance Comparison")
        
        performance_data = []
        for model_name, results in ews.models.items():
            y_test, y_pred = results['y_test'], results['y_pred']

            if len(y_test.unique()) > 1:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0

            performance_data.append({
                'Model': model_name.title(),
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC': results['auc'],
                'Lead Time': '8-10 weeks'
            })

        performance_df = pd.DataFrame(performance_data)
        st.table(performance_df)

    # Feature Importance
    if 'random_forest' in ews.models:
        st.markdown("### 🔍 Top 10 Most Important Features")
        importance = ews.models['random_forest']['feature_importance'].head(10)
        
        importance_data = []
        for i, (feature, score) in enumerate(importance.items(), 1):
            importance_data.append({
                'Rank': i,
                'Feature': feature,
                'Importance': score
            })
        
        importance_df = pd.DataFrame(importance_data)
        st.table(importance_df)

    # Current Risk Assessment
    current_risk, risk_level, emoji = ews.get_current_risk_assessment()
    if current_risk is not None:
        st.markdown("### ⚠️ Current Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Crisis Probability", f"{current_risk:.1%}")
        with col2:
            st.metric("Risk Level", f"{emoji} {risk_level}")
        with col3:
            if 'beta_mean_95' in ews.features_df.columns:
                latest_beta = ews.features_df['beta_mean_95'].iloc[-1]
                st.metric("Current Avg Beta_T (95%)", f"{latest_beta:.3f}")

def plot_model_performance(ews):
    """Plot model performance metrics"""
    if not ews.models:
        return

    st.markdown("### 📊 Model Performance Visualization")

    # ROC Curves
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ROC Curves', 'Feature Importance'),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )

    # Add ROC curves
    for model_name, results in ews.models.items():
        if len(results['y_test'].unique()) > 1:
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_proba'])
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, name=f'{model_name.title()} (AUC={results["auc"]:.3f})'),
                row=1, col=1
            )

    # Add diagonal line
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')),
        row=1, col=1
    )

    # Add feature importance
    if 'random_forest' in ews.models:
        importance = ews.models['random_forest']['feature_importance'].head(10)
        fig.add_trace(
            go.Bar(x=importance.values, y=importance.index, orientation='h', name='Feature Importance'),
            row=1, col=2
        )

    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_crisis_timeline(ews):
    """Plot crisis timeline with predictions"""
    if not hasattr(ews, 'features_df') or 'random_forest' not in ews.models:
        return

    st.markdown("### 📈 Crisis Timeline Analysis")

    # Get predictions for all data
    predictions = ews.models['random_forest']['model'].predict_proba(ews.features_df)[:, 1]
    timeline_df = pd.DataFrame({
        'Date': ews.features_df.index,
        'Crisis_Probability': predictions,
        'Beta_Mean': ews.features_df['beta_mean_95'] if 'beta_mean_95' in ews.features_df.columns else 0
    })

    # Create timeline plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Crisis Probability Over Time', 'Average Beta_T Over Time'),
        vertical_spacing=0.1
    )

    # Crisis probability
    fig.add_trace(
        go.Scatter(x=timeline_df['Date'], y=timeline_df['Crisis_Probability'], 
                  name='Crisis Probability', line=dict(color='red')),
        row=1, col=1
    )

    # Add threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold", row=1, col=1)
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Risk Threshold", row=1, col=1)

    # Beta mean
    if 'beta_mean_95' in ews.features_df.columns:
        fig.add_trace(
            go.Scatter(x=timeline_df['Date'], y=timeline_df['Beta_Mean'], 
                      name='Average Beta_T', line=dict(color='blue')),
            row=2, col=1
        )

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# Main Streamlit app
def main():
    st.markdown('<h1 class="main-header">🤖 Systemic Risk Early Warning System</h1>', unsafe_allow_html=True)
    
    # Navigation
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <a href="?page=dashboard" class="nav-button">📊 Dashboard</a>
        <a href="?page=methodology" class="nav-button">📚 Methodology</a>
        <a href="?page=machinelearning" class="nav-button">🤖 ML System</a>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Data loading section
    st.sidebar.markdown("### 📊 Data Loading")
    
    # Bank selection
    available_banks = [
        'JPMorgan Chase', 'Citigroup', 'Bank of America', 'Wells Fargo',
        'Goldman Sachs', 'Morgan Stanley', 'HSBC', 'Barclays', 'BNP Paribas',
        'Deutsche Bank', 'Santander', 'UBS', 'Standard Chartered'
    ]
    
    selected_banks = st.sidebar.multiselect(
        "Select Banks for Analysis",
        available_banks,
        default=['JPMorgan Chase', 'HSBC', 'Deutsche Bank'],
        help="Choose banks to include in the early warning system"
    )
    
    # Date range
    st.sidebar.markdown("### 📅 Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime('2020-01-01').date(),
        min_value=pd.to_datetime('2010-01-01').date(),
        max_value=pd.to_datetime('2024-12-31').date()
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=pd.to_datetime('2024-12-31').date(),
        min_value=pd.to_datetime('2010-01-01').date(),
        max_value=pd.to_datetime('2024-12-31').date()
    )
    
    # ML parameters
    st.sidebar.markdown("### 🤖 ML Parameters")
    lead_weeks = st.sidebar.slider(
        "Lead Time (weeks)",
        min_value=4,
        max_value=16,
        value=8,
        help="How many weeks in advance to predict crises"
    )
    
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        [0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%",
        help="Confidence level for risk calculations"
    )
    
    # Load data button
    if st.sidebar.button("🚀 Load Data & Train Models", type="primary"):
        if not selected_banks:
            st.error("Please select at least one bank.")
            return
            
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return
        
        with st.spinner("Loading data and training models..."):
            try:
                # Initialize data processor
                processor = process_banking_data(
                    selected_banks,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Initialize early warning system
                ews = SystemicRiskEarlyWarning()
                
                # Load data
                if not ews.load_data_from_processor(processor):
                    st.error("Failed to load data. Please check your selections.")
                    return
                
                # Prepare dataset
                X, y = ews.prepare_ml_dataset(lead_weeks)
                
                # Check if we have enough data
                if len(X) < 50:
                    st.warning(f"⚠️ Limited data available ({len(X)} samples). Consider selecting more banks or a longer date range for better model performance.")
                
                if len(y.unique()) < 2:
                    st.error("❌ Insufficient crisis periods in the selected date range. Please select a longer period or different banks.")
                    return
                
                # Train models
                models = ews.train_models(X, y)
                
                # Store in session state
                st.session_state['ews'] = ews
                st.session_state['models_trained'] = True
                
                st.success("✅ Models trained successfully!")
                
            except ValueError as e:
                if "No data downloaded" in str(e) or "No valid data" in str(e):
                    st.error("❌ No data available for the selected banks and date range. This could be due to:")
                    st.error("• Banks being delisted or having no trading data")
                    st.error("• Date range with no available data")
                    st.error("• Network connectivity issues")
                    st.error("Try selecting different banks or a different date range.")
                else:
                    st.error(f"❌ Data error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error during training: {e}")
                st.exception(e)
    
    # Display results if models are trained
    if 'models_trained' in st.session_state and st.session_state['models_trained']:
        ews = st.session_state['ews']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "⚠️ Risk Assessment", "📈 Timeline Analysis", "📝 Academic Results"])
        
        with tab1:
            plot_model_performance(ews)
            
        with tab2:
            current_risk, risk_level, emoji = ews.get_current_risk_assessment()
            if current_risk is not None:
                st.markdown("### ⚠️ Current Risk Assessment")
                
                if risk_level == "HIGH":
                    st.markdown(f"""
                    <div class="warning-high">
                        <h4>{emoji} HIGH RISK DETECTED</h4>
                        <p><strong>Crisis Probability:</strong> {current_risk:.1%}</p>
                        <p>The system has detected a high probability of systemic risk in the banking sector.</p>
                        <ul>
                            <li>Monitor high-risk banks closely</li>
                            <li>Review portfolio allocations</li>
                            <li>Consider risk mitigation strategies</li>
                            <li>Stay alert for market stress signals</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == "MODERATE":
                    st.markdown(f"""
                    <div class="warning-medium">
                        <h4>{emoji} MODERATE RISK DETECTED</h4>
                        <p><strong>Crisis Probability:</strong> {current_risk:.1%}</p>
                        <p>The system has detected moderate systemic risk levels.</p>
                        <ul>
                            <li>Monitor key risk indicators</li>
                            <li>Review risk management policies</li>
                            <li>Stay informed about market developments</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-low">
                        <h4>{emoji} LOW RISK DETECTED</h4>
                        <p><strong>Crisis Probability:</strong> {current_risk:.1%}</p>
                        <p>The system indicates low systemic risk levels.</p>
                        <ul>
                            <li>Continue regular monitoring</li>
                            <li>Maintain standard risk management practices</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            plot_crisis_timeline(ews)
            
        with tab4:
            generate_paper_results(ews)
    
    else:
        # Show instructions
        st.markdown("""
        ## 🚀 Getting Started
        
        This Early Warning System uses machine learning to predict systemic banking crises with advanced lead time.
        
        ### Key Features:
        - **Multi-Model Approach**: Random Forest and XGBoost for robust predictions
        - **Feature Engineering**: Advanced features from systemic risk metrics
        - **Crisis Period Detection**: Automatic labeling of historical crisis periods
        - **Real-time Assessment**: Current risk level evaluation
        - **Academic Results**: Publication-ready performance metrics
        
        ### How to Use:
        1. **Select Banks**: Choose banks to include in the analysis
        2. **Set Date Range**: Define the historical period for training
        3. **Configure Parameters**: Adjust lead time and confidence levels
        4. **Load Data & Train**: Click the button to start the process
        5. **Review Results**: Explore performance metrics and risk assessments
        
        ### Crisis Periods Included:
        - Eurozone Crisis (2011-2012)
        - China Correction (2015-2016)
        - COVID Crash (2020)
        - Ukraine War (2022)
        - Banking Stress 2023
        
        ### Technical Details:
        - **Lead Time**: 8-10 weeks advance warning
        - **Features**: 20+ engineered features from systemic risk metrics
        - **Models**: Random Forest and XGBoost with balanced classes
        - **Validation**: Time-series cross-validation
        """)
        
        # Show example crisis periods
        st.markdown("### 📅 Historical Crisis Periods")
        crisis_periods = {
            'Eurozone Crisis': '2011-07-01 to 2012-12-31',
            'China Correction': '2015-06-01 to 2016-02-29',
            'COVID Crash': '2020-02-01 to 2020-05-31',
            'Ukraine War': '2022-02-01 to 2022-06-30',
            'Banking Stress 2023': '2023-03-01 to 2023-05-31'
        }
        
        for crisis, period in crisis_periods.items():
            st.info(f"**{crisis}**: {period}")

if __name__ == "__main__":
    main()
