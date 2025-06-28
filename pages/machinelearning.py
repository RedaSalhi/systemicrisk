import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import CRISIS_PERIODS, REGION_MAP

st.set_page_config(
    page_title="Machine Learning",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .model-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .performance-card {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #22C55E;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
        margin: 0.5rem 0;
    }
    .feature-box {
        background-color: #FFF7ED;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #FDBA74;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SystemicRiskML:
    def __init__(self):
        self.crisis_periods = CRISIS_PERIODS
        self.models = {}
        self.features_df = None
        self.labels = None

    def create_crisis_labels(self, metrics_data, lead_weeks=8):
        """Create binary crisis labels with lead time"""
        dates = metrics_data.index.get_level_values(0).unique().sort_values()
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

        return labels

    def engineer_features(self, metrics_data):
        """Engineer features from systemic risk metrics"""
        st.info("🔧 Engineering features from systemic risk metrics...")
        
        features_list = []
        dates = metrics_data.index.get_level_values(0).unique().sort_values()
        
        progress_bar = st.progress(0)
        
        for i, date in enumerate(dates):
            if i % 20 == 0:
                progress_bar.progress(i / len(dates))
            
            try:
                date_data = metrics_data.loc[date]
                feature_dict = {'Date': date}

                # Basic aggregate features
                if 'Beta_T' in date_data.columns:
                    beta_values = date_data['Beta_T'].dropna()
                    if len(beta_values) > 0:
                        feature_dict.update({
                            'beta_mean': beta_values.mean(),
                            'beta_std': beta_values.std(),
                            'beta_max': beta_values.max(),
                            'beta_75pct': beta_values.quantile(0.75),
                            'beta_90pct': beta_values.quantile(0.90),
                            'beta_high_risk_count': (beta_values > 2.0).sum(),
                            'beta_skew': beta_values.skew() if len(beta_values) > 2 else 0,
                            'beta_kurt': beta_values.kurtosis() if len(beta_values) > 3 else 0
                        })

                # VaR features
                if 'VaR_95' in date_data.columns:
                    var_values = date_data['VaR_95'].dropna()
                    if len(var_values) > 0:
                        feature_dict.update({
                            'var_mean': var_values.mean(),
                            'var_std': var_values.std(),
                            'var_extreme_count': (var_values < -0.1).sum(),
                            'var_max': var_values.max()
                        })

                # Tail dependence features
                if 'Tau_95' in date_data.columns:
                    tau_values = date_data['Tau_95'].dropna()
                    if len(tau_values) > 0:
                        feature_dict.update({
                            'tau_mean': tau_values.mean(),
                            'tau_max': tau_values.max(),
                            'tau_std': tau_values.std(),
                            'tau_high_count': (tau_values > 0.7).sum()
                        })

                # Hill estimator features
                if 'Hill_95' in date_data.columns:
                    hill_values = date_data['Hill_95'].dropna()
                    if len(hill_values) > 0:
                        feature_dict.update({
                            'hill_mean': hill_values.mean(),
                            'hill_max': hill_values.max(),
                            'hill_std': hill_values.std()
                        })

                # Regional breakdown
                for region in ['Americas', 'Europe', 'Asia/Pacific']:
                    region_banks = [bank for bank, r in REGION_MAP.items() if r == region]
                    region_data = date_data[date_data.index.isin(region_banks)]
                    
                    if len(region_data) > 0 and 'Beta_T' in region_data.columns:
                        region_beta = region_data['Beta_T'].dropna()
                        if len(region_beta) > 0:
                            feature_dict[f'beta_mean_{region.lower().replace("/", "_")}'] = region_beta.mean()
                            feature_dict[f'beta_max_{region.lower().replace("/", "_")}'] = region_beta.max()

                features_list.append(feature_dict)

            except Exception as e:
                continue

        progress_bar.progress(1.0)
        progress_bar.empty()

        # Create DataFrame
        features_df = pd.DataFrame(features_list).set_index('Date')

        # Add rolling features
        st.info("🔄 Adding rolling and trend features...")
        for window in [4, 8, 12, 24]:
            if 'beta_mean' in features_df.columns:
                features_df[f'beta_mean_roll_{window}w'] = features_df['beta_mean'].rolling(window).mean()
                features_df[f'beta_std_roll_{window}w'] = features_df['beta_std'].rolling(window).mean()
                
                # Trend features
                features_df[f'beta_mean_trend_{window}w'] = features_df['beta_mean'].rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window and not np.isnan(x).any() else np.nan
                )

        # Cross-metric features
        if 'beta_mean' in features_df.columns and 'var_mean' in features_df.columns:
            features_df['beta_var_ratio'] = features_df['beta_mean'] / features_df['var_mean'].abs()
        
        if 'beta_mean' in features_df.columns and 'tau_mean' in features_df.columns:
            features_df['beta_tau_product'] = features_df['beta_mean'] * features_df['tau_mean']

        # Volatility features
        for col in ['beta_mean', 'var_mean', 'tau_mean']:
            if col in features_df.columns:
                features_df[f'{col}_volatility_4w'] = features_df[col].rolling(4).std()
                features_df[f'{col}_volatility_12w'] = features_df[col].rolling(12).std()

        # Clean data
        features_df = features_df.dropna(thresh=len(features_df.columns) * 0.5)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')

        self.features_df = features_df
        return features_df

    def prepare_ml_dataset(self, metrics_data, lead_weeks=8):
        """Prepare dataset for ML training"""
        features_df = self.engineer_features(metrics_data)
        labels = self.create_crisis_labels(metrics_data, lead_weeks)

        # Align data
        common_dates = features_df.index.intersection(labels.index)
        X = features_df.loc[common_dates]
        y = labels.loc[common_dates]

        self.labels = y
        return X, y

    def train_models(self, X, y):
        """Train ML models"""
        st.info("🤖 Training machine learning models...")
        
        # Time-based split (preserve temporal order)
        split_idx = int(len(X) * 0.75)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Handle class imbalance
        class_weight = 'balanced'
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_proba) if len(y_test.unique()) > 1 else 0.5

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)

        xgb_pred = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_proba) if len(y_test.unique()) > 1 else 0.5

        # Store results
        self.models = {
            'random_forest': {
                'model': rf_model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': rf_pred,
                'y_proba': rf_proba,
                'auc': rf_auc,
                'feature_importance': pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            },
            'xgboost': {
                'model': xgb_model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': xgb_pred,
                'y_proba': xgb_proba,
                'auc': xgb_auc,
                'feature_importance': pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            }
        }

        return self.models

@st.cache_data
def load_ml_data():
    """Load sample data for ML demonstration"""
    # Since we don't have the actual metrics data, create a realistic sample
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range('2011-01-01', '2024-12-31', freq='W-FRI')
    n_dates = len(dates)
    n_banks = 20  # Reduced for demo
    
    # Create sample metrics data
    data = []
    for i, date in enumerate(dates):
        for bank_idx in range(n_banks):
            bank_name = f"Bank_{bank_idx:02d}"
            
            # Add some trend and volatility
            trend = 0.001 * i / n_dates
            volatility = 0.02 + 0.01 * np.sin(2 * np.pi * i / 52)  # Annual cycle
            
            # Crisis periods (higher risk)
            crisis_multiplier = 1.0
            for crisis_name, (start, end) in CRISIS_PERIODS.items():
                if start <= date <= end:
                    crisis_multiplier = 1.5 + np.random.normal(0, 0.3)
                    break
            
            data.append({
                'Date': date,
                'Bank': bank_name,
                'Beta_T': max(0.1, np.random.lognormal(0, 0.5) * crisis_multiplier),
                'VaR_95': -abs(np.random.normal(0.04, volatility) * crisis_multiplier),
                'Tau_95': np.random.beta(2, 3) * crisis_multiplier,
                'Hill_95': np.random.gamma(2, 0.1)
            })
    
    df = pd.DataFrame(data).set_index(['Date', 'Bank'])
    return df

def main():
    st.title("🤖 Machine Learning Early Warning System")
    st.markdown("**Crisis prediction using systemic risk features and advanced ML algorithms**")
    
    # Initialize ML system
    ml_system = SystemicRiskML()
    
    # Sidebar controls
    st.sidebar.header("🎛️ ML Controls")
    
    # Data options
    data_source = st.sidebar.selectbox(
        "📊 Data Source",
        ["Demo Data (Sample)", "Upload Metrics", "Real Data (if available)"]
    )
    
    if data_source == "Demo Data (Sample)":
        st.info("📊 Using simulated sample data for demonstration")
        metrics_data = load_ml_data()
    else:
        st.warning("📁 Please upload your metrics data or connect to real data source")
        st.stop()
    
    # ML parameters
    lead_weeks = st.sidebar.slider("⏰ Lead Time (weeks)", 4, 16, 8)
    test_split = st.sidebar.slider("📊 Test Split (%)", 10, 40, 25)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Overview", 
        "🔧 Feature Engineering", 
        "🤖 Model Training", 
        "📊 Performance", 
        "🔮 Predictions"
    ])
    
    with tab1:
        show_ml_overview(ml_system, metrics_data, lead_weeks)
    
    with tab2:
        show_feature_engineering(ml_system, metrics_data)
    
    with tab3:
        show_model_training(ml_system, metrics_data, lead_weeks)
    
    with tab4:
        show_model_performance(ml_system)
    
    with tab5:
        show_predictions(ml_system, metrics_data)

def show_ml_overview(ml_system, metrics_data, lead_weeks):
    """ML overview and crisis labeling"""
    
    st.subheader("🎯 Early Warning System Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        The Machine Learning Early Warning System uses engineered features from systemic risk metrics 
        to predict banking crises with a configurable lead time.
        
        ### 🔬 Methodology
        
        1. **Feature Engineering**: Extract statistical features from rolling systemic risk metrics
        2. **Crisis Labeling**: Label crisis periods and pre-crisis periods based on historical events
        3. **Model Training**: Train Random Forest and XGBoost classifiers with temporal splits
        4. **Performance Evaluation**: Assess predictive performance using precision, recall, and AUC
        5. **Real-time Prediction**: Generate crisis probability forecasts for current conditions
        
        ### 📊 Crisis Definition
        
        Binary classification where 1 indicates crisis/pre-crisis periods:
        - **Crisis Periods**: Actual financial stress events
        - **Pre-crisis Periods**: N weeks before crisis onset (lead time)
        - **Normal Periods**: All other time periods
        """)
    
    with col2:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Dataset Statistics")
        
        total_weeks = len(metrics_data.index.get_level_values(0).unique())
        total_banks = len(metrics_data.index.get_level_values(1).unique())
        
        st.metric("Total Weeks", f"{total_weeks:,}")
        st.metric("Banks Analyzed", f"{total_banks}")
        st.metric("Lead Time", f"{lead_weeks} weeks")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Crisis period visualization
    st.markdown("### ⚠️ Crisis Period Labeling")
    
    labels = ml_system.create_crisis_labels(metrics_data, lead_weeks)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Crisis Timeline")
        
        crisis_data = []
        for date in labels.index:
            crisis_data.append({
                'Date': date,
                'Crisis_Label': labels[date],
                'Type': 'Crisis/Pre-Crisis' if labels[date] == 1 else 'Normal'
            })
        
        crisis_df = pd.DataFrame(crisis_data)
        
        fig = px.scatter(crisis_df, x='Date', y='Crisis_Label', color='Type',
                        title=f"Crisis Labels (Lead Time: {lead_weeks} weeks)",
                        labels={'Crisis_Label': 'Crisis Indicator'})
        fig.update_traces(marker=dict(size=3))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Crisis Statistics")
        
        crisis_count = labels.sum()
        total_count = len(labels)
        crisis_ratio = crisis_count / total_count
        
        st.metric("Crisis Periods", f"{crisis_count} weeks")
        st.metric("Normal Periods", f"{total_count - crisis_count} weeks")
        st.metric("Crisis Ratio", f"{crisis_ratio:.1%}")
        
        # Crisis breakdown by period
        st.markdown("**Crisis Breakdown:**")
        for crisis_name, (start, end) in ml_system.crisis_periods.items():
            crisis_weeks = ((labels.index >= start) & (labels.index <= end)).sum()
            st.write(f"• {crisis_name.replace('_', ' ').title()}: {crisis_weeks} weeks")

def show_feature_engineering(ml_system, metrics_data):
    """Feature engineering interface"""
    
    st.subheader("🔧 Feature Engineering")
    
    if st.button("🚀 Engineer Features"):
        with st.spinner("🔧 Engineering features..."):
            features_df = ml_system.engineer_features(metrics_data)
        
        st.success(f"✅ Created {len(features_df.columns)} features for {len(features_df)} time periods")
        
        # Feature overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Feature Categories")
            
            feature_categories = {
                'Beta Features': [col for col in features_df.columns if 'beta' in col],
                'VaR Features': [col for col in features_df.columns if 'var' in col],
                'Tail Dependence': [col for col in features_df.columns if 'tau' in col],
                'Hill Features': [col for col in features_df.columns if 'hill' in col],
                'Rolling Features': [col for col in features_df.columns if 'roll' in col],
                'Trend Features': [col for col in features_df.columns if 'trend' in col],
                'Regional Features': [col for col in features_df.columns if any(region in col for region in ['americas', 'europe', 'asia'])]
            }
            
            for category, features in feature_categories.items():
                if features:
                    st.markdown(f"**{category}**: {len(features)} features")
        
        with col2:
            st.markdown("#### Feature Preview")
            st.dataframe(features_df.head(10), use_container_width=True)
        
        # Feature correlation heatmap
        st.markdown("#### Feature Correlation Matrix")
        
        # Select key features for correlation
        key_features = [col for col in features_df.columns if not any(x in col for x in ['roll', 'trend', 'volatility'])]
        key_features = key_features[:15]  # Limit to 15 for visibility
        
        if len(key_features) > 1:
            corr_matrix = features_df[key_features].corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, ax=ax, fmt='.2f')
            plt.title("Key Feature Correlations")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
        
        # Feature distributions
        st.markdown("#### Feature Distributions")
        
        numeric_features = features_df.select_dtypes(include=[np.number]).columns[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(numeric_features):
            if i < len(axes):
                axes[i].hist(features_df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature}')
                axes[i].grid(True, alpha=0.3)
        
        for i in range(len(numeric_features), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    else:
        st.info("👆 Click 'Engineer Features' to extract ML features from systemic risk metrics")

def show_model_training(ml_system, metrics_data, lead_weeks):
    """Model training interface"""
    
    st.subheader("🤖 Model Training")
    
    if ml_system.features_df is None:
        st.warning("⚠️ Please engineer features first in the 'Feature Engineering' tab")
        return
    
    if st.button("🚀 Train Models"):
        with st.spinner("🤖 Training ML models..."):
            X, y = ml_system.prepare_ml_dataset(metrics_data, lead_weeks)
            models = ml_system.train_models(X, y)
        
        st.success("✅ Models trained successfully!")
        
        # Training summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="performance-card">', unsafe_allow_html=True)
            st.markdown("#### Random Forest")
            rf_results = models['random_forest']
            st.metric("AUC Score", f"{rf_results['auc']:.3f}")
            st.metric("Training Samples", f"{len(rf_results['X_train'])}")
            st.metric("Test Samples", f"{len(rf_results['X_test'])}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="performance-card">', unsafe_allow_html=True)
            st.markdown("#### XGBoost")
            xgb_results = models['xgboost']
            st.metric("AUC Score", f"{xgb_results['auc']:.3f}")
            st.metric("Training Samples", f"{len(xgb_results['X_train'])}")
            st.metric("Test Samples", f"{len(xgb_results['X_test'])}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model comparison
        st.markdown("#### Model Comparison")
        
        comparison_data = []
        for model_name, results in models.items():
            y_test, y_pred = results['y_test'], results['y_pred']
            
            if len(y_test.unique()) > 1:
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
            else:
                precision = recall = f1 = 0.0
            
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC': results['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.round(3), use_container_width=True)
    
    else:
        st.info("👆 Click 'Train Models' to train Random Forest and XGBoost classifiers")

def show_model_performance(ml_system):
    """Model performance analysis"""
    
    st.subheader("📊 Model Performance Analysis")
    
    if not ml_system.models:
        st.warning("⚠️ Please train models first in the 'Model Training' tab")
        return
    
    # Model selection
    model_name = st.selectbox(
        "🔍 Select Model for Analysis",
        options=list(ml_system.models.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    model_results = ml_system.models[model_name]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve
        st.markdown("#### ROC Curve")
        
        y_test = model_results['y_test']
        y_proba = model_results['y_proba']
        
        if len(y_test.unique()) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = model_results['auc']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_score:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ ROC curve not available - insufficient class diversity in test set")
    
    with col2:
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        
        y_test = model_results['y_test']
        y_pred = model_results['y_pred']
        
        if len(y_test.unique()) > 1:
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        else:
            st.info("⚠️ Confusion matrix not available - insufficient class diversity")
    
    # Feature Importance
    st.markdown("#### Feature Importance")
    
    importance = model_results['feature_importance'].head(15)
    
    fig = px.bar(
        x=importance.values,
        y=importance.index,
        orientation='h',
        title=f"Top 15 Features - {model_name.replace('_', ' ').title()}",
        labels={'x': 'Importance', 'y': 'Features'}
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    st.markdown("#### Top 20 Most Important Features")
    importance_df = model_results['feature_importance'].head(20).reset_index()
    importance_df.columns = ['Feature', 'Importance']
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    st.dataframe(importance_df[['Rank', 'Feature', 'Importance']].round(4), use_container_width=True)

def show_predictions(ml_system, metrics_data):
    """Current predictions and risk assessment"""
    
    st.subheader("🔮 Crisis Probability Predictions")
    
    if not ml_system.models:
        st.warning("⚠️ Please train models first in the 'Model Training' tab")
        return
    
    # Current risk assessment
    if ml_system.features_df is not None and len(ml_system.features_df) > 0:
        
        # Get latest features
        latest_features = ml_system.features_df.tail(1)
        
        col1, col2, col3 = st.columns(3)
        
        # Random Forest prediction
        with col1:
            rf_model = ml_system.models['random_forest']['model']
            rf_prob = rf_model.predict_proba(latest_features)[0, 1]
            
            risk_level_rf = "🔴 HIGH" if rf_prob > 0.7 else "🟡 MODERATE" if rf_prob > 0.4 else "🟢 LOW"
            
            st.markdown('<div class="performance-card">', unsafe_allow_html=True)
            st.markdown("#### Random Forest")
            st.metric("Crisis Probability", f"{rf_prob:.1%}")
            st.markdown(f"**Risk Level**: {risk_level_rf}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # XGBoost prediction
        with col2:
            xgb_model = ml_system.models['xgboost']['model']
            xgb_prob = xgb_model.predict_proba(latest_features)[0, 1]
            
            risk_level_xgb = "🔴 HIGH" if xgb_prob > 0.7 else "🟡 MODERATE" if xgb_prob > 0.4 else "🟢 LOW"
            
            st.markdown('<div class="performance-card">', unsafe_allow_html=True)
            st.markdown("#### XGBoost")
            st.metric("Crisis Probability", f"{xgb_prob:.1%}")
            st.markdown(f"**Risk Level**: {risk_level_xgb}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Ensemble prediction
        with col3:
            ensemble_prob = (rf_prob + xgb_prob) / 2
            risk_level_ens = "🔴 HIGH" if ensemble_prob > 0.7 else "🟡 MODERATE" if ensemble_prob > 0.4 else "🟢 LOW"
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("#### Ensemble")
            st.metric("Crisis Probability", f"{ensemble_prob:.1%}")
            st.markdown(f"**Risk Level**: {risk_level_ens}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Historical predictions
        st.markdown("#### Historical Prediction Performance")
        
        # Get predictions for test period
        rf_results = ml_system.models['random_forest']
        X_test = rf_results['X_test']
        y_test = rf_results['y_test']
        
        # Generate predictions for visualization
        rf_test_proba = ml_system.models['random_forest']['model'].predict_proba(X_test)[:, 1]
        xgb_test_proba = ml_system.models['xgboost']['model'].predict_proba(X_test)[:, 1]
        
        pred_data = []
        for i, date in enumerate(X_test.index):
            pred_data.append({
                'Date': date,
                'Actual': y_test.iloc[i],
                'RF_Probability': rf_test_proba[i],
                'XGB_Probability': xgb_test_proba[i],
                'Ensemble': (rf_test_proba[i] + xgb_test_proba[i]) / 2
            })
        
        pred_df = pd.DataFrame(pred_data)
        
        # Plot predictions vs actual
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Crisis Probabilities", "Actual Crisis Labels"),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(x=pred_df['Date'], y=pred_df['RF_Probability'], 
                      mode='lines', name='Random Forest', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=pred_df['Date'], y=pred_df['XGB_Probability'], 
                      mode='lines', name='XGBoost', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=pred_df['Date'], y=pred_df['Ensemble'], 
                      mode='lines', name='Ensemble', line=dict(color='green', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=pred_df['Date'], y=pred_df['Actual'], 
                      mode='markers', name='Actual Crisis', 
                      marker=dict(color='orange', size=6)),
            row=2, col=1
        )
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=1, col=1)
        
        fig.update_layout(height=600, title="Model Predictions vs Actual Crisis Events")
        fig.update_yaxes(title_text="Crisis Probability", row=1, col=1)
        fig.update_yaxes(title_text="Crisis Label", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("#### 🔍 Key Insights")
        
        insights = [
            f"**Current Ensemble Risk**: {ensemble_prob:.1%} crisis probability",
            f"**Model Agreement**: RF and XGBoost predictions differ by {abs(rf_prob - xgb_prob):.1%}",
            f"**Test Performance**: Average AUC of {np.mean([rf_results['auc'], ml_system.models['xgboost']['auc']]):.3f}",
            f"**Feature Count**: {len(latest_features.columns)} engineered features used",
            f"**Data Recency**: Latest prediction based on {latest_features.index[0].strftime('%Y-%m-%d')} data"
        ]
        
        for insight in insights:
            st.markdown(f"• {insight}")
    
    else:
        st.info("⚠️ No features available for prediction. Please run feature engineering first.")

if __name__ == "__main__":
    main()
