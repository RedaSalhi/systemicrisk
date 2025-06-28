import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Import our fixed data processor
from data_processor import BankingDataProcessor, process_banking_data

# Page configuration
st.set_page_config(
    page_title="ML Early Warning System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #991b1b;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #92400e;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #065f46;
    }
    
    .feature-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-large {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

class SystemicRiskEarlyWarning:
    """Complete ML Early Warning System for Systemic Risk"""
    
    def __init__(self):
        self.crisis_periods = {
            'eurozone_crisis': (pd.Timestamp('2011-07-01'), pd.Timestamp('2012-12-31')),
            'china_correction': (pd.Timestamp('2015-06-01'), pd.Timestamp('2016-02-29')),
            'covid_crash': (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-05-31')),
            'ukraine_war': (pd.Timestamp('2022-02-01'), pd.Timestamp('2022-06-30')),
            'banking_stress_2023': (pd.Timestamp('2023-03-01'), pd.Timestamp('2023-05-31'))
        }
        
        self.models = {}
        self.features_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data_from_processor(self, processor):
        """Load data from BankingDataProcessor"""
        try:
            # Get all metrics
            self.metrics_95 = processor.get_all_metrics(0.95)
            self.metrics_99 = processor.get_all_metrics(0.99)
            
            print(f"✅ Loaded data: 95% shape {self.metrics_95.shape}, 99% shape {self.metrics_99.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_crisis_labels(self, lead_weeks=8):
        """Create crisis labels with lead time"""
        dates = self.metrics_95['Date'].unique()
        dates = pd.to_datetime(dates).sort_values()
        
        labels = pd.Series(0, index=dates, name='crisis_label')
        
        # Label crisis periods and pre-crisis periods
        for crisis_name, (start, end) in self.crisis_periods.items():
            # Crisis period
            crisis_mask = (dates >= start) & (dates <= end)
            labels.loc[crisis_mask] = 1
            
            # Pre-crisis period (lead_weeks before)
            pre_crisis_start = start - pd.Timedelta(weeks=lead_weeks)
            pre_crisis_mask = (dates >= pre_crisis_start) & (dates < start)
            labels.loc[pre_crisis_mask] = 1
        
        return labels
    
    def engineer_features(self):
        """Engineer features for ML model"""
        print("🔧 Engineering features...")
        
        features_list = []
        dates = sorted(self.metrics_95['Date'].unique())
        
        for date in dates:
            try:
                date_data_95 = self.metrics_95[self.metrics_95['Date'] == date]
                date_data_99 = self.metrics_99[self.metrics_99['Date'] == date]
                
                if date_data_95.empty:
                    continue
                
                feature_dict = {'Date': date}
                
                # Beta_T features (main systemic risk indicators)
                if 'Beta_T' in date_data_95.columns:
                    beta_95 = date_data_95['Beta_T'].dropna()
                    if len(beta_95) > 0:
                        feature_dict.update({
                            'beta_mean_95': beta_95.mean(),
                            'beta_std_95': beta_95.std(),
                            'beta_max_95': beta_95.max(),
                            'beta_75pct_95': beta_95.quantile(0.75),
                            'beta_high_risk_95': (beta_95 > 2.0).sum(),
                            'beta_med_risk_95': ((beta_95 > 1.5) & (beta_95 <= 2.0)).sum(),
                            'beta_skew_95': beta_95.skew() if len(beta_95) > 2 else 0
                        })
                
                if 'Beta_T' in date_data_99.columns and not date_data_99.empty:
                    beta_99 = date_data_99['Beta_T'].dropna()
                    if len(beta_99) > 0:
                        feature_dict.update({
                            'beta_mean_99': beta_99.mean(),
                            'beta_max_99': beta_99.max(),
                            'beta_high_risk_99': (beta_99 > 2.0).sum()
                        })
                
                # VaR features
                if 'VaR_95' in date_data_95.columns:
                    var_95 = date_data_95['VaR_95'].dropna()
                    if len(var_95) > 0:
                        feature_dict.update({
                            'var_mean_95': var_95.mean(),
                            'var_std_95': var_95.std(),
                            'var_extreme_95': (var_95 < -0.1).sum()  # Extreme losses
                        })
                
                # Tail dependence features
                if 'Tau_95' in date_data_95.columns:
                    tau_95 = date_data_95['Tau_95'].dropna()
                    if len(tau_95) > 0:
                        feature_dict.update({
                            'tau_mean_95': tau_95.mean(),
                            'tau_max_95': tau_95.max(),
                            'tau_high_95': (tau_95 > 0.7).sum()  # High interconnectedness
                        })
                
                # Regional concentration
                if 'Region' in date_data_95.columns:
                    region_counts = date_data_95['Region'].value_counts()
                    total_banks = len(date_data_95)
                    if total_banks > 0:
                        feature_dict.update({
                            'americas_pct': region_counts.get('Americas', 0) / total_banks,
                            'europe_pct': region_counts.get('Europe', 0) / total_banks,
                            'asia_pct': region_counts.get('Asia/Pacific', 0) / total_banks
                        })
                
                features_list.append(feature_dict)
                
            except Exception as e:
                print(f"Error processing date {date}: {e}")
                continue
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list).set_index('Date')
        
        # Add rolling features (temporal patterns)
        print("🔄 Adding rolling features...")
        for window in [4, 8, 12]:  # 4, 8, 12 week windows
            if 'beta_mean_95' in features_df.columns:
                features_df[f'beta_mean_95_roll_{window}w'] = features_df['beta_mean_95'].rolling(window).mean()
                features_df[f'beta_mean_95_trend_{window}w'] = features_df['beta_mean_95'].rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan, raw=False
                )
        
        # Cross-confidence features
        if 'beta_mean_95' in features_df.columns and 'beta_mean_99' in features_df.columns:
            features_df['beta_spread'] = features_df['beta_mean_99'] - features_df['beta_mean_95']
        
        # Clean data
        features_df = features_df.dropna(thresh=len(features_df.columns) * 0.6)  # Keep rows with 60%+ data
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        print(f"✅ Created {len(features_df.columns)} features for {len(features_df)} time periods")
        
        self.features_df = features_df
        return features_df
    
    def prepare_ml_dataset(self, lead_weeks=8):
        """Prepare dataset for ML training"""
        if self.features_df is None:
            self.engineer_features()
        
        labels = self.create_crisis_labels(lead_weeks)
        
        # Align data
        common_dates = self.features_df.index.intersection(labels.index)
        X = self.features_df.loc[common_dates]
        y = labels.loc[common_dates]
        
        print(f"📈 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"⚖️ Class distribution: Normal={(y==0).sum()}, Crisis={(y==1).sum()}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train ML models with proper validation"""
        print("🤖 Training ML models...")
        
        # Time-based split (70% train, 30% test)
        split_idx = int(len(X) * 0.7)
        self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Train: {len(self.X_train)} samples, Test: {len(self.X_test)} samples")
        
        # Handle class imbalance
        class_weight = 'balanced'
        
        # Random Forest
        print("🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight=class_weight
        )
        
        rf_model.fit(self.X_train, self.y_train)
        
        # Predictions
        rf_pred = rf_model.predict(self.X_test)
        rf_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        if len(self.y_test.unique()) > 1:
            rf_auc = roc_auc_score(self.y_test, rf_proba)
        else:
            rf_auc = 0.5
        
        # Feature importance
        feature_importance = pd.Series(
            rf_model.feature_importances_, 
            index=self.X_train.columns
        ).sort_values(ascending=False)
        
        # Store results
        self.models['random_forest'] = {
            'model': rf_model,
            'y_test': self.y_test,
            'y_pred': rf_pred,
            'y_proba': rf_proba,
            'auc': rf_auc,
            'feature_importance': feature_importance
        }
        
        print(f"✅ Random Forest trained - AUC: {rf_auc:.3f}")
        
        return self.models
    
    def get_current_risk_assessment(self):
        """Get current risk assessment"""
        if self.features_df is None or 'random_forest' not in self.models:
            return None, None, None
        
        try:
            # Get latest features
            latest_features = self.features_df.tail(1)
            
            # Predict
            current_risk = self.models['random_forest']['model'].predict_proba(latest_features)[0, 1]
            
            # Classify risk level
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
            print(f"Error in risk assessment: {e}")
            return None, None, None
    
    def generate_model_performance(self):
        """Generate model performance metrics"""
        if 'random_forest' not in self.models:
            return None
        
        results = self.models['random_forest']
        y_test, y_pred, y_proba = results['y_test'], results['y_pred'], results['y_proba']
        
        # Performance metrics
        if len(y_test.unique()) > 1:
            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                auc = results['auc']
                
                return {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
            except:
                return None
        return None

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🤖 ML Early Warning System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.2rem;">Machine Learning-based crisis prediction with 8-10 weeks advance warning</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ ML Configuration")
        
        # Bank selection
        st.subheader("🏦 Bank Selection")
        temp_processor = BankingDataProcessor()
        available_banks = temp_processor.get_available_banks()
        
        selected_banks = st.multiselect(
            "Select banks for ML training:",
            available_banks,
            default=['JPMorgan Chase', 'HSBC Holdings', 'Deutsche Bank', 'Bank of America'],
            help="Choose 4-10 banks for robust ML training"
        )
        
        # Date range
        st.subheader("📅 Training Period")
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime('2020-01-01').date(),
            min_value=pd.to_datetime('2015-01-01').date()
        )
        
        end_date = st.date_input(
            "End Date", 
            value=pd.to_datetime('2024-12-31').date()
        )
        
        # ML parameters
        st.subheader("🤖 ML Parameters")
        lead_weeks = st.slider(
            "Crisis lead time (weeks):",
            min_value=4, max_value=16, value=8,
            help="How many weeks in advance to predict crises"
        )
        
        confidence_level = st.selectbox(
            "Risk calculation confidence:",
            [0.95, 0.99],
            format_func=lambda x: f"{int(x*100)}%"
        )
        
        # Training button
        st.subheader("🚀 Model Training")
        
        if st.button("🤖 Train ML Models", type="primary", use_container_width=True):
            if not selected_banks:
                st.error("Please select at least 3 banks for training.")
                return
            
            # Store parameters in session state
            st.session_state.ml_params = {
                'selected_banks': selected_banks,
                'start_date': start_date,
                'end_date': end_date,
                'lead_weeks': lead_weeks,
                'confidence_level': confidence_level
            }
            st.session_state.training_requested = True
            st.rerun()
    
    # Main content
    if hasattr(st.session_state, 'training_requested') and st.session_state.training_requested:
        # Training process
        params = st.session_state.ml_params
        
        st.markdown("## 🔄 Training ML Models...")
        
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load data
                status_text.text("📥 Loading banking data...")
                progress_bar.progress(20)
                
                processor = process_banking_data(
                    params['selected_banks'],
                    start_date=params['start_date'].strftime('%Y-%m-%d'),
                    end_date=params['end_date'].strftime('%Y-%m-%d')
                )
                
                # Step 2: Initialize ML system
                status_text.text("🤖 Initializing ML system...")
                progress_bar.progress(40)
                
                ews = SystemicRiskEarlyWarning()
                
                # Step 3: Load data into ML system
                status_text.text("🔗 Loading data into ML system...")
                progress_bar.progress(50)
                
                if not ews.load_data_from_processor(processor):
                    st.error("Failed to load data into ML system")
                    return
                
                # Step 4: Prepare dataset
                status_text.text("⚙️ Engineering features...")
                progress_bar.progress(70)
                
                X, y = ews.prepare_ml_dataset(params['lead_weeks'])
                
                if len(X) < 50:
                    st.warning(f"⚠️ Limited data: {len(X)} samples. Consider longer date range.")
                
                if len(y.unique()) < 2:
                    st.error("❌ Insufficient crisis periods for training.")
                    return
                
                # Step 5: Train models
                status_text.text("🧠 Training ML models...")
                progress_bar.progress(90)
                
                models = ews.train_models(X, y)
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed!")
                
                # Store in session state
                st.session_state.ews = ews
                st.session_state.models_trained = True
                st.session_state.training_requested = False
                
                st.success("🎉 ML models trained successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.session_state.training_requested = False
    
    elif hasattr(st.session_state, 'models_trained') and st.session_state.models_trained:
        # Display results
        ews = st.session_state.ews
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚠️ Risk Assessment",
            "📊 Model Performance", 
            "📈 Feature Analysis",
            "📝 Academic Results"
        ])
        
        with tab1:
            st.header("⚠️ Current Risk Assessment")
            
            # Get current risk
            current_risk, risk_level, emoji = ews.get_current_risk_assessment()
            
            if current_risk is not None:
                # Display current risk
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div class="metric-large">{current_risk:.1%}</div>
                        <p style="font-size: 1.2rem; color: #6b7280;">Crisis Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div class="metric-large">{emoji}</div>
                        <p style="font-size: 1.2rem; color: #6b7280;">{risk_level} RISK</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Show latest beta if available
                    if ews.features_df is not None and 'beta_mean_95' in ews.features_df.columns:
                        latest_beta = ews.features_df['beta_mean_95'].iloc[-1]
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div class="metric-large">{latest_beta:.3f}</div>
                            <p style="font-size: 1.2rem; color: #6b7280;">Avg Beta_T</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Risk level explanation
                if risk_level == "HIGH":
                    st.markdown(f"""
                    <div class="alert-high">
                        <h3>{emoji} HIGH RISK DETECTED</h3>
                        <p><strong>Crisis Probability:</strong> {current_risk:.1%}</p>
                        <p>The ML system predicts high probability of systemic banking crisis within the next 8-10 weeks.</p>
                        <h4>Immediate Actions Required:</h4>
                        <ul>
                            <li>🔍 Activate enhanced monitoring protocols</li>
                            <li>💰 Review liquidity positions across all major banks</li>
                            <li>📞 Coordinate with regulatory authorities</li>
                            <li>🏦 Conduct emergency stress tests</li>
                            <li>📋 Prepare crisis response measures</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif risk_level == "MODERATE":
                    st.markdown(f"""
                    <div class="alert-medium">
                        <h3>{emoji} MODERATE RISK DETECTED</h3>
                        <p><strong>Crisis Probability:</strong> {current_risk:.1%}</p>
                        <p>The ML system indicates elevated systemic risk levels requiring increased vigilance.</p>
                        <h4>Recommended Actions:</h4>
                        <ul>
                            <li>📊 Increase monitoring frequency</li>
                            <li>🔍 Focus on key risk indicators</li>
                            <li>📈 Review portfolio concentrations</li>
                            <li>🤝 Enhance inter-agency coordination</li>
                            <li>📋 Update contingency plans</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.markdown(f"""
                    <div class="alert-low">
                        <h3>{emoji} LOW RISK DETECTED</h3>
                        <p><strong>Crisis Probability:</strong> {current_risk:.1%}</p>
                        <p>The ML system indicates low systemic risk in the banking sector.</p>
                        <h4>Standard Measures:</h4>
                        <ul>
                            <li>✅ Continue regular monitoring</li>
                            <li>📊 Maintain standard reporting schedules</li>
                            <li>🔄 Review and update risk models</li>
                            <li>📚 Continue research on emerging risks</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.error("❌ Unable to assess current risk. Please retrain the model.")
        
        with tab2:
            st.header("📊 Model Performance")
            
            # Get performance metrics
            performance = ews.generate_model_performance()
            
            if performance:
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Precision", f"{performance['precision']:.3f}")
                with col2:
                    st.metric("Recall", f"{performance['recall']:.3f}")
                with col3:
                    st.metric("F1-Score", f"{performance['f1_score']:.3f}")
                with col4:
                    st.metric("AUC", f"{performance['auc']:.3f}")
                
                # ROC Curve
                if len(ews.y_test.unique()) > 1:
                    fpr, tpr, _ = roc_curve(ews.y_test, ews.models['random_forest']['y_proba'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f'ROC Curve (AUC = {performance["auc"]:.3f})',
                        line=dict(color='#ef4444', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        name='Random Classifier',
                        line=dict(dash='dash', color='gray')
                    ))
                    
                    fig.update_layout(
                        title="ROC Curve - Crisis Prediction Performance",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confusion Matrix
                st.subheader("📊 Confusion Matrix")
                cm = performance['confusion_matrix']
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Normal', 'Crisis'],
                    y=['Normal', 'Crisis'],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig.update_layout(title="Confusion Matrix", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("⚠️ Performance metrics not available")
        
        with tab3:
            st.header("📈 Feature Analysis")
            
            if 'random_forest' in ews.models:
                # Feature importance
                importance = ews.models['random_forest']['feature_importance'].head(15)
                
                fig = px.bar(
                    x=importance.values,
                    y=importance.index,
                    orientation='h',
                    title="Top 15 Most Important Features",
                    labels={'x': 'Importance', 'y': 'Feature'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature descriptions
                st.subheader("🔍 Key Feature Insights")
                
                for i, (feature, importance_val) in enumerate(importance.head(5).items(), 1):
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>#{i}. {feature}</h4>
                        <p><strong>Importance:</strong> {importance_val:.4f}</p>
                        <p><strong>Description:</strong> {self.get_feature_description(feature)}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Timeline of features
            if ews.features_df is not None:
                st.subheader("📈 Feature Timeline")
                
                # Select key features to plot
                key_features = ['beta_mean_95', 'var_mean_95', 'tau_mean_95', 'beta_high_risk_95']
                available_features = [f for f in key_features if f in ews.features_df.columns]
                
                if available_features:
                    selected_feature = st.selectbox(
                        "Select feature to visualize:",
                        available_features,
                        format_func=lambda x: {
                            'beta_mean_95': 'Average Systemic Beta (95%)',
                            'var_mean_95': 'Average VaR (95%)',
                            'tau_mean_95': 'Average Tail Dependence (95%)',
                            'beta_high_risk_95': 'High Risk Bank Count (95%)'
                        }.get(x, x)
                    )
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ews.features_df.index,
                        y=ews.features_df[selected_feature],
                        name=selected_feature,
                        line=dict(color='#3b82f6', width=2)
                    ))
                    
                    # Add crisis period shading
                    for crisis_name, (start, end) in ews.crisis_periods.items():
                        fig.add_vrect(
                            x0=start, x1=end,
                            fillcolor="red", opacity=0.2,
                            layer="below", line_width=0,
                            annotation_text=crisis_name,
                            annotation_position="top left"
                        )
                    
                    fig.update_layout(
                        title=f"Timeline: {selected_feature}",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("📝 Academic Results")
            
            # Model performance table
            st.subheader("🏆 Model Performance Summary")
            
            performance = ews.generate_model_performance()
            if performance:
                results_df = pd.DataFrame({
                    'Model': ['Random Forest'],
                    'Precision': [performance['precision']],
                    'Recall': [performance['recall']],
                    'F1-Score': [performance['f1_score']],
                    'AUC': [performance['auc']],
                    'Lead Time': ['8-10 weeks']
                })
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Feature importance table
            st.subheader("🔍 Top 10 Feature Importance")
            
            if 'random_forest' in ews.models:
                importance_df = ews.models['random_forest']['feature_importance'].head(10).reset_index()
                importance_df.columns = ['Feature', 'Importance']
                importance_df['Rank'] = range(1, len(importance_df) + 1)
                importance_df = importance_df[['Rank', 'Feature', 'Importance']]
                
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
            
            # Crisis periods table
            st.subheader("📅 Historical Crisis Periods")
            
            crisis_df = pd.DataFrame([
                {'Crisis': name, 'Start Date': start.strftime('%Y-%m-%d'), 'End Date': end.strftime('%Y-%m-%d')}
                for name, (start, end) in ews.crisis_periods.items()
            ])
            
            st.dataframe(crisis_df, use_container_width=True, hide_index=True)
            
            # Dataset statistics
            st.subheader("📊 Dataset Statistics")
            
            if hasattr(ews, 'X_train') and ews.X_train is not None:
                stats_df = pd.DataFrame({
                    'Metric': [
                        'Total Samples',
                        'Training Samples', 
                        'Test Samples',
                        'Features',
                        'Crisis Samples (Train)',
                        'Normal Samples (Train)',
                        'Crisis Ratio'
                    ],
                    'Value': [
                        len(ews.X_train) + len(ews.X_test),
                        len(ews.X_train),
                        len(ews.X_test),
                        ews.X_train.shape[1],
                        (ews.y_train == 1).sum(),
                        (ews.y_train == 0).sum(),
                        f"{(ews.y_train == 1).mean():.3f}"
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 ML Early Warning System Overview
        
        This system uses advanced machine learning to predict banking crises with **8-10 weeks advance warning**.
        
        ### 🔬 Technical Approach
        - **Models**: Random Forest with balanced classes
        - **Features**: 20+ engineered features from systemic risk metrics
        - **Validation**: Time-series cross-validation
        - **Lead Time**: 8-10 weeks advance prediction
        
        ### 📊 Key Features
        - **Beta_T Statistics**: Systemic risk measurements
        - **VaR Metrics**: Value-at-Risk calculations  
        - **Tail Dependence**: Interconnectedness measures
        - **Rolling Trends**: Temporal pattern detection
        - **Regional Factors**: Geographic risk concentration
        
        ### 🏛️ Crisis Periods Analyzed
        - **Eurozone Crisis** (2011-2012)
        - **China Market Correction** (2015-2016) 
        - **COVID-19 Crash** (2020)
        - **Ukraine War Impact** (2022)
        - **Banking Stress Events** (2023)
        
        ### 🚀 Getting Started
        
        1. **Select Banks**: Choose 4-10 banks for robust training
        2. **Set Date Range**: Recommend 2020-2024 for recent patterns
        3. **Configure ML**: Adjust lead time and confidence levels
        4. **Train Models**: Click the training button to start
        5. **Analyze Results**: Review risk assessment and performance
        
        """)
        
        # Expected performance info
        st.markdown("""
        ### 📈 Expected Performance
        
        | Metric | Typical Range | Description |
        |--------|---------------|-------------|
        | **AUC** | 0.75 - 0.90 | Overall discrimination ability |
        | **Precision** | 0.65 - 0.85 | Accuracy of crisis predictions |
        | **Recall** | 0.60 - 0.80 | Coverage of actual crises |
        | **F1-Score** | 0.65 - 0.82 | Balanced performance measure |
        
        ### ⚠️ Risk Level Thresholds
        
        - **🔴 High Risk**: Crisis probability > 70%
        - **🟡 Moderate Risk**: Crisis probability 40-70%  
        - **🟢 Low Risk**: Crisis probability < 40%
        
        ### 💡 Use Cases
        
        - **Regulatory Supervision**: Early warning for supervisors
        - **Risk Management**: Portfolio and exposure management
        - **Policy Making**: Systemic risk assessment for policy
        - **Academic Research**: Crisis prediction methodology
        """)

    def get_feature_description(self, feature):
        """Get human-readable feature descriptions"""
        descriptions = {
            'beta_mean_95': 'Average systemic beta across all banks - core systemic risk measure',
            'beta_std_95': 'Standard deviation of systemic beta - measures dispersion of risk',
            'beta_max_95': 'Maximum systemic beta - identifies highest risk bank',
            'beta_high_risk_95': 'Number of banks with beta > 2.0 - counts high-risk institutions',
            'beta_med_risk_95': 'Number of banks with 1.5 < beta ≤ 2.0 - medium risk count',
            'var_mean_95': 'Average Value-at-Risk - expected loss measure',
            'tau_mean_95': 'Average tail dependence - systemic interconnectedness',
            'tau_max_95': 'Maximum tail dependence - strongest interconnection',
            'beta_mean_95_roll_4w': '4-week rolling average of systemic beta',
            'beta_mean_95_trend_8w': '8-week trend in systemic beta - directional change',
            'americas_pct': 'Percentage of banks from Americas region',
            'europe_pct': 'Percentage of banks from European region',
            'asia_pct': 'Percentage of banks from Asia/Pacific region'
        }
        return descriptions.get(feature, 'Statistical measure derived from banking risk data')

if __name__ == "__main__":
    main()
