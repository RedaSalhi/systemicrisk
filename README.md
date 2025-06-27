# Systemic Risk Analysis Dashboard

A comprehensive banking systemic risk analysis platform using Extreme Value Theory (EVT) with real-time data from Yahoo Finance.

## 🚀 Features

- **Real-time Data**: Live banking data from Yahoo Finance for 28 global banks
- **User-Configurable Bank Selection**: Choose from banks across Americas, Europe, and Asia/Pacific regions
- **Extreme Value Theory Analysis**: Advanced statistical methods for tail risk assessment
- **Interactive Dashboard**: Streamlit-based interface with multiple visualization options
- **Systemic Risk Metrics**: VaR, Hill Estimator, Tail Dependence, and Systemic Beta calculations
- **Regional Analysis**: Compare risk profiles across different geographical regions
- **Risk Alerts**: Automated identification of high and medium-risk banks

## 📊 Available Banks

### 🌎 Americas
- JPMorgan Chase
- Citigroup
- Bank of America
- Wells Fargo
- Goldman Sachs
- Morgan Stanley
- Bank of New York Mellon
- State Street
- Royal Bank of Canada
- Toronto Dominion

### 🇪🇺 Europe
- HSBC
- Barclays
- BNP Paribas
- Groupe Crédit Agricole
- ING
- Deutsche Bank
- Santander
- Société Générale
- UBS
- Standard Chartered

### 🌏 Asia/Pacific
- Agricultural Bank of China
- Bank of China
- China Construction Bank
- ICBC
- Bank of Communications
- Mitsubishi UFJ FG
- Mizuho FG
- Sumitomo Mitsui FG

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd systemicrisk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 🔧 Troubleshooting SciPy Import Error

If you encounter the error:
```
ImportError: cannot import name '_lazywhere' from 'scipy._lib._util'
```

This is a known compatibility issue with Python 3.13 and newer versions of SciPy.

Building SciPy on Python 3.13 requires a Fortran compiler such as `gfortran`. You can install it on Debian-based systems with:
```bash
sudo apt-get install gfortran
```

Here are the solutions:

#### Option 1: Automatic Fix (Recommended)
Run the provided fix script:
```bash
python fix_scipy_issue.py
```

#### Option 2: Manual Fix
Use the fixed requirements file:
```bash
pip install -r requirements-fixed.txt
```

#### Option 3: Manual Package Installation
```bash
pip uninstall -y scipy numpy pandas scikit-learn statsmodels
pip install numpy==1.25.2
pip install scipy==1.11.4
pip install pandas==2.1.4
pip install scikit-learn==1.3.2
pip install statsmodels==0.14.0
pip install -r requirements.txt
```

## 🚀 Usage

### Interactive Dashboard

Run the main dashboard (or start from the landing page with `streamlit run app.py`):
```bash
streamlit run pages/dashboard.py
```

The dashboard provides:
- **Configuration Panel**: Select banks, date range, and confidence levels
- **Overview Tab**: Summary statistics and latest risk metrics
- **Bank Analysis Tab**: Individual bank risk profiles and time series
- **Time Series Tab**: Historical analysis of selected metrics
- **Regional Analysis Tab**: Comparative analysis across regions
- **Risk Alerts Tab**: Automated risk identification and alerts

### Programmatic Usage

Use the data processor directly in your code:

```python
from data_processor import process_banking_data

# Select banks for analysis
selected_banks = ['JPMorgan Chase', 'HSBC', 'Deutsche Bank']

# Process data
processor = process_banking_data(
    selected_banks, 
    start_date='2020-01-01', 
    end_date='2024-12-31'
)

# Get latest metrics
latest_metrics = processor.get_latest_metrics(0.95)
print(latest_metrics)

# Get time series for specific bank
jpm_beta = processor.get_bank_time_series('JPMorgan Chase', 'Beta_T', 0.95)
print(jpm_beta)
```

### Example Script

Run the example script to see the data processor in action:
```bash
python example_usage.py
```

## 📈 Key Metrics

### Systemic Beta (βT)
- Measures a bank's contribution to systemic risk
- Values > 2.0 indicate high systemic risk
- Values > 1.5 indicate medium risk
- Values ≤ 1.5 indicate low risk

### Value at Risk (VaR)
- Measures potential losses at specified confidence levels
- Available at 95% and 99% confidence levels
- Higher values indicate greater risk

### Tail Dependence (τ)
- Measures correlation between bank and market extreme losses
- Values closer to 1 indicate stronger systemic connections
- Critical for understanding contagion risk

### Hill Estimator (ξ)
- Estimates the tail index of return distributions
- Higher values indicate heavier tails (more extreme events)
- Essential for EVT-based risk modeling

## 🔧 Technical Details

### Data Processing
- **Frequency**: Weekly returns (Friday close prices)
- **Window Size**: 52 weeks (1 year) rolling window
- **Data Source**: Yahoo Finance via yfinance library
- **Regional Indices**: S&P 500, FTSE 100, DAX, Nikkei 225, etc.

### Methodology
The analysis uses Extreme Value Theory with:
1. **Hill Estimator**: For tail index estimation
2. **Tail Dependence**: For measuring extreme event correlations
3. **Systemic Beta**: For quantifying systemic risk contribution
4. **Rolling Windows**: For time-varying risk assessment

### Confidence Levels
- **95%**: Standard risk assessment level
- **99%**: Extreme risk assessment level

## 📋 Files Structure

```
systemicrisk/
├── pages/
│   ├── dashboard.py          # Main Streamlit dashboard
│   ├── methodology.py       # EVT methodology explainer
│   └── machinelearning.py   # ML early warning system
├── data_processor.py      # Core data processing module
├── example_usage.py      # Example script
├── app.py               # Main landing page
├── requirements.txt     # Python dependencies
├── requirements-fixed.txt # Fixed versions for compatibility
├── fix_scipy_issue.py   # Automatic fix script
└── README.md           # This file
```

## 🎯 Use Cases

- **Regulatory Compliance**: Monitor systemic risk for regulatory reporting
- **Risk Management**: Identify high-risk banks and potential contagion
- **Investment Analysis**: Assess banking sector risk for portfolio decisions
- **Academic Research**: Study systemic risk patterns and methodologies
- **Policy Making**: Inform financial stability policies

## ⚠️ Risk Thresholds

- **High Risk**: Systemic Beta > 2.0
- **Medium Risk**: 1.5 < Systemic Beta ≤ 2.0
- **Low Risk**: Systemic Beta ≤ 1.5

## 🔄 Data Updates

The dashboard automatically downloads the latest data when you click "Load Data". Data is fetched from Yahoo Finance and includes:
- Daily closing prices for all selected banks
- Corresponding regional market indices
- Weekly return calculations
- Rolling window risk metrics

## 📝 Notes

- Data loading may take several minutes depending on the number of selected banks
- Some banks may have limited data availability
- The analysis uses weekly data to reduce noise and focus on systematic patterns
- All calculations are based on the accurate methodology provided

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.