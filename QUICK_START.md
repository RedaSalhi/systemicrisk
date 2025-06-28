# 🚀 Quick Start Guide

## For Server Environments (Linux/Cloud)

If you're getting Fortran compilation errors, use this:

```bash
# 1. Run the automatic server fix
python fix_server_install.py

# 2. Start the application
streamlit run app.py
```

## For Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the application
streamlit run app.py
```

## If You Still Have Issues

### Option 1: Use Conda (Recommended)
```bash
conda create -n systemicrisk python=3.11
conda activate systemicrisk
conda install -c conda-forge scipy pandas numpy scikit-learn statsmodels streamlit plotly yfinance
```

### Option 2: Use Server Requirements
```bash
pip install -r requirements-server.txt
```

### Option 3: Install Fortran Compiler
```bash
# Ubuntu/Debian
sudo apt-get install gfortran

# CentOS/RHEL
sudo yum install gcc-gfortran
```

## 🎯 What You'll See

1. **Landing Page**: Overview of the systemic risk platform
2. **Navigation**: Use the sidebar to access different pages:
   - 📊 Dashboard: Main analysis interface
   - 📚 Methodology: EVT explanation
   - 🤖 Machine Learning: ML early warning system

3. **Dashboard Features**:
   - Select banks by region
   - Choose date ranges
   - Set confidence levels
   - View risk metrics and alerts

## 🔧 Troubleshooting

- **Port already in use**: Change port with `streamlit run app.py --server.port 8502`
- **Memory issues**: Reduce number of selected banks
- **Slow loading**: Data fetching takes time, be patient

## 📞 Need Help?

Check the main README.md for detailed documentation and troubleshooting. 