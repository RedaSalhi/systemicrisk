import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from scipy.stats import genpareto
import warnings
warnings.filterwarnings('ignore')

# Bank ticker to name mapping
BANK_DICT = {
    # Americas
    'JPM': 'JPMorgan Chase',
    'C': 'Citigroup',
    'BAC': 'Bank of America',
    'WFC': 'Wells Fargo',
    'GS': 'Goldman Sachs',
    'MS': 'Morgan Stanley',
    'BK': 'Bank of New York Mellon',
    'STT': 'State Street',
    'RY': 'Royal Bank of Canada',
    'TD': 'Toronto Dominion',
    # Europe
    'HSBA.L': 'HSBC',
    'BARC.L': 'Barclays',
    'BNP.PA': 'BNP Paribas',
    'ACA.PA': 'Groupe Crédit Agricole',
    'INGA.AS': 'ING',
    'DBK.DE': 'Deutsche Bank',
    'SAN.MC': 'Santander',
    'GLE.PA': 'Société Générale',
    'UBSG.SW': 'UBS',
    'STAN.L': 'Standard Chartered',
    # Asia/Pacific
    '1288.HK': 'Agricultural Bank of China',
    '3988.HK': 'Bank of China',
    '0939.HK': 'China Construction Bank',
    '1398.HK': 'ICBC',
    '3328.HK': 'Bank of Communications',
    '8306.T': 'Mitsubishi UFJ FG',
    '8411.T': 'Mizuho FG',
    '8316.T': 'Sumitomo Mitsui FG'
}

# Bank to regional index mapping
INDEX_MAP = {
    # US banks → S&P 500
    'JPMorgan Chase': '^GSPC',
    'Citigroup': '^GSPC',
    'Bank of America': '^GSPC',
    'Wells Fargo': '^GSPC',
    'Goldman Sachs': '^GSPC',
    'Morgan Stanley': '^GSPC',
    'Bank of New York Mellon': '^GSPC',
    'State Street': '^GSPC',
    # Canada → TSX Composite
    'Royal Bank of Canada': '^GSPTSE',
    'Toronto Dominion': '^GSPTSE',
    # France → CAC 40
    'BNP Paribas': '^FCHI',
    'Groupe Crédit Agricole': '^FCHI',
    'Société Générale': '^FCHI',
    # Spain → IBEX 35
    'Santander': '^IBEX',
    # UK → FTSE 100
    'HSBC': '^FTSE',
    'Barclays': '^FTSE',
    'Standard Chartered': '^FTSE',
    # Germany → DAX
    'Deutsche Bank': '^GDAXI',
    # Switzerland → SMI
    'UBS': '^SSMI',
    # Netherlands → AEX
    'ING': '^AEX',
    # China → Shanghai Composite
    'China Construction Bank': '000001.SS',
    'Agricultural Bank of China': '000001.SS',
    'ICBC': '000001.SS',
    'Bank of Communications': '000001.SS',
    'Bank of China': '000001.SS',
    # Japan → Nikkei 225
    'Mitsubishi UFJ FG': '^N225',
    'Sumitomo Mitsui FG': '^N225',
    'Mizuho FG': '^N225',
}

# Index ticker to name mapping
IDX_NAME_MAP = {
    '^GSPC': 'S&P 500',
    '^GSPTSE': 'TSX Composite',
    '^FCHI': 'CAC 40',
    '^IBEX': 'IBEX 35',
    '^FTSE': 'FTSE 100',
    '^GDAXI': 'DAX',
    '^SSMI': 'SMI',
    '^N225': 'Nikkei 225',
    '000001.SS': 'Shanghai Composite',
    '^AEX': 'AEX',
}

# Regional mapping
REGION_MAP = {
    # Americas
    'JPMorgan Chase': 'Americas',
    'Citigroup': 'Americas',
    'Bank of America': 'Americas',
    'Wells Fargo': 'Americas',
    'Goldman Sachs': 'Americas',
    'Morgan Stanley': 'Americas',
    'Bank of New York Mellon': 'Americas',
    'State Street': 'Americas',
    'Royal Bank of Canada': 'Americas',
    'Toronto Dominion': 'Americas',
    # Europe
    'HSBC': 'Europe',
    'Barclays': 'Europe',
    'BNP Paribas': 'Europe',
    'Groupe Crédit Agricole': 'Europe',
    'ING': 'Europe',
    'Deutsche Bank': 'Europe',
    'Santander': 'Europe',
    'Société Générale': 'Europe',
    'UBS': 'Europe',
    'Standard Chartered': 'Europe',
    # Asia/Pacific
    'Agricultural Bank of China': 'Asia/Pacific',
    'Bank of China': 'Asia/Pacific',
    'China Construction Bank': 'Asia/Pacific',
    'ICBC': 'Asia/Pacific',
    'Bank of Communications': 'Asia/Pacific',
    'Mitsubishi UFJ FG': 'Asia/Pacific',
    'Mizuho FG': 'Asia/Pacific',
    'Sumitomo Mitsui FG': 'Asia/Pacific'
}

@st.cache_data
def download_data(start_date='2011-01-01', end_date='2024-12-31'):
    """Download bank and index data with caching"""
    
    # Download bank data
    tickers = list(BANK_DICT.keys())
    raw_banks = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Remove tickers with no data
    no_data = [t for t in raw_banks.columns if raw_banks[t].dropna().empty]
    if no_data:
        raw_banks = raw_banks.drop(columns=no_data)
    
    # Rename to bank names
    available_banks = {k: v for k, v in BANK_DICT.items() if k in raw_banks.columns}
    raw_banks.rename(columns=available_banks, inplace=True)
    
    # Download index data
    index_tickers = list(set(INDEX_MAP.values()))
    raw_indices = yf.download(index_tickers, start=start_date, end=end_date)['Close']
    
    # Remove indices with no data
    no_data_idx = [t for t in raw_indices.columns if raw_indices[t].dropna().empty]
    if no_data_idx:
        raw_indices = raw_indices.drop(columns=no_data_idx)
    
    # Rename indices
    raw_indices.rename(columns=IDX_NAME_MAP, inplace=True)
    
    return raw_banks, raw_indices

def prepare_returns(price_data_banks, price_data_indices):
    """Convert prices to weekly log returns"""
    
    # Resample to weekly (Friday close)
    weekly_banks = price_data_banks.resample('W-FRI').last().ffill().dropna()
    weekly_indices = price_data_indices.resample('W-FRI').last().ffill().dropna()
    
    # Compute log returns
    bank_returns = np.log(weekly_banks / weekly_banks.shift(1)).dropna()
    index_returns = np.log(weekly_indices / weekly_indices.shift(1)).dropna()
    
    # Merge on dates
    combined = bank_returns.join(index_returns, how='inner')
    
    return combined, bank_returns.columns.tolist()

# EVT and Systemic Risk Functions
def rolling_var(x, alpha=0.95):
    """Calculate rolling Value-at-Risk"""
    return -np.percentile(x, 100*(1-alpha))

def hill_estimator(x, threshold_quantile=0.95, min_excesses=5):
    """Hill estimator for tail index"""
    candidate_q = np.linspace(0.90, threshold_quantile, 50)
    
    for q in reversed(candidate_q):
        p = 1 - q
        u = np.quantile(x, p)
        losses = -x[x < u]
        losses = losses[losses > 0]
        
        if len(losses) >= min_excesses:
            min_loss = losses.min()
            return np.mean(np.log(losses / min_loss))
    
    return np.nan

def tail_dependence(x, y, u=0.95):
    """Calculate tail dependence coefficient"""
    qx, qy = np.quantile(x, u), np.quantile(y, u)
    mask = x < qx  # left-tail dependence
    return np.sum(y[mask] < qy) / np.sum(mask) if np.sum(mask) > 0 else np.nan

def systemic_beta(x, y, u=0.95):
    """Calculate systemic beta as per van Oordt & Zhou (2018)"""
    VaR_x = rolling_var(x, alpha=u)
    VaR_y = rolling_var(y, alpha=u)
    xi_y = hill_estimator(y, threshold_quantile=u)
    tau = tail_dependence(x, y, u=u)
    
    if xi_y is None or xi_y == 0 or np.isnan(tau) or VaR_y == 0:
        return np.nan
    
    return (tau ** (1.0/xi_y)) * (VaR_x / VaR_y)

@st.cache_data
def compute_rolling_metrics(combined_data, bank_names, window_size=52):
    """Compute rolling systemic risk metrics"""
    
    results_95 = []
    results_99 = []
    
    # Progress bar
    progress_bar = st.progress(0)
    dates = combined_data.index[window_size:]
    
    for i, date in enumerate(dates):
        # Update progress
        progress = (i + 1) / len(dates)
        progress_bar.progress(progress)
        
        # Get rolling window
        window = combined_data.loc[:date].tail(window_size)
        
        for bank in bank_names:
            if bank not in combined_data.columns:
                continue
                
            # Get corresponding index
            if bank not in INDEX_MAP:
                continue
                
            idx_ticker = INDEX_MAP[bank]
            if idx_ticker not in IDX_NAME_MAP:
                continue
                
            idx_name = IDX_NAME_MAP[idx_ticker]
            if idx_name not in combined_data.columns:
                continue
            
            x_bank = window[bank].values
            x_index = window[idx_name].values
            
            # 95% metrics
            results_95.append({
                'Date': date,
                'Bank': bank,
                'VaR_95': rolling_var(x_bank, alpha=0.95),
                'Hill_95': hill_estimator(x_bank, threshold_quantile=0.95),
                'Tau_95': tail_dependence(x_bank, x_index, u=0.95),
                'Beta_T': systemic_beta(x_bank, x_index, u=0.95)
            })
            
            # 99% metrics
            results_99.append({
                'Date': date,
                'Bank': bank,
                'VaR_99': rolling_var(x_bank, alpha=0.99),
                'Hill_99': hill_estimator(x_bank, threshold_quantile=0.99),
                'Tau_99': tail_dependence(x_bank, x_index, u=0.99),
                'Beta_T': systemic_beta(x_bank, x_index, u=0.99)
            })
    
    progress_bar.empty()
    
    # Convert to DataFrames
    df_95 = pd.DataFrame(results_95).set_index(['Date', 'Bank'])
    df_99 = pd.DataFrame(results_99).set_index(['Date', 'Bank'])
    
    return df_95, df_99

def stress_test_spillover(metrics_df, shock_magnitude=0.2, affected_banks=None):
    """Perform spillover-aware stress testing"""
    
    if affected_banks is None:
        # Default: stress US banks
        affected_banks = [bank for bank, region in REGION_MAP.items() 
                         if region == 'Americas']
    
    # Get latest metrics
    latest_date = metrics_df.index.get_level_values(0).max()
    latest_metrics = metrics_df.loc[latest_date]
    
    results = []
    
    for bank in latest_metrics.index:
        if bank in affected_banks:
            # Direct impact
            shocked_var = latest_metrics.loc[bank, 'VaR_95'] * (1 + shock_magnitude)
            impact_type = 'Direct'
        else:
            # Spillover impact based on systemic beta
            beta_t = latest_metrics.loc[bank, 'Beta_T']
            if pd.isna(beta_t):
                beta_t = 1.0
            
            spillover_factor = min(beta_t * 0.3, 0.5)  # Cap spillover at 50%
            shocked_var = latest_metrics.loc[bank, 'VaR_95'] * (1 + shock_magnitude * spillover_factor)
            impact_type = 'Spillover'
        
        results.append({
            'Bank': bank,
            'Region': REGION_MAP.get(bank, 'Unknown'),
            'Original_VaR': latest_metrics.loc[bank, 'VaR_95'],
            'Shocked_VaR': shocked_var,
            'Impact_Magnitude': (shocked_var / latest_metrics.loc[bank, 'VaR_95'] - 1) * 100,
            'Impact_Type': impact_type,
            'Systemic_Beta': latest_metrics.loc[bank, 'Beta_T']
        })
    
    return pd.DataFrame(results)

# Crisis periods for ML labeling
CRISIS_PERIODS = {
    'eurozone_crisis': (pd.Timestamp('2011-07-01'), pd.Timestamp('2012-12-31')),
    'china_correction': (pd.Timestamp('2015-06-01'), pd.Timestamp('2016-02-29')),
    'covid_crash': (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-05-31')),
    'ukraine_war': (pd.Timestamp('2022-02-01'), pd.Timestamp('2022-06-30')),
    'banking_stress_2023': (pd.Timestamp('2023-03-01'), pd.Timestamp('2023-05-31'))
}
