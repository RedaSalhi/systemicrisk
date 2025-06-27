# Data manipulation and numerical tools
import pandas as pd
import numpy as np

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Yahoo Finance data fetching
import yfinance as yf

# For statistical and EVT functions
import scipy.stats as stats
from scipy.stats import genpareto  # Generalized Pareto for EVT
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

# For estimating tail dependence
from scipy.stats import rankdata

# Date handling
import datetime as dt

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

class BankingDataProcessor:
    """
    A comprehensive data processor for banking systemic risk analysis
    using the accurate methodology provided.
    """
    
    def __init__(self):
        # Bank dictionary with ticker to name mapping
        self.bank_dict = {
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
        
        # Index mapping for each bank
        self.index_map = {
            # US banks → S&P 500
            'JPMorgan Chase':        '^GSPC',
            'Citigroup':             '^GSPC',
            'Bank of America':       '^GSPC',
            'Wells Fargo':           '^GSPC',
            'Goldman Sachs':         '^GSPC',
            'Morgan Stanley':        '^GSPC',
            'Bank of New York Mellon':'^GSPC',
            'State Street':          '^GSPC',

            # Canada → TSX Composite
            'Royal Bank of Canada':  '^GSPTSE',
            'Toronto Dominion':      '^GSPTSE',

            # France → CAC 40
            'BNP Paribas':           '^FCHI',
            'Groupe Crédit Agricole':'^FCHI',
            'Société Générale':      '^FCHI',

            # Spain → IBEX 35
            'Santander':             '^IBEX',

            # UK → FTSE 100
            'HSBC':                  '^FTSE',
            'Barclays':              '^FTSE',
            'Standard Chartered':    '^FTSE',

            # Germany → DAX
            'Deutsche Bank':         '^GDAXI',

            # Switzerland → SMI
            'UBS':                   '^SSMI',

            # Netherlands → AEX  (for ING)
            'ING':                   '^AEX',

            # China → Shanghai Composite
            'China Construction Bank':'000001.SS',
            'Agricultural Bank of China':'000001.SS',
            'ICBC':                  '000001.SS',
            'Bank of Communications':'000001.SS',
            'Bank of China':         '000001.SS',

            # Japan → Nikkei 225
            'Mitsubishi UFJ FG':     '^N225',
            'Sumitomo Mitsui FG':    '^N225',
            'Mizuho FG':             '^N225',
        }

        # Map index tickers → nice names
        self.idx_name_map = {
            '^GSPC':     'S&P 500',
            '^GSPTSE':   'TSX Composite',
            '^FCHI':     'CAC 40',
            '^IBEX':     'IBEX 35',
            '^FTSE':     'FTSE 100',
            '^GDAXI':    'DAX',
            '^SSMI':     'SMI',
            '^N225':     'Nikkei 225',
            '000001.SS': 'Shanghai Composite',
            '^AEX':       'AEX',
        }

        # Region mapping
        self.region_map = {
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
        
        # Initialize data storage
        self.weekly_returns = None
        self.weekly_idx_returns = None
        self.combined_data = None
        self.results_95 = None
        self.results_99 = None
        
    def get_available_banks(self):
        """Get list of available banks for user selection"""
        return list(self.bank_dict.values())
    
    def get_banks_by_region(self):
        """Get banks grouped by region"""
        regions = {}
        for bank, region in self.region_map.items():
            if region not in regions:
                regions[region] = []
            regions[region].append(bank)
        return regions
    
    def download_bank_data(self, selected_banks, start_date='2010-01-01', end_date='2024-12-31'):
        """
        Download banking data for selected banks
        
        Parameters:
        - selected_banks: list of bank names to include
        - start_date: start date for data download
        - end_date: end date for data download
        
        Returns:
        - weekly_returns: DataFrame with weekly returns for selected banks
        """
        # Get tickers for selected banks
        ticker_to_name = {v: k for k, v in self.bank_dict.items()}
        selected_tickers = [ticker_to_name[bank] for bank in selected_banks if bank in ticker_to_name]
        
        if not selected_tickers:
            raise ValueError("No valid banks selected")
        
        # Download daily Close prices
        raw = yf.download(selected_tickers, start=start_date, end=end_date)['Close']
        
        # Find & drop any tickers with *no* data at all
        no_data = [t for t in raw.columns if raw[t].dropna().empty]
        if no_data:
            print("Dropping (no data):", no_data)
        
        # Define price_data as the cleaned raw data
        price_data = raw.drop(columns=no_data)
        
        # Map tickers to bank names
        price_data.rename(columns=self.bank_dict, inplace=True)
        
        # Resample and compute returns
        weekly_prices = (
            price_data
            .resample('W-FRI').last()
            .ffill()
            .dropna()
        )
        
        weekly_returns = np.log(weekly_prices / weekly_prices.shift(1)).dropna()
        
        self.weekly_returns = weekly_returns
        return weekly_returns
    
    def download_index_data(self, start_date='2010-01-01', end_date='2024-12-31'):
        """
        Download index data for all regions
        
        Parameters:
        - start_date: start date for data download
        - end_date: end date for data download
        
        Returns:
        - weekly_idx_returns: DataFrame with weekly index returns
        """
        # Get unique index tickers
        index_tickers = list(set(self.index_map.values()))
        raw_idx = yf.download(index_tickers, start=start_date, end=end_date)['Close']
        
        # Drop index tickers with no data at all
        no_data_idx = [t for t in raw_idx.columns if raw_idx[t].dropna().empty]
        if no_data_idx:
            print("Dropping indices with no data:", no_data_idx)
        raw_idx = raw_idx.drop(columns=no_data_idx)
        
        # Weekly resample & compute log‐returns
        weekly_idx_prices = (
            raw_idx
            .resample('W-FRI').last()
            .ffill()
            .dropna()
        )
        weekly_idx_returns = np.log(weekly_idx_prices / weekly_idx_prices.shift(1)).dropna()
        
        # Rename to human‐readable index names
        weekly_idx_returns.rename(columns=self.idx_name_map, inplace=True)
        
        self.weekly_idx_returns = weekly_idx_returns
        return weekly_idx_returns
    
    def combine_data(self):
        """
        Combine bank and index data via inner-join on Date
        
        Returns:
        - combined: DataFrame with both bank and index returns
        """
        if self.weekly_returns is None or self.weekly_idx_returns is None:
            raise ValueError("Must download bank and index data first")
        
        # Merge with bank returns via an inner‐join on Date
        combined = self.weekly_returns.join(self.weekly_idx_returns, how='inner')
        
        self.combined_data = combined
        return combined
    
    # Metric functions
    def rolling_var(self, x, alpha=0.95):
        """Calculate rolling Value at Risk"""
        return -np.percentile(x, 100*(1-alpha))
    
    def hill_estimator(self, x, threshold_quantile=0.99, min_excesses=5):
        """
        Estimate the tail index via Hill, but only at the
        largest threshold_quantile such that at least min_excesses are in the tail.
        """
        # sort unique quantiles from, say, 90% up to desired level
        candidate_q = np.linspace(0.90, threshold_quantile, 50)
        for q in reversed(candidate_q):
            p = 1 - q
            u = np.quantile(x, p)
            losses = -x[x < u]
            losses = losses[losses > 0]
            if len(losses) >= min_excesses:
                min_loss = losses.min()
                return np.mean(np.log(losses / min_loss))
        # if even 90% gives too few points, fall back or return NaN
        return np.nan
    
    def tail_dependence(self, x, y, u=0.95):
        """Calculate tail dependence coefficient"""
        qx, qy = np.quantile(x, u), np.quantile(y, u)
        mask = x < qx   # left‐tail dependence (for losses)
        return np.sum(y[mask] < qy) / np.sum(mask) if np.sum(mask)>0 else np.nan
    
    def systemic_beta(self, x, y, u=0.95):
        """Calculate systemic beta"""
        VaR_x = self.rolling_var(x, alpha=u)
        VaR_y = self.rolling_var(y, alpha=u)
        xi_y  = self.hill_estimator(y, threshold_quantile=u)
        tau   = self.tail_dependence(x, y, u=u)
        if xi_y is None or xi_y==0 or np.isnan(tau):
            return np.nan
        return (tau ** (1.0/xi_y)) * (VaR_x / VaR_y)
    
    def calculate_rolling_metrics(self, window_size=52):
        """
        Calculate rolling window metrics for all banks
        
        Parameters:
        - window_size: size of rolling window in weeks (default 52 = 1 year)
        
        Returns:
        - results_95: DataFrame with 95% confidence level metrics
        - results_99: DataFrame with 99% confidence level metrics
        """
        if self.combined_data is None:
            raise ValueError("Must combine data first")
        
        results_95 = []
        results_99 = []
        dates = self.combined_data.index[window_size:]   # start at week 53
        
        for date in dates:
            window = self.combined_data.loc[:date].tail(window_size)
            for bank in self.weekly_returns.columns:
                idx = self.index_map[bank]  # e.g. 'JPMorgan Chase' → 'S&P 500'
                x_b = window[bank].values
                x_i = window[self.idx_name_map[idx]].values

                results_95.append({
                    'Date':    date,
                    'Bank':    bank,
                    'Region':  self.region_map.get(bank, 'Unknown'),
                    'VaR_95':  self.rolling_var(x_b, alpha=0.95),
                    'Hill_95': self.hill_estimator(x_b, threshold_quantile=0.95),
                    'Tau_95':  self.tail_dependence(x_b, x_i, u=0.95),
                    'Beta_T':  self.systemic_beta(x_b, x_i, u=0.95)
                })
                results_99.append({
                    'Date':    date,
                    'Bank':    bank,
                    'Region':  self.region_map.get(bank, 'Unknown'),
                    'VaR_99':  self.rolling_var(x_b, alpha=0.99),
                    'Hill_99': self.hill_estimator(x_b, threshold_quantile=0.99),
                    'Tau_99':  self.tail_dependence(x_b, x_i, u=0.99),
                    'Beta_T':  self.systemic_beta(x_b, x_i, u=0.99)
                })
        
        self.results_95 = pd.DataFrame(results_95)
        self.results_99 = pd.DataFrame(results_99)
        
        return self.results_95, self.results_99
    
    def get_latest_metrics(self, confidence_level=0.95):
        """
        Get the latest metrics for all banks
        
        Parameters:
        - confidence_level: 0.95 or 0.99
        
        Returns:
        - latest_metrics: DataFrame with latest metrics
        """
        if confidence_level == 0.95:
            if self.results_95 is None:
                raise ValueError("Must calculate rolling metrics first")
            latest_date = self.results_95['Date'].max()
            return self.results_95[self.results_95['Date'] == latest_date]
        elif confidence_level == 0.99:
            if self.results_99 is None:
                raise ValueError("Must calculate rolling metrics first")
            latest_date = self.results_99['Date'].max()
            return self.results_99[self.results_99['Date'] == latest_date]
        else:
            raise ValueError("Confidence level must be 0.95 or 0.99")
    
    def get_bank_time_series(self, bank_name, metric='Beta_T', confidence_level=0.95):
        """
        Get time series for a specific bank and metric
        
        Parameters:
        - bank_name: name of the bank
        - metric: metric to retrieve ('Beta_T', 'VaR_95', 'VaR_99', 'Hill_95', 'Hill_99', 'Tau_95', 'Tau_99')
        - confidence_level: 0.95 or 0.99
        
        Returns:
        - time_series: Series with dates and metric values
        """
        if confidence_level == 0.95:
            if self.results_95 is None:
                raise ValueError("Must calculate rolling metrics first")
            data = self.results_95[self.results_95['Bank'] == bank_name]
        elif confidence_level == 0.99:
            if self.results_99 is None:
                raise ValueError("Must calculate rolling metrics first")
            data = self.results_99[self.results_99['Bank'] == bank_name]
        else:
            raise ValueError("Confidence level must be 0.95 or 0.99")
        
        if metric not in data.columns:
            raise ValueError(f"Metric {metric} not available")
        
        return data.set_index('Date')[metric]
    
    def get_summary_statistics(self, confidence_level=0.95):
        """
        Get summary statistics across all banks
        
        Parameters:
        - confidence_level: 0.95 or 0.99
        
        Returns:
        - summary_stats: DataFrame with summary statistics
        """
        if confidence_level == 0.95:
            if self.results_95 is None:
                raise ValueError("Must calculate rolling metrics first")
            data = self.results_95
        elif confidence_level == 0.99:
            if self.results_99 is None:
                raise ValueError("Must calculate rolling metrics first")
            data = self.results_99
        else:
            raise ValueError("Confidence level must be 0.95 or 0.99")
        
        # Get latest data
        latest_date = data['Date'].max()
        latest_data = data[data['Date'] == latest_date]
        
        # Calculate summary statistics
        summary = latest_data.groupby('Region').agg({
            'Beta_T': ['mean', 'std', 'min', 'max'],
            'VaR_95' if confidence_level == 0.95 else 'VaR_99': ['mean', 'std', 'min', 'max'],
            'Hill_95' if confidence_level == 0.95 else 'Hill_99': ['mean', 'std', 'min', 'max'],
            'Tau_95' if confidence_level == 0.95 else 'Tau_99': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        return summary

# Example usage function
def process_banking_data(selected_banks, start_date='2010-01-01', end_date='2024-12-31'):
    """
    Convenience function to process banking data for selected banks
    
    Parameters:
    - selected_banks: list of bank names to include
    - start_date: start date for data download
    - end_date: end date for data download
    
    Returns:
    - processor: BankingDataProcessor instance with all data loaded
    """
    processor = BankingDataProcessor()
    
    # Download data
    processor.download_bank_data(selected_banks, start_date, end_date)
    processor.download_index_data(start_date, end_date)
    
    # Combine data
    processor.combine_data()
    
    # Calculate metrics
    processor.calculate_rolling_metrics()
    
    return processor
