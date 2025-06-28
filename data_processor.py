# Data manipulation and numerical tools
import pandas as pd
import numpy as np

# Yahoo Finance data fetching
import yfinance as yf

# For statistical and EVT functions
import scipy.stats as stats
from scipy.stats import genpareto  # Generalized Pareto for EVT
from scipy.stats import norm

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
    using accurate Extreme Value Theory methodology.
    """
    
    def __init__(self):
        # Bank dictionary with ticker to name mapping (updated with reliable symbols)
        self.bank_dict = {
            # Americas - Major US Banks (using more reliable symbols)
            'JPM': 'JPMorgan Chase',
            'C': 'Citigroup', 
            'BAC': 'Bank of America',
            'WFC': 'Wells Fargo',
            'GS': 'Goldman Sachs',
            'MS': 'Morgan Stanley',
            'BK': 'Bank of New York Mellon',
            'STT': 'State Street',
            'RY.TO': 'Royal Bank of Canada',
            'TD.TO': 'Toronto Dominion',
            
            # Europe - Major European Banks
            'HSBC': 'HSBC Holdings',
            'BCS': 'Barclays',
            'BNPQY': 'BNP Paribas',
            'CRARY': 'Groupe Crédit Agricole',
            'ING': 'ING Group',
            'DB': 'Deutsche Bank',
            'SAN': 'Santander',
            'SCGLY': 'Société Générale',
            'UBS': 'UBS Group',
            'SCBFF': 'Standard Chartered',
            
            # Asia/Pacific - Major Asian Banks
            'ACGBY': 'Agricultural Bank of China',
            'BACHY': 'Bank of China',
            'CICHY': 'China Construction Bank',
            'IDCBY': 'ICBC',
            'BKFCF': 'Bank of Communications',
            'MUFG': 'Mitsubishi UFJ FG',
            'MFG': 'Mizuho FG',
            'SMFG': 'Sumitomo Mitsui FG'
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
            'HSBC Holdings':         '^FTSE',
            'Barclays':              '^FTSE',
            'Standard Chartered':    '^FTSE',

            # Germany → DAX
            'Deutsche Bank':         '^GDAXI',

            # Switzerland → SMI
            'UBS Group':             '^SSMI',

            # Netherlands → AEX
            'ING Group':             '^AEX',

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
            '^AEX':      'AEX',
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
            'HSBC Holdings': 'Europe',
            'Barclays': 'Europe',
            'BNP Paribas': 'Europe',
            'Groupe Crédit Agricole': 'Europe',
            'ING Group': 'Europe',
            'Deutsche Bank': 'Europe',
            'Santander': 'Europe',
            'Société Générale': 'Europe',
            'UBS Group': 'Europe',
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
        
        # Handle single ticker case
        if len(selected_tickers) == 1:
            raw = raw.to_frame(selected_tickers[0])
        
        # Check if we have any data at all
        if raw.empty:
            raise ValueError("No data downloaded for any selected banks")
        
        # Ensure index is DatetimeIndex
        if not isinstance(raw.index, pd.DatetimeIndex):
            try:
                raw.index = pd.to_datetime(raw.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to DatetimeIndex: {e}")
        
        # Find & drop any tickers with *no* data at all
        no_data = [t for t in raw.columns if raw[t].dropna().empty]
        if no_data:
            print("Dropping (no data):", no_data)
        
        # Define price_data as the cleaned raw data
        price_data = raw.drop(columns=no_data) if no_data else raw
        
        # Check if we still have data after cleaning
        if price_data.empty:
            raise ValueError("No valid data remaining after cleaning")
        
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
        
        # Handle single index case
        if len(index_tickers) == 1:
            raw_idx = raw_idx.to_frame(index_tickers[0])
        
        # Check if we have any data at all
        if raw_idx.empty:
            raise ValueError("No index data downloaded")
        
        # Ensure index is DatetimeIndex
        if not isinstance(raw_idx.index, pd.DatetimeIndex):
            try:
                raw_idx.index = pd.to_datetime(raw_idx.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to DatetimeIndex: {e}")
        
        # Drop index tickers with no data at all
        no_data_idx = [t for t in raw_idx.columns if raw_idx[t].dropna().empty]
        if no_data_idx:
            print("Dropping indices with no data:", no_data_idx)
        raw_idx = raw_idx.drop(columns=no_data_idx) if no_data_idx else raw_idx
        
        # Check if we still have data after cleaning
        if raw_idx.empty:
            raise ValueError("No valid index data remaining after cleaning")
        
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
    
    # Accurate EVT metric functions
    def calculate_var(self, returns, alpha=0.95):
        """Calculate Value at Risk using empirical quantile"""
        if len(returns) == 0:
            return np.nan
        # VaR is the negative of the quantile (since we want losses)
        return -np.percentile(returns, (1-alpha)*100)
    
    def hill_estimator(self, returns, threshold_quantile=0.95, min_excesses=10):
        """
        Accurate Hill estimator for tail index estimation
        
        Parameters:
        - returns: array of returns
        - threshold_quantile: quantile for threshold selection
        - min_excesses: minimum number of exceedances required
        
        Returns:
        - hill_estimate: tail index estimate
        """
        if len(returns) < min_excesses:
            return np.nan
            
        # Convert returns to losses (negative returns)
        losses = -returns
        
        # Remove any non-positive losses
        losses = losses[losses > 0]
        
        if len(losses) < min_excesses:
            return np.nan
        
        # Sort losses in descending order
        sorted_losses = np.sort(losses)[::-1]
        
        # Find threshold based on quantile
        threshold = np.quantile(losses, threshold_quantile)
        
        # Get exceedances above threshold
        exceedances = sorted_losses[sorted_losses > threshold]
        
        if len(exceedances) < min_excesses:
            # Try progressively lower thresholds
            for q in np.arange(threshold_quantile - 0.01, 0.85, -0.01):
                threshold = np.quantile(losses, q)
                exceedances = sorted_losses[sorted_losses > threshold]
                if len(exceedances) >= min_excesses:
                    break
            
        if len(exceedances) < min_excesses:
            return np.nan
        
        # Calculate Hill estimator
        k = len(exceedances)
        log_ratios = np.log(exceedances / threshold)
        hill_estimate = np.mean(log_ratios)
        
        return max(hill_estimate, 0.01)  # Ensure positive value
    
    def tail_dependence_coefficient(self, x, y, threshold_quantile=0.95):
        """
        Calculate tail dependence coefficient for left tail (losses)
        
        Parameters:
        - x, y: arrays of returns
        - threshold_quantile: quantile for threshold selection
        
        Returns:
        - tau: tail dependence coefficient
        """
        if len(x) != len(y) or len(x) == 0:
            return np.nan
        
        # Calculate thresholds for both series (using losses)
        x_threshold = np.quantile(x, 1 - threshold_quantile)  # Left tail threshold
        y_threshold = np.quantile(y, 1 - threshold_quantile)  # Left tail threshold
        
        # Count joint exceedances in left tail (losses)
        x_exceeds = x <= x_threshold
        y_exceeds = y <= y_threshold
        joint_exceeds = x_exceeds & y_exceeds
        
        # Calculate conditional probability
        if np.sum(x_exceeds) == 0:
            return np.nan
        
        tau = np.sum(joint_exceeds) / np.sum(x_exceeds)
        
        return min(max(tau, 0.0), 1.0)  # Ensure tau is in [0,1]
    
    def systemic_beta(self, bank_returns, market_returns, alpha=0.95):
        """
        Calculate systemic beta using accurate EVT methodology
        
        Parameters:
        - bank_returns: array of bank returns
        - market_returns: array of market returns
        - alpha: confidence level
        
        Returns:
        - beta_t: systemic beta coefficient
        """
        if len(bank_returns) != len(market_returns) or len(bank_returns) == 0:
            return np.nan
        
        # Calculate components
        var_bank = self.calculate_var(bank_returns, alpha)
        var_market = self.calculate_var(market_returns, alpha)
        
        # Use threshold quantile corresponding to alpha
        threshold_quantile = alpha
        
        hill_market = self.hill_estimator(market_returns, threshold_quantile)
        tau = self.tail_dependence_coefficient(bank_returns, market_returns, threshold_quantile)
        
        # Check for valid values
        if (np.isnan(var_bank) or np.isnan(var_market) or 
            np.isnan(hill_market) or np.isnan(tau) or 
            var_market == 0 or hill_market == 0):
            return np.nan
        
        # Calculate systemic beta
        try:
            beta_t = (tau ** (1.0 / hill_market)) * (var_bank / var_market)
            return max(beta_t, 0.0)  # Ensure non-negative
        except (ZeroDivisionError, OverflowError):
            return np.nan
    
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
        
        # Get bank columns
        bank_columns = [col for col in self.combined_data.columns if col in self.weekly_returns.columns]
        
        # Calculate metrics for each date and bank
        for i in range(window_size, len(self.combined_data)):
            end_date = self.combined_data.index[i]
            start_idx = i - window_size
            window_data = self.combined_data.iloc[start_idx:i+1]
            
            for bank in bank_columns:
                # Get corresponding market index
                if bank not in self.index_map:
                    continue
                    
                market_idx_ticker = self.index_map[bank]
                if market_idx_ticker not in self.idx_name_map:
                    continue
                    
                market_name = self.idx_name_map[market_idx_ticker]
                
                if market_name not in window_data.columns:
                    continue
                
                bank_returns = window_data[bank].dropna().values
                market_returns = window_data[market_name].dropna().values
                
                # Ensure we have enough data
                if len(bank_returns) < 30 or len(market_returns) < 30:
                    continue
                
                # Calculate metrics for 95% confidence
                var_95 = self.calculate_var(bank_returns, alpha=0.95)
                hill_95 = self.hill_estimator(bank_returns, threshold_quantile=0.95)
                tau_95 = self.tail_dependence_coefficient(bank_returns, market_returns, threshold_quantile=0.95)
                beta_95 = self.systemic_beta(bank_returns, market_returns, alpha=0.95)
                
                results_95.append({
                    'Date': end_date,
                    'Bank': bank,
                    'Region': self.region_map.get(bank, 'Unknown'),
                    'VaR_95': var_95,
                    'Hill_95': hill_95,
                    'Tau_95': tau_95,
                    'Beta_T': beta_95
                })
                
                # Calculate metrics for 99% confidence
                var_99 = self.calculate_var(bank_returns, alpha=0.99)
                hill_99 = self.hill_estimator(bank_returns, threshold_quantile=0.99)
                tau_99 = self.tail_dependence_coefficient(bank_returns, market_returns, threshold_quantile=0.99)
                beta_99 = self.systemic_beta(bank_returns, market_returns, alpha=0.99)
                
                results_99.append({
                    'Date': end_date,
                    'Bank': bank,
                    'Region': self.region_map.get(bank, 'Unknown'),
                    'VaR_99': var_99,
                    'Hill_99': hill_99,
                    'Tau_99': tau_99,
                    'Beta_T': beta_99
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
            if self.results_95 is None or self.results_95.empty:
                raise ValueError("Must calculate rolling metrics first")
            latest_date = self.results_95['Date'].max()
            return self.results_95[self.results_95['Date'] == latest_date].reset_index(drop=True)
        elif confidence_level == 0.99:
            if self.results_99 is None or self.results_99.empty:
                raise ValueError("Must calculate rolling metrics first")
            latest_date = self.results_99['Date'].max()
            return self.results_99[self.results_99['Date'] == latest_date].reset_index(drop=True)
        else:
            raise ValueError("Confidence level must be 0.95 or 0.99")
    
    def get_bank_time_series(self, bank_name, metric='Beta_T', confidence_level=0.95):
        """
        Get time series for a specific bank and metric
        
        Parameters:
        - bank_name: name of the bank
        - metric: metric to retrieve
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
        var_col = 'VaR_95' if confidence_level == 0.95 else 'VaR_99'
        hill_col = 'Hill_95' if confidence_level == 0.95 else 'Hill_99'
        tau_col = 'Tau_95' if confidence_level == 0.95 else 'Tau_99'
        
        summary = latest_data.groupby('Region').agg({
            'Beta_T': ['mean', 'std', 'min', 'max', 'count'],
            var_col: ['mean', 'std', 'min', 'max'],
            hill_col: ['mean', 'std', 'min', 'max'],
            tau_col: ['mean', 'std', 'min', 'max']
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
