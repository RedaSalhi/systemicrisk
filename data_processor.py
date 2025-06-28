# Fixed Data Processor with Complete EVT Implementation
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as stats
from scipy.stats import genpareto
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

class BankingDataProcessor:
    """
    Complete and accurate banking data processor for systemic risk analysis
    """
    
    def __init__(self):
        # Updated bank dictionary with working tickers
        self.bank_dict = {
            # Americas - Major US Banks
            'JPM': 'JPMorgan Chase',
            'C': 'Citigroup', 
            'BAC': 'Bank of America',
            'WFC': 'Wells Fargo',
            'GS': 'Goldman Sachs',
            'MS': 'Morgan Stanley',
            'BK': 'Bank of New York Mellon',
            'STT': 'State Street',
            'RY': 'Royal Bank of Canada',  # Fixed ticker
            'TD': 'Toronto Dominion',       # Fixed ticker
            
            # Europe - Major European Banks  
            'HSBC': 'HSBC Holdings',
            'BARC.L': 'Barclays',           # Fixed ticker
            'BNP.PA': 'BNP Paribas',        # Fixed ticker
            'ACA.PA': 'Credit Agricole',    # Fixed ticker
            'ING': 'ING Group',
            'DB': 'Deutsche Bank',
            'SAN': 'Santander',
            'GLE.PA': 'Societe Generale',   # Fixed ticker
            'UBS': 'UBS Group',
            'STAN.L': 'Standard Chartered', # Fixed ticker
            
            # Asia/Pacific - Major Asian Banks
            '1288.HK': 'ABC China',         # Fixed ticker
            '3988.HK': 'Bank of China',     # Fixed ticker
            '0939.HK': 'CCB China',         # Fixed ticker
            '1398.HK': 'ICBC',              # Fixed ticker
            '8411.T': 'Mizuho FG',          # Fixed ticker
            '8306.T': 'MUFG',               # Fixed ticker
            '8316.T': 'Sumitomo Mitsui FG'  # Fixed ticker
        }
        
        # Index mapping for each bank
        self.index_map = {
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
            'Credit Agricole': '^FCHI',
            'Societe Generale': '^FCHI',

            # Spain → IBEX 35
            'Santander': '^IBEX',

            # UK → FTSE 100
            'HSBC Holdings': '^FTSE',
            'Barclays': '^FTSE',
            'Standard Chartered': '^FTSE',

            # Germany → DAX
            'Deutsche Bank': '^GDAXI',

            # Switzerland → SMI
            'UBS Group': '^SSMI',

            # Netherlands → AEX
            'ING Group': '^AEX',

            # Hong Kong → Hang Seng
            'ABC China': '^HSI',
            'Bank of China': '^HSI',
            'CCB China': '^HSI',
            'ICBC': '^HSI',

            # Japan → Nikkei 225
            'Mizuho FG': '^N225',
            'MUFG': '^N225',
            'Sumitomo Mitsui FG': '^N225'
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
            'Credit Agricole': 'Europe',
            'ING Group': 'Europe',
            'Deutsche Bank': 'Europe',
            'Santander': 'Europe',
            'Societe Generale': 'Europe',
            'UBS Group': 'Europe',
            'Standard Chartered': 'Europe',
            # Asia/Pacific
            'ABC China': 'Asia/Pacific',
            'Bank of China': 'Asia/Pacific',
            'CCB China': 'Asia/Pacific',
            'ICBC': 'Asia/Pacific',
            'Mizuho FG': 'Asia/Pacific',
            'MUFG': 'Asia/Pacific',
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
    
    def download_bank_data(self, selected_banks, start_date='2015-01-01', end_date='2024-12-31'):
        """Download banking data for selected banks with robust error handling"""
        # Get tickers for selected banks
        ticker_to_name = {v: k for k, v in self.bank_dict.items()}
        selected_tickers = []
        
        for bank in selected_banks:
            if bank in ticker_to_name:
                selected_tickers.append(ticker_to_name[bank])
            else:
                print(f"Warning: Bank '{bank}' not found in ticker mapping")
        
        if not selected_tickers:
            raise ValueError("No valid banks selected")
        
        print(f"Downloading data for tickers: {selected_tickers}")
        
        # Download data with error handling
        successful_data = {}
        
        for ticker in selected_tickers:
            try:
                print(f"Downloading {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty and 'Close' in data.columns:
                    # Handle multi-level columns if present
                    if isinstance(data.columns, pd.MultiIndex):
                        close_data = data['Close'].iloc[:, 0] if data['Close'].shape[1] > 0 else data['Close']
                    else:
                        close_data = data['Close']
                    
                    # Ensure we have enough data
                    if len(close_data.dropna()) > 100:
                        successful_data[ticker] = close_data
                        print(f"✅ Successfully downloaded {ticker}: {len(close_data)} data points")
                    else:
                        print(f"❌ Insufficient data for {ticker}: {len(close_data.dropna())} points")
                else:
                    print(f"❌ No valid data for {ticker}")
                    
            except Exception as e:
                print(f"❌ Error downloading {ticker}: {e}")
                continue
        
        if not successful_data:
            raise ValueError("No data successfully downloaded for any banks")
        
        # Create DataFrame from successful downloads
        price_data = pd.DataFrame(successful_data)
        
        # Map tickers to bank names
        name_mapping = {ticker: self.bank_dict[ticker] for ticker in successful_data.keys()}
        price_data.rename(columns=name_mapping, inplace=True)
        
        # Resample to weekly and compute returns
        weekly_prices = price_data.resample('W-FRI').last().ffill().dropna()
        weekly_returns = np.log(weekly_prices / weekly_prices.shift(1)).dropna()
        
        print(f"✅ Final data shape: {weekly_returns.shape}")
        print(f"Date range: {weekly_returns.index.min().date()} to {weekly_returns.index.max().date()}")
        
        self.weekly_returns = weekly_returns
        return weekly_returns
    
    def download_index_data(self, start_date='2015-01-01', end_date='2024-12-31'):
        """Download index data with robust error handling"""
        # Get unique index tickers
        index_tickers = list(set(self.index_map.values()))
        
        successful_indices = {}
        
        for ticker in index_tickers:
            try:
                print(f"Downloading index {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty and 'Close' in data.columns:
                    # Handle multi-level columns
                    if isinstance(data.columns, pd.MultiIndex):
                        close_data = data['Close'].iloc[:, 0] if data['Close'].shape[1] > 0 else data['Close']
                    else:
                        close_data = data['Close']
                    
                    if len(close_data.dropna()) > 100:
                        successful_indices[ticker] = close_data
                        print(f"✅ Successfully downloaded index {ticker}")
                    else:
                        print(f"❌ Insufficient data for index {ticker}")
                else:
                    print(f"❌ No valid data for index {ticker}")
                    
            except Exception as e:
                print(f"❌ Error downloading index {ticker}: {e}")
                continue
        
        if not successful_indices:
            raise ValueError("No index data successfully downloaded")
        
        # Create DataFrame and resample
        idx_data = pd.DataFrame(successful_indices)
        weekly_idx_prices = idx_data.resample('W-FRI').last().ffill().dropna()
        weekly_idx_returns = np.log(weekly_idx_prices / weekly_idx_prices.shift(1)).dropna()
        
        print(f"✅ Index data shape: {weekly_idx_returns.shape}")
        
        self.weekly_idx_returns = weekly_idx_returns
        return weekly_idx_returns
    
    def combine_data(self):
        """Combine bank and index data"""
        if self.weekly_returns is None or self.weekly_idx_returns is None:
            raise ValueError("Must download bank and index data first")
        
        combined = self.weekly_returns.join(self.weekly_idx_returns, how='inner')
        
        print(f"✅ Combined data shape: {combined.shape}")
        print(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}")
        
        self.combined_data = combined
        return combined
    
    # ACCURATE EVT FUNCTIONS
    def calculate_var(self, returns, alpha=0.95):
        """Calculate Value at Risk using empirical quantile"""
        # Convert to pandas Series if it's a numpy array
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if len(returns) == 0:
            return np.nan

        # Handle NaNs for both pandas and numpy objects
        try:
            if hasattr(returns, 'isna') and callable(getattr(returns, 'isna', None)):
                if returns.isna().all():
                    return np.nan
            elif np.isnan(returns).all():
                return np.nan
        except (AttributeError, TypeError):
            # Fallback to numpy check if pandas method fails
            if np.isnan(returns).all():
                return np.nan
        
        # Clean the data
        try:
            if hasattr(returns, 'dropna') and callable(getattr(returns, 'dropna', None)):
                clean_returns = returns.dropna()
            else:
                clean_returns = returns[~np.isnan(returns)] if len(returns) > 0 else returns
        except (AttributeError, TypeError):
            # Fallback to numpy method
            clean_returns = returns[~np.isnan(returns)] if len(returns) > 0 else returns
        
        if len(clean_returns) == 0:
            return np.nan
            
        # VaR is the negative of the lower quantile
        return -np.percentile(clean_returns, (1-alpha)*100)
    
    def hill_estimator(self, returns, threshold_quantile=0.95, min_excesses=10):
        """Accurate Hill estimator for tail index"""
        # Convert to pandas Series if it's a numpy array
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if len(returns) < min_excesses:
            return np.nan
        
        # Clean the data
        try:
            if hasattr(returns, 'dropna') and callable(getattr(returns, 'dropna', None)):
                clean_returns = returns.dropna()
            else:
                clean_returns = returns[~np.isnan(returns)] if len(returns) > 0 else returns
        except (AttributeError, TypeError):
            # Fallback to numpy method
            clean_returns = returns[~np.isnan(returns)] if len(returns) > 0 else returns
            
        if len(clean_returns) < min_excesses:
            return np.nan
        
        # Convert to losses (negative returns for left tail)
        losses = -clean_returns
        losses = losses[losses > 0]  # Keep positive losses only
        
        if len(losses) < min_excesses:
            return np.nan
        
        # Calculate threshold
        threshold = np.quantile(losses, threshold_quantile)
        
        # Get exceedances above threshold
        exceedances = losses[losses > threshold]
        
        if len(exceedances) < min_excesses:
            # Try lower thresholds
            for q in np.arange(threshold_quantile - 0.01, 0.85, -0.01):
                threshold = np.quantile(losses, q)
                exceedances = losses[losses > threshold]
                if len(exceedances) >= min_excesses:
                    break
        
        if len(exceedances) < min_excesses:
            return np.nan
        
        # Hill estimator calculation
        log_ratios = np.log(exceedances / threshold)
        hill_estimate = np.mean(log_ratios)
        
        return max(hill_estimate, 0.01)  # Ensure positive
    
    def tail_dependence_coefficient(self, x, y, threshold_quantile=0.95):
        """Calculate tail dependence coefficient for left tail"""
        if len(x) != len(y) or len(x) == 0:
            return np.nan
        
        # Remove missing values
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(df) == 0:
            return np.nan
        
        x_clean, y_clean = df['x'].values, df['y'].values
        
        # Calculate thresholds for left tail (extreme losses)
        x_threshold = np.quantile(x_clean, 1 - threshold_quantile)
        y_threshold = np.quantile(y_clean, 1 - threshold_quantile)
        
        # Count exceedances in left tail
        x_exceeds = x_clean <= x_threshold
        y_exceeds = y_clean <= y_threshold
        joint_exceeds = x_exceeds & y_exceeds
        
        # Calculate conditional probability
        if np.sum(x_exceeds) == 0:
            return np.nan
        
        tau = np.sum(joint_exceeds) / np.sum(x_exceeds)
        
        return np.clip(tau, 0.0, 1.0)  # Ensure [0,1] range
    
    def systemic_beta(self, bank_returns, market_returns, alpha=0.95):
        """Calculate systemic beta using accurate EVT methodology"""
        # Convert to pandas Series if numpy arrays
        if isinstance(bank_returns, np.ndarray):
            bank_returns = pd.Series(bank_returns)
        if isinstance(market_returns, np.ndarray):
            market_returns = pd.Series(market_returns)
            
        if len(bank_returns) != len(market_returns) or len(bank_returns) == 0:
            return np.nan
        
        # Remove missing values
        df = pd.DataFrame({'bank': bank_returns, 'market': market_returns}).dropna()
        if len(df) < 30:  # Need sufficient data
            return np.nan
        
        bank_clean = df['bank'].values
        market_clean = df['market'].values
        
        # Calculate components
        var_bank = self.calculate_var(bank_clean, alpha)
        var_market = self.calculate_var(market_clean, alpha)
        
        threshold_quantile = alpha
        hill_market = self.hill_estimator(market_clean, threshold_quantile)
        tau = self.tail_dependence_coefficient(bank_clean, market_clean, threshold_quantile)
        
        # Check for valid values
        if any(np.isnan([var_bank, var_market, hill_market, tau])) or var_market == 0 or hill_market == 0:
            return np.nan
        
        # Calculate systemic beta
        try:
            beta_t = (tau ** (1.0 / hill_market)) * (var_bank / var_market)
            return max(beta_t, 0.0)  # Ensure non-negative
        except (ZeroDivisionError, OverflowError, ValueError):
            return np.nan
    
    def calculate_rolling_metrics(self, window_size=52):
        """Calculate rolling window metrics for all banks"""
        if self.combined_data is None:
            raise ValueError("Must combine data first")
        
        results_95 = []
        results_99 = []
        
        # Get bank columns
        bank_columns = [col for col in self.combined_data.columns if col in self.weekly_returns.columns]
        
        print(f"Calculating metrics for {len(bank_columns)} banks with {window_size}-week windows...")
        
        total_iterations = len(self.combined_data) - window_size
        
        for i in range(window_size, len(self.combined_data)):
            if i % 10 == 0:  # Progress indicator
                progress = (i - window_size) / total_iterations * 100
                print(f"Progress: {progress:.1f}%")
            
            end_date = self.combined_data.index[i]
            window_data = self.combined_data.iloc[i-window_size:i+1]
            
            for bank in bank_columns:
                # Get corresponding market index
                if bank not in self.index_map:
                    continue
                    
                market_idx_ticker = self.index_map[bank]
                
                # Find the market index column
                market_col = None
                for col in window_data.columns:
                    if market_idx_ticker in str(col) or col == market_idx_ticker:
                        market_col = col
                        break
                
                if market_col is None:
                    continue
                
                bank_returns = window_data[bank].dropna()
                market_returns = window_data[market_col].dropna()
                
                # Ensure we have enough data and convert to numpy arrays
                if len(bank_returns) < 30 or len(market_returns) < 30:
                    continue
                
                # Convert to numpy arrays for consistent processing
                bank_returns_array = bank_returns.values
                market_returns_array = market_returns.values
                
                # Calculate metrics for 95% confidence
                var_95 = self.calculate_var(bank_returns_array, alpha=0.95)
                hill_95 = self.hill_estimator(bank_returns_array, threshold_quantile=0.95)
                tau_95 = self.tail_dependence_coefficient(bank_returns_array, market_returns_array, threshold_quantile=0.95)
                beta_95 = self.systemic_beta(bank_returns_array, market_returns_array, alpha=0.95)
                
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
                var_99 = self.calculate_var(bank_returns_array, alpha=0.99)
                hill_99 = self.hill_estimator(bank_returns_array, threshold_quantile=0.99)
                tau_99 = self.tail_dependence_coefficient(bank_returns_array, market_returns_array, threshold_quantile=0.99)
                beta_99 = self.systemic_beta(bank_returns_array, market_returns_array, alpha=0.99)
                
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
        
        print(f"✅ Calculated metrics: {len(self.results_95)} records for 95%, {len(self.results_99)} records for 99%")
        
        return self.results_95, self.results_99
    
    # COMPLETE ACCESSOR METHODS
    def get_all_metrics(self, confidence_level=0.95):
        """Get all metrics for the specified confidence level"""
        if confidence_level == 0.95:
            if self.results_95 is None or self.results_95.empty:
                raise ValueError("No 95% metrics available. Run calculate_rolling_metrics first.")
            return self.results_95
        elif confidence_level == 0.99:
            if self.results_99 is None or self.results_99.empty:
                raise ValueError("No 99% metrics available. Run calculate_rolling_metrics first.")
            return self.results_99
        else:
            raise ValueError("Confidence level must be 0.95 or 0.99")
    
    def get_latest_metrics(self, confidence_level=0.95):
        """Get the latest metrics for all banks"""
        data = self.get_all_metrics(confidence_level)
        latest_date = data['Date'].max()
        return data[data['Date'] == latest_date].reset_index(drop=True)
    
    def get_bank_time_series(self, bank_name, metric='Beta_T', confidence_level=0.95):
        """Get time series for a specific bank and metric"""
        data = self.get_all_metrics(confidence_level)
        bank_data = data[data['Bank'] == bank_name]
        
        if bank_data.empty:
            raise ValueError(f"No data found for bank: {bank_name}")
        
        if metric not in bank_data.columns:
            raise ValueError(f"Metric {metric} not available")
        
        return bank_data.set_index('Date')[metric].sort_index()
    
    def get_summary_statistics(self, confidence_level=0.95):
        """Get summary statistics across all banks"""
        latest_data = self.get_latest_metrics(confidence_level)
        
        if latest_data.empty:
            return pd.DataFrame()
        
        # Get the correct column names
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

# Convenience function
def process_banking_data(selected_banks, start_date='2015-01-01', end_date='2024-12-31'):
    """Process banking data for selected banks with complete error handling"""
    print(f"🚀 Processing data for {len(selected_banks)} banks...")
    print(f"📅 Date range: {start_date} to {end_date}")
    
    processor = BankingDataProcessor()
    
    try:
        # Download data
        print("📥 Downloading bank data...")
        processor.download_bank_data(selected_banks, start_date, end_date)
        
        print("📊 Downloading index data...")
        processor.download_index_data(start_date, end_date)
        
        # Combine data
        print("🔗 Combining datasets...")
        processor.combine_data()
        
        # Calculate metrics
        print("📈 Calculating rolling metrics...")
        processor.calculate_rolling_metrics()
        
        print("✅ Processing completed successfully!")
        return processor
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise
