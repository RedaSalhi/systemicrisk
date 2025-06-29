"""
Enhanced Data Processor for Systemic Risk Analysis
Implements advanced EVT methodologies with improved accuracy and robustness
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gamma
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "Low"
    MEDIUM = "Medium" 
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class EVTParams:
    """Parameters for EVT calculations"""
    confidence_levels: List[float] = None
    min_exceedances: int = 10
    threshold_range: Tuple[float, float] = (0.85, 0.98)
    hill_threshold_strategy: str = "adaptive"  # "adaptive", "fixed", "optimal"
    bootstrap_samples: int = 100
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]

@dataclass
class SystemicRiskMetrics:
    """Container for systemic risk metrics"""
    date: pd.Timestamp
    bank: str
    var_95: float
    var_99: float
    hill_95: float
    hill_99: float
    tau_95: float
    tau_99: float
    beta_t_95: float
    beta_t_99: float
    conditional_var: float
    marginal_expected_shortfall: float
    confidence_intervals: Dict[str, Tuple[float, float]]

# Enhanced Bank and Index Mappings
ENHANCED_BANK_DICT = {
    # US Banks
    'JPM': 'JPMorgan Chase', 'C': 'Citigroup', 'BAC': 'Bank of America',
    'WFC': 'Wells Fargo', 'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley',
    'BK': 'Bank of New York Mellon', 'STT': 'State Street',
    
    # Canadian Banks
    'RY': 'Royal Bank of Canada', 'TD': 'Toronto Dominion',
    
    # European Banks
    'HSBA.L': 'HSBC', 'BARC.L': 'Barclays', 'BNP.PA': 'BNP Paribas',
    'ACA.PA': 'Groupe Crédit Agricole', 'INGA.AS': 'ING', 'DBK.DE': 'Deutsche Bank',
    'SAN.MC': 'Santander', 'GLE.PA': 'Société Générale', 'UBSG.SW': 'UBS',
    'STAN.L': 'Standard Chartered',
    
    # Asian Banks
    '1288.HK': 'Agricultural Bank of China', '3988.HK': 'Bank of China',
    '0939.HK': 'China Construction Bank', '1398.HK': 'ICBC',
    '3328.HK': 'Bank of Communications', '8306.T': 'Mitsubishi UFJ FG',
    '8411.T': 'Mizuho FG', '8316.T': 'Sumitomo Mitsui FG'
}

ENHANCED_INDEX_MAP = {
    # Americas
    'JPMorgan Chase': '^GSPC', 'Citigroup': '^GSPC', 'Bank of America': '^GSPC',
    'Wells Fargo': '^GSPC', 'Goldman Sachs': '^GSPC', 'Morgan Stanley': '^GSPC',
    'Bank of New York Mellon': '^GSPC', 'State Street': '^GSPC',
    'Royal Bank of Canada': '^GSPTSE', 'Toronto Dominion': '^GSPTSE',
    
    # Europe
    'BNP Paribas': '^FCHI', 'Groupe Crédit Agricole': '^FCHI', 'Société Générale': '^FCHI',
    'Santander': '^IBEX', 'HSBC': '^FTSE', 'Barclays': '^FTSE', 'Standard Chartered': '^FTSE',
    'Deutsche Bank': '^GDAXI', 'UBS': '^SSMI', 'ING': '^AEX',
    
    # Asia/Pacific
    'China Construction Bank': '000001.SS', 'Agricultural Bank of China': '000001.SS',
    'ICBC': '000001.SS', 'Bank of Communications': '000001.SS', 'Bank of China': '000001.SS',
    'Mitsubishi UFJ FG': '^N225', 'Sumitomo Mitsui FG': '^N225', 'Mizuho FG': '^N225'
}

ENHANCED_IDX_NAME_MAP = {
    '^GSPC': 'S&P 500', '^GSPTSE': 'TSX Composite', '^FCHI': 'CAC 40',
    '^IBEX': 'IBEX 35', '^FTSE': 'FTSE 100', '^GDAXI': 'DAX',
    '^SSMI': 'SMI', '^N225': 'Nikkei 225', '000001.SS': 'Shanghai Composite', '^AEX': 'AEX'
}

REGION_MAP = {
    'JPMorgan Chase': 'Americas', 'Citigroup': 'Americas', 'Bank of America': 'Americas',
    'Wells Fargo': 'Americas', 'Goldman Sachs': 'Americas', 'Morgan Stanley': 'Americas',
    'Bank of New York Mellon': 'Americas', 'State Street': 'Americas',
    'Royal Bank of Canada': 'Americas', 'Toronto Dominion': 'Americas',
    'HSBC': 'Europe', 'Barclays': 'Europe', 'BNP Paribas': 'Europe',
    'Groupe Crédit Agricole': 'Europe', 'ING': 'Europe', 'Deutsche Bank': 'Europe',
    'Santander': 'Europe', 'Société Générale': 'Europe', 'UBS': 'Europe',
    'Standard Chartered': 'Europe', 'Agricultural Bank of China': 'Asia/Pacific',
    'Bank of China': 'Asia/Pacific', 'China Construction Bank': 'Asia/Pacific',
    'ICBC': 'Asia/Pacific', 'Bank of Communications': 'Asia/Pacific',
    'Mitsubishi UFJ FG': 'Asia/Pacific', 'Mizuho FG': 'Asia/Pacific',
    'Sumitomo Mitsui FG': 'Asia/Pacific'
}

CRISIS_PERIODS = {
    'eurozone_crisis': (pd.Timestamp('2011-07-01'), pd.Timestamp('2012-12-31')),
    'china_correction': (pd.Timestamp('2015-06-01'), pd.Timestamp('2016-02-29')),
    'covid_crash': (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-05-31')),
    'ukraine_war': (pd.Timestamp('2022-02-01'), pd.Timestamp('2022-06-30')),
    'banking_stress_2023': (pd.Timestamp('2023-03-01'), pd.Timestamp('2023-05-31'))
}

class EnhancedEVTCalculator:
    """Enhanced Extreme Value Theory calculator with improved accuracy"""
    
    def __init__(self, params: EVTParams = None):
        self.params = params or EVTParams()
        
    def calculate_var(self, returns: np.ndarray, alpha: float = 0.95, 
                     method: str = "empirical") -> float:
        """
        Calculate Value-at-Risk with multiple estimation methods
        
        Args:
            returns: Array of return observations
            alpha: Confidence level
            method: Estimation method ("empirical", "parametric", "cornish_fisher")
        
        Returns:
            VaR estimate (positive value)
        """
        if len(returns) == 0 or np.isnan(returns).all():
            return np.nan
            
        clean_returns = returns[~np.isnan(returns)]
        
        if method == "empirical":
            return -np.percentile(clean_returns, 100 * (1 - alpha))
        
        elif method == "threshold":
            # Threshold-based method with automatic threshold selection
            thresholds_x = np.linspace(0.8, 0.98, 20)
            tau_estimates = []
            
            for q in thresholds_x:
                threshold_x = np.quantile(x_clean, q)
                threshold_y = np.quantile(y_clean, q)
                
                tail_x_mask = x_clean <= threshold_x
                n_tail = np.sum(tail_x_mask)
                
                if n_tail >= 10:  # Minimum observations in tail
                    joint_tail = np.sum((x_clean <= threshold_x) & (y_clean <= threshold_y))
                    tau_est = joint_tail / n_tail if n_tail > 0 else 0
                    tau_estimates.append(tau_est)
            
            return np.mean(tau_estimates) if tau_estimates else np.nan
        
        elif method == "copula":
            # Empirical copula-based estimation
            from scipy.stats import rankdata
            
            # Convert to pseudo-observations (ranks)
            u = rankdata(x_clean) / (len(x_clean) + 1)
            v = rankdata(y_clean) / (len(y_clean) + 1)
            
            # Calculate tail dependence using copula
            threshold = 1 - confidence_level
            tail_mask = u <= threshold
            
            if np.sum(tail_mask) == 0:
                return np.nan
            
            return np.sum((u <= threshold) & (v <= threshold)) / np.sum(tail_mask)
        
        else:
            raise ValueError(f"Unknown tail dependence method: {method}")
    
    def systemic_beta_enhanced(self, bank_returns: np.ndarray, system_returns: np.ndarray,
                             confidence_level: float = 0.95) -> Tuple[float, Dict]:
        """
        Enhanced systemic beta calculation with confidence intervals
        
        Args:
            bank_returns: Individual bank return series
            system_returns: System/index return series
            confidence_level: Confidence level for calculations
        
        Returns:
            Tuple of (systemic_beta, diagnostics_dict)
        """
        if len(bank_returns) != len(system_returns) or len(bank_returns) == 0:
            return np.nan, {}
        
        # Calculate components
        var_bank = self.calculate_var(bank_returns, confidence_level, method="cornish_fisher")
        var_system = self.calculate_var(system_returns, confidence_level, method="cornish_fisher")
        
        hill_system, hill_threshold = self.hill_estimator_adaptive(system_returns, confidence_level)
        tau = self.tail_dependence_coefficient(bank_returns, system_returns, 
                                             confidence_level, method="threshold")
        
        # Diagnostics
        diagnostics = {
            'var_bank': var_bank,
            'var_system': var_system,
            'hill_system': hill_system,
            'hill_threshold': hill_threshold,
            'tau': tau,
            'n_observations': len(bank_returns),
            'method': 'enhanced'
        }
        
        # Check validity of components
        if (np.isnan(hill_system) or hill_system <= 0 or 
            np.isnan(tau) or var_system <= 0):
            return np.nan, diagnostics
        
        # Calculate systemic beta with numerical stability
        try:
            tau_adj = np.clip(tau, 1e-6, 1-1e-6)  # Avoid extreme values
            hill_adj = np.clip(hill_system, 1e-6, 10)  # Reasonable range for hill estimator
            
            beta_t = (tau_adj ** (1.0 / hill_adj)) * (var_bank / var_system)
            
            # Additional stability check
            if not np.isfinite(beta_t) or beta_t < 0:
                return np.nan, diagnostics
            
            return beta_t, diagnostics
            
        except (ZeroDivisionError, OverflowError, ValueError):
            return np.nan, diagnostics
    
    def calculate_expected_shortfall(self, returns: np.ndarray, 
                                   alpha: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Array of return observations
            alpha: Confidence level
        
        Returns:
            Expected shortfall estimate
        """
        if len(returns) == 0:
            return np.nan
        
        clean_returns = returns[~np.isnan(returns)]
        var_threshold = -self.calculate_var(clean_returns, alpha)
        
        # Calculate ES as mean of returns below VaR threshold
        tail_returns = clean_returns[clean_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return np.nan
        
        return -np.mean(tail_returns)
    
    def bootstrap_confidence_intervals(self, bank_returns: np.ndarray, 
                                     system_returns: np.ndarray,
                                     confidence_level: float = 0.95,
                                     n_bootstrap: int = 100,
                                     ci_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for systemic risk metrics
        
        Args:
            bank_returns: Individual bank return series
            system_returns: System return series
            confidence_level: Confidence level for risk metrics
            n_bootstrap: Number of bootstrap samples
            ci_level: Confidence level for confidence intervals
        
        Returns:
            Dictionary of confidence intervals for each metric
        """
        if len(bank_returns) < 20:
            return {}
        
        n_obs = len(bank_returns)
        bootstrap_results = {
            'beta_t': [],
            'var_bank': [],
            'tau': [],
            'hill_system': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            boot_bank = bank_returns[indices]
            boot_system = system_returns[indices]
            
            # Calculate metrics
            try:
                beta_t, diagnostics = self.systemic_beta_enhanced(boot_bank, boot_system, confidence_level)
                
                if not np.isnan(beta_t):
                    bootstrap_results['beta_t'].append(beta_t)
                    bootstrap_results['var_bank'].append(diagnostics.get('var_bank', np.nan))
                    bootstrap_results['tau'].append(diagnostics.get('tau', np.nan))
                    bootstrap_results['hill_system'].append(diagnostics.get('hill_system', np.nan))
            except:
                continue
        
        # Calculate confidence intervals
        alpha_ci = 1 - ci_level
        confidence_intervals = {}
        
        for metric, values in bootstrap_results.items():
            if len(values) > 10:  # Minimum successful bootstrap samples
                lower = np.percentile(values, 100 * alpha_ci / 2)
                upper = np.percentile(values, 100 * (1 - alpha_ci / 2))
                confidence_intervals[metric] = (lower, upper)
        
        return confidence_intervals


class EnhancedDataProcessor:
    """Enhanced data processor with improved accuracy and robustness"""
    
    def __init__(self, evt_params: EVTParams = None):
        self.evt_params = evt_params or EVTParams()
        self.evt_calculator = EnhancedEVTCalculator(self.evt_params)
        self.cache = {}
        
    def download_data(self, start_date: str = '2011-01-01', 
                     end_date: str = '2024-12-31',
                     selected_banks: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Enhanced data download with error handling and data quality checks
        
        Args:
            start_date: Start date for data download
            end_date: End date for data download
            selected_banks: List of bank names to download (None for all)
        
        Returns:
            Tuple of (bank_prices, index_prices)
        """
        cache_key = f"{start_date}_{end_date}_{hash(tuple(selected_banks or []))}"
        
        if cache_key in self.cache:
            logger.info("Using cached data")
            return self.cache[cache_key]
        
        try:
            # Filter banks if specified
            if selected_banks:
                bank_tickers = {k: v for k, v in ENHANCED_BANK_DICT.items() 
                              if v in selected_banks}
            else:
                bank_tickers = ENHANCED_BANK_DICT
            
            logger.info(f"Downloading data for {len(bank_tickers)} banks")
            
            # Download bank data with retry logic
            bank_data = {}
            failed_tickers = []
            
            for ticker, name in bank_tickers.items():
                try:
                    data = yf.download(ticker, start=start_date, end=end_date, 
                                     progress=False, threads=True)
                    if not data.empty and 'Close' in data.columns:
                        bank_data[name] = data['Close']
                    else:
                        failed_tickers.append(ticker)
                        logger.warning(f"No data available for {ticker} ({name})")
                except Exception as e:
                    failed_tickers.append(ticker)
                    logger.error(f"Failed to download {ticker}: {str(e)}")
            
            bank_prices = pd.DataFrame(bank_data)
            
            # Download index data
            unique_indices = list(set(ENHANCED_INDEX_MAP.values()))
            index_data = {}
            
            for idx_ticker in unique_indices:
                try:
                    data = yf.download(idx_ticker, start=start_date, end=end_date, 
                                     progress=False, threads=True)
                    if not data.empty and 'Close' in data.columns:
                        idx_name = ENHANCED_IDX_NAME_MAP.get(idx_ticker, idx_ticker)
                        index_data[idx_name] = data['Close']
                except Exception as e:
                    logger.error(f"Failed to download index {idx_ticker}: {str(e)}")
            
            index_prices = pd.DataFrame(index_data)
            
            # Data quality checks
            bank_prices = self._quality_check_data(bank_prices, "bank")
            index_prices = self._quality_check_data(index_prices, "index")
            
            # Cache results
            self.cache[cache_key] = (bank_prices, index_prices)
            
            logger.info(f"Successfully downloaded data: {len(bank_prices.columns)} banks, "
                       f"{len(index_prices.columns)} indices")
            
            return bank_prices, index_prices
            
        except Exception as e:
            logger.error(f"Data download failed: {str(e)}")
            raise
    
    def _quality_check_data(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Perform data quality checks and cleaning
        
        Args:
            data: Price data DataFrame
            data_type: Type of data ("bank" or "index")
        
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data
        
        original_cols = len(data.columns)
        
        # Remove columns with insufficient data (less than 60% coverage)
        min_obs = len(data) * 0.6
        data = data.dropna(thresh=min_obs, axis=1)
        
        # Remove columns with constant values
        data = data.loc[:, data.std() > 1e-8]
        
        # Forward fill missing values (max 5 consecutive days)
        data = data.fillna(method='ffill', limit=5)
        
        # Remove remaining NaN values
        data = data.dropna()
        
        removed_cols = original_cols - len(data.columns)
        if removed_cols > 0:
            logger.info(f"Removed {removed_cols} {data_type} series due to quality issues")
        
        return data
    
    def prepare_returns(self, bank_prices: pd.DataFrame, 
                       index_prices: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Enhanced return calculation with outlier detection and adjustment
        
        Args:
            bank_prices: Bank price data
            index_prices: Index price data
        
        Returns:
            Tuple of (combined_returns, bank_names)
        """
        # Resample to weekly frequency (Friday close)
        weekly_banks = bank_prices.resample('W-FRI').last()
        weekly_indices = index_prices.resample('W-FRI').last()
        
        # Calculate log returns
        bank_returns = np.log(weekly_banks / weekly_banks.shift(1)).dropna()
        index_returns = np.log(weekly_indices / weekly_indices.shift(1)).dropna()
        
        # Outlier detection and treatment
        bank_returns = self._treat_outliers(bank_returns)
        index_returns = self._treat_outliers(index_returns)
        
        # Combine data
        combined_returns = bank_returns.join(index_returns, how='inner')
        
        # Remove any remaining columns with insufficient data
        min_obs = len(combined_returns) * 0.8
        combined_returns = combined_returns.dropna(thresh=min_obs, axis=1)
        
        bank_names = [col for col in combined_returns.columns 
                     if col in bank_returns.columns]
        
        logger.info(f"Prepared returns: {len(bank_names)} banks, "
                   f"{len(combined_returns)} weekly observations")
        
        return combined_returns, bank_names
    
    def _treat_outliers(self, returns: pd.DataFrame, 
                       method: str = "winsorize", threshold: float = 0.01) -> pd.DataFrame:
        """
        Detect and treat outliers in return data
        
        Args:
            returns: Return data
            method: Treatment method ("winsorize", "cap", "remove")
            threshold: Threshold for outlier detection
        
        Returns:
            Treated return data
        """
        if method == "winsorize":
            # Winsorize at specified percentiles
            lower = returns.quantile(threshold)
            upper = returns.quantile(1 - threshold)
            
            return returns.clip(lower=lower, upper=upper, axis=1)
        
        elif method == "cap":
            # Cap extreme values using IQR method
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            
            return returns.clip(lower=lower, upper=upper, axis=1)
        
        elif method == "remove":
            # Remove extreme observations
            z_scores = np.abs(stats.zscore(returns, nan_policy='omit'))
            return returns.where(z_scores < 3.5, np.nan)
        
        else:
            return returns
    
    def compute_rolling_metrics(self, combined_data: pd.DataFrame, 
                              bank_names: List[str],
                              window_size: int = 52,
                              min_periods: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Enhanced rolling metrics computation with confidence intervals
        
        Args:
            combined_data: Combined return data
            bank_names: List of bank names
            window_size: Rolling window size
            min_periods: Minimum periods for calculation
        
        Returns:
            Dictionary of DataFrames with metrics for different confidence levels
        """
        results = {f'metrics_{int(alpha*100)}': [] for alpha in self.evt_params.confidence_levels}
        
        dates = combined_data.index[max(window_size, min_periods):]
        total_iterations = len(dates) * len(bank_names)
        current_iteration = 0
        
        logger.info(f"Computing rolling metrics for {len(dates)} periods and {len(bank_names)} banks")
        
        for date in dates:
            # Get rolling window
            window_data = combined_data.loc[:date].tail(window_size)
            
            if len(window_data) < min_periods:
                continue
            
            for bank in bank_names:
                current_iteration += 1
                
                if bank not in combined_data.columns:
                    continue
                
                # Get corresponding regional index
                if bank not in ENHANCED_INDEX_MAP:
                    continue
                
                idx_ticker = ENHANCED_INDEX_MAP[bank]
                idx_name = ENHANCED_IDX_NAME_MAP.get(idx_ticker, idx_ticker)
                
                if idx_name not in combined_data.columns:
                    continue
                
                # Extract return series
                bank_returns = window_data[bank].dropna().values
                index_returns = window_data[idx_name].dropna().values
                
                # Align series
                min_len = min(len(bank_returns), len(index_returns))
                if min_len < min_periods:
                    continue
                
                bank_returns = bank_returns[-min_len:]
                index_returns = index_returns[-min_len:]
                
                # Calculate metrics for each confidence level
                for alpha in self.evt_params.confidence_levels:
                    metrics_key = f'metrics_{int(alpha*100)}'
                    
                    # Enhanced VaR calculation
                    var_bank = self.evt_calculator.calculate_var(bank_returns, alpha, 
                                                               method="cornish_fisher")
                    var_index = self.evt_calculator.calculate_var(index_returns, alpha,
                                                                method="cornish_fisher")
                    
                    # Enhanced Hill estimator
                    hill_bank, _ = self.evt_calculator.hill_estimator_adaptive(bank_returns, alpha)
                    hill_index, _ = self.evt_calculator.hill_estimator_adaptive(index_returns, alpha)
                    
                    # Enhanced tail dependence
                    tau = self.evt_calculator.tail_dependence_coefficient(
                        bank_returns, index_returns, alpha, method="threshold")
                    
                    # Enhanced systemic beta
                    beta_t, diagnostics = self.evt_calculator.systemic_beta_enhanced(
                        bank_returns, index_returns, alpha)
                    
                    # Expected Shortfall
                    es_bank = self.evt_calculator.calculate_expected_shortfall(bank_returns, alpha)
                    
                    # Confidence intervals (every 10th calculation to save time)
                    confidence_intervals = {}
                    if current_iteration % 10 == 0:
                        confidence_intervals = self.evt_calculator.bootstrap_confidence_intervals(
                            bank_returns, index_returns, alpha, n_bootstrap=50)
                    
                    # Store results
                    result_dict = {
                        'Date': date,
                        'Bank': bank,
                        'Region': REGION_MAP.get(bank, 'Unknown'),
                        f'VaR_{int(alpha*100)}': var_bank,
                        f'VaR_Index_{int(alpha*100)}': var_index,
                        f'Hill_{int(alpha*100)}': hill_bank,
                        f'Hill_Index_{int(alpha*100)}': hill_index,
                        f'Tau_{int(alpha*100)}': tau,
                        f'Beta_T_{int(alpha*100)}': beta_t,
                        f'ES_{int(alpha*100)}': es_bank,
                        'n_observations': len(bank_returns),
                        'data_quality_score': self._calculate_data_quality_score(bank_returns)
                    }
                    
                    # Add confidence intervals if available
                    for metric, (lower, upper) in confidence_intervals.items():
                        result_dict[f'{metric}_ci_lower'] = lower
                        result_dict[f'{metric}_ci_upper'] = upper
                    
                    results[metrics_key].append(result_dict)
        
        # Convert to DataFrames
        output_dfs = {}
        for key, data in results.items():
            if data:
                df = pd.DataFrame(data)
                df = df.set_index(['Date', 'Bank'])
                output_dfs[key] = df
                logger.info(f"Computed {key}: {len(df)} observations")
        
        return output_dfs
    
    def _calculate_data_quality_score(self, returns: np.ndarray) -> float:
        """
        Calculate a data quality score for the return series
        
        Args:
            returns: Return observations
        
        Returns:
            Quality score between 0 and 1
        """
        if len(returns) == 0:
            return 0.0
        
        # Factors affecting quality
        completeness = 1.0  # Assumed complete since we pre-filter
        
        # Volatility reasonableness (penalize extreme volatility)
        vol = np.std(returns)
        vol_score = 1.0 / (1.0 + max(0, vol - 0.1) * 10)  # Penalize vol > 10%
        
        # Return distribution reasonableness
        skewness = abs(stats.skew(returns))
        skew_score = 1.0 / (1.0 + max(0, skewness - 2) * 0.5)  # Penalize extreme skew
        
        # Number of observations
        n_score = min(1.0, len(returns) / 52)  # Full score for 1+ year of data
        
        return np.mean([completeness, vol_score, skew_score, n_score])
    
    def get_risk_classification(self, beta_t: float) -> RiskLevel:
        """
        Classify systemic risk level based on beta_t value
        
        Args:
            beta_t: Systemic beta value
        
        Returns:
            Risk level classification
        """
        if np.isnan(beta_t):
            return RiskLevel.LOW
        elif beta_t >= 3.0:
            return RiskLevel.CRITICAL
        elif beta_t >= 2.0:
            return RiskLevel.HIGH
        elif beta_t >= 1.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def stress_test_enhanced(self, metrics_df: pd.DataFrame, 
                           shock_scenarios: Dict[str, Dict] = None) -> pd.DataFrame:
        """
        Enhanced stress testing with multiple scenarios and spillover effects
        
        Args:
            metrics_df: Latest metrics DataFrame
            shock_scenarios: Dictionary of shock scenarios
        
        Returns:
            Stress test results DataFrame
        """
        if shock_scenarios is None:
            shock_scenarios = {
                'moderate_stress': {'shock_magnitude': 0.15, 'affected_regions': ['Americas']},
                'severe_stress': {'shock_magnitude': 0.30, 'affected_regions': ['Americas', 'Europe']},
                'extreme_stress': {'shock_magnitude': 0.50, 'affected_regions': ['Americas', 'Europe', 'Asia/Pacific']}
            }
        
        latest_date = metrics_df.index.get_level_values(0).max()
        latest_metrics = metrics_df.loc[latest_date]
        
        stress_results = []
        
        for scenario_name, scenario_params in shock_scenarios.items():
            shock_magnitude = scenario_params['shock_magnitude']
            affected_regions = scenario_params['affected_regions']
            
            for bank in latest_metrics.index:
                bank_region = REGION_MAP.get(bank, 'Unknown')
                
                # Determine shock type and magnitude
                if bank_region in affected_regions:
                    # Direct shock
                    shock_multiplier = 1 + shock_magnitude
                    impact_type = 'Direct'
                else:
                    # Spillover effect based on systemic beta
                    beta_t = latest_metrics.loc[bank, 'Beta_T_95']
                    if pd.isna(beta_t):
                        beta_t = 1.0
                    
                    spillover_factor = min(beta_t * 0.4, 0.8)  # Max 80% spillover
                    shock_multiplier = 1 + shock_magnitude * spillover_factor
                    impact_type = 'Spillover'
                
                # Calculate shocked metrics
                original_var = latest_metrics.loc[bank, 'VaR_95']
                shocked_var = original_var * shock_multiplier
                
                # Enhanced impact calculation
                var_impact = (shocked_var / original_var - 1) * 100
                
                # Risk level change
                original_beta = latest_metrics.loc[bank, 'Beta_T_95']
                original_risk = self.get_risk_classification(original_beta)
                
                # Estimate shocked beta (simplified)
                shocked_beta = original_beta * (shock_multiplier ** 0.5) if not pd.isna(original_beta) else np.nan
                shocked_risk = self.get_risk_classification(shocked_beta)
                
                stress_results.append({
                    'Scenario': scenario_name,
                    'Bank': bank,
                    'Region': bank_region,
                    'Impact_Type': impact_type,
                    'Original_VaR': original_var,
                    'Shocked_VaR': shocked_var,
                    'VaR_Impact_Pct': var_impact,
                    'Original_Beta': original_beta,
                    'Shocked_Beta': shocked_beta,
                    'Original_Risk_Level': original_risk.value,
                    'Shocked_Risk_Level': shocked_risk.value,
                    'Risk_Level_Change': shocked_risk != original_risk
                })
        
        return pd.DataFrame(stress_results)


# Factory function for easy initialization
def create_enhanced_processor(confidence_levels: List[float] = None,
                            min_exceedances: int = 10,
                            threshold_strategy: str = "adaptive") -> EnhancedDataProcessor:
    """
    Factory function to create an enhanced data processor
    
    Args:
        confidence_levels: List of confidence levels to calculate
        min_exceedances: Minimum number of exceedances for Hill estimator
        threshold_strategy: Strategy for threshold selection
    
    Returns:
        Configured EnhancedDataProcessor instance
    """
    params = EVTParams(
        confidence_levels=confidence_levels or [0.95, 0.99],
        min_exceedances=min_exceedances,
        hill_threshold_strategy=threshold_strategy
    )
    
    return EnhancedDataProcessor(params)


# Example usage and testing functions
def run_enhanced_analysis_example():
    """
    Example function demonstrating enhanced analysis capabilities
    """
    # Create processor
    processor = create_enhanced_processor(
        confidence_levels=[0.95, 0.99],
        min_exceedances=15,
        threshold_strategy="adaptive"
    )
    
    # Download data for selected banks
    selected_banks = ['JPMorgan Chase', 'Bank of America', 'Deutsche Bank', 'HSBC']
    
    try:
        bank_prices, index_prices = processor.download_data(
            start_date='2020-01-01',
            end_date='2024-12-31',
            selected_banks=selected_banks
        )
        
        # Prepare returns
        combined_returns, bank_names = processor.prepare_returns(bank_prices, index_prices)
        
        # Compute rolling metrics
        metrics_dict = processor.compute_rolling_metrics(
            combined_returns, bank_names, window_size=52, min_periods=30
        )
        
        # Display results
        for confidence_level, metrics_df in metrics_dict.items():
            print(f"\n{confidence_level.upper()} Results:")
            print(f"Shape: {metrics_df.shape}")
            print(f"Latest date: {metrics_df.index.get_level_values(0).max()}")
            
            # Show summary statistics
            if not metrics_df.empty:
                latest_metrics = metrics_df.loc[metrics_df.index.get_level_values(0).max()]
                print(f"Average Systemic Beta: {latest_metrics['Beta_T_95'].mean():.3f}")
                print(f"Banks with High Risk (β > 2.0): {(latest_metrics['Beta_T_95'] > 2.0).sum()}")
        
        return metrics_dict
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Run example analysis
    results = run_enhanced_analysis_example()
    if results:
        print("Enhanced analysis completed successfully!")
    else:
        print("Enhanced analysis failed. Check logs for details.")d == "parametric":
            mu, sigma = np.mean(clean_returns), np.std(clean_returns)
            z_score = stats.norm.ppf(1 - alpha)
            return -(mu + sigma * z_score)
        
        elif method == "cornish_fisher":
            # Cornish-Fisher expansion for non-normal distributions
            mu, sigma = np.mean(clean_returns), np.std(clean_returns)
            skew = stats.skew(clean_returns)
            kurt = stats.kurtosis(clean_returns)
            
            z = stats.norm.ppf(1 - alpha)
            z_cf = (z + (z**2 - 1) * skew / 6 + 
                   (z**3 - 3*z) * kurt / 24 - 
                   (2*z**3 - 5*z) * skew**2 / 36)
            
            return -(mu + sigma * z_cf)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def hill_estimator_adaptive(self, returns: np.ndarray, 
                              confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Enhanced Hill estimator with adaptive threshold selection
        
        Args:
            returns: Array of return observations
            confidence_level: Confidence level for threshold selection
        
        Returns:
            Tuple of (hill_estimate, optimal_threshold)
        """
        if len(returns) < self.params.min_exceedances * 2:
            return np.nan, np.nan
            
        clean_returns = returns[~np.isnan(returns)]
        losses = -clean_returns[clean_returns < 0]
        
        if len(losses) < self.params.min_exceedances:
            return np.nan, np.nan
        
        # Generate candidate thresholds
        sorted_losses = np.sort(losses)[::-1]  # Descending order
        min_idx = self.params.min_exceedances
        max_idx = min(len(sorted_losses) - 1, int(len(sorted_losses) * 0.5))
        
        if max_idx <= min_idx:
            return np.nan, np.nan
        
        candidates = sorted_losses[min_idx:max_idx]
        hill_estimates = []
        
        for threshold in candidates:
            exceedances = losses[losses >= threshold]
            if len(exceedances) >= self.params.min_exceedances:
                # Hill estimator calculation
                log_ratios = np.log(exceedances / threshold)
                hill_est = np.mean(log_ratios)
                hill_estimates.append((hill_est, threshold, len(exceedances)))
        
        if not hill_estimates:
            return np.nan, np.nan
        
        # Select optimal threshold based on stability criterion
        if self.params.hill_threshold_strategy == "adaptive":
            # Choose threshold that minimizes variance while maintaining stability
            hill_vals = [h[0] for h in hill_estimates]
            n_exceedances = [h[2] for h in hill_estimates]
            
            # Weighted by sample size and stability
            weights = np.array(n_exceedances) / np.sum(n_exceedances)
            stability_scores = 1.0 / (1.0 + np.array([abs(h - np.median(hill_vals)) 
                                                    for h in hill_vals]))
            
            combined_weights = weights * stability_scores
            optimal_idx = np.argmax(combined_weights)
            
            return hill_estimates[optimal_idx][0], hill_estimates[optimal_idx][1]
        
        elif self.params.hill_threshold_strategy == "fixed":
            # Use fixed quantile threshold
            target_quantile = 1 - confidence_level
            threshold = np.quantile(losses, 1 - target_quantile)
            exceedances = losses[losses >= threshold]
            
            if len(exceedances) >= self.params.min_exceedances:
                log_ratios = np.log(exceedances / threshold)
                return np.mean(log_ratios), threshold
            else:
                return np.nan, np.nan
        
        else:  # optimal
            # Use method with minimum mean squared error
            hill_vals = [h[0] for h in hill_estimates]
            mse_scores = []
            
            for i, (hill_est, threshold, n_exc) in enumerate(hill_estimates):
                # Bootstrap estimation of MSE
                bootstrap_estimates = []
                for _ in range(min(50, self.params.bootstrap_samples)):
                    try:
                        boot_sample = np.random.choice(losses[losses >= threshold], 
                                                     size=min(n_exc, len(losses[losses >= threshold])), 
                                                     replace=True)
                        boot_hill = np.mean(np.log(boot_sample / threshold))
                        bootstrap_estimates.append(boot_hill)
                    except:
                        continue
                
                if bootstrap_estimates:
                    mse = np.var(bootstrap_estimates)
                    mse_scores.append(mse)
                else:
                    mse_scores.append(np.inf)
            
            if mse_scores and not all(np.isinf(mse_scores)):
                optimal_idx = np.argmin(mse_scores)
                return hill_estimates[optimal_idx][0], hill_estimates[optimal_idx][1]
            else:
                return hill_estimates[0][0], hill_estimates[0][1]
    
    def tail_dependence_coefficient(self, x: np.ndarray, y: np.ndarray, 
                                  confidence_level: float = 0.95,
                                  method: str = "empirical") -> float:
        """
        Enhanced tail dependence coefficient calculation
        
        Args:
            x, y: Return series for bank and system
            confidence_level: Confidence level for tail definition
            method: Estimation method ("empirical", "threshold", "copula")
        
        Returns:
            Tail dependence coefficient
        """
        if len(x) != len(y) or len(x) == 0:
            return np.nan
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 20:  # Minimum sample size
            return np.nan
        
        if method == "empirical":
            # Standard empirical method
            threshold_x = np.quantile(x_clean, 1 - confidence_level)
            threshold_y = np.quantile(y_clean, 1 - confidence_level)
            
            tail_x_mask = x_clean <= threshold_x
            if np.sum(tail_x_mask) == 0:
                return np.nan
            
            joint_tail_prob = np.sum((x_clean <= threshold_x) & (y_clean <= threshold_y))
            marginal_tail_prob = np.sum(tail_x_mask)
            
            return joint_tail_prob / marginal_tail_prob
        
        elif metho
