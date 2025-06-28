#!/usr/bin/env python3
"""
Test script using sample data to verify the numpy.ndarray isna fix
"""

import numpy as np
import pandas as pd
from data_processor import BankingDataProcessor
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create realistic sample banking data for testing"""
    print("📊 Creating sample banking data...")
    
    # Create date range
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='W-FRI')
    
    # Create sample bank returns with realistic patterns
    np.random.seed(42)  # For reproducible results
    
    banks = [
        "JPMorgan Chase",
        "Citigroup", 
        "Bank of America",
        "HSBC Holdings",
        "Barclays",
        "BNP Paribas",
        "ABC China",
        "Bank of China"
    ]
    
    # Create sample returns data
    bank_returns = {}
    for bank in banks:
        # Generate realistic returns with some NaN values
        returns = np.random.normal(0.001, 0.03, len(dates))  # Weekly returns ~0.1%, std ~3%
        
        # Add some NaN values (missing data)
        nan_indices = np.random.choice(len(dates), size=int(len(dates) * 0.05), replace=False)
        returns[nan_indices] = np.nan
        
        bank_returns[bank] = returns
    
    # Create sample index returns
    indices = {
        '^GSPC': np.random.normal(0.001, 0.025, len(dates)),  # S&P 500
        '^FTSE': np.random.normal(0.0008, 0.025, len(dates)),  # FTSE 100
        '^FCHI': np.random.normal(0.0008, 0.026, len(dates)),  # CAC 40
        '^HSI': np.random.normal(0.0012, 0.028, len(dates)),   # Hang Seng
    }
    
    # Add some NaN values to indices too
    for idx_name, idx_returns in indices.items():
        nan_indices = np.random.choice(len(dates), size=int(len(dates) * 0.03), replace=False)
        idx_returns[nan_indices] = np.nan
    
    # Create DataFrames
    bank_df = pd.DataFrame(bank_returns, index=dates)
    index_df = pd.DataFrame(indices, index=dates)
    
    print(f"✅ Created sample data:")
    print(f"   Banks: {bank_df.shape}")
    print(f"   Indices: {index_df.shape}")
    print(f"   Date range: {dates.min().date()} to {dates.max().date()}")
    
    return bank_df, index_df

def test_individual_methods():
    """Test individual methods with numpy arrays"""
    print("🧪 Testing individual methods with numpy arrays...")
    
    processor = BankingDataProcessor()
    
    # Create test data with NaN values
    test_returns = np.array([0.01, 0.02, np.nan, -0.01, 0.03, np.nan, 0.01, -0.02, 0.015, -0.005])
    test_market = np.array([0.008, 0.015, np.nan, -0.008, 0.025, 0.005, 0.012, -0.015, 0.018, -0.003])
    
    print(f"Test returns: {test_returns}")
    print(f"Test market: {test_market}")
    print(f"Contains NaN: {np.isnan(test_returns).any()}")
    
    try:
        # Test calculate_var
        var_result = processor.calculate_var(test_returns, alpha=0.95)
        print(f"✅ calculate_var result: {var_result:.6f}")
        
        # Test hill_estimator
        hill_result = processor.hill_estimator(test_returns, threshold_quantile=0.95)
        print(f"✅ hill_estimator result: {hill_result:.6f}")
        
        # Test tail_dependence_coefficient
        tau_result = processor.tail_dependence_coefficient(test_returns, test_market, threshold_quantile=0.95)
        print(f"✅ tail_dependence result: {tau_result:.6f}")
        
        # Test systemic_beta
        beta_result = processor.systemic_beta(test_returns, test_market, alpha=0.95)
        print(f"✅ systemic_beta result: {beta_result:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in individual methods: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_sample_data():
    """Test the data processor with sample data"""
    print("🧪 Testing with sample data...")
    
    # Create sample data
    bank_returns, index_returns = create_sample_data()
    
    # Create processor and inject sample data
    processor = BankingDataProcessor()
    processor.weekly_returns = bank_returns
    processor.weekly_idx_returns = index_returns
    
    # Test data combination
    print("\n1. Testing data combination...")
    combined_data = processor.combine_data()
    print(f"✅ Combined data shape: {combined_data.shape}")
    
    # Test individual methods with numpy arrays
    print("\n2. Testing individual methods...")
    
    # Get some sample data
    sample_bank = combined_data.iloc[:100, 0].values  # First bank, first 100 observations
    sample_market = combined_data.iloc[:100, -1].values  # Last column (market index)
    
    print(f"Sample bank data type: {type(sample_bank)}")
    print(f"Sample market data type: {type(sample_market)}")
    print(f"Sample bank data shape: {sample_bank.shape}")
    print(f"Contains NaN: {np.isnan(sample_bank).any()}")
    
    try:
        # Test calculate_var
        var_result = processor.calculate_var(sample_bank, alpha=0.95)
        print(f"✅ calculate_var result: {var_result:.6f}")
        
        # Test hill_estimator
        hill_result = processor.hill_estimator(sample_bank, threshold_quantile=0.95)
        print(f"✅ hill_estimator result: {hill_result:.6f}")
        
        # Test tail_dependence_coefficient
        tau_result = processor.tail_dependence_coefficient(sample_bank, sample_market, threshold_quantile=0.95)
        print(f"✅ tail_dependence result: {tau_result:.6f}")
        
        # Test systemic_beta
        beta_result = processor.systemic_beta(sample_bank, sample_market, alpha=0.95)
        print(f"✅ systemic_beta result: {beta_result:.6f}")
        
    except Exception as e:
        print(f"❌ Error in individual methods: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test a simplified rolling calculation
    print("\n3. Testing simplified rolling calculation...")
    try:
        # Manually calculate metrics for one bank at one time point
        bank_name = "JPMorgan Chase"
        market_name = "^GSPC"
        
        # Get data for the last 52 weeks
        window_data = combined_data.iloc[-52:]
        bank_data = window_data[bank_name].dropna()
        market_data = window_data[market_name].dropna()
        
        print(f"Bank data length: {len(bank_data)}")
        print(f"Market data length: {len(market_data)}")
        
        if len(bank_data) >= 30 and len(market_data) >= 30:
            # Convert to numpy arrays
            bank_array = bank_data.values
            market_array = market_data.values
            
            # Calculate metrics
            var_95 = processor.calculate_var(bank_array, alpha=0.95)
            hill_95 = processor.hill_estimator(bank_array, threshold_quantile=0.95)
            tau_95 = processor.tail_dependence_coefficient(bank_array, market_array, threshold_quantile=0.95)
            beta_95 = processor.systemic_beta(bank_array, market_array, alpha=0.95)
            
            print(f"✅ Manual calculation results:")
            print(f"   VaR_95: {var_95:.6f}")
            print(f"   Hill_95: {hill_95:.6f}")
            print(f"   Tau_95: {tau_95:.6f}")
            print(f"   Beta_T: {beta_95:.6f}")
        else:
            print("⚠️ Insufficient data for manual calculation")
        
    except Exception as e:
        print(f"❌ Error in manual calculation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! The numpy.ndarray isna error has been successfully fixed.")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING NUMPY.NDARRAY ISNA FIX")
    print("=" * 60)
    
    # Test 1: Individual methods
    print("\n" + "=" * 40)
    print("TEST 1: Individual Methods")
    print("=" * 40)
    test1_passed = test_individual_methods()
    
    # Test 2: Sample data
    print("\n" + "=" * 40)
    print("TEST 2: Sample Data")
    print("=" * 40)
    test2_passed = test_with_sample_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED! The numpy.ndarray isna error has been fixed.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print("=" * 60) 