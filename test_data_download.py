#!/usr/bin/env python3
"""
Quick test script to verify data download fixes
"""

import pandas as pd
import numpy as np
from data_processor import BankingDataProcessor, process_banking_data

def test_data_download():
    """Test data download functionality"""
    print("🧪 Testing Data Download Fixes")
    print("=" * 40)
    
    # Test 1: Basic data processor initialization
    print("1. Testing basic initialization...")
    try:
        processor = BankingDataProcessor()
        print("✅ BankingDataProcessor initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test 2: Get available banks
    print("2. Testing bank availability...")
    try:
        banks = processor.get_available_banks()
        print(f"✅ Found {len(banks)} available banks")
        print(f"   Sample banks: {banks[:5]}")
    except Exception as e:
        print(f"❌ Bank list failed: {e}")
        return False
    
    # Test 3: Test data download with working banks
    print("3. Testing data download...")
    try:
        # Use banks that are more likely to have data
        test_banks = ['JPMorgan Chase', 'Bank of America', 'Wells Fargo']
        
        # Test the process_banking_data function
        result = process_banking_data(
            test_banks,
            start_date='2023-01-01',
            end_date='2024-01-01'
        )
        
        print("✅ Data download successful")
        print(f"   Data shape: {result.weekly_returns.shape}")
        print(f"   Date range: {result.weekly_returns.index.min()} to {result.weekly_returns.index.max()}")
        
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return False
    
    # Test 4: Test with problematic banks (should handle gracefully)
    print("4. Testing error handling...")
    try:
        # Try with some banks that might not have data
        problematic_banks = ['Test Bank 1', 'Test Bank 2']
        
        try:
            result = process_banking_data(
                problematic_banks,
                start_date='2023-01-01',
                end_date='2024-01-01'
            )
            print("⚠️  Unexpected success with invalid banks")
        except ValueError as e:
            print(f"✅ Properly handled invalid banks: {e}")
        except Exception as e:
            print(f"❌ Unexpected error with invalid banks: {e}")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

def test_ml_integration():
    """Test ML system integration"""
    print("\n🤖 Testing ML Integration")
    print("=" * 40)
    
    try:
        from pages.machinelearning import SystemicRiskEarlyWarning
        
        # Initialize ML system
        ews = SystemicRiskEarlyWarning()
        print("✅ SystemicRiskEarlyWarning initialized")
        
        # Test data loading
        test_banks = ['JPMorgan Chase', 'Bank of America']
        processor = process_banking_data(
            test_banks,
            start_date='2023-01-01',
            end_date='2024-01-01'
        )
        
        # Test data loading into ML system
        success = ews.load_data_from_processor(processor)
        if success:
            print("✅ ML data loading successful")
            
            # Test feature engineering
            features_df = ews.engineer_features()
            print(f"✅ Feature engineering successful: {features_df.shape}")
            
            # Test dataset preparation
            X, y = ews.prepare_ml_dataset(lead_weeks=8)
            print(f"✅ Dataset preparation successful: {X.shape}, {y.shape}")
            
        else:
            print("❌ ML data loading failed")
            return False
            
    except Exception as e:
        print(f"❌ ML integration test failed: {e}")
        return False
    
    print("🎉 ML integration tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Systemic Risk Platform - Data Download Test")
    print("=" * 50)
    
    # Run tests
    data_test = test_data_download()
    ml_test = test_ml_integration()
    
    if data_test and ml_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("The data download fixes are working correctly.")
    else:
        print("\n❌ Some tests failed.")
        print("Please check the error messages above.") 