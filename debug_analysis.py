#!/usr/bin/env python3
"""
Debug script to identify the exact location of the numpy.ndarray isna error
"""

import traceback
import sys
from data_processor import BankingDataProcessor

def debug_analysis():
    """Run analysis with detailed error tracking"""
    
    # Selected banks
    selected_banks = [
        "JPMorgan Chase",
        "Citigroup", 
        "Bank of America",
        "HSBC Holdings",
        "Barclays",
        "BNP Paribas",
        "ABC China",
        "Bank of China"
    ]
    
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    
    print(f"🔍 Debugging analysis for {len(selected_banks)} banks")
    print(f"📅 Date range: {start_date} to {end_date}")
    
    try:
        # Step 1: Initialize processor
        print("\n1. Initializing processor...")
        processor = BankingDataProcessor()
        
        # Step 2: Download bank data
        print("\n2. Downloading bank data...")
        bank_data = processor.download_bank_data(selected_banks, start_date, end_date)
        print(f"✅ Bank data shape: {bank_data.shape}")
        
        # Step 3: Download index data
        print("\n3. Downloading index data...")
        index_data = processor.download_index_data(start_date, end_date)
        print(f"✅ Index data shape: {index_data.shape}")
        
        # Step 4: Combine data
        print("\n4. Combining data...")
        combined_data = processor.combine_data()
        print(f"✅ Combined data shape: {combined_data.shape}")
        
        # Step 5: Calculate rolling metrics
        print("\n5. Calculating rolling metrics...")
        results_95, results_99 = processor.calculate_rolling_metrics(window_size=52)
        print(f"✅ Results 95%: {len(results_95)} records")
        print(f"✅ Results 99%: {len(results_99)} records")
        
        # Step 6: Get latest metrics
        print("\n6. Getting latest metrics...")
        latest_95 = processor.get_latest_metrics(0.95)
        latest_99 = processor.get_latest_metrics(0.99)
        print(f"✅ Latest 95%: {len(latest_95)} banks")
        print(f"✅ Latest 99%: {len(latest_99)} banks")
        
        print("\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        
        # Additional debugging info
        print(f"\n🔧 Python version: {sys.version}")
        print(f"🔧 NumPy version: {getattr(__import__('numpy'), '__version__', 'Unknown')}")
        print(f"🔧 Pandas version: {getattr(__import__('pandas'), '__version__', 'Unknown')}")

if __name__ == "__main__":
    debug_analysis() 