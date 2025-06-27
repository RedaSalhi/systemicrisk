#!/usr/bin/env python3
"""
Example usage of the BankingDataProcessor

This script demonstrates how to use the new data processor
to analyze systemic risk for user-selected banks.
"""

from data_processor import BankingDataProcessor, process_banking_data
import pandas as pd

def main():
    print("🏦 Banking Systemic Risk Analysis Example")
    print("=" * 50)
    
    # Initialize the processor
    processor = BankingDataProcessor()
    
    # Show available banks
    print("\n📋 Available Banks by Region:")
    banks_by_region = processor.get_banks_by_region()
    
    for region, banks in banks_by_region.items():
        print(f"\n{region}:")
        for bank in banks:
            print(f"  • {bank}")
    
    # Example: Select some banks for analysis
    print("\n" + "=" * 50)
    print("🔍 Example Analysis with Selected Banks")
    print("=" * 50)
    
    # Select a few banks from different regions
    selected_banks = [
        'JPMorgan Chase',      # Americas
        'Bank of America',     # Americas
        'HSBC',               # Europe
        'Deutsche Bank',      # Europe
        'Agricultural Bank of China'  # Asia/Pacific
    ]
    
    print(f"\nSelected banks: {', '.join(selected_banks)}")
    
    try:
        # Process the data
        print("\n📥 Downloading and processing data...")
        processor = process_banking_data(
            selected_banks, 
            start_date='2020-01-01', 
            end_date='2024-12-31'
        )
        
        print("✅ Data processing completed!")
        
        # Get latest metrics
        print("\n📊 Latest Risk Metrics (95% confidence):")
        latest_95 = processor.get_latest_metrics(0.95)
        print(latest_95[['Bank', 'Region', 'Beta_T', 'VaR_95', 'Tau_95']].round(4))
        
        print("\n📊 Latest Risk Metrics (99% confidence):")
        latest_99 = processor.get_latest_metrics(0.99)
        print(latest_99[['Bank', 'Region', 'Beta_T', 'VaR_99', 'Tau_99']].round(4))
        
        # Get summary statistics
        print("\n📈 Regional Summary Statistics (95% confidence):")
        summary_95 = processor.get_summary_statistics(0.95)
        print(summary_95)
        
        # Show time series for one bank
        print("\n📈 Time Series Example - JPMorgan Chase Systemic Beta:")
        jpm_beta = processor.get_bank_time_series('JPMorgan Chase', 'Beta_T', 0.95)
        print(f"Latest value: {jpm_beta.iloc[-1]:.4f}")
        print(f"Maximum value: {jpm_beta.max():.4f}")
        print(f"Minimum value: {jpm_beta.min():.4f}")
        print(f"Average value: {jpm_beta.mean():.4f}")
        
        # Risk assessment
        print("\n⚠️ Risk Assessment:")
        latest_metrics = processor.get_latest_metrics(0.95)
        high_risk = latest_metrics[latest_metrics['Beta_T'] > 2.0]
        medium_risk = latest_metrics[
            (latest_metrics['Beta_T'] > 1.5) & 
            (latest_metrics['Beta_T'] <= 2.0)
        ]
        
        if not high_risk.empty:
            print("🔴 High Risk Banks (Beta_T > 2.0):")
            for _, bank in high_risk.iterrows():
                print(f"  • {bank['Bank']}: {bank['Beta_T']:.3f}")
        else:
            print("✅ No high-risk banks detected")
            
        if not medium_risk.empty:
            print("🟡 Medium Risk Banks (1.5 < Beta_T ≤ 2.0):")
            for _, bank in medium_risk.iterrows():
                print(f"  • {bank['Bank']}: {bank['Beta_T']:.3f}")
        else:
            print("✅ No medium-risk banks detected")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("This might be due to network issues or unavailable data.")

if __name__ == "__main__":
    main() 