#!/usr/bin/env python3
"""
Analyze selected banks for systemic risk metrics
"""

import pandas as pd
import numpy as np
from data_processor import BankingDataProcessor
import datetime as dt

def analyze_banks(selected_banks):
    """Analyze systemic risk metrics for selected banks"""
    print("🚀 Starting systemic risk analysis...")
    print(f"📊 Analyzing {len(selected_banks)} banks:")
    for bank in selected_banks:
        print(f"   • {bank}")
    
    # Create data processor
    processor = BankingDataProcessor()
    
    # Set date range (last 5 years)
    end_date = dt.datetime.now().strftime('%Y-%m-%d')
    start_date = '2019-01-01'
    
    print(f"\n📅 Date range: {start_date} to {end_date}")
    
    try:
        # Download bank data
        print("\n📥 Downloading bank data...")
        processor.download_bank_data(selected_banks, start_date=start_date, end_date=end_date)
        
        # Download market index data
        print("\n📥 Downloading market index data...")
        processor.download_index_data(start_date=start_date, end_date=end_date)
        
        # Combine data
        print("\n🔗 Combining data...")
        processor.combine_data()
        
        # Calculate rolling metrics
        print("\n📈 Calculating rolling metrics...")
        results_95, results_99 = processor.calculate_rolling_metrics(window_size=52)
        
        # Get latest metrics
        print("\n📊 Latest metrics (95% confidence):")
        latest_95 = processor.get_latest_metrics(confidence_level=0.95)
        print(latest_95.to_string(index=False))
        
        print("\n📊 Latest metrics (99% confidence):")
        latest_99 = processor.get_latest_metrics(confidence_level=0.99)
        print(latest_99.to_string(index=False))
        
        # Get summary statistics
        print("\n📋 Summary statistics by region (95% confidence):")
        summary_95 = processor.get_summary_statistics(confidence_level=0.95)
        print(summary_95)
        
        print("\n📋 Summary statistics by region (99% confidence):")
        summary_99 = processor.get_summary_statistics(confidence_level=0.99)
        print(summary_99)
        
        # Save results to CSV
        results_95.to_csv('systemic_risk_results_95.csv', index=False)
        results_99.to_csv('systemic_risk_results_99.csv', index=False)
        print(f"\n💾 Results saved to:")
        print(f"   • systemic_risk_results_95.csv ({len(results_95)} records)")
        print(f"   • systemic_risk_results_99.csv ({len(results_99)} records)")
        
        return processor, results_95, results_99
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None, None, None

def main():
    """Main function to run the analysis"""
    # Banks selected by user
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
    
    # Run analysis
    processor, results_95, results_99 = analyze_banks(selected_banks)
    
    if processor is not None:
        print("\n✅ Analysis completed successfully!")
        print("\n🔍 Key insights:")
        
        # Find highest systemic risk banks
        if results_95 is not None and not results_95.empty:
            latest_95 = processor.get_latest_metrics(confidence_level=0.95)
            if not latest_95.empty:
                highest_beta = latest_95.loc[latest_95['Beta_T'].idxmax()]
                print(f"   • Highest systemic risk (Beta_T): {highest_beta['Bank']} ({highest_beta['Beta_T']:.4f})")
                
                highest_var = latest_95.loc[latest_95['VaR_95'].idxmax()]
                print(f"   • Highest VaR: {highest_var['Bank']} ({highest_var['VaR_95']:.4f})")
                
                highest_tau = latest_95.loc[latest_95['Tau_95'].idxmax()]
                print(f"   • Highest tail dependence: {highest_tau['Bank']} ({highest_tau['Tau_95']:.4f})")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 