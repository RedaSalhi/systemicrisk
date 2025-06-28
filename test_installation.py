#!/usr/bin/env python3
"""
Systemic Risk Platform - Installation Test Script
Verifies that all components are working correctly
"""

import sys
import importlib
import traceback

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: {module_name} - {e}")
        return False

def test_data_processor():
    """Test the data processor functionality"""
    try:
        from data_processor import BankingDataProcessor
        
        # Test basic functionality
        processor = BankingDataProcessor()
        print("✅ Data processor: Basic initialization")
        
        # Test bank list
        banks = processor.get_available_banks()
        if len(banks) > 0:
            print(f"✅ Data processor: Found {len(banks)} banks")
        else:
            print("❌ Data processor: No banks found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_pages():
    """Test that Streamlit pages can be imported"""
    try:
        # Test main app
        import app
        print("✅ Streamlit app: Main landing page")
        
        # Test dashboard
        import pages.dashboard
        print("✅ Streamlit app: Dashboard page")
        
        # Test methodology
        import pages.methodology
        print("✅ Streamlit app: Methodology page")
        
        # Test machine learning
        import pages.machinelearning
        print("✅ Streamlit app: Machine Learning page")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit pages test failed: {e}")
        traceback.print_exc()
        return False

def test_example_usage():
    """Test the example usage script"""
    try:
        import example_usage
        print("✅ Example usage: Script imports successfully")
        return True
    except Exception as e:
        print(f"❌ Example usage test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Systemic Risk Platform - Installation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test core dependencies
    print("\n📦 Testing Core Dependencies:")
    dependencies = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("statsmodels", "Statsmodels"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("yfinance", "YFinance"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn")
    ]
    
    for module, desc in dependencies:
        total_tests += 1
        if test_import(module, desc):
            tests_passed += 1
    
    # Test data processor
    print("\n🔧 Testing Data Processor:")
    total_tests += 1
    if test_data_processor():
        tests_passed += 1
    
    # Test Streamlit pages
    print("\n📱 Testing Streamlit Pages:")
    total_tests += 1
    if test_streamlit_pages():
        tests_passed += 1
    
    # Test example usage
    print("\n📝 Testing Example Usage:")
    total_tests += 1
    if test_example_usage():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The platform is ready to use.")
        print("\n🚀 To start the application:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Try running the installation fix:")
        print("   python fix_server_install.py")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 