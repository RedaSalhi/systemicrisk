#!/usr/bin/env python3
"""
Script to fix SciPy import error for Python 3.13
Run this script if you encounter: ImportError: cannot import name '_lazywhere' from 'scipy._lib._util'
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def fix_scipy_issue():
    """Fix the SciPy import issue"""
    print("🔧 Fixing SciPy import issue...")
    
    # Step 1: Uninstall problematic packages
    print("1. Uninstalling problematic packages...")
    packages_to_remove = ["scipy", "numpy", "pandas", "scikit-learn", "statsmodels"]
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}")
    
    # Step 2: Install compatible versions
    print("2. Installing compatible versions...")
    
    # Install numpy first (dependency for others)
    run_command("pip install numpy==1.25.2")
    
    # Install scipy with specific version
    run_command("pip install scipy==1.11.4")
    
    # Install other packages
    run_command("pip install pandas==2.1.4")
    run_command("pip install scikit-learn==1.3.2")
    run_command("pip install statsmodels==0.14.0")
    
    # Step 3: Verify installation
    print("3. Verifying installation...")
    try:
        import scipy
        import numpy
        import pandas
        import sklearn
        import statsmodels
        print("✅ All packages installed successfully!")
        print(f"   SciPy version: {scipy.__version__}")
        print(f"   NumPy version: {numpy.__version__}")
        print(f"   Pandas version: {pandas.__version__}")
        print(f"   Scikit-learn version: {sklearn.__version__}")
        print(f"   Statsmodels version: {statsmodels.__version__}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def install_requirements():
    """Install all requirements"""
    print("📦 Installing all requirements...")
    
    # Check if requirements-fixed.txt exists
    if os.path.exists("requirements-fixed.txt"):
        print("Using requirements-fixed.txt...")
        result = run_command("pip install -r requirements-fixed.txt")
    else:
        print("Using requirements.txt...")
        result = run_command("pip install -r requirements.txt")
    
    if result:
        print("✅ Requirements installed successfully!")
    else:
        print("❌ Error installing requirements")

if __name__ == "__main__":
    print("🚀 Systemic Risk Platform - SciPy Fix Script")
    print("=" * 50)
    
    # Fix SciPy issue
    if fix_scipy_issue():
        # Install remaining requirements
        install_requirements()
        print("\n🎉 Setup completed successfully!")
        print("You can now run: streamlit run app.py")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1) 