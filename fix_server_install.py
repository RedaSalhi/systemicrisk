#!/usr/bin/env python3
"""
Server Installation Fix Script
Resolves Fortran compilation issues on servers by using pre-compiled wheels
"""

import subprocess
import sys
import os
import platform

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def check_python_version():
    """Check Python version and provide recommendations"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 13:
        print("✅ Python 3.13+ detected - using latest compatible versions")
        return True
    else:
        print("⚠️  Consider upgrading to Python 3.13+ for better compatibility")
        return False

def install_server_requirements():
    """Install server-compatible requirements"""
    print("🔧 Installing server-compatible requirements...")
    
    # Step 1: Upgrade pip first
    print("1. Upgrading pip...")
    run_command("pip install --upgrade pip")
    
    # Step 2: Install numpy first (dependency for others)
    print("2. Installing numpy...")
    run_command("pip install numpy==2.3.1")
    
    # Step 3: Install scipy with pre-compiled wheel
    print("3. Installing scipy (pre-compiled wheel)...")
    result = run_command("pip install scipy==1.13.0")
    if not result:
        print("❌ Failed to install scipy. Trying alternative version...")
        run_command("pip install scipy==1.12.0")
    
    # Step 4: Install other packages
    print("4. Installing other packages...")
    packages = [
        "pandas==2.3.0",
        "scikit-learn==1.4.0",
        "statsmodels==0.14.1",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "plotly==6.2.0",
        "yfinance==0.2.28",
        "xgboost==2.0.3",
        "streamlit==1.46.1"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        run_command(f"pip install {package}")
    
    return True

def verify_installation():
    """Verify that all packages are installed correctly"""
    print("🔍 Verifying installation...")
    
    try:
        import scipy
        import numpy
        import pandas
        import sklearn
        import statsmodels
        import streamlit
        import plotly
        import yfinance
        import xgboost
        
        print("✅ All packages installed successfully!")
        print(f"   SciPy version: {scipy.__version__}")
        print(f"   NumPy version: {numpy.__version__}")
        print(f"   Pandas version: {pandas.__version__}")
        print(f"   Scikit-learn version: {sklearn.__version__}")
        print(f"   Statsmodels version: {statsmodels.__version__}")
        print(f"   Streamlit version: {streamlit.__version__}")
        print(f"   Plotly version: {plotly.__version__}")
        print(f"   YFinance version: {yfinance.__version__}")
        print(f"   XGBoost version: {xgboost.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_alternative_requirements():
    """Create alternative requirements file if needed"""
    print("📝 Creating alternative requirements file...")
    
    alt_requirements = """# Alternative requirements for servers with compilation issues
# Use these if the main requirements fail

# Core packages with minimal dependencies
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
xgboost>=1.7.0

# Use conda-forge versions if available
# conda install -c conda-forge scipy scikit-learn statsmodels

# Or use system package manager
# apt-get install python3-scipy python3-sklearn python3-statsmodels
"""
    
    with open("requirements-alternative.txt", "w") as f:
        f.write(alt_requirements)
    
    print("✅ Created requirements-alternative.txt")

def main():
    """Main installation process"""
    print("🚀 Systemic Risk Platform - Server Installation Fix")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Check if we're on a server environment
    if os.path.exists("/etc/debian_version") or os.path.exists("/etc/redhat-release"):
        print("🖥️  Server environment detected")
    else:
        print("💻 Local environment detected")
    
    # Install requirements
    if install_server_requirements():
        # Verify installation
        if verify_installation():
            print("\n🎉 Installation completed successfully!")
            print("You can now run: streamlit run app.py")
            
            # Test import
            print("\n🧪 Testing imports...")
            try:
                from data_processor import BankingDataProcessor
                print("✅ Data processor imports successfully!")
            except Exception as e:
                print(f"⚠️  Data processor import issue: {e}")
                print("This is normal if data_processor.py is not in the current directory")
        else:
            print("\n❌ Installation verification failed")
            create_alternative_requirements()
            sys.exit(1)
    else:
        print("\n❌ Installation failed")
        create_alternative_requirements()
        sys.exit(1)

if __name__ == "__main__":
    main() 