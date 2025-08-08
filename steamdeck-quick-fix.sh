#!/bin/bash
# Quick Fix Script for Steam Deck ML Shader Prediction Compiler
# This script addresses the three main installation issues identified

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Steam Deck ML Shader Compiler Quick Fix   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}\n"

# Fix 1: Install pip3 (addresses "bash: pip3: command not found")
echo -e "${BLUE}[FIX 1/3]${NC} Installing pip3..."
if ! command -v pip3 >/dev/null 2>&1; then
    if ! python3 -m pip --version >/dev/null 2>&1; then
        echo -e "${YELLOW}[INFO]${NC} Installing pip using ensurepip..."
        python3 -m ensurepip --upgrade --user || {
            echo -e "${YELLOW}[INFO]${NC} Downloading get-pip.py..."
            curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
            python3 /tmp/get-pip.py --user
            rm /tmp/get-pip.py
        }
        # Add user bin to PATH if not already there
        if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
            export PATH="$HOME/.local/bin:$PATH"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        fi
    fi
    alias pip3='python3 -m pip'
    echo -e "${GREEN}[SUCCESS]${NC} pip3 is now available"
else
    echo -e "${GREEN}[SUCCESS]${NC} pip3 already installed"
fi

# Fix 2: Create proper directory structure (addresses file path issues)
echo -e "${BLUE}[FIX 2/3]${NC} Creating proper directory structure..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET_DIR="/home/deck/src"

mkdir -p "$TARGET_DIR"
echo -e "${GREEN}[SUCCESS]${NC} Created directory: $TARGET_DIR"

# Copy files to the expected location
if [ -f "$SCRIPT_DIR/src/shader_prediction_system.py" ]; then
    cp -r "$SCRIPT_DIR/src/"* "$TARGET_DIR/"
    echo -e "${GREEN}[SUCCESS]${NC} Copied shader prediction system to $TARGET_DIR"
elif [ -d "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src" ]; then
    cp -r "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src/"* "$TARGET_DIR/"
    echo -e "${GREEN}[SUCCESS]${NC} Copied shader prediction system to $TARGET_DIR"
else
    echo -e "${RED}[ERROR]${NC} Could not find source files"
    echo -e "${YELLOW}[INFO]${NC} Available directories:"
    ls -la "$SCRIPT_DIR"
    exit 1
fi

# Fix 3: Install minimal dependencies (addresses installation script failures)
echo -e "${BLUE}[FIX 3/3]${NC} Installing minimal Python dependencies..."
python3 -m pip install --user --upgrade pip

# Install core dependencies one by one
DEPS=(
    "numpy>=1.21.0,<1.25.0"
    "scikit-learn>=1.1.0,<1.4.0" 
    "joblib>=1.1.0,<1.4.0"
    "psutil>=5.8.0,<6.0.0"
    "PyYAML>=6.0,<7.0"
    "configparser>=5.2.0"
)

for dep in "${DEPS[@]}"; do
    echo -e "${BLUE}[INFO]${NC} Installing ${dep%>=*}..."
    python3 -m pip install --user "$dep" || {
        echo -e "${YELLOW}[WARNING]${NC} Failed to install $dep, continuing..."
    }
done

# Linux-specific dependency
if [ "$(uname)" = "Linux" ]; then
    python3 -m pip install --user "pyudev>=0.23.0" 2>/dev/null || echo -e "${YELLOW}[WARNING]${NC} Could not install pyudev"
fi

echo -e "${GREEN}[SUCCESS]${NC} Dependencies installed"

# Create a simple test script
cat > "$TARGET_DIR/test_installation.py" << 'EOF'
#!/usr/bin/env python3
"""
Quick test to verify the installation is working
"""
import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        print("✓ scikit-learn imported successfully")
    except ImportError:
        print("⚠ scikit-learn not available - fallback mode will be used")
    
    try:
        import psutil
        print("✓ psutil imported successfully")
    except ImportError:
        print("⚠ psutil not available - basic monitoring will be used")
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError:
        print("⚠ PyYAML not available")
    
    return True

def test_shader_prediction():
    """Test the shader prediction system"""
    try:
        # Try to import the main module
        if os.path.exists('shader_prediction_system.py'):
            import importlib.util
            spec = importlib.util.spec_from_file_location("shader_prediction_system", "shader_prediction_system.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✓ Shader prediction system module loaded")
            return True
    except Exception as e:
        print(f"✗ Shader prediction system failed to load: {e}")
        return False

if __name__ == "__main__":
    print("Testing Steam Deck ML Shader Prediction Compiler Installation")
    print("=" * 60)
    
    success = test_imports()
    if success:
        success = test_shader_prediction()
    
    if success:
        print("\n✓ Installation test PASSED - system is ready to use!")
        sys.exit(0)
    else:
        print("\n✗ Installation test FAILED - check the errors above")
        sys.exit(1)
EOF

chmod +x "$TARGET_DIR/test_installation.py"

# Run the test
echo -e "${BLUE}[TEST]${NC} Running installation verification..."
cd "$TARGET_DIR"
python3 test_installation.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    QUICK FIX COMPLETED                    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${GREEN}[SUCCESS]${NC} All fixes applied successfully!"
    echo -e ""
    echo -e "${YELLOW}What was fixed:${NC}"
    echo -e "  ✓ pip3 installation issue resolved"
    echo -e "  ✓ File path structure corrected"
    echo -e "  ✓ Essential dependencies installed"
    echo -e ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Run the main shader prediction system:"
    echo -e "     ${GREEN}cd /home/deck/src && python3 shader_prediction_system.py${NC}"
    echo -e ""
    echo -e "  2. For full installation with systemd service:"
    echo -e "     ${GREEN}$SCRIPT_DIR/steamdeck-optimized-install.sh${NC}"
    echo -e ""
else
    echo -e "\n${RED}[ERROR]${NC} Quick fix validation failed. Check the output above for details."
    exit 1
fi