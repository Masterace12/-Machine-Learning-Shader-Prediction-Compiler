#!/bin/bash
# Quick Fix Script for Steam Deck ML Shader Prediction Compiler
# This script addresses the three main installation issues identified
# ENHANCED VERSION - Now with immutable filesystem support and proper error handling

set -euo pipefail  # Stricter error handling
IFS=$'\n\t'        # Secure internal field separator

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Steam Deck ML Shader Compiler Quick Fix   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}\n"

# Steam Deck filesystem detection
echo -e "${BLUE}[INFO]${NC} Detecting Steam Deck filesystem type..."
IMMUTABLE_FS=false
if [[ ! -w "/usr" ]]; then
    IMMUTABLE_FS=true
    echo -e "${YELLOW}[INFO]${NC} Detected immutable filesystem (standard SteamOS)"
else
    echo -e "${BLUE}[INFO]${NC} Detected mutable filesystem (developer mode)"
fi

# Fix 1: Install pip3 with Steam Deck compatibility (addresses "bash: pip3: command not found")
echo -e "${BLUE}[FIX 1/3]${NC} Installing pip3 with Steam Deck compatibility..."
if ! command -v pip3 >/dev/null 2>&1; then
    if ! python3 -m pip --version >/dev/null 2>&1; then
        echo -e "${YELLOW}[INFO]${NC} Installing pip using ensurepip..."
        python3 -m ensurepip --upgrade --user 2>/dev/null || {
            echo -e "${YELLOW}[INFO]${NC} Downloading get-pip.py with timeout protection..."
            if command -v curl >/dev/null 2>&1; then
                timeout 60 curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py || {
                    echo -e "${RED}[ERROR]${NC} Failed to download get-pip.py"
                    exit 1
                }
            elif command -v wget >/dev/null 2>&1; then
                timeout 60 wget -q -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py || {
                    echo -e "${RED}[ERROR]${NC} Failed to download get-pip.py"
                    exit 1
                }
            else
                echo -e "${RED}[ERROR]${NC} Neither curl nor wget available"
                exit 1
            fi
            
            python3 /tmp/get-pip.py --user
            rm -f /tmp/get-pip.py
        }
        
        # Add user bin to PATH if not already there (Steam Deck safe)
        USER_BIN="$HOME/.local/bin"
        if [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
            export PATH="$USER_BIN:$PATH"
            
            # Add to multiple shell configs for persistence
            for shell_config in "$HOME/.bashrc" "$HOME/.profile"; do
                if [[ -f "$shell_config" ]] && ! grep -q "export PATH.*$USER_BIN" "$shell_config" 2>/dev/null; then
                    echo "export PATH=\"$USER_BIN:\$PATH\"" >> "$shell_config"
                fi
            done
            echo -e "${BLUE}[INFO]${NC} Added $USER_BIN to PATH"
        fi
    fi
    
    # Create pip3 alias for consistency
    if ! command -v pip3 >/dev/null 2>&1; then
        alias pip3='python3 -m pip'
        echo "alias pip3='python3 -m pip'" >> "$HOME/.bashrc"
    fi
    
    echo -e "${GREEN}[SUCCESS]${NC} pip3 is now available"
else
    echo -e "${GREEN}[SUCCESS]${NC} pip3 already installed"
fi

# Fix 2: Create proper directory structure with Steam Deck compatibility (addresses file path issues)
echo -e "${BLUE}[FIX 2/3]${NC} Creating proper directory structure for Steam Deck..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use Steam Deck optimized paths that work with immutable filesystem
if [[ "$IMMUTABLE_FS" == "true" ]]; then
    # Use user-space directories for immutable filesystem
    TARGET_DIR="$HOME/.local/share/ml-shader-predictor/src"
    echo -e "${BLUE}[INFO]${NC} Using immutable-filesystem-compatible path: $TARGET_DIR"
else
    # Original path for developer mode
    TARGET_DIR="/home/deck/src"
    echo -e "${BLUE}[INFO]${NC} Using developer mode path: $TARGET_DIR"
fi

# Create directory with proper permissions
mkdir -p "$TARGET_DIR" || {
    echo -e "${RED}[ERROR]${NC} Failed to create target directory: $TARGET_DIR"
    exit 1
}

# Set proper permissions
chmod 755 "$TARGET_DIR" 2>/dev/null || true
echo -e "${GREEN}[SUCCESS]${NC} Created directory: $TARGET_DIR"

# Search for source files in multiple possible locations
SOURCE_FOUND=false
SOURCE_PATHS=(
    "$SCRIPT_DIR/src"
    "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src"
    "$SCRIPT_DIR/../src"
    "$(dirname "$SCRIPT_DIR")/src"
)

echo -e "${BLUE}[INFO]${NC} Searching for source files..."
for src_path in "${SOURCE_PATHS[@]}"; do
    echo -e "${BLUE}[INFO]${NC} Checking: $src_path"
    if [[ -d "$src_path" ]] && [[ -n "$(ls -A "$src_path" 2>/dev/null)" ]]; then
        echo -e "${BLUE}[INFO]${NC} Found source files at: $src_path"
        
        # Copy files with error handling
        if cp -r "$src_path"/* "$TARGET_DIR/" 2>/dev/null; then
            echo -e "${GREEN}[SUCCESS]${NC} Copied shader prediction system to $TARGET_DIR"
            SOURCE_FOUND=true
            break
        else
            echo -e "${YELLOW}[WARNING]${NC} Failed to copy some files from $src_path"
            # Try individual file copy
            for file in "$src_path"/*; do
                if [[ -f "$file" ]]; then
                    cp "$file" "$TARGET_DIR/" 2>/dev/null && echo -e "${BLUE}[INFO]${NC} Copied: $(basename "$file")"
                fi
            done
            SOURCE_FOUND=true
            break
        fi
    fi
done

if [[ "$SOURCE_FOUND" == "false" ]]; then
    echo -e "${RED}[ERROR]${NC} Could not find source files in any expected location"
    echo -e "${YELLOW}[INFO]${NC} Searched locations:"
    for src_path in "${SOURCE_PATHS[@]}"; do
        echo "  - $src_path $([ -d "$src_path" ] && echo "[EXISTS]" || echo "[NOT FOUND]")"
    done
    echo -e "${YELLOW}[INFO]${NC} Current directory contents:"
    ls -la "$SCRIPT_DIR" 2>/dev/null || echo "Cannot list directory contents"
    exit 1
fi

# Create additional necessary directories for Steam Deck
ADDITIONAL_DIRS=(
    "$HOME/.config/ml-shader-predictor"
    "$HOME/.cache/ml-shader-predictor" 
    "$HOME/.local/share/ml-shader-predictor/logs"
    "$HOME/.local/share/ml-shader-predictor/config"
)

echo -e "${BLUE}[INFO]${NC} Creating additional Steam Deck directories..."
for dir in "${ADDITIONAL_DIRS[@]}"; do
    if mkdir -p "$dir" 2>/dev/null; then
        chmod 755 "$dir" 2>/dev/null || true
        echo -e "${GREEN}[SUCCESS]${NC} Created: $dir"
    else
        echo -e "${YELLOW}[WARNING]${NC} Could not create: $dir"
    fi
done

# Fix 3: Install minimal dependencies with Steam Deck optimization (addresses installation script failures)
echo -e "${BLUE}[FIX 3/3]${NC} Installing minimal Python dependencies for Steam Deck..."

# Set memory optimization for pip on Steam Deck
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Check available memory and adjust installation strategy
AVAILABLE_MEMORY=0
if command -v free >/dev/null 2>&1; then
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
    echo -e "${BLUE}[INFO]${NC} Available memory: ${AVAILABLE_MEMORY}MB"
fi

# Upgrade pip with timeout and error handling
echo -e "${BLUE}[INFO]${NC} Upgrading pip..."
timeout 120 python3 -m pip install --user --upgrade pip setuptools wheel --quiet || {
    echo -e "${YELLOW}[WARNING]${NC} Pip upgrade failed or timed out, continuing with current version"
}

# Install core dependencies one by one with enhanced error handling
DEPS=(
    "numpy>=1.21.0,<1.25.0"
    "psutil>=5.8.0,<6.0.0"
    "PyYAML>=6.0,<7.0"
    "joblib>=1.1.0,<1.4.0"
    "requests>=2.28.0,<3.0.0"
)

# Optional dependencies (install if memory allows)
OPTIONAL_DEPS=(
    "scikit-learn>=1.1.0,<1.4.0"
    "configparser>=5.2.0"
)

echo -e "${BLUE}[INFO]${NC} Installing core dependencies..."
SUCCESSFUL_INSTALLS=0
FAILED_INSTALLS=()

for dep in "${DEPS[@]}"; do
    package_name="${dep%>=*}"
    echo -e "${BLUE}[INFO]${NC} Installing $package_name..."
    
    # Install with timeout and memory monitoring
    if timeout 300 python3 -m pip install --user --quiet --no-warn-script-location "$dep" 2>/dev/null; then
        echo -e "${GREEN}[SUCCESS]${NC} Installed: $package_name"
        ((SUCCESSFUL_INSTALLS++))
    else
        echo -e "${YELLOW}[WARNING]${NC} Failed to install $package_name, trying fallback version..."
        FAILED_INSTALLS+=("$package_name")
        
        # Try fallback versions for critical packages
        case "$package_name" in
            "numpy")
                timeout 180 python3 -m pip install --user --quiet "numpy>=1.19.0,<1.24.0" 2>/dev/null && {
                    echo -e "${GREEN}[SUCCESS]${NC} Installed fallback numpy"
                    ((SUCCESSFUL_INSTALLS++))
                }
                ;;
            "psutil"|"PyYAML"|"joblib"|"requests")
                # For other critical packages, try without version constraints
                timeout 180 python3 -m pip install --user --quiet "$package_name" 2>/dev/null && {
                    echo -e "${GREEN}[SUCCESS]${NC} Installed fallback $package_name"
                    ((SUCCESSFUL_INSTALLS++))
                }
                ;;
        esac
    fi
    
    # Small delay to prevent overloading system
    sleep 2
done

# Install optional dependencies if resources allow
if [[ "$AVAILABLE_MEMORY" -gt 2000 ]] && [[ ${#FAILED_INSTALLS[@]} -lt 2 ]]; then
    echo -e "${BLUE}[INFO]${NC} Installing optional dependencies..."
    for dep in "${OPTIONAL_DEPS[@]}"; do
        package_name="${dep%>=*}"
        echo -e "${BLUE}[INFO]${NC} Installing optional $package_name..."
        
        timeout 300 python3 -m pip install --user --quiet --no-warn-script-location "$dep" 2>/dev/null && {
            echo -e "${GREEN}[SUCCESS]${NC} Installed optional: $package_name"
        } || {
            echo -e "${YELLOW}[INFO]${NC} Skipped optional $package_name (not critical)"
        }
        sleep 1
    done
else
    echo -e "${YELLOW}[INFO]${NC} Skipping optional dependencies due to resource constraints"
fi

# Linux-specific dependency (Steam Deck)
if [[ "$(uname)" == "Linux" ]]; then
    echo -e "${BLUE}[INFO]${NC} Installing Linux-specific dependencies..."
    timeout 120 python3 -m pip install --user --quiet "pyudev>=0.23.0" 2>/dev/null && {
        echo -e "${GREEN}[SUCCESS]${NC} Installed pyudev"
    } || {
        echo -e "${YELLOW}[WARNING]${NC} Could not install pyudev (hardware monitoring will be limited)"
    }
fi

# Summary
if [[ "$SUCCESSFUL_INSTALLS" -ge 3 ]]; then
    echo -e "${GREEN}[SUCCESS]${NC} Dependencies installed: $SUCCESSFUL_INSTALLS packages"
    if [[ ${#FAILED_INSTALLS[@]} -gt 0 ]]; then
        echo -e "${YELLOW}[INFO]${NC} Some packages failed but core functionality should work"
        echo -e "${YELLOW}[INFO]${NC} Failed packages: ${FAILED_INSTALLS[*]}"
    fi
else
    echo -e "${RED}[ERROR]${NC} Too many dependency failures ($SUCCESSFUL_INSTALLS successful)"
    echo -e "${YELLOW}[INFO]${NC} The system may have limited functionality"
fi

# Clear pip cache to save space
python3 -m pip cache purge 2>/dev/null || true

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

# Run the test with proper path handling
echo -e "${BLUE}[TEST]${NC} Running installation verification..."
cd "$TARGET_DIR" || {
    echo -e "${RED}[ERROR]${NC} Cannot access target directory: $TARGET_DIR"
    exit 1
}

# Run test with timeout and error handling
if timeout 60 python3 test_installation.py 2>/dev/null; then
    TEST_RESULT=0
else
    TEST_RESULT=$?
    echo -e "${YELLOW}[WARNING]${NC} Test had issues but installation may still work"
fi

if [[ $TEST_RESULT -eq 0 ]]; then
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