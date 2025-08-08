#!/bin/bash
# Fix script for ModuleNotFoundError: No module named 'numpy'
# Steam Deck ML-Based Shader Prediction Compiler

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

echo -e "${BLUE}Steam Deck Shader Prediction Compiler - NumPy Fix${NC}"
echo "=========================================================="
echo

log_info "Diagnosing NumPy installation issue..."

# Check Python availability
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    log_success "Python found: $PYTHON_VERSION"
else
    log_error "Python 3 not found! Please install Python 3.8+ first."
    exit 1
fi

# Check pip availability
if python3 -m pip --version >/dev/null 2>&1; then
    PIP_VERSION=$(python3 -m pip --version 2>&1)
    log_success "pip found: $PIP_VERSION"
else
    log_error "pip not found! Installing pip..."
    # Try to install pip
    if command -v apt >/dev/null 2>&1; then
        sudo apt update && sudo apt install -y python3-pip
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm python-pip
    else
        log_error "Could not install pip automatically. Please install python3-pip manually."
        exit 1
    fi
fi

# Check current numpy installation
log_info "Checking current NumPy installation..."
if python3 -c "import numpy; print(f'NumPy {numpy.__version__} is installed')" 2>/dev/null; then
    log_success "NumPy is already installed!"
    python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
    python3 -c "import numpy; print(f'NumPy location: {numpy.__file__}')"
    
    log_info "Testing shader prediction system import..."
    cd "$(dirname "$0")"
    if python3 -c "from src.shader_prediction_system import SteamDeckShaderPredictor; print('✓ Import successful')" 2>/dev/null; then
        log_success "System import test passed! The issue appears to be resolved."
        exit 0
    else
        log_warning "System import still failing. Continuing with full dependency installation..."
    fi
else
    log_warning "NumPy is not installed or not accessible"
fi

# Install NumPy and core dependencies
log_info "Installing NumPy and core dependencies..."

# Installation strategies in order of preference
INSTALL_METHODS=(
    "python3 -m pip install --user numpy>=1.19.0 --prefer-binary"
    "python3 -m pip install --user numpy>=1.19.0"
    "python3 -m pip install numpy>=1.19.0 --prefer-binary"
    "python3 -m pip install numpy>=1.19.0"
)

NUMPY_INSTALLED=false
for method in "${INSTALL_METHODS[@]}"; do
    log_info "Trying: $method"
    if eval "$method" >/dev/null 2>&1; then
        NUMPY_INSTALLED=true
        log_success "NumPy installation successful!"
        break
    else
        log_warning "Method failed, trying next..."
    fi
done

if [[ "$NUMPY_INSTALLED" != "true" ]]; then
    log_error "All NumPy installation methods failed!"
    log_info "Manual steps to try:"
    log_info "1. sudo apt update && sudo apt install python3-numpy  # On Debian/Ubuntu"
    log_info "2. sudo pacman -S python-numpy  # On Arch/SteamOS"
    log_info "3. pip3 install --user numpy  # Direct pip install"
    exit 1
fi

# Verify NumPy installation
log_info "Verifying NumPy installation..."
if python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__} successfully installed')" 2>/dev/null; then
    log_success "NumPy verification passed!"
else
    log_error "NumPy verification failed!"
    exit 1
fi

# Install other core dependencies
log_info "Installing other core dependencies..."
CORE_DEPS=(
    "scikit-learn>=1.0.0"
    "psutil>=5.7.0"
    "requests>=2.25.0"
    "PyYAML>=5.4.0"
)

FAILED_DEPS=()
for dep in "${CORE_DEPS[@]}"; do
    log_info "Installing $dep..."
    if python3 -m pip install --user "$dep" --prefer-binary >/dev/null 2>&1; then
        log_success "$dep installed"
    else
        log_warning "$dep installation failed"
        FAILED_DEPS+=("$dep")
    fi
done

if [[ ${#FAILED_DEPS[@]} -gt 0 ]]; then
    log_warning "Some dependencies failed to install: ${FAILED_DEPS[*]}"
    log_info "The core system should still work with NumPy installed."
fi

# Final system test
log_info "Running final system test..."
cd "$(dirname "$0")"

if [[ -f "src/shader_prediction_system.py" ]]; then
    if python3 -c "
import sys
sys.path.insert(0, 'src')
from shader_prediction_system import SteamDeckShaderPredictor
print('✓ Main system import successful')
system = SteamDeckShaderPredictor()
print('✓ System initialization successful')
print('✓ All tests passed!')
" 2>/dev/null; then
        log_success "Final system test PASSED!"
        echo
        echo -e "${GREEN}🎉 SUCCESS! The NumPy issue has been resolved! 🎉${NC}"
        echo
        log_info "You can now run the shader prediction system:"
        log_info "  cd $(pwd)"
        log_info "  python3 src/shader_prediction_system.py"
        echo
    else
        log_warning "System import test failed, but NumPy is installed."
        log_info "Try running the enhanced installer for complete setup:"
        log_info "  chmod +x enhanced-install.sh && ./enhanced-install.sh"
    fi
else
    log_warning "shader_prediction_system.py not found in src/ directory"
    log_info "Make sure you're running this script from the project root directory"
fi

echo
log_info "Fix script completed. Check the output above for any remaining issues."