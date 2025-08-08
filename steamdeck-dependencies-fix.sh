#!/bin/bash
#
# Steam Deck Dependencies Fix Script
# Handles pip3 installation issues and immutable filesystem constraints
#
# Usage: bash steamdeck-dependencies-fix.sh
#

set -euo pipefail

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1" >&2; }

# ============================================================================
# STEAM DECK SPECIFIC DEPENDENCY HANDLING
# ============================================================================

detect_environment() {
    log_info "Detecting Steam Deck environment..."
    
    # Check if we're on SteamOS
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        if [[ "${ID:-}" == "steamos" ]]; then
            log_success "SteamOS detected"
            IS_STEAMOS=true
        else
            log_info "Non-SteamOS Linux detected: ${NAME:-unknown}"
            IS_STEAMOS=false
        fi
    fi
    
    # Check filesystem mutability
    if [[ -w /usr ]] && [[ -w /opt ]]; then
        IS_IMMUTABLE=false
        log_info "Mutable filesystem detected"
    else
        IS_IMMUTABLE=true
        log_info "Immutable filesystem detected"
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df "$HOME" | awk 'NR==2 {print $4}')
    if [[ $AVAILABLE_SPACE -lt 1000000 ]]; then  # Less than 1GB
        log_warning "Low disk space: $(($AVAILABLE_SPACE / 1024))MB available"
    fi
}

setup_pip_environment() {
    log_info "Setting up pip environment for Steam Deck..."
    
    # Ensure user directories exist
    mkdir -p "$HOME/.local/bin"
    mkdir -p "$HOME/.local/lib"
    
    # Add to PATH if not already there
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        export PATH="$HOME/.local/bin:$PATH"
        
        # Add to bashrc for persistence
        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        fi
        
        log_info "Added ~/.local/bin to PATH"
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $PYTHON_VERSION"
    
    # Upgrade pip, setuptools, wheel in user space
    log_info "Upgrading pip in user space..."
    python3 -m pip install --user --upgrade pip setuptools wheel --quiet --no-warn-script-location
    
    # Verify pip installation
    if python3 -m pip --version >/dev/null 2>&1; then
        log_success "pip is working correctly"
    else
        log_error "pip installation failed"
        return 1
    fi
}

install_core_dependencies() {
    log_info "Installing core Python dependencies..."
    
    # Core packages that must succeed
    local core_packages=(
        "psutil>=5.8.0,<6.0.0"
        "requests>=2.28.0,<3.0.0" 
        "PyYAML>=6.0,<7.0"
    )
    
    local failed_core=()
    for package in "${core_packages[@]}"; do
        log_info "Installing core package: $package"
        if ! python3 -m pip install --user --quiet --no-warn-script-location "$package"; then
            log_error "Failed to install critical package: $package"
            failed_core+=("$package")
        else
            log_success "Installed: $package"
        fi
    done
    
    if [[ ${#failed_core[@]} -ne 0 ]]; then
        log_error "Critical packages failed: ${failed_core[*]}"
        return 1
    fi
    
    log_success "Core dependencies installed"
}

install_ml_dependencies() {
    log_info "Installing ML dependencies (with fallbacks)..."
    
    # ML packages with specific Steam Deck tested versions
    local ml_packages=(
        "numpy>=1.21.0,<1.26.0"
        "scipy>=1.8.0,<1.12.0"
        "scikit-learn>=1.1.0,<1.4.0"
        "joblib>=1.1.0,<1.4.0"
        "pandas>=1.4.0,<2.1.0"
    )
    
    local installed_ml=()
    local failed_ml=()
    
    for package in "${ml_packages[@]}"; do
        log_info "Installing ML package: $package"
        
        # Try with timeout to prevent hanging
        if timeout 300 python3 -m pip install --user --quiet --no-warn-script-location "$package"; then
            log_success "Installed: $package"
            installed_ml+=("$package")
        else
            log_warning "Failed to install ML package: $package"
            failed_ml+=("$package")
            
            # Try installing a simpler/older version as fallback
            case "$package" in
                numpy*)
                    log_info "Trying fallback numpy version..."
                    if python3 -m pip install --user --quiet --no-warn-script-location "numpy>=1.19.0,<1.24.0"; then
                        log_success "Installed fallback numpy"
                        installed_ml+=("numpy")
                    fi
                    ;;
                scikit-learn*)
                    log_info "Trying fallback scikit-learn version..."
                    if python3 -m pip install --user --quiet --no-warn-script-location "scikit-learn>=1.0.0,<1.3.0"; then
                        log_success "Installed fallback scikit-learn"
                        installed_ml+=("scikit-learn")
                    fi
                    ;;
            esac
        fi
    done
    
    log_info "ML dependencies result: ${#installed_ml[@]} installed, ${#failed_ml[@]} failed"
    
    if [[ ${#installed_ml[@]} -ge 2 ]]; then
        log_success "Sufficient ML dependencies installed"
        return 0
    else
        log_warning "Limited ML functionality due to missing dependencies"
        return 1
    fi
}

install_networking_dependencies() {
    log_info "Installing networking dependencies..."
    
    local net_packages=(
        "aiohttp>=3.8.0,<4.0.0"
        "cryptography>=40.0.0,<42.0.0"
    )
    
    for package in "${net_packages[@]}"; do
        log_info "Installing networking package: $package"
        if ! python3 -m pip install --user --quiet --no-warn-script-location "$package"; then
            log_warning "Networking package failed: $package"
        else
            log_success "Installed: $package"
        fi
    done
    
    log_success "Networking dependencies processed"
}

install_fallback_packages() {
    log_info "Installing fallback packages for missing dependencies..."
    
    # Check what's missing and install minimal alternatives
    local missing_packages=()
    
    # Test imports
    if ! python3 -c "import numpy" 2>/dev/null; then
        log_warning "numpy missing - installing minimal version"
        python3 -m pip install --user --quiet --no-warn-script-location "numpy>=1.19.0" || true
    fi
    
    if ! python3 -c "import sklearn" 2>/dev/null; then
        log_warning "scikit-learn missing - ML features will be limited"
        # Create a dummy sklearn module for graceful degradation
        mkdir -p "$HOME/.local/lib/python${PYTHON_VERSION}/site-packages/sklearn_fallback"
        cat > "$HOME/.local/lib/python${PYTHON_VERSION}/site-packages/sklearn_fallback/__init__.py" << 'EOF'
# Fallback sklearn module for graceful degradation
class DummyClassifier:
    def fit(self, X, y): return self
    def predict(self, X): return [0] * len(X)
    def predict_proba(self, X): return [[0.5, 0.5]] * len(X)

class DummyRegressor:
    def fit(self, X, y): return self
    def predict(self, X): return [0.0] * len(X)

# Minimal sklearn-like interface
ensemble = type('ensemble', (), {
    'RandomForestClassifier': DummyClassifier,
    'RandomForestRegressor': DummyRegressor
})()

linear_model = type('linear_model', (), {
    'LinearRegression': DummyRegressor,
    'LogisticRegression': DummyClassifier
})()

model_selection = type('model_selection', (), {
    'train_test_split': lambda *args, **kwargs: args[:2]
})()

preprocessing = type('preprocessing', (), {
    'StandardScaler': lambda: type('StandardScaler', (), {
        'fit': lambda self, X: self,
        'transform': lambda self, X: X,
        'fit_transform': lambda self, X: X
    })()
})()

metrics = type('metrics', (), {
    'accuracy_score': lambda y_true, y_pred: 0.5,
    'mean_absolute_error': lambda y_true, y_pred: 1.0
})()
EOF
    fi
    
    log_info "Fallback packages configured"
}

create_requirements_steamdeck() {
    log_info "Creating Steam Deck specific requirements file..."
    
    cat > "$HOME/.cache/steamdeck-requirements.txt" << 'EOF'
# Steam Deck Optimized Python Requirements
# Version-locked for compatibility and stability

# Core system packages (must install)
psutil>=5.8.0,<6.0.0
requests>=2.28.0,<3.0.0
PyYAML>=6.0,<7.0
toml>=0.10.2,<1.0.0

# ML packages (with fallbacks)
numpy>=1.21.0,<1.26.0
scipy>=1.8.0,<1.12.0; python_version >= "3.8"
scikit-learn>=1.1.0,<1.4.0; python_version >= "3.8"
joblib>=1.1.0,<1.4.0
pandas>=1.4.0,<2.1.0; python_version >= "3.8"

# Networking (optional)
aiohttp>=3.8.0,<4.0.0
cryptography>=40.0.0,<42.0.0

# GPU support (optional - will fail gracefully)
torch>=2.0.0,<2.2.0; python_version >= "3.8" and platform_system == "Linux"

# Steam Deck specific
py-cpuinfo>=9.0.0,<10.0.0
EOF
    
    log_success "Requirements file created at ~/.cache/steamdeck-requirements.txt"
}

verify_installation() {
    log_info "Verifying dependency installation..."
    
    local test_results=()
    
    # Test core imports
    if python3 -c "import psutil, requests, yaml" 2>/dev/null; then
        log_success "Core dependencies: OK"
        test_results+=("core:OK")
    else
        log_error "Core dependencies: FAILED"
        test_results+=("core:FAILED")
    fi
    
    # Test ML imports
    if python3 -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>/dev/null; then
        log_success "NumPy: OK"
        test_results+=("numpy:OK")
    else
        log_warning "NumPy: MISSING"
        test_results+=("numpy:MISSING")
    fi
    
    if python3 -c "import sklearn; print(f'Scikit-learn {sklearn.__version__}')" 2>/dev/null; then
        log_success "Scikit-learn: OK" 
        test_results+=("sklearn:OK")
    else
        log_warning "Scikit-learn: MISSING (fallback available)"
        test_results+=("sklearn:FALLBACK")
    fi
    
    # Test networking imports
    if python3 -c "import aiohttp" 2>/dev/null; then
        log_success "Networking: OK"
        test_results+=("networking:OK")
    else
        log_warning "Networking: LIMITED"
        test_results+=("networking:LIMITED")
    fi
    
    # Create summary
    echo
    log_info "Installation Summary:"
    for result in "${test_results[@]}"; do
        echo "  - $result"
    done
    
    # Overall assessment
    if [[ " ${test_results[*]} " == *"core:FAILED"* ]]; then
        log_error "Installation FAILED - core dependencies missing"
        return 1
    elif [[ " ${test_results[*]} " == *"numpy:MISSING"* ]] && [[ " ${test_results[*]} " == *"sklearn:FALLBACK"* ]]; then
        log_warning "Installation PARTIAL - limited ML functionality"
        return 2
    else
        log_success "Installation SUCCESSFUL"
        return 0
    fi
}

create_dependency_test() {
    log_info "Creating dependency test script..."
    
    cat > "$HOME/.local/bin/test-shader-deps" << 'EOF'
#!/usr/bin/env python3
"""
Steam Deck Shader Prediction Compiler - Dependency Test
Tests all required and optional dependencies
"""

import sys
import subprocess
import importlib

def test_import(module_name, required=True):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {module_name}: {version}")
        return True
    except ImportError:
        status = "REQUIRED" if required else "optional"
        print(f"✗ {module_name}: missing ({status})")
        return not required

def test_system_resources():
    print("\n=== System Resources ===")
    try:
        import psutil
        print(f"CPU count: {psutil.cpu_count()}")
        print(f"Memory: {psutil.virtual_memory().total // (1024**3)}GB")
        print(f"Available: {psutil.virtual_memory().available // (1024**3)}GB")
        
        # Check thermal zones
        import glob
        thermal_zones = glob.glob('/sys/class/thermal/thermal_zone*/temp')
        if thermal_zones:
            with open(thermal_zones[0]) as f:
                temp = int(f.read().strip()) / 1000
                print(f"CPU Temperature: {temp}°C")
    except Exception as e:
        print(f"Resource check failed: {e}")

def main():
    print("Steam Deck Shader Prediction Compiler - Dependency Test")
    print("=" * 60)
    
    # Required dependencies
    print("\n=== Required Dependencies ===")
    required_ok = all([
        test_import('psutil'),
        test_import('requests'),
        test_import('yaml'),
    ])
    
    # ML dependencies
    print("\n=== ML Dependencies ===")
    ml_ok = all([
        test_import('numpy', required=False),
        test_import('sklearn', required=False),
        test_import('scipy', required=False),
        test_import('pandas', required=False),
        test_import('joblib', required=False),
    ])
    
    # Networking dependencies
    print("\n=== Networking Dependencies ===")
    net_ok = all([
        test_import('aiohttp', required=False),
        test_import('cryptography', required=False),
    ])
    
    # System resources
    test_system_resources()
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Required dependencies: {'OK' if required_ok else 'FAILED'}")
    print(f"ML functionality: {'Full' if ml_ok else 'Limited'}")
    print(f"P2P networking: {'Enabled' if net_ok else 'Disabled'}")
    
    if not required_ok:
        print("\n❌ Installation needs repair - core dependencies missing")
        sys.exit(1)
    elif not ml_ok:
        print("\n⚠️  Partial installation - ML features limited")
        sys.exit(2)
    else:
        print("\n✅ All dependencies OK")
        sys.exit(0)

if __name__ == '__main__':
    main()
EOF
    
    chmod +x "$HOME/.local/bin/test-shader-deps"
    log_success "Dependency test script created at ~/.local/bin/test-shader-deps"
}

main() {
    echo -e "${BLUE}Steam Deck Dependencies Fix Script${NC}"
    echo "=================================="
    echo
    
    detect_environment
    setup_pip_environment
    create_requirements_steamdeck
    
    if install_core_dependencies; then
        install_ml_dependencies || log_warning "ML dependencies partially installed"
        install_networking_dependencies || log_warning "Networking dependencies partially installed"
    else
        log_error "Core dependency installation failed"
        exit 1
    fi
    
    install_fallback_packages
    create_dependency_test
    
    echo
    verification_result=$(verify_installation)
    exit_code=$?
    
    echo
    case $exit_code in
        0)
            echo -e "${GREEN}✅ Dependencies successfully installed and verified${NC}"
            echo "Run 'test-shader-deps' anytime to check dependency status"
            ;;
        1)
            echo -e "${RED}❌ Dependency installation failed${NC}"
            echo "Please check the error messages above and try manual installation"
            ;;
        2)
            echo -e "${YELLOW}⚠️  Partial installation completed${NC}"
            echo "Some ML features may be limited, but basic functionality is available"
            ;;
    esac
    
    echo
    echo "Next steps:"
    echo "1. Test dependencies: test-shader-deps"
    echo "2. Run main installer: bash steamdeck-optimized-install.sh"
    
    exit $exit_code
}

main "$@"