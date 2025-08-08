#!/bin/bash
# Shader Predictive Compiler - Optimized Universal Installer
# Single installer that handles all GitHub download issues and installation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'  
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# Navigate to shader-predict-compile directory
cd_to_project() {
    if [ -d "shader-predict-compile" ]; then
        cd shader-predict-compile
        log_success "Navigating to shader-predict-compile directory"
    else
        log_error "shader-predict-compile directory not found!"
        log_info "Make sure you extracted the ZIP file correctly"
        exit 1
    fi
}

# Fix all GitHub download issues
fix_github_issues() {
    log_info "Fixing GitHub download issues..."
    
    # Fix permissions
    find . -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    chmod +x install install-manual 2>/dev/null || true
    find . -name "*.py" -exec chmod +x {} \; 2>/dev/null || true
    
    # Fix line endings
    if command -v dos2unix &>/dev/null; then
        find . \( -name "*.sh" -o -name "*.py" -o -name "install*" \) -exec dos2unix {} \; 2>/dev/null || true
    else
        # Use sed as fallback
        for file in install *.sh src/*.py ui/*.py; do
            [ -f "$file" ] && { sed -i 's/\r$//' "$file" 2>/dev/null || sed -i '' 's/\r$//' "$file" 2>/dev/null; }
        done
    fi
    
    log_success "GitHub download issues fixed"
}

# Main installation
main() {
    echo "🚀 Shader Predictive Compiler - Optimized Installer"
    echo "=================================================="
    echo
    
    # Step 1: Navigate to project directory  
    cd_to_project
    
    # Step 2: Fix GitHub issues
    fix_github_issues
    
    # Step 3: Run the main installer
    log_info "Running main installer..."
    if [ -x "./install" ]; then
        ./install "$@"
    elif [ -f "./install" ]; then
        bash ./install "$@"
    else
        log_error "Main installer not found!"
        exit 1
    fi
}

main "$@"