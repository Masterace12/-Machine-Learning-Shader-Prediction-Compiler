#!/bin/bash
# Shader Predictive Compiler - One-Liner GitHub Installer
# This script downloads and installs directly from GitHub with a single command

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
REPO_URL="https://github.com/Masterace12/shader-prediction-compilation"
REPO_BRANCH="main"
TEMP_DIR="/tmp/shader-predict-install-$$"
INSTALL_NAME="shader-prediction-compilation-main"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_header() { echo -e "\n${BOLD}${BLUE}$1${NC}"; }

# Cleanup function
cleanup() {
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Banner
echo -e "${BOLD}${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           SHADER PREDICTIVE COMPILER INSTALLER               ║"
echo "║                                                               ║"
echo "║  One-liner installation from GitHub                          ║"
echo "║  Automatically downloads, fixes, and installs                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_error "Do not run as root! The installer will ask for sudo when needed."
    exit 1
fi

# Check for required tools
log_header "Checking Requirements"

missing_tools=()

if ! command -v git &>/dev/null && ! command -v curl &>/dev/null && ! command -v wget &>/dev/null; then
    missing_tools+=("git, curl, or wget")
fi

if ! command -v unzip &>/dev/null; then
    missing_tools+=("unzip")
fi

if [ ${#missing_tools[@]} -ne 0 ]; then
    log_error "Missing required tools: ${missing_tools[*]}"
    log_info "Install with: sudo pacman -S git curl wget unzip"
    exit 1
fi

log_success "All required tools available"

# Download methods in order of preference
download_repo() {
    log_header "Downloading from GitHub"
    
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Method 1: Git clone (best)
    if command -v git &>/dev/null; then
        log_info "Using git clone..."
        if git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL.git" "$INSTALL_NAME" 2>/dev/null; then
            log_success "Downloaded with git clone"
            return 0
        else
            log_warning "Git clone failed, trying alternative methods..."
        fi
    fi
    
    # Method 2: curl
    if command -v curl &>/dev/null; then
        log_info "Using curl to download ZIP..."
        DOWNLOAD_URL="$REPO_URL/archive/refs/heads/$REPO_BRANCH.zip"
        if curl -sL "$DOWNLOAD_URL" -o repo.zip; then
            unzip -q repo.zip
            mv "*-$REPO_BRANCH" "$INSTALL_NAME" 2>/dev/null || mv shader-prediction-compilation* "$INSTALL_NAME"
            log_success "Downloaded with curl"
            return 0
        else
            log_warning "curl download failed, trying wget..."
        fi
    fi
    
    # Method 3: wget
    if command -v wget &>/dev/null; then
        log_info "Using wget to download ZIP..."
        DOWNLOAD_URL="$REPO_URL/archive/refs/heads/$REPO_BRANCH.zip"
        if wget -q "$DOWNLOAD_URL" -O repo.zip; then
            unzip -q repo.zip  
            mv "*-$REPO_BRANCH" "$INSTALL_NAME" 2>/dev/null || mv shader-prediction-compilation* "$INSTALL_NAME"
            log_success "Downloaded with wget"
            return 0
        fi
    fi
    
    log_error "All download methods failed"
    log_info "Please check your internet connection and GitHub repository URL"
    return 1
}

# Fix GitHub download issues
fix_github_issues() {
    log_header "Fixing GitHub Download Issues"
    
    cd "$TEMP_DIR/$INSTALL_NAME"
    
    # Fix permissions
    log_info "Fixing file permissions..."
    find . -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    chmod +x INSTALL.sh 2>/dev/null || true
    find . -name "install" -exec chmod +x {} \; 2>/dev/null || true
    find . -name "*.py" -exec chmod +x {} \; 2>/dev/null || true
    
    # Fix line endings if needed
    if command -v dos2unix &>/dev/null; then
        log_info "Converting line endings with dos2unix..."
        find . \( -name "*.sh" -o -name "*.py" -o -name "install*" \) -exec dos2unix {} \; 2>/dev/null || true
    else
        log_info "Converting line endings with sed..."
        find . \( -name "*.sh" -o -name "*.py" -o -name "install*" \) -type f -exec sed -i 's/\r$//' {} \; 2>/dev/null || true
    fi
    
    log_success "GitHub issues fixed"
}

# Run the installation
run_installation() {
    log_header "Running Installation"
    
    cd "$TEMP_DIR/$INSTALL_NAME"
    
    # Try different installation methods
    if [ -x "./INSTALL.sh" ]; then
        log_info "Running universal installer..."
        ./INSTALL.sh "$@"
    elif [ -d "shader-predict-compile" ] && [ -x "shader-predict-compile/install" ]; then
        log_info "Running main installer..."
        cd shader-predict-compile
        ./install "$@"
    elif [ -x "./install" ]; then
        log_info "Running direct installer..."
        ./install "$@"
    else
        log_error "No installer found!"
        log_info "Available files:"
        ls -la
        exit 1
    fi
}

# Main installation process
main() {
    log_info "Starting one-liner installation from GitHub..."
    
    # Download repository
    if ! download_repo; then
        exit 1
    fi
    
    # Fix GitHub issues
    fix_github_issues
    
    # Run installation
    run_installation "$@"
    
    # Success message
    echo
    log_success "Installation completed successfully!"
    log_info "Shader Predictive Compiler is now installed on your Steam Deck!"
    echo
    log_info "You can find it in:"
    echo "  • Gaming Mode: Library → Non-Steam → Shader Predictive Compiler"
    echo "  • Desktop Mode: Applications → Games → Shader Predictive Compiler"
    echo
}

# Run main function with all arguments
main "$@"