#!/bin/bash

# Enhanced Dependency Installer for Shader Predictive Compiler
# Handles all dependency installation with PGP signature fix support

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions with timestamps
log_info() { echo -e "${BLUE}[$(date +'%H:%M:%S') INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[$(date +'%H:%M:%S') ✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[$(date +'%H:%M:%S') !]${NC} $1"; }
log_error() { echo -e "${RED}[$(date +'%H:%M:%S') ✗]${NC} $1"; }
log_debug() { echo -e "${PURPLE}[$(date +'%H:%M:%S') DEBUG]${NC} $1"; }
log_header() { echo -e "\n${BOLD}${BLUE}$1${NC}"; }

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_error "Do not run this script as root!"
    log_info "The script will ask for sudo when needed."
    exit 1
fi

# Enhanced Banner
echo -e "${BOLD}${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║      ENHANCED DEPENDENCY INSTALLER FOR STEAM DECK            ║"
echo "║                                                               ║"
echo "║  🔧 PGP Signature Fix    🚀 Performance Optimized            ║"
echo "║  📦 Robust Installation  🛡️  Error Recovery                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Function to check if a command exists
command_exists() {
    command -v "$1" &>/dev/null 2>&1
}

# Function to check Python module
python_module_exists() {
    python3 -c "import $1" &>/dev/null 2>&1
}

# Enhanced PGP signature fix for pacman
fix_pacman_pgp() {
    log_header "Fixing Pacman PGP Signature Issues"
    
    # Check system time first
    log_info "Verifying system time synchronization..."
    if command_exists timedatectl; then
        local time_status=$(timedatectl status 2>/dev/null | grep "System clock synchronized" | cut -d':' -f2 | xargs || echo "unknown")
        if [[ "$time_status" != "yes" ]]; then
            log_warning "System clock may not be synchronized"
            log_info "Attempting to synchronize system clock..."
            sudo timedatectl set-ntp true 2>/dev/null || log_warning "Could not enable NTP"
            sleep 2
        else
            log_success "System clock is synchronized"
        fi
    fi
    
    # Clear pacman cache to remove potentially corrupted packages
    log_info "Clearing pacman cache..."
    sudo pacman -Scc --noconfirm 2>/dev/null || log_warning "Could not clear cache completely"
    
    # Remove and reinitialize GPG keyring
    log_info "Reinitializing pacman keyring (this may take several minutes)..."
    sudo rm -rf /etc/pacman.d/gnupg 2>/dev/null || true
    
    # Initialize new keyring
    log_info "Creating new keyring..."
    sudo pacman-key --init
    
    # Populate with Arch Linux keys
    log_info "Populating Arch Linux keys..."
    sudo pacman-key --populate archlinux
    
    # For SteamOS, also try to populate SteamOS keys
    if [[ -f "/etc/os-release" ]] && grep -q -i "steamos\|steam" /etc/os-release 2>/dev/null; then
        log_info "SteamOS detected - populating SteamOS keys..."
        sudo pacman-key --populate steamos 2>/dev/null || {
            log_warning "SteamOS keys not available, continuing with Arch keys only"
        }
    fi
    
    # Refresh keys from keyserver (with timeout)
    log_info "Refreshing keys from keyserver (timeout: 60s)..."
    timeout 60 sudo pacman-key --refresh-keys 2>/dev/null || {
        log_warning "Key refresh timed out or failed - continuing anyway"
    }
    
    # Test keyring with a minimal update
    log_info "Testing keyring with package database update..."
    if sudo pacman -Sy --noconfirm 2>/dev/null; then
        log_success "Pacman keyring repair completed successfully"
        return 0
    else
        log_warning "Standard keyring repair failed, trying alternative approach..."
        
        # Alternative: temporarily disable signature checking for critical updates
        log_info "Attempting emergency package database update..."
        sudo pacman -Sy --disable-download-timeout 2>/dev/null || {
            log_error "Could not repair pacman keyring"
            return 1
        }
        
        log_warning "Emergency update succeeded - keyring partially repaired"
        return 0
    fi
}

# Update package database with enhanced error handling
update_package_db() {
    log_header "Updating Package Database"
    
    if command_exists pacman; then
        log_info "Updating pacman database..."
        
        # First, try normal update
        if sudo pacman -Sy --noconfirm 2>/dev/null; then
            log_success "Package database updated successfully"
            return 0
        fi
        
        log_warning "Initial package database update failed"
        
        # Check if this is a PGP signature issue
        local update_error
        update_error=$(sudo pacman -Sy --noconfirm 2>&1) || true
        
        if echo "$update_error" | grep -q -i "signature\|pgp\|gpg\|key"; then
            log_warning "PGP signature issue detected - attempting repair..."
            
            if fix_pacman_pgp; then
                log_success "PGP issues resolved, package database updated"
                return 0
            else
                log_error "Could not resolve PGP signature issues"
                return 1
            fi
        else
            log_error "Package database update failed for unknown reason"
            log_debug "Error output: $update_error"
            return 1
        fi
        
    elif command_exists apt; then
        log_info "Updating APT package database..."
        if sudo apt update 2>/dev/null; then
            log_success "APT package database updated"
        else
            log_warning "APT update failed - attempting with fix-missing"
            sudo apt update --fix-missing 2>/dev/null || {
                log_error "APT update failed"
                return 1
            }
        fi
        
    elif command_exists dnf; then
        log_info "Updating DNF package database..."
        sudo dnf makecache --refresh 2>/dev/null || {
            log_error "DNF update failed"
            return 1
        }
        log_success "DNF package database updated"
        
    else
        log_warning "No supported package manager found"
        return 1
    fi
}

# Install system dependencies
install_system_deps() {
    log_header "Installing System Dependencies"
    
    local deps_to_install=()
    
    # Check Python 3
    if ! command_exists python3; then
        deps_to_install+=("python")
    else
        log_success "Python 3: Already installed"
    fi
    
    # Check pip
    if ! python3 -m pip --version &>/dev/null; then
        deps_to_install+=("python-pip")
    else
        log_success "pip: Already installed"
    fi
    
    # Check GTK bindings
    if ! python_module_exists gi; then
        deps_to_install+=("python-gobject")
    else
        log_success "GTK bindings: Already installed"
    fi
    
    # Check git (for updates)
    if ! command_exists git; then
        deps_to_install+=("git")
    else
        log_success "git: Already installed"
    fi
    
    # Install missing dependencies
    if [ ${#deps_to_install[@]} -gt 0 ]; then
        log_info "Installing: ${deps_to_install[*]}"
        
        if command_exists pacman; then
            sudo pacman -S --needed --noconfirm "${deps_to_install[@]}" || {
                log_error "Failed to install system dependencies"
                log_info "You may need to install these manually:"
                for dep in "${deps_to_install[@]}"; do
                    echo "  - $dep"
                done
                return 1
            }
        else
            log_error "Package manager not found"
            log_info "Please install these packages manually:"
            for dep in "${deps_to_install[@]}"; do
                echo "  - $dep"
            done
            return 1
        fi
        
        log_success "System dependencies installed"
    else
        log_success "All system dependencies already installed"
    fi
}

# Install Python dependencies
install_python_deps() {
    log_header "Installing Python Dependencies"
    
    # Ensure pip is up to date
    log_info "Updating pip..."
    python3 -m pip install --user --upgrade pip || {
        log_warning "Failed to upgrade pip"
    }
    
    # Install from requirements.txt if it exists
    if [ -f "requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        python3 -m pip install --user -r requirements.txt || {
            log_warning "Some packages failed to install"
            log_info "Trying to install packages individually..."
            
            # Try installing each package individually
            while IFS= read -r line; do
                # Skip comments and empty lines
                [[ "$line" =~ ^#.*$ ]] && continue
                [[ -z "$line" ]] && continue
                
                package=$(echo "$line" | sed 's/[<>=].*//g')
                log_info "Installing $package..."
                python3 -m pip install --user "$line" || {
                    log_warning "Failed to install $package"
                }
            done < requirements.txt
        }
    else
        # Manual installation of known dependencies
        log_info "No requirements.txt found, installing known dependencies..."
        
        local python_packages=(
            "PyGObject>=3.40.0"
            "psutil>=5.8.0"
            "numpy>=1.20.0"
        )
        
        for package in "${python_packages[@]}"; do
            package_name=$(echo "$package" | sed 's/[<>=].*//g')
            if python_module_exists "$package_name"; then
                log_success "$package_name: Already installed"
            else
                log_info "Installing $package..."
                python3 -m pip install --user "$package" || {
                    log_warning "Failed to install $package"
                }
            fi
        done
    fi
    
    # Verify installations
    log_header "Verifying Python Dependencies"
    
    if python_module_exists gi; then
        log_success "GTK bindings (PyGObject): OK"
    else
        log_error "GTK bindings (PyGObject): FAILED"
    fi
    
    if python_module_exists psutil; then
        PSUTIL_VERSION=$(python3 -c "import psutil; print(psutil.__version__)")
        log_success "psutil: OK (version $PSUTIL_VERSION)"
    else
        log_warning "psutil: Not available (optional)"
    fi
    
    if python_module_exists numpy; then
        NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
        log_success "numpy: OK (version $NUMPY_VERSION)"
    else
        log_warning "numpy: Not available (optional)"
    fi
}

# Steam Deck specific optimizations
steam_deck_optimizations() {
    log_header "Steam Deck Optimizations"
    
    # Add user to necessary groups
    if groups | grep -q video; then
        log_success "User already in 'video' group"
    else
        log_info "Adding user to 'video' group for GPU access..."
        sudo usermod -a -G video "$USER" || {
            log_warning "Failed to add user to video group"
        }
    fi
    
    # Create cache directories
    log_info "Creating cache directories..."
    mkdir -p "$HOME/.cache/shader-predict-compile"
    mkdir -p "$HOME/.config/shader-predict-compile"
    
    # Set up Python user path
    PYTHON_USER_BASE=$(python3 -m site --user-base)
    PYTHON_USER_SITE=$(python3 -m site --user-site)
    
    log_info "Python user base: $PYTHON_USER_BASE"
    log_info "Python user site: $PYTHON_USER_SITE"
    
    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$PYTHON_USER_BASE/bin:"* ]]; then
        log_info "Adding Python user bin to PATH..."
        echo "export PATH=\"\$PATH:$PYTHON_USER_BASE/bin\"" >> "$HOME/.bashrc"
        export PATH="$PATH:$PYTHON_USER_BASE/bin"
        log_success "PATH updated"
    fi
    
    log_success "Steam Deck optimizations applied"
}

# Main installation process
main() {
    log_info "Starting dependency installation..."
    
    # Check if we're on Steam Deck
    if [ -f "/sys/devices/virtual/dmi/id/product_name" ]; then
        PRODUCT=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "Unknown")
        if [[ "$PRODUCT" == *"Jupiter"* ]]; then
            log_success "Steam Deck detected"
        else
            log_warning "Not running on Steam Deck - some optimizations may not apply"
        fi
    fi
    
    # Update package database
    update_package_db
    
    # Install system dependencies
    if ! install_system_deps; then
        log_error "Failed to install system dependencies"
        exit 1
    fi
    
    # Install Python dependencies
    install_python_deps
    
    # Apply Steam Deck optimizations
    steam_deck_optimizations
    
    # Summary
    echo
    log_header "Installation Summary"
    
    local all_good=true
    
    # Check critical dependencies
    if command_exists python3 && python_module_exists gi; then
        log_success "Core dependencies: OK"
    else
        log_error "Core dependencies: MISSING"
        all_good=false
    fi
    
    # Check optional dependencies
    if python_module_exists psutil && python_module_exists numpy; then
        log_success "Optional dependencies: OK"
    else
        log_warning "Optional dependencies: PARTIAL"
    fi
    
    echo
    if [ "$all_good" = true ]; then
        echo -e "${GREEN}${BOLD}✓ Dependencies installed successfully!${NC}"
        echo
        echo "You can now run the installer: ./install"
    else
        echo -e "${YELLOW}${BOLD}⚠ Some dependencies could not be installed${NC}"
        echo
        echo "The application may still work with limited functionality."
        echo "Check the errors above and try installing missing packages manually."
    fi
    
    # Note about relogin
    if groups | grep -q video; then
        echo
        log_info "Note: You may need to log out and back in for group changes to take effect."
    fi
}

# Run main function
main "$@"