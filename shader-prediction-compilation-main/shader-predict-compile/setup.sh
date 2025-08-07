#!/bin/bash

# Post-Download Setup Script for Shader Predictive Compiler
# Automatically fixes permissions and prepares installation after GitHub download

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "${BOLD}${BLUE}$1${NC}"; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Banner
show_banner() {
    clear
    echo -e "${BOLD}${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║              SHADER PREDICTIVE COMPILER                      ║"
    echo "║                   Post-Download Setup                         ║"
    echo "║                                                               ║"
    echo "║  Preparing installation files after GitHub download          ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
}

# Fix file permissions
fix_permissions() {
    log_header "🔧 Fixing File Permissions"
    
    # List of files that need execute permissions
    local executable_files=(
        "install"
        "install-manual"
        "auto_launcher.sh"
        "gaming_mode_launcher.sh"
        "launcher.sh"
        "uninstall.sh"
        "scripts/install.sh"
    )
    
    # Also fix any .sh files in the directory
    find "$SCRIPT_DIR" -name "*.sh" -type f ! -executable -exec chmod +x {} \;
    
    local fixed_count=0
    
    for file in "${executable_files[@]}"; do
        local filepath="$SCRIPT_DIR/$file"
        if [ -f "$filepath" ]; then
            if [ ! -x "$filepath" ]; then
                chmod +x "$filepath"
                log_info "Made executable: $file"
                ((fixed_count++))
            else
                log_info "Already executable: $file"
            fi
        else
            log_warning "File not found: $file"
        fi
    done
    
    # Fix Python files that should be executable
    local python_executables=(
        "src/background_service.py"
        "src/gaming_mode_ui.py"
        "ui/main_window.py"
        "test_game_detection.py"
        "restore_defaults.py"
    )
    
    for file in "${python_executables[@]}"; do
        local filepath="$SCRIPT_DIR/$file"
        if [ -f "$filepath" ]; then
            if [ ! -x "$filepath" ]; then
                chmod +x "$filepath"
                log_info "Made executable: $file"
                ((fixed_count++))
            fi
        fi
    done
    
    if [ $fixed_count -gt 0 ]; then
        log_success "Fixed permissions for $fixed_count files"
    else
        log_success "All files already have correct permissions"
    fi
    echo
}

# Verify installation files
verify_files() {
    log_header "🔍 Verifying Installation Files"
    
    local required_files=(
        "install"
        "install-manual"
        "auto_launcher.sh"
        "src/background_service.py"
        "ui/main_window.py"
        "config/default_settings.json"
        "requirements.txt"
        "README.md"
    )
    
    local missing_files=()
    local present_files=0
    
    for file in "${required_files[@]}"; do
        if [ -f "$SCRIPT_DIR/$file" ]; then
            ((present_files++))
            log_info "✓ Found: $file"
        else
            missing_files+=("$file")
            log_error "✗ Missing: $file"
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        log_success "All required files are present ($present_files files)"
    else
        log_error "Missing ${#missing_files[@]} required files"
        log_error "This may indicate an incomplete download"
        echo
        log_info "Missing files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        return 1
    fi
    echo
}

# Create convenience symlinks
create_convenience_links() {
    log_header "🔗 Creating Convenience Links"
    
    # Create a simple 'install' link if it doesn't exist
    if [ ! -f "$SCRIPT_DIR/install" ] && [ -f "$SCRIPT_DIR/scripts/install.sh" ]; then
        ln -sf "scripts/install.sh" "$SCRIPT_DIR/install"
        log_info "Created install symlink"
    fi
    
    # Create README symlink in root if needed
    if [ -f "$SCRIPT_DIR/../README.md" ] && [ ! -f "$SCRIPT_DIR/README.md" ]; then
        ln -sf "../README.md" "$SCRIPT_DIR/README.md"
        log_info "Linked README.md"
    fi
    
    log_success "Convenience links created"
    echo
}

# Show next steps
show_next_steps() {
    log_header "🚀 Setup Complete - Next Steps"
    
    echo "Your Shader Predictive Compiler is now ready for installation!"
    echo
    echo "📋 ${BOLD}Installation Options:${NC}"
    echo
    echo "  ${GREEN}1. Easy Installation (Recommended):${NC}"
    echo "     ${BLUE}./install${NC}"
    echo "     • Auto-detects your Steam Deck model"
    echo "     • Installs all dependencies"
    echo "     • Sets up background service"
    echo "     • Creates desktop entries"
    echo
    echo "  ${GREEN}2. Manual Installation (If auto-install fails):${NC}"
    echo "     ${BLUE}./install-manual${NC}"
    echo "     • Bypasses pacman if it's having issues"
    echo "     • Uses pip for Python dependencies"
    echo "     • Manual dependency resolution"
    echo
    echo "  ${GREEN}3. Quick Test (Before installing):${NC}"
    echo "     ${BLUE}./auto_launcher.sh${NC}"
    echo "     • Tests the application without installing"
    echo "     • Checks dependencies"
    echo "     • Shows what would be installed"
    echo
    echo "📁 ${BOLD}Current Directory:${NC}"
    echo "   $(pwd)"
    echo
    echo "📖 ${BOLD}Documentation:${NC}"
    echo "   README.md - Full documentation and troubleshooting"
    echo
    echo "🎮 ${BOLD}After Installation:${NC}"
    echo "   • Gaming Mode: Library → Non-Steam → Shader Predictive Compiler"
    echo "   • Desktop Mode: Applications → Games → Shader Predictive Compiler"
    echo
}

# Main execution
main() {
    show_banner
    
    log_info "Setting up Shader Predictive Compiler from downloaded files..."
    log_info "Working directory: $SCRIPT_DIR"
    echo
    
    # Verify files exist
    if ! verify_files; then
        log_error "Setup cannot continue with missing files"
        log_info "Please re-download the project from GitHub"
        exit 1
    fi
    
    # Fix permissions
    fix_permissions
    
    # Create convenience links
    create_convenience_links
    
    # Show next steps
    show_next_steps
    
    log_success "Setup completed successfully!"
}

# Run main function
main "$@"