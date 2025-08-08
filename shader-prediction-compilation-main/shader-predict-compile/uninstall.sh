#!/bin/bash

# Shader Predictive Compiler Uninstall Script for Steam Deck
# Removes all traces of the application from the system

set -e

# Configuration
APP_NAME="shader-predict-compile"
INSTALL_DIR="/opt/${APP_NAME}"
DESKTOP_FILE="/usr/share/applications/${APP_NAME}.desktop"
SYSTEMD_SERVICE="/etc/systemd/system/${APP_NAME}.service"
CONFIG_DIR="$HOME/.config/${APP_NAME}"
CACHE_DIR="$HOME/.cache/${APP_NAME}"
AUTOSTART_FILE="$HOME/.config/autostart/${APP_NAME}.desktop"
FOSSILIZE_CONFIG="$HOME/.config/fossilize"
LOCAL_DESKTOP_FILE="$HOME/.local/share/applications/${APP_NAME}.desktop"

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

# Check if running as root
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        log_error "Do not run this uninstaller as root!"
        log_info "The uninstaller will ask for sudo when needed."
        exit 1
    fi
}

# Banner
show_banner() {
    clear
    echo -e "${BOLD}${RED}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                  SHADER PREDICTIVE COMPILER                  ║"
    echo "║                      UNINSTALLER                             ║"
    echo "║                                                               ║"
    echo "║  This will completely remove all traces of the application   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
}

# Show what will be removed
show_removal_plan() {
    log_header "🗑️  Uninstall Plan"
    echo
    echo "The following will be removed:"
    echo
    
    # System files
    echo "${BOLD}System Files:${NC}"
    [ -d "$INSTALL_DIR" ] && echo "  • $INSTALL_DIR (application files)"
    [ -f "$DESKTOP_FILE" ] && echo "  • $DESKTOP_FILE (desktop entry)"
    [ -f "$SYSTEMD_SERVICE" ] && echo "  • $SYSTEMD_SERVICE (background service)"
    
    # User files
    echo
    echo "${BOLD}User Files:${NC}"
    [ -d "$CONFIG_DIR" ] && echo "  • $CONFIG_DIR (configuration files)"
    [ -f "$AUTOSTART_FILE" ] && echo "  • $AUTOSTART_FILE (autostart entry)"
    [ -f "$LOCAL_DESKTOP_FILE" ] && echo "  • $LOCAL_DESKTOP_FILE (user desktop entry)"
    
    # Cache and optional files
    echo
    echo "${BOLD}Optional (will ask before removing):${NC}"
    [ -d "$CACHE_DIR" ] && echo "  • $CACHE_DIR (shader cache - frees disk space)"
    [ -d "$FOSSILIZE_CONFIG" ] && echo "  • $FOSSILIZE_CONFIG (Fossilize configuration)"
    
    # Gaming Mode integration
    echo
    echo "${BOLD}Gaming Mode Integration:${NC}"
    echo "  • Non-Steam game entries (if added)"
    echo "  • Steam shortcuts (if created)"
    
    echo
}

# Check what's actually installed
check_installation() {
    log_header "🔍 Checking Installation"
    
    local found_something=false
    
    # Check system installation
    if [ -d "$INSTALL_DIR" ] || [ -f "$DESKTOP_FILE" ] || [ -f "$SYSTEMD_SERVICE" ]; then
        log_info "Found system installation"
        found_something=true
    fi
    
    # Check user installation
    if [ -d "$CONFIG_DIR" ] || [ -d "$CACHE_DIR" ] || [ -f "$AUTOSTART_FILE" ]; then
        log_info "Found user configuration"
        found_something=true
    fi
    
    # Check if service is running
    if systemctl is-active --quiet "${APP_NAME}.service" 2>/dev/null; then
        log_warning "Background service is currently running"
        found_something=true
    fi
    
    if [ "$found_something" = false ]; then
        log_info "No installation found"
        echo
        log_success "Shader Predictive Compiler is not installed"
        exit 0
    fi
    
    echo
}

# Stop and remove systemd service
remove_service() {
    log_header "⚙️  Removing Background Service"
    
    # Check if service exists
    if [ -f "$SYSTEMD_SERVICE" ]; then
        # Stop the service
        if systemctl is-active --quiet "${APP_NAME}.service" 2>/dev/null; then
            log_info "Stopping background service..."
            sudo systemctl stop "${APP_NAME}.service" || true
        fi
        
        # Disable the service
        if systemctl is-enabled --quiet "${APP_NAME}.service" 2>/dev/null; then
            log_info "Disabling background service..."
            sudo systemctl disable "${APP_NAME}.service" || true
        fi
        
        # Remove service file
        log_info "Removing service file..."
        sudo rm -f "$SYSTEMD_SERVICE"
        
        # Reload systemd
        sudo systemctl daemon-reload
        
        log_success "Background service removed"
    else
        log_info "No systemd service found"
    fi
    
    echo
}

# Remove system files
remove_system_files() {
    log_header "🗂️  Removing System Files"
    
    # Remove main application directory
    if [ -d "$INSTALL_DIR" ]; then
        log_info "Removing application directory: $INSTALL_DIR"
        sudo rm -rf "$INSTALL_DIR"
        log_success "Application directory removed"
    else
        log_info "Application directory not found"
    fi
    
    # Remove desktop entry
    if [ -f "$DESKTOP_FILE" ]; then
        log_info "Removing desktop entry: $DESKTOP_FILE"
        sudo rm -f "$DESKTOP_FILE"
        log_success "Desktop entry removed"
    else
        log_info "System desktop entry not found"
    fi
    
    echo
}

# Remove user files
remove_user_files() {
    log_header "👤 Removing User Files"
    
    # Remove configuration directory
    if [ -d "$CONFIG_DIR" ]; then
        log_info "Removing configuration directory: $CONFIG_DIR"
        rm -rf "$CONFIG_DIR"
        log_success "Configuration directory removed"
    else
        log_info "Configuration directory not found"
    fi
    
    # Remove autostart entry
    if [ -f "$AUTOSTART_FILE" ]; then
        log_info "Removing autostart entry: $AUTOSTART_FILE"
        rm -f "$AUTOSTART_FILE"
        log_success "Autostart entry removed"
    else
        log_info "Autostart entry not found"
    fi
    
    # Remove user desktop entry
    if [ -f "$LOCAL_DESKTOP_FILE" ]; then
        log_info "Removing user desktop entry: $LOCAL_DESKTOP_FILE"
        rm -f "$LOCAL_DESKTOP_FILE"
        log_success "User desktop entry removed"
    else
        log_info "User desktop entry not found"
    fi
    
    echo
}

# Remove cache (optional)
remove_cache() {
    if [ -d "$CACHE_DIR" ]; then
        log_header "💾 Cache Removal"
        echo
        echo "Shader cache found at: $CACHE_DIR"
        echo "Removing this will:"
        echo "  • Free up disk space"
        echo "  • Require shader recompilation for games"
        echo "  • Not affect game functionality"
        echo
        
        read -p "Remove shader cache? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing shader cache..."
            rm -rf "$CACHE_DIR"
            log_success "Shader cache removed"
        else
            log_info "Shader cache preserved"
        fi
    else
        log_info "No shader cache found"
    fi
    
    echo
}

# Remove Fossilize integration (optional)
remove_fossilize_config() {
    if [ -d "$FOSSILIZE_CONFIG" ] && [ -f "$FOSSILIZE_CONFIG/config.json" ]; then
        # Check if our config is there
        if grep -q "shader-predict-compile" "$FOSSILIZE_CONFIG/config.json" 2>/dev/null; then
            log_header "🔧 Fossilize Integration"
            echo
            echo "Fossilize configuration with our integration found."
            echo "This configuration was created by Shader Predictive Compiler."
            echo
            
            read -p "Remove Fossilize configuration? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Removing Fossilize configuration..."
                rm -rf "$FOSSILIZE_CONFIG"
                log_success "Fossilize configuration removed"
            else
                log_info "Fossilize configuration preserved"
            fi
        else
            log_info "Fossilize configuration exists but not created by our app"
        fi
    else
        log_info "No Fossilize integration found"
    fi
    
    echo
}

# Remove Gaming Mode integration
remove_gaming_mode_integration() {
    log_header "🎮 Gaming Mode Integration"
    
    # Try to remove from Steam's non-steam games
    if [ -d "$HOME/.steam" ] || [ -d "$HOME/.local/share/Steam" ]; then
        log_info "Checking for Gaming Mode integration..."
        
        # Look for shortcuts files
        local steam_dirs=(
            "$HOME/.steam/steam"
            "$HOME/.local/share/Steam"
        )
        
        local found_shortcuts=false
        
        for steam_dir in "${steam_dirs[@]}"; do
            if [ -d "$steam_dir" ]; then
                # Check for shortcuts
                local shortcuts_vdf="$steam_dir/userdata/*/config/shortcuts.vdf"
                if ls $shortcuts_vdf 2>/dev/null | head -1 >/dev/null; then
                    log_warning "Found Steam shortcuts - manual removal may be needed"
                    log_info "You may need to manually remove from Gaming Mode:"
                    log_info "  Library → Non-Steam Games → Right-click → Remove"
                    found_shortcuts=true
                fi
            fi
        done
        
        if [ "$found_shortcuts" = false ]; then
            log_info "No Gaming Mode shortcuts found"
        fi
    else
        log_info "Steam not found - no Gaming Mode integration to remove"
    fi
    
    echo
}

# Clean up any remaining traces
cleanup_remaining_traces() {
    log_header "🧹 Final Cleanup"
    
    # Update desktop database
    if command -v update-desktop-database >/dev/null 2>&1; then
        log_info "Updating desktop database..."
        update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
        sudo update-desktop-database "/usr/share/applications" 2>/dev/null || true
    fi
    
    # Update icon cache
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        log_info "Updating icon cache..."
        gtk-update-icon-cache -f -t "$HOME/.local/share/icons" 2>/dev/null || true
        sudo gtk-update-icon-cache -f -t "/usr/share/icons" 2>/dev/null || true
    fi
    
    # Clear any potential Python cache
    find "$HOME" -name "__pycache__" -path "*shader-predict-compile*" -type d -exec rm -rf {} + 2>/dev/null || true
    
    log_success "Final cleanup completed"
    echo
}

# Confirmation prompt
confirm_uninstall() {
    echo "${BOLD}${RED}⚠️  WARNING: This action cannot be undone!${NC}"
    echo
    read -p "Are you sure you want to completely uninstall Shader Predictive Compiler? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Uninstall cancelled"
        exit 0
    fi
    echo
}

# Uninstall complete message
uninstall_complete() {
    log_header "✅ Uninstall Complete"
    
    echo -e "${GREEN}Shader Predictive Compiler has been completely removed!${NC}"
    echo
    echo "What was removed:"
    echo "  ✓ Application files"
    echo "  ✓ System services"
    echo "  ✓ Desktop integration"
    echo "  ✓ User configuration"
    echo "  ✓ Autostart entries"
    echo
    
    echo "${BOLD}Notes:${NC}"
    echo "  • Some cache files may have been preserved (if you chose to keep them)"
    echo "  • Gaming Mode shortcuts may need manual removal"
    echo "  • Your Steam games and saves are unaffected"
    echo "  • Fossilize will continue to work normally"
    echo
    
    echo "Thank you for using Shader Predictive Compiler!"
    echo
}

# Dry run mode
dry_run() {
    log_header "🔍 Dry Run Mode (showing what would be removed)"
    echo
    
    show_removal_plan
    
    echo
    log_info "This was a dry run. Nothing was actually removed."
    log_info "Run without --dry-run to perform the actual uninstall."
}

# Help function
show_help() {
    echo "Shader Predictive Compiler Uninstaller for Steam Deck"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --dry-run     Show what would be removed without actually removing it"
    echo "  --force       Skip confirmation prompts (use with caution)"
    echo "  --keep-cache  Keep shader cache (preserve compiled shaders)"
    echo "  --help, -h    Show this help message"
    echo
    echo "Default action is to interactively uninstall the application."
    echo
    echo "Examples:"
    echo "  $0                    # Interactive uninstall"
    echo "  $0 --dry-run         # Show what would be removed"
    echo "  $0 --force          # Uninstall without prompts"
    echo "  $0 --keep-cache     # Uninstall but keep shader cache"
}

# Main uninstall function
main() {
    local dry_run_mode=false
    local force_mode=false
    local keep_cache=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run_mode=true
                shift
                ;;
            --force)
                force_mode=true
                shift
                ;;
            --keep-cache)
                keep_cache=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    show_banner
    check_permissions
    check_installation
    show_removal_plan
    
    if [ "$dry_run_mode" = true ]; then
        dry_run
        exit 0
    fi
    
    if [ "$force_mode" = false ]; then
        confirm_uninstall
    fi
    
    # Perform the actual uninstall
    remove_service
    remove_system_files
    remove_user_files
    
    if [ "$keep_cache" = false ]; then
        remove_cache
    else
        log_info "Keeping shader cache as requested"
        echo
    fi
    
    remove_fossilize_config
    remove_gaming_mode_integration
    cleanup_remaining_traces
    
    uninstall_complete
}

# Run main function
main "$@"