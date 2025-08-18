#!/bin/bash
set -euo pipefail

#=============================================================================
# Enhanced ML Shader Prediction Compiler Uninstaller
# Production-grade removal with complete cleanup and data preservation options
#=============================================================================

VERSION="2.1.0-enhanced"
UNINSTALL_LOG="/tmp/shader-predict-uninstall-$(date +%s).log"

# Installation directories
INSTALL_BASE="$HOME/.local"
INSTALL_DIR="$INSTALL_BASE/shader-predict-compile"
CONFIG_DIR="$HOME/.config/shader-predict-compile"
CACHE_DIR="$HOME/.cache/shader-predict-compile"
DATA_DIR="$HOME/.local/share/shader-predict-compile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$UNINSTALL_LOG"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$UNINSTALL_LOG"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$UNINSTALL_LOG"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$UNINSTALL_LOG"; }

# Options
PRESERVE_CONFIG=false
PRESERVE_CACHE=false
FORCE_REMOVE=false
BACKUP_DATA=true

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --preserve-config)
                PRESERVE_CONFIG=true
                shift
                ;;
            --preserve-cache)
                PRESERVE_CACHE=true
                shift
                ;;
            --preserve-all)
                PRESERVE_CONFIG=true
                PRESERVE_CACHE=true
                BACKUP_DATA=true
                shift
                ;;
            --force)
                FORCE_REMOVE=true
                shift
                ;;
            --no-backup)
                BACKUP_DATA=false
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
}

# Help function
show_help() {
    cat << EOF
ML Shader Prediction Compiler Enhanced Uninstaller v$VERSION

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --help                Show this help message
    --preserve-config     Keep configuration files
    --preserve-cache      Keep cache data
    --preserve-all        Keep all user data and settings
    --force              Skip confirmation prompts
    --no-backup          Skip creating backup before removal

EXAMPLES:
    $0                   Standard uninstallation with backup
    $0 --preserve-all    Remove software but keep all data
    $0 --force           Force removal without prompts

EOF
}

# Check if installation exists
check_installation() {
    log_info "Checking for ML Shader Prediction Compiler installation"
    
    local found_components=()
    
    if [[ -d "$INSTALL_DIR" ]]; then
        found_components+=("Main installation: $INSTALL_DIR")
    fi
    
    if [[ -d "$CONFIG_DIR" ]]; then
        found_components+=("Configuration: $CONFIG_DIR")
    fi
    
    if [[ -f "$HOME/.local/bin/shader-predict" ]]; then
        found_components+=("Launcher script: $HOME/.local/bin/shader-predict")
    fi
    
    # Check systemd services
    local services=($(systemctl --user list-unit-files | grep ml-shader | awk '{print $1}'))
    if [[ ${#services[@]} -gt 0 ]]; then
        found_components+=("Systemd services: ${services[*]}")
    fi
    
    if [[ ${#found_components[@]} -eq 0 ]]; then
        log_warn "No ML Shader Prediction Compiler installation found"
        return 1
    fi
    
    log_info "Found installation components:"
    printf '%s\n' "${found_components[@]}" | sed 's/^/  - /'
    
    return 0
}

# Stop and disable systemd services
stop_services() {
    log_info "Stopping and disabling systemd services"
    
    local services=(
        "ml-shader-predictor.service"
        "ml-shader-thermal.service"
        "ml-shader-predictor-maintenance.service"
        "ml-shader-predictor-maintenance.timer"
    )
    
    for service in "${services[@]}"; do
        if systemctl --user is-active --quiet "$service" 2>/dev/null; then
            log_info "Stopping $service"
            systemctl --user stop "$service" || log_warn "Failed to stop $service"
        fi
        
        if systemctl --user is-enabled --quiet "$service" 2>/dev/null; then
            log_info "Disabling $service"
            systemctl --user disable "$service" || log_warn "Failed to disable $service"
        fi
        
        # Remove service file
        local service_file="$HOME/.config/systemd/user/$service"
        if [[ -f "$service_file" ]]; then
            rm -f "$service_file"
            log_info "Removed $service_file"
        fi
    done
    
    # Reload systemd
    systemctl --user daemon-reload
    log_success "Systemd services cleaned up"
}

# Create backup of user data
create_backup() {
    if [[ "$BACKUP_DATA" != true ]]; then
        return 0
    fi
    
    log_info "Creating backup of user data"
    
    local backup_dir="$HOME/shader-predict-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup configuration
    if [[ -d "$CONFIG_DIR" ]]; then
        cp -r "$CONFIG_DIR" "$backup_dir/config" 2>/dev/null || true
        log_info "Configuration backed up to: $backup_dir/config"
    fi
    
    # Backup cache data (if small enough)
    if [[ -d "$CACHE_DIR" ]]; then
        local cache_size=$(du -sb "$CACHE_DIR" 2>/dev/null | cut -f1)
        if [[ $cache_size -lt 104857600 ]]; then # Less than 100MB
            cp -r "$CACHE_DIR" "$backup_dir/cache" 2>/dev/null || true
            log_info "Cache backed up to: $backup_dir/cache"
        else
            log_warn "Cache too large ($(numfmt --to=iec $cache_size)), skipping backup"
        fi
    fi
    
    # Backup logs
    if [[ -d "$DATA_DIR/logs" ]]; then
        cp -r "$DATA_DIR/logs" "$backup_dir/logs" 2>/dev/null || true
    fi
    
    # Create backup metadata
    cat > "$backup_dir/backup_info.txt" << EOF
ML Shader Prediction Compiler Backup
Created: $(date)
Version: $VERSION
Original install dir: $INSTALL_DIR
Original config dir: $CONFIG_DIR

To restore:
1. Reinstall ML Shader Prediction Compiler
2. Copy config/* to $CONFIG_DIR
3. Copy cache/* to $CACHE_DIR (if desired)
EOF
    
    log_success "Backup created at: $backup_dir"
}

# Remove main installation
remove_installation() {
    log_info "Removing main installation"
    
    if [[ -d "$INSTALL_DIR" ]]; then
        rm -rf "$INSTALL_DIR"
        log_info "Removed installation directory: $INSTALL_DIR"
    fi
    
    # Remove launcher script
    if [[ -f "$HOME/.local/bin/shader-predict" ]]; then
        rm -f "$HOME/.local/bin/shader-predict"
        log_info "Removed launcher script"
    fi
    
    # Remove desktop files
    if [[ -f "$HOME/.local/share/applications/shader-predict.desktop" ]]; then
        rm -f "$HOME/.local/share/applications/shader-predict.desktop"
        log_info "Removed desktop file"
    fi
    
    log_success "Main installation removed"
}

# Remove configuration and data
remove_user_data() {
    if [[ "$PRESERVE_CONFIG" != true ]] && [[ -d "$CONFIG_DIR" ]]; then
        rm -rf "$CONFIG_DIR"
        log_info "Removed configuration directory: $CONFIG_DIR"
    elif [[ "$PRESERVE_CONFIG" == true ]]; then
        log_info "Preserving configuration directory: $CONFIG_DIR"
    fi
    
    if [[ "$PRESERVE_CACHE" != true ]] && [[ -d "$CACHE_DIR" ]]; then
        rm -rf "$CACHE_DIR"
        log_info "Removed cache directory: $CACHE_DIR"
    elif [[ "$PRESERVE_CACHE" == true ]]; then
        log_info "Preserving cache directory: $CACHE_DIR"
    fi
    
    # Always remove data directory unless specifically preserving all
    if [[ "$PRESERVE_CONFIG" != true ]] || [[ "$PRESERVE_CACHE" != true ]]; then
        if [[ -d "$DATA_DIR" ]]; then
            rm -rf "$DATA_DIR"
            log_info "Removed data directory: $DATA_DIR"
        fi
    fi
}

# Clean up environment
cleanup_environment() {
    log_info "Cleaning up environment"
    
    # Remove from PATH additions (if any)
    local bashrc="$HOME/.bashrc"
    local profile="$HOME/.profile"
    
    for file in "$bashrc" "$profile"; do
        if [[ -f "$file" ]] && grep -q "shader-predict" "$file" 2>/dev/null; then
            log_info "Cleaning shader-predict references from $file"
            sed -i '/shader-predict/d' "$file" 2>/dev/null || true
        fi
    done
    
    # Clean up any remaining processes
    pkill -f "shader-predict" 2>/dev/null || true
    pkill -f "ml-shader" 2>/dev/null || true
    
    log_success "Environment cleanup completed"
}

# Verify removal
verify_removal() {
    log_info "Verifying removal completion"
    
    local remaining_items=()
    
    if [[ -d "$INSTALL_DIR" ]]; then
        remaining_items+=("Installation directory: $INSTALL_DIR")
    fi
    
    if [[ -f "$HOME/.local/bin/shader-predict" ]]; then
        remaining_items+=("Launcher script")
    fi
    
    if systemctl --user list-unit-files | grep -q ml-shader 2>/dev/null; then
        remaining_items+=("Systemd services")
    fi
    
    if [[ ${#remaining_items[@]} -gt 0 ]]; then
        log_warn "Some items may not have been completely removed:"
        printf '%s\n' "${remaining_items[@]}" | sed 's/^/  - /'
        return 1
    else
        log_success "All components successfully removed"
        return 0
    fi
}

# Main uninstallation function
main_uninstall() {
    log_info "Starting ML Shader Prediction Compiler uninstallation v$VERSION"
    log_info "Uninstallation log: $UNINSTALL_LOG"
    
    # Check if installation exists
    if ! check_installation; then
        if [[ "$FORCE_REMOVE" != true ]]; then
            exit 0
        fi
        log_info "Forcing removal of any remaining components"
    fi
    
    # Confirm uninstallation
    if [[ "$FORCE_REMOVE" != true ]]; then
        echo ""
        echo "⚠️  This will remove ML Shader Prediction Compiler from your system"
        if [[ "$PRESERVE_CONFIG" == true ]] || [[ "$PRESERVE_CACHE" == true ]]; then
            echo "📁 User data will be preserved as requested"
        fi
        if [[ "$BACKUP_DATA" == true ]]; then
            echo "💾 A backup will be created before removal"
        fi
        echo ""
        read -p "Continue with uninstallation? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Uninstallation cancelled by user"
            exit 0
        fi
    fi
    
    # Create backup if requested
    create_backup
    
    # Stop services
    stop_services
    
    # Remove installation
    remove_installation
    
    # Remove user data (respecting preservation flags)
    remove_user_data
    
    # Clean up environment
    cleanup_environment
    
    # Verify removal
    if verify_removal; then
        log_success "Uninstallation completed successfully!"
        
        if [[ "$PRESERVE_CONFIG" == true ]] || [[ "$PRESERVE_CACHE" == true ]]; then
            echo ""
            echo "📁 Preserved data locations:"
            [[ "$PRESERVE_CONFIG" == true ]] && [[ -d "$CONFIG_DIR" ]] && echo "   Configuration: $CONFIG_DIR"
            [[ "$PRESERVE_CACHE" == true ]] && [[ -d "$CACHE_DIR" ]] && echo "   Cache: $CACHE_DIR"
        fi
        
        if [[ "$BACKUP_DATA" == true ]]; then
            echo ""
            echo "💾 Backup created in ~/shader-predict-backup-*"
        fi
        
        echo ""
        echo "✨ ML Shader Prediction Compiler has been successfully removed"
        echo "📋 Uninstallation log saved to: $UNINSTALL_LOG"
    else
        log_error "Uninstallation may be incomplete. Check log for details."
        exit 1
    fi
}

# Parse arguments and run
parse_arguments "$@"
main_uninstall