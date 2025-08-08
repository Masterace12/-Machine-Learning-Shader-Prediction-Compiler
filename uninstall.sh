#!/bin/bash

# Shader Predictive Compiler - One-Line Uninstaller
# Complete removal with backup and recovery options
#
# One-line uninstall command:
#   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash
#
# Safe uninstall (creates backup):
#   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash -s -- --backup
#
# For security-conscious users:
#   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh -o uninstall.sh
#   less uninstall.sh  # Inspect the script
#   chmod +x uninstall.sh && ./uninstall.sh

set -euo pipefail

# Wrap in function to prevent partial execution
main() {

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

readonly SCRIPT_VERSION="1.2.0"
readonly REPO_OWNER="YourUsername"
readonly REPO_NAME="shader-prediction-compilation"

readonly INSTALL_DIR="/opt/shader-predict-compile"
readonly CONFIG_DIR="${HOME}/.config/shader-predict-compile"
readonly CACHE_DIR="${HOME}/.cache/shader-predict-compile"
readonly BACKUP_DIR="${HOME}/.local/share/shader-predict-compile-backup"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Uninstall options
CREATE_BACKUP=false
FORCE_REMOVE=false
KEEP_CONFIG=false
KEEP_CACHE=false
DRY_RUN=false

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_header() {
    echo -e "\n${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}\n"
}

log_info() { 
    echo -e "${BLUE}[INFO]${NC} $1" 
}

log_success() { 
    echo -e "${GREEN}[✓]${NC} $1" 
}

log_warning() { 
    echo -e "${YELLOW}[!]${NC} $1" 
}

log_error() { 
    echo -e "${RED}[✗]${NC} $1" >&2
}

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

detect_installation() {
    log_info "Detecting Shader Prediction Compiler installation..."
    
    local found_components=()
    local missing_components=()
    
    # Check main installation
    if [[ -d "$INSTALL_DIR" ]]; then
        found_components+=("Main Installation ($INSTALL_DIR)")
    else
        missing_components+=("Main Installation")
    fi
    
    # Check configuration
    if [[ -d "$CONFIG_DIR" ]]; then
        found_components+=("Configuration ($CONFIG_DIR)")
    else
        missing_components+=("Configuration")
    fi
    
    # Check cache
    if [[ -d "$CACHE_DIR" ]]; then
        local cache_size
        cache_size=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "unknown")
        found_components+=("Cache ($CACHE_DIR) - $cache_size")
    else
        missing_components+=("Cache")
    fi
    
    # Check desktop integration
    if [[ -f "$HOME/.local/share/applications/shader-predict-compile.desktop" ]]; then
        found_components+=("Desktop Integration")
    fi
    
    if [[ -f "$HOME/.config/autostart/shader-predict-compile.desktop" ]]; then
        found_components+=("Autostart Service")
    fi
    
    # Check system launchers
    if [[ -f "/usr/local/bin/shader-predict-compile" ]]; then
        found_components+=("System Launcher")
    fi
    
    if [[ -f "/usr/local/bin/uninstall-shader-predict-compile" ]]; then
        found_components+=("Uninstaller Command")
    fi
    
    # Report findings
    if [[ ${#found_components[@]} -eq 0 ]]; then
        log_warning "No Shader Prediction Compiler installation detected"
        log_info "The application may have been manually removed or never installed"
        return 1
    fi
    
    log_success "Found ${#found_components[@]} installed components:"
    for component in "${found_components[@]}"; do
        echo "    • $component"
    done
    
    if [[ ${#missing_components[@]} -gt 0 ]]; then
        log_warning "Missing components (possibly already removed):"
        for component in "${missing_components[@]}"; do
            echo "    • $component"
        done
    fi
    
    return 0
}

check_running_processes() {
    log_info "Checking for running processes..."
    
    local running_processes=()
    
    # Check for shader prediction processes
    if pgrep -f "shader_prediction_system.py" >/dev/null 2>&1; then
        running_processes+=("shader_prediction_system.py")
    fi
    
    if pgrep -f "shader-predict-compile" >/dev/null 2>&1; then
        running_processes+=("shader-predict-compile")
    fi
    
    if [[ ${#running_processes[@]} -gt 0 ]]; then
        log_warning "Found ${#running_processes[@]} running processes:"
        for process in "${running_processes[@]}"; do
            echo "    • $process (PID: $(pgrep -f "$process" | head -1))"
        done
        
        if [[ "$FORCE_REMOVE" != "true" ]]; then
            echo
            echo -n "Stop running processes and continue? [Y/n]: "
            read -r response
            case "$response" in
                [nN][oO]|[nN])
                    log_info "Uninstallation cancelled by user"
                    exit 0
                    ;;
                *)
                    log_info "Will stop running processes"
                    ;;
            esac
        fi
        
        return 1  # Processes found
    else
        log_success "No running processes found"
        return 0  # No processes
    fi
}

# ============================================================================
# BACKUP FUNCTIONS
# ============================================================================

create_backup() {
    if [[ "$CREATE_BACKUP" != "true" ]]; then
        return 0
    fi
    
    log_info "Creating backup before uninstallation..."
    
    local backup_timestamp
    backup_timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_path="$BACKUP_DIR/$backup_timestamp"
    
    mkdir -p "$backup_path"
    
    # Backup configuration
    if [[ -d "$CONFIG_DIR" ]]; then
        log_info "Backing up configuration..."
        cp -r "$CONFIG_DIR" "$backup_path/config"
    fi
    
    # Backup cache (only ML models and important data, not raw cache)
    if [[ -d "$CACHE_DIR" ]]; then
        log_info "Backing up important cache data..."
        mkdir -p "$backup_path/cache"
        
        # Backup ML models if they exist
        if [[ -d "$CACHE_DIR/ml_models" ]]; then
            cp -r "$CACHE_DIR/ml_models" "$backup_path/cache/"
        fi
        
        # Backup user customizations
        if [[ -d "$CACHE_DIR/user_profiles" ]]; then
            cp -r "$CACHE_DIR/user_profiles" "$backup_path/cache/"
        fi
    fi
    
    # Create backup manifest
    cat > "$backup_path/backup_manifest.txt" << EOF
Shader Prediction Compiler Backup
Created: $(date)
Version: $(cat "$CONFIG_DIR/config.json" 2>/dev/null | grep -o '"version": "[^"]*"' | cut -d'"' -f4 || echo "unknown")
System: $(uname -a)

Backed up components:
- Configuration files from $CONFIG_DIR
- ML models and user profiles from $CACHE_DIR
- This manifest file

To restore:
1. Reinstall Shader Prediction Compiler
2. Copy contents from backup/config/ to ~/.config/shader-predict-compile/
3. Copy contents from backup/cache/ to ~/.cache/shader-predict-compile/

Backup location: $backup_path
EOF
    
    log_success "Backup created at: $backup_path"
    log_info "Use this backup to restore settings if you reinstall later"
}

# ============================================================================
# REMOVAL FUNCTIONS
# ============================================================================

stop_services() {
    log_info "Stopping Shader Prediction Compiler services..."
    
    # Stop background processes gracefully
    if pgrep -f "shader_prediction_system.py.*--service" >/dev/null 2>&1; then
        log_info "Stopping background service..."
        pkill -TERM -f "shader_prediction_system.py.*--service" 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if pgrep -f "shader_prediction_system.py.*--service" >/dev/null 2>&1; then
            log_warning "Force stopping background service..."
            pkill -KILL -f "shader_prediction_system.py.*--service" 2>/dev/null || true
        fi
    fi
    
    # Stop any other shader prediction processes
    if pgrep -f "shader.*predict" >/dev/null 2>&1; then
        log_info "Stopping other shader prediction processes..."
        pkill -TERM -f "shader.*predict" 2>/dev/null || true
        sleep 1
        pkill -KILL -f "shader.*predict" 2>/dev/null || true
    fi
    
    log_success "Services stopped"
}

remove_installation_directory() {
    if [[ ! -d "$INSTALL_DIR" ]]; then
        log_info "Installation directory not found, skipping"
        return 0
    fi
    
    log_info "Removing installation directory: $INSTALL_DIR"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would remove: $INSTALL_DIR"
        return 0
    fi
    
    if sudo rm -rf "$INSTALL_DIR" 2>/dev/null; then
        log_success "Installation directory removed"
    else
        log_error "Failed to remove installation directory"
        log_info "You may need to remove it manually: sudo rm -rf $INSTALL_DIR"
    fi
}

remove_user_config() {
    if [[ "$KEEP_CONFIG" == "true" ]]; then
        log_info "Keeping configuration files (--keep-config specified)"
        return 0
    fi
    
    if [[ ! -d "$CONFIG_DIR" ]]; then
        log_info "Configuration directory not found, skipping"
        return 0
    fi
    
    log_info "Removing configuration directory: $CONFIG_DIR"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would remove: $CONFIG_DIR"
        return 0
    fi
    
    rm -rf "$CONFIG_DIR"
    log_success "Configuration directory removed"
}

remove_user_cache() {
    if [[ "$KEEP_CACHE" == "true" ]]; then
        log_info "Keeping cache files (--keep-cache specified)"
        return 0
    fi
    
    if [[ ! -d "$CACHE_DIR" ]]; then
        log_info "Cache directory not found, skipping"
        return 0
    fi
    
    local cache_size
    cache_size=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "unknown")
    log_info "Removing cache directory: $CACHE_DIR ($cache_size)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would remove: $CACHE_DIR ($cache_size)"
        return 0
    fi
    
    rm -rf "$CACHE_DIR"
    log_success "Cache directory removed"
}

remove_desktop_integration() {
    log_info "Removing desktop integration..."
    
    local removed_count=0
    
    # Desktop entry
    if [[ -f "$HOME/.local/share/applications/shader-predict-compile.desktop" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would remove desktop entry"
        else
            rm -f "$HOME/.local/share/applications/shader-predict-compile.desktop"
            ((removed_count++))
        fi
    fi
    
    # Autostart entry
    if [[ -f "$HOME/.config/autostart/shader-predict-compile.desktop" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would remove autostart entry"
        else
            rm -f "$HOME/.config/autostart/shader-predict-compile.desktop"
            ((removed_count++))
        fi
    fi
    
    if [[ "$DRY_RUN" != "true" ]]; then
        if [[ $removed_count -gt 0 ]]; then
            log_success "Removed $removed_count desktop integration files"
        else
            log_info "No desktop integration files found"
        fi
    fi
}

remove_system_launchers() {
    log_info "Removing system launchers..."
    
    local removed_count=0
    
    # Main launcher
    if [[ -f "/usr/local/bin/shader-predict-compile" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would remove system launcher"
        else
            if sudo rm -f "/usr/local/bin/shader-predict-compile" 2>/dev/null; then
                ((removed_count++))
            else
                log_warning "Could not remove system launcher (permission denied)"
            fi
        fi
    fi
    
    # Uninstaller launcher
    if [[ -f "/usr/local/bin/uninstall-shader-predict-compile" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would remove uninstaller command"
        else
            if sudo rm -f "/usr/local/bin/uninstall-shader-predict-compile" 2>/dev/null; then
                ((removed_count++))
            else
                log_warning "Could not remove uninstaller command (permission denied)"
            fi
        fi
    fi
    
    if [[ "$DRY_RUN" != "true" ]]; then
        if [[ $removed_count -gt 0 ]]; then
            log_success "Removed $removed_count system launchers"
        else
            log_info "No system launchers found"
        fi
    fi
}

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup)
                CREATE_BACKUP=true
                log_info "Backup mode enabled"
                shift
                ;;
            --force)
                FORCE_REMOVE=true
                log_info "Force removal enabled"
                shift
                ;;
            --keep-config)
                KEEP_CONFIG=true
                log_info "Will preserve configuration files"
                shift
                ;;
            --keep-cache)
                KEEP_CACHE=true
                log_info "Will preserve cache files"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                log_info "Dry run mode - no files will be deleted"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --version|-v)
                echo "Shader Prediction Compiler Uninstaller v$SCRIPT_VERSION"
                exit 0
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
}

show_help() {
    cat << 'HELP_EOF'
Shader Prediction Compiler - One-Line Uninstaller

USAGE:
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash
    
    # Or with options:
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash -s -- [OPTIONS]

OPTIONS:
    --backup              Create backup of configuration and important data
    --force               Don't prompt for confirmation, force removal
    --keep-config         Preserve configuration files
    --keep-cache          Preserve cache files  
    --dry-run             Show what would be removed without actually removing
    --help, -h            Show this help message
    --version, -v         Show uninstaller version

EXAMPLES:
    # Standard uninstall
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash
    
    # Safe uninstall with backup
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash -s -- --backup
    
    # Keep user data, remove only application
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash -s -- --keep-config --keep-cache
    
    # Preview what would be removed
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash -s -- --dry-run

REMOVED COMPONENTS:
    • Main installation (/opt/shader-predict-compile)
    • Configuration files (~/.config/shader-predict-compile)
    • Cache files (~/.cache/shader-predict-compile)
    • Desktop integration and shortcuts
    • System launcher commands
    • Autostart services

PRESERVED:
    • Steam game shader caches (untouched)
    • Steam integration settings
    • System packages and dependencies
    • Other applications and data

SECURITY:
    For security-conscious users, inspect the script first:
    
    curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh -o uninstall.sh
    less uninstall.sh  # Inspect the script
    chmod +x uninstall.sh && ./uninstall.sh

MORE INFO:
    Repository: https://github.com/YourUsername/shader-prediction-compilation
    Issues: https://github.com/YourUsername/shader-prediction-compilation/issues
HELP_EOF
}

# ============================================================================
# MAIN UNINSTALLATION FLOW
# ============================================================================

show_banner() {
    echo -e "${BOLD}${YELLOW}"
    cat << 'BANNER_EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           🗑️  SHADER PREDICTION COMPILER - UNINSTALLER                       ║
║                                                                              ║
║           Complete Removal with Backup Options                              ║
║           Safe • Thorough • Reversible                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
BANNER_EOF
    echo -e "${NC}"
    
    log_info "Uninstaller Version: $SCRIPT_VERSION"
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN MODE - No files will actually be removed"
    fi
    echo
}

confirm_uninstall() {
    if [[ "$FORCE_REMOVE" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    echo -e "${BOLD}${RED}⚠️  WARNING: This will completely remove Shader Prediction Compiler!${NC}"
    echo
    echo "The following will be removed:"
    echo "  • Main application and all files"
    if [[ "$KEEP_CONFIG" != "true" ]]; then
        echo "  • Configuration files and settings"
    fi
    if [[ "$KEEP_CACHE" != "true" ]]; then
        echo "  • Cache files and compiled shaders"
    fi
    echo "  • Desktop shortcuts and integration"
    echo "  • System launcher commands"
    echo
    
    if [[ "$CREATE_BACKUP" == "true" ]]; then
        echo -e "${GREEN}✓${NC} Configuration and important data will be backed up"
        echo
    fi
    
    echo -n "Are you sure you want to continue? [y/N]: "
    read -r response
    case "$response" in
        [yY][eE][sS]|[yY])
            log_info "Proceeding with uninstallation..."
            return 0
            ;;
        *)
            log_info "Uninstallation cancelled by user"
            exit 0
            ;;
    esac
}

run_uninstallation() {
    log_header "DETECTING INSTALLATION"
    if ! detect_installation; then
        if [[ "$FORCE_REMOVE" != "true" ]]; then
            echo
            echo -n "Continue with cleanup of any remaining files? [y/N]: "
            read -r response
            case "$response" in
                [yY][eE][sS]|[yY])
                    log_info "Continuing with cleanup..."
                    ;;
                *)
                    log_info "Cleanup cancelled"
                    exit 0
                    ;;
            esac
        fi
    fi
    
    check_running_processes
    
    log_header "PREPARING FOR REMOVAL"
    create_backup
    confirm_uninstall
    
    log_header "STOPPING SERVICES"
    stop_services
    
    log_header "REMOVING COMPONENTS"
    remove_installation_directory
    remove_user_config
    remove_user_cache
    remove_desktop_integration
    remove_system_launchers
    
    log_header "CLEANUP COMPLETE"
    show_completion_message
}

show_completion_message() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${BOLD}${BLUE}🔍 DRY RUN COMPLETE${NC}"
        echo
        log_info "This was a dry run - no files were actually removed"
        log_info "Run without --dry-run to perform actual uninstallation"
        return 0
    fi
    
    echo -e "${BOLD}${GREEN}✅ UNINSTALLATION COMPLETE${NC}"
    echo
    log_success "Shader Prediction Compiler has been successfully removed!"
    echo
    
    echo -e "${BOLD}What was removed:${NC}"
    echo "  🗂️  Application files and installation"
    if [[ "$KEEP_CONFIG" != "true" ]]; then
        echo "  ⚙️  Configuration files and settings"
    fi
    if [[ "$KEEP_CACHE" != "true" ]]; then
        echo "  💾 Cache files and compiled shaders"
    fi
    echo "  🖥️  Desktop integration and shortcuts"
    echo "  🔧 System launcher commands"
    echo
    
    if [[ "$CREATE_BACKUP" == "true" ]]; then
        echo -e "${BOLD}Backup Information:${NC}"
        echo "  📦 Backup created at: $BACKUP_DIR"
        echo "  🔄 Use backup to restore settings if you reinstall"
        echo
    fi
    
    echo -e "${BOLD}What was preserved:${NC}"
    echo "  🎮 Steam game shader caches (untouched)"
    echo "  🔗 Steam integration settings"
    echo "  📦 System packages and dependencies"
    echo "  💻 Other applications and data"
    echo
    
    echo -e "${BOLD}To reinstall in the future:${NC}"
    echo "  curl -fsSL https://raw.githubusercontent.com/$REPO_OWNER/$REPO_NAME/main/install.sh | bash"
    echo
    
    log_info "Thank you for using Shader Prediction Compiler!"
    log_info "If you have feedback, please visit: https://github.com/$REPO_OWNER/$REPO_NAME/issues"
}

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_error "Do not run the uninstaller as root!"
    log_info "The uninstaller will ask for sudo access when needed."
    exit 1
fi

# Parse command line arguments
parse_arguments "$@"

# Show banner
show_banner

# Run the uninstallation
run_uninstallation

# End of main function
}

# Call main function to prevent partial execution
main "$@"