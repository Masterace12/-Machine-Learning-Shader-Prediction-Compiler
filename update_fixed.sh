#!/usr/bin/env bash

# Steam Deck ML Shader Prediction Compiler - Update Script
# Automatically checks for installation and updates if needed
# Version: 2.1.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration - MUST match install.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_NAME="shader-predict-compile"
GITHUB_REPO="https://github.com/Masterace12/Machine-Learning-Shader-Prediction-Compiler"
GITHUB_API="https://api.github.com/repos/Masterace12/Machine-Learning-Shader-Prediction-Compiler"

# Installation paths - MUST match install.sh exactly
INSTALL_DIR="${HOME}/.local/${PROJECT_NAME}"
CONFIG_DIR="${HOME}/.config/${PROJECT_NAME}"
CACHE_DIR="${HOME}/.cache/${PROJECT_NAME}"
BACKUP_DIR=""

# Update options
FORCE_UPDATE=false
SKIP_BACKUP=false
SKIP_SERVICES=false
DRY_RUN=false
VERBOSE=false
AUTO_UPDATE=true
ROLLBACK=false

# Version information
CURRENT_VERSION=""
LATEST_VERSION=""
UPDATE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Temporary directories
TEMP_DIR="/tmp/${PROJECT_NAME}_update_${UPDATE_TIMESTAMP}"
DOWNLOAD_DIR="${TEMP_DIR}/download"
BACKUP_MANIFEST="${TEMP_DIR}/backup_manifest.json"

# Logging
LOG_FILE="${TEMP_DIR}/update.log"
ERROR_LOG="${TEMP_DIR}/update_errors.log"

# Service management
SERVICES_TO_RESTART=()
SERVICES_STOPPED=false

# Installation status
INSTALLATION_FOUND=false
UPDATE_NEEDED=false

# Trap for cleanup
trap 'cleanup_on_exit' EXIT
trap 'handle_error $? $LINENO' ERR

# Error handling
handle_error() {
    local exit_code=$1
    local line_number=$2
    
    error "Update failed at line $line_number (exit code: $exit_code)"
    
    if [[ -f "$ERROR_LOG" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: Line $line_number, Exit code: $exit_code" >> "$ERROR_LOG"
    fi
    
    if [[ "$SERVICES_STOPPED" == "true" ]]; then
        warn "Attempting to restart services..."
        restart_services
    fi
    
    error "Update failed. Check logs at: $LOG_FILE"
    
    if [[ -d "$BACKUP_DIR" ]]; then
        echo
        echo -e "${YELLOW}A backup was created at: ${BACKUP_DIR}${NC}"
        echo -e "${YELLOW}You can restore it manually if needed.${NC}"
    fi
    
    exit $exit_code
}

# Cleanup function
cleanup_on_exit() {
    if [[ "$DRY_RUN" == "true" ]]; then
        debug "Dry run completed, cleaning up temporary files..."
    fi
    
    # Remove temporary download directory
    if [[ -d "$DOWNLOAD_DIR" ]]; then
        rm -rf "$DOWNLOAD_DIR"
    fi
    
    # Keep logs for debugging
    if [[ -f "$LOG_FILE" ]] && [[ ! -s "$LOG_FILE" ]]; then
        rm -f "$LOG_FILE"
    fi
}

# Logging functions
log() {
    local message="$*"
    echo -e "${GREEN}[INFO]${NC} $message"
    if [[ -f "$LOG_FILE" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $message" >> "$LOG_FILE"
    fi
}

warn() {
    local message="$*"
    echo -e "${YELLOW}[WARN]${NC} $message"
    if [[ -f "$LOG_FILE" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] $message" >> "$LOG_FILE"
    fi
}

error() {
    local message="$*"
    echo -e "${RED}[ERROR]${NC} $message" >&2
    if [[ -f "$LOG_FILE" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $message" >> "$LOG_FILE"
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $message" >> "$ERROR_LOG"
    fi
}

debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*"
        if [[ -f "$LOG_FILE" ]]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [DEBUG] $*" >> "$LOG_FILE"
        fi
    fi
}

success() {
    echo -e "${GREEN}✓${NC} $*"
    if [[ -f "$LOG_FILE" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] $*" >> "$LOG_FILE"
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE_UPDATE=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-services)
                SKIP_SERVICES=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --no-auto)
                AUTO_UPDATE=false
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Show usage
show_usage() {
    echo "ML Shader Prediction Compiler - Update Script"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Default behavior:"
    echo "  - Checks for existing installation at: ${INSTALL_DIR}"
    echo "  - Compares versions and updates if needed"
    echo "  - No action taken if already up-to-date"
    echo
    echo "Options:"
    echo "  --force          Force update even if versions match"
    echo "  --skip-backup    Skip creating backup (not recommended)"
    echo "  --skip-services  Don't stop/restart services during update"
    echo "  --dry-run        Simulate update without making changes"
    echo "  --no-auto        Prompt before updating (disable auto-update)"
    echo "  --rollback       Rollback to previous version from backup"
    echo "  --verbose        Enable verbose output"
    echo "  --help           Show this help message"
    echo
    echo "Examples:"
    echo "  $0               # Auto-check and update if needed"
    echo "  $0 --no-auto     # Check and prompt before updating"
    echo "  $0 --force       # Force reinstall latest version"
    echo "  $0 --dry-run     # Test update process without changes"
    echo "  $0 --rollback    # Restore previous version"
}

# Find installation - matches install.sh paths exactly
find_installation() {
    debug "Searching for existing installation..."
    
    # Check multiple possible installation paths from install.sh
    local possible_dirs=(
        "${HOME}/.local/share/${PROJECT_NAME}"     # User-space install location
        "${HOME}/.local/${PROJECT_NAME}"           # Standard install location
        "/opt/${PROJECT_NAME}"                     # System-wide install
        "/usr/local/${PROJECT_NAME}"               # Alternative system install
    )
    
    for dir in "${possible_dirs[@]}"; do
        debug "Checking install directory: $dir"
        if [[ -d "$dir" ]] && [[ -f "$dir/main.py" ]]; then
            INSTALL_DIR="$dir"
            INSTALLATION_FOUND=true
            debug "Found installation at: $INSTALL_DIR"
            
            # Update other directories based on found installation
            if [[ "$INSTALL_DIR" == *"/share/"* ]]; then
                # User-space installation paths
                CONFIG_DIR="${HOME}/.config/${PROJECT_NAME}"
                CACHE_DIR="${HOME}/.cache/${PROJECT_NAME}"
            else
                # Standard installation paths
                CONFIG_DIR="${HOME}/.config/${PROJECT_NAME}"
                CACHE_DIR="${HOME}/.cache/${PROJECT_NAME}"
            fi
            
            return 0
        fi
    done
    
    debug "No installation found at any expected location"
    INSTALLATION_FOUND=false
    return 1
}

# Get current version
get_current_version() {
    CURRENT_VERSION="unknown"
    
    # Try to get version from various sources
    if [[ -f "$INSTALL_DIR/version.txt" ]]; then
        CURRENT_VERSION=$(cat "$INSTALL_DIR/version.txt" 2>/dev/null || echo "unknown")
    elif [[ -f "$CONFIG_DIR/config.json" ]]; then
        CURRENT_VERSION=$(python3 -c "import json; print(json.load(open('$CONFIG_DIR/config.json')).get('version', 'unknown'))" 2>/dev/null || echo "unknown")
    elif [[ -f "$INSTALL_DIR/.git/HEAD" ]]; then
        CURRENT_VERSION=$(cd "$INSTALL_DIR" && git describe --tags --always 2>/dev/null || echo "unknown")
    fi
    
    debug "Current version: $CURRENT_VERSION"
}

# Check for latest version
check_latest_version() {
    debug "Checking for latest version from GitHub..."
    
    # Try to get latest release from GitHub API
    if command -v curl &>/dev/null; then
        local api_response
        api_response=$(curl -s "${GITHUB_API}/releases/latest" 2>/dev/null || echo "{}")
        
        if [[ -n "$api_response" ]] && [[ "$api_response" != "{}" ]]; then
            LATEST_VERSION=$(echo "$api_response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('tag_name', 'main'))" 2>/dev/null || echo "main")
        else
            # Fallback to main branch
            LATEST_VERSION="main"
        fi
    else
        LATEST_VERSION="main"
    fi
    
    debug "Latest version: $LATEST_VERSION"
}

# Determine if update is needed
check_update_needed() {
    # Force update if requested
    if [[ "$FORCE_UPDATE" == "true" ]]; then
        UPDATE_NEEDED=true
        return 0
    fi
    
    # Check if versions differ
    if [[ "$CURRENT_VERSION" == "$LATEST_VERSION" ]] && [[ "$LATEST_VERSION" != "main" ]]; then
        UPDATE_NEEDED=false
    else
        UPDATE_NEEDED=true
    fi
    
    return 0
}

# Stop services
stop_services() {
    if [[ "$SKIP_SERVICES" == "true" ]]; then
        debug "Skipping service management (--skip-services)"
        return 0
    fi
    
    log "Checking for running services..."
    
    # Check systemd user services
    local services=(
        "shader-predict-compile.service"
        "shader-predict-compile-maintenance.timer"
        "ml-shader-predictor.service"
        "ml-shader-predictor-maintenance.timer"
    )
    
    for service in "${services[@]}"; do
        if systemctl --user is-active --quiet "$service" 2>/dev/null; then
            SERVICES_TO_RESTART+=("$service")
            log "Stopping $service..."
            
            if [[ "$DRY_RUN" != "true" ]]; then
                systemctl --user stop "$service" || warn "Failed to stop $service"
            fi
        fi
    done
    
    if [[ ${#SERVICES_TO_RESTART[@]} -gt 0 ]]; then
        SERVICES_STOPPED=true
        success "Services stopped"
    else
        debug "No services were running"
    fi
}

# Restart services
restart_services() {
    if [[ "$SKIP_SERVICES" == "true" ]] || [[ ${#SERVICES_TO_RESTART[@]} -eq 0 ]]; then
        return 0
    fi
    
    log "Restarting services..."
    
    for service in "${SERVICES_TO_RESTART[@]}"; do
        log "Starting $service..."
        
        if [[ "$DRY_RUN" != "true" ]]; then
            systemctl --user start "$service" || warn "Failed to start $service"
        fi
    done
    
    # Reload systemd if service files were updated
    if [[ "$DRY_RUN" != "true" ]]; then
        systemctl --user daemon-reload
    fi
    
    SERVICES_STOPPED=false
    success "Services restarted"
}

# Create backup
create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        warn "Skipping backup (--skip-backup)"
        return 0
    fi
    
    log "Creating backup..."
    
    BACKUP_DIR="${HOME}/.local/share/${PROJECT_NAME}_backups/backup_${UPDATE_TIMESTAMP}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        debug "Would create backup at: $BACKUP_DIR"
        return 0
    fi
    
    mkdir -p "$BACKUP_DIR"
    
    # Ensure backup subdirectories exist
    mkdir -p "$BACKUP_DIR/install" || {
        error "Failed to create backup install directory"
        return 1
    }
    
    # Backup main installation
    if [[ -d "$INSTALL_DIR" ]]; then
        log "Backing up installation directory..."
        cp -r "$INSTALL_DIR"/* "$BACKUP_DIR/install/" 2>/dev/null || warn "Installation backup may be incomplete"
    fi
    
    # Backup configuration
    if [[ -d "$CONFIG_DIR" ]]; then
        mkdir -p "$BACKUP_DIR/config" || {
            warn "Failed to create backup config directory"
        }
        log "Backing up configuration..."
        cp -r "$CONFIG_DIR"/* "$BACKUP_DIR/config/" 2>/dev/null || warn "Configuration backup may be incomplete"
    fi
    
    # Create backup manifest
    cat > "$BACKUP_DIR/manifest.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "version": "$CURRENT_VERSION",
    "install_dir": "$INSTALL_DIR",
    "config_dir": "$CONFIG_DIR",
    "update_version": "$LATEST_VERSION"
}
EOF
    
    # Keep only last 3 backups
    local backup_parent=$(dirname "$BACKUP_DIR")
    if [[ -d "$backup_parent" ]]; then
        local old_backups=($(ls -t "$backup_parent" | grep "^backup_" | tail -n +4))
        for old_backup in "${old_backups[@]}"; do
            debug "Removing old backup: $old_backup"
            rm -rf "$backup_parent/$old_backup"
        done
    fi
    
    success "Backup created at: $BACKUP_DIR"
}

# Download latest version (FIXED: directory creation logic)
download_latest() {
    log "Downloading latest version..."
    
    # Create temporary directory first
    mkdir -p "$TEMP_DIR" || {
        error "Failed to create temporary directory: $TEMP_DIR"
        return 1
    }
    
    if [[ "$DRY_RUN" == "true" ]]; then
        debug "Would download from: $GITHUB_REPO to $DOWNLOAD_DIR"
        return 0
    fi
    
    # Clone or download the repository
    if command -v git &>/dev/null; then
        log "Using git to fetch latest version..."
        # Clone directly to download dir (git will create it)
        git clone --depth 1 "$GITHUB_REPO" "$DOWNLOAD_DIR" 2>/dev/null || {
            error "Failed to clone repository"
            return 1
        }
    else
        error "Git is required for updating"
        return 1
    fi
    
    # Verify download succeeded
    if [[ ! -d "$DOWNLOAD_DIR" ]]; then
        error "Download failed - directory not created"
        return 1
    fi
    
    success "Downloaded latest version"
}

# Update files (FIXED: directory verification)
update_files() {
    log "Updating files..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        debug "Would update files from $DOWNLOAD_DIR to $INSTALL_DIR"
        return 0
    fi
    
    # Verify source directory exists
    if [[ ! -d "$DOWNLOAD_DIR" ]]; then
        error "Download directory not found: $DOWNLOAD_DIR"
        return 1
    fi
    
    # Ensure install directory exists
    mkdir -p "$INSTALL_DIR" || {
        error "Failed to create install directory: $INSTALL_DIR"
        return 1
    }
    
    # Update main files
    local update_items=(
        "main.py"
        "src"
        "security"
        "scripts"
        "requirements-optimized.txt"
        "requirements-legacy.txt"
        "README.md"
        "update.sh"
        "install.sh"
        "uninstall.sh"
    )
    
    # Update each item
    for item in "${update_items[@]}"; do
        if [[ -e "$DOWNLOAD_DIR/$item" ]]; then
            debug "Updating $item..."
            
            # Remove old version if it exists
            if [[ -e "$INSTALL_DIR/$item" ]]; then
                rm -rf "$INSTALL_DIR/$item"
            fi
            
            # Ensure parent directory exists for the item
            local item_parent="$INSTALL_DIR/$(dirname "$item")"
            if [[ "$(dirname "$item")" != "." ]]; then
                mkdir -p "$item_parent"
            fi
            
            # Copy new version
            cp -r "$DOWNLOAD_DIR/$item" "$INSTALL_DIR/$item" || {
                warn "Failed to update $item"
            }
        fi
    done
    
    # Update version file
    echo "$LATEST_VERSION" > "$INSTALL_DIR/version.txt"
    
    # Preserve executable permissions
    local executables=(
        "main.py"
        "install.sh"
        "uninstall.sh"
        "update.sh"
    )
    
    for exe in "${executables[@]}"; do
        if [[ -f "$INSTALL_DIR/$exe" ]]; then
            chmod +x "$INSTALL_DIR/$exe"
        fi
    done
    
    success "Files updated"
}

# Update Python dependencies
update_dependencies() {
    log "Updating Python dependencies..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        debug "Would update Python dependencies in virtual environment"
        return 0
    fi
    
    # Check if virtual environment exists
    local venv_path="$INSTALL_DIR/venv"
    
    if [[ ! -d "$venv_path" ]]; then
        warn "Virtual environment not found, skipping dependency update"
        return 0
    fi
    
    # Activate virtual environment
    source "$venv_path/bin/activate"
    
    # Upgrade pip
    log "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel &>/dev/null
    
    # Update requirements
    local requirements_file="$INSTALL_DIR/requirements-optimized.txt"
    
    if [[ ! -f "$requirements_file" ]]; then
        requirements_file="$INSTALL_DIR/requirements-legacy.txt"
    fi
    
    if [[ -f "$requirements_file" ]]; then
        log "Installing updated dependencies..."
        python -m pip install --upgrade -r "$requirements_file" &>/dev/null || {
            warn "Some dependencies failed to update"
        }
    fi
    
    # Deactivate virtual environment
    deactivate
    
    success "Dependencies updated"
}

# Perform rollback
perform_rollback() {
    log "Performing rollback to previous version..."
    
    # Find most recent backup
    local backup_parent="${HOME}/.local/share/${PROJECT_NAME}_backups"
    
    if [[ ! -d "$backup_parent" ]]; then
        error "No backups found"
        exit 1
    fi
    
    local latest_backup=$(ls -t "$backup_parent" | grep "^backup_" | head -1)
    
    if [[ -z "$latest_backup" ]]; then
        error "No backup available for rollback"
        exit 1
    fi
    
    BACKUP_DIR="$backup_parent/$latest_backup"
    
    log "Rolling back from backup: $BACKUP_DIR"
    
    # Stop services
    stop_services
    
    # Restore installation
    if [[ -d "$BACKUP_DIR/install" ]]; then
        log "Restoring installation directory..."
        rm -rf "$INSTALL_DIR"
        cp -r "$BACKUP_DIR/install" "$INSTALL_DIR"
    fi
    
    # Restore configuration
    if [[ -d "$BACKUP_DIR/config" ]]; then
        log "Restoring configuration..."
        rm -rf "$CONFIG_DIR"
        cp -r "$BACKUP_DIR/config" "$CONFIG_DIR"
    fi
    
    # Restart services
    restart_services
    
    success "Rollback completed successfully"
}

# Verify update
verify_update() {
    log "Verifying update..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        debug "Would verify update installation"
        return 0
    fi
    
    # Check main script exists
    if [[ ! -f "$INSTALL_DIR/main.py" ]]; then
        error "Main script not found after update"
        return 1
    fi
    
    # Test import of main modules
    if [[ -d "$INSTALL_DIR/venv" ]]; then
        source "$INSTALL_DIR/venv/bin/activate"
        
        python3 -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
try:
    import main
    print('✓ Main module loads successfully')
except ImportError as e:
    print(f'✗ Failed to load main module: {e}')
    sys.exit(1)
" || {
            deactivate
            error "Update verification failed"
            return 1
        }
        
        deactivate
    fi
    
    success "Update verified successfully"
}

# Perform the update
perform_update() {
    log "Starting update from $CURRENT_VERSION to $LATEST_VERSION"
    
    # Stop services
    stop_services
    
    # Create backup
    create_backup
    
    # Download latest version
    download_latest
    
    # Update files
    update_files
    
    # Update dependencies
    update_dependencies
    
    # Verify update
    verify_update
    
    # Restart services
    restart_services
}

# Show update status
show_update_status() {
    echo
    echo -e "${BLUE}════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  ML Shader Prediction Compiler - Update Status${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════${NC}"
    echo
    
    if [[ "$INSTALLATION_FOUND" == "false" ]]; then
        echo -e "${RED}✗ No installation found${NC}"
        echo
        echo -e "${YELLOW}Please install the program first:${NC}"
        echo -e "  ${CYAN}cd $(dirname "$0")${NC}"
        echo -e "  ${CYAN}./install.sh${NC}"
        echo
        return 1
    fi
    
    echo -e "${GREEN}✓ Installation found${NC}"
    echo -e "  Location: ${YELLOW}$INSTALL_DIR${NC}"
    echo -e "  Current version: ${YELLOW}$CURRENT_VERSION${NC}"
    echo -e "  Latest version: ${GREEN}$LATEST_VERSION${NC}"
    echo
    
    if [[ "$UPDATE_NEEDED" == "true" ]]; then
        echo -e "${YELLOW}⚡ Update available!${NC}"
        
        if [[ "$AUTO_UPDATE" == "true" ]]; then
            echo -e "${BLUE}→ Auto-updating in 5 seconds...${NC}"
            echo -e "  Press Ctrl+C to cancel"
            echo
            
            # Countdown
            for i in {5..1}; do
                echo -ne "\r  Starting in ${i}..."
                sleep 1
            done
            echo -ne "\r                    \r"
            
            return 0
        else
            echo
            read -p "Do you want to update now? [Y/n]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                return 0
            else
                echo -e "${YELLOW}Update cancelled${NC}"
                return 1
            fi
        fi
    else
        echo -e "${GREEN}✓ Already up-to-date!${NC}"
        echo -e "  No update needed"
        echo
        echo -e "${BLUE}Options:${NC}"
        echo -e "  Use ${CYAN}$0 --force${NC} to reinstall"
        echo -e "  Use ${CYAN}$0 --rollback${NC} to restore previous version"
        echo
        return 1
    fi
}

# Show completion message
show_completion() {
    echo
    echo -e "${GREEN}════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✨ Update Completed Successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════${NC}"
    echo
    echo -e "${BLUE}Update Summary:${NC}"
    echo -e "  • Previous: ${YELLOW}$CURRENT_VERSION${NC}"
    echo -e "  • Current:  ${GREEN}$LATEST_VERSION${NC}"
    echo -e "  • Backup:   ${CYAN}$BACKUP_DIR${NC}"
    echo
    
    if [[ ${#SERVICES_TO_RESTART[@]} -gt 0 ]]; then
        echo -e "${BLUE}Services restarted:${NC}"
        for service in "${SERVICES_TO_RESTART[@]}"; do
            echo -e "  • ${GREEN}✓${NC} $service"
        done
        echo
    fi
    
    echo -e "${PURPLE}What's next:${NC}"
    echo -e "  • Check status: ${YELLOW}shader-predict-status${NC}"
    echo -e "  • View logs:    ${YELLOW}journalctl --user -u shader-predict-compile${NC}"
    echo -e "  • Rollback:     ${YELLOW}$0 --rollback${NC}"
    echo
    echo -e "${CYAN}Thank you for keeping your installation up to date! 🎮${NC}"
}

# Main update function
main() {
    # Parse arguments
    parse_arguments "$@"
    
    # Create temp directory with proper error handling
    mkdir -p "$TEMP_DIR" || {
        echo -e "${RED}[ERROR]${NC} Failed to create temporary directory: $TEMP_DIR" >&2
        exit 1
    }
    
    # Create log file with parent directory
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE" || {
        echo -e "${YELLOW}[WARN]${NC} Failed to create log file, continuing without logging" >&2
    }
    
    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        echo -e "${CYAN}🔄 ML Shader Prediction Compiler - Rollback${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════${NC}"
        echo
        
        if find_installation; then
            perform_rollback
        else
            error "No installation found to rollback"
            exit 1
        fi
        exit 0
    fi
    
    # Check for installation
    if ! find_installation; then
        show_update_status
        exit 1
    fi
    
    # Get version information
    get_current_version
    check_latest_version
    
    # Check if update is needed
    check_update_needed
    
    # Show status and decide whether to update
    if show_update_status; then
        # Perform update
        perform_update
        
        # Show completion message
        show_completion
    fi
}

# Run main function
main "$@"