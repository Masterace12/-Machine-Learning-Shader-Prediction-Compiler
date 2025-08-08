#!/bin/bash
#
# Steam Deck Optimized Installation Script
# Handles immutable filesystem, systemd services, and Flatpak packaging
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/steamdeck-deployment/scripts/deck-install.sh | bash
#
# Advanced options:
#   bash deck-install.sh --flatpak-only    # Install only Flatpak version
#   bash deck-install.sh --service-only    # Install only systemd service
#   bash deck-install.sh --dev             # Install development version
#

set -euo pipefail

# ============================================================================
# CONSTANTS
# ============================================================================

readonly SCRIPT_VERSION="2.0.0"
readonly DECK_USER="deck"
readonly INSTALL_BASE="/home/${DECK_USER}/.local/share/shader-predict-compile"
readonly CONFIG_DIR="/home/${DECK_USER}/.config/shader-predict-compile"
readonly CACHE_DIR="/home/${DECK_USER}/.cache/shader-predict-compile"
readonly SYSTEMD_USER_DIR="/home/${DECK_USER}/.config/systemd/user"
readonly FLATPAK_REPO="flathub"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Installation modes
INSTALL_MODE="full"  # full, flatpak-only, service-only
DEV_MODE=false
FORCE_INSTALL=false
SKIP_DEPENDENCIES=false

# ============================================================================
# STEAM DECK DETECTION
# ============================================================================

detect_steam_deck() {
    local is_deck=false
    
    # Check for Steam Deck hardware identifiers
    if [[ -f /sys/devices/virtual/dmi/id/product_name ]]; then
        local product_name=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "")
        if [[ "$product_name" == *"Jupiter"* ]] || [[ "$product_name" == *"Galileo"* ]]; then
            is_deck=true
        fi
    fi
    
    # Check for SteamOS
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        if [[ "${ID:-}" == "steamos" ]] || [[ "${NAME:-}" == *"SteamOS"* ]]; then
            is_deck=true
        fi
    fi
    
    # Check for gamescope session
    if pgrep -x "gamescope" > /dev/null 2>&1; then
        is_deck=true
    fi
    
    if [[ "$is_deck" == true ]]; then
        echo -e "${GREEN}✓${NC} Steam Deck detected!"
        return 0
    else
        echo -e "${YELLOW}!${NC} Not running on Steam Deck - installing in compatibility mode"
        return 1
    fi
}

# ============================================================================
# THERMAL MANAGEMENT
# ============================================================================

check_thermal_state() {
    local temp_file="/sys/class/thermal/thermal_zone0/temp"
    local gpu_temp_file="/sys/class/hwmon/hwmon4/temp1_input"
    
    if [[ -f "$temp_file" ]]; then
        local cpu_temp=$(cat "$temp_file")
        cpu_temp=$((cpu_temp / 1000))
        
        if [[ $cpu_temp -gt 75 ]]; then
            echo -e "${YELLOW}!${NC} High CPU temperature detected (${cpu_temp}°C)"
            echo -e "${YELLOW}!${NC} Waiting for system to cool down..."
            while [[ $cpu_temp -gt 70 ]]; do
                sleep 5
                cpu_temp=$(cat "$temp_file")
                cpu_temp=$((cpu_temp / 1000))
            done
        fi
    fi
}

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

install_dependencies() {
    echo -e "${BLUE}→${NC} Installing dependencies..."
    
    # Check if running in read-only filesystem
    if [[ -w /usr ]]; then
        # Mutable filesystem - can use pacman
        local packages=(
            "python"
            "python-pip"
            "python-numpy"
            "python-scikit-learn"
            "python-pyqt6"
            "vulkan-tools"
            "vulkan-headers"
        )
        
        for pkg in "${packages[@]}"; do
            if ! pacman -Q "$pkg" &>/dev/null; then
                echo -e "${BLUE}→${NC} Installing $pkg..."
                sudo pacman -S --noconfirm "$pkg" || true
            fi
        done
    else
        # Immutable filesystem - use Flatpak for dependencies
        echo -e "${YELLOW}!${NC} Read-only filesystem detected - using Flatpak runtime"
        
        # Ensure Flatpak is available
        if ! command -v flatpak &>/dev/null; then
            echo -e "${RED}✗${NC} Flatpak not found - cannot continue"
            exit 1
        fi
        
        # Add Flathub if not present
        flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo --user
        
        # Install required runtimes
        flatpak install -y --user flathub org.kde.Platform//6.5 || true
        flatpak install -y --user flathub org.kde.Sdk//6.5 || true
    fi
    
    # Install Python packages in user space
    echo -e "${BLUE}→${NC} Installing Python dependencies..."
    pip install --user --upgrade pip
    pip install --user \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        numpy scikit-learn joblib \
        psutil pyqt6 \
        aiohttp asyncio \
        cryptography
}

# ============================================================================
# SYSTEMD SERVICE INSTALLATION
# ============================================================================

install_systemd_service() {
    echo -e "${BLUE}→${NC} Installing systemd services..."
    
    # Create systemd user directory
    mkdir -p "$SYSTEMD_USER_DIR"
    
    # Copy service files
    cp ../systemd/*.service "$SYSTEMD_USER_DIR/"
    cp ../systemd/*.timer "$SYSTEMD_USER_DIR/"
    
    # Adjust paths in service files
    sed -i "s|/home/deck/|/home/${DECK_USER}/|g" "$SYSTEMD_USER_DIR"/*.service
    sed -i "s|/home/deck/|/home/${DECK_USER}/|g" "$SYSTEMD_USER_DIR"/*.timer
    
    # Reload systemd
    systemctl --user daemon-reload
    
    # Enable services (but don't start yet)
    systemctl --user enable shader-predict.service
    systemctl --user enable shader-predict-thermal.timer
    
    # Enable Game Mode integration if available
    if pgrep -x "gamescope" > /dev/null 2>&1; then
        systemctl --user enable shader-predict-gamemode.service
    fi
    
    echo -e "${GREEN}✓${NC} Systemd services installed"
}

# ============================================================================
# FLATPAK INSTALLATION
# ============================================================================

install_flatpak_app() {
    echo -e "${BLUE}→${NC} Building and installing Flatpak..."
    
    # Create build directory
    local build_dir="/tmp/shader-predict-flatpak-$$"
    mkdir -p "$build_dir"
    
    # Copy Flatpak files
    cp -r ../flatpak/* "$build_dir/"
    cp -r ../../src "$build_dir/"
    cp -r ../../scripts "$build_dir/"
    cp -r ../../config "$build_dir/"
    
    # Build Flatpak
    cd "$build_dir"
    flatpak-builder --user --install --force-clean build-dir com.shaderpredict.Compiler.yml
    
    # Clean up
    cd - > /dev/null
    rm -rf "$build_dir"
    
    echo -e "${GREEN}✓${NC} Flatpak application installed"
}

# ============================================================================
# APPLICATION INSTALLATION
# ============================================================================

install_application() {
    echo -e "${BLUE}→${NC} Installing application files..."
    
    # Create directories
    mkdir -p "$INSTALL_BASE"/{src,scripts,config,data,logs}
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$CACHE_DIR"
    
    # Copy application files
    cp -r ../../src/* "$INSTALL_BASE/src/"
    cp -r ../../scripts/* "$INSTALL_BASE/scripts/"
    cp -r ../../config/* "$INSTALL_BASE/config/"
    
    # Copy security modules
    if [[ -d "../../security" ]]; then
        cp -r ../../security "$INSTALL_BASE/"
    fi
    
    # Make scripts executable
    chmod +x "$INSTALL_BASE/scripts"/*.sh
    
    # Create default configuration
    cat > "$CONFIG_DIR/settings.json" << EOF
{
    "mode": "auto",
    "ml_enabled": true,
    "p2p_enabled": true,
    "thermal_monitoring": true,
    "battery_aware": true,
    "max_cpu_percent": 10,
    "max_memory_mb": 500,
    "cache_size_gb": 2,
    "gamemode_throttle": true,
    "auto_start": false,
    "log_level": "info"
}
EOF
    
    # Create launcher script
    cat > "$INSTALL_BASE/launch.sh" << 'EOF'
#!/bin/bash
export SHADER_PREDICT_HOME="$(dirname "$0")"
export PYTHONPATH="${SHADER_PREDICT_HOME}/src:${PYTHONPATH}"
cd "$SHADER_PREDICT_HOME"
exec python3 src/main.py "$@"
EOF
    chmod +x "$INSTALL_BASE/launch.sh"
    
    echo -e "${GREEN}✓${NC} Application files installed"
}

# ============================================================================
# STEAM INTEGRATION
# ============================================================================

setup_steam_integration() {
    echo -e "${BLUE}→${NC} Setting up Steam integration..."
    
    local steam_dir="/home/${DECK_USER}/.local/share/Steam"
    
    if [[ -d "$steam_dir" ]]; then
        # Create compatibility tool entry
        local compat_dir="$steam_dir/compatibilitytools.d/shader-predict"
        mkdir -p "$compat_dir"
        
        cat > "$compat_dir/compatibilitytool.vdf" << EOF
"compatibilitytools"
{
    "compat_tools"
    {
        "shader-predict"
        {
            "install_path" "."
            "display_name" "Shader Prediction Compiler"
            "from_oslist" "windows"
            "to_oslist" "linux"
        }
    }
}
EOF
        
        # Create launch wrapper
        cat > "$compat_dir/shader-predict" << EOF
#!/bin/bash
# Enable shader prediction for this game
export SHADER_PREDICT_ENABLED=1
export SHADER_PREDICT_GAME_ID="\$SteamAppId"
systemctl --user start shader-predict.service 2>/dev/null || true
exec "\$@"
EOF
        chmod +x "$compat_dir/shader-predict"
        
        echo -e "${GREEN}✓${NC} Steam integration configured"
    else
        echo -e "${YELLOW}!${NC} Steam directory not found - skipping integration"
    fi
}

# ============================================================================
# POST-INSTALLATION
# ============================================================================

post_install_setup() {
    echo -e "${BLUE}→${NC} Running post-installation setup..."
    
    # Initialize ML models
    python3 "$INSTALL_BASE/src/initialize_models.py" 2>/dev/null || true
    
    # Create desktop entry for KDE
    local desktop_file="/home/${DECK_USER}/.local/share/applications/shader-predict.desktop"
    mkdir -p "$(dirname "$desktop_file")"
    cp ../flatpak/com.shaderpredict.Compiler.desktop "$desktop_file"
    
    # Update desktop database
    update-desktop-database "/home/${DECK_USER}/.local/share/applications" 2>/dev/null || true
    
    # Set up log rotation
    cat > "/home/${DECK_USER}/.config/logrotate.d/shader-predict" << EOF
${INSTALL_BASE}/logs/*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    maxsize 10M
}
EOF
    
    echo -e "${GREEN}✓${NC} Post-installation setup complete"
}

# ============================================================================
# MAIN INSTALLATION
# ============================================================================

main() {
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}    Shader Prediction Compiler - Steam Deck Installer${NC}"
    echo -e "${BOLD}${BLUE}    Version: ${SCRIPT_VERSION}${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════${NC}\n"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --flatpak-only)
                INSTALL_MODE="flatpak-only"
                shift
                ;;
            --service-only)
                INSTALL_MODE="service-only"
                shift
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            --force)
                FORCE_INSTALL=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPENDENCIES=true
                shift
                ;;
            *)
                echo -e "${RED}✗${NC} Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check if running as deck user
    if [[ "$USER" != "$DECK_USER" ]] && [[ "$USER" != "root" ]]; then
        echo -e "${YELLOW}!${NC} Not running as deck user - some features may not work"
    fi
    
    # Detect Steam Deck
    detect_steam_deck || true
    
    # Check thermal state
    check_thermal_state
    
    # Install based on mode
    case "$INSTALL_MODE" in
        full)
            [[ "$SKIP_DEPENDENCIES" == false ]] && install_dependencies
            install_application
            install_systemd_service
            install_flatpak_app
            setup_steam_integration
            post_install_setup
            ;;
        flatpak-only)
            [[ "$SKIP_DEPENDENCIES" == false ]] && install_dependencies
            install_flatpak_app
            ;;
        service-only)
            [[ "$SKIP_DEPENDENCIES" == false ]] && install_dependencies
            install_application
            install_systemd_service
            setup_steam_integration
            post_install_setup
            ;;
    esac
    
    # Success message
    echo -e "\n${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${GREEN}    Installation Complete!${NC}"
    echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${GREEN}✓${NC} Shader Prediction Compiler has been installed successfully!"
    echo -e "\nTo get started:"
    echo -e "  • GUI: Launch from KDE Discover or application menu"
    echo -e "  • Service: ${BLUE}systemctl --user start shader-predict.service${NC}"
    echo -e "  • Status: ${BLUE}systemctl --user status shader-predict.service${NC}"
    echo -e "  • Logs: ${BLUE}journalctl --user -u shader-predict -f${NC}"
    echo -e "\nFor Game Mode, the service will automatically activate."
    echo -e "Configuration: ${CONFIG_DIR}/settings.json"
}

# Run main function
main "$@"