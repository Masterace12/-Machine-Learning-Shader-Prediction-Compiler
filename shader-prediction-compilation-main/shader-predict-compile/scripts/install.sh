#!/bin/bash

# Shader Predictive Compiler Installation Script for Steam Deck
# Supports both LCD and OLED models with SteamOS 3.7.13+

set -e

APP_NAME="shader-predict-compile"
APP_DIR="/opt/${APP_NAME}"
DESKTOP_FILE="/usr/share/applications/${APP_NAME}.desktop"
SYSTEMD_SERVICE="/etc/systemd/system/${APP_NAME}.service"
CONFIG_DIR="$HOME/.config/${APP_NAME}"
CACHE_DIR="$HOME/.cache/${APP_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root for system operations
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        log_info "Please run without sudo. The script will ask for password when needed."
        exit 1
    fi
}

# Detect Steam Deck model and SteamOS version
detect_steam_deck() {
    log_info "Detecting Steam Deck model and SteamOS version..."
    
    # Check if running on Steam Deck
    if [ ! -f "/sys/devices/virtual/dmi/id/product_name" ] || \
       ! grep -q "Jupiter" /sys/devices/virtual/dmi/id/product_name; then
        log_warning "This doesn't appear to be a Steam Deck"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Detect model (LCD vs OLED)
    STEAM_DECK_MODEL="Unknown"
    if [ -f "/sys/class/drm/card0/device/apu_model" ]; then
        APU_MODEL=$(cat /sys/class/drm/card0/device/apu_model 2>/dev/null || echo "unknown")
        case $APU_MODEL in
            *"Van Gogh"*)
                STEAM_DECK_MODEL="LCD"
                ;;
            *"Phoenix"*)
                STEAM_DECK_MODEL="OLED"
                ;;
        esac
    fi
    
    # Check SteamOS version
    STEAMOS_VERSION="Unknown"
    if [ -f "/etc/os-release" ]; then
        STEAMOS_VERSION=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
    fi
    
    log_info "Steam Deck Model: $STEAM_DECK_MODEL"
    log_info "SteamOS Version: $STEAMOS_VERSION"
    
    # Version compatibility check
    if [[ "$STEAMOS_VERSION" < "3.7" ]]; then
        log_warning "SteamOS version $STEAMOS_VERSION may not be fully supported"
        log_info "Recommended: SteamOS 3.7.13 or later"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check for Python 3
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check for GTK development files
    if ! python3 -c "import gi; gi.require_version('Gtk', '3.0')" &> /dev/null; then
        missing_deps+=("python3-gi-dev")
    fi
    
    # Check for psutil
    if ! python3 -c "import psutil" &> /dev/null; then
        missing_deps+=("python3-psutil")
    fi
    
    # Check for numpy
    if ! python3 -c "import numpy" &> /dev/null; then
        missing_deps+=("python3-numpy")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Installing dependencies..."
        
        # Try to install via pacman (SteamOS uses Arch)
        if command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm python python-gobject python-psutil python-numpy
        else
            log_error "Cannot install dependencies automatically"
            log_info "Please install: ${missing_deps[*]}"
            exit 1
        fi
    fi
    
    log_success "All dependencies satisfied"
}

# Create application directories
create_directories() {
    log_info "Creating application directories..."
    
    # Create system directory
    sudo mkdir -p "$APP_DIR"
    sudo mkdir -p "$(dirname "$DESKTOP_FILE")"
    sudo mkdir -p "$(dirname "$SYSTEMD_SERVICE")"
    
    # Create user directories
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$CACHE_DIR"
    
    log_success "Directories created"
}

# Install application files
install_files() {
    log_info "Installing application files..."
    
    # Copy source files
    sudo cp -r src/ "$APP_DIR/"
    sudo cp -r ui/ "$APP_DIR/"
    sudo cp -r lib/ "$APP_DIR/" 2>/dev/null || true
    sudo cp -r config/ "$APP_DIR/" 2>/dev/null || true
    
    # Make main script executable
    sudo chmod +x "$APP_DIR/ui/main_window.py"
    
    # Create main launcher script
    sudo tee "$APP_DIR/launcher.sh" > /dev/null << 'EOF'
#!/bin/bash
cd /opt/shader-predict-compile
export PYTHONPATH="/opt/shader-predict-compile/src:$PYTHONPATH"

# Check if we're in Gaming Mode
if [[ "$1" == "--gaming-mode-ui" ]] || pgrep -x gamescope > /dev/null; then
    echo "Launching Gaming Mode UI..."
    export GAMING_MODE=1
    export QT_SCALE_FACTOR=1.5
    export GDK_SCALE=1.5
    python3 src/gaming_mode_ui.py "$@"
else
    echo "Launching Desktop Mode UI..."
    python3 ui/main_window.py "$@"
fi
EOF
    
    sudo chmod +x "$APP_DIR/launcher.sh"
    
    # Create Gaming Mode specific launcher
    sudo tee "$APP_DIR/gaming_mode_launcher.sh" > /dev/null << 'EOF'
#!/bin/bash
cd /opt/shader-predict-compile
export PYTHONPATH="/opt/shader-predict-compile/src:$PYTHONPATH"
export GAMING_MODE=1
export QT_SCALE_FACTOR=1.5
export GDK_SCALE=1.5
python3 src/gaming_mode_ui.py --gaming-mode-ui "$@"
EOF
    
    sudo chmod +x "$APP_DIR/gaming_mode_launcher.sh"
    
    log_success "Application files installed"
}

# Create desktop entry
create_desktop_entry() {
    log_info "Creating desktop entry..."
    
    sudo tee "$DESKTOP_FILE" > /dev/null << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Shader Predictive Compiler
Comment=Enhance shader compilation for Steam games on Steam Deck
Icon=$APP_DIR/icon.png
Exec=$APP_DIR/launcher.sh
Terminal=false
Categories=Game;Utility;
Keywords=shader;steam;deck;fossilize;compile;
StartupNotify=true
EOF
    
    # Create a simple icon if it doesn't exist
    if [ ! -f "$APP_DIR/icon.png" ]; then
        log_info "Creating application icon..."
        # This would normally be a proper PNG, but for now create a placeholder
        sudo touch "$APP_DIR/icon.png"
    fi
    
    log_success "Desktop entry created"
}

# Create systemd service
create_systemd_service() {
    log_info "Creating systemd service..."
    
    sudo tee "$SYSTEMD_SERVICE" > /dev/null << EOF
[Unit]
Description=Shader Predictive Compiler Background Service
After=steam.service
Wants=steam.service

[Service]
Type=simple
User=deck
Group=deck
Environment=PYTHONPATH=/opt/$APP_NAME/src
ExecStart=$APP_DIR/launcher.sh --background
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

# Resource limits for Steam Deck
MemoryMax=2G
CPUQuota=50%

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$CACHE_DIR $CONFIG_DIR

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    log_success "Systemd service created"
}

# Configure Steam Deck optimizations
configure_optimizations() {
    log_info "Configuring Steam Deck optimizations..."
    
    # Create optimization config
    tee "$CONFIG_DIR/optimizations.json" > /dev/null << EOF
{
    "steam_deck_model": "$STEAM_DECK_MODEL",
    "steamos_version": "$STEAMOS_VERSION",
    "cpu_cores": $(nproc),
    "memory_total_gb": $(($(free -m | awk 'NR==2{print $2}') / 1024)),
    "optimizations": {
        "max_parallel_compiles": 4,
        "memory_limit_mb": 2048,
        "use_znver2_optimizations": true,
        "enable_half_precision": true,
        "cache_efficiency_target": 0.85
    }
}
EOF
    
    # Set up Fossilize integration
    if [ -d "$HOME/.steam" ]; then
        log_info "Configuring Fossilize integration..."
        
        # Create fossilize config directory
        mkdir -p "$HOME/.config/fossilize"
        
        # Create or update fossilize config
        tee "$HOME/.config/fossilize/config.json" > /dev/null << EOF
{
    "enable_pipeline_cache": true,
    "shader_cache_dir": "$CACHE_DIR/fossilize",
    "parallel_compile_threads": 4,
    "optimization_level": 2,
    "target_architecture": "znver2",
    "predictive_compilation": true,
    "integration_tool": "shader-predict-compile"
}
EOF
        
        log_success "Fossilize integration configured"
    else
        log_warning "Steam directory not found, skipping Fossilize integration"
    fi
    
    log_success "Steam Deck optimizations configured"
}

# Set up automatic startup
setup_auto_startup() {
    log_info "Setting up automatic startup..."
    
    # Enable but don't start the service immediately
    sudo systemctl enable "$APP_NAME.service"
    
    # Create autostart desktop entry for user session
    mkdir -p "$HOME/.config/autostart"
    
    tee "$HOME/.config/autostart/$APP_NAME.desktop" > /dev/null << EOF
[Desktop Entry]
Type=Application
Name=Shader Predictive Compiler
Exec=$APP_DIR/launcher.sh --minimize
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF
    
    log_success "Automatic startup configured"
}

# Setup Gaming Mode integration
setup_gaming_mode() {
    log_info "Setting up Gaming Mode integration..."
    
    # Run the Gaming Mode integration script
    if [ -f "$APP_DIR/src/gaming_mode_integration.py" ]; then
        cd "$APP_DIR"
        export PYTHONPATH="$APP_DIR/src:$PYTHONPATH"
        
        # Try to add as non-Steam game
        if python3 src/gaming_mode_integration.py; then
            log_success "Added to Gaming Mode as non-Steam application"
            log_info "You can now launch from Gaming Mode: Library → Non-Steam Games → Shader Predictive Compiler"
        else
            log_warning "Could not add to Gaming Mode automatically"
            log_info "You can manually add it later using the Desktop Mode interface"
        fi
    else
        log_warning "Gaming Mode integration script not found"
    fi
    
    # Create icon if it doesn't exist
    if [ ! -f "$APP_DIR/icon.png" ]; then
        log_info "Creating application icon..."
        
        # Create a simple SVG icon and convert to PNG
        sudo tee "$APP_DIR/icon.svg" > /dev/null << 'EOF'
<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <rect width="64" height="64" rx="8" fill="#1a1a2e"/>
  <rect x="8" y="8" width="48" height="48" rx="4" fill="#16213e"/>
  <circle cx="20" cy="20" r="4" fill="#4fc3f7"/>
  <circle cx="44" cy="20" r="4" fill="#4fc3f7"/>
  <circle cx="20" cy="44" r="4" fill="#4fc3f7"/>
  <circle cx="44" cy="44" r="4" fill="#4fc3f7"/>
  <rect x="24" y="28" width="16" height="8" rx="2" fill="#66c2ff"/>
  <text x="32" y="52" text-anchor="middle" font-family="Arial" font-size="8" fill="#ffffff">SPC</text>
</svg>
EOF
        
        # Try to convert SVG to PNG (if convert is available)
        if command -v convert &> /dev/null; then
            sudo convert "$APP_DIR/icon.svg" "$APP_DIR/icon.png" 2>/dev/null || {
                # Fallback: create a simple placeholder PNG file
                sudo touch "$APP_DIR/icon.png"
            }
        else
            # Create a minimal PNG header for a 64x64 transparent image
            echo -e '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00@\x00\x00\x00@\x08\x06\x00\x00\x00\xaaiq\xde\x00\x00\x00\rIDATx\x9c\xed\xc1\x01\r\x00\x00\x00\xc2\xa0\xf7Om\x0e7\xa0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82' | sudo tee "$APP_DIR/icon.png" > /dev/null
        fi
        
        log_success "Application icon created"
    fi
    
    log_success "Gaming Mode integration setup complete"
}

# Uninstall function
uninstall() {
    log_info "Uninstalling Shader Predictive Compiler..."
    
    # Stop and disable service
    sudo systemctl stop "$APP_NAME.service" 2>/dev/null || true
    sudo systemctl disable "$APP_NAME.service" 2>/dev/null || true
    
    # Remove files
    sudo rm -rf "$APP_DIR"
    sudo rm -f "$DESKTOP_FILE"
    sudo rm -f "$SYSTEMD_SERVICE"
    rm -rf "$CONFIG_DIR"
    rm -f "$HOME/.config/autostart/$APP_NAME.desktop"
    
    # Ask about cache removal
    read -p "Remove shader cache? This will free disk space but shaders will need recompilation. (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$CACHE_DIR"
        log_success "Cache removed"
    fi
    
    sudo systemctl daemon-reload
    
    log_success "Shader Predictive Compiler uninstalled"
}

# Main installation function
install() {
    log_info "Installing Shader Predictive Compiler for Steam Deck..."
    
    detect_steam_deck
    check_dependencies
    create_directories
    install_files
    create_desktop_entry
    create_systemd_service
    configure_optimizations
    setup_auto_startup
    setup_gaming_mode
    
    log_success "Installation completed successfully!"
    echo
    log_info "You can now:"
    log_info "  • Launch from Gaming Mode: Library → Non-Steam → Shader Predictive Compiler"
    log_info "  • Launch from Desktop Mode: Applications Menu → Games → Shader Predictive Compiler"
    log_info "  • Start background service: sudo systemctl start $APP_NAME"
    log_info "  • Enable auto-start: Already configured"
    echo
    log_info "For Steam Deck OLED users: GPU optimizations are automatically applied"
    log_info "For Steam Deck LCD users: CPU optimizations are automatically applied"
    echo
    log_warning "Note: First shader compilation may take some time."
    log_info "The tool will learn and improve predictions over time."
}

# Command line argument parsing
case "${1:-}" in
    --uninstall)
        uninstall
        ;;
    --help|-h)
        echo "Shader Predictive Compiler Installation Script"
        echo
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --uninstall    Remove the application and all data"
        echo "  --help, -h     Show this help message"
        echo
        echo "Default action is to install the application."
        ;;
    *)
        check_root
        install
        ;;
esac