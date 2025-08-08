#!/bin/bash
# Steam Deck Shader Prediction Compilation - Optimized Installer
# Fixes all identified issues and provides enhanced installation experience

set -euo pipefail

# Enhanced Colors and Formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Logging Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_debug() { echo -e "${PURPLE}[DEBUG]${NC} $1"; }
log_step() { echo -e "${CYAN}${BOLD}[STEP]${NC} $1"; }

# Configuration
readonly INSTALL_DIR="/opt/shader-predict-compile"
readonly CONFIG_DIR="$HOME/.config/shader-predict-compile"
readonly CACHE_DIR="$HOME/.cache/shader-predict-compile"
readonly LOG_FILE="/tmp/shader-predict-install.log"

# Steam Deck Detection
detect_steam_deck() {
    local is_steam_deck=false
    local model="Unknown"
    
    # Check DMI product name
    if [[ -f "/sys/devices/virtual/dmi/id/product_name" ]]; then
        local product_name=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "")
        if [[ "$product_name" == *"Jupiter"* ]]; then
            is_steam_deck=true
            
            # Try to detect LCD vs OLED
            if [[ -f "/sys/class/drm/card0/device/apu_model" ]]; then
                local apu_model=$(cat /sys/class/drm/card0/device/apu_model 2>/dev/null || echo "")
                if [[ "$apu_model" == *"Van Gogh"* ]]; then
                    model="LCD"
                elif [[ "$apu_model" == *"Phoenix"* ]]; then
                    model="OLED"
                fi
            fi
            
            # Fallback detection
            if [[ "$model" == "Unknown" ]]; then
                if command -v xrandr >/dev/null 2>&1; then
                    local resolution=$(xrandr 2>/dev/null | grep "connected primary" || echo "")
                    if [[ "$resolution" == *"1280x800"* ]]; then
                        model="LCD"
                    elif [[ "$resolution" == *"1280x720"* ]]; then
                        model="OLED"
                    else
                        model="LCD"  # Default assumption
                    fi
                else
                    model="LCD"  # Conservative default
                fi
            fi
        fi
    fi
    
    echo "$is_steam_deck,$model"
}

# Enhanced dependency checking
check_dependencies() {
    log_step "Checking system dependencies..."
    
    local missing_deps=()
    local python_version=""
    
    # Check Python 3
    if command -v python3 >/dev/null 2>&1; then
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_success "Python 3 found: $python_version"
    else
        missing_deps+=("python3")
    fi
    
    # Check pip
    if ! command -v pip3 >/dev/null 2>&1; then
        missing_deps+=("python3-pip")
    fi
    
    # Check git (for potential updates)
    if ! command -v git >/dev/null 2>&1; then
        log_warning "Git not found - updates from repository will not be available"
    fi
    
    # Steam Deck specific checks
    local steam_deck_info=$(detect_steam_deck)
    local is_steam_deck=$(echo "$steam_deck_info" | cut -d',' -f1)
    local deck_model=$(echo "$steam_deck_info" | cut -d',' -f2)
    
    if [[ "$is_steam_deck" == "true" ]]; then
        log_success "Steam Deck detected: $deck_model model"
        
        # Check SteamOS version
        if [[ -f "/etc/os-release" ]]; then
            local steamos_version=$(grep 'VERSION_ID=' /etc/os-release | cut -d'"' -f2)
            log_info "SteamOS version: $steamos_version"
            
            # Check if version is sufficient (3.7+)
            local major=$(echo "$steamos_version" | cut -d'.' -f1)
            local minor=$(echo "$steamos_version" | cut -d'.' -f2)
            if [[ "$major" -gt 3 ]] || [[ "$major" -eq 3 && "$minor" -ge 7 ]]; then
                log_success "SteamOS version is compatible"
            else
                log_warning "SteamOS version may not be fully supported (recommend 3.7+)"
            fi
        fi
        
        # Check for Fossilize
        if command -v fossilize-replay >/dev/null 2>&1; then
            log_success "Fossilize found - enhanced shader caching available"
        else
            log_warning "Fossilize not found - basic caching only"
        fi
        
        # Check Vulkan support
        if [[ -d "/usr/share/vulkan/icd.d" ]] && [[ -n "$(ls -A /usr/share/vulkan/icd.d 2>/dev/null)" ]]; then
            log_success "Vulkan ICD found"
        else
            log_warning "Vulkan ICD not detected - checking alternative locations"
        fi
        
        # Export Steam Deck model for later use
        export STEAM_DECK_MODEL="$deck_model"
        
    else
        log_info "Not running on Steam Deck - generic Linux configuration will be used"
        export STEAM_DECK_MODEL="Generic"
    fi
    
    # Install missing dependencies
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_warning "Missing dependencies: ${missing_deps[*]}"
        log_info "Attempting to install missing dependencies..."
        
        # Try different package managers
        if command -v pacman >/dev/null 2>&1; then
            # Arch Linux / SteamOS
            sudo pacman -S --noconfirm "${missing_deps[@]}" || {
                log_error "Failed to install dependencies with pacman"
                return 1
            }
        elif command -v apt >/dev/null 2>&1; then
            # Debian/Ubuntu
            sudo apt update && sudo apt install -y "${missing_deps[@]}" || {
                log_error "Failed to install dependencies with apt"
                return 1
            }
        elif command -v dnf >/dev/null 2>&1; then
            # Fedora
            sudo dnf install -y "${missing_deps[@]}" || {
                log_error "Failed to install dependencies with dnf"
                return 1
            }
        else
            log_error "No supported package manager found"
            log_error "Please install manually: ${missing_deps[*]}"
            return 1
        fi
        
        log_success "Dependencies installed successfully"
    fi
    
    return 0
}

# Create optimized virtual environment
create_venv() {
    log_step "Setting up optimized Python environment..."
    
    local venv_dir="$INSTALL_DIR/venv"
    
    # Create virtual environment
    python3 -m venv "$venv_dir" --prompt "shader-predict"
    
    # Activate virtual environment
    source "$venv_dir/bin/activate"
    
    # Upgrade pip to latest version
    pip install --upgrade pip wheel setuptools
    
    log_success "Virtual environment created: $venv_dir"
}

# Install Python dependencies with optimizations
install_python_deps() {
    log_step "Installing optimized Python dependencies..."
    
    source "$INSTALL_DIR/venv/bin/activate"
    
    # Install core dependencies with Steam Deck optimizations
    cat > /tmp/requirements_optimized.txt << 'EOF'
# Core ML and Data Science (Steam Deck optimized versions)
numpy>=1.21.0,<2.0.0
scikit-learn>=1.1.0,<1.5.0
pandas>=1.4.0,<2.2.0
scipy>=1.8.0,<1.12.0
joblib>=1.1.0,<1.4.0

# Networking and P2P  
aiohttp>=3.8.0,<4.0.0
cryptography>=3.4.8,<42.0.0

# System Integration (Steam Deck compatible)
psutil>=5.8.0,<6.0.0

# Steam Deck Hardware (Linux only)
pyudev>=0.23.2,<1.0.0; platform_system=="Linux"
dbus-python>=1.2.18,<2.0.0; platform_system=="Linux"

# Configuration and Validation
requests>=2.28.0,<3.0.0
PyYAML>=6.0,<7.0.0

# Minimal GUI support
PyGObject>=3.40.0; platform_system=="Linux"
EOF

    # Install with optimizations for Steam Deck
    pip install -r /tmp/requirements_optimized.txt --prefer-binary --no-cache-dir
    
    # Try to install PyTorch for GPU acceleration (optional)
    if [[ "${STEAM_DECK_MODEL:-Generic}" != "Generic" ]]; then
        log_info "Attempting to install PyTorch for GPU acceleration..."
        pip install torch>=1.12.0 --index-url https://download.pytorch.org/whl/rocm5.4.2 --no-cache-dir 2>/dev/null || {
            log_warning "PyTorch GPU acceleration not available - using CPU inference"
            pip install torch>=1.12.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir || {
                log_warning "PyTorch installation failed - ML features may be limited"
            }
        }
    fi
    
    rm -f /tmp/requirements_optimized.txt
    log_success "Python dependencies installed"
}

# Setup configuration files
setup_config() {
    log_step "Configuring system for Steam Deck optimization..."
    
    # Create configuration directories
    mkdir -p "$CONFIG_DIR" "$CACHE_DIR"
    
    # Copy optimized configuration
    if [[ -f "shader-predict-compile/config/steam_deck_optimized.json" ]]; then
        cp "shader-predict-compile/config/steam_deck_optimized.json" "$CONFIG_DIR/config.json"
        log_success "Optimized configuration installed"
    else
        log_warning "Optimized config not found - creating default"
        cat > "$CONFIG_DIR/config.json" << EOF
{
  "system": {
    "steam_deck_model": "${STEAM_DECK_MODEL:-Generic}",
    "auto_optimize": true
  },
  "compilation": {
    "max_threads": 4,
    "memory_limit_mb": 2048,
    "thermal_aware": true
  },
  "ml_models": {
    "type": "ensemble",
    "cache_size": 2000
  }
}
EOF
    fi
    
    # Setup environment variables for RADV optimization
    local env_file="$CONFIG_DIR/radv_env.sh"
    cat > "$env_file" << 'EOF'
#!/bin/bash
# RADV optimizations for Steam Deck
export RADV_PERFTEST=aco,nggc,sam
export RADV_DEBUG=noshaderdb,nocompute  
export MESA_VK_DEVICE_SELECT=1002:163f
export RADV_LOWER_DISCARD_TO_DEMOTE=1
EOF
    chmod +x "$env_file"
    
    log_success "Configuration files created"
}

# Install systemd service
install_service() {
    log_step "Installing systemd service..."
    
    local service_file="/etc/systemd/system/shader-predict-compile.service"
    
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=Steam Deck Shader Prediction Compilation Service
After=graphical-session.target steam.service
Wants=steam.service

[Service]
Type=notify
User=deck
Group=deck
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/src/main.py --daemon
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=5
Environment=HOME=/home/deck

# Resource limits optimized for Steam Deck
CPUQuota=25%
MemoryMax=1G
MemoryHigh=512M
IOWeight=10
TasksMax=50

# Security settings
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=$CONFIG_DIR $CACHE_DIR /tmp
PrivateTmp=yes

# Allow hardware access
DeviceAllow=/dev/dri/card0 rw
DeviceAllow=/dev/dri/renderD128 rw

# Environment variables for RADV
EnvironmentFile=$CONFIG_DIR/radv_env.sh

[Install]
WantedBy=graphical-session.target
EOF

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable shader-predict-compile.service
    
    log_success "Systemd service installed and enabled"
}

# Create desktop entry
create_desktop_entry() {
    log_step "Creating desktop integration..."
    
    local desktop_file="$HOME/.local/share/applications/shader-predict-compile.desktop"
    mkdir -p "$(dirname "$desktop_file")"
    
    cat > "$desktop_file" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Shader Prediction Compiler
Comment=Intelligent shader compilation optimization for Steam Deck
Exec=$INSTALL_DIR/launcher.sh
Icon=$INSTALL_DIR/icon.png
Terminal=false
Categories=Game;Utility;System;
Keywords=shader;steam;deck;compilation;optimization;
StartupWMClass=shader-predict-compile
EOF

    # Create launcher script
    cat > "$INSTALL_DIR/launcher.sh" << EOF
#!/bin/bash
# Steam Deck Shader Prediction Compiler Launcher

# Source RADV optimizations
source "$CONFIG_DIR/radv_env.sh" 2>/dev/null || true

# Activate virtual environment
source "$INSTALL_DIR/venv/bin/activate"

# Launch application
cd "$INSTALL_DIR"
python src/main.py "\$@"
EOF
    
    chmod +x "$INSTALL_DIR/launcher.sh"
    
    # Update desktop database
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    
    log_success "Desktop integration created"
}

# Validate installation
validate_installation() {
    log_step "Validating installation..."
    
    local errors=0
    
    # Check Python environment
    if [[ -f "$INSTALL_DIR/venv/bin/python" ]]; then
        log_success "Python environment: OK"
    else
        log_error "Python environment: FAILED"
        ((errors++))
    fi
    
    # Check main script
    if [[ -f "$INSTALL_DIR/src/main.py" ]]; then
        log_success "Main application: OK"
    else
        log_error "Main application: MISSING"
        ((errors++))
    fi
    
    # Check configuration
    if [[ -f "$CONFIG_DIR/config.json" ]]; then
        log_success "Configuration: OK"
    else
        log_error "Configuration: MISSING"
        ((errors++))
    fi
    
    # Check systemd service
    if systemctl --user is-enabled shader-predict-compile.service >/dev/null 2>&1; then
        log_success "Systemd service: ENABLED"
    else
        log_warning "Systemd service: NOT ENABLED"
    fi
    
    # Test import of key modules
    source "$INSTALL_DIR/venv/bin/activate"
    if python -c "import numpy, sklearn, psutil; print('Core dependencies: OK')" 2>/dev/null; then
        log_success "Python dependencies: OK"
    else
        log_error "Python dependencies: FAILED"
        ((errors++))
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "Installation validation: PASSED"
        return 0
    else
        log_error "Installation validation: FAILED ($errors errors)"
        return 1
    fi
}

# Main installation function
main() {
    echo -e "${CYAN}${BOLD}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║         Steam Deck Shader Prediction Compiler Installer      ║
║                     Optimized Version 1.1                    ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "Do not run this installer as root!"
        log_info "Run as regular user (deck on Steam Deck)"
        exit 1
    fi
    
    # Detect Steam Deck and show info
    local steam_deck_info=$(detect_steam_deck)
    local is_steam_deck=$(echo "$steam_deck_info" | cut -d',' -f1)
    local deck_model=$(echo "$steam_deck_info" | cut -d',' -f2)
    
    if [[ "$is_steam_deck" == "true" ]]; then
        log_success "Steam Deck $deck_model detected - using optimized configuration"
    else
        log_info "Generic Linux system detected"
    fi
    
    # Start logging
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
    
    log_info "Installation started: $(date)"
    log_info "Installation log: $LOG_FILE"
    
    # Run installation steps
    if ! check_dependencies; then
        log_error "Dependency check failed"
        exit 1
    fi
    
    # Ensure we're in the right directory
    if [[ ! -d "shader-prediction-compilation-main" ]] && [[ ! -d "shader-predict-compile" ]]; then
        log_error "Project directory not found!"
        log_info "Make sure you extracted the ZIP file and are running from the correct directory"
        exit 1
    fi
    
    # Navigate to project directory  
    if [[ -d "shader-prediction-compilation-main" ]]; then
        cd shader-prediction-compilation-main
    fi
    
    if [[ -d "shader-predict-compile" ]]; then
        cd shader-predict-compile
    fi
    
    # Create installation directory
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown -R "$USER:$USER" "$INSTALL_DIR"
    
    # Copy source files
    log_step "Copying application files..."
    cp -r . "$INSTALL_DIR/"
    
    # Setup environment
    create_venv
    install_python_deps
    setup_config
    
    # Install system integration
    install_service
    create_desktop_entry
    
    # Validate everything works
    if validate_installation; then
        echo -e "${GREEN}${BOLD}"
        cat << 'EOF'

╔═══════════════════════════════════════════════════════════════╗
║                    🎉 INSTALLATION SUCCESSFUL! 🎉             ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Shader Prediction Compiler is now installed and optimized   ║
║  for your Steam Deck!                                        ║
║                                                               ║
║  🎮 The service will start automatically                      ║
║  📱 Find it in Applications → Games                          ║
║  ⚙️  Configuration: ~/.config/shader-predict-compile/        ║
║  📊 Logs: journalctl -u shader-predict-compile               ║
║                                                               ║
║  🚀 Enjoy enhanced gaming performance!                       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
        echo -e "${NC}"
        
        log_info "Installation completed successfully at: $(date)"
        
        # Ask if user wants to start the service now
        read -p "Start the shader prediction service now? [Y/n]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            sudo systemctl start shader-predict-compile.service
            log_success "Service started!"
        fi
        
    else
        log_error "Installation validation failed - please check the errors above"
        exit 1
    fi
}

# Run main function
main "$@"