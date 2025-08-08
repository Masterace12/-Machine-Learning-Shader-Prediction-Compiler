#!/bin/bash
# Steam Deck Optimized ML Shader Prediction Compiler Installation Script
# Designed specifically for Steam Deck hardware constraints and SteamOS
# Version: 2.0.0 - Memory Optimized Edition

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR="$HOME/.local/share/shader-prediction-compiler"
SERVICE_DIR="$HOME/.config/systemd/user"
LOG_FILE="$HOME/.local/share/shader-prediction-compiler/install.log"

# Steam Deck Detection
detect_steam_deck() {
    echo -e "${BLUE}[INFO]${NC} Detecting Steam Deck hardware..."
    
    # Check for Steam Deck specific identifiers
    if [ -f "/sys/class/dmi/id/product_name" ]; then
        PRODUCT_NAME=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "unknown")
        if [[ "$PRODUCT_NAME" == *"Jupiter"* ]] || [[ "$PRODUCT_NAME" == *"Steam Deck"* ]]; then
            echo -e "${GREEN}[SUCCESS]${NC} Detected Steam Deck hardware: $PRODUCT_NAME"
            STEAM_DECK_MODEL="LCD"  # Default to LCD, can be detected more specifically later
            return 0
        fi
    fi
    
    # Check for AMD Van Gogh APU
    if command -v lscpu >/dev/null 2>&1; then
        if lscpu | grep -q "AMD Custom APU"; then
            echo -e "${GREEN}[SUCCESS]${NC} Detected Steam Deck APU"
            STEAM_DECK_MODEL="LCD"
            return 0
        fi
    fi
    
    echo -e "${YELLOW}[WARNING]${NC} Not running on Steam Deck hardware - continuing with compatibility mode"
    STEAM_DECK_MODEL="LCD"
    return 1
}

# Check available memory and adjust installation accordingly
check_memory_constraints() {
    echo -e "${BLUE}[INFO]${NC} Checking memory constraints..."
    
    # Get available memory in MB
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
    TOTAL_MEMORY=$(free -m | awk 'NR==2{print $2}')
    
    echo -e "${BLUE}[INFO]${NC} Total memory: ${TOTAL_MEMORY}MB, Available: ${AVAILABLE_MEMORY}MB"
    
    # Adjust installation profile based on available memory
    if [ "$AVAILABLE_MEMORY" -lt 2000 ]; then
        echo -e "${YELLOW}[WARNING]${NC} Limited memory available - using minimal installation profile"
        INSTALL_PROFILE="minimal"
        export PIP_NO_CACHE_DIR=1  # Disable pip cache to save memory
    elif [ "$AVAILABLE_MEMORY" -lt 4000 ]; then
        echo -e "${BLUE}[INFO]${NC} Using optimized installation profile"
        INSTALL_PROFILE="optimized"
    else
        echo -e "${GREEN}[SUCCESS]${NC} Using full installation profile"
        INSTALL_PROFILE="full"
    fi
}

# Function to handle different package managers
install_system_dependencies() {
    echo -e "${BLUE}[INFO]${NC} Installing system dependencies..."
    
    # Try different package managers in order of preference
    if command -v pacman >/dev/null 2>&1; then
        # Arch-based system (SteamOS)
        echo -e "${BLUE}[INFO]${NC} Detected pacman (Arch/SteamOS)"
        
        # Check if system is read-only (immutable)
        if [ -f "/etc/os-release" ] && grep -q "steamos" /etc/os-release; then
            echo -e "${YELLOW}[WARNING]${NC} SteamOS immutable filesystem detected"
            echo -e "${BLUE}[INFO]${NC} Using Flatpak for dependencies where possible"
            
            # Try to install Flatpak Python runtime
            if command -v flatpak >/dev/null 2>&1; then
                flatpak install --user --assumeyes org.freedesktop.Platform.Locale 22.08 2>/dev/null || true
                flatpak install --user --assumeyes org.freedesktop.Sdk.Extension.python3 22.08 2>/dev/null || true
            fi
        else
            # Try to install via pacman (requires developer mode)
            echo -e "${BLUE}[INFO]${NC} Attempting to install via pacman..."
            if command -v sudo >/dev/null 2>&1; then
                sudo pacman -S --noconfirm python python-pip python-numpy python-scikit-learn 2>/dev/null || echo -e "${YELLOW}[WARNING]${NC} Could not install via pacman"
            fi
        fi
        
    elif command -v apt-get >/dev/null 2>&1; then
        # Debian/Ubuntu based
        echo -e "${BLUE}[INFO]${NC} Detected apt (Debian/Ubuntu)"
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-numpy python3-sklearn
        
    elif command -v dnf >/dev/null 2>&1; then
        # Fedora/RedHat based
        echo -e "${BLUE}[INFO]${NC} Detected dnf (Fedora/RHEL)"
        sudo dnf install -y python3 python3-pip python3-numpy python3-scikit-learn
        
    else
        echo -e "${YELLOW}[WARNING]${NC} No supported package manager found, relying on pip installation"
    fi
}

# Enhanced Python environment setup for Steam Deck
setup_python_environment() {
    echo -e "${BLUE}[INFO]${NC} Setting up Python environment for Steam Deck..."
    
    # Check for Python 3
    if ! command -v python3 >/dev/null 2>&1; then
        echo -e "${RED}[ERROR]${NC} Python 3 not found. Please install Python 3."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}[SUCCESS]${NC} Found Python $PYTHON_VERSION"
    
    # Check for pip3, try multiple approaches
    if ! command -v pip3 >/dev/null 2>&1; then
        echo -e "${YELLOW}[WARNING]${NC} pip3 not found, trying alternative methods..."
        
        # Try python3 -m pip
        if python3 -m pip --version >/dev/null 2>&1; then
            echo -e "${GREEN}[SUCCESS]${NC} Using python3 -m pip"
            alias pip3='python3 -m pip'
        else
            # Try to install pip
            echo -e "${BLUE}[INFO]${NC} Attempting to install pip..."
            python3 -m ensurepip --upgrade 2>/dev/null || {
                # Download get-pip.py as last resort
                echo -e "${BLUE}[INFO]${NC} Downloading get-pip.py..."
                curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
                python3 /tmp/get-pip.py --user
                rm /tmp/get-pip.py
            }
        fi
    fi
    
    # Verify pip is working
    if ! python3 -m pip --version >/dev/null 2>&1; then
        echo -e "${RED}[ERROR]${NC} Could not set up pip. Installation cannot continue."
        exit 1
    fi
    
    echo -e "${GREEN}[SUCCESS]${NC} pip is available"
}

# Install Python dependencies with memory optimization
install_python_dependencies() {
    echo -e "${BLUE}[INFO]${NC} Installing Python dependencies (profile: $INSTALL_PROFILE)..."
    
    # Create requirements file based on memory constraints
    case "$INSTALL_PROFILE" in
        "minimal")
            REQUIREMENTS_FILE="$SCRIPT_DIR/requirements-minimal.txt"
            echo -e "${BLUE}[INFO]${NC} Using minimal requirements for memory conservation"
            ;;
        "optimized")
            REQUIREMENTS_FILE="$SCRIPT_DIR/requirements-minimal.txt"
            echo -e "${BLUE}[INFO]${NC} Using optimized requirements"
            ;;
        "full")
            REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
            echo -e "${BLUE}[INFO]${NC} Using full requirements"
            ;;
    esac
    
    # Check if requirements file exists
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo -e "${YELLOW}[WARNING]${NC} Requirements file not found, using fallback installation"
        
        # Fallback minimal installation
        echo -e "${BLUE}[INFO]${NC} Installing essential packages..."
        python3 -m pip install --user --upgrade pip
        python3 -m pip install --user "numpy>=1.21.0,<1.25.0"
        python3 -m pip install --user "scikit-learn>=1.1.0,<1.4.0"
        python3 -m pip install --user "joblib>=1.1.0,<1.4.0"
        python3 -m pip install --user "psutil>=5.8.0,<6.0.0"
        python3 -m pip install --user "PyYAML>=6.0,<7.0"
        
        # Add Linux-specific packages
        if [ "$(uname)" = "Linux" ]; then
            python3 -m pip install --user "pyudev>=0.23.0" 2>/dev/null || echo -e "${YELLOW}[WARNING]${NC} Could not install pyudev"
        fi
    else
        echo -e "${BLUE}[INFO]${NC} Installing from requirements file: $REQUIREMENTS_FILE"
        
        # Install with memory optimization flags
        python3 -m pip install --user --upgrade pip
        
        # Install packages one by one to handle memory pressure
        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ "$line" =~ ^#.*$ ]] && continue
            [[ -z "$line" ]] && continue
            
            # Extract package name (before >= or ==)
            PACKAGE_NAME=$(echo "$line" | sed 's/[>=<].*//')
            
            echo -e "${BLUE}[INFO]${NC} Installing $PACKAGE_NAME..."
            python3 -m pip install --user "$line" || {
                echo -e "${YELLOW}[WARNING]${NC} Failed to install $line, continuing..."
            }
            
            # Small delay to prevent memory pressure
            sleep 1
        done < "$REQUIREMENTS_FILE"
    fi
    
    echo -e "${GREEN}[SUCCESS]${NC} Python dependencies installed"
}

# Create optimized directories for Steam Deck
create_directories() {
    echo -e "${BLUE}[INFO]${NC} Creating directory structure..."
    
    # Create main directories
    mkdir -p "$INSTALL_DIR"/{src,config,logs,models,cache,temp}
    mkdir -p "$SERVICE_DIR"
    mkdir -p "$HOME/.local/bin"
    
    # Set appropriate permissions
    chmod 755 "$INSTALL_DIR"
    chmod 700 "$INSTALL_DIR/config"
    chmod 755 "$INSTALL_DIR/logs"
    
    echo -e "${GREEN}[SUCCESS]${NC} Directory structure created"
}

# Install the shader prediction system
install_shader_system() {
    echo -e "${BLUE}[INFO]${NC} Installing shader prediction system..."
    
    # Copy core files
    if [ -d "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src" ]; then
        cp -r "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src/"* "$INSTALL_DIR/src/"
        echo -e "${GREEN}[SUCCESS]${NC} Copied shader prediction system"
    elif [ -d "$SCRIPT_DIR/src" ]; then
        cp -r "$SCRIPT_DIR/src/"* "$INSTALL_DIR/src/"
        echo -e "${GREEN}[SUCCESS]${NC} Copied shader prediction system"
    else
        echo -e "${RED}[ERROR]${NC} Could not find source files"
        exit 1
    fi
    
    # Copy configuration files
    if [ -f "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/config/steam_deck_optimized.json" ]; then
        cp "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/config/steam_deck_optimized.json" "$INSTALL_DIR/config/"
    fi
    
    # Create Steam Deck specific configuration
    cat > "$INSTALL_DIR/config/steamdeck_config.json" << EOF
{
  "predictor": {
    "model_type": "lightweight",
    "cache_size": 500,
    "max_temp": 83.0,
    "power_budget": 12.0,
    "sequence_length": 50,
    "buffer_size": 5000,
    "auto_train_interval": 1000,
    "min_training_samples": 100
  },
  "monitor_interval": 1.0,
  "thermal_protection": true,
  "battery_optimization": true,
  "performance_mode": "balanced",
  "steam_deck_model": "$STEAM_DECK_MODEL",
  "memory_optimization": true,
  "max_memory_usage_mb": 200,
  "game_profiles": {
    "1091500": {
      "name": "Cyberpunk 2077",
      "shader_complexity_modifier": 1.2,
      "thermal_limit": 80.0,
      "power_budget": 11.0,
      "cache_size": 300
    },
    "1245620": {
      "name": "ELDEN RING", 
      "shader_complexity_modifier": 1.1,
      "thermal_limit": 83.0,
      "power_budget": 12.0,
      "cache_size": 400
    }
  }
}
EOF
    
    echo -e "${GREEN}[SUCCESS]${NC} Configuration created"
}

# Create launcher script
create_launcher() {
    echo -e "${BLUE}[INFO]${NC} Creating launcher script..."
    
    cat > "$HOME/.local/bin/shader-prediction-compiler" << 'EOF'
#!/bin/bash
# Steam Deck ML Shader Prediction Compiler Launcher

INSTALL_DIR="$HOME/.local/share/shader-prediction-compiler"
CONFIG_FILE="$INSTALL_DIR/config/steamdeck_config.json"
LOG_FILE="$INSTALL_DIR/logs/shader_predictor.log"

cd "$INSTALL_DIR"

# Set PYTHONPATH
export PYTHONPATH="$INSTALL_DIR/src:$PYTHONPATH"

# Memory optimization environment variables
export PYTHONOPTIMIZE=1
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# Start the shader prediction system
python3 -O "$INSTALL_DIR/src/steam_deck_integration.py" --config "$CONFIG_FILE" "$@" 2>&1 | tee -a "$LOG_FILE"
EOF
    
    chmod +x "$HOME/.local/bin/shader-prediction-compiler"
    echo -e "${GREEN}[SUCCESS]${NC} Launcher created at ~/.local/bin/shader-prediction-compiler"
}

# Create systemd user service for background operation
create_systemd_service() {
    echo -e "${BLUE}[INFO]${NC} Creating systemd user service..."
    
    cat > "$SERVICE_DIR/shader-prediction-compiler.service" << EOF
[Unit]
Description=Steam Deck ML Shader Prediction Compiler
After=graphical-session.target

[Service]
Type=simple
ExecStart=$HOME/.local/bin/shader-prediction-compiler --daemon
Restart=on-failure
RestartSec=5
Environment=PYTHONPATH=$INSTALL_DIR/src
Environment=PYTHONOPTIMIZE=1
StandardOutput=append:$INSTALL_DIR/logs/service.log
StandardError=append:$INSTALL_DIR/logs/service.log

# Resource limits for Steam Deck
MemoryMax=300M
CPUQuota=50%

[Install]
WantedBy=default.target
EOF
    
    echo -e "${GREEN}[SUCCESS]${NC} Systemd service created"
    
    # Enable and start the service
    systemctl --user daemon-reload
    systemctl --user enable shader-prediction-compiler.service
    
    echo -e "${GREEN}[SUCCESS]${NC} Service enabled"
}

# Create desktop entry
create_desktop_entry() {
    echo -e "${BLUE}[INFO]${NC} Creating desktop entry..."
    
    DESKTOP_FILE="$HOME/.local/share/applications/shader-prediction-compiler.desktop"
    
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=ML Shader Prediction Compiler
Comment=Machine Learning-based Shader Compilation Predictor for Steam Deck
Exec=$HOME/.local/bin/shader-prediction-compiler --gui
Icon=applications-games
Terminal=false
Type=Application
Categories=Game;System;
Keywords=shader;steam;deck;gaming;ml;prediction;
StartupNotify=true
EOF
    
    chmod +x "$DESKTOP_FILE"
    echo -e "${GREEN}[SUCCESS]${NC} Desktop entry created"
}

# Validation and testing
run_validation() {
    echo -e "${BLUE}[INFO]${NC} Running validation tests..."
    
    # Test Python imports
    python3 -c "
import sys
import os
sys.path.insert(0, '$INSTALL_DIR/src')

try:
    import numpy as np
    print('✓ NumPy import successful')
except ImportError as e:
    print('✗ NumPy import failed:', e)
    exit(1)

try:
    from sklearn.ensemble import ExtraTreesRegressor
    print('✓ scikit-learn import successful')
except ImportError as e:
    print('✓ scikit-learn not available, fallback mode will be used')

try:
    import psutil
    print('✓ psutil import successful')
except ImportError as e:
    print('✓ psutil not available, basic monitoring will be used')

try:
    from ml_shader_predictor import SteamDeckMLPredictor
    print('✓ ML predictor import successful')
except ImportError as e:
    print('✗ ML predictor import failed:', e)
    exit(1)

print('✓ All validation tests passed')
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Validation tests passed"
        return 0
    else
        echo -e "${RED}[ERROR]${NC} Validation tests failed"
        return 1
    fi
}

# Performance test
run_performance_test() {
    echo -e "${BLUE}[INFO]${NC} Running performance test..."
    
    python3 -c "
import sys
import time
sys.path.insert(0, '$INSTALL_DIR/src')

from ml_shader_predictor import SteamDeckMLPredictor, ShaderFeatures

# Test predictor creation
start_time = time.time()
predictor = SteamDeckMLPredictor()
creation_time = time.time() - start_time
print(f'✓ Predictor creation time: {creation_time:.3f}s')

# Test prediction
test_features = ShaderFeatures(
    shader_hash='test_shader',
    instruction_count=150,
    complexity_score=0.6,
    stage_type='fragment',
    uses_textures=True,
    uses_uniforms=True,
    branch_count=3,
    loop_count=1,
    register_pressure=0.5,
    memory_access_pattern='textured',
    engine_type='unity',
    game_id='test_game',
    rdna2_optimal=True,
    memory_bandwidth_sensitive=False,
    thermal_sensitive=False
)

start_time = time.time()
predicted_time, confidence = predictor.predict_compilation_time(test_features)
prediction_time = time.time() - start_time

print(f'✓ Prediction time: {prediction_time*1000:.1f}ms')
print(f'✓ Predicted compilation time: {predicted_time:.1f}ms (confidence: {confidence:.2f})')

if prediction_time < 0.01:  # Less than 10ms
    print('✓ Performance test passed - predictions are fast enough for real-time use')
else:
    print('⚠ Performance warning - predictions may be too slow for real-time use')
"
    
    echo -e "${GREEN}[SUCCESS]${NC} Performance test completed"
}

# Cleanup function
cleanup_install() {
    echo -e "${BLUE}[INFO]${NC} Cleaning up temporary files..."
    
    # Clean pip cache if we disabled it
    if [ "$INSTALL_PROFILE" = "minimal" ]; then
        python3 -m pip cache purge 2>/dev/null || true
    fi
    
    # Clean temporary files
    rm -f /tmp/get-pip.py 2>/dev/null || true
    
    echo -e "${GREEN}[SUCCESS]${NC} Cleanup completed"
}

# Show post-installation information
show_post_install_info() {
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    INSTALLATION COMPLETED                       ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${BLUE}[INFO]${NC} Steam Deck ML Shader Prediction Compiler has been installed successfully!"
    echo -e ""
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  • Command line: ${GREEN}shader-prediction-compiler${NC}"
    echo -e "  • Background service: ${GREEN}systemctl --user start shader-prediction-compiler${NC}"
    echo -e "  • Desktop application: Launch from Gaming Mode or Desktop"
    echo -e ""
    echo -e "${YELLOW}Configuration:${NC}"
    echo -e "  • Config file: ${GREEN}$INSTALL_DIR/config/steamdeck_config.json${NC}"
    echo -e "  • Logs: ${GREEN}$INSTALL_DIR/logs/${NC}"
    echo -e "  • Installation profile: ${GREEN}$INSTALL_PROFILE${NC}"
    echo -e "  • Steam Deck model: ${GREEN}$STEAM_DECK_MODEL${NC}"
    echo -e ""
    echo -e "${YELLOW}System Integration:${NC}"
    echo -e "  • Systemd service: ${GREEN}Enabled${NC}"
    echo -e "  • Memory limit: ${GREEN}300MB${NC}"
    echo -e "  • CPU limit: ${GREEN}50%${NC}"
    echo -e "  • Thermal protection: ${GREEN}Enabled${NC}"
    echo -e ""
    echo -e "${BLUE}[INFO]${NC} The system will automatically start optimizing shader compilation"
    echo -e "       for your Steam Deck hardware constraints."
    echo -e ""
    echo -e "${YELLOW}To start immediately:${NC} ${GREEN}systemctl --user start shader-prediction-compiler${NC}"
    echo -e ""
}

# Main installation function
main() {
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║        Steam Deck ML Shader Prediction Compiler Installer       ║${NC}"
    echo -e "${GREEN}║                     Memory Optimized Edition                     ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
    
    # Create log file directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Start logging
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1
    
    echo "Installation started at $(date)"
    echo "Script directory: $SCRIPT_DIR"
    
    # Run installation steps
    detect_steam_deck
    check_memory_constraints
    install_system_dependencies
    setup_python_environment
    install_python_dependencies
    create_directories
    install_shader_system
    create_launcher
    create_systemd_service
    create_desktop_entry
    
    # Validation and testing
    if run_validation; then
        run_performance_test
        cleanup_install
        show_post_install_info
        
        echo -e "\n${GREEN}[SUCCESS]${NC} Installation completed successfully!"
        echo -e "Installation log saved to: $LOG_FILE"
        
        exit 0
    else
        echo -e "\n${RED}[ERROR]${NC} Installation failed validation. Check the log for details."
        echo -e "Log file: $LOG_FILE"
        exit 1
    fi
}

# Trap cleanup on exit
trap cleanup_install EXIT

# Run main installation
main "$@"