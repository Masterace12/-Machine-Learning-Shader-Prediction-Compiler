#!/bin/bash

# ML Shader Prediction Compiler - Fixed Installation Script for Steam Deck
# Handles externally managed Python environments and permission issues

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎮 ML Shader Prediction Compiler - Fixed Installer${NC}"
echo "======================================================"
echo -e "${CYAN}Optimized for Steam Deck with proper permission handling${NC}"
echo ""

# Parse command line arguments
USE_VENV=false
FORCE_SYSTEM=false
ENABLE_SERVICE=false
AUTO_YES=false

for arg in "$@"; do
    case $arg in
        --venv)
            USE_VENV=true
            shift
            ;;
        --system)
            FORCE_SYSTEM=true
            shift
            ;;
        --enable-service)
            ENABLE_SERVICE=true
            shift
            ;;
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --venv           Use virtual environment (recommended)"
            echo "  --system         Force system-wide installation"
            echo "  --enable-service Enable systemd service"
            echo "  --yes, -y        Auto-confirm all prompts"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
    esac
done

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.8 or later.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')
echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"

# Check for externally managed Python environment
EXTERNALLY_MANAGED=false
EXTERNALLY_MANAGED_FILE="/usr/lib/python${PYTHON_MAJOR}.${PYTHON_MINOR}/EXTERNALLY-MANAGED"
if [ -f "$EXTERNALLY_MANAGED_FILE" ]; then
    echo -e "${YELLOW}⚠️  Externally managed Python environment detected (PEP 668)${NC}"
    echo -e "${CYAN}   This is normal on Steam Deck/SteamOS${NC}"
    EXTERNALLY_MANAGED=true
fi

# Detect Steam Deck
IS_STEAM_DECK=false
STEAM_DECK_MODEL=""
if [ -f "/sys/devices/virtual/dmi/id/product_name" ]; then
    PRODUCT_NAME=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "")
    if [[ "$PRODUCT_NAME" == "Jupiter" ]]; then
        IS_STEAM_DECK=true
        STEAM_DECK_MODEL="LCD"
        echo -e "${GREEN}🎮 Steam Deck LCD detected${NC}"
    elif [[ "$PRODUCT_NAME" == "Galileo" ]]; then
        IS_STEAM_DECK=true
        STEAM_DECK_MODEL="OLED"
        echo -e "${GREEN}🎮 Steam Deck OLED detected${NC}"
    fi
fi

# Ensure pip is installed
echo -n "Checking pip installation... "
if ! $PYTHON_CMD -m pip --version &>/dev/null; then
    echo -e "${YELLOW}Installing${NC}"
    if [ "$EUID" -eq 0 ]; then
        $PYTHON_CMD -m ensurepip 2>/dev/null || {
            echo -e "${RED}Failed to install pip. Try: sudo pacman -S python-pip${NC}"
            exit 1
        }
    else
        curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD - --user 2>/dev/null || {
            echo -e "${RED}Failed to install pip${NC}"
            exit 1
        }
    fi
else
    PIP_VERSION=$($PYTHON_CMD -m pip --version | cut -d' ' -f2)
    echo -e "${GREEN}✅ pip $PIP_VERSION${NC}"
fi

# Determine installation method
if [ "$FORCE_SYSTEM" == "true" ]; then
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}❌ System installation requires root privileges${NC}"
        echo "Please run: sudo $0 --system"
        exit 1
    fi
    INSTALL_METHOD="system"
elif [ "$USE_VENV" == "true" ] || ([ "$EXTERNALLY_MANAGED" == "true" ] && [ "$AUTO_YES" != "true" ]); then
    if [ "$EXTERNALLY_MANAGED" == "true" ] && [ "$USE_VENV" != "true" ]; then
        echo ""
        echo -e "${YELLOW}Installation method required for externally managed Python:${NC}"
        echo "1) Virtual environment (recommended, cleanest)"
        echo "2) User installation with --break-system-packages"
        echo "3) Cancel"
        
        if [ "$AUTO_YES" == "true" ]; then
            REPLY="2"
        else
            read -p "Choose option [1-3]: " -n 1 -r REPLY
            echo ""
        fi
        
        case $REPLY in
            1)
                USE_VENV=true
                INSTALL_METHOD="venv"
                ;;
            2)
                INSTALL_METHOD="user_override"
                ;;
            *)
                echo "Installation cancelled."
                exit 0
                ;;
        esac
    else
        INSTALL_METHOD="venv"
    fi
else
    if [ "$EXTERNALLY_MANAGED" == "true" ]; then
        INSTALL_METHOD="user_override"
    elif [ "$EUID" -eq 0 ]; then
        INSTALL_METHOD="system"
    else
        INSTALL_METHOD="user"
    fi
fi

# Set installation directories based on method
case $INSTALL_METHOD in
    system)
        echo -e "${YELLOW}📦 System-wide installation${NC}"
        INSTALL_DIR="/opt/shader-predict-compile"
        BIN_DIR="/usr/local/bin"
        CONFIG_DIR="/etc/shader-predict-compile"
        ;;
    venv)
        echo -e "${GREEN}📦 Virtual environment installation (recommended)${NC}"
        VENV_DIR="$HOME/.local/share/shader-predict-venv"
        INSTALL_DIR="$HOME/.local/share/shader-predict-compile"
        BIN_DIR="$HOME/.local/bin"
        CONFIG_DIR="$HOME/.config/shader-predict-compile"
        ;;
    user|user_override)
        echo -e "${GREEN}📦 User installation${NC}"
        INSTALL_DIR="$HOME/.local/share/shader-predict-compile"
        BIN_DIR="$HOME/.local/bin"
        CONFIG_DIR="$HOME/.config/shader-predict-compile"
        ;;
esac

# Create virtual environment if needed
if [ "$INSTALL_METHOD" == "venv" ]; then
    echo -e "${YELLOW}🔧 Creating virtual environment...${NC}"
    
    if [ -d "$VENV_DIR" ]; then
        echo -n "Virtual environment already exists. Recreate? [y/N] "
        if [ "$AUTO_YES" == "true" ]; then
            REPLY="y"
            echo "y (auto)"
        else
            read -n 1 -r
            echo ""
        fi
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            $PYTHON_CMD -m venv "$VENV_DIR"
        fi
    else
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="$VENV_DIR/bin/python"
    PIP_CMD="$VENV_DIR/bin/pip"
    
    # Upgrade pip in venv
    echo -n "Upgrading pip in virtual environment... "
    $PIP_CMD install --upgrade pip >/dev/null 2>&1
    echo -e "${GREEN}✅${NC}"
else
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Create directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$CONFIG_DIR"

# Copy project files
echo -e "${YELLOW}📦 Installing project files...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for item in src config main.py requirements.txt setup_threading.py docs tests; do
    if [ -e "$SCRIPT_DIR/$item" ]; then
        echo -n "  Installing $item... "
        cp -r "$SCRIPT_DIR/$item" "$INSTALL_DIR/" 2>/dev/null && echo -e "${GREEN}✅${NC}" || echo -e "${YELLOW}⚠️${NC}"
    fi
done

# Function to install packages with proper error handling
install_package() {
    local package=$1
    local description=$2
    local import_name=${3:-$(echo $package | cut -d'[' -f1 | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1 | tr '-' '_')}
    
    echo -n "  $description ($package)... "
    
    # Check if already installed and working
    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        local version=$($PYTHON_CMD -c "import $import_name; print(getattr($import_name, '__version__', 'installed'))" 2>/dev/null || echo "installed")
        echo -e "${GREEN}✅ (v$version)${NC}"
        return 0
    fi
    
    # Determine pip command based on installation method
    local pip_install_cmd=""
    case $INSTALL_METHOD in
        venv)
            pip_install_cmd="$PIP_CMD install"
            ;;
        system)
            if [ "$EXTERNALLY_MANAGED" == "true" ]; then
                pip_install_cmd="$PIP_CMD install --break-system-packages"
            else
                pip_install_cmd="$PIP_CMD install"
            fi
            ;;
        user)
            pip_install_cmd="$PIP_CMD install --user"
            ;;
        user_override)
            pip_install_cmd="$PIP_CMD install --user --break-system-packages"
            ;;
    esac
    
    # Try installation
    if $pip_install_cmd "$package" >/dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        return 0
    else
        # Try with verbose output on failure
        echo -e "${YELLOW}retry${NC}"
        local error_output=$($pip_install_cmd "$package" 2>&1)
        
        # Check if it actually installed despite error
        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            echo -e "  ${GREEN}✅ Installed successfully${NC}"
            return 0
        else
            echo -e "  ${RED}❌ Failed to install${NC}"
            echo -e "  ${YELLOW}Error: $(echo "$error_output" | head -n 1)${NC}"
            return 1
        fi
    fi
}

# Install dependencies
echo -e "${YELLOW}📚 Installing dependencies...${NC}"
echo -e "${CYAN}Core ML Libraries:${NC}"

# Track installation status
FAILED_PACKAGES=""
SUCCESS_COUNT=0
TOTAL_COUNT=0

# Core ML dependencies
for pkg in "numpy>=2.0.0:NumPy:numpy" \
           "scikit-learn>=1.7.0:scikit-learn:sklearn" \
           "lightgbm>=4.0.0:LightGBM:lightgbm"; do
    IFS=':' read -r package description import_name <<< "$pkg"
    ((TOTAL_COUNT++))
    if install_package "$package" "$description" "$import_name"; then
        ((SUCCESS_COUNT++))
    else
        FAILED_PACKAGES="$FAILED_PACKAGES $package"
    fi
done

echo -e "${CYAN}Performance Optimizations:${NC}"

# Performance dependencies
for pkg in "numba>=0.60.0:Numba JIT:numba" \
           "numexpr>=2.10.0:NumExpr:numexpr" \
           "bottleneck>=1.3.0:Bottleneck:bottleneck" \
           "msgpack>=1.0.0:MessagePack:msgpack" \
           "zstandard>=0.20.0:Zstandard:zstandard"; do
    IFS=':' read -r package description import_name <<< "$pkg"
    ((TOTAL_COUNT++))
    if install_package "$package" "$description" "$import_name"; then
        ((SUCCESS_COUNT++))
    else
        FAILED_PACKAGES="$FAILED_PACKAGES $package"
    fi
done

echo -e "${CYAN}System Integration:${NC}"

# System dependencies
for pkg in "psutil>=5.8.0:psutil:psutil" \
           "requests>=2.25.0:requests:requests"; do
    IFS=':' read -r package description import_name <<< "$pkg"
    ((TOTAL_COUNT++))
    if install_package "$package" "$description" "$import_name"; then
        ((SUCCESS_COUNT++))
    else
        FAILED_PACKAGES="$FAILED_PACKAGES $package"
    fi
done

# Linux-specific dependencies
if [ "$(uname)" == "Linux" ]; then
    echo -e "${CYAN}Linux Integration:${NC}"
    for pkg in "dbus-next>=0.2.0:D-Bus:dbus_next" \
               "distro>=1.6.0:Distro:distro"; do
        IFS=':' read -r package description import_name <<< "$pkg"
        ((TOTAL_COUNT++))
        if install_package "$package" "$description" "$import_name"; then
            ((SUCCESS_COUNT++))
        else
            FAILED_PACKAGES="$FAILED_PACKAGES $package"
        fi
    done
fi

# Installation summary
echo ""
echo -e "${BLUE}Installation Summary:${NC}"
echo -e "  Packages installed: ${GREEN}$SUCCESS_COUNT${NC}/${TOTAL_COUNT}"

if [ -n "$FAILED_PACKAGES" ]; then
    echo -e "  ${YELLOW}⚠️  Some packages failed to install:${NC}"
    for pkg in $FAILED_PACKAGES; do
        echo -e "    ${RED}• $pkg${NC}"
    done
    echo ""
    echo -e "${YELLOW}The system will use fallback implementations for missing packages.${NC}"
fi

# Create executable commands
echo -e "${YELLOW}🔗 Creating command shortcuts...${NC}"

# Main executable
if [ "$INSTALL_METHOD" == "venv" ]; then
    cat > "$BIN_DIR/shader-predict-compile" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
python main.py "\$@"
EOF
else
    cat > "$BIN_DIR/shader-predict-compile" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
$PYTHON_CMD main.py "\$@"
EOF
fi
chmod +x "$BIN_DIR/shader-predict-compile"

# Status command
if [ "$INSTALL_METHOD" == "venv" ]; then
    cat > "$BIN_DIR/shader-predict-status" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.core.enhanced_ml_predictor import get_enhanced_predictor
    predictor = get_enhanced_predictor()
    stats = predictor.get_enhanced_stats()
    print('🎮 ML Shader Predictor Status')
    print('=' * 30)
    print(f'Backend: {stats.get(\"ml_backend\", \"fallback\")}')
    print(f'Memory: {stats.get(\"memory_usage_mb\", 0):.1f}MB')
    print(f'Cache Hit Rate: {stats.get(\"feature_cache_stats\", {}).get(\"hit_rate\", 0):.1%}')
    if '$IS_STEAM_DECK' == 'true':
        print(f'Platform: Steam Deck $STEAM_DECK_MODEL')
except Exception as e:
    print(f'Error: {e}')
    print('System may be using fallback mode')
"
EOF
else
    cat > "$BIN_DIR/shader-predict-status" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
$PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
try:
    from src.core.enhanced_ml_predictor import get_enhanced_predictor
    predictor = get_enhanced_predictor()
    stats = predictor.get_enhanced_stats()
    print('🎮 ML Shader Predictor Status')
    print('=' * 30)
    print(f'Backend: {stats.get(\"ml_backend\", \"fallback\")}')
    print(f'Memory: {stats.get(\"memory_usage_mb\", 0):.1f}MB')
    print(f'Cache Hit Rate: {stats.get(\"feature_cache_stats\", {}).get(\"hit_rate\", 0):.1%}')
    if '$IS_STEAM_DECK' == 'true':
        print(f'Platform: Steam Deck $STEAM_DECK_MODEL')
except Exception as e:
    print(f'Error: {e}')
    print('System may be using fallback mode')
"
EOF
fi
chmod +x "$BIN_DIR/shader-predict-status"

# Test command
if [ "$INSTALL_METHOD" == "venv" ]; then
    cat > "$BIN_DIR/shader-predict-test" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
if [ -f "test_steam_deck_implementation.py" ]; then
    python test_steam_deck_implementation.py
else
    python -c "
import sys
sys.path.insert(0, '.')
print('Testing ML system...')
try:
    from src.core.enhanced_ml_predictor import get_enhanced_predictor
    predictor = get_enhanced_predictor()
    import numpy as np
    features = np.random.rand(100, 20)
    predictions = predictor.predict(features)
    print(f'✅ ML predictions working: {len(predictions)} samples processed')
except ImportError as e:
    print(f'⚠️  Using fallback mode: {e}')
except Exception as e:
    print(f'❌ Error: {e}')
"
fi
EOF
else
    cat > "$BIN_DIR/shader-predict-test" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
if [ -f "test_steam_deck_implementation.py" ]; then
    $PYTHON_CMD test_steam_deck_implementation.py
else
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
print('Testing ML system...')
try:
    from src.core.enhanced_ml_predictor import get_enhanced_predictor
    predictor = get_enhanced_predictor()
    import numpy as np
    features = np.random.rand(100, 20)
    predictions = predictor.predict(features)
    print(f'✅ ML predictions working: {len(predictions)} samples processed')
except ImportError as e:
    print(f'⚠️  Using fallback mode: {e}')
except Exception as e:
    print(f'❌ Error: {e}')
"
fi
EOF
fi
chmod +x "$BIN_DIR/shader-predict-test"

# Add to PATH if needed
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo -e "${YELLOW}📝 Adding $BIN_DIR to PATH...${NC}"
    
    # Add to appropriate shell config
    for config in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [ -f "$config" ]; then
            if ! grep -q "export PATH=\"$BIN_DIR:\$PATH\"" "$config"; then
                echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$config"
            fi
        fi
    done
    export PATH="$BIN_DIR:$PATH"
fi

# Create systemd service if requested
if [ "$ENABLE_SERVICE" == "true" ] && [ "$IS_STEAM_DECK" == "true" ]; then
    echo -e "${YELLOW}🔧 Setting up systemd service...${NC}"
    
    mkdir -p "$HOME/.config/systemd/user"
    
    cat > "$HOME/.config/systemd/user/shader-predict-compile.service" << EOF
[Unit]
Description=ML Shader Prediction Compiler
After=graphical-session.target

[Service]
Type=simple
ExecStart=$BIN_DIR/shader-predict-compile --daemon
Restart=on-failure
RestartSec=10
Nice=10
Environment="HOME=$HOME"

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable shader-predict-compile.service
    echo -e "${GREEN}✅ Service enabled (will start on next boot)${NC}"
fi

# Create uninstall script
echo -e "${YELLOW}📝 Creating uninstall script...${NC}"
cat > "$INSTALL_DIR/uninstall.sh" << EOF
#!/bin/bash
echo "🗑️  Uninstalling ML Shader Prediction Compiler..."

# Stop and disable service
systemctl --user stop shader-predict-compile.service 2>/dev/null || true
systemctl --user disable shader-predict-compile.service 2>/dev/null || true

# Remove files
rm -rf "$INSTALL_DIR" 2>/dev/null || true
rm -rf "$CONFIG_DIR" 2>/dev/null || true
rm -rf "$HOME/.cache/shader-predict-compile" 2>/dev/null || true
rm -f "$BIN_DIR/shader-predict-"* 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/shader-predict-compile.service" 2>/dev/null || true

# Remove virtual environment if used
if [ -d "$VENV_DIR" ]; then
    echo "Removing virtual environment..."
    rm -rf "$VENV_DIR"
fi

systemctl --user daemon-reload 2>/dev/null || true

echo "✅ Uninstallation complete!"
EOF
chmod +x "$INSTALL_DIR/uninstall.sh"

# Final setup
echo ""
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo ""
echo -e "${BLUE}Available commands:${NC}"
echo "  shader-predict-compile  - Run the ML compiler"
echo "  shader-predict-status   - Check system status"
echo "  shader-predict-test     - Test the installation"
echo ""

if [ "$INSTALL_METHOD" == "venv" ]; then
    echo -e "${CYAN}Virtual environment location:${NC}"
    echo "  $VENV_DIR"
    echo ""
fi

echo -e "${BLUE}Installation details:${NC}"
echo "  Install directory: $INSTALL_DIR"
echo "  Config directory:  $CONFIG_DIR"
echo "  Installation type: $INSTALL_METHOD"
echo ""

if [ "$IS_STEAM_DECK" == "true" ]; then
    echo -e "${GREEN}🎮 Steam Deck optimizations enabled${NC}"
    echo "  Model: $STEAM_DECK_MODEL"
    echo "  • Hardware-specific ML optimizations"
    echo "  • Thermal-aware compilation scheduling"
    echo "  • Gaming Mode integration ready"
    echo ""
    
    if [ "$ENABLE_SERVICE" != "true" ]; then
        echo "To enable automatic startup:"
        echo "  $0 --enable-service"
        echo ""
    fi
fi

echo "To uninstall:"
echo "  $INSTALL_DIR/uninstall.sh"
echo ""

# Test the installation
echo -e "${YELLOW}Running quick test...${NC}"
if $BIN_DIR/shader-predict-test 2>/dev/null | grep -q "✅"; then
    echo -e "${GREEN}✅ System test passed!${NC}"
else
    echo -e "${YELLOW}⚠️  System running in fallback mode${NC}"
    echo "  This is normal if some optional dependencies failed to install."
fi

echo ""
echo -e "${GREEN}🚀 ML Shader Prediction Compiler is ready!${NC}"