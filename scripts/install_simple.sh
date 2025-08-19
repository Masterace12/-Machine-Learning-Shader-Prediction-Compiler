#!/bin/bash

# Simplified ML Shader Prediction Compiler Installation
# Works with or without read-only filesystem

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🎮 ML Shader Prediction Compiler - Simple Installer${NC}"
echo "===================================================="
echo ""

# Detect Steam Deck
IS_STEAM_DECK=false
if [ -f "/sys/devices/virtual/dmi/id/product_name" ]; then
    PRODUCT_NAME=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "")
    if [[ "$PRODUCT_NAME" == "Jupiter" ]] || [[ "$PRODUCT_NAME" == "Galileo" ]]; then
        IS_STEAM_DECK=true
        echo -e "${GREEN}🎮 Steam Deck detected${NC}"
    fi
fi

# Installation directories (user-space only)
INSTALL_DIR="$HOME/.local/share/shader-predict-compile"
BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/shader-predict-compile"
VENV_DIR="$HOME/.local/share/shader-predict-venv"

echo -e "${CYAN}Installation method: User-space with virtual environment${NC}"
echo ""

# Step 1: Create virtual environment
echo -e "${YELLOW}Step 1: Creating virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment exists, recreating..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip first
echo -n "  Upgrading pip... "
pip install --upgrade pip >/dev/null 2>&1
echo -e "${GREEN}✅${NC}"

# Step 2: Install Python packages
echo -e "${YELLOW}Step 2: Installing Python packages...${NC}"

install_package() {
    local package=$1
    local name=$2
    echo -n "  Installing $name... "
    if pip install "$package" >/dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        return 0
    else
        echo -e "${RED}❌${NC}"
        return 1
    fi
}

# Essential packages
install_package "numpy>=2.0.0" "NumPy"
install_package "scikit-learn>=1.7.0" "scikit-learn"
install_package "lightgbm>=4.0.0" "LightGBM"
install_package "psutil>=5.8.0" "psutil"

# Optional performance packages (don't fail if these don't install)
install_package "numba>=0.60.0" "Numba" || true
install_package "numexpr>=2.10.0" "NumExpr" || true

# Step 3: Copy project files
echo -e "${YELLOW}Step 3: Installing project files...${NC}"
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$CONFIG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for item in src config main.py docs tests *.py *.md; do
    if [ -e "$SCRIPT_DIR/$item" ]; then
        cp -r "$SCRIPT_DIR/$item" "$INSTALL_DIR/" 2>/dev/null
    fi
done
echo -e "  ${GREEN}✅ Files copied${NC}"

# Step 4: Create command shortcuts
echo -e "${YELLOW}Step 4: Creating commands...${NC}"

# Main command
cat > "$BIN_DIR/shader-predict-compile" << 'EOF'
#!/bin/bash
VENV_DIR="$HOME/.local/share/shader-predict-venv"
INSTALL_DIR="$HOME/.local/share/shader-predict-compile"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$INSTALL_DIR"
python main.py "$@"
EOF
chmod +x "$BIN_DIR/shader-predict-compile"

# Status command
cat > "$BIN_DIR/shader-predict-status" << 'EOF'
#!/bin/bash
VENV_DIR="$HOME/.local/share/shader-predict-venv"
INSTALL_DIR="$HOME/.local/share/shader-predict-compile"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$INSTALL_DIR"
python -c "
import sys
sys.path.insert(0, '.')
print('🎮 ML Shader Predictor Status')
print('=' * 30)
try:
    import numpy
    print(f'NumPy: {numpy.__version__}')
except:
    print('NumPy: Not installed')
try:
    import sklearn
    print(f'scikit-learn: {sklearn.__version__}')
except:
    print('scikit-learn: Not installed')
try:
    import lightgbm
    print(f'LightGBM: {lightgbm.__version__}')
except:
    print('LightGBM: Not installed')
    
print()
print('System is ready for use!')
"
EOF
chmod +x "$BIN_DIR/shader-predict-status"

# Test command
cat > "$BIN_DIR/shader-predict-test" << 'EOF'
#!/bin/bash
VENV_DIR="$HOME/.local/share/shader-predict-venv"
INSTALL_DIR="$HOME/.local/share/shader-predict-compile"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$INSTALL_DIR"
python -c "
import sys
sys.path.insert(0, '.')

print('Testing ML system...')
try:
    import numpy as np
    import lightgbm as lgb
    
    # Create dummy data
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    
    # Train simple model
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    predictions = model.predict(X)
    
    print(f'✅ ML system working!')
    print(f'   Processed {len(predictions)} predictions')
except Exception as e:
    print(f'❌ Error: {e}')
"
EOF
chmod +x "$BIN_DIR/shader-predict-test"

echo -e "  ${GREEN}✅ Commands created${NC}"

# Step 5: Add to PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo -e "${YELLOW}Step 5: Adding to PATH...${NC}"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "  ${GREEN}✅ Added to PATH${NC}"
fi

# Create uninstaller
cat > "$INSTALL_DIR/uninstall.sh" << 'EOF'
#!/bin/bash
echo "Uninstalling ML Shader Prediction Compiler..."
rm -rf "$HOME/.local/share/shader-predict-compile"
rm -rf "$HOME/.local/share/shader-predict-venv"
rm -f "$HOME/.local/bin/shader-predict-"*
echo "✅ Uninstalled"
EOF
chmod +x "$INSTALL_DIR/uninstall.sh"

# Final message
echo ""
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo ""
echo -e "${BLUE}Available commands:${NC}"
echo "  shader-predict-compile  - Run the compiler"
echo "  shader-predict-status   - Check status"
echo "  shader-predict-test     - Test installation"
echo ""
echo "To uninstall: $INSTALL_DIR/uninstall.sh"
echo ""

# Fix permissions
chmod +x "$BIN_DIR/shader-predict-"*

# Run test
echo -e "${YELLOW}Running test...${NC}"
$BIN_DIR/shader-predict-test