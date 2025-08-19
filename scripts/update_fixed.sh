#!/bin/bash

# ML Shader Prediction Compiler - Update Script
# Updates existing installation with fixed permission handling

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔄 ML Shader Prediction Compiler - Updater${NC}"
echo "=============================================="
echo ""

# Find existing installation
if [ -d "$HOME/.local/share/shader-predict-compile" ]; then
    INSTALL_DIR="$HOME/.local/share/shader-predict-compile"
    echo -e "${GREEN}✅ Found installation at: $INSTALL_DIR${NC}"
elif [ -d "/opt/shader-predict-compile" ]; then
    INSTALL_DIR="/opt/shader-predict-compile"
    echo -e "${GREEN}✅ Found system installation at: $INSTALL_DIR${NC}"
else
    echo -e "${RED}❌ No existing installation found${NC}"
    echo "Please run install_fixed.sh first"
    exit 1
fi

# Check for virtual environment
VENV_DIR="$HOME/.local/share/shader-predict-venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✅ Virtual environment detected${NC}"
    USE_VENV=true
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="$VENV_DIR/bin/python"
else
    USE_VENV=false
    PYTHON_CMD="python3"
fi

# Update project files
echo -e "${YELLOW}📦 Updating project files...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for item in src config main.py docs tests; do
    if [ -e "$SCRIPT_DIR/$item" ]; then
        echo -n "  Updating $item... "
        cp -r "$SCRIPT_DIR/$item" "$INSTALL_DIR/" 2>/dev/null && echo -e "${GREEN}✅${NC}" || echo -e "${RED}❌${NC}"
    fi
done

# Update dependencies if using venv
if [ "$USE_VENV" == "true" ]; then
    echo -e "${YELLOW}📚 Updating dependencies...${NC}"
    pip install --upgrade pip >/dev/null 2>&1
    
    # Core packages
    for pkg in numpy scikit-learn lightgbm numba psutil; do
        echo -n "  Updating $pkg... "
        pip install --upgrade "$pkg" >/dev/null 2>&1 && echo -e "${GREEN}✅${NC}" || echo -e "${YELLOW}⚠️${NC}"
    done
else
    echo -e "${YELLOW}⚠️  Skipping dependency updates (no venv)${NC}"
    echo "  Run install_fixed.sh --venv for better dependency management"
fi

echo ""
echo -e "${GREEN}✅ Update complete!${NC}"
echo ""
echo "Run 'shader-predict-status' to check system status"