#!/bin/bash

# Web Installer for ML Shader Prediction Compiler
# Downloads repository and runs installation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🎮 ML Shader Prediction Compiler - Web Installer${NC}"
echo "=================================================="
echo ""

# Check if git is available
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is required but not installed${NC}"
    echo "Please install git and try again:"
    echo "  sudo pacman -S git    # On Steam Deck/Arch"
    echo "  sudo apt install git  # On Ubuntu/Debian"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo -e "${YELLOW}📂 Downloading to temporary directory: $TEMP_DIR${NC}"

# Clone repository
echo -e "${YELLOW}📥 Downloading ML Shader Prediction Compiler...${NC}"
git clone https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler.git "$TEMP_DIR/shader-compiler" >/dev/null 2>&1

# Run installer
echo -e "${YELLOW}🚀 Running installation...${NC}"
cd "$TEMP_DIR/shader-compiler"
chmod +x scripts/install_simple.sh
./scripts/install_simple.sh

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up temporary files...${NC}"
cd "$HOME"
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "Available commands:"
echo "  shader-predict-compile  - Run the ML compiler"
echo "  shader-predict-status   - Check system status"
echo "  shader-predict-test     - Test installation"
echo ""
echo -e "${BLUE}Repository available at:${NC}"
echo "  https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler"