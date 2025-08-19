#!/bin/bash

# ML Shader Prediction Compiler - Simple Uninstaller
# Removes all traces of the system

echo "🗑️  ML Shader Prediction Compiler - Uninstaller"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Confirm uninstallation
read -p "Are you sure you want to uninstall? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 0
fi

echo -e "${YELLOW}Removing ML Shader Prediction Compiler...${NC}"

# Stop services
echo "Stopping services..."
systemctl --user stop shader-predict-compile.service 2>/dev/null || true
systemctl --user disable shader-predict-compile.service 2>/dev/null || true

# Remove user installation
echo "Removing user files..."
rm -rf "$HOME/.local/share/shader-predict-compile" 2>/dev/null || true
rm -rf "$HOME/.config/shader-predict-compile" 2>/dev/null || true
rm -rf "$HOME/.cache/shader-predict-compile" 2>/dev/null || true

# Remove commands
echo "Removing commands..."
rm -f "$HOME/.local/bin/shader-predict-compile" 2>/dev/null || true
rm -f "$HOME/.local/bin/shader-predict-status" 2>/dev/null || true
rm -f "$HOME/.local/bin/shader-predict-test" 2>/dev/null || true

# Remove systemd services
echo "Removing services..."
rm -f "$HOME/.config/systemd/user/shader-predict-compile.service" 2>/dev/null || true

# Remove system-wide installation (if user has sudo)
if command -v sudo &> /dev/null; then
    echo "Removing system-wide files (if any)..."
    sudo rm -rf "/opt/shader-predict-compile" 2>/dev/null || true
    sudo rm -f "/usr/local/bin/shader-predict-compile" 2>/dev/null || true
    sudo rm -f "/usr/local/bin/shader-predict-status" 2>/dev/null || true
    sudo rm -f "/usr/local/bin/shader-predict-test" 2>/dev/null || true
fi

# Reload systemd
systemctl --user daemon-reload 2>/dev/null || true

echo ""
echo -e "${GREEN}✅ Uninstallation complete!${NC}"
echo ""
echo "All ML Shader Prediction Compiler files have been removed."
echo "Python packages installed by pip were left intact."
echo ""
echo "Thank you for using ML Shader Prediction Compiler!"