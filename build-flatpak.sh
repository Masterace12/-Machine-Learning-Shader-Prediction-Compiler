#!/bin/bash
#
# Build script for ML Shader Predictor Flatpak
# Optimized for Steam Deck deployment
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building ML Shader Prediction Compiler Flatpak..."
echo "=================================================="

# Check if flatpak-builder is installed
if ! command -v flatpak-builder &> /dev/null; then
    echo "Error: flatpak-builder not found. Please install it first:"
    echo "  Debian/Ubuntu: sudo apt install flatpak-builder"
    echo "  Fedora: sudo dnf install flatpak-builder"
    echo "  Arch: sudo pacman -S flatpak-builder"
    exit 1
fi

# Check if required runtimes are installed
echo "Checking for required Flatpak runtimes..."
if ! flatpak list --runtime | grep -q "org.kde.Platform.*6.6"; then
    echo "Installing KDE Platform 6.6 runtime..."
    flatpak install -y flathub org.kde.Platform//6.6 org.kde.Sdk//6.6
fi

# Create build directory
BUILD_DIR="build-flatpak"
REPO_DIR="repo-flatpak"

rm -rf "$BUILD_DIR" "$REPO_DIR"
mkdir -p "$BUILD_DIR" "$REPO_DIR"

echo "Building Flatpak package..."
echo "Build directory: $BUILD_DIR"
echo "Repository: $REPO_DIR"

# Build the Flatpak
flatpak-builder \
    --force-clean \
    --disable-rofiles-fuse \
    --repo="$REPO_DIR" \
    --subject="ML Shader Prediction Compiler for Steam Deck" \
    --body="Optimized ML shader prediction system with Steam Deck integration" \
    "$BUILD_DIR" \
    com.shaderpredict.MLCompiler.yml

echo ""
echo "Build complete!"
echo "==============="

# Install locally for testing
echo "Installing locally for testing..."
flatpak install --user -y "$REPO_DIR" com.shaderpredict.MLCompiler

echo ""
echo "Installation complete!"
echo "You can now run the application with:"
echo "  flatpak run com.shaderpredict.MLCompiler"
echo ""
echo "Or test different modes:"
echo "  flatpak run com.shaderpredict.MLCompiler --status"
echo "  flatpak run com.shaderpredict.MLCompiler --setup-steam"
echo "  flatpak run com.shaderpredict.MLCompiler --help"
echo ""
echo "To create a Flatpak bundle for distribution:"
echo "  flatpak build-bundle '$REPO_DIR' ml-shader-predictor-steamdeck.flatpak com.shaderpredict.MLCompiler"
echo ""