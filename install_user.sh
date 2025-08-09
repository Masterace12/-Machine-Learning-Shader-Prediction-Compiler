#!/bin/bash
# User-space Installation Script for ML Shader Prediction Compiler
# Modified to install in user directory without sudo

set -euo pipefail

# Configuration
readonly REPO_URL="https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler"
readonly INSTALL_DIR="${HOME}/.local/shader-predict-compile"
readonly CONFIG_DIR="${HOME}/.config/shader-predict-compile"
readonly CACHE_DIR="${HOME}/.cache/shader-predict-compile"
readonly TEMP_DIR="/tmp/shader-install-$$"

# Colors
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1" >&2; }

cleanup() {
    [[ -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Main installation
main() {
    log_info "Starting user-space installation..."
    
    # Create directories
    mkdir -p "$INSTALL_DIR" "$CONFIG_DIR" "$CACHE_DIR" "$TEMP_DIR"
    
    # Download source
    log_info "Downloading source code..."
    cd "$TEMP_DIR"
    if ! git clone --depth 1 "$REPO_URL" repo 2>/dev/null; then
        log_info "Git not available, downloading as archive..."
        curl -fsSL "https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/archive/main.tar.gz" | tar xz
        mv *Machine-Learning-Shader-Prediction-Compiler* repo
    fi
    
    # Copy files
    log_info "Installing files..."
    cp -r repo/* "$INSTALL_DIR/"
    
    # Detect Steam Deck
    STEAM_DECK="false"
    STEAM_DECK_MODEL="unknown"
    if grep -q "Jupiter\|Galileo" /sys/class/dmi/id/board_name 2>/dev/null; then
        STEAM_DECK="true"
        if grep -q "Galileo" /sys/class/dmi/id/board_name 2>/dev/null; then
            STEAM_DECK_MODEL="OLED"
        else
            STEAM_DECK_MODEL="LCD"
        fi
    fi
    
    # Create configuration
    log_info "Creating configuration..."
    cat > "$CONFIG_DIR/config.json" << EOF
{
  "version": "1.0.0",
  "installation_date": "$(date -Iseconds)",
  "system": {
    "steam_deck": $(if [ "$STEAM_DECK" = "true" ]; then echo "true"; else echo "false"; fi),
    "steam_deck_model": "$STEAM_DECK_MODEL",
    "cache_size_mb": $(if [ "$STEAM_DECK" = "true" ]; then echo "1024"; else echo "2048"; fi)
  },
  "ml_models": {
    "enabled": true,
    "model_path": "$CACHE_DIR/models"
  },
  "p2p_network": {
    "enabled": true,
    "bandwidth_limit_kbps": $(if [ "$STEAM_DECK" = "true" ]; then echo "1024"; else echo "5120"; fi)
  },
  "paths": {
    "install_dir": "$INSTALL_DIR",
    "cache_dir": "$CACHE_DIR",
    "config_dir": "$CONFIG_DIR"
  }
}
EOF
    
    # Install Python dependencies in virtual environment
    log_info "Setting up Python environment..."
    cd "$INSTALL_DIR"
    python3 -m venv venv
    source venv/bin/activate
    
    # Install requirements if they exist
    if [ -f requirements.txt ]; then
        pip install --upgrade pip
        pip install -r requirements.txt 2>/dev/null || log_info "Some Python packages failed to install"
    fi
    
    # Create launcher script
    mkdir -p "$HOME/.local/bin"
    cat > "$HOME/.local/bin/shader-predict" << 'EOF'
#!/bin/bash
source "$HOME/.local/shader-predict-compile/venv/bin/activate"
python "$HOME/.local/shader-predict-compile/main.py" "$@"
EOF
    chmod +x "$HOME/.local/bin/shader-predict"
    
    log_success "Installation complete!"
    log_info "Installation directory: $INSTALL_DIR"
    log_info "Configuration: $CONFIG_DIR/config.json"
    log_info "To run: ~/.local/bin/shader-predict"
    
    # Add to PATH if not already there
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        log_info "Add $HOME/.local/bin to your PATH to run 'shader-predict' from anywhere"
    fi
}

main "$@"