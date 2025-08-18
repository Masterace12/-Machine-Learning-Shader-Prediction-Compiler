# Installation Guide

## Quick Installation

### One-Line Installation (Recommended)
```bash
curl -fsSL https://raw.githubusercontent.com/user/shader-predict-compile/main/install.sh | bash
```

### Manual Installation (Safer)
```bash
# Download and inspect the installer
wget https://raw.githubusercontent.com/user/shader-predict-compile/main/install.sh
chmod +x install.sh

# Review the script (recommended)
less install.sh

# Run the installer
./install.sh
```

## Installation Options

### Standard Installation
```bash
./install.sh
```
- Installs core components
- Sets up systemd service
- Configures automatic startup

### Steam Deck Specific
```bash
./install.sh --steamdeck
```
- Optimized for Steam Deck hardware
- Configures thermal management
- Sets up gaming mode integration

### Development Installation
```bash
./install.sh --dev
```
- Includes development dependencies
- Sets up testing framework
- Enables debug logging

### Minimal Installation
```bash
./install.sh --minimal
```
- Core functionality only
- Reduced memory footprint
- Basic thermal management

## System Requirements

### Minimum Requirements
- **Steam Deck**: SteamOS 3.7+ (any model)
- **Linux Desktop**: Ubuntu 20.04+, Debian 11+, Arch Linux, Fedora 35+
- **Python**: 3.8 or higher
- **Memory**: 2GB RAM available
- **Storage**: 500MB free space
- **Network**: Internet connection for initial setup

### Recommended
- **Steam Deck OLED**: Better thermal headroom for aggressive optimization
- **Storage**: 2GB+ for optimal cache performance
- **Network**: Stable connection for model updates

## Post-Installation Setup

### Automatic Configuration
The installer automatically:
- Detects Steam Deck model (LCD/OLED)
- Configures optimal settings
- Sets up system service
- Creates necessary directories

### Manual Configuration (Optional)
```bash
# Edit configuration
~/.config/shader-predict-compile/config.json

# Test installation
shader-predict-test

# View status
shader-predict-status
```

## Verification

### System Health Check
```bash
shader-predict-test
```

Expected output:
```
✅ System configuration valid
✅ ML models loaded successfully
✅ Steam integration active
✅ Thermal monitoring functional
✅ Cache system operational
```

### Performance Validation
```bash
shader-predict-compile --benchmark
```

This runs a comprehensive benchmark to verify performance improvements.

## Troubleshooting

### Installation Fails
```bash
# Check dependencies
python3 --version
pip3 --version

# Install missing dependencies
sudo apt install python3-pip python3-venv  # Ubuntu/Debian
sudo dnf install python3-pip               # Fedora
sudo pacman -S python-pip                  # Arch Linux
```

### Permission Issues
```bash
# Fix user permissions
chmod +x ~/.local/shader-predict-compile/install.sh

# Restart user services
systemctl --user daemon-reload
```

### Steam Deck Specific Issues
```bash
# Enable developer mode
sudo steamos-devmode

# Install in desktop mode
systemctl --user stop steam
# Run installer
systemctl --user start steam
```

## Uninstallation

### Complete Removal
```bash
~/.local/shader-predict-compile/uninstall.sh
```

### Manual Cleanup
```bash
# Stop services
systemctl --user stop shader-predict-compile
systemctl --user disable shader-predict-compile

# Remove files
rm -rf ~/.local/shader-predict-compile
rm -rf ~/.config/shader-predict-compile
rm -rf ~/.cache/shader-predict-compile

# Remove PATH entries (if added)
# Edit ~/.bashrc or ~/.profile as needed
```

## Advanced Installation

### Custom Installation Directory
```bash
INSTALL_DIR="/opt/shader-predict" ./install.sh
```

### Proxy Configuration
```bash
export HTTP_PROXY="http://proxy:8080"
export HTTPS_PROXY="http://proxy:8080"
./install.sh
```

### Offline Installation
1. Download installer and dependencies on connected machine
2. Transfer to offline system
3. Run with `--offline` flag

```bash
./install.sh --offline --deps-dir ./offline-deps/
```