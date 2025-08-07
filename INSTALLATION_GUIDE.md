# 🚀 Complete Installation Guide

This comprehensive guide covers all installation methods for the Shader Prediction Compiler, from one-line installation to advanced custom deployments.

## 🎯 Quick Installation (Recommended)

### One-Line Installation

```bash
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash
```

This command will:
- ✅ Automatically detect your system (Steam Deck LCD/OLED, Linux distribution)
- ✅ Install all required dependencies
- ✅ Download and verify the latest release
- ✅ Configure optimization settings for your hardware
- ✅ Set up desktop integration and autostart
- ✅ Connect to the P2P shader network

### Security-Conscious Installation

For users who prefer to inspect scripts before execution:

```bash
# Download the installer
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh -o install.sh

# Inspect the script (recommended)
less install.sh

# Make executable and run
chmod +x install.sh && ./install.sh
```

## 🔧 Advanced Installation Options

### Custom Installation with Options

```bash
# Development version with latest features
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- --dev

# Minimal installation (no P2P, no ML)
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- --no-p2p --no-ml

# Installation without autostart
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- --no-autostart

# Force reinstall over existing installation
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- --force

# Skip system dependency installation
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash -s -- --skip-deps
```

### Installation Options Reference

| Option | Description | Use Case |
|--------|-------------|----------|
| `--dev` | Install development version | Latest features and fixes |
| `--no-p2p` | Disable P2P sharing features | Privacy-focused or offline use |
| `--no-ml` | Disable ML prediction features | Minimal resource usage |
| `--no-autostart` | Don't enable automatic startup | Manual control preference |
| `--force` | Force reinstall over existing | Fix corrupted installation |
| `--skip-deps` | Skip system dependency installation | Custom dependency management |
| `--help` | Show help message | View all available options |

## 🎮 Steam Deck Installation

### Gaming Mode Installation (Recommended)

1. **Switch to Desktop Mode**
   - Hold Power button → Switch to Desktop

2. **Open Terminal (Konsole)**
   - Find in Applications menu or press Ctrl+Alt+T

3. **Run Installation Command**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash
   ```

4. **Return to Gaming Mode**
   - The application will be available in your library

### What Gets Installed on Steam Deck

- 📁 **Main Installation**: `/opt/shader-predict-compile/`
- 🔧 **Configuration**: `~/.config/shader-predict-compile/`
- 💾 **Cache**: `~/.cache/shader-predict-compile/`
- 🎮 **Desktop Shortcut**: Available in Gaming Mode library
- 🚀 **Autostart Service**: Runs automatically in background
- 🔧 **System Commands**: `shader-predict-compile` and `uninstall-shader-predict-compile`

## 💻 Linux Distribution Support

### Ubuntu/Debian

```bash
# Standard installation works out of the box
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash

# Manual dependency installation if needed
sudo apt update && sudo apt install python3 python3-pip python3-gi mesa-utils vulkan-tools
```

### Fedora/CentOS/RHEL

```bash
# Standard installation with automatic package manager detection
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash

# Manual dependency installation if needed
sudo dnf install python3 python3-pip python3-gobject mesa-utils vulkan-tools
```

### Arch Linux/Manjaro

```bash
# Works perfectly with pacman
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash

# Manual dependency installation if needed
sudo pacman -S python python-pip python-gobject gtk3 mesa-utils vulkan-tools
```

### openSUSE

```bash
# Standard installation with zypper support
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash

# Manual dependency installation if needed
sudo zypper install python3 python3-pip python3-gobject-Gdk typelib-1_0-Gtk-3_0 Mesa-utils vulkan-tools
```

## 🪟 Windows Installation (Beta)

### Windows Subsystem for Linux (WSL)

1. **Install WSL2 with Ubuntu**
   ```powershell
   wsl --install -d Ubuntu
   ```

2. **Install in WSL**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash
   ```

### Native Windows Support

Currently in beta. The installer will attempt to set up a compatible environment using:
- Python for Windows
- Windows Subsystem for Linux integration
- Native Windows shader cache integration

## 🔒 Manual Installation (Maximum Security)

For users who require complete control over the installation process:

### Step 1: Download and Verify

```bash
# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download latest release
LATEST_VERSION=$(curl -s https://api.github.com/repos/YourUsername/shader-prediction-compilation/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
curl -L "https://github.com/YourUsername/shader-prediction-compilation/archive/refs/tags/${LATEST_VERSION}.tar.gz" -o shader-compiler.tar.gz

# Download checksums
curl -L "https://github.com/YourUsername/shader-prediction-compilation/releases/download/${LATEST_VERSION}/SHA256SUMS" -o SHA256SUMS

# Verify checksums
sha256sum -c SHA256SUMS --ignore-missing
```

### Step 2: Extract and Inspect

```bash
# Extract archive
tar -xzf shader-compiler.tar.gz
cd shader-prediction-compilation-*

# Inspect contents
ls -la
cat README.md
less install.sh
```

### Step 3: Manual Installation

```bash
# Install dependencies manually
sudo apt update && sudo apt install python3 python3-pip python3-gi mesa-utils vulkan-tools

# Install Python packages
python3 -m pip install --user numpy psutil requests PyGObject

# Create directories
sudo mkdir -p /opt/shader-predict-compile
mkdir -p ~/.config/shader-predict-compile ~/.cache/shader-predict-compile

# Copy files
sudo cp -r src/* /opt/shader-predict-compile/
sudo cp -r security/ /opt/shader-predict-compile/
cp -r config/* ~/.config/shader-predict-compile/ 2>/dev/null || true

# Set permissions
sudo chown -R $(whoami):$(whoami) /opt/shader-predict-compile
find /opt/shader-predict-compile -name "*.py" -exec chmod +x {} \;

# Create launcher
sudo tee /usr/local/bin/shader-predict-compile >/dev/null << 'EOF'
#!/bin/bash
cd /opt/shader-predict-compile
python3 shader_prediction_system.py "$@"
EOF
sudo chmod +x /usr/local/bin/shader-predict-compile

# Test installation
shader-predict-compile --version
```

## 🧪 Development Installation

For contributors and developers:

### Clone and Build

```bash
# Clone repository
git clone https://github.com/YourUsername/shader-prediction-compilation.git
cd shader-prediction-compilation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Run tests
python -m pytest tests/ -v

# Install pre-commit hooks
pre-commit install
```

### Development Environment

```bash
# Start development server
python src/shader_prediction_system.py --dev --debug

# Run with live reload
watchexec -e py "python src/shader_prediction_system.py --dev"

# Run P2P network in simulation mode
python src/p2p_demonstration.py
```

## 🔄 Container Installation

### Docker

```dockerfile
# Dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-gi \
    mesa-utils \
    vulkan-tools

RUN curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh | bash

CMD ["shader-predict-compile", "--service"]
```

```bash
# Build and run
docker build -t shader-compiler .
docker run -d --name shader-service shader-compiler
```

### Flatpak (Planned)

```bash
# Future Flatpak installation
flatpak install flathub com.github.YourUsername.ShaderPredictionCompiler
```

## ✅ Post-Installation Verification

### System Check

```bash
# Check installation
shader-predict-compile --version
shader-predict-compile --check-system

# Test hardware detection
shader-predict-compile --detect-hardware

# Verify P2P connectivity
shader-predict-compile --network-test

# Test ML model loading
shader-predict-compile --test-ml
```

### Performance Test

```bash
# Run built-in benchmark
shader-predict-compile --benchmark

# Test with a specific game
shader-predict-compile --game-id 1091500 --test-compilation
```

## 🗑️ Uninstallation

### One-Line Uninstall

```bash
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash
```

### Uninstall Options

```bash
# Uninstall with backup
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash -s -- --backup

# Keep configuration files
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash -s -- --keep-config

# Preview what would be removed
curl -fsSL https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/uninstall.sh | bash -s -- --dry-run
```

### Manual Uninstallation

```bash
# Stop services
pkill -f shader_prediction_system.py

# Remove installation directory
sudo rm -rf /opt/shader-predict-compile

# Remove user data
rm -rf ~/.config/shader-predict-compile
rm -rf ~/.cache/shader-predict-compile

# Remove desktop integration
rm -f ~/.local/share/applications/shader-predict-compile.desktop
rm -f ~/.config/autostart/shader-predict-compile.desktop

# Remove system commands
sudo rm -f /usr/local/bin/shader-predict-compile
sudo rm -f /usr/local/bin/uninstall-shader-predict-compile
```

## 🚨 Troubleshooting Installation

### Common Issues

**Permission Denied**
```bash
# Fix with sudo access
sudo chown -R $(whoami):$(whoami) /opt/shader-predict-compile
```

**Python Module Not Found**
```bash
# Reinstall Python dependencies
python3 -m pip install --user --force-reinstall numpy psutil requests
```

**Network Timeout**
```bash
# Use alternative download method
wget https://raw.githubusercontent.com/YourUsername/shader-prediction-compilation/main/install.sh
chmod +x install.sh && ./install.sh
```

**Steam Deck Not Detected**
```bash
# Check DMI information
cat /sys/class/dmi/id/board_name
cat /sys/class/dmi/id/product_name

# Manual override
echo "STEAM_DECK_MODEL=oled" >> ~/.config/shader-predict-compile/override.conf
```

### Getting Help

If you encounter issues:

1. 📖 Check the [Troubleshooting Wiki](https://github.com/YourUsername/shader-prediction-compilation/wiki/Troubleshooting)
2. 🐛 Search [GitHub Issues](https://github.com/YourUsername/shader-prediction-compilation/issues)
3. 💬 Ask in [GitHub Discussions](https://github.com/YourUsername/shader-prediction-compilation/discussions)
4. 🆕 Create a [New Issue](https://github.com/YourUsername/shader-prediction-compilation/issues/new) with:
   - System information (`uname -a`)
   - Installation method used
   - Error messages and logs
   - Expected vs actual behavior

## 📊 Installation Statistics

The installer collects anonymous statistics to improve compatibility:

- ✅ **System Detection**: Hardware model, OS version
- ✅ **Installation Success**: Whether installation completed
- ✅ **Performance Metrics**: Basic system performance data
- ❌ **No Personal Data**: No usernames, game libraries, or identifying information

Statistics can be disabled with `--no-telemetry` flag.

---

**Installation successful?** 🎉 Check out the [User Guide](https://github.com/YourUsername/shader-prediction-compilation/wiki/User-Guide) to get started!