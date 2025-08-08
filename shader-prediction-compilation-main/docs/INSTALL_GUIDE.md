# Shader Predictive Compiler - Steam Deck Installation Guide

## Overview

This guide will help you install the Shader Predictive Compiler on your Steam Deck. The installation process has been optimized for Steam Deck and includes automatic dependency handling.

## Prerequisites

- Steam Deck (LCD or OLED model)
- SteamOS 3.7 or later (recommended)
- At least 500MB of free space
- Internet connection for downloading dependencies

## Quick Installation

### Method 1: Automatic Installation (Recommended)

1. **Download the project** from GitHub to your Steam Deck
2. **Extract the archive** to a convenient location (e.g., your home directory)
3. **Open Konsole** (Desktop Mode) and navigate to the extracted directory:
   ```bash
   cd ~/shader-prediction-compilation-main/shader-predict-compile
   ```
4. **Run the installer**:
   ```bash
   ./install
   ```

The installer will:
- Detect your Steam Deck model
- Install all required dependencies
- Set up the background service
- Create desktop shortcuts
- Configure Steam integration

### Method 2: Step-by-Step Installation

If the automatic installer fails, follow these steps:

1. **Check system compatibility**:
   ```bash
   ./check_steam_deck.sh
   ```

2. **Install dependencies**:
   ```bash
   ./install_dependencies.sh
   ```

3. **Run the manual installer**:
   ```bash
   ./install-manual
   ```

## Post-Installation

### Starting the Application

**Desktop Mode:**
- Find "Shader Predictive Compiler" in the application menu under Games
- Or run from terminal: `/opt/shader-predict-compile/launcher.sh`

**Gaming Mode:**
- Add as a Non-Steam Game in your library
- The application will automatically detect Gaming Mode and adjust the UI

### Managing the Background Service

The background service optimizes shader compilation automatically:

```bash
# Check status
systemctl status shader-predict-compile

# Start service
sudo systemctl start shader-predict-compile

# Stop service
sudo systemctl stop shader-predict-compile

# Enable auto-start on boot
sudo systemctl enable shader-predict-compile
```

## Troubleshooting

### Common Issues

**1. Permission Denied Errors**
```bash
# Fix permissions
bash setup.sh
```

**2. Missing Dependencies**
```bash
# Install Python GTK bindings
sudo pacman -S python-gobject

# Install Python packages
python3 -m pip install --user psutil numpy PyGObject
```

**3. Service Not Starting**
- Check logs: `journalctl -u shader-predict-compile -f`
- Verify installation: `ls -la /opt/shader-predict-compile/`

**4. GUI Not Opening**
- Ensure you're in Desktop Mode
- Check GTK installation: `python3 -c "import gi; gi.require_version('Gtk', '3.0')"`

### Steam Deck Specific Issues

**Pacman Key Errors:**
```bash
sudo pacman-key --init
sudo pacman-key --populate
sudo pacman -Sy
```

**Read-Only Filesystem:**
- The installer uses `/opt` which should be writable
- User configs are stored in `~/.config/shader-predict-compile/`

## Configuration

### Settings Location
- System: `/opt/shader-predict-compile/config/`
- User: `~/.config/shader-predict-compile/settings.json`

### Performance Tuning

The application automatically detects your Steam Deck model and applies optimizations:

- **LCD Model**: 4 compilation threads, 2GB memory limit
- **OLED Model**: 6 compilation threads, 2.5GB memory limit

You can adjust these in the settings file if needed.

## Uninstallation

To completely remove the application:

```bash
/opt/shader-predict-compile/install --uninstall
```

Or manually:
```bash
cd ~/shader-prediction-compilation-main/shader-predict-compile
./uninstall.sh
```

## Getting Help

### Logs
- Application logs: `~/.cache/shader-predict-compile/`
- Service logs: `journalctl -u shader-predict-compile`

### Support
- Check the README.md for detailed documentation
- Report issues on the GitHub repository
- Join the community Discord for help

## Tips for Steam Deck Users

1. **Gaming Mode Integration**: The app works in both Desktop and Gaming Mode
2. **Battery Life**: The service is optimized to minimize battery impact
3. **Storage**: Shader caches are stored in `~/.cache/shader-predict-compile/`
4. **Updates**: Pull the latest version from GitHub and run the installer again

## Advanced Usage

### Command Line Options
```bash
# Check service status
/opt/shader-predict-compile/launcher.sh --status

# Open settings
/opt/shader-predict-compile/launcher.sh --settings

# Force recompilation
/opt/shader-predict-compile/launcher.sh --force-compile
```

### Integration with Decky Loader
If you have Decky Loader installed, you can create a plugin for easier access in Gaming Mode.

## Security Notes

- The application runs with user privileges
- Background service has resource limits to prevent system impact
- No network access required after installation
- All shader data stays local to your device

---

For more information, see the main README.md file.