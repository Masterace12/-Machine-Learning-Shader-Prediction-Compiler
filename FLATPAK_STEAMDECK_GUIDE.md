# ML Shader Predictor - Steam Deck Flatpak Guide

Complete guide for building and using the ML Shader Prediction Compiler Flatpak on Steam Deck.

## Features

### Steam Deck Optimizations
- **Thermal Management**: Automatic thermal throttling when temperature exceeds 75°C
- **Battery Awareness**: Power-saving mode when battery is below 20%
- **Gaming Mode Detection**: Minimal resource usage during gameplay
- **RADV Optimizations**: Pre-configured AMD GPU settings for Steam Deck
- **Memory Constraints**: Optimized for Steam Deck's 16GB RAM with swap limitations

### Flatpak Benefits
- **Sandboxed Security**: Isolated from system with controlled permissions
- **Easy Installation**: Single command installation via KDE Discover
- **Automatic Updates**: Integrated with Steam Deck's update system
- **Clean Uninstall**: No system files left behind when removed
- **Steam Integration**: Compatible with both Gaming Mode and Desktop Mode

## Quick Installation

### Method 1: One-Command Build (Recommended)
```bash
# On Steam Deck Desktop Mode
cd ~/Downloads
wget -O build-ml-shader.sh https://raw.githubusercontent.com/your-repo/build-flatpak.sh
chmod +x build-ml-shader.sh
./build-ml-shader.sh
```

### Method 2: Manual Build
```bash
# Install Flatpak Builder (if not present)
sudo steamos-readonly disable
sudo pacman -S flatpak-builder
sudo steamos-readonly enable

# Clone and build
git clone https://github.com/your-repo/ML-Shader-Predictor.git
cd ML-Shader-Predictor
./build-flatpak.sh
```

## Verified Dependencies

All Python packages use **verified SHA256 hashes from PyPI**:

| Package | Version | SHA256 Hash (First 16 chars) |
|---------|---------|------------------------------|
| numpy | 1.24.4 | 222e40d0e2548690... |
| scikit-learn | 1.3.2 | fc4144a5004a676d... |
| psutil | 5.9.6 | 748c9dd2583ed863... |
| PyYAML | 6.0.1 | 7b5eefd033ab7199... |
| requests | 2.31.0 | 58cd2187c01e70e6... |
| joblib | 1.3.2 | ef4331c65f239985... |
| py-cpuinfo | 9.0.0 | e0a9023730ba63db... |

## Usage

### Basic Commands
```bash
# Launch GUI (default)
flatpak run com.shaderpredict.MLCompiler

# Run as background service
flatpak run com.shaderpredict.MLCompiler --service

# Check system status
flatpak run com.shaderpredict.MLCompiler --status

# Setup Steam integration
flatpak run com.shaderpredict.MLCompiler --setup-steam
```

### Steam Deck Specific
```bash
# Gaming Mode launcher (minimal resources)
flatpak run com.shaderpredict.MLCompiler --gaming-mode

# Desktop Mode (full features)
flatpak run com.shaderpredict.MLCompiler --desktop-mode

# Check Steam Deck hardware status
flatpak run com.shaderpredict.MLCompiler --steamdeck-info
```

## Steam Integration

### Automatic Setup
After installation, run:
```bash
flatpak run com.shaderpredict.MLCompiler --setup-steam
```

This creates:
- Steam compatibility tool entry
- Shader cache integration hooks
- Gaming Mode optimization scripts

### Manual Steam Setup
1. Open Steam in Desktop Mode
2. Go to Settings → Compatibility
3. Select "ML Shader Prediction (Flatpak)" as compatibility tool
4. Apply to games that benefit from shader prediction

## Configuration

### Steam Deck Specific Settings
Configuration is stored in: `~/.var/app/com.shaderpredict.MLCompiler/config/shader-predict-ml/`

Default Steam Deck configuration:
```json
{
  "steam_deck": {
    "thermal_limit_celsius": 80,
    "battery_threshold_percent": 20,
    "gaming_mode_cpu_limit": 5,
    "memory_limit_mb": 256
  },
  "performance": {
    "max_worker_threads": 1,
    "thermal_throttling": true,
    "battery_aware": true
  }
}
```

### Environment Variables
The Flatpak automatically sets:
- `RADV_PERFTEST=aco` - Enable AMD's ACO shader compiler
- `MESA_GLSL_CACHE_DISABLE=0` - Enable shader caching
- `AMD_VULKAN_ICD=RADV` - Use RADV Vulkan driver
- `SHADER_PREDICT_STEAMDECK=1` - Enable Steam Deck optimizations

## Troubleshooting

### Build Issues
```bash
# If build fails due to missing runtimes
flatpak install flathub org.kde.Platform//6.6 org.kde.Sdk//6.6

# If hash verification fails
# This indicates a corrupted download - rebuild will re-download
rm -rf build-flatpak repo-flatpak
./build-flatpak.sh
```

### Runtime Issues
```bash
# Check Flatpak permissions
flatpak info --show-permissions com.shaderpredict.MLCompiler

# Debug mode
flatpak run --devel com.shaderpredict.MLCompiler --debug

# Reset configuration
rm -rf ~/.var/app/com.shaderpredict.MLCompiler/config/
```

### Steam Deck Specific Issues

#### Gaming Mode Not Detected
```bash
# Verify gamescope is running
ps aux | grep gamescope

# Force gaming mode
flatpak run com.shaderpredict.MLCompiler --force-gaming-mode
```

#### Thermal Throttling Too Aggressive
```bash
# Check current temperature
cat /sys/class/thermal/thermal_zone0/temp

# Adjust thermal limit (temporarily)
flatpak run com.shaderpredict.MLCompiler --thermal-limit 85
```

#### Memory Issues
```bash
# Check available memory
free -h

# Force low-memory mode
flatpak run com.shaderpredict.MLCompiler --low-memory-mode
```

## Performance Optimization

### Gaming Mode Resources
- **CPU Limit**: 5% to avoid interfering with games
- **Memory Limit**: 200MB including ML models
- **I/O Priority**: Background to prevent storage bottlenecks
- **GPU Access**: Read-only monitoring, no compute usage

### Desktop Mode Resources
- **CPU Limit**: 10% for responsive operation
- **Memory Limit**: 400MB for full ML functionality
- **Thermal Monitoring**: Active protection at 80°C
- **Battery Monitoring**: Power-saving below 20%

## File Locations

### Flatpak Sandbox Paths
```
Application: /app/lib/shader-predict-ml/
Config: ~/.var/app/com.shaderpredict.MLCompiler/config/
Cache: ~/.var/app/com.shaderpredict.MLCompiler/cache/
Data: ~/.var/app/com.shaderpredict.MLCompiler/data/
```

### Steam Integration
```
Steam Compat Tools: ~/.local/share/Steam/compatibilitytools.d/shader-predict-ml-flatpak/
Shader Caches: ~/.local/share/Steam/steamapps/shadercache/ (read-write)
```

## Uninstallation

```bash
# Remove application
flatpak uninstall com.shaderpredict.MLCompiler

# Remove user data (optional)
rm -rf ~/.var/app/com.shaderpredict.MLCompiler/

# Remove Steam integration
rm -rf ~/.local/share/Steam/compatibilitytools.d/shader-predict-ml-flatpak/
```

## Security

The Flatpak runs with minimal permissions:
- **No system file access** - Only user data directories
- **Controlled network access** - For shader sharing only
- **Hardware monitoring** - Read-only thermal/power sensors
- **Steam integration** - Limited to shader cache directories
- **GPU access** - Device access for monitoring, no compute

All Python packages are verified with SHA256 hashes to prevent supply chain attacks.

## Support

For Steam Deck specific issues:
1. Check thermal status: `cat /sys/class/thermal/thermal_zone0/temp`
2. Verify SteamOS version: `cat /etc/os-release`
3. Test in Desktop Mode first
4. Enable debug logging: `flatpak run com.shaderpredict.MLCompiler --debug`

The Flatpak is designed to fail gracefully - if ML dependencies fail to load, it falls back to a basic mode that still provides shader cache management.