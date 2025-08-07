# Shader Predictive Compiler for Steam Deck

A lightweight, heuristic-based shader compilation optimizer that enhances Valve's Fossilize system by predicting and prioritizing shader compilation on Steam Deck (both LCD and OLED models).

## Features

- 🎯 **Smart Shader Prediction**: Analyzes game files to predict common shader patterns and variants
- 🚀 **Fossilize Integration**: Enhances Valve's existing shader compilation system
- 🎮 **Steam Deck Optimized**: Specifically tuned for both LCD and OLED Steam Deck models
- 🔧 **User-Friendly GUI**: GTK-based interface for easy management
- ⚙️ **Background Service**: Automatic shader compilation when system is idle
- 🔄 **Easy Install/Uninstall**: Simple installation script with complete removal option
- 📊 **Coverage Maximization**: Prioritizes shaders for maximum game performance impact
- 🔍 **Auto-Detection**: Automatically finds Steam games across all libraries (including SD cards)
- 🎮 **Gaming Mode Support**: Optimized interface and behavior for Steam Deck Gaming Mode
- 📱 **Controller-Friendly UI**: Large buttons and navigation optimized for handheld use

## Compatibility

- **Steam Deck LCD**: Fully supported with power-optimized settings
- **Steam Deck OLED**: Fully supported with enhanced GPU optimizations  
- **SteamOS Version**: 3.7.13+ (earlier versions may work with limitations)
- **Dependencies**: Python 3.8+, GTK 3, Fossilize

## Installation

### Quick Install

```bash
cd shader-predict-compile
chmod +x scripts/install.sh
./scripts/install.sh
```

### What the installer does:

1. **Detects your Steam Deck model** (LCD vs OLED) and SteamOS version
2. **Checks dependencies** and installs missing packages
3. **Installs the application** to `/opt/shader-predict-compile`
4. **Creates desktop entry** for Gaming Mode and Desktop Mode access
5. **Sets up systemd service** for background operation
6. **Configures optimizations** specific to your Steam Deck model
7. **Integrates with Fossilize** for enhanced shader compilation

## Usage

### Auto-Launcher

The new auto-launcher automatically detects your environment and launches the appropriate interface:

```bash
# Automatically detects Gaming Mode vs Desktop Mode
./auto_launcher.sh
```

### GUI Mode

Launch from:
- **Gaming Mode**: Library → Non-Steam → Shader Predictive Compiler (controller-friendly interface)
- **Desktop Mode**: Applications → Games → Shader Predictive Compiler (full desktop interface)

The GUI provides:
- **Auto-Detection**: Automatically finds all Steam libraries (including SD cards and external drives)
- **Game Library Analysis**: Comprehensive scanning of all Steam game installations
- **Gaming Mode Optimization**: Specialized interface for handheld use with large buttons
- **Real-time System Monitoring**: Battery, thermal, and performance awareness
- **Game-specific shader compilation**: Intelligent prioritization based on game engines
- **Power Management**: Adaptive compilation based on battery level and thermal state
- **Settings and cache management**: Easy configuration for different usage scenarios

### Background Service

The service automatically:
- Monitors game launches
- Analyzes newly installed games
- Compiles shaders during idle periods
- Manages thermal and battery constraints
- Integrates compiled shaders with Steam

### Command Line

```bash
# Start background service manually
sudo systemctl start shader-predict-compile

# Check service status with auto-detection info
python3 src/background_service.py --status

# Scan for games without starting service
python3 src/background_service.py --scan-only

# Test auto-detection features
python3 test_game_detection.py

# View logs
journalctl -u shader-predict-compile -f

# Run auto-launcher
./auto_launcher.sh
```

## How It Works

### 1. Enhanced Game Analysis with Auto-Detection
- **Multi-Library Scanning**: Automatically discovers Steam libraries across all storage devices
- **SD Card Support**: Detects and monitors games on removable storage (SD cards, USB drives)
- **Game Engine Detection**: Identifies game engines (Unreal, Unity, Source2, etc.) for optimized compilation
- **Shader Analysis**: Analyzes shader file patterns and estimates compilation requirements
- **Real-time Monitoring**: Tracks game installations and removals across all libraries

### 2. Shader Prediction
- Uses heuristics to predict common shader variants
- Prioritizes shaders based on usage patterns
- Applies Steam Deck specific optimizations
- Generates compilation hints for Fossilize

### 3. Smart Compilation
- Compiles high-priority shaders first
- Monitors system resources (CPU, memory, thermal)
- Adjusts compilation intensity based on battery/power state
- Integrates results with Steam's shader cache

### 4. Fossilize Enhancement
- Works alongside Valve's existing system
- Provides priority hints for compilation order
- Optimizes compilation flags for Steam Deck hardware
- Maintains compatibility with Steam updates

## Steam Deck Specific Optimizations

### LCD Model
- Conservative power settings for better battery life
- 4-thread compilation limit
- Thermal throttling awareness
- Power-saving compilation flags

### OLED Model  
- Enhanced GPU optimization flags
- 6-thread compilation capability
- Variable rate shading support
- Improved memory efficiency

### Both Models
- AMD Zen 2 architecture targeting (`-march=znver2`)
- RADV GPU driver optimizations
- Adaptive resource management
- Integration with Steam's power profiles

### Gaming Mode Enhancements
- **Auto-Detection**: Detects Gaming Mode vs Desktop Mode automatically
- **Controller-Friendly Interface**: Large buttons and easy navigation for handheld use
- **Battery Awareness**: Adjusts compilation intensity based on battery level and charging state
- **Thermal Management**: Monitors CPU/GPU temperatures and adjusts workload accordingly
- **Power Profile Integration**: Adapts to Steam Deck power profiles (battery saver, balanced, performance)
- **Background Behavior**: Intelligently pauses during active gaming sessions
- **Gaming Mode UI**: Specialized interface optimized for the Steam Deck screen and controls

## Configuration

Configuration files are stored in `~/.config/shader-predict-compile/`:

- `settings.json`: GUI preferences and user settings
- `service.json`: Background service configuration
- `optimizations.json`: Hardware-specific optimizations

### Example service configuration:

```json
{
  "max_cpu_usage": 50.0,
  "max_memory_mb": 2048,
  "check_interval": 300,
  "auto_compile_threshold": 5,
  "thermal_throttling": true,
  "battery_aware": true
}
```

## Performance Impact

- **Reduced Shader Stutter**: Pre-compiled shaders eliminate in-game compilation
- **Faster Game Loading**: Shaders ready before game launch
- **Improved Frame Rates**: Optimized shader variants for Steam Deck
- **Better Power Efficiency**: Background compilation during optimal conditions

## Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check dependencies
sudo pacman -S python python-gobject python-psutil python-numpy

# Check permissions
sudo systemctl daemon-reload
sudo systemctl enable shader-predict-compile
```

**High CPU usage:**
- Adjust `max_cpu_usage` in service configuration
- Enable thermal throttling
- Reduce `max_parallel_compiles` setting

**Storage space issues:**
- Run cache cleanup in GUI settings
- Reduce cache retention period
- Use external storage for shader cache

**GUI won't launch:**
```bash
# Check GTK dependencies
python3 -c "import gi; gi.require_version('Gtk', '3.0')"

# Run with debug output
/opt/shader-predict-compile/launcher.sh --debug
```

### Logs

- **Service logs**: `journalctl -u shader-predict-compile`
- **Application logs**: `~/.cache/shader-predict-compile/logs/`
- **Compilation logs**: View in GUI or service logs

## Uninstallation

```bash
# Complete removal (includes cache)
sudo ./scripts/install.sh --uninstall

# Or via GUI settings
# Settings tab → Uninstall section → Uninstall button
```

This removes:
- Application files
- Desktop entries  
- Systemd service
- Configuration files
- Optionally: shader cache

## Contributing

This is a defensive security tool designed to enhance gaming performance on Steam Deck. Please ensure any contributions maintain compatibility with SteamOS and follow security best practices.

## Technical Details

### Architecture
- **Core Engine**: Python-based heuristic analysis
- **UI Framework**: GTK 3 for native Linux integration
- **Service**: systemd for reliable background operation
- **Integration**: Fossilize API for shader compilation
- **Storage**: Efficient cache management with automatic cleanup

### Shader Detection
- File extension analysis (`.hlsl`, `.glsl`, `.spv`, `.dxbc`)
- Magic number detection for binary shaders
- Content analysis for shader type classification
- Engine-specific pattern recognition

### Prediction Algorithm
- Frequency-based priority calculation
- Game engine shader pattern database
- Hardware capability consideration
- Runtime performance feedback

## License

This project is designed for defensive security purposes only - enhancing gaming performance and user experience on Steam Deck through optimized shader compilation.

## Compatibility Notes

- **SteamOS Updates**: Designed to work across SteamOS updates
- **Steam Client**: Compatible with Steam client updates
- **Fossilize**: Works with existing Fossilize installations
- **Performance**: Minimal impact on system resources
- **Storage**: Efficient cache management with configurable limits