# Gaming Mode Integration Guide

The Shader Predictive Compiler has been enhanced with full Gaming Mode support, allowing you to configure and monitor shader compilation directly from Steam's Gaming Mode interface.

## 🎮 Gaming Mode Features

### Automatic Integration
- **Non-Steam Game**: Automatically added to your Steam library during installation
- **Controller Navigation**: Full controller support with Gaming Mode optimized UI
- **Real-time Monitoring**: Monitor system performance and shader compilation in real-time
- **Power Management**: Adaptive power profiles based on battery and docked status

### Gaming Mode UI Features
- 📊 **Status Page**: Real-time system monitoring with thermal and battery awareness
- ⚙️ **Settings Page**: Configure power management and compilation settings
- 🎮 **Games Page**: View detected games and manage shader compilation
- 📈 **Performance Page**: Monitor cache statistics and system performance

## 🚀 Installation

The Gaming Mode integration is automatically set up during installation:

```bash
cd shader-predict-compile
chmod +x scripts/install.sh
./scripts/install.sh
```

### What Gets Installed:
1. **Gaming Mode Launcher**: Controller-friendly UI optimized for 1280x800 resolution
2. **Non-Steam Game Entry**: Added to your Steam library automatically
3. **Adaptive Power Management**: Battery-aware compilation with thermal protection
4. **Real-time Monitoring**: <0.1% CPU overhead system monitoring

## 🎯 Gaming Mode Access

### Method 1: Steam Library
1. Enter Gaming Mode (if not already active)
2. Navigate to **Library**
3. Go to **Non-Steam Games**
4. Select **Shader Predictive Compiler**
5. Press A to launch

### Method 2: Desktop Mode
- Launch from Applications Menu → Games → Shader Predictive Compiler
- The app will automatically detect if Gaming Mode is active

## 🔧 Gaming Mode Controls

### Navigation
- **D-Pad/Left Stick**: Navigate between UI elements
- **A Button**: Select/Activate
- **B Button**: Back/Cancel
- **X Button**: Quick actions (context-dependent)
- **Y Button**: Settings/Options

### Page Navigation
- **L1/R1 Bumpers**: Switch between main pages
- **Start Button**: Quick access to status
- **Select Button**: Open settings

## ⚙️ Gaming Mode Settings

### Power Management
- **Adaptive Power**: Automatically adjusts TDP and CPU frequency based on:
  - Battery level and charge status
  - Thermal conditions
  - Gaming Mode activity
  - Docked vs handheld mode

### Compilation Settings
- **Gaming Mode Pause**: Automatically pause shader compilation when games are running
- **Battery Awareness**: Reduce compilation intensity on low battery
- **Thermal Protection**: Stop compilation if temperatures exceed safe limits
- **Thread Limiting**: Adaptive thread count based on system load

### Profiles
- **Battery Save** (4W TDP): Maximum battery life
- **Handheld Gaming** (12W TDP): Optimized for 40-60 FPS gaming
- **Docked Gaming** (15W TDP): Maximum performance when plugged in

## 📊 Gaming Mode Monitoring

### Real-time Metrics
- **CPU/GPU Temperature**: With color-coded thermal warnings
- **Battery Status**: Level, charging state, time remaining
- **Power Draw**: Current consumption in watts
- **Fan Speed**: RPM monitoring
- **Memory Usage**: System and shader cache usage

### Performance Indicators
- 🟢 **Green**: Optimal conditions
- 🟡 **Yellow**: Caution (high temps, low battery)
- 🔴 **Red**: Critical (thermal throttling, very low battery)

## 🎮 Game Detection

The system automatically detects games from:
- **Main Steam Library**: `~/.steam/steam/steamapps/common/`
- **SD Card Libraries**: Additional library folders on SD cards
- **Multiple Steam Installations**: Handles various Steam installation paths

### Game Analysis
- **Automatic Scanning**: New games are detected and queued for analysis
- **Shader Prediction**: Uses heuristics to predict common shader variants
- **Priority Calculation**: Larger games and more frequently played games get higher priority

## 🔥 Gaming Mode Optimizations

### Steam Deck Specific
- **LCD Model**: Conservative power settings for better battery life
- **OLED Model**: Enhanced GPU optimizations with ray tracing support
- **Thermal Design**: 90°C GPU target with gradual throttling
- **Memory Management**: Unified memory architecture optimizations

### SteamOS Integration
- **Fossilize Integration**: Works with Valve's shader pre-compilation system
- **RADV Optimizations**: Latest AMD driver optimizations for Steam Deck
- **Cache Management**: Single-file compression and automatic cleanup
- **Transcoded Videos**: Support for Proton game cutscene optimization

## 🛠️ Troubleshooting

### Gaming Mode Not Detecting App
1. Restart Steam: `systemctl --user restart steam`
2. Check non-Steam games in Desktop Mode
3. Manually re-run: `python3 src/gaming_mode_integration.py`

### Controller Not Working
1. Ensure Steam Input is enabled in Steam settings
2. Check controller configuration in Steam
3. Try restarting the application

### High Temperature Warnings
- The system will automatically reduce compilation intensity
- Ensure adequate ventilation
- Check fan operation
- Consider using a cooling stand

### Battery Drain Issues
- Enable "Battery Aware Compilation" in settings
- Use "Battery Save" power profile
- Reduce max compilation threads
- Enable "Pause During Gaming"

## 📱 Gaming Mode vs Desktop Mode

| Feature | Gaming Mode | Desktop Mode |
|---------|-------------|--------------|
| **Interface** | Controller optimized | Mouse/keyboard optimized |
| **Resolution** | 1280x800 scaled | Native resolution |
| **Navigation** | D-pad/analog stick | Mouse cursor |
| **Power Management** | Automatic profiles | Manual configuration |
| **Real-time Monitoring** | Simplified metrics | Detailed graphs |
| **Game Detection** | Background only | Interactive scanning |

## 🔄 Updates and Maintenance

### Automatic Updates
- The Gaming Mode integration survives SteamOS updates
- Non-Steam game entries are preserved
- Settings are maintained across updates

### Manual Maintenance
```bash
# Test game detection
./test_game_detection.py

# Update Gaming Mode integration
python3 src/gaming_mode_integration.py

# Check service status
systemctl --user status shader-predict-compile

# View logs
journalctl --user -u shader-predict-compile -f
```

## 🎯 Performance Impact

The Gaming Mode UI is designed for minimal performance impact:
- **Memory Usage**: <50MB RAM
- **CPU Overhead**: <0.1% when monitoring
- **GPU Impact**: None (CPU-only monitoring)
- **Storage**: <2MB for UI components

## 🤝 Integration with Steam Features

### Steam Input
- Full controller mapping support
- Custom controller configurations
- Haptic feedback for notifications

### Steam Overlay
- Can be launched from Steam overlay
- Transparent overlay integration
- No interference with game performance

### Steam Cloud
- Settings can be synced across devices (if configured)
- Game preferences are preserved

## 📋 Quick Reference

### Common Tasks
- **Start/Stop Service**: Status page → Service toggle
- **Optimize Cache**: Status page → Quick Actions → Optimize Cache
- **Check Game**: Games page → Select game → Compile Selected
- **Monitor Performance**: Performance page → Real-time metrics
- **Change Power Profile**: Settings page → Power Management

### Keyboard Shortcuts (when available)
- `Ctrl+R`: Refresh games list
- `Ctrl+S`: Save current settings
- `Ctrl+Q`: Quit application
- `F5`: Refresh status
- `F11`: Toggle fullscreen

This Gaming Mode integration makes shader optimization easily accessible without leaving the gaming environment, providing a seamless experience for Steam Deck users.