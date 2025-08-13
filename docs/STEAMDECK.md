# Steam Deck Optimization Guide

## Hardware-Specific Features

### Automatic Model Detection

The system automatically detects your Steam Deck model and optimizes accordingly:

**LCD Model (Van Gogh APU)**:
- 4 compilation threads maximum
- Conservative thermal limits (95°C max)
- 150MB memory limit
- Optimized for 15W TDP

**OLED Model (Phoenix APU)**:
- 6 compilation threads maximum  
- Higher thermal limits (97°C max)
- 200MB memory limit
- Optimized for 18W TDP

### Configuration Files

Located at `~/.config/shader-predict-compile/`:

**steamdeck_lcd_config.json**:
```json
{
  "hardware": {
    "model": "LCD",
    "apu": "Van Gogh",
    "tdp_limit": 15,
    "memory_gb": 16,
    "display": "LCD 800p"
  },
  "optimization": {
    "compilation_threads": 4,
    "memory_limit_mb": 150,
    "thermal_target": 80.0,
    "thermal_max": 95.0,
    "prediction_threshold": 0.85
  },
  "ml_config": {
    "backend": "lightgbm",
    "model_size": "compact",
    "inference_threads": 2
  }
}
```

**steamdeck_oled_config.json**:
```json
{
  "hardware": {
    "model": "OLED", 
    "apu": "Phoenix",
    "tdp_limit": 18,
    "memory_gb": 16,
    "display": "OLED HDR 800p"
  },
  "optimization": {
    "compilation_threads": 6,
    "memory_limit_mb": 200,
    "thermal_target": 82.0,
    "thermal_max": 97.0,
    "prediction_threshold": 0.87
  },
  "ml_config": {
    "backend": "lightgbm_optimized",
    "model_size": "full",
    "inference_threads": 3
  }
}
```

## Gaming Mode Integration

### Seamless Operation
- Automatically starts with Steam
- Works in Big Picture mode
- No interaction required during gaming
- Respects Steam's power management

### D-Bus Integration
Monitors Steam activity via D-Bus:
```python
# Steam game launch detection
self.monitor_steam_dbus()
def on_game_start(app_id):
    self.activate_prediction_for_game(app_id)
```

### Gaming Mode Status
Check system status from command line:
```bash
# Quick status (works in Gaming Mode)
shader-predict-status --brief

# Detailed status (use in Desktop Mode)  
shader-predict-status --full
```

## Thermal Management

### Temperature Monitoring
Direct hardware sensor integration:
```python
# Temperature sensors accessed
/sys/class/thermal/thermal_zone*/temp
/sys/class/hwmon/hwmon*/temp*_input
```

### Adaptive Behavior

**Cool State (< 60°C)**:
- Aggressive shader compilation
- Full ML model utilization
- 6 threads (OLED) / 4 threads (LCD)

**Optimal State (60-70°C)**:
- Standard operation
- Normal prediction confidence
- Full feature set enabled

**Normal State (70-80°C)**:
- Slight reduction in aggressiveness  
- 4 threads (OLED) / 3 threads (LCD)
- Maintained prediction quality

**Warm Predictive (Trend → 85°C)**:
- Preemptive thread reduction
- Essential predictions only
- 2-3 threads maximum

**Hot State (85-95°C)**:
- Minimal compilation activity
- 1 thread maximum
- Safety-first approach

**Critical State (> 95°C)**:
- Emergency shutdown of compilation
- System protection mode
- Automatic service restart when cool

### Game-Specific Thermal Profiles

**Cyberpunk 2077**:
```json
{
  "thermal_profile": "aggressive",
  "max_threads": 2,
  "thermal_threshold": 78.0,
  "prediction_confidence": 0.92,
  "notes": "Known to cause thermal stress"
}
```

**Elden Ring**:
```json
{
  "thermal_profile": "moderate", 
  "max_threads": 3,
  "thermal_threshold": 82.0,
  "prediction_confidence": 0.88,
  "notes": "Balanced thermal behavior"
}
```

## Power Management

### Battery Awareness
- Detects AC adapter connection
- Reduces activity on battery power
- Configurable battery thresholds

### Power Profiles
```bash
# Battery mode (reduced performance)
shader-predict-compile --power-mode battery

# Balanced mode (default)
shader-predict-compile --power-mode balanced  

# Performance mode (AC power)
shader-predict-compile --power-mode performance
```

### TDP Integration
Respects Steam Deck's TDP settings:
- Monitors `/sys/class/hwmon/hwmon*/power*`
- Adjusts compilation intensity accordingly
- Prevents power budget conflicts

## Storage Optimization

### Cache Management
Optimized for Steam Deck's limited storage:

**Cache Hierarchy**:
```
~/.cache/shader-predict-compile/
├── hot/          # 50MB, fastest access
├── warm/         # 150MB, compressed  
└── cold/         # 300MB, deep storage
```

**Automatic Cleanup**:
- LRU eviction when space limited
- Game-specific cache priorities
- Automatic compression of old entries

### microSD Card Support
- Detects microSD installation
- Optimizes for slower storage speeds
- Reduced I/O operations on microSD

## Network Optimization

### Downloading Mode Detection
- Pauses intensive operations during downloads
- Resumes automatically when idle
- Prevents network congestion

### Offline Capability
- Full functionality without internet
- Local model inference only
- Cached predictions available

## Gaming Performance Impact

### CPU Core Utilization
Steam Deck's 4-core/8-thread Zen 2 optimization:
- Uses spare threads during gaming
- Monitors game CPU usage
- Automatically reduces threads under load

### Memory Management
Respects Steam Deck's shared memory:
- 16GB total system memory
- Reserves memory for games
- Adaptive memory limits based on usage

### GPU Coordination
Works with RDNA2 GPU scheduling:
- Avoids conflicting with game rendering
- Respects GPU power limits
- Coordinates with Mesa driver

## Troubleshooting Steam Deck Issues

### Service Not Starting
```bash
# Check systemd status
systemctl --user status shader-predict-compile

# Restart in Desktop Mode
systemctl --user restart shader-predict-compile

# Check logs
journalctl --user -u shader-predict-compile -n 50
```

### Performance Issues
```bash
# Check thermal throttling
shader-predict-status --thermal

# Monitor resource usage
shader-predict-status --resources

# Reduce aggressiveness
shader-predict-compile --config thermal.max_threads=2
```

### Gaming Mode Problems
```bash
# Force enable in Gaming Mode
shader-predict-compile --force-enable

# Check D-Bus connection
shader-predict-compile --test-steam-integration

# Reset to defaults
shader-predict-compile --reset-config
```

### Storage Issues
```bash
# Clean cache
shader-predict-compile --clean-cache

# Check disk usage  
du -sh ~/.cache/shader-predict-compile/

# Move cache to microSD
shader-predict-compile --cache-dir /run/media/mmcblk0p1/shader-cache/
```

## Developer Mode

### Additional Features in Developer Mode
```bash
# Enable detailed logging
shader-predict-compile --debug --verbose

# Performance profiling
shader-predict-compile --profile

# Raw sensor data
shader-predict-compile --sensors-raw
```

### Testing in Developer Mode
```bash
# Comprehensive system test
shader-predict-test --hardware

# Thermal stress test
shader-predict-test --thermal-stress

# Memory leak detection
shader-predict-test --memory-profile
```

## SteamOS Updates

### Compatibility
- Survives SteamOS updates
- Automatically detects OS changes
- Reapplies optimizations post-update

### Update Procedure
1. Service automatically pauses during update
2. Detects new SteamOS version on restart
3. Revalidates hardware configuration
4. Updates thermal profiles if needed
5. Resumes normal operation

### Recovery Mode
If issues after SteamOS update:
```bash
# Reinstall service
~/.local/shader-predict-compile/install.sh --repair

# Reset configuration
shader-predict-compile --reset-all --reconfigure
```