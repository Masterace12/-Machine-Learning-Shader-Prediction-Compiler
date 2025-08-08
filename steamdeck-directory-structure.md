# Steam Deck Directory Structure for ML Shader Prediction Compiler

## Overview

This guide outlines the directory structure optimized for Steam Deck's immutable filesystem, ensuring data persistence across SteamOS updates and proper isolation from system files.

## Core Principles

1. **Immutable Filesystem Compatibility**: All application data stored in user directories
2. **Update Persistence**: Data survives SteamOS updates and factory resets (with user data preservation)
3. **Performance Optimization**: Proper cache placement for SSD longevity
4. **Security**: Minimal permissions required, sandboxed execution
5. **Steam Integration**: Compatible with Steam's directory structure

## Directory Structure

### Main Application Directory
```
/home/deck/.local/share/shader-predict-ml/
├── src/                              # Application source code
│   ├── main.py                       # Main entry point
│   ├── ml_shader_predictor.py        # ML prediction system
│   ├── steam_deck_integration.py     # Steam Deck specific code
│   ├── thermal_manager.py            # Thermal monitoring
│   ├── p2p_shader_network.py         # P2P networking
│   └── security/                     # Security modules
│       ├── sandbox_executor.py
│       ├── signature_verification.py
│       └── privacy_protection.py
├── scripts/                          # Helper scripts
│   ├── thermal-monitor.sh            # Thermal monitoring script
│   ├── service-wrapper.sh            # Systemd service wrapper
│   └── steam-integration.sh          # Steam hook scripts
├── models/                           # ML models and data
│   ├── shader_predictor.pkl          # Trained prediction model
│   ├── thermal_model.pkl             # Thermal prediction model
│   └── performance_profiles/         # Device-specific profiles
│       ├── lcd_model.json
│       └── oled_model.json
├── logs/                            # Application logs
│   ├── shader-predict.log           # Main application log
│   ├── thermal.log                  # Thermal monitoring log
│   └── p2p.log                      # P2P networking log
├── launch.sh                        # Main launcher script
├── TROUBLESHOOTING.md               # Troubleshooting guide
└── VERSION                          # Version information
```

### Configuration Directory
```
/home/deck/.config/shader-predict-ml/
├── config.json                      # Main configuration
├── steamdeck-config.json           # Steam Deck specific settings
├── games/                           # Per-game configurations
│   ├── 440_csgo.json              # Game-specific settings
│   ├── 570_dota2.json
│   └── default.json               # Default game settings
├── profiles/                        # User profiles
│   ├── performance.json            # Performance-focused profile
│   ├── battery.json               # Battery-saving profile
│   └── balanced.json              # Balanced profile
├── security/                        # Security configurations
│   ├── trusted_peers.json          # P2P trusted peers
│   ├── signature_keys.json         # Signature verification keys
│   └── sandbox_rules.json          # Sandboxing rules
└── ml/                             # ML configuration
    ├── training_params.json         # Training parameters
    ├── model_config.json           # Model configuration
    └── feature_weights.json        # Feature importance weights
```

### Cache Directory
```
/home/deck/.cache/shader-predict-ml/
├── shaders/                         # Compiled shader cache
│   ├── hash_index.db               # Shader hash database
│   ├── compiled/                   # Successfully compiled shaders
│   │   ├── game_440/               # Per-game shader cache
│   │   └── game_570/
│   └── failed/                     # Failed compilation attempts
├── ml_models/                      # Cached ML model data
│   ├── training_data.pkl           # Training dataset cache
│   ├── feature_cache.pkl           # Preprocessed features
│   └── prediction_cache.json       # Recent predictions
├── p2p/                            # P2P networking cache
│   ├── peer_database.json          # Known peers
│   ├── shared_shaders/             # Downloaded shaders
│   └── reputation_data.json        # Peer reputation data
├── temp/                           # Temporary files
│   ├── compilation/                # Temp compilation workspace
│   ├── downloads/                  # Temporary downloads
│   └── analysis/                   # Shader analysis temp files
└── thermal/                        # Thermal monitoring cache
    ├── temperature_history.json    # Historical temperature data
    └── throttling_events.json      # Throttling event history
```

### Data Directory
```
/home/deck/.local/share/shader-predict-ml/data/
├── training/                        # ML training data
│   ├── shader_samples/             # Shader training samples
│   ├── performance_data/           # Performance measurements
│   ├── thermal_data/               # Thermal correlation data
│   └── user_feedback/              # User feedback data
├── compiled/                       # Compiled shader database
│   ├── metadata.db                # Shader metadata database
│   ├── shaders.db                 # Main shader database
│   └── statistics.json            # Compilation statistics
├── profiles/                       # User and system profiles
│   ├── hardware_profile.json      # Hardware-specific data
│   ├── gaming_patterns.json       # Gameplay pattern analysis
│   └── performance_baseline.json  # Performance baseline data
└── exports/                        # Data exports
    ├── training_data_backup.json  # Training data backup
    └── user_statistics.json       # User statistics export
```

### Systemd Service Directory
```
/home/deck/.config/systemd/user/
├── shader-predict-ml.service       # Main service file
├── shader-predict-ml-thermal.timer # Thermal monitoring timer
├── shader-predict-ml-thermal.service # Thermal service
├── shader-predict-ml-gaming.service # Gaming mode service
└── shader-predict-ml.target        # Service target
```

### Steam Integration
```
/home/deck/.local/share/Steam/
├── compatibilitytools.d/
│   └── shader-predict-ml/          # Steam compatibility tool
│       ├── compatibilitytool.vdf   # Steam tool definition
│       ├── shader-predict-ml       # Wrapper script
│       └── tool.vdf                # Tool metadata
└── config/
    └── shader_predict_hooks/       # Steam launch hooks
        ├── pre_launch.sh
        └── post_launch.sh
```

## Permission Requirements

### File System Permissions
```bash
# Application directory (755)
chmod 755 /home/deck/.local/share/shader-predict-ml
chmod +x /home/deck/.local/share/shader-predict-ml/launch.sh

# Configuration directory (700 for security)
chmod 700 /home/deck/.config/shader-predict-ml
chmod 600 /home/deck/.config/shader-predict-ml/security/*

# Cache directory (755, optimized for performance)
chmod 755 /home/deck/.cache/shader-predict-ml
chmod 777 /home/deck/.cache/shader-predict-ml/temp

# Data directory (755)
chmod 755 /home/deck/.local/share/shader-predict-ml/data
```

### System Permissions Required
- Read access to `/sys/class/thermal/` (thermal monitoring)
- Read access to `/sys/class/hwmon/` (hardware monitoring)
- Read access to `/proc/cpuinfo` and `/proc/meminfo` (system info)
- Network access for P2P functionality
- GPU device access (`/dev/dri/card0`, `/dev/dri/renderD128`)

## Storage Considerations

### Disk Usage Estimates
- **Application**: ~50MB (source code, scripts, configs)
- **ML Models**: ~100MB (trained models, cached data)
- **Shader Cache**: ~500MB - 2GB (depends on games played)
- **Training Data**: ~200MB (accumulated over time)
- **Logs**: ~50MB (with rotation)
- **Total**: ~1-3GB typical usage

### SSD Longevity Optimizations
- Logs rotated weekly, compressed
- Temporary files cleaned on service restart
- Cache size limits enforced
- Bulk writes batched to reduce wear
- Frequent small writes avoided

## Backup and Migration

### Critical Files to Backup
```bash
# Configuration (user preferences)
/home/deck/.config/shader-predict-ml/

# Training data (valuable accumulated data)
/home/deck/.local/share/shader-predict-ml/data/training/

# Compiled shader cache (performance benefit)
/home/deck/.cache/shader-predict-ml/shaders/compiled/

# ML models (if custom trained)
/home/deck/.local/share/shader-predict-ml/models/
```

### Migration Script
```bash
#!/bin/bash
# Backup critical data before SteamOS update
tar -czf ~/shader-predict-backup.tar.gz \
    ~/.config/shader-predict-ml/ \
    ~/.local/share/shader-predict-ml/data/ \
    ~/.cache/shader-predict-ml/shaders/compiled/ \
    ~/.local/share/shader-predict-ml/models/
```

## Troubleshooting Directory Issues

### Check Directory Permissions
```bash
# Verify all directories exist and have correct permissions
find ~/.local/share/shader-predict-ml -type d ! -perm 755 -ls
find ~/.config/shader-predict-ml -type d ! -perm 700 -ls
```

### Clean Corrupted Cache
```bash
# Safe cache cleanup
rm -rf ~/.cache/shader-predict-ml/temp/*
rm -rf ~/.cache/shader-predict-ml/p2p/shared_shaders/*
systemctl --user restart shader-predict-ml.service
```

### Repair Directory Structure
```bash
# Recreate missing directories
mkdir -p ~/.local/share/shader-predict-ml/{src,scripts,models,logs}
mkdir -p ~/.config/shader-predict-ml/{games,profiles,security,ml}
mkdir -p ~/.cache/shader-predict-ml/{shaders,ml_models,p2p,temp,thermal}
mkdir -p ~/.local/share/shader-predict-ml/data/{training,compiled,profiles,exports}
```

## Integration with Steam Deck Updates

### Update-Safe Locations
All directories under `/home/deck/` are preserved during:
- SteamOS updates
- Steam client updates  
- System recovery (if user data preservation is enabled)

### Not Preserved During Factory Reset
- Complete factory reset will remove all user data
- Recommend backing up to external storage or cloud before factory reset

### Flatpak Considerations
When using Flatpak packaging:
```
~/.var/app/com.shaderpredict.MLCompiler/
├── config/shader-predict-ml/        # Mapped to ~/.config/
├── cache/shader-predict-ml/         # Mapped to ~/.cache/
└── data/shader-predict-ml/          # Mapped to ~/.local/share/
```

This structure ensures compatibility with both native and Flatpak installations while maintaining Steam Deck optimization and data persistence.