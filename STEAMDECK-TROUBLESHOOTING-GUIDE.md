# Steam Deck ML Shader Prediction Compiler - Troubleshooting Guide

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Problems](#runtime-problems) 
3. [Performance Issues](#performance-issues)
4. [Steam Integration Problems](#steam-integration-problems)
5. [Thermal and Battery Management](#thermal-and-battery-management)
6. [Dependency Issues](#dependency-issues)
7. [Flatpak Specific Issues](#flatpak-specific-issues)
8. [Network and P2P Problems](#network-and-p2p-problems)
9. [Diagnostic Tools](#diagnostic-tools)
10. [Recovery Procedures](#recovery-procedures)

---

## Installation Issues

### Problem: Installation Script Fails with Permission Errors

**Symptoms:**
- "Permission denied" errors during installation
- Cannot create directories in ~/.local/share/
- Failed to install Python packages

**Solutions:**

1. **Fix directory permissions:**
   ```bash
   # Ensure user owns local directories
   sudo chown -R deck:deck ~/.local/
   sudo chown -R deck:deck ~/.config/
   sudo chown -R deck:deck ~/.cache/
   
   # Fix permissions
   chmod 755 ~/.local/share/
   chmod 755 ~/.config/
   chmod 755 ~/.cache/
   ```

2. **Run dependency fix script:**
   ```bash
   bash steamdeck-dependencies-fix.sh
   ```

3. **Manual installation in user space:**
   ```bash
   # Force user-only installation
   bash steamdeck-optimized-install.sh --user-only --force
   ```

### Problem: Immutable Filesystem Prevents Installation

**Symptoms:**
- Cannot write to /usr or /opt directories
- "Read-only file system" errors
- Package manager commands fail

**Solutions:**

1. **Use Flatpak installation (recommended):**
   ```bash
   bash steamdeck-optimized-install.sh --flatpak
   ```

2. **Enable developer mode temporarily:**
   ```bash
   # Switch to read-write mode (temporary)
   sudo steamos-readonly disable
   
   # Install dependencies
   sudo pacman -S python python-pip python-numpy
   
   # Re-enable read-only mode
   sudo steamos-readonly enable
   ```

3. **Container-based installation:**
   ```bash
   # Install in distrobox container
   distrobox create --name shader-predict --image registry.fedoraproject.org/fedora:38
   distrobox enter shader-predict
   # Run installation inside container
   ```

### Problem: Python Dependencies Fail to Install

**Symptoms:**
- pip install commands timeout or fail
- "Building wheel failed" errors
- Missing compiler or development headers

**Solutions:**

1. **Use pre-built wheels:**
   ```bash
   # Install from PyPI with pre-built wheels
   python3 -m pip install --user --only-binary=all numpy scikit-learn
   ```

2. **Install minimal dependencies:**
   ```bash
   # Core dependencies only
   python3 -m pip install --user psutil requests PyYAML
   ```

3. **Use system packages when available:**
   ```bash
   # On mutable filesystem
   sudo pacman -S python-numpy python-scikit-learn python-requests
   ```

---

## Runtime Problems

### Problem: Service Fails to Start

**Symptoms:**
- `systemctl --user status shader-predict-ml.service` shows failed
- Service exits immediately after starting
- No response from launcher script

**Diagnostics:**
```bash
# Check service status
systemctl --user status shader-predict-ml.service

# View detailed logs
journalctl --user -u shader-predict-ml.service -f

# Test launcher directly
~/.local/share/shader-predict-ml/launch.sh --test
```

**Solutions:**

1. **Check dependencies:**
   ```bash
   ~/.local/bin/test-shader-deps
   ```

2. **Recreate directories:**
   ```bash
   mkdir -p ~/.local/share/shader-predict-ml/{src,scripts,models}
   mkdir -p ~/.config/shader-predict-ml
   mkdir -p ~/.cache/shader-predict-ml
   ```

3. **Reset service:**
   ```bash
   systemctl --user stop shader-predict-ml.service
   systemctl --user daemon-reload
   systemctl --user start shader-predict-ml.service
   ```

### Problem: High CPU/Memory Usage

**Symptoms:**
- System becomes slow during gaming
- Thermal throttling triggered
- Battery drains quickly

**Solutions:**

1. **Check thermal manager status:**
   ```bash
   python3 ~/.local/share/shader-predict-ml/steamdeck-thermal-manager.py --status
   ```

2. **Manually apply gaming profile:**
   ```bash
   systemctl --user set-property shader-predict-ml.service CPUQuota=3%
   systemctl --user set-property shader-predict-ml.service MemoryMax=150M
   ```

3. **Enable thermal monitoring:**
   ```bash
   systemctl --user start shader-predict-ml-thermal.timer
   ```

### Problem: Application Crashes or Freezes

**Symptoms:**
- Process stops responding
- Segmentation faults in logs
- Python import errors

**Diagnostics:**
```bash
# Check for core dumps
coredumpctl list
coredumpctl info <PID>

# Run with debug output
DEBUG=1 ~/.local/share/shader-predict-ml/launch.sh --test

# Check system resources
free -h
df -h ~/.local/share/
```

**Solutions:**

1. **Clean corrupted cache:**
   ```bash
   rm -rf ~/.cache/shader-predict-ml/temp/*
   rm -rf ~/.cache/shader-predict-ml/ml_models/*
   systemctl --user restart shader-predict-ml.service
   ```

2. **Reinstall with fallback mode:**
   ```bash
   export SHADER_PREDICT_FALLBACK_MODE=1
   bash steamdeck-dependencies-fix.sh
   ```

---

## Performance Issues

### Problem: Slow Shader Compilation

**Symptoms:**
- Games take longer to load than expected
- Compilation queue backs up
- High I/O wait times

**Solutions:**

1. **Check cache size and location:**
   ```bash
   du -sh ~/.cache/shader-predict-ml/
   df -h ~/.cache/
   
   # Clean old cache entries
   find ~/.cache/shader-predict-ml/shaders/ -mtime +7 -delete
   ```

2. **Optimize for Steam Deck storage:**
   ```bash
   # Move cache to faster storage if available
   mkdir -p /tmp/shader-predict-cache
   ln -sf /tmp/shader-predict-cache ~/.cache/shader-predict-ml/temp
   ```

3. **Adjust compilation threads:**
   ```bash
   # Edit config to use fewer threads
   jq '.performance.max_worker_threads = 1' ~/.config/shader-predict-ml/config.json
   ```

### Problem: Memory Leaks

**Symptoms:**
- Memory usage grows over time
- System becomes sluggish
- Out of memory errors

**Diagnostics:**
```bash
# Monitor memory usage
watch -n 5 'ps aux | grep shader-predict'

# Check systemd memory accounting
systemctl --user show shader-predict-ml.service | grep Memory
```

**Solutions:**

1. **Enable memory limits:**
   ```bash
   systemctl --user set-property shader-predict-ml.service MemoryMax=300M
   systemctl --user restart shader-predict-ml.service
   ```

2. **Periodic service restart:**
   ```bash
   # Create timer for periodic restart
   systemctl --user edit --force --full shader-predict-ml-restart.timer
   # Add restart logic
   ```

---

## Steam Integration Problems

### Problem: Steam Doesn't Detect Compatibility Tool

**Symptoms:**
- Shader prediction tool not visible in Steam
- No option to enable per-game
- Steam integration not working

**Solutions:**

1. **Manually setup Steam integration:**
   ```bash
   # For native Steam
   ~/.local/share/shader-predict-ml/setup-steam-integration.sh
   
   # For Flatpak Steam
   flatpak run com.shaderpredict.MLCompiler --setup-steam
   ```

2. **Check Steam directories:**
   ```bash
   # Verify compatibility tool directory
   ls -la ~/.local/share/Steam/compatibilitytools.d/shader-predict-ml/
   
   # Check for Flatpak Steam
   ls -la ~/.var/app/com.valvesoftware.Steam/.local/share/Steam/compatibilitytools.d/
   ```

3. **Restart Steam:**
   ```bash
   steam -shutdown
   # Wait a few seconds
   steam &
   ```

### Problem: Per-Game Settings Not Applied

**Symptoms:**
- Game launches without shader prediction
- No performance improvement observed
- Game-specific profiles not loaded

**Solutions:**

1. **Verify game configuration:**
   ```bash
   ls ~/.config/shader-predict-ml/games/
   cat ~/.config/shader-predict-ml/games/<GAME_ID>.json
   ```

2. **Check Steam app ID detection:**
   ```bash
   # Monitor Steam environment variables
   grep -r SteamAppId /proc/*/environ 2>/dev/null
   ```

3. **Manual game profile creation:**
   ```bash
   # Create profile for specific game
   cat > ~/.config/shader-predict-ml/games/440_csgo.json << EOF
   {
     "game_id": "440",
     "name": "Counter-Strike: Global Offensive", 
     "shader_prediction_enabled": true,
     "optimization_level": "aggressive"
   }
   EOF
   ```

---

## Thermal and Battery Management

### Problem: Thermal Throttling Too Aggressive

**Symptoms:**
- Performance drops at low temperatures
- Service stops working in warm conditions
- Gaming performance affected

**Solutions:**

1. **Adjust thermal thresholds:**
   ```bash
   # Edit thermal configuration
   jq '.thermal.cpu_temp_warning = 85' ~/.config/shader-predict-ml/thermal-config.json > temp.json
   mv temp.json ~/.config/shader-predict-ml/thermal-config.json
   ```

2. **Check thermal zones:**
   ```bash
   python3 ~/.local/share/shader-predict-ml/steamdeck-thermal-manager.py --status
   ```

3. **Manual profile override:**
   ```bash
   python3 ~/.local/share/shader-predict-ml/steamdeck-thermal-manager.py --profile balanced
   ```

### Problem: Battery Drain Too High

**Symptoms:**
- Battery life significantly reduced
- High power consumption during idle
- Power saving not activating

**Solutions:**

1. **Force power saving mode:**
   ```bash
   python3 ~/.local/share/shader-predict-ml/steamdeck-thermal-manager.py --profile power_save
   ```

2. **Check battery monitoring:**
   ```bash
   cat /sys/class/power_supply/BAT1/capacity
   cat /sys/class/power_supply/BAT1/power_now
   ```

3. **Disable features for battery saving:**
   ```bash
   # Edit config to disable power-hungry features
   jq '.p2p_network.enabled = false | .ml_prediction.training_enabled = false' \
      ~/.config/shader-predict-ml/config.json
   ```

---

## Dependency Issues

### Problem: NumPy/SciKit-Learn Import Errors

**Symptoms:**
- "No module named numpy" errors
- ML features disabled
- Fallback mode activated

**Solutions:**

1. **Reinstall ML dependencies:**
   ```bash
   python3 -m pip uninstall numpy scikit-learn scipy -y
   python3 -m pip install --user --upgrade numpy==1.24.4 scikit-learn==1.3.2
   ```

2. **Use conda environment (if available):**
   ```bash
   conda create -n shader-predict python=3.10 numpy scikit-learn
   conda activate shader-predict
   ```

3. **Build from source (last resort):**
   ```bash
   # Install build dependencies
   sudo pacman -S gcc python-setuptools
   python3 -m pip install --user numpy --no-binary numpy
   ```

### Problem: Network Library Import Failures

**Symptoms:**
- P2P features not available
- "No module named aiohttp" errors
- Network functionality limited

**Solutions:**

1. **Install networking dependencies:**
   ```bash
   python3 -m pip install --user aiohttp cryptography requests
   ```

2. **Disable P2P if not needed:**
   ```bash
   export SHADER_PREDICT_P2P_DISABLED=1
   ~/.local/share/shader-predict-ml/launch.sh
   ```

---

## Flatpak Specific Issues

### Problem: Flatpak Build Fails

**Symptoms:**
- flatpak-builder exits with errors
- Missing dependencies in build
- Permission issues during build

**Solutions:**

1. **Install required SDKs:**
   ```bash
   flatpak install org.kde.Sdk//6.6 org.kde.Platform//6.6
   ```

2. **Build with verbose output:**
   ```bash
   flatpak-builder --verbose --repo=repo build-dir com.shaderpredict.MLCompiler.yml
   ```

3. **Clean build environment:**
   ```bash
   rm -rf build-dir .flatpak-builder
   flatpak-builder --force-clean build-dir com.shaderpredict.MLCompiler.yml
   ```

### Problem: Sandboxing Issues

**Symptoms:**
- Cannot access Steam directories
- Thermal monitoring not working
- File permission errors

**Solutions:**

1. **Grant additional permissions:**
   ```bash
   flatpak override --user --filesystem=~/.local/share/Steam:ro com.shaderpredict.MLCompiler
   flatpak override --user --device=all com.shaderpredict.MLCompiler
   ```

2. **Check current permissions:**
   ```bash
   flatpak info --show-permissions com.shaderpredict.MLCompiler
   ```

---

## Network and P2P Problems

### Problem: P2P Network Not Connecting

**Symptoms:**
- No peer connections established
- Network timeouts
- Firewall blocking connections

**Solutions:**

1. **Check network connectivity:**
   ```bash
   curl -I https://github.com
   ping -c 3 8.8.8.8
   ```

2. **Configure firewall:**
   ```bash
   # Open ports for P2P (if using iptables)
   sudo iptables -A INPUT -p tcp --dport 17700:17800 -j ACCEPT
   ```

3. **Disable P2P temporarily:**
   ```bash
   export SHADER_PREDICT_P2P_DISABLED=1
   ```

---

## Diagnostic Tools

### System Information Collection Script

```bash
#!/bin/bash
# Steam Deck system information collector

echo "=== Steam Deck System Information ==="
echo "Date: $(date)"
echo

echo "=== Hardware Information ==="
echo "Product: $(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'Unknown')"
echo "Board: $(cat /sys/devices/virtual/dmi/id/board_name 2>/dev/null || echo 'Unknown')"
echo "CPU: $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//')"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo

echo "=== Thermal State ==="
for zone in /sys/class/thermal/thermal_zone*/temp; do
    if [ -f "$zone" ]; then
        temp=$(cat "$zone")
        temp_c=$((temp / 1000))
        zone_name=$(basename $(dirname "$zone"))
        echo "$zone_name: ${temp_c}°C"
    fi
done
echo

echo "=== Battery State ==="
if [ -f /sys/class/power_supply/BAT1/capacity ]; then
    echo "Battery: $(cat /sys/class/power_supply/BAT1/capacity)%"
    echo "Status: $(cat /sys/class/power_supply/BAT1/status)"
fi
echo

echo "=== Service Status ==="
systemctl --user is-active shader-predict-ml.service
systemctl --user status shader-predict-ml.service --no-pager -l
echo

echo "=== Installation Check ==="
echo "Install directory: $(ls -la ~/.local/share/shader-predict-ml/ 2>/dev/null || echo 'Not found')"
echo "Config directory: $(ls -la ~/.config/shader-predict-ml/ 2>/dev/null || echo 'Not found')"
echo "Launcher: $(test -x ~/.local/share/shader-predict-ml/launch.sh && echo 'OK' || echo 'Missing')"
echo

echo "=== Dependency Check ==="
~/.local/bin/test-shader-deps 2>/dev/null || echo "Test script not found"
echo

echo "=== Recent Logs ==="
journalctl --user -u shader-predict-ml.service --since "1 hour ago" --no-pager
```

### Log Analysis Script

```bash
#!/bin/bash
# Analyze shader prediction logs for common issues

LOG_FILE="$HOME/.cache/shader-predict-ml/thermal_manager.log"
SERVICE_LOGS=$(journalctl --user -u shader-predict-ml.service --since "1 day ago" --no-pager)

echo "=== Log Analysis Results ==="

# Check for common error patterns
echo "Thermal emergencies: $(echo "$SERVICE_LOGS" | grep -c "thermal_emergency")"
echo "Memory warnings: $(echo "$SERVICE_LOGS" | grep -c "MemoryMax")"
echo "Dependency errors: $(echo "$SERVICE_LOGS" | grep -c "ImportError\|ModuleNotFoundError")"
echo "Network errors: $(echo "$SERVICE_LOGS" | grep -c "ConnectionError\|TimeoutError")"

# Recent critical events
echo -e "\n=== Recent Critical Events ==="
echo "$SERVICE_LOGS" | grep -E "(CRITICAL|ERROR)" | tail -5
```

---

## Recovery Procedures

### Complete Reinstallation

```bash
#!/bin/bash
# Complete reinstallation procedure

echo "Starting complete reinstallation..."

# Stop services
systemctl --user stop shader-predict-ml.service 2>/dev/null || true
systemctl --user disable shader-predict-ml.service 2>/dev/null || true

# Backup important data
mkdir -p ~/shader-predict-backup
cp -r ~/.config/shader-predict-ml/ ~/shader-predict-backup/ 2>/dev/null || true
cp -r ~/.local/share/shader-predict-ml/data/ ~/shader-predict-backup/ 2>/dev/null || true

# Remove old installation
rm -rf ~/.local/share/shader-predict-ml/
rm -rf ~/.config/shader-predict-ml/
rm -rf ~/.cache/shader-predict-ml/
rm -f ~/.config/systemd/user/shader-predict-ml*

# Reinstall
bash steamdeck-optimized-install.sh --force

echo "Reinstallation complete. Restore backup data if needed:"
echo "cp -r ~/shader-predict-backup/* ~/.config/shader-predict-ml/"
```

### Factory Reset Configuration

```bash
#!/bin/bash
# Reset to factory default configuration

# Backup current config
cp ~/.config/shader-predict-ml/config.json ~/.config/shader-predict-ml/config.json.backup

# Remove custom configurations
rm -rf ~/.config/shader-predict-ml/games/
rm -rf ~/.config/shader-predict-ml/profiles/
rm -f ~/.config/shader-predict-ml/thermal-config.json

# Restart service to regenerate defaults
systemctl --user restart shader-predict-ml.service

echo "Configuration reset to defaults. Backup saved as config.json.backup"
```

### Emergency Thermal Protection

```bash
#!/bin/bash
# Emergency procedure for thermal issues

echo "EMERGENCY: Activating thermal protection"

# Force thermal emergency profile
python3 ~/.local/share/shader-predict-ml/steamdeck-thermal-manager.py --profile thermal_emergency

# Stop resource-intensive services
systemctl --user stop shader-predict-ml.service

# Set maximum throttling
echo "Applying maximum resource limits..."
systemctl --user set-property shader-predict-ml.service CPUQuota=1%
systemctl --user set-property shader-predict-ml.service MemoryMax=50M

# Monitor temperature
echo "Monitoring temperature (Ctrl+C to stop):"
while true; do
    temp=$(cat /sys/class/thermal/thermal_zone0/temp)
    temp_c=$((temp / 1000))
    echo "CPU Temperature: ${temp_c}°C"
    if [ $temp_c -lt 70 ]; then
        echo "Temperature safe - consider restarting normal operation"
        break
    fi
    sleep 5
done
```

---

## Getting Help

### Information to Include in Bug Reports

1. **System Information:**
   ```bash
   uname -a
   cat /etc/os-release
   cat /sys/devices/virtual/dmi/id/product_name
   ```

2. **Installation Details:**
   ```bash
   ls -la ~/.local/share/shader-predict-ml/
   ~/.local/bin/test-shader-deps
   ```

3. **Service Status:**
   ```bash
   systemctl --user status shader-predict-ml.service
   journalctl --user -u shader-predict-ml.service --since "1 hour ago"
   ```

4. **Configuration:**
   ```bash
   cat ~/.config/shader-predict-ml/config.json
   python3 ~/.local/share/shader-predict-ml/steamdeck-thermal-manager.py --status
   ```

### Contact and Support

- **GitHub Issues**: [Repository Issues Page]
- **Community Discord**: [Discord Server Link]
- **Documentation**: [Wiki/Documentation Link]
- **Email Support**: support@shaderpredict.com

### Quick Reference Commands

```bash
# Service management
systemctl --user start shader-predict-ml.service
systemctl --user stop shader-predict-ml.service
systemctl --user status shader-predict-ml.service
systemctl --user restart shader-predict-ml.service

# Monitoring
journalctl --user -u shader-predict-ml.service -f
~/.local/bin/test-shader-deps
python3 ~/.local/share/shader-predict-ml/steamdeck-thermal-manager.py --status

# Configuration
cat ~/.config/shader-predict-ml/config.json
ls ~/.config/shader-predict-ml/games/
cat ~/.cache/shader-predict-ml/thermal_manager.log

# Troubleshooting
DEBUG=1 ~/.local/share/shader-predict-ml/launch.sh --test
bash steamdeck-dependencies-fix.sh
systemctl --user daemon-reload
```

---

This troubleshooting guide should help you resolve most common issues with the ML Shader Prediction Compiler on Steam Deck. If you encounter problems not covered here, please check the documentation or report an issue with detailed system information.