# Steam Deck Installation Fix Guide

## Problem Summary

The Steam Deck ML-Based Shader Prediction Compiler installation is failing due to several Steam Deck-specific issues:

### 1. **Missing pip3 Command**
- **Error**: `bash: pip3: command not found`
- **Cause**: Steam Deck's immutable filesystem and Python packaging differences
- **Solution**: Use Python virtual environments instead of system pip

### 2. **Incorrect File Paths**
- **Error**: `[Errno 2] No such file or directory: '/home/deck/src/shader_prediction_system.py'`
- **Cause**: Installation script assumes files are in `/home/deck/src/` instead of proper installation directory
- **Solution**: Correct all paths to use `~/.local/share/shader-predict-compile/`

### 3. **Immutable Filesystem Issues**
- **Cause**: Steam Deck's `/usr` is read-only, preventing system-wide installations
- **Solution**: Install everything in user directories (`~/.local/`)

## Quick Fix Instructions

### Method 1: Automated Fix Script (Recommended)

1. **Navigate to the project directory**:
   ```bash
   cd ~/Downloads/-Machine-Learning-Shader-Prediction-Compiler-main
   ```

2. **Run the fix script**:
   ```bash
   bash steam-deck-fix.sh
   ```

   This script will:
   - Create a Python virtual environment
   - Install all dependencies correctly
   - Fix all file paths
   - Create proper launcher scripts
   - Set up systemd services with resource limits

3. **Verify installation**:
   ```bash
   ~/.local/share/shader-predict-compile/launcher.sh --help
   ```

### Method 2: Manual Fix Steps

If the automated script fails, follow these manual steps:

#### Step 1: Fix Python Environment

```bash
# Install Python if not present
sudo pacman -S python python-pip

# Create installation directory
mkdir -p ~/.local/share/shader-predict-compile
cd ~/.local/share/shader-predict-compile

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 2: Install Dependencies

```bash
# Core dependencies
pip install numpy>=1.19.0 psutil>=5.8.0 requests>=2.25.0

# Optional ML dependencies
pip install scikit-learn pandas joblib

# Optional GUI dependencies
pip install PyQt6  # or PyQt5 if PyQt6 fails
```

#### Step 3: Copy Files to Correct Location

```bash
# Assuming you're in the downloaded project directory
PROJECT_DIR="~/Downloads/-Machine-Learning-Shader-Prediction-Compiler-main"
INSTALL_DIR="~/.local/share/shader-predict-compile"

# Copy source files
cp -r $PROJECT_DIR/src $INSTALL_DIR/
cp -r $PROJECT_DIR/config ~/.config/shader-predict-compile/

# Also check nested directories
if [ -d "$PROJECT_DIR/shader-prediction-compilation-main/shader-predict-compile/src" ]; then
    cp -r $PROJECT_DIR/shader-prediction-compilation-main/shader-predict-compile/src/* $INSTALL_DIR/src/
fi
```

#### Step 4: Create Launcher Script

Create `~/.local/share/shader-predict-compile/launcher.sh`:

```bash
#!/bin/bash
export SHADER_PREDICT_HOME="$HOME/.local/share/shader-predict-compile"
export PYTHONPATH="${SHADER_PREDICT_HOME}/src:${PYTHONPATH}"
cd "$SHADER_PREDICT_HOME"
source venv/bin/activate

if [ -f "src/shader_prediction_system.py" ]; then
    exec python src/shader_prediction_system.py "$@"
elif [ -f "src/main.py" ]; then
    exec python src/main.py "$@"
else
    echo "Error: No main Python file found"
    exit 1
fi
```

Make it executable:
```bash
chmod +x ~/.local/share/shader-predict-compile/launcher.sh
```

## Steam Deck-Specific Optimizations

### Resource Limits

The fix script configures the following Steam Deck-appropriate limits:

- **Memory**: 500MB maximum (to prevent impacting games)
- **CPU**: 10% quota (runs on efficiency cores when possible)
- **Thermal**: Pauses at 75°C to prevent throttling
- **I/O**: Low priority to prevent storage bottlenecks

### Systemd Service

After installation, you can manage the service with:

```bash
# Start service
systemctl --user start shader-predict.service

# Stop service
systemctl --user stop shader-predict.service

# Check status
systemctl --user status shader-predict.service

# Enable auto-start
systemctl --user enable shader-predict.service

# View logs
journalctl --user -u shader-predict -f
```

### Game Mode Integration

The service automatically detects when you're in Game Mode and adjusts its behavior:
- Reduces resource usage during gameplay
- Increases activity during game downloads/updates
- Pauses during thermal events

## Troubleshooting

### Issue: "Permission denied" errors

```bash
# Fix permissions
chmod -R u+rwX ~/.local/share/shader-predict-compile
chmod -R u+rwX ~/.config/shader-predict-compile
chmod -R u+rwX ~/.cache/shader-predict-compile
```

### Issue: Python module not found

```bash
# Ensure virtual environment is activated
source ~/.local/share/shader-predict-compile/venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Service won't start

```bash
# Check service logs
journalctl --user -u shader-predict -n 50

# Reset service
systemctl --user daemon-reload
systemctl --user reset-failed shader-predict.service
```

### Issue: High CPU/Memory usage

Edit `~/.config/shader-predict-compile/settings.json`:

```json
{
    "resource_limits": {
        "max_memory_mb": 300,
        "max_cpu_percent": 5,
        "thermal_throttle_temp": 70
    }
}
```

## Verification Steps

After installation, verify everything works:

1. **Check installation**:
   ```bash
   ls -la ~/.local/share/shader-predict-compile/
   ```

2. **Test Python environment**:
   ```bash
   ~/.local/share/shader-predict-compile/venv/bin/python -c "import numpy; print('NumPy OK')"
   ```

3. **Test launcher**:
   ```bash
   ~/.local/share/shader-predict-compile/launcher.sh --version
   ```

4. **Check service**:
   ```bash
   systemctl --user status shader-predict.service
   ```

## Complete Uninstall

To completely remove the installation:

```bash
# Stop and disable service
systemctl --user stop shader-predict.service
systemctl --user disable shader-predict.service

# Remove files
rm -rf ~/.local/share/shader-predict-compile
rm -rf ~/.config/shader-predict-compile
rm -rf ~/.cache/shader-predict-compile
rm -f ~/.local/bin/shader-predict-compile
rm -f ~/.local/share/applications/shader-predict-compile.desktop
rm -f ~/.config/systemd/user/shader-predict.service

# Reload systemd
systemctl --user daemon-reload
```

## Support

If you continue to experience issues after following this guide:

1. Run the fix script with verbose mode:
   ```bash
   bash steam-deck-fix.sh --verbose > fix-log.txt 2>&1
   ```

2. Check the log file for specific errors

3. Report issues with the log file attached

## Summary

The main issues were:
1. **pip3 not found** - Resolved by using Python virtual environments
2. **Wrong file paths** - Fixed by installing to `~/.local/share/` instead of `/home/deck/src/`
3. **Immutable filesystem** - Handled by using user directories exclusively

The provided `steam-deck-fix.sh` script automates all these fixes and creates a proper Steam Deck-optimized installation with:
- Correct file locations
- Python virtual environment with all dependencies
- Resource-limited systemd service
- Proper launcher scripts
- Steam Deck thermal management