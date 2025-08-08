# 📦 Multiple Installation Methods Guide

## Steam Deck ML-Based Shader Prediction Compiler

This guide provides **multiple installation approaches** with automatic fallback options to ensure successful installation on any Steam Deck configuration.

---

## 🚀 **Method 1: Enhanced One-Line Installer (Recommended)**

**Best for**: Most users, automatic PGP fix, zero configuration

```bash
curl -fsSL https://raw.githubusercontent.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/main/enhanced-install.sh | bash
```

**Features:**
- ✅ Automatic PGP signature repair
- ✅ Steam Deck hardware detection (LCD/OLED)
- ✅ Thermal and power optimization
- ✅ Complete dependency resolution
- ✅ Service installation and configuration
- ✅ Comprehensive error recovery

---

## 🔧 **Method 2: Local Enhanced Installer**

**Best for**: Users who want to inspect the script first, or offline installation

```bash
# Download and inspect
curl -fsSL https://raw.githubusercontent.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/main/enhanced-install.sh -o enhanced-install.sh

# Review the script (optional)
less enhanced-install.sh

# Make executable and run
chmod +x enhanced-install.sh
./enhanced-install.sh
```

**Advantages:**
- 🔍 Can review script before execution
- 💾 Works with slow/intermittent internet
- 🛡️ Security-conscious approach

---

## 📁 **Method 3: Git Clone Installation**

**Best for**: Developers, users wanting latest features, version control

```bash
# Clone the repository
git clone https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler.git
cd -Machine-Learning-Shader-Prediction-Compiler

# Run enhanced installer
chmod +x enhanced-install.sh
./enhanced-install.sh

# OR use traditional installer
chmod +x optimized-install.sh
./optimized-install.sh
```

**Benefits:**
- 📈 Easy updates with `git pull`
- 🧪 Access to development features
- 📝 Full source code available
- 🔄 Version control integration

---

## 📦 **Method 4: Package-by-Package Installation**

**Best for**: Systems with persistent PGP issues, manual control

```bash
# Step 1: Fix PGP signatures manually
sudo pacman -Scc --noconfirm
sudo rm -rf /etc/pacman.d/gnupg
sudo pacman-key --init
sudo pacman-key --populate archlinux steamos
sudo pacman-key --refresh-keys
sudo pacman -Sy

# Step 2: Install core dependencies
sudo pacman -S --needed --noconfirm python python-pip python-gobject git curl

# Step 3: Download project
curl -L https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/archive/main.tar.gz | tar xz
cd -Machine-Learning-Shader-Prediction-Compiler-main

# Step 4: Python dependencies only
python3 -m venv ~/.local/share/shader-predict-venv
source ~/.local/share/shader-predict-venv/bin/activate
pip install -r requirements.txt

# Step 5: Manual configuration
mkdir -p ~/.config/shader-predict-compile
cp config/steam_deck_optimized.json ~/.config/shader-predict-compile/config.json
```

---

## 🐍 **Method 5: Python Virtual Environment Only**

**Best for**: Users without sudo access, minimal system impact

```bash
# Create isolated Python environment
python3 -m venv ~/shader-predict-env
source ~/shader-predict-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install numpy>=1.21.0 scikit-learn>=1.1.0 pandas>=1.4.0 psutil>=5.8.0 requests>=2.28.0 PyYAML>=6.0

# Install optional dependencies
pip install cryptography aiohttp pyudev dbus-python  # Linux-specific

# Download source
curl -L https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/archive/main.tar.gz | tar xz
cd -Machine-Learning-Shader-Prediction-Compiler-main

# Create configuration
mkdir -p ~/.config/shader-predict-compile
cat > ~/.config/shader-predict-compile/config.json << 'EOF'
{
  "system": {
    "steam_deck": true,
    "steam_deck_model": "LCD",
    "auto_optimize": true
  },
  "compilation": {
    "max_threads": 4,
    "memory_limit_mb": 2048,
    "thermal_aware": true
  },
  "ml_models": {
    "type": "ensemble",
    "cache_size": 1000
  }
}
EOF

# Create simple launcher
cat > ~/shader-predict-launcher.sh << 'EOF'
#!/bin/bash
source ~/shader-predict-env/bin/activate
cd ~/"-Machine-Learning-Shader-Prediction-Compiler-main"
python src/shader_prediction_system.py "$@"
EOF
chmod +x ~/shader-predict-launcher.sh

echo "✅ Installation complete! Run with: ~/shader-predict-launcher.sh"
```

---

## 🏗️ **Method 6: Development Installation**

**Best for**: Contributors, testers, advanced users

```bash
# Clone with development branches
git clone --recurse-submodules https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler.git
cd -Machine-Learning-Shader-Prediction-Compiler

# Switch to development branch (if available)
git checkout develop || git checkout main

# Install development dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run development installer
./enhanced-install.sh --dev
```

**Features:**
- 🧪 Latest development features
- 🔬 Debug logging enabled
- 📊 Performance profiling tools
- 🧹 Development utilities included

---

## 🚨 **Method 7: Emergency Fallback (Manual)**

**Best for**: When all other methods fail

```bash
# Step 1: Create minimal directory structure
mkdir -p ~/.local/bin ~/.config/shader-predict-compile ~/.cache/shader-predict-compile

# Step 2: Download core files manually
cd ~/.local/bin
curl -O https://raw.githubusercontent.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/main/src/shader_prediction_system.py
curl -O https://raw.githubusercontent.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/main/config/steam_deck_optimized.json

# Step 3: Move config
mv steam_deck_optimized.json ~/.config/shader-predict-compile/config.json

# Step 4: Install minimal Python dependencies
python3 -m pip install --user numpy scikit-learn psutil requests

# Step 5: Create executable launcher
cat > ~/.local/bin/shader-predict-compile << 'EOF'
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
exec(open('shader_prediction_system.py').read())
EOF
chmod +x ~/.local/bin/shader-predict-compile

# Step 6: Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo "✅ Emergency installation complete!"
```

---

## 📊 **Installation Method Comparison**

| Method | Difficulty | Features | Recovery | Best For |
|--------|------------|----------|----------|----------|
| **Enhanced One-Line** | ⭐ Easy | 🌟 Full | 🛡️ Auto | Most users |
| **Local Enhanced** | ⭐⭐ Easy+ | 🌟 Full | 🛡️ Auto | Security-conscious |
| **Git Clone** | ⭐⭐ Moderate | 🌟 Full + Updates | 🛡️ Auto | Developers |
| **Package-by-Package** | ⭐⭐⭐ Advanced | 🌟 Full | 🔧 Manual | PGP issues |
| **Python Venv Only** | ⭐⭐ Moderate | ⭐ Core | 🔧 Manual | Limited access |
| **Development** | ⭐⭐⭐⭐ Expert | 🌟 Full + Debug | 🛡️ Auto | Contributors |
| **Emergency Fallback** | ⭐⭐⭐ Advanced | ⭐ Minimal | 🔧 Manual | Last resort |

---

## 🧪 **Testing Your Installation**

After any installation method, verify it works:

```bash
# Method 1: Check installation paths
ls -la ~/.config/shader-predict-compile/
ls -la ~/.cache/shader-predict-compile/

# Method 2: Test Python dependencies
python3 -c "
import sys
modules = ['numpy', 'sklearn', 'psutil', 'requests']
for mod in modules:
    try:
        __import__(mod)
        print(f'✅ {mod}: OK')
    except ImportError:
        print(f'❌ {mod}: MISSING')
"

# Method 3: Test application startup
shader-predict-compile --version 2>/dev/null || echo "Command not in PATH"

# Method 4: Check Steam Deck optimizations
python3 -c "
import json, os
config_path = os.path.expanduser('~/.config/shader-predict-compile/config.json')
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    print(f'Steam Deck detected: {config.get(\"system\", {}).get(\"steam_deck\", False)}')
    print(f'Model: {config.get(\"system\", {}).get(\"steam_deck_model\", \"Unknown\")}')
    print(f'Threads: {config.get(\"compilation\", {}).get(\"max_threads\", 0)}')
else:
    print('❌ Configuration file not found')
"
```

### **Expected Output:**
```bash
✅ numpy: OK
✅ sklearn: OK  
✅ psutil: OK
✅ requests: OK
Steam Deck detected: True
Model: OLED  # or LCD
Threads: 6   # or 4 for LCD
```

---

## 🔄 **Switching Between Methods**

You can switch installation methods at any time:

```bash
# Remove current installation
uninstall-shader-predict-compile  # if system service installed
# OR
rm -rf ~/.config/shader-predict-compile ~/.cache/shader-predict-compile

# Install using different method
# (follow any method above)
```

---

## 🆘 **Getting Help**

**If installation fails:**

1. **Check our troubleshooting guide**: `TROUBLESHOOTING_PGP.md`
2. **Try the next method** in this list
3. **Report the issue**: Include output from failed method

**For support:**
- 📖 **Documentation**: [GitHub Wiki](https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/wiki)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/issues)
- 💬 **Community**: [GitHub Discussions](https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/discussions)

---

## 📈 **Performance Verification**

After installation, verify shader prediction is working:

```bash
# Start the service
systemctl --user start shader-predict-compile  # System service method
# OR
~/shader-predict-launcher.sh --service &  # Virtual env method

# Monitor logs
journalctl --user -u shader-predict-compile -f  # System service
# OR  
tail -f ~/.cache/shader-predict-compile/logs/service.log  # Virtual env

# Launch a game and observe shader compilation improvements
# Look for: "Predicted shader compilation" messages
```

**Success indicators:**
- 📊 Reduced shader compilation stutters
- ⚡ Faster game loading times
- 🎮 Smoother gameplay experience
- 📈 Background compilation during loading screens

**🎉 Enjoy your enhanced Steam Deck gaming performance!**