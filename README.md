# Shader Predictive Compiler for Steam Deck

Enhance shader compilation for Steam games with intelligent prediction and compilation prioritization.

## ✨ Features

- **Intelligent Shader Prediction**: Pre-compiles shaders based on game patterns
- **Steam Deck Optimized**: Automatically detects LCD vs OLED models
- **Background Service**: Runs silently in the background
- **Gaming Mode Integration**: Works in both Desktop and Gaming modes
- **Resource Management**: CPU and memory limits to prevent system impact

## 🚀 **One-Line Installation (Recommended)**

Copy and paste this command into **Konsole** for instant installation:

```bash
curl -sL https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/web-install.sh | bash
```

**That's it!** The installer will automatically:
- ✅ Download the latest version from GitHub
- ✅ Fix all GitHub download issues (permissions, line endings)
- ✅ Install dependencies
- ✅ Detect your Steam Deck model (LCD vs OLED)
- ✅ Configure optimizations for your hardware
- ✅ Set up Steam integration

## 🔄 **Alternative Installation Methods**

**Git Clone Method:**
```bash
git clone https://github.com/Masterace12/shader-prediction-compilation.git /tmp/shader && cd /tmp/shader && bash INSTALL.sh
```

**Wget Method:**
```bash
wget -qO- https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/web-install.sh | bash
```

**Manual Download:**
1. Download ZIP from GitHub
2. Extract and navigate to folder
3. Run: `bash INSTALL.sh`

## 🎮 Steam Deck Compatibility

### Automatically Detects:
- **LCD Model**: 4 threads, 2GB memory limit
- **OLED Model**: 6 threads, 2.5GB memory limit  
- **SteamOS Version**: Ensures compatibility

### Works With:
- ✅ SteamOS 3.7+
- ✅ Desktop Mode
- ✅ Gaming Mode
- ✅ GitHub ZIP downloads
- ✅ Git clone installations

## 🔧 GitHub Download Issues - FIXED!

This installer automatically fixes all common GitHub download issues:
- ✅ Line ending problems (CRLF → LF)
- ✅ File permission issues
- ✅ Shebang line errors
- ✅ Missing files and symlinks

No separate fix scripts needed!

## 📁 What Gets Installed

- **Application**: `/opt/shader-predict-compile/`
- **Desktop Entry**: Applications → Games → Shader Predictive Compiler
- **Background Service**: Auto-starts with system
- **Configuration**: `~/.config/shader-predict-compile/`
- **Cache**: `~/.cache/shader-predict-compile/`

## 🛠️ Usage

### Desktop Mode
- Find in Applications → Games → Shader Predictive Compiler
- Or run: `/opt/shader-predict-compile/launcher.sh`

### Gaming Mode
- Library → Non-Steam → Shader Predictive Compiler

### Command Line
```bash
# Check status
systemctl status shader-predict-compile

# View logs
journalctl -u shader-predict-compile

# Manual launch
/opt/shader-predict-compile/launcher.sh
```

## 🔧 Troubleshooting

### Installation Issues
```bash
# If permission errors
bash INSTALL.sh

# Check dependencies
cd shader-predict-compile && ./check_dependencies.sh

# Validate installation
cd shader-predict-compile && ./validate_installation.sh
```

### Common Problems
- **"Permission denied"**: Run `bash INSTALL.sh`
- **"Not in directory"**: Make sure you're in the extracted folder
- **Missing dependencies**: The installer handles this automatically

## 📚 Documentation

See the `docs/` folder for detailed documentation:
- Installation guides
- Troubleshooting
- GitHub download fixes
- Configuration options

## 🗑️ Uninstall

```bash
cd shader-predict-compile
./install --uninstall
```

## 🎉 Success!

After installation, the Shader Predictive Compiler will:
1. Run automatically in the background
2. Detect when you launch Steam games
3. Pre-compile shaders intelligently
4. Improve game loading times
5. Optimize for your specific Steam Deck model

Enjoy enhanced gaming performance on your Steam Deck! 🎮
