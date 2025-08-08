# Getting Started - Steam Deck Installation

## For Users Downloading from GitHub

If you've just downloaded this project from GitHub, follow these simple steps:

### Step 1: Extract and Navigate
```bash
# After downloading the zip file from GitHub:
unzip shader-prediction-compilation-main.zip
cd shader-prediction-compilation-main/shader-predict-compile
```

### Step 2: Run Setup (REQUIRED)
```bash
# This fixes file permissions and prepares installation
chmod +x setup.sh
./setup.sh
```

**Why is this needed?** When you download files from GitHub, they lose their executable permissions. The setup script automatically fixes this common issue.

### Step 3: Install
```bash
# After setup is complete, install with:
./install
```

That's it! The installer will:
- Auto-detect your Steam Deck model (LCD vs OLED)
- Install all dependencies
- Set up the background service
- Create desktop entries for Gaming Mode and Desktop Mode

## Troubleshooting

### "Permission denied" errors
```bash
# If you get permission errors, run setup first:
chmod +x setup.sh
./setup.sh
```

### Installation fails
```bash
# Try the manual installer:
./install-manual
```

### Want to test before installing?
```bash
# Run without installing:
./auto_launcher.sh
```

## What Gets Installed

- **Application**: `/opt/shader-predict-compile/`
- **Desktop Entry**: Gaming Mode and Desktop Mode integration
- **Background Service**: Automatic shader compilation
- **Configuration**: `~/.config/shader-predict-compile/`
- **Cache**: `~/.cache/shader-predict-compile/`

## Using the Application

### Gaming Mode
- Open Library → Non-Steam → "Shader Predictive Compiler"
- Controller-friendly interface optimized for handheld use

### Desktop Mode  
- Applications → Games → "Shader Predictive Compiler"
- Full desktop interface with all features

### Command Line
```bash
# Check service status
systemctl status shader-predict-compile

# View logs
journalctl -u shader-predict-compile -f

# Manual run
/opt/shader-predict-compile/launcher.sh
```

## Uninstalling

```bash
# Complete removal
./install --uninstall
```

This removes all files and optionally clears the shader cache.

---

For detailed documentation, see [README.md](README.md).