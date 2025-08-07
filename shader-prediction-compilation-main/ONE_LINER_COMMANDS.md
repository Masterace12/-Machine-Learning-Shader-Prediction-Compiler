# One-Liner Installation Commands

## 🚀 **Ultimate One-Liner Commands**

Copy and paste any of these commands into Konsole for instant installation:

### **Method 1: Direct Download & Install (Recommended)**
```bash
curl -sL https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/one-liner-install.sh | bash
```

### **Method 2: Git Clone & Install**
```bash
git clone https://github.com/Masterace12/shader-prediction-compilation.git && cd shader-prediction-compilation && bash INSTALL.sh
```

### **Method 3: Wget Download & Install**
```bash
wget -qO- https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/one-liner-install.sh | bash
```

### **Method 4: ZIP Download & Install**
```bash
curl -sL https://github.com/Masterace12/shader-prediction-compilation/archive/main.zip -o temp.zip && unzip -q temp.zip && cd shader-prediction-compilation-main && bash INSTALL.sh && cd .. && rm -rf temp.zip shader-prediction-compilation-main
```

## 🎯 **What These Commands Do**

1. **Automatically download** the project from GitHub
2. **Fix all GitHub issues** (permissions, line endings, etc.)
3. **Install dependencies** (Python, GTK, etc.)
4. **Install the application** to `/opt/shader-predict-compile`
5. **Set up desktop integration** and background service
6. **Configure Steam Deck optimizations** automatically
7. **Clean up** temporary files

## 🛠️ **Advanced One-Liners**

### **Install with Custom Options**
```bash
curl -sL https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/one-liner-install.sh | bash -s -- --option-name
```

### **Silent Installation (No Prompts)**
```bash
curl -sL https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/one-liner-install.sh | bash -s -- --silent
```

### **Install to Custom Directory**
```bash
curl -sL https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/one-liner-install.sh | bash -s -- --install-dir /custom/path
```

## ✨ **Benefits of One-Liner Installation**

- ✅ **Zero manual steps** - just paste and run
- ✅ **Automatic GitHub issue fixing** - no permission or line ending errors
- ✅ **Smart download method** - tries git, curl, wget automatically
- ✅ **Self-contained** - downloads and installs everything
- ✅ **Error handling** - fails gracefully with helpful messages
- ✅ **Clean installation** - removes temporary files automatically

## 🎮 **For Steam Deck Users**

Just open **Konsole** and paste one of these commands:

**Easiest:**
```bash
curl -sL https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/one-liner-install.sh | bash
```

That's it! The installer will:
1. Detect your Steam Deck model (LCD/OLED)
2. Install all dependencies automatically
3. Set up the background service
4. Add to your Steam library
5. Configure optimizations for your specific model

## 🗑️ **One-Liner Uninstall**

```bash
/opt/shader-predict-compile/install --uninstall
```

## 🛡️ **Security Note**

These commands download and execute scripts from GitHub. Only run commands from trusted sources. You can always:

1. **Inspect first**: Download the script and read it before running
2. **Use git clone**: Clone the repository and run the installer manually
3. **Check the source**: Review the GitHub repository before installing

## 📋 **Requirements**

The one-liner installer requires one of these tools (usually pre-installed on Steam Deck):
- `curl` or `wget` or `git`
- `unzip`
- `bash`

If missing, install with:
```bash
sudo pacman -S git curl wget unzip
```

## 🎉 **Example Usage**

```bash
# Open Konsole on Steam Deck
# Paste this command:
curl -sL https://raw.githubusercontent.com/Masterace12/shader-prediction-compilation/main/one-liner-install.sh | bash

# Watch the magic happen:
# ✓ Downloading from GitHub...
# ✓ Fixing GitHub issues...
# ✓ Installing dependencies...
# ✓ Installing application... 
# ✓ Setting up Steam integration...
# ✓ Installation complete!
```

**That's it!** Your Shader Predictive Compiler is now installed and ready to use! 🚀