# 🛠️ PGP Signature Troubleshooting Guide

## Steam Deck Shader Prediction Compiler Installation Issues

This guide addresses **PGP signature verification failures** that can occur during installation on Steam Deck and other Linux systems.

---

## 🔍 **Problem Identification**

### Common Error Messages:
```bash
error: failed to commit transaction (invalid or corrupted package (PGP signature))
error: database file for 'core' does not exist (use '-Sy' to download)
error: target not found: python
error: failed to prepare transaction (invalid or corrupted package)
```

### Root Causes:
- **Outdated or corrupted PGP keyring**
- **System clock synchronization issues**
- **Corrupted package cache files**
- **Network connectivity problems during downloads**
- **SteamOS-specific keyring conflicts**

---

## 🚀 **Quick Fix Solutions**

### **Option 1: Use Enhanced Installer (Recommended)**
```bash
# Use our enhanced installer that automatically fixes PGP issues
chmod +x enhanced-install.sh
./enhanced-install.sh
```

### **Option 2: Manual PGP Fix**
```bash
# Step 1: Clear package cache
sudo pacman -Scc --noconfirm

# Step 2: Reinitialize keyring
sudo rm -rf /etc/pacman.d/gnupg
sudo pacman-key --init

# Step 3: Populate keys
sudo pacman-key --populate archlinux
sudo pacman-key --populate steamos  # SteamOS only

# Step 4: Refresh from keyserver
sudo pacman-key --refresh-keys

# Step 5: Update package database
sudo pacman -Sy
```

### **Option 3: Emergency Installation**
```bash
# If all else fails, use dependency installer with PGP fixes
cd shader-predict-compile
chmod +x install_dependencies.sh
./install_dependencies.sh
```

---

## 🔧 **Detailed Solutions**

### **1. System Clock Synchronization**

PGP signatures are time-sensitive. Ensure your system clock is accurate:

```bash
# Check current time status
timedatectl status

# Enable automatic time synchronization
sudo timedatectl set-ntp true

# Force immediate sync
sudo chrony sources -v  # if using chrony
# OR
sudo systemctl restart systemd-timesyncd  # if using systemd-timesyncd

# Wait a few seconds, then check again
timedatectl status
```

**Expected Output:**
```
System clock synchronized: yes
NTP service: active
```

### **2. Network Connectivity Issues**

Verify you can reach package repositories:

```bash
# Test connectivity to Arch repositories
curl -I https://mirror.rackspace.com/archlinux/

# Test connectivity to keyserver
nc -zv keyserver.ubuntu.com 11371

# If behind a proxy, ensure it's configured:
export http_proxy="http://your-proxy:port"
export https_proxy="https://your-proxy:port"
```

### **3. SteamOS-Specific Issues**

SteamOS may have additional keyring requirements:

```bash
# Check SteamOS version
cat /etc/os-release

# For SteamOS 3.7+, ensure both keyrings are populated
sudo pacman-key --populate archlinux steamos

# Check available keyrings
ls -la /usr/share/pacman/keyrings/

# If steamos.gpg is missing, update SteamOS first
sudo steamos-update check
```

### **4. Corrupted Package Cache**

Clear and rebuild the package cache:

```bash
# Method 1: Complete cache cleanup
sudo pacman -Scc --noconfirm
sudo pacman -Syy

# Method 2: Remove specific corrupted packages
sudo rm -rf /var/cache/pacman/pkg/*
sudo pacman -Sy

# Method 3: Rebuild package database
sudo pacman-db-upgrade
sudo pacman -Sy
```

### **5. Alternative Package Sources**

If official repositories are problematic:

```bash
# Backup current mirrorlist
sudo cp /etc/pacman.d/mirrorlist /etc/pacman.d/mirrorlist.backup

# Use a fast, reliable mirror
echo 'Server = https://mirror.rackspace.com/archlinux/$repo/os/$arch' | sudo tee /etc/pacman.d/mirrorlist

# Update with new mirror
sudo pacman -Sy

# Restore original mirrorlist after installation
sudo mv /etc/pacman.d/mirrorlist.backup /etc/pacman.d/mirrorlist
```

---

## 🐛 **Advanced Debugging**

### **Verbose Package Manager Output**

Get detailed error information:

```bash
# Run with maximum verbosity
sudo pacman -Sy --debug --verbose

# Check pacman logs
sudo tail -f /var/log/pacman.log

# Check system journal for errors
journalctl -u pacman-init --no-pager
```

### **Manual Key Import**

If automatic key population fails:

```bash
# Import specific keys manually
sudo pacman-key --keyserver keyserver.ubuntu.com --recv-keys <KEY_ID>

# Example for Arch Linux master keys
sudo pacman-key --keyserver keyserver.ubuntu.com --recv-keys 4AA4767BBC9C4B1D18AE28B77F2D434B9741E8AC

# Trust imported keys
sudo pacman-key --lsign-key <KEY_ID>
```

### **Bypass Signature Verification (LAST RESORT)**

⚠️ **Security Warning**: Only use this temporarily and re-enable signature checking immediately after installation.

```bash
# Backup pacman.conf
sudo cp /etc/pacman.conf /etc/pacman.conf.backup

# Temporarily disable signature checking
sudo sed -i 's/SigLevel = Required DatabaseOptional/SigLevel = Never/' /etc/pacman.conf

# Install required packages
sudo pacman -S python python-pip python-gobject git

# IMMEDIATELY restore signature checking
sudo mv /etc/pacman.conf.backup /etc/pacman.conf

# Fix keyring properly after installation
sudo pacman-key --init
sudo pacman-key --populate archlinux
```

---

## 📋 **Installation Verification**

After fixing PGP issues, verify the installation:

```bash
# Test package manager
sudo pacman -Sy --noconfirm

# Verify Python installation
python3 --version
python3 -m pip --version

# Test critical modules
python3 -c "import gi; gi.require_version('Gtk', '3.0'); print('GTK bindings: OK')"
python3 -c "import psutil; print('psutil:', psutil.__version__)"

# Run our dependency checker
cd shader-predict-compile
chmod +x check_dependencies.sh
./check_dependencies.sh
```

---

## 🚨 **Emergency Fallback Methods**

### **Method 1: Use Flatpak Version**
```bash
# Install via Flatpak (isolated from system packages)
flatpak install flathub org.freedesktop.Platform.html5-codecs//21.08
# Note: Full Flatpak version in development
```

### **Method 2: Python Virtual Environment Only**
```bash
# Skip system packages, use Python virtual environment
python3 -m venv ~/.local/share/shader-predict-venv
source ~/.local/share/shader-predict-venv/bin/activate
pip install --upgrade pip
pip install numpy scikit-learn pandas psutil requests PyYAML

# Install our application in the virtual environment
cd shader-prediction-compilation
pip install -e .
```

### **Method 3: Docker Container**
```bash
# Run in isolated container (if Docker available)
docker run -it --rm \
  -v "$(pwd)":/workspace \
  -v "$HOME/.steam":/steam:ro \
  archlinux:latest bash

# Inside container:
pacman -Sy --noconfirm python python-pip git
cd /workspace
./install_dependencies.sh
```

---

## 📞 **Getting Help**

If these solutions don't work:

### **Collect Debug Information**
```bash
# System information
uname -a > debug_info.txt
cat /etc/os-release >> debug_info.txt
pacman --version >> debug_info.txt
python3 --version >> debug_info.txt

# Package manager status
pacman-key --list-keys | head -20 >> debug_info.txt
ls -la /etc/pacman.d/gnupg/ >> debug_info.txt

# Recent errors
journalctl -p err --since "1 hour ago" >> debug_info.txt
tail -50 /var/log/pacman.log >> debug_info.txt
```

### **Report Issues**
- **GitHub Issues**: [Create detailed bug report](https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/issues)
- **Include**: Your debug_info.txt file
- **Specify**: Exact error messages and steps taken

---

## ✅ **Prevention Tips**

### **Keep System Updated**
```bash
# Regular system maintenance
sudo pacman -Syu --noconfirm

# Update keyring regularly
sudo pacman -S archlinux-keyring
sudo pacman-key --refresh-keys
```

### **Monitor System Health**
```bash
# Check for corrupted packages
sudo pacman -Dk

# Verify package integrity
sudo pacman -Qkk | grep -v "0 missing files"

# Monitor disk space
df -h /var/cache/pacman/
```

### **Backup Critical Configurations**
```bash
# Backup working keyring
sudo tar czf ~/.cache/pacman-keyring-backup.tar.gz /etc/pacman.d/gnupg

# Backup working mirrorlist
cp /etc/pacman.d/mirrorlist ~/.cache/mirrorlist-backup
```

---

## 🎯 **Success Indicators**

Installation is ready when you see:

```bash
✓ System clock is synchronized
✓ Package database updated successfully  
✓ Python 3 found: 3.11.x
✓ pip is available
✓ GTK bindings: OK
✓ All dependencies satisfied
```

**Your Steam Deck Shader Prediction Compiler is ready for enhanced gaming performance!** 🎮

---

*This troubleshooting guide is specifically designed for the PGP signature verification issues commonly encountered during Steam Deck installations. For other installation problems, please refer to our main documentation.*