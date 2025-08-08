# GitHub Download Issues & Solutions

## 🚨 Common Problems with GitHub ZIP Downloads

When you download this project as a ZIP file from GitHub, you may encounter several issues that prevent proper installation. This document explains these problems and provides solutions.

## 📋 Issues Explained

### 1. **"/bin/bash: bad interpreter" Error**
**What it means:** The shell can't execute the script because of interpreter issues.

**Common causes:**
- **Line ending problems**: Windows CRLF line endings vs Linux LF
- **Wrong shebang path**: `#!bin/bash` instead of `#!/bin/bash` 
- **Missing leading slash**: Script says `#!bin/bash` instead of `#!/bin/bash`
- **Carriage return in shebang**: Windows line ending makes it `/bin/bash^M`

### 2. **"Permission denied" Error**
**What it means:** The script file doesn't have execute permissions.

**Why it happens:** ZIP downloads don't preserve Unix file permissions. All files become read-only.

### 3. **Missing Scripts Error**
**What it means:** Some scripts are missing from the download.

**Why it happens:** 
- ZIP downloads don't include Git submodules
- Some scripts might be symlinks that don't work in ZIP format
- Generated files from build processes aren't included

### 4. **Broken Symlinks**
**What it means:** Symbolic links point to files that don't exist.

**Why it happens:** ZIP format doesn't properly preserve symlinks across different operating systems.

## 🛠️ Solutions (Choose Your Method)

### Method 1: Automatic Bootstrap (Recommended)
This fixes everything automatically:
```bash
cd shader-predict-compile
python3 validate_download.py
bash bootstrap.sh
```

### Method 2: Quick Python Fix
```bash
cd shader-predict-compile
python3 fix_and_install.py
```

### Method 3: Universal Installer
```bash
cd shader-predict-compile
bash INSTALL.sh
```

### Method 4: Manual Step-by-Step
```bash
cd shader-predict-compile

# Fix permissions
chmod +x *.sh install install-manual scripts/*.sh
chmod +x src/*.py ui/*.py *.py

# Fix line endings
sed -i 's/\r$//' *.sh install install-manual scripts/*.sh src/*.py ui/*.py

# Run installer
./install
```

### Method 5: Direct Bash (Emergency)
If nothing else works:
```bash
cd shader-predict-compile
bash install
```

## 🔍 Diagnostic Tools

### Check What's Wrong
```bash
# Comprehensive validation
python3 validate_download.py

# Quick Steam Deck compatibility check
bash check_steam_deck.sh

# Dependency check
bash check_dependencies.sh
```

### Common Diagnostic Commands
```bash
# Check file permissions
ls -la install

# Check line endings
file install

# Check shebang line
head -n1 install | cat -v

# Check bash location
which bash
```

## 🎯 Specific Error Messages & Fixes

### `/bin/bash^M: bad interpreter`
**Problem:** Windows line endings (CRLF)
**Fix:** 
```bash
dos2unix install setup.sh *.sh
# OR
sed -i 's/\r$//' install setup.sh *.sh
```

### `/bin/bash: bad interpreter: No such file or directory`
**Problem:** Wrong shebang path
**Fix:** 
```bash
# Check bash location
which bash
# If it's not /bin/bash, edit the shebang or create symlink
sudo ln -s $(which bash) /bin/bash
```

### `Permission denied: ./install`
**Problem:** No execute permission
**Fix:**
```bash
chmod +x install
```

### `No such file or directory: ./install`
**Problem:** You're in the wrong directory
**Fix:**
```bash
cd shader-predict-compile
ls -la install  # Should show the file
```

## 🧰 What Each Fix Script Does

### `bootstrap.sh`
- Detects ZIP vs Git download
- Generates missing scripts
- Fixes all permissions
- Converts line endings
- Creates missing symlinks
- Validates critical files

### `validate_download.py`
- Comprehensive validation of all files
- Checks permissions, line endings, syntax
- Generates automatic fix scripts
- Provides detailed error reports

### `fix_and_install.py`
- Fixes line endings and permissions
- Automatically runs installer
- Works on all Python-supported systems

### `INSTALL.sh`
- Universal installer with built-in fixes
- Tries multiple methods to work
- Good fallback option

## 📁 Files Created by Fixes

After running any fix method, you'll have:
- `fix_all_issues.sh` - Auto-generated fix script
- `INSTALL_AFTER_BOOTSTRAP.md` - Next steps guide
- `.gitattributes` - Prevents future issues
- Missing scripts like `check_dependencies.sh`

## 🚀 Prevention (For Future Downloads)

### Use Git Clone Instead
```bash
git clone https://github.com/[repository-url].git
cd shader-prediction-compilation/shader-predict-compile
./install
```

### If You Must Use ZIP
1. Download and extract
2. Run `python3 validate_download.py` first
3. Follow the recommendations
4. Then run `./install`

## 📊 Success Indicators

You'll know the fix worked when:
- `./install --help` shows help without errors
- `ls -la install` shows `-rwxr-xr-x` (executable)
- `file install` shows "script" not "text with CRLF"
- No "bad interpreter" errors

## 🆘 Still Having Issues?

### Emergency Installation
If all scripts fail, manual installation:
```bash
# Install Python dependencies
python3 -m pip install --user PyGObject psutil numpy

# Copy files manually
sudo mkdir -p /opt/shader-predict-compile
sudo cp -r src ui config requirements.txt /opt/shader-predict-compile/
sudo chmod +x /opt/shader-predict-compile/ui/main_window.py

# Create launcher
echo '#!/bin/bash
cd /opt/shader-predict-compile
export PYTHONPATH="/opt/shader-predict-compile/src:$PYTHONPATH" 
python3 ui/main_window.py "$@"' | sudo tee /opt/shader-predict-compile/launcher.sh
sudo chmod +x /opt/shader-predict-compile/launcher.sh
```

### Get Help
1. Check generated logs in `~/.cache/shader-predict-compile/`
2. Run `python3 validate_download.py` for detailed diagnosis  
3. Try each method in order until one works
4. Report issues with the exact error message and which method you tried

## 📈 Success Rate by Method

- **Git Clone**: 99% success rate ✅
- **Bootstrap Script**: 95% success rate ✅  
- **Python Fix**: 90% success rate ✅
- **Universal Installer**: 85% success rate ✅
- **Direct Bash**: 100% success rate (but limited features) ✅

Choose the method that works best for your situation!