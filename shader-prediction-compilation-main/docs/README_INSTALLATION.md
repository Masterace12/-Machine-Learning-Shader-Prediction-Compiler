# Shader Predictive Compiler - Installation Methods

If you're getting "/bin/bash: bad interpreter" errors when trying to run the installation scripts after downloading from GitHub, this is a common issue caused by line ending differences between Windows and Linux systems.

## Quick Fix Methods

### Method 1: Python Fix (Recommended)
This method works on all systems and fixes line endings automatically:

```bash
cd shader-predict-compile
python3 fix_and_install.py
```

### Method 2: Universal Shell Script
```bash
cd shader-predict-compile
bash INSTALL.sh
```

### Method 3: Manual Line Ending Fix
If you have `dos2unix` installed:
```bash
cd shader-predict-compile
dos2unix install setup.sh *.sh scripts/*.sh
chmod +x install setup.sh *.sh scripts/*.sh
./install
```

If you don't have `dos2unix`:
```bash
cd shader-predict-compile
bash quick_fix.sh
./install
```

### Method 4: Direct Bash Execution
Bypass the interpreter error by running scripts with bash directly:
```bash
cd shader-predict-compile
bash install
```

## Step-by-Step Troubleshooting

### 1. Check Your Location
Make sure you're in the correct directory:
```bash
cd /home/deck/Downloads/shader-prediction-compilation-main/shader-predict-compile
ls -la install
```

### 2. Fix Line Endings
The most common issue is Windows line endings (CRLF) vs Linux line endings (LF):

**Option A - Install dos2unix:**
```bash
sudo pacman -S dos2unix
dos2unix install setup.sh *.sh
```

**Option B - Use sed:**
```bash
sed -i 's/\r$//' install setup.sh *.sh scripts/*.sh
```

**Option C - Use the Python fixer:**
```bash
python3 fix_and_install.py
```

### 3. Fix Permissions
```bash
chmod +x install setup.sh *.sh scripts/*.sh
chmod +x src/*.py ui/*.py *.py
```

### 4. Check Bash Location
```bash
which bash
```
If bash is not at `/bin/bash`, you may need to edit the shebang lines or create a symlink.

## Alternative Installation Methods

### If Standard Install Fails:

1. **Check compatibility first:**
   ```bash
   bash check_steam_deck.sh
   ```

2. **Install dependencies separately:**
   ```bash
   bash install_dependencies.sh
   ```

3. **Run manual installer:**
   ```bash
   bash install-manual
   ```

### Emergency Installation (No Scripts Working):

If none of the scripts work, you can install manually:

1. **Install Python dependencies:**
   ```bash
   python3 -m pip install --user PyGObject psutil numpy
   ```

2. **Copy files manually:**
   ```bash
   sudo mkdir -p /opt/shader-predict-compile
   sudo cp -r src ui config requirements.txt /opt/shader-predict-compile/
   sudo chmod +x /opt/shader-predict-compile/ui/main_window.py
   sudo chmod +x /opt/shader-predict-compile/src/background_service.py
   ```

3. **Create launcher:**
   ```bash
   echo '#!/bin/bash
   cd /opt/shader-predict-compile
   export PYTHONPATH="/opt/shader-predict-compile/src:$PYTHONPATH"
   python3 ui/main_window.py "$@"' | sudo tee /opt/shader-predict-compile/launcher.sh
   sudo chmod +x /opt/shader-predict-compile/launcher.sh
   ```

## Common Error Messages and Fixes

### "/bin/bash: bad interpreter: No such file or directory"
- **Cause:** Wrong shebang line or line ending issues
- **Fix:** Run `bash script_name.sh` instead of `./script_name.sh`

### "/bin/bash^M: bad interpreter"
- **Cause:** Windows line endings (CRLF)
- **Fix:** Use `dos2unix script_name.sh` or `sed -i 's/\r$//' script_name.sh`

### "Permission denied"
- **Cause:** File not executable
- **Fix:** Run `chmod +x script_name.sh`

### "No such file or directory" (for the script itself)
- **Cause:** Wrong directory or filename
- **Fix:** Check you're in the right directory with `ls -la`

## Testing the Fix

After fixing, test with:
```bash
./install --help
```

If this works, you can proceed with the full installation:
```bash
./install
```

## Getting Help

If you're still having issues:

1. **Check the logs:**
   ```bash
   cat ~/.cache/shader-predict-compile/launcher.log
   ```

2. **Try the compatibility checker:**
   ```bash
   bash check_steam_deck.sh
   ```

3. **Report the issue with:**
   - Your operating system version
   - The exact error message
   - Which method you tried

## Files Created by This Fix

- `fix_and_install.py` - Python-based universal fixer
- `INSTALL.sh` - Universal shell installer
- `quick_fix.sh` - Simple line ending fixer
- `fix_github_download.sh` - Comprehensive fix script
- `.gitattributes` - Prevents future line ending issues

Choose the method that works best for your system!