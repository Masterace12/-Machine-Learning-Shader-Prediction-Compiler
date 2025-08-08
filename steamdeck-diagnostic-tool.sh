#!/bin/bash
# Steam Deck ML Shader Compiler - Comprehensive Diagnostic Tool
# Identifies root causes of installation failures and provides specific fixes
# Version: 1.0 - Bulletproof Diagnostic Edition

set -euo pipefail
IFS=$'\n\t'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGNOSTIC_LOG="/tmp/steamdeck-diagnostic-$(date +%Y%m%d-%H%M%S).log"
FAILURE_COUNT=0
WARNING_COUNT=0
SUCCESS_COUNT=0

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          Steam Deck ML Shader Compiler - Diagnostic Tool         ║${NC}"
echo -e "${CYAN}║                    Installation Failure Analysis                 ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"

# Logging function
log_result() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" >> "$DIAGNOSTIC_LOG"
    
    case "$level" in
        "PASS")
            echo -e "${GREEN}✓${NC} $message"
            ((SUCCESS_COUNT++))
            ;;
        "FAIL")
            echo -e "${RED}✗${NC} $message"
            ((FAILURE_COUNT++))
            ;;
        "WARN")
            echo -e "${YELLOW}⚠${NC} $message"
            ((WARNING_COUNT++))
            ;;
        "INFO")
            echo -e "${BLUE}ℹ${NC} $message"
            ;;
    esac
}

# Test 1: Immutable Filesystem Detection
test_filesystem_state() {
    echo -e "\n${BLUE}[TEST 1/11]${NC} Checking Steam Deck filesystem state..."
    
    local is_immutable=false
    local root_readonly=false
    local system_writable=false
    
    # Check if root is mounted read-only
    if findmnt / | grep -q "ro,"; then
        root_readonly=true
        is_immutable=true
        log_result "INFO" "Root filesystem is read-only (immutable mode)"
    else
        log_result "PASS" "Root filesystem is writable (developer mode)"
    fi
    
    # Check SteamOS detection
    if [[ -f "/etc/os-release" ]] && grep -q "steamos" /etc/os-release; then
        local steamos_version=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
        log_result "INFO" "Detected SteamOS version: $steamos_version"
        
        if [[ "$root_readonly" == "true" ]]; then
            log_result "WARN" "SteamOS immutable filesystem detected - system packages unavailable"
            echo -e "   ${YELLOW}→${NC} Solution: Use user-space installation only"
        fi
    fi
    
    # Check writable directories
    local writable_dirs=("/home" "/tmp" "/var/tmp" "/opt")
    for dir in "${writable_dirs[@]}"; do
        if [[ -w "$dir" ]]; then
            log_result "PASS" "$dir is writable"
        else
            log_result "FAIL" "$dir is not writable"
        fi
    done
    
    # Test temporary directory space
    local tmp_avail=$(df -BM /tmp | tail -1 | awk '{print $4}' | sed 's/M//')
    if [[ "$tmp_avail" -gt 500 ]]; then
        log_result "PASS" "/tmp has sufficient space (${tmp_avail}MB available)"
    else
        log_result "FAIL" "/tmp has insufficient space (${tmp_avail}MB available)"
        echo -e "   ${RED}→${NC} Solution: Clear /tmp or use alternative temp directory"
        echo -e "   ${BLUE}Command:${NC} sudo rm -rf /tmp/dumps/* /tmp/steam_* /tmp/*.tmp"
    fi
}

# Test 2: Storage Space Analysis
test_storage_space() {
    echo -e "\n${BLUE}[TEST 2/11]${NC} Analyzing storage space..."
    
    # Check system partition
    local system_avail=$(df -BM / | tail -1 | awk '{print $4}' | sed 's/M//')
    if [[ "$system_avail" -gt 100 ]]; then
        log_result "PASS" "System partition has adequate space (${system_avail}MB)"
    else
        log_result "FAIL" "System partition critically low on space (${system_avail}MB)"
    fi
    
    # Check home partition  
    local home_avail=$(df -BM /home | tail -1 | awk '{print $4}' | sed 's/M//')
    if [[ "$home_avail" -gt 1000 ]]; then
        log_result "PASS" "Home partition has adequate space (${home_avail}MB)"
    elif [[ "$home_avail" -gt 500 ]]; then
        log_result "WARN" "Home partition space is limited (${home_avail}MB)"
    else
        log_result "FAIL" "Home partition critically low on space (${home_avail}MB)"
        echo -e "   ${RED}→${NC} Solution: Free up space in /home/deck"
        echo -e "   ${BLUE}Command:${NC} du -sh /home/deck/* | sort -hr | head -10"
    fi
    
    # Check for Steam dumps filling /tmp
    if [[ -d "/tmp/dumps" ]]; then
        local dumps_size=$(du -sm /tmp/dumps 2>/dev/null | cut -f1 || echo "0")
        if [[ "$dumps_size" -gt 100 ]]; then
            log_result "FAIL" "Steam dumps consuming excessive space (${dumps_size}MB in /tmp/dumps)"
            echo -e "   ${RED}→${NC} Solution: Clear Steam dumps"
            echo -e "   ${BLUE}Command:${NC} sudo rm -rf /tmp/dumps/*"
        fi
    fi
    
    # Check inode usage
    local inode_usage=$(df -i / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ "$inode_usage" -gt 90 ]]; then
        log_result "FAIL" "Inode usage critically high (${inode_usage}%)"
    elif [[ "$inode_usage" -gt 70 ]]; then
        log_result "WARN" "Inode usage elevated (${inode_usage}%)"
    else
        log_result "PASS" "Inode usage normal (${inode_usage}%)"
    fi
}

# Test 3: Build Dependencies Check
test_build_dependencies() {
    echo -e "\n${BLUE}[TEST 3/11]${NC} Checking build dependencies..."
    
    # Essential build tools
    local build_tools=("gcc" "g++" "make" "python3" "python3-dev" "pkg-config")
    local missing_tools=()
    
    for tool in "${build_tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_result "PASS" "$tool is available"
        else
            log_result "FAIL" "$tool is missing"
            missing_tools+=("$tool")
        fi
    done
    
    # Check for Python development headers
    if python3 -c "import sysconfig; print(sysconfig.get_path('include'))" >/dev/null 2>&1; then
        local python_include=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
        if [[ -d "$python_include" ]]; then
            log_result "PASS" "Python development headers available at $python_include"
        else
            log_result "FAIL" "Python development headers directory not found"
        fi
    else
        log_result "FAIL" "Python development configuration unavailable"
    fi
    
    # Check for Linux headers
    if [[ -d "/usr/include/linux" ]]; then
        log_result "PASS" "Linux headers available"
    else
        log_result "FAIL" "Linux headers missing"
        echo -e "   ${RED}→${NC} Solution: Install base-devel group or use pre-compiled wheels"
    fi
    
    # Check pacman availability
    if command -v pacman >/dev/null 2>&1; then
        if pacman -Sy --print 2>/dev/null >/dev/null; then
            log_result "PASS" "pacman is functional"
        else
            log_result "FAIL" "pacman database is locked or system is immutable"
            echo -e "   ${RED}→${NC} Solution: Use Flatpak or user-space installations"
        fi
    else
        log_result "INFO" "pacman not available (non-Arch system)"
    fi
    
    # Provide solutions for missing tools
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        echo -e "   ${RED}→${NC} Missing tools: ${missing_tools[*]}"
        echo -e "   ${BLUE}Solution 1:${NC} Enable developer mode and install base-devel"
        echo -e "   ${BLUE}Solution 2:${NC} Use pre-compiled wheels with: pip install --only-binary=all"
    fi
}

# Test 4: Permission and PATH Analysis  
test_permissions_and_path() {
    echo -e "\n${BLUE}[TEST 4/11]${NC} Analyzing permissions and PATH..."
    
    # Check current user
    local current_user=$(whoami)
    log_result "INFO" "Running as user: $current_user"
    
    # Check if script is executable
    if [[ -x "$0" ]]; then
        log_result "PASS" "Diagnostic script has execute permissions"
    else
        log_result "WARN" "Diagnostic script lacks execute permissions"
    fi
    
    # Check PATH configuration
    echo -e "   Current PATH: $PATH" >> "$DIAGNOSTIC_LOG"
    
    # Check for user bin in PATH
    if [[ ":$PATH:" == *":$HOME/.local/bin:"* ]]; then
        log_result "PASS" "User bin directory ($HOME/.local/bin) in PATH"
    else
        log_result "WARN" "User bin directory not in PATH"
        echo -e "   ${YELLOW}→${NC} Solution: Add to PATH with: export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
    
    # Check LD_PRELOAD interference
    if [[ -n "${LD_PRELOAD:-}" ]]; then
        log_result "WARN" "LD_PRELOAD is set: $LD_PRELOAD"
        echo -e "   ${YELLOW}→${NC} This may interfere with script execution"
        echo -e "   ${BLUE}Solution:${NC} unset LD_PRELOAD before running scripts"
    else
        log_result "PASS" "LD_PRELOAD is not set"
    fi
    
    # Check sudo availability
    if command -v sudo >/dev/null 2>&1; then
        if timeout 1 sudo -n true 2>/dev/null; then
            log_result "PASS" "sudo access available without password"
        else
            log_result "WARN" "sudo requires password or is unavailable"
            echo -e "   ${YELLOW}→${NC} User-space installation required"
        fi
    else
        log_result "INFO" "sudo not available"
    fi
    
    # Test directory creation permissions
    local test_dir="/tmp/steamdeck-permission-test-$$"
    if mkdir -p "$test_dir" 2>/dev/null; then
        log_result "PASS" "Can create directories in /tmp"
        rmdir "$test_dir" 2>/dev/null
    else
        log_result "FAIL" "Cannot create directories in /tmp"
    fi
    
    # Test home directory permissions
    local home_test_dir="$HOME/.steamdeck-test-$$"
    if mkdir -p "$home_test_dir" 2>/dev/null; then
        log_result "PASS" "Can create directories in home"
        rmdir "$home_test_dir" 2>/dev/null
    else
        log_result "FAIL" "Cannot create directories in home"
    fi
}

# Test 5: Thermal State Analysis
test_thermal_state() {
    echo -e "\n${BLUE}[TEST 5/11]${NC} Checking thermal state..."
    
    local thermal_zones=(/sys/class/thermal/thermal_zone*/temp)
    local max_temp=0
    local temp_count=0
    
    for zone in "${thermal_zones[@]}"; do
        if [[ -r "$zone" ]]; then
            local temp=$(cat "$zone" 2>/dev/null)
            if [[ -n "$temp" && "$temp" =~ ^[0-9]+$ ]]; then
                # Convert millicelsius to celsius
                temp=$((temp / 1000))
                if [[ "$temp" -gt "$max_temp" ]]; then
                    max_temp=$temp
                fi
                ((temp_count++))
            fi
        fi
    done
    
    if [[ "$temp_count" -gt 0 ]]; then
        if [[ "$max_temp" -lt 60 ]]; then
            log_result "PASS" "System temperature normal (${max_temp}°C)"
        elif [[ "$max_temp" -lt 75 ]]; then
            log_result "WARN" "System temperature elevated (${max_temp}°C)"
            echo -e "   ${YELLOW}→${NC} Installation may be slower due to thermal throttling"
        else
            log_result "FAIL" "System temperature critical (${max_temp}°C)"
            echo -e "   ${RED}→${NC} Installation should be delayed until system cools"
            echo -e "   ${BLUE}Solution:${NC} Wait or improve ventilation"
        fi
    else
        log_result "WARN" "Cannot read thermal sensors"
    fi
    
    # Check for thermal throttling in dmesg
    if dmesg 2>/dev/null | tail -20 | grep -qi "thermal\|throttl"; then
        log_result "WARN" "Recent thermal throttling detected in system logs"
    fi
    
    # Check CPU frequency to detect throttling
    if [[ -r /proc/cpuinfo ]]; then
        local cpu_freq=$(grep "cpu MHz" /proc/cpuinfo | head -1 | awk '{print $4}')
        if [[ -n "$cpu_freq" ]]; then
            local freq_int=${cpu_freq%.*}
            if [[ "$freq_int" -lt 2000 ]]; then
                log_result "WARN" "CPU frequency possibly throttled (${cpu_freq} MHz)"
            else
                log_result "PASS" "CPU frequency normal (${cpu_freq} MHz)"
            fi
        fi
    fi
}

# Test 6: Memory Analysis
test_memory_state() {
    echo -e "\n${BLUE}[TEST 6/11]${NC} Analyzing memory state..."
    
    # Get memory information
    local total_mem=$(free -m | awk 'NR==2{print $2}')
    local available_mem=$(free -m | awk 'NR==2{print $7}')
    local used_mem=$(free -m | awk 'NR==2{print $3}')
    local free_mem=$(free -m | awk 'NR==2{print $4}')
    
    log_result "INFO" "Memory: ${total_mem}MB total, ${available_mem}MB available, ${used_mem}MB used"
    
    # Check if sufficient memory for installation
    if [[ "$available_mem" -gt 2000 ]]; then
        log_result "PASS" "Sufficient memory for full installation (${available_mem}MB available)"
    elif [[ "$available_mem" -gt 1000 ]]; then
        log_result "WARN" "Limited memory - recommend minimal installation (${available_mem}MB available)"
    else
        log_result "FAIL" "Insufficient memory for installation (${available_mem}MB available)"
        echo -e "   ${RED}→${NC} Solution: Close other applications or restart system"
    fi
    
    # Check swap usage
    local swap_total=$(free -m | awk 'NR==3{print $2}')
    local swap_used=$(free -m | awk 'NR==3{print $3}')
    
    if [[ "$swap_total" -gt 0 ]]; then
        local swap_percent=$(( swap_used * 100 / swap_total ))
        if [[ "$swap_percent" -gt 50 ]]; then
            log_result "WARN" "Heavy swap usage (${swap_percent}%) may slow installation"
        else
            log_result "PASS" "Swap usage normal (${swap_percent}%)"
        fi
    else
        log_result "INFO" "No swap configured"
    fi
    
    # Check for memory pressure
    if [[ -f /proc/pressure/memory ]]; then
        local mem_pressure=$(grep "avg10=" /proc/pressure/memory | cut -d'=' -f2 | cut -d' ' -f1)
        if [[ "${mem_pressure%.*}" -gt 10 ]]; then
            log_result "WARN" "System under memory pressure (${mem_pressure})"
        fi
    fi
    
    # Check for OOM killer activity
    if dmesg 2>/dev/null | tail -50 | grep -qi "killed process\|out of memory"; then
        log_result "FAIL" "Recent out-of-memory kills detected"
        echo -e "   ${RED}→${NC} System has been killing processes due to memory pressure"
        echo -e "   ${BLUE}Solution:${NC} Restart system or close memory-intensive applications"
    fi
}

# Test 7: Network Connectivity
test_network_connectivity() {
    echo -e "\n${BLUE}[TEST 7/11]${NC} Testing network connectivity..."
    
    # Test basic connectivity
    if ping -c 1 -W 5 8.8.8.8 >/dev/null 2>&1; then
        log_result "PASS" "Internet connectivity available"
    else
        log_result "FAIL" "No internet connectivity"
        echo -e "   ${RED}→${NC} Solution: Check network connection or use offline installation"
        return
    fi
    
    # Test DNS resolution
    if nslookup pypi.org >/dev/null 2>&1; then
        log_result "PASS" "DNS resolution working"
    else
        log_result "FAIL" "DNS resolution failed"
        echo -e "   ${RED}→${NC} Solution: Check DNS settings or use IP addresses"
    fi
    
    # Test PyPI connectivity
    if timeout 10 curl -s -I https://pypi.org/ >/dev/null 2>&1; then
        log_result "PASS" "PyPI repository accessible"
    else
        log_result "FAIL" "Cannot reach PyPI repository"
        echo -e "   ${RED}→${NC} Solution: Check firewall or use alternative package index"
    fi
    
    # Test GitHub connectivity (for potential downloads)
    if timeout 10 curl -s -I https://github.com/ >/dev/null 2>&1; then
        log_result "PASS" "GitHub accessible"
    else
        log_result "WARN" "GitHub not accessible"
    fi
    
    # Test download speed
    local speed_test=$(timeout 10 curl -s -o /dev/null -w "%{speed_download}" http://speedtest.tele2.net/1MB.zip 2>/dev/null || echo "0")
    local speed_mb=$(echo "scale=1; $speed_test / 1048576" | bc 2>/dev/null || echo "0")
    
    if [[ "${speed_mb%.*}" -gt 0 ]]; then
        log_result "PASS" "Download speed: ${speed_mb} MB/s"
    else
        log_result "WARN" "Could not determine download speed"
    fi
}

# Test 8: Python Environment
test_python_environment() {
    echo -e "\n${BLUE}[TEST 8/11]${NC} Checking Python environment..."
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        local major_version=$(echo "$python_version" | cut -d'.' -f1)
        local minor_version=$(echo "$python_version" | cut -d'.' -f2)
        
        if [[ "$major_version" -eq 3 && "$minor_version" -ge 8 ]]; then
            log_result "PASS" "Python version compatible ($python_version)"
        else
            log_result "FAIL" "Python version incompatible ($python_version, need 3.8+)"
        fi
    else
        log_result "FAIL" "Python 3 not found"
        return
    fi
    
    # Check pip availability
    if python3 -m pip --version >/dev/null 2>&1; then
        local pip_version=$(python3 -m pip --version | awk '{print $2}')
        log_result "PASS" "pip available (version $pip_version)"
    else
        log_result "FAIL" "pip not available"
        echo -e "   ${RED}→${NC} Solution: python3 -m ensurepip --upgrade --user"
    fi
    
    # Check virtual environment capability
    if python3 -m venv --help >/dev/null 2>&1; then
        log_result "PASS" "Virtual environment support available"
    else
        log_result "WARN" "Virtual environment support missing"
    fi
    
    # Test basic module imports
    local test_modules=("ssl" "urllib" "json" "os" "sys")
    for module in "${test_modules[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            log_result "PASS" "Python module '$module' available"
        else
            log_result "FAIL" "Python module '$module' missing"
        fi
    done
    
    # Check site-packages permissions
    local site_packages=$(python3 -c "import site; print(site.getusersitepackages())")
    if [[ -d "$site_packages" ]]; then
        if [[ -w "$site_packages" ]]; then
            log_result "PASS" "User site-packages writable ($site_packages)"
        else
            log_result "FAIL" "User site-packages not writable ($site_packages)"
        fi
    else
        if mkdir -p "$site_packages" 2>/dev/null; then
            log_result "PASS" "Created user site-packages directory ($site_packages)"
        else
            log_result "FAIL" "Cannot create user site-packages directory ($site_packages)"
        fi
    fi
}

# Test 9: Steam Deck Hardware Detection
test_steam_deck_hardware() {
    echo -e "\n${BLUE}[TEST 9/11]${NC} Steam Deck hardware detection..."
    
    local is_steam_deck=false
    local steam_deck_model="unknown"
    
    # Check DMI information
    if [[ -f "/sys/class/dmi/id/product_name" ]]; then
        local product_name=$(cat /sys/class/dmi/id/product_name 2>/dev/null)
        if [[ "$product_name" == *"Jupiter"* ]]; then
            is_steam_deck=true
            steam_deck_model="LCD"
            log_result "PASS" "Steam Deck LCD detected via DMI ($product_name)"
        elif [[ "$product_name" == *"Galileo"* ]]; then
            is_steam_deck=true
            steam_deck_model="OLED"
            log_result "PASS" "Steam Deck OLED detected via DMI ($product_name)"
        elif [[ "$product_name" == *"Steam Deck"* ]]; then
            is_steam_deck=true
            log_result "PASS" "Steam Deck detected via DMI ($product_name)"
        fi
    fi
    
    # Check CPU information
    if grep -q "AMD Custom APU" /proc/cpuinfo; then
        is_steam_deck=true
        log_result "PASS" "Steam Deck APU detected"
    fi
    
    # Check hostname
    if [[ "$(hostname)" == *"steamdeck"* ]]; then
        is_steam_deck=true
        log_result "PASS" "Steam Deck hostname detected"
    fi
    
    if [[ "$is_steam_deck" == "false" ]]; then
        log_result "WARN" "Not running on Steam Deck hardware"
        echo -e "   ${YELLOW}→${NC} Continuing in compatibility mode"
    fi
    
    # Check gaming vs desktop mode
    if pgrep -f "steam.*-tenfoot" >/dev/null 2>&1; then
        log_result "INFO" "Currently in Gaming Mode"
        echo -e "   ${BLUE}→${NC} Some operations may require Desktop Mode"
    else
        log_result "INFO" "Currently in Desktop Mode"
    fi
    
    # Check memory configuration to differentiate models
    if [[ "$steam_deck_model" == "unknown" && "$is_steam_deck" == "true" ]]; then
        local total_mem=$(free -m | awk 'NR==2{print $2}')
        if [[ "$total_mem" -gt 14000 ]]; then
            steam_deck_model="OLED"
            log_result "INFO" "Steam Deck OLED detected via memory size (${total_mem}MB)"
        else
            steam_deck_model="LCD"
            log_result "INFO" "Steam Deck LCD detected via memory size (${total_mem}MB)"
        fi
    fi
    
    # Store model information for other tests
    echo "STEAM_DECK_MODEL=$steam_deck_model" >> "$DIAGNOSTIC_LOG"
}

# Test 10: Existing Installation Check
test_existing_installation() {
    echo -e "\n${BLUE}[TEST 10/11]${NC} Checking for existing installations..."
    
    # Check for existing installations in common locations
    local install_locations=(
        "$HOME/.local/share/shader-prediction-compiler"
        "$HOME/.local/share/ml-shader-predictor"
        "/home/deck/src"
        "/opt/shader-predict-compile"
    )
    
    local found_installations=()
    
    for location in "${install_locations[@]}"; do
        if [[ -d "$location" ]]; then
            found_installations+=("$location")
            log_result "WARN" "Existing installation found at $location"
        fi
    done
    
    if [[ ${#found_installations[@]} -eq 0 ]]; then
        log_result "PASS" "No conflicting installations found"
    else
        echo -e "   ${YELLOW}→${NC} Found ${#found_installations[@]} existing installation(s)"
        echo -e "   ${BLUE}Solution:${NC} Remove or backup existing installations before proceeding"
        for install in "${found_installations[@]}"; do
            echo -e "   ${BLUE}Command:${NC} mv '$install' '$install.backup.$(date +%Y%m%d)'"
        done
    fi
    
    # Check for running services
    if systemctl --user list-units 2>/dev/null | grep -q shader; then
        log_result "WARN" "Shader-related services already running"
        echo -e "   ${YELLOW}→${NC} Stop existing services before installation"
        echo -e "   ${BLUE}Command:${NC} systemctl --user stop shader-prediction-compiler.service"
    fi
    
    # Check for Python packages that might conflict
    local conflicting_packages=("numpy" "scikit-learn" "tensorflow" "torch")
    local found_conflicts=()
    
    for package in "${conflicting_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            local version=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null)
            log_result "INFO" "Found existing $package ($version)"
        fi
    done
}

# Test 11: Source Files Check
test_source_files() {
    echo -e "\n${BLUE}[TEST 11/11]${NC} Checking source files..."
    
    # Check if we can find source files
    local source_paths=(
        "$SCRIPT_DIR/src"
        "$SCRIPT_DIR/shader-prediction-compilation-main/shader-predict-compile/src"
        "$SCRIPT_DIR/../src"
        "$(dirname "$SCRIPT_DIR")/src"
    )
    
    local found_sources=false
    
    for src_path in "${source_paths[@]}"; do
        if [[ -d "$src_path" && -n "$(ls -A "$src_path" 2>/dev/null)" ]]; then
            log_result "PASS" "Source files found at: $src_path"
            found_sources=true
            
            # Check for key files
            local key_files=("shader_prediction_system.py" "ml_shader_predictor.py")
            for file in "${key_files[@]}"; do
                if [[ -f "$src_path/$file" ]]; then
                    log_result "PASS" "Key file found: $file"
                else
                    log_result "WARN" "Key file missing: $file"
                fi
            done
            break
        fi
    done
    
    if [[ "$found_sources" == "false" ]]; then
        log_result "FAIL" "No source files found"
        echo -e "   ${RED}→${NC} Searched locations:"
        for src_path in "${source_paths[@]}"; do
            echo -e "     - $src_path $([ -d "$src_path" ] && echo "[EXISTS]" || echo "[NOT FOUND]")"
        done
        echo -e "   ${BLUE}Solution:${NC} Ensure you're running from the correct directory"
    fi
    
    # Check script permissions
    local scripts=("steamdeck-quick-fix.sh" "steamdeck-optimized-install.sh" "install.sh")
    for script in "${scripts[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            if [[ -x "$SCRIPT_DIR/$script" ]]; then
                log_result "PASS" "Script '$script' is executable"
            else
                log_result "FAIL" "Script '$script' is not executable"
                echo -e "   ${RED}→${NC} Solution: chmod +x '$SCRIPT_DIR/$script'"
            fi
        fi
    done
}

# Generate comprehensive report
generate_report() {
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                        DIAGNOSTIC SUMMARY                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${GREEN}Passed:${NC} $SUCCESS_COUNT tests"
    echo -e "${YELLOW}Warnings:${NC} $WARNING_COUNT tests"  
    echo -e "${RED}Failed:${NC} $FAILURE_COUNT tests"
    
    echo -e "\n${BLUE}Diagnostic log saved to:${NC} $DIAGNOSTIC_LOG"
    
    # Provide recommendations based on failure patterns
    if [[ "$FAILURE_COUNT" -eq 0 ]]; then
        echo -e "\n${GREEN}✓ SYSTEM READY FOR INSTALLATION${NC}"
        echo -e "No critical issues detected. You can proceed with installation."
    elif [[ "$FAILURE_COUNT" -le 2 ]]; then
        echo -e "\n${YELLOW}⚠ MINOR ISSUES DETECTED${NC}"
        echo -e "Installation may succeed but consider addressing warnings first."
    else
        echo -e "\n${RED}✗ CRITICAL ISSUES DETECTED${NC}"
        echo -e "Installation likely to fail. Address critical issues before proceeding."
    fi
    
    # Specific recommendations
    echo -e "\n${BLUE}RECOMMENDED INSTALLATION METHOD:${NC}"
    
    local total_mem=$(free -m | awk 'NR==2{print $2}')
    local available_mem=$(free -m | awk 'NR==2{print $7}')
    
    if [[ "$FAILURE_COUNT" -le 1 && "$available_mem" -gt 2000 ]]; then
        echo -e "→ Use: ${GREEN}./steamdeck-bulletproof-installer.sh --full${NC}"
    elif [[ "$available_mem" -gt 1000 ]]; then
        echo -e "→ Use: ${YELLOW}./steamdeck-bulletproof-installer.sh --minimal${NC}"
    else
        echo -e "→ Use: ${RED}./steamdeck-recovery-tool.sh${NC} (after addressing issues)"
    fi
    
    echo -e "\n${BLUE}For detailed analysis, review:${NC} $DIAGNOSTIC_LOG"
}

# Main execution
main() {
    echo "Starting Steam Deck diagnostic at $(date)" > "$DIAGNOSTIC_LOG"
    
    test_filesystem_state
    test_storage_space  
    test_build_dependencies
    test_permissions_and_path
    test_thermal_state
    test_memory_state
    test_network_connectivity
    test_python_environment
    test_steam_deck_hardware
    test_existing_installation
    test_source_files
    
    generate_report
}

# Run main function
main "$@"

echo -e "\n${CYAN}Diagnostic complete. Run with specific installation script based on recommendations above.${NC}"