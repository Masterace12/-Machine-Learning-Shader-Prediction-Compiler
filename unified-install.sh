#!/bin/bash
# ML Shader Prediction Compiler - Unified Installation Script
# Consolidates all installation methods into a single bulletproof installer
# Supports: Steam Deck (LCD/OLED), Generic Linux, Flatpak, Container deployment

set -euo pipefail

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

readonly SCRIPT_VERSION="2.0.0"
readonly REPO_OWNER="Masterace12"
readonly REPO_NAME="-Machine-Learning-Shader-Prediction-Compiler"
readonly REPO_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}"
readonly RAW_URL="https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/main"
readonly API_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}"

# Installation paths
readonly SYSTEM_INSTALL_DIR="/opt/ml-shader-predictor"
readonly USER_INSTALL_DIR="${HOME}/.local/share/ml-shader-predictor"
readonly CONFIG_DIR="${HOME}/.config/ml-shader-predictor"
readonly CACHE_DIR="${HOME}/.cache/ml-shader-predictor"
readonly TEMP_DIR="/tmp/ml-shader-install-$$"

# Colors and formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly PURPLE='\033[0;35m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Installation options (defaults)
INSTALL_MODE="auto"          # auto, system, user, flatpak
INSTALL_DEV=false
ENABLE_AUTOSTART=true
ENABLE_P2P=true
ENABLE_ML=true
ENABLE_GPU=true
ENABLE_THERMAL=true
FORCE_REINSTALL=false
SKIP_DEPS=false
QUIET_MODE=false
DRY_RUN=false

# Detected system information
DETECTED_OS=""
DETECTED_ARCH=""
DETECTED_DISTRO=""
PKG_MANAGER=""
STEAM_DECK_DETECTED=false
STEAM_DECK_MODEL="unknown"
GAMING_MODE_ACTIVE=false

# ============================================================================
# LOGGING AND OUTPUT FUNCTIONS
# ============================================================================

log_header() {
    if [[ "$QUIET_MODE" != "true" ]]; then
        echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}${CYAN}  $1${NC}"
        echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}\n"
    fi
}

log_info() { 
    echo -e "${BLUE}[INFO]${NC} $1" 
}

log_success() { 
    echo -e "${GREEN}[✓]${NC} $1" 
}

log_warning() { 
    echo -e "${YELLOW}[!]${NC} $1" 
}

log_error() { 
    echo -e "${RED}[✗]${NC} $1" >&2
}

log_debug() {
    if [[ "${DEBUG:-}" == "1" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1" >&2
    fi
}

log_step() {
    if [[ "$QUIET_MODE" != "true" ]]; then
        echo -e "${CYAN}${BOLD}[STEP]${NC} $1"
    fi
}

# ============================================================================
# ERROR HANDLING AND CLEANUP
# ============================================================================

cleanup() {
    local exit_code=$?
    
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR" 2>/dev/null || true
        log_debug "Cleaned up temporary directory: $TEMP_DIR"
    fi
    
    # Stop any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Installation failed with exit code $exit_code"
        log_info "For support, visit: ${REPO_URL}/issues"
        log_info "Include system information: $(uname -a)"
    fi
    
    exit $exit_code
}

trap cleanup EXIT SIGINT SIGTERM

error_handler() {
    local line_no=$1
    local error_code=$2
    log_error "Error occurred at line $line_no: exit code $error_code"
    log_info "Run with DEBUG=1 for detailed logging"
    exit "$error_code"
}

trap 'error_handler ${LINENO} $?' ERR

# ============================================================================
# SYSTEM DETECTION AND ANALYSIS
# ============================================================================

detect_system() {
    log_step "Analyzing system configuration..."
    
    # Detect OS
    case "$OSTYPE" in
        linux-gnu*)
            DETECTED_OS="linux"
            ;;
        msys|cygwin)
            DETECTED_OS="windows"
            log_warning "Windows detected - using WSL compatibility mode"
            ;;
        darwin*)
            DETECTED_OS="macos"
            log_error "macOS is not currently supported"
            exit 1
            ;;
        *)
            DETECTED_OS="unknown"
            log_warning "Unknown OS: $OSTYPE"
            ;;
    esac
    
    # Detect architecture
    DETECTED_ARCH=$(uname -m)
    case $DETECTED_ARCH in
        x86_64|amd64) DETECTED_ARCH="x86_64" ;;
        aarch64|arm64) DETECTED_ARCH="arm64" ;;
        armv7l) DETECTED_ARCH="armv7" ;;
        *) log_warning "Unsupported architecture: $DETECTED_ARCH" ;;
    esac
    
    # Detect Linux distribution
    if [[ "$DETECTED_OS" == "linux" ]]; then
        if [[ -f "/etc/os-release" ]]; then
            . /etc/os-release
            DETECTED_DISTRO="$ID"
            log_debug "Detected distribution: $PRETTY_NAME"
            
            # Check for SteamOS (Steam Deck)
            if [[ "$ID" == "steamos" ]] || [[ "${ID_LIKE:-}" == *"arch"* && "$NAME" == *"SteamOS"* ]]; then
                STEAM_DECK_DETECTED=true
                log_info "SteamOS detected - enabling Steam Deck optimizations"
            fi
        fi
    fi
    
    # Detect package manager
    if command -v pacman >/dev/null 2>&1; then
        PKG_MANAGER="pacman"
    elif command -v apt >/dev/null 2>&1; then
        PKG_MANAGER="apt"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
    elif command -v zypper >/dev/null 2>&1; then
        PKG_MANAGER="zypper"
    elif command -v apk >/dev/null 2>&1; then
        PKG_MANAGER="apk"
    else
        PKG_MANAGER="unknown"
        log_warning "No supported package manager found"
    fi
    
    # Enhanced Steam Deck detection
    detect_steam_deck_hardware
    
    # Check if Gaming Mode is active
    if pgrep -x "gamescope" >/dev/null 2>&1 || [[ "${XDG_CURRENT_DESKTOP:-}" == "gamescope" ]]; then
        GAMING_MODE_ACTIVE=true
        log_info "Gaming Mode detected - adjusting installation approach"
    fi
    
    log_success "System detected: $DETECTED_OS ($DETECTED_ARCH), Package Manager: $PKG_MANAGER"
    if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
        log_success "Steam Deck Model: $STEAM_DECK_MODEL"
    fi
}

detect_steam_deck_hardware() {
    # Use enhanced Steam Deck detection from our specialized module
    local confidence=0
    local detection_methods=0
    
    # Method 1: DMI detection
    local dmi_paths=(
        "/sys/class/dmi/id/board_name"
        "/sys/devices/virtual/dmi/id/product_name"
        "/sys/class/dmi/id/product_name"
    )
    
    for path in "${dmi_paths[@]}"; do
        if [[ -f "$path" ]]; then
            local dmi_value
            dmi_value=$(cat "$path" 2>/dev/null || echo "")
            if [[ "$dmi_value" == *"Jupiter"* ]]; then
                STEAM_DECK_DETECTED=true
                STEAM_DECK_MODEL="LCD"
                ((confidence += 30))
                ((detection_methods++))
            elif [[ "$dmi_value" == *"Galileo"* ]]; then
                STEAM_DECK_DETECTED=true
                STEAM_DECK_MODEL="OLED"
                ((confidence += 30))
                ((detection_methods++))
            elif [[ "$dmi_value" == *"Valve"* ]]; then
                STEAM_DECK_DETECTED=true
                ((confidence += 20))
                ((detection_methods++))
            fi
        fi
    done
    
    # Method 2: CPU detection
    if [[ -f "/proc/cpuinfo" ]]; then
        local cpu_info
        cpu_info=$(cat /proc/cpuinfo)
        if [[ "$cpu_info" == *"Custom APU 0405"* ]]; then
            STEAM_DECK_DETECTED=true
            STEAM_DECK_MODEL="LCD"
            ((confidence += 25))
            ((detection_methods++))
        elif [[ "$cpu_info" == *"Custom APU 0932"* ]]; then
            STEAM_DECK_DETECTED=true
            STEAM_DECK_MODEL="OLED" 
            ((confidence += 25))
            ((detection_methods++))
        fi
    fi
    
    # Method 3: GPU detection
    if command -v lspci >/dev/null 2>&1; then
        local pci_info
        pci_info=$(lspci -nn 2>/dev/null || echo "")
        if [[ "$pci_info" == *"1002:163f"* ]]; then
            STEAM_DECK_DETECTED=true
            STEAM_DECK_MODEL="LCD"
            ((confidence += 20))
            ((detection_methods++))
        elif [[ "$pci_info" == *"1002:15bf"* ]]; then
            STEAM_DECK_DETECTED=true
            STEAM_DECK_MODEL="OLED"
            ((confidence += 20))
            ((detection_methods++))
        fi
    fi
    
    # Method 4: Battery capacity detection
    local battery_paths=(
        "/sys/class/power_supply/BAT1/energy_full_design"
        "/sys/class/power_supply/BAT0/energy_full_design"
    )
    
    for path in "${battery_paths[@]}"; do
        if [[ -f "$path" ]]; then
            local capacity_uw
            capacity_uw=$(cat "$path" 2>/dev/null || echo "0")
            local capacity_wh=$((capacity_uw / 1000000))
            
            if [[ $capacity_wh -ge 38 && $capacity_wh -le 42 ]]; then
                STEAM_DECK_DETECTED=true
                STEAM_DECK_MODEL="LCD"
                ((confidence += 15))
                ((detection_methods++))
            elif [[ $capacity_wh -ge 48 && $capacity_wh -le 52 ]]; then
                STEAM_DECK_DETECTED=true
                STEAM_DECK_MODEL="OLED"
                ((confidence += 15))
                ((detection_methods++))
            fi
            break
        fi
    done
    
    # Method 5: Filesystem detection
    local deck_indicators=(
        "/home/deck"
        "/usr/bin/steamos-session-select"
        "/etc/systemd/system/steam-deck-oled-display.service"
    )
    
    for indicator in "${deck_indicators[@]}"; do
        if [[ -e "$indicator" ]]; then
            STEAM_DECK_DETECTED=true
            ((confidence += 10))
            ((detection_methods++))
            if [[ "$indicator" == *"oled"* ]]; then
                STEAM_DECK_MODEL="OLED"
            fi
        fi
    done
    
    log_debug "Steam Deck detection: confidence=$confidence, methods=$detection_methods"
    
    if [[ "$STEAM_DECK_DETECTED" == "true" && "$STEAM_DECK_MODEL" == "unknown" ]]; then
        # Default to LCD if we detected Steam Deck but not model
        STEAM_DECK_MODEL="LCD"
    fi
}

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

check_requirements() {
    log_step "Checking system requirements..."
    
    local missing_tools=()
    local warnings=()
    
    # Essential tools
    local required_tools=("curl" "wget" "python3" "git" "unzip")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    # Python version check
    if command -v python3 >/dev/null 2>&1; then
        local python_version
        python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        local major minor
        IFS='.' read -r major minor <<< "$python_version"
        if [[ $major -lt 3 || ($major -eq 3 && $minor -lt 8) ]]; then
            warnings+=("Python $python_version detected, recommend ≥3.8")
        else
            log_success "Python $python_version (compatible)"
        fi
    fi
    
    # pip check
    if ! python3 -m pip --version >/dev/null 2>&1; then
        missing_tools+=("python3-pip")
    fi
    
    # System libraries for shader compilation
    local system_libs=()
    if [[ "$ENABLE_GPU" == "true" ]]; then
        system_libs+=("vulkan-loader" "mesa-vulkan-drivers")
        
        # Check for Vulkan support
        if [[ -d "/usr/share/vulkan/icd.d" ]] && [[ -n "$(ls -A /usr/share/vulkan/icd.d 2>/dev/null)" ]]; then
            log_success "Vulkan ICD detected"
        else
            warnings+=("Vulkan ICD not found - GPU features may not work")
        fi
    fi
    
    # Steam Deck specific checks
    if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
        log_info "Running Steam Deck specific checks..."
        
        # Check for Fossilize (Steam's shader system)
        if command -v fossilize-replay >/dev/null 2>&1; then
            log_success "Fossilize found - enhanced Steam integration available"
        else
            warnings+=("Fossilize not found - basic shader caching only")
        fi
        
        # Check GPU device access
        if [[ -c "/dev/dri/card0" ]]; then
            log_success "GPU device access available"
        else
            warnings+=("GPU device not accessible - may need group membership")
        fi
        
        # Check thermal sensors
        if [[ -d "/sys/class/thermal" ]] && [[ -n "$(ls -A /sys/class/thermal 2>/dev/null)" ]]; then
            log_success "Thermal sensors available"
        else
            warnings+=("Thermal sensors not accessible")
        fi
    fi
    
    # Report findings
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install with your package manager:"
        case $PKG_MANAGER in
            pacman) log_info "  sudo pacman -S ${missing_tools[*]}" ;;
            apt) log_info "  sudo apt update && sudo apt install ${missing_tools[*]}" ;;
            dnf) log_info "  sudo dnf install ${missing_tools[*]}" ;;
            zypper) log_info "  sudo zypper install ${missing_tools[*]}" ;;
            *) log_info "  Use your system's package manager to install these tools" ;;
        esac
        return 1
    fi
    
    if [[ ${#warnings[@]} -gt 0 ]]; then
        for warning in "${warnings[@]}"; do
            log_warning "$warning"
        done
    fi
    
    # Network connectivity test
    if ! curl -fsSL --connect-timeout 10 "https://api.github.com" >/dev/null 2>&1; then
        log_error "Cannot connect to GitHub. Check internet connection."
        return 1
    fi
    
    log_success "System requirements satisfied"
    return 0
}

install_system_dependencies() {
    if [[ "$SKIP_DEPS" == "true" ]]; then
        log_info "Skipping system dependency installation"
        return 0
    fi
    
    log_step "Installing system dependencies..."
    
    local deps=()
    local optional_deps=()
    
    # Base dependencies
    deps=("python3" "python3-pip" "git" "curl" "wget" "unzip")
    
    # Vulkan and graphics dependencies
    if [[ "$ENABLE_GPU" == "true" ]]; then
        case $PKG_MANAGER in
            pacman)
                deps+=("vulkan-icd-loader" "mesa" "vulkan-mesa-layers")
                optional_deps+=("vulkan-tools" "mesa-utils")
                ;;
            apt)
                deps+=("vulkan-loader" "mesa-vulkan-drivers" "vulkan-validationlayers")
                optional_deps+=("vulkan-tools" "mesa-utils")
                ;;
            dnf)
                deps+=("vulkan-loader" "mesa-vulkan-drivers" "vulkan-validation-layers")
                optional_deps+=("vulkan-tools" "mesa-utils")
                ;;
        esac
    fi
    
    # Steam Deck specific dependencies
    if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
        case $PKG_MANAGER in
            pacman)
                optional_deps+=("spirv-tools" "shaderc" "fossilize")
                ;;
            apt)
                optional_deps+=("spirv-tools" "libshaderc-dev")
                ;;
            dnf)
                optional_deps+=("spirv-tools" "shaderc-devel")
                ;;
        esac
    fi
    
    # Install dependencies
    local install_success=false
    case $PKG_MANAGER in
        pacman)
            if [[ "$DRY_RUN" != "true" ]]; then
                if sudo pacman -Sy --needed --noconfirm "${deps[@]}" 2>/dev/null; then
                    install_success=true
                    # Try optional deps
                    sudo pacman -S --needed --noconfirm "${optional_deps[@]}" 2>/dev/null || log_warning "Some optional packages failed to install"
                fi
            else
                log_info "DRY RUN: Would install: ${deps[*]} ${optional_deps[*]}"
                install_success=true
            fi
            ;;
        apt)
            if [[ "$DRY_RUN" != "true" ]]; then
                sudo apt update >/dev/null 2>&1 || true
                if sudo apt install -y "${deps[@]}" 2>/dev/null; then
                    install_success=true
                    sudo apt install -y "${optional_deps[@]}" 2>/dev/null || log_warning "Some optional packages failed to install"
                fi
            else
                log_info "DRY RUN: Would install: ${deps[*]} ${optional_deps[*]}"
                install_success=true
            fi
            ;;
        dnf)
            if [[ "$DRY_RUN" != "true" ]]; then
                if sudo dnf install -y "${deps[@]}" 2>/dev/null; then
                    install_success=true
                    sudo dnf install -y "${optional_deps[@]}" 2>/dev/null || log_warning "Some optional packages failed to install"
                fi
            else
                log_info "DRY RUN: Would install: ${deps[*]} ${optional_deps[*]}"
                install_success=true
            fi
            ;;
        *)
            log_warning "Cannot install system dependencies automatically with $PKG_MANAGER"
            log_info "Please install manually: ${deps[*]}"
            install_success=true  # Continue anyway
            ;;
    esac
    
    if [[ "$install_success" == "true" ]]; then
        log_success "System dependencies installed"
    else
        log_error "Failed to install system dependencies"
        return 1
    fi
}

# ============================================================================
# INSTALLATION MODE SELECTION
# ============================================================================

determine_install_mode() {
    log_step "Determining optimal installation mode..."
    
    if [[ "$INSTALL_MODE" != "auto" ]]; then
        log_info "Installation mode manually set to: $INSTALL_MODE"
        return 0
    fi
    
    # Decision matrix for installation mode
    local system_install_score=0
    local user_install_score=0
    local flatpak_install_score=0
    
    # Scoring factors
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root - defaulting to system install"
        INSTALL_MODE="system"
        return 0
    fi
    
    # Steam Deck preferences
    if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
        if [[ "$GAMING_MODE_ACTIVE" == "true" ]]; then
            ((flatpak_install_score += 30))  # Flatpak integrates better with Gaming Mode
            ((user_install_score += 20))
        else
            ((system_install_score += 20))  # System install better for desktop mode
            ((user_install_score += 25))
        fi
    fi
    
    # Check available installation methods
    if command -v flatpak >/dev/null 2>&1; then
        ((flatpak_install_score += 20))
    else
        flatpak_install_score=0  # Flatpak not available
    fi
    
    # Permission checks
    if [[ -w "/opt" ]] || groups | grep -q sudo; then
        ((system_install_score += 15))
    else
        system_install_score=0  # No sudo access
    fi
    
    # User directory writable (always true)
    ((user_install_score += 10))
    
    # Immutable filesystem check (SteamOS)
    if [[ -f "/.immutable" ]] || mount | grep -q "/ .*ro,"; then
        system_install_score=0  # Immutable filesystem
        ((flatpak_install_score += 25))
    fi
    
    # Determine best mode
    if [[ $flatpak_install_score -gt $system_install_score && $flatpak_install_score -gt $user_install_score ]]; then
        INSTALL_MODE="flatpak"
    elif [[ $system_install_score -gt $user_install_score ]]; then
        INSTALL_MODE="system"
    else
        INSTALL_MODE="user"
    fi
    
    log_success "Selected installation mode: $INSTALL_MODE"
    log_debug "Scores - System: $system_install_score, User: $user_install_score, Flatpak: $flatpak_install_score"
}

# ============================================================================
# INSTALLATION IMPLEMENTATIONS
# ============================================================================

download_source() {
    log_step "Downloading ML Shader Predictor source..."
    
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    local download_url
    if [[ "$INSTALL_DEV" == "true" ]]; then
        download_url="$REPO_URL/archive/refs/heads/main.tar.gz"
        log_info "Downloading development version..."
    else
        # Try to get latest release
        local latest_release
        latest_release=$(curl -fsSL "$API_URL/releases/latest" 2>/dev/null | grep '"tag_name"' | cut -d'"' -f4 || echo "")
        
        if [[ -n "$latest_release" ]]; then
            download_url="$REPO_URL/archive/refs/tags/$latest_release.tar.gz"
            log_info "Downloading latest release: $latest_release"
        else
            download_url="$REPO_URL/archive/refs/heads/main.tar.gz"
            log_info "Latest release not found, downloading main branch..."
        fi
    fi
    
    # Download with progress if not in quiet mode
    if [[ "$QUIET_MODE" == "true" ]]; then
        curl -fsSL "$download_url" -o "source.tar.gz"
    else
        curl -fsSL --progress-bar "$download_url" -o "source.tar.gz"
    fi
    
    # Extract
    tar -xzf "source.tar.gz" --strip-components=1
    rm "source.tar.gz"
    
    log_success "Source code downloaded and extracted"
}

install_system_mode() {
    log_step "Installing in system mode..."
    
    local install_dir="$SYSTEM_INSTALL_DIR"
    INSTALL_DIR="$install_dir"  # Set global variable
    PYTHON_CMD="$install_dir/venv/bin/python"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would install to $install_dir"
        return 0
    fi
    
    # Create system directories
    sudo mkdir -p "$install_dir" "$install_dir/bin"
    sudo chown "$USER:$USER" "$install_dir"
    
    # Copy source files
    cp -r src/* "$install_dir/" 2>/dev/null || true
    cp -r security "$install_dir/" 2>/dev/null || true
    cp requirements*.txt "$install_dir/" 2>/dev/null || true
    cp bin/* "$install_dir/bin/" 2>/dev/null || true
    cp ml-shader-predictor.desktop "$install_dir/" 2>/dev/null || true
    
    # Create virtual environment
    python3 -m venv "$install_dir/venv"
    source "$install_dir/venv/bin/activate"
    
    # Upgrade pip and install dependencies
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "$install_dir/requirements.txt"
    
    # Create system launcher script
    sudo ln -sf "$install_dir/bin/ml-shader-predictor" /usr/local/bin/ml-shader-predictor
    
    # Install systemd services
    install_systemd_services "system" "$install_dir"
    
    log_success "System installation completed: $install_dir"
}

install_user_mode() {
    log_step "Installing in user mode..."
    
    local install_dir="$USER_INSTALL_DIR"
    INSTALL_DIR="$install_dir"  # Set global variable
    PYTHON_CMD="$install_dir/venv/bin/python"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would install to $install_dir"
        return 0
    fi
    
    # Create user directories
    mkdir -p "$install_dir" "$install_dir/bin" "$CONFIG_DIR" "$CACHE_DIR"
    
    # Copy source files
    cp -r src/* "$install_dir/" 2>/dev/null || true
    cp -r security "$install_dir/" 2>/dev/null || true
    cp requirements*.txt "$install_dir/" 2>/dev/null || true
    cp bin/* "$install_dir/bin/" 2>/dev/null || true
    cp ml-shader-predictor.desktop "$install_dir/" 2>/dev/null || true
    
    # Create virtual environment
    python3 -m venv "$install_dir/venv"
    source "$install_dir/venv/bin/activate"
    
    # Install dependencies
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "$install_dir/requirements.txt"
    
    # Create user launcher script
    mkdir -p "$HOME/.local/bin"
    ln -sf "$install_dir/bin/ml-shader-predictor" "$HOME/.local/bin/ml-shader-predictor"
    
    # Install user systemd services
    install_systemd_services "user" "$install_dir"
    
    log_success "User installation completed: $install_dir"
}

install_flatpak_mode() {
    log_step "Installing as Flatpak..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would install Flatpak package"
        return 0
    fi
    
    if ! command -v flatpak >/dev/null 2>&1; then
        log_error "Flatpak not available"
        return 1
    fi
    
    # Check if Flathub is available
    if ! flatpak remotes | grep -q flathub; then
        log_info "Adding Flathub repository..."
        flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
    fi
    
    # Build and install local Flatpak
    if [[ -f "com.shaderpredict.MLCompiler.yml" ]]; then
        log_info "Building Flatpak from source..."
        
        # Install flatpak-builder if needed
        case $PKG_MANAGER in
            pacman) sudo pacman -S --needed --noconfirm flatpak-builder 2>/dev/null || true ;;
            apt) sudo apt install -y flatpak-builder 2>/dev/null || true ;;
            dnf) sudo dnf install -y flatpak-builder 2>/dev/null || true ;;
        esac
        
        # Build Flatpak
        flatpak-builder --user --install --force-clean build-dir com.shaderpredict.MLCompiler.yml
        
        log_success "Flatpak installation completed"
    else
        log_error "Flatpak manifest not found"
        return 1
    fi
}

install_systemd_services() {
    local mode="$1"
    local install_dir="$2"
    
    log_info "Installing systemd services ($mode mode)..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would install systemd services"
        return 0
    fi
    
    local service_files=("ml-shader-predictor.service" "ml-shader-predictor-user.service" 
                         "ml-shader-predictor-thermal.service" "ml-shader-predictor-maintenance.service"
                         "ml-shader-predictor-maintenance.timer")
    
    for service_file in "${service_files[@]}"; do
        if [[ -f "$service_file" ]]; then
            local service_content
            service_content=$(cat "$service_file")
            
            # Replace template variables
            service_content="${service_content//\%i/$USER}"
            service_content="${service_content//\/opt\/ml-shader-predictor/$install_dir}"
            
            if [[ "$mode" == "system" ]]; then
                # Install system service
                echo "$service_content" | sudo tee "/etc/systemd/system/$service_file" >/dev/null
                sudo systemctl daemon-reload
                sudo systemctl enable "$service_file" 2>/dev/null || true
            else
                # Install user service
                mkdir -p "$HOME/.config/systemd/user"
                echo "$service_content" > "$HOME/.config/systemd/user/$service_file"
                systemctl --user daemon-reload
                systemctl --user enable "$service_file" 2>/dev/null || true
            fi
            
            log_success "Installed $service_file"
        fi
    done
}

# ============================================================================
# CONFIGURATION AND INTEGRATION
# ============================================================================

create_configuration() {
    log_step "Creating configuration files..."
    
    mkdir -p "$CONFIG_DIR" "$CACHE_DIR"
    
    # Generate hardware-optimized configuration
    local config_file="$CONFIG_DIR/config.json"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create configuration at $config_file"
        return 0
    fi
    
    # Use our enhanced Steam Deck detection for configuration
    local cache_size_mb=2048
    local max_jobs=4
    local thermal_cpu_limit=85.0
    local thermal_gpu_limit=90.0
    
    if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
        if [[ "$STEAM_DECK_MODEL" == "OLED" ]]; then
            cache_size_mb=2560
            max_jobs=6
            thermal_cpu_limit=87.0
            thermal_gpu_limit=92.0
        fi
    fi
    
    cat > "$config_file" << EOF
{
    "version": "2.0.0",
    "installation": {
        "mode": "$INSTALL_MODE",
        "install_date": "$(date -Iseconds)",
        "installer_version": "$SCRIPT_VERSION"
    },
    "hardware": {
        "steam_deck_detected": $STEAM_DECK_DETECTED,
        "steam_deck_model": "$STEAM_DECK_MODEL",
        "gaming_mode_active": $GAMING_MODE_ACTIVE,
        "os": "$DETECTED_OS",
        "arch": "$DETECTED_ARCH",
        "distro": "$DETECTED_DISTRO"
    },
    "performance": {
        "cache_size_mb": $cache_size_mb,
        "max_parallel_jobs": $max_jobs,
        "enable_gpu_acceleration": $ENABLE_GPU,
        "enable_ml_prediction": $ENABLE_ML,
        "enable_thermal_management": $ENABLE_THERMAL
    },
    "thermal_management": {
        "cpu_temp_limit_celsius": $thermal_cpu_limit,
        "gpu_temp_limit_celsius": $thermal_gpu_limit,
        "enable_thermal_throttling": true,
        "thermal_monitoring_interval": 5
    },
    "p2p_network": {
        "enabled": $ENABLE_P2P,
        "max_connections": 50,
        "bandwidth_limit_kbps": 2048,
        "community_sharing": true
    },
    "steam_integration": {
        "enable_fossilize_integration": $STEAM_DECK_DETECTED,
        "steam_shader_cache_path": "~/.local/share/Steam/steamapps/shadercache",
        "gaming_mode_optimization": $GAMING_MODE_ACTIVE
    },
    "security": {
        "verify_shader_checksums": true,
        "enable_shader_validation": true,
        "sandbox_shader_compilation": true
    },
    "logging": {
        "level": "INFO",
        "max_log_size_mb": 10,
        "max_log_files": 5
    }
}
EOF
    
    # Create RADV optimization environment file for Steam Deck
    if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
        cat > "$CONFIG_DIR/radv_optimizations.sh" << 'EOF'
#!/bin/bash
# Steam Deck RADV optimizations
export RADV_PERFTEST=aco,nggc,sam
export RADV_DEBUG=noshaderdb,nocompute
export MESA_VK_DEVICE_SELECT=1002:163f
export RADV_LOWER_DISCARD_TO_DEMOTE=1
export MESA_GLSL_CACHE_DISABLE=0
export MESA_GLSL_CACHE_MAX_SIZE=1G
export __GL_SHADER_DISK_CACHE=1
export __GL_SHADER_DISK_CACHE_SIZE=1073741824
export DXVK_ASYNC=1
EOF
        chmod +x "$CONFIG_DIR/radv_optimizations.sh"
        log_success "RADV optimizations configured for Steam Deck"
    fi
    
    log_success "Configuration created: $config_file"
}

create_desktop_integration() {
    log_step "Creating desktop integration..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create desktop integration"
        return 0
    fi
    
    # Use Python desktop integration installer
    if "$PYTHON_CMD" "$INSTALL_DIR/src/desktop_integration.py" --install; then
        log_success "Desktop integration installed"
    else
        log_warning "Desktop integration installation failed"
        return 1
    fi
    
    # Install Gaming Mode integration if on Steam Deck
    if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
        log_info "Installing Gaming Mode integration for Steam Deck..."
        
        if "$PYTHON_CMD" "$INSTALL_DIR/src/gaming_mode_integration.py" --install "$INSTALL_DIR"; then
            log_success "Gaming Mode integration installed"
            
            # Create Steam library compatibility tool
            if "$PYTHON_CMD" -c "
from src.gaming_mode_integration import GamingModeIntegration
from pathlib import Path
integration = GamingModeIntegration()
success = integration.create_steam_library_integration(Path('$INSTALL_DIR'))
exit(0 if success else 1)
" 2>/dev/null; then
                log_success "Steam library integration created"
            else
                log_warning "Steam library integration creation failed"
            fi
        else
            log_warning "Gaming Mode integration installation failed"
        fi
    fi
    
    log_success "Desktop integration created"
}

# ============================================================================
# VALIDATION AND POST-INSTALL
# ============================================================================

validate_installation() {
    log_step "Validating installation..."
    
    local errors=0
    local warnings=0
    
    # Determine installation directory
    local install_dir
    case "$INSTALL_MODE" in
        system) install_dir="$SYSTEM_INSTALL_DIR" ;;
        user) install_dir="$USER_INSTALL_DIR" ;;
        flatpak) 
            if flatpak list --user | grep -q "com.shaderpredict.MLCompiler"; then
                log_success "Flatpak installation verified"
                return 0
            else
                log_error "Flatpak installation verification failed"
                return 1
            fi
            ;;
        *) log_error "Unknown installation mode: $INSTALL_MODE"; return 1 ;;
    esac
    
    # Check core files
    local required_files=(
        "$install_dir/shader_prediction_system.py"
        "$install_dir/venv/bin/python"
        "$CONFIG_DIR/config.json"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "File exists: $(basename "$file")"
        else
            log_error "Missing file: $file"
            ((errors++))
        fi
    done
    
    # Test Python environment
    if [[ -f "$install_dir/venv/bin/python" ]]; then
        if "$install_dir/venv/bin/python" --version >/dev/null 2>&1; then
            log_success "Python virtual environment working"
            
            # Test critical imports
            local test_imports=("numpy" "requests" "psutil" "yaml")
            local import_failures=0
            
            for module in "${test_imports[@]}"; do
                if "$install_dir/venv/bin/python" -c "import $module" 2>/dev/null; then
                    log_success "Python module '$module' available"
                else
                    log_warning "Python module '$module' not available"
                    ((import_failures++))
                fi
            done
            
            if [[ $import_failures -gt 2 ]]; then
                log_error "Too many Python modules missing ($import_failures/${#test_imports[@]})"
                ((errors++))
            fi
        else
            log_error "Python virtual environment not working"
            ((errors++))
        fi
    fi
    
    # Test executable
    local exec_cmd
    case "$INSTALL_MODE" in
        system) exec_cmd="/usr/local/bin/ml-shader-predictor" ;;
        user) exec_cmd="$HOME/.local/bin/ml-shader-predictor" ;;
    esac
    
    if [[ -n "$exec_cmd" && -x "$exec_cmd" ]]; then
        log_success "Executable created: $exec_cmd"
    elif [[ "$INSTALL_MODE" != "flatpak" ]]; then
        log_error "Executable not found or not executable: $exec_cmd"
        ((errors++))
    fi
    
    # Check systemd services
    if [[ "$INSTALL_MODE" != "flatpak" ]]; then
        local service_check_cmd="systemctl"
        if [[ "$INSTALL_MODE" == "user" ]]; then
            service_check_cmd="systemctl --user"
        fi
        
        if $service_check_cmd is-enabled ml-shader-predictor.service >/dev/null 2>&1; then
            log_success "SystemD service enabled"
        else
            log_warning "SystemD service not enabled"
            ((warnings++))
        fi
    fi
    
    # Steam Deck specific validation
    if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
        if [[ -f "$CONFIG_DIR/radv_optimizations.sh" ]]; then
            log_success "RADV optimizations configured"
        else
            log_warning "RADV optimizations not configured"
            ((warnings++))
        fi
        
        # Check GPU access
        if [[ -c "/dev/dri/card0" ]]; then
            log_success "GPU device access available"
        else
            log_warning "GPU device access may be restricted"
            ((warnings++))
        fi
    fi
    
    # Summary
    if [[ $errors -eq 0 ]]; then
        if [[ $warnings -eq 0 ]]; then
            log_success "Installation validation: PASSED (perfect)"
        else
            log_success "Installation validation: PASSED ($warnings warnings)"
        fi
        return 0
    else
        log_error "Installation validation: FAILED ($errors errors, $warnings warnings)"
        return 1
    fi
}

create_uninstaller() {
    log_step "Creating uninstaller..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create uninstaller"
        return 0
    fi
    
    local uninstall_script
    case "$INSTALL_MODE" in
        system) 
            uninstall_script="/usr/local/bin/uninstall-ml-shader-predictor"
            sudo tee "$uninstall_script" >/dev/null << 'UNINSTALL_EOF'
#!/bin/bash
# ML Shader Predictor Uninstaller (System)
echo "Uninstalling ML Shader Predictor (system installation)..."

# Stop and disable services
sudo systemctl stop ml-shader-predictor.service 2>/dev/null || true
sudo systemctl disable ml-shader-predictor.service 2>/dev/null || true

# Remove files
sudo rm -rf /opt/ml-shader-predictor
sudo rm -f /usr/local/bin/ml-shader-predictor
sudo rm -f /etc/systemd/system/ml-shader-predictor*.service
sudo rm -f /etc/systemd/system/ml-shader-predictor*.timer

# Reload systemd
sudo systemctl daemon-reload

# Remove user data (ask first)
echo -n "Remove user configuration and cache? [y/N]: "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    rm -rf ~/.config/ml-shader-predictor
    rm -rf ~/.cache/ml-shader-predictor
    rm -f ~/.local/share/applications/ml-shader-predictor.desktop
fi

echo "Uninstallation complete!"
UNINSTALL_EOF
            sudo chmod +x "$uninstall_script"
            ;;
        user)
            uninstall_script="$HOME/.local/bin/uninstall-ml-shader-predictor"
            cat > "$uninstall_script" << 'UNINSTALL_EOF'
#!/bin/bash
# ML Shader Predictor Uninstaller (User)
echo "Uninstalling ML Shader Predictor (user installation)..."

# Stop and disable user services
systemctl --user stop ml-shader-predictor-user.service 2>/dev/null || true
systemctl --user disable ml-shader-predictor-user.service 2>/dev/null || true

# Remove files
rm -rf ~/.local/share/ml-shader-predictor
rm -f ~/.local/bin/ml-shader-predictor
rm -f ~/.config/systemd/user/ml-shader-predictor*.service
rm -f ~/.config/systemd/user/ml-shader-predictor*.timer

# Reload systemd
systemctl --user daemon-reload

# Remove user data
echo -n "Remove configuration and cache? [y/N]: "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    rm -rf ~/.config/ml-shader-predictor
    rm -rf ~/.cache/ml-shader-predictor
fi

# Remove desktop integration
rm -f ~/.local/share/applications/ml-shader-predictor.desktop

echo "Uninstallation complete!"
UNINSTALL_EOF
            chmod +x "$uninstall_script"
            ;;
        flatpak)
            uninstall_script="$HOME/.local/bin/uninstall-ml-shader-predictor"
            cat > "$uninstall_script" << 'UNINSTALL_EOF'
#!/bin/bash
# ML Shader Predictor Uninstaller (Flatpak)
echo "Uninstalling ML Shader Predictor (Flatpak)..."

flatpak uninstall --user com.shaderpredict.MLCompiler

echo "Uninstallation complete!"
UNINSTALL_EOF
            chmod +x "$uninstall_script"
            ;;
    esac
    
    log_success "Uninstaller created: $uninstall_script"
}

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                INSTALL_MODE="$2"
                shift 2
                ;;
            --system)
                INSTALL_MODE="system"
                shift
                ;;
            --user)
                INSTALL_MODE="user"
                shift
                ;;
            --flatpak)
                INSTALL_MODE="flatpak"
                shift
                ;;
            --dev|--development)
                INSTALL_DEV=true
                log_info "Development mode enabled"
                shift
                ;;
            --no-autostart)
                ENABLE_AUTOSTART=false
                shift
                ;;
            --no-p2p)
                ENABLE_P2P=false
                shift
                ;;
            --no-ml)
                ENABLE_ML=false
                shift
                ;;
            --no-gpu)
                ENABLE_GPU=false
                shift
                ;;
            --no-thermal)
                ENABLE_THERMAL=false
                shift
                ;;
            --force)
                FORCE_REINSTALL=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --quiet|-q)
                QUIET_MODE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                log_info "DRY RUN mode - no changes will be made"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --version|-v)
                echo "ML Shader Predictor Unified Installer v$SCRIPT_VERSION"
                exit 0
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
}

show_help() {
    cat << 'HELP_EOF'
ML Shader Predictor - Unified Installation Script

USAGE:
    ./unified-install.sh [OPTIONS]

INSTALLATION MODES:
    --system            Install system-wide (requires sudo)
    --user              Install for current user only
    --flatpak           Install as Flatpak package
    --mode MODE         Specify installation mode explicitly

OPTIONS:
    --dev               Install development version
    --no-autostart      Don't enable automatic startup
    --no-p2p            Disable P2P shader sharing
    --no-ml             Disable ML prediction features
    --no-gpu            Disable GPU acceleration
    --no-thermal        Disable thermal management
    --force             Force reinstallation
    --skip-deps         Skip system dependency installation
    --quiet, -q         Quiet installation
    --dry-run           Show what would be done without making changes
    --help, -h          Show this help message
    --version, -v       Show installer version

EXAMPLES:
    # Automatic installation (recommended)
    ./unified-install.sh

    # System installation with development version
    ./unified-install.sh --system --dev

    # User installation without P2P features
    ./unified-install.sh --user --no-p2p

    # Flatpak installation for Steam Deck Gaming Mode
    ./unified-install.sh --flatpak

    # Dry run to see what would be installed
    ./unified-install.sh --dry-run

STEAM DECK:
    The installer automatically detects Steam Deck hardware and applies
    appropriate optimizations. Gaming Mode users should prefer Flatpak
    installation for better integration.

UNINSTALLATION:
    Use the created uninstaller:
    uninstall-ml-shader-predictor

MORE INFO:
    Repository: https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler
    Issues: https://github.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/issues
HELP_EOF
}

# ============================================================================
# MAIN INSTALLATION FLOW
# ============================================================================

show_banner() {
    if [[ "$QUIET_MODE" != "true" ]]; then
        echo -e "${BOLD}${CYAN}"
        cat << 'BANNER_EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           🤖 ML SHADER PREDICTION COMPILER - UNIFIED INSTALLER               ║
║                          v2.0.0 - Bulletproof Edition                       ║
║                                                                              ║
║   🎮 Steam Deck Optimized  🔧 Multi-Mode Install  🛡️  Security Hardened    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
BANNER_EOF
        echo -e "${NC}"
    fi
}

main() {
    # Initialize
    show_banner
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Prevent running as root unless system install explicitly requested
    if [[ $EUID -eq 0 && "$INSTALL_MODE" != "system" ]]; then
        log_error "Do not run as root unless using --system mode"
        log_info "Run as regular user for automatic mode selection"
        exit 1
    fi
    
    # System detection and analysis
    detect_system
    
    # Check requirements
    if ! check_requirements; then
        exit 1
    fi
    
    # Install system dependencies
    install_system_dependencies
    
    # Determine installation mode
    determine_install_mode
    
    # Check for existing installation
    if [[ "$FORCE_REINSTALL" != "true" ]]; then
        local existing_install=""
        if [[ -d "$SYSTEM_INSTALL_DIR" ]]; then
            existing_install="system"
        elif [[ -d "$USER_INSTALL_DIR" ]]; then
            existing_install="user"
        elif flatpak list --user | grep -q "com.shaderpredict.MLCompiler" 2>/dev/null; then
            existing_install="flatpak"
        fi
        
        if [[ -n "$existing_install" ]]; then
            log_warning "Existing $existing_install installation detected"
            if [[ "$QUIET_MODE" != "true" ]]; then
                echo -n "Reinstall? [y/N]: "
                read -r response
                if [[ ! "$response" =~ ^[Yy]$ ]]; then
                    log_info "Installation cancelled"
                    exit 0
                fi
            fi
        fi
    fi
    
    # Download source code
    download_source
    
    # Execute installation based on mode
    case "$INSTALL_MODE" in
        system)
            install_system_mode
            ;;
        user)
            install_user_mode
            ;;
        flatpak)
            install_flatpak_mode
            ;;
        *)
            log_error "Invalid installation mode: $INSTALL_MODE"
            exit 1
            ;;
    esac
    
    # Create configuration
    create_configuration
    
    # Create desktop integration
    create_desktop_integration
    
    # Create uninstaller
    create_uninstaller
    
    # Validate installation
    if validate_installation; then
        log_header "🎉 INSTALLATION SUCCESSFUL! 🎉"
        
        echo -e "${GREEN}${BOLD}ML Shader Predictor has been successfully installed!${NC}\n"
        
        case "$INSTALL_MODE" in
            system)
                echo -e "${BOLD}System Installation:${NC}"
                echo "  📁 Installation: $SYSTEM_INSTALL_DIR"
                echo "  🚀 Command: ml-shader-predictor"
                echo "  🔧 Service: sudo systemctl start ml-shader-predictor"
                ;;
            user)
                echo -e "${BOLD}User Installation:${NC}"
                echo "  📁 Installation: $USER_INSTALL_DIR"
                echo "  🚀 Command: ml-shader-predictor (add ~/.local/bin to PATH)"
                echo "  🔧 Service: systemctl --user start ml-shader-predictor-user"
                ;;
            flatpak)
                echo -e "${BOLD}Flatpak Installation:${NC}"
                echo "  🚀 Command: flatpak run com.shaderpredict.MLCompiler"
                echo "  📱 Available in application menu"
                ;;
        esac
        
        echo ""
        echo -e "${BOLD}Configuration:${NC}"
        echo "  ⚙️  Config: $CONFIG_DIR/config.json"
        echo "  💾 Cache: $CACHE_DIR"
        
        if [[ "$STEAM_DECK_DETECTED" == "true" ]]; then
            echo ""
            echo -e "${BOLD}Steam Deck Optimizations:${NC}"
            echo "  🎮 Model: $STEAM_DECK_MODEL detected"
            echo "  🔥 RADV optimizations: Enabled"
            echo "  🌡️  Thermal management: Enabled"
            if [[ "$GAMING_MODE_ACTIVE" == "true" ]]; then
                echo "  🎯 Gaming Mode: Optimized integration"
            fi
        fi
        
        echo ""
        echo -e "${BOLD}Next Steps:${NC}"
        echo "  1. 🔄 Restart your session or run the service manually"
        echo "  2. 🎮 Launch a game to see shader optimization in action"
        echo "  3. 📊 Monitor performance improvements"
        
        if [[ "$ENABLE_AUTOSTART" == "true" && "$INSTALL_MODE" != "flatpak" ]]; then
            echo ""
            echo -n "Start the service now? [Y/n]: "
            if [[ "$QUIET_MODE" != "true" ]]; then
                read -r response
            else
                response="y"
            fi
            
            if [[ "$response" =~ ^[Nn]$ ]]; then
                log_info "Service startup skipped"
            else
                case "$INSTALL_MODE" in
                    system)
                        if sudo systemctl start ml-shader-predictor.service; then
                            log_success "Service started successfully!"
                        else
                            log_warning "Service failed to start - check logs with: sudo journalctl -u ml-shader-predictor"
                        fi
                        ;;
                    user)
                        if systemctl --user start ml-shader-predictor-user.service; then
                            log_success "User service started successfully!"
                        else
                            log_warning "Service failed to start - check logs with: journalctl --user -u ml-shader-predictor-user"
                        fi
                        ;;
                esac
            fi
        fi
        
        echo ""
        log_info "🗑️  To uninstall: uninstall-ml-shader-predictor"
        log_info "🆘 For support: ${REPO_URL}/issues"
        
    else
        log_error "Installation validation failed"
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"