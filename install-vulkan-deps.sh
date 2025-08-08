#!/bin/bash
# Vulkan Dependencies Installer for ML Shader Predictor
# Ensures all required Vulkan, SPIRV, and graphics dependencies are installed

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1" >&2; }

# Detect system
detect_system() {
    if command -v pacman >/dev/null 2>&1; then
        PKG_MANAGER="pacman"
        DISTRO="arch"
    elif command -v apt >/dev/null 2>&1; then
        PKG_MANAGER="apt"
        DISTRO="debian"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
        DISTRO="fedora"
    elif command -v zypper >/dev/null 2>&1; then
        PKG_MANAGER="zypper"
        DISTRO="opensuse"
    else
        log_error "Unsupported package manager"
        exit 1
    fi
    
    # Detect Steam Deck
    STEAM_DECK=false
    if [[ -f "/etc/os-release" ]] && grep -q "steamos" /etc/os-release; then
        STEAM_DECK=true
    fi
}

# Check current Vulkan installation status
check_vulkan_status() {
    log_info "Checking current Vulkan installation..."
    
    local vulkan_loader_found=false
    local vulkan_drivers_found=false
    local spirv_tools_found=false
    local validation_layers_found=false
    
    # Check for Vulkan loader
    if ldconfig -p | grep -q libvulkan.so; then
        vulkan_loader_found=true
        log_success "Vulkan loader: Present"
    else
        log_warning "Vulkan loader: Missing"
    fi
    
    # Check for Vulkan drivers (ICD files)
    local icd_dirs=("/usr/share/vulkan/icd.d" "/etc/vulkan/icd.d")
    for dir in "${icd_dirs[@]}"; do
        if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
            vulkan_drivers_found=true
            log_success "Vulkan drivers: Found in $dir"
            break
        fi
    done
    
    if [[ "$vulkan_drivers_found" != "true" ]]; then
        log_warning "Vulkan drivers: Missing"
    fi
    
    # Check for SPIRV tools
    if command -v spirv-as >/dev/null 2>&1; then
        spirv_tools_found=true
        log_success "SPIRV tools: Present"
    else
        log_warning "SPIRV tools: Missing"
    fi
    
    # Check for validation layers
    local layer_dirs=("/usr/share/vulkan/explicit_layer.d" "/etc/vulkan/explicit_layer.d")
    for dir in "${layer_dirs[@]}"; do
        if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
            validation_layers_found=true
            log_success "Validation layers: Found in $dir"
            break
        fi
    done
    
    if [[ "$validation_layers_found" != "true" ]]; then
        log_warning "Validation layers: Missing"
    fi
    
    # Steam Deck specific checks
    if [[ "$STEAM_DECK" == "true" ]]; then
        log_info "Running Steam Deck specific checks..."
        
        # Check AMD GPU detection
        if lspci | grep -q "AMD.*VGA\|AMD.*Display"; then
            log_success "AMD GPU detected"
        else
            log_warning "AMD GPU not detected"
        fi
        
        # Check RADV driver
        if [[ -f "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json" ]]; then
            log_success "RADV driver: Present"
        else
            log_warning "RADV driver: Missing"
        fi
        
        # Check Mesa version
        if command -v mesa-vulkan-drivers >/dev/null 2>&1; then
            local mesa_version
            mesa_version=$(mesa-vulkan-drivers --version 2>/dev/null || echo "unknown")
            log_info "Mesa version: $mesa_version"
        fi
    fi
    
    # Overall status
    if [[ "$vulkan_loader_found" == "true" && "$vulkan_drivers_found" == "true" ]]; then
        log_success "Vulkan installation: Functional"
        return 0
    else
        log_warning "Vulkan installation: Incomplete"
        return 1
    fi
}

# Install Vulkan dependencies based on package manager
install_vulkan_deps() {
    log_info "Installing Vulkan dependencies for $DISTRO..."
    
    local base_packages=()
    local dev_packages=()
    local optional_packages=()
    
    case "$PKG_MANAGER" in
        pacman)
            # Arch Linux / SteamOS packages
            base_packages+=(
                "vulkan-icd-loader"      # Vulkan loader library
                "mesa"                   # Mesa 3D graphics library
                "lib32-mesa"             # 32-bit Mesa for compatibility
                "vulkan-mesa-layers"     # Mesa Vulkan layers
                "lib32-vulkan-mesa-layers" # 32-bit Mesa Vulkan layers
                "mesa-utils"             # Mesa utilities
            )
            
            dev_packages+=(
                "vulkan-headers"         # Vulkan development headers
                "spirv-tools"            # SPIRV toolkit
                "shaderc"                # Google's HLSL->SPIRV compiler
                "vulkan-validation-layers" # Validation layers
                "lib32-vulkan-validation-layers" # 32-bit validation layers
            )
            
            optional_packages+=(
                "vulkan-tools"           # Vulkan utilities (vulkaninfo, etc.)
                "renderdoc"              # Graphics debugging tool
                "fossilize"              # Fossilize for Steam integration
            )
            ;;
            
        apt)
            # Ubuntu/Debian packages
            base_packages+=(
                "libvulkan1"             # Vulkan loader
                "mesa-vulkan-drivers"    # Mesa Vulkan drivers
                "mesa-vulkan-drivers:i386" # 32-bit Mesa drivers
                "libc6:i386"             # 32-bit C library
                "mesa-utils"             # Mesa utilities
            )
            
            dev_packages+=(
                "libvulkan-dev"          # Vulkan development files
                "spirv-tools"            # SPIRV toolkit
                "libshaderc-dev"         # Shaderc development files
                "vulkan-validationlayers" # Validation layers
                "vulkan-validationlayers-dev" # Validation layer development
            )
            
            optional_packages+=(
                "vulkan-tools"           # Vulkan utilities
                "mesa-utils-extra"       # Additional Mesa utilities
            )
            ;;
            
        dnf)
            # Fedora/RHEL packages
            base_packages+=(
                "vulkan-loader"          # Vulkan loader
                "mesa-vulkan-drivers"    # Mesa Vulkan drivers
                "mesa-vulkan-drivers.i686" # 32-bit Mesa drivers
                "mesa-utils"             # Mesa utilities
            )
            
            dev_packages+=(
                "vulkan-headers"         # Vulkan headers
                "spirv-tools"            # SPIRV toolkit
                "shaderc-devel"          # Shaderc development
                "vulkan-validation-layers" # Validation layers
            )
            
            optional_packages+=(
                "vulkan-tools"           # Vulkan tools
                "renderdoc"              # Graphics debugging
            )
            ;;
            
        zypper)
            # openSUSE packages
            base_packages+=(
                "libvulkan1"             # Vulkan loader
                "Mesa-vulkan-device-select" # Mesa Vulkan drivers
                "Mesa-dri"               # Mesa DRI drivers
            )
            
            dev_packages+=(
                "vulkan-devel"           # Vulkan development
                "spirv-tools"            # SPIRV tools
                "shaderc-devel"          # Shaderc development
            )
            
            optional_packages+=(
                "vulkan-tools"           # Vulkan utilities
            )
            ;;
    esac
    
    # Install base packages
    log_info "Installing base Vulkan packages..."
    case "$PKG_MANAGER" in
        pacman)
            sudo pacman -Sy --needed --noconfirm "${base_packages[@]}" || {
                log_error "Failed to install base packages"
                return 1
            }
            ;;
        apt)
            sudo apt update
            sudo apt install -y "${base_packages[@]}" || {
                log_error "Failed to install base packages" 
                return 1
            }
            ;;
        dnf)
            sudo dnf install -y "${base_packages[@]}" || {
                log_error "Failed to install base packages"
                return 1
            }
            ;;
        zypper)
            sudo zypper install -y "${base_packages[@]}" || {
                log_error "Failed to install base packages"
                return 1
            }
            ;;
    esac
    
    log_success "Base Vulkan packages installed"
    
    # Install development packages
    log_info "Installing development packages..."
    case "$PKG_MANAGER" in
        pacman)
            sudo pacman -S --needed --noconfirm "${dev_packages[@]}" 2>/dev/null || {
                log_warning "Some development packages failed to install"
            }
            ;;
        apt)
            sudo apt install -y "${dev_packages[@]}" 2>/dev/null || {
                log_warning "Some development packages failed to install"
            }
            ;;
        dnf)
            sudo dnf install -y "${dev_packages[@]}" 2>/dev/null || {
                log_warning "Some development packages failed to install"
            }
            ;;
        zypper)
            sudo zypper install -y "${dev_packages[@]}" 2>/dev/null || {
                log_warning "Some development packages failed to install"
            }
            ;;
    esac
    
    # Install optional packages (non-critical)
    log_info "Installing optional packages..."
    case "$PKG_MANAGER" in
        pacman)
            sudo pacman -S --needed --noconfirm "${optional_packages[@]}" 2>/dev/null || {
                log_info "Optional packages installation skipped or failed"
            }
            ;;
        apt)
            sudo apt install -y "${optional_packages[@]}" 2>/dev/null || {
                log_info "Optional packages installation skipped or failed" 
            }
            ;;
        dnf)
            sudo dnf install -y "${optional_packages[@]}" 2>/dev/null || {
                log_info "Optional packages installation skipped or failed"
            }
            ;;
        zypper)
            sudo zypper install -y "${optional_packages[@]}" 2>/dev/null || {
                log_info "Optional packages installation skipped or failed"
            }
            ;;
    esac
}

# Steam Deck specific optimizations
install_steamdeck_optimizations() {
    if [[ "$STEAM_DECK" != "true" ]]; then
        return 0
    fi
    
    log_info "Applying Steam Deck specific Vulkan optimizations..."
    
    # Create RADV optimization script
    local radv_script="$HOME/.local/bin/setup-radv-optimizations.sh"
    mkdir -p "$(dirname "$radv_script")"
    
    cat > "$radv_script" << 'EOF'
#!/bin/bash
# Steam Deck RADV Optimizations for ML Shader Predictor

# RADV (AMD Radeon Vulkan) optimizations
export RADV_PERFTEST=aco,nggc,sam,rt,ngg_streamout
export RADV_DEBUG=noshaderdb,nocompute
export MESA_VK_DEVICE_SELECT=1002:163f  # Steam Deck APU
export RADV_LOWER_DISCARD_TO_DEMOTE=1

# Mesa optimizations  
export MESA_GLSL_CACHE_DISABLE=0
export MESA_GLSL_CACHE_MAX_SIZE=1073741824  # 1GB
export MESA_SHADER_CACHE_DISABLE=0

# OpenGL optimizations
export __GL_SHADER_DISK_CACHE=1
export __GL_SHADER_DISK_CACHE_SIZE=1073741824  # 1GB

# DXVK optimizations
export DXVK_ASYNC=1
export DXVK_STATE_CACHE=1
export DXVK_CONFIG_FILE="$HOME/.config/dxvk.conf"

# VKD3D-Proton optimizations
export VKD3D_CONFIG=dxr,dxr11
export VKD3D_SHADER_CACHE_PATH="$HOME/.cache/vkd3d-proton"

# Create DXVK config for Steam Deck
mkdir -p "$HOME/.config"
cat > "$HOME/.config/dxvk.conf" << 'DXVK_EOF'
# DXVK configuration for Steam Deck
dxgi.maxFrameLatency = 1
d3d11.constantBufferRangeCheck = false  
d3d11.strictDivision = false
d3d11.floatControls = false
d3d9.deferSurfaceCreation = true
d3d9.memoryTrackTest = false
DXVK_EOF

echo "RADV optimizations applied for Steam Deck"
EOF
    
    chmod +x "$radv_script"
    log_success "Steam Deck optimizations script created: $radv_script"
    
    # Create systemd environment file
    local env_file="$HOME/.config/environment.d/50-vulkan-steamdeck.conf"
    mkdir -p "$(dirname "$env_file")"
    
    cat > "$env_file" << 'EOF'
# Steam Deck Vulkan Environment Variables
RADV_PERFTEST=aco,nggc,sam
RADV_DEBUG=noshaderdb,nocompute  
MESA_VK_DEVICE_SELECT=1002:163f
RADV_LOWER_DISCARD_TO_DEMOTE=1
MESA_GLSL_CACHE_DISABLE=0
MESA_GLSL_CACHE_MAX_SIZE=1073741824
__GL_SHADER_DISK_CACHE=1
__GL_SHADER_DISK_CACHE_SIZE=1073741824
DXVK_ASYNC=1
EOF
    
    log_success "Steam Deck environment variables configured: $env_file"
}

# Validate Vulkan installation
validate_vulkan_installation() {
    log_info "Validating Vulkan installation..."
    
    local validation_success=true
    
    # Test Vulkan loader
    if ! ldconfig -p | grep -q libvulkan.so; then
        log_error "Vulkan loader not found in library path"
        validation_success=false
    fi
    
    # Test vulkaninfo if available
    if command -v vulkaninfo >/dev/null 2>&1; then
        log_info "Running vulkaninfo test..."
        if vulkaninfo >/dev/null 2>&1; then
            log_success "vulkaninfo test passed"
        else
            log_warning "vulkaninfo test failed - may indicate driver issues"
        fi
    else
        log_info "vulkaninfo not available - skipping test"
    fi
    
    # Test SPIRV tools
    if command -v spirv-as >/dev/null 2>&1; then
        log_success "SPIRV tools are available"
    else
        log_warning "SPIRV tools not found"
    fi
    
    # Check ICD files
    local icd_found=false
    for dir in "/usr/share/vulkan/icd.d" "/etc/vulkan/icd.d"; do
        if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
            log_success "Vulkan ICD files found in $dir"
            icd_found=true
            break
        fi
    done
    
    if [[ "$icd_found" != "true" ]]; then
        log_error "No Vulkan ICD files found"
        validation_success=false
    fi
    
    # Steam Deck specific validation
    if [[ "$STEAM_DECK" == "true" ]]; then
        # Check for RADV driver specifically
        if [[ -f "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json" ]]; then
            log_success "RADV driver ICD found"
        else
            log_warning "RADV driver ICD not found"
        fi
        
        # Check GPU device access
        if [[ -c "/dev/dri/card0" ]]; then
            log_success "GPU device access available"
        else
            log_warning "GPU device access may be restricted"
        fi
    fi
    
    if [[ "$validation_success" == "true" ]]; then
        log_success "Vulkan installation validation passed"
        return 0
    else
        log_error "Vulkan installation validation failed"
        return 1
    fi
}

# Add Python Vulkan bindings
install_python_vulkan() {
    log_info "Installing Python Vulkan bindings..."
    
    # Check if we have a virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log_info "Using active virtual environment: $VIRTUAL_ENV"
        pip_cmd="pip"
    elif [[ -f "/opt/ml-shader-predictor/venv/bin/pip" ]]; then
        log_info "Using system installation virtual environment"
        pip_cmd="/opt/ml-shader-predictor/venv/bin/pip"
    elif [[ -f "$HOME/.local/share/ml-shader-predictor/venv/bin/pip" ]]; then
        log_info "Using user installation virtual environment"  
        pip_cmd="$HOME/.local/share/ml-shader-predictor/venv/bin/pip"
    else
        log_info "Using user pip installation"
        pip_cmd="python3 -m pip install --user"
    fi
    
    # Install Vulkan Python bindings
    local python_packages=(
        "vulkan>=1.3.0"          # Python Vulkan bindings
    )
    
    for package in "${python_packages[@]}"; do
        if $pip_cmd install "$package" 2>/dev/null; then
            log_success "Installed Python package: $package"
        else
            log_warning "Failed to install Python package: $package"
        fi
    done
}

# Main function
main() {
    echo -e "${BOLD}${BLUE}ML Shader Predictor - Vulkan Dependencies Installer${NC}"
    echo "====================================================="
    echo
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "Do not run as root - script will request sudo when needed"
        exit 1
    fi
    
    # Detect system
    detect_system
    log_info "Detected system: $DISTRO ($PKG_MANAGER)"
    
    if [[ "$STEAM_DECK" == "true" ]]; then
        log_info "Steam Deck detected - enabling optimizations"
    fi
    
    # Check current status
    if check_vulkan_status; then
        echo
        echo -n "Vulkan appears functional. Reinstall/update dependencies? [y/N]: "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Installation skipped"
            exit 0
        fi
    fi
    
    echo
    log_info "Installing Vulkan dependencies..."
    
    # Install dependencies
    if install_vulkan_deps; then
        log_success "Vulkan dependencies installed"
    else
        log_error "Vulkan dependency installation failed"
        exit 1
    fi
    
    # Steam Deck optimizations
    install_steamdeck_optimizations
    
    # Install Python bindings
    install_python_vulkan
    
    echo
    log_info "Validating installation..."
    if validate_vulkan_installation; then
        echo
        log_success "🎉 Vulkan dependencies successfully installed and validated!"
        echo
        echo -e "${BOLD}Next steps:${NC}"
        echo "1. Restart your session to load environment variables"
        echo "2. Test with: vulkaninfo"
        echo "3. Run ML Shader Predictor to verify GPU acceleration"
        
        if [[ "$STEAM_DECK" == "true" ]]; then
            echo
            echo -e "${BOLD}Steam Deck users:${NC}"
            echo "• RADV optimizations are now active"
            echo "• Shader caches will use 1GB storage"  
            echo "• GPU acceleration enabled for shader compilation"
        fi
    else
        log_error "Validation failed - some issues remain"
        echo
        echo "Common solutions:"
        echo "1. Restart your system"
        echo "2. Check if your GPU supports Vulkan"
        echo "3. Update GPU drivers"
        echo "4. Run: sudo ldconfig"
        exit 1
    fi
}

# Run main function
main "$@"