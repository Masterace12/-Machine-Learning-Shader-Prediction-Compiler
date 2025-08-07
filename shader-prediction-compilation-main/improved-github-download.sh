#!/bin/bash
# Improved GitHub Download Script for Shader Predictive Compiler
# Enhanced with better error handling, progress indication, and reliability

set -e

# Configuration
GITHUB_USER="Masterace12"
GITHUB_REPO="shader-prediction-compilation"
GITHUB_BRANCH="main"
REPO_URL="https://github.com/${GITHUB_USER}/${GITHUB_REPO}"
CACHE_DIR="$HOME/.cache/shader-predict-compile"
TEMP_DIR="/tmp/shader-install-$$"
MAX_RETRIES=3
DOWNLOAD_TIMEOUT=300  # 5 minutes

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Progress bar characters
PROGRESS_FILLED="█"
PROGRESS_EMPTY="░"

# Logging functions with timestamps
log_info() { echo -e "[$(date '+%H:%M:%S')] ${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "[$(date '+%H:%M:%S')] ${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "[$(date '+%H:%M:%S')] ${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "[$(date '+%H:%M:%S')] ${RED}[✗]${NC} $1"; }
log_debug() { [ "${DEBUG:-0}" = "1" ] && echo -e "[$(date '+%H:%M:%S')] ${PURPLE}[DEBUG]${NC} $1"; }

# Cleanup function
cleanup() {
    [ -d "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percent=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' "$PROGRESS_FILLED"
    printf "%${empty}s" | tr ' ' "$PROGRESS_EMPTY"
    printf "] %3d%%" "$percent"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    local missing=()
    local optional=()
    
    # Required tools
    for tool in bash; do
        command -v "$tool" &>/dev/null || missing+=("$tool")
    done
    
    # At least one download tool required
    local has_download_tool=false
    for tool in git curl wget; do
        command -v "$tool" &>/dev/null && has_download_tool=true
    done
    [ "$has_download_tool" = false ] && missing+=("git, curl, or wget")
    
    # Optional but recommended tools
    for tool in unzip tar gzip dos2unix sha256sum; do
        command -v "$tool" &>/dev/null || optional+=("$tool")
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_info "Install with: sudo pacman -S ${missing[*]}"
        return 1
    fi
    
    if [ ${#optional[@]} -gt 0 ]; then
        log_warning "Optional tools missing: ${optional[*]}"
        log_info "Consider installing for better functionality"
    fi
    
    log_success "All required tools available"
    return 0
}

# Check if we have a valid cached download
check_cache() {
    local cache_file="$CACHE_DIR/repo-${GITHUB_BRANCH}.tar.gz"
    local cache_info="$CACHE_DIR/repo-${GITHUB_BRANCH}.info"
    
    if [ -f "$cache_file" ] && [ -f "$cache_info" ]; then
        local cached_date=$(cat "$cache_info" 2>/dev/null || echo "0")
        local current_date=$(date +%s)
        local cache_age=$((current_date - cached_date))
        local max_age=$((24 * 60 * 60))  # 24 hours
        
        if [ $cache_age -lt $max_age ]; then
            log_info "Found recent cached download ($(($cache_age / 3600)) hours old)"
            echo "$cache_file"
            return 0
        else
            log_info "Cache expired, will download fresh copy"
            rm -f "$cache_file" "$cache_info"
        fi
    fi
    
    return 1
}

# Download with progress indication
download_with_progress() {
    local url=$1
    local output=$2
    local tool=$3
    
    log_info "Downloading from $url..."
    
    case "$tool" in
        curl)
            curl -L --progress-bar --connect-timeout 30 --max-time "$DOWNLOAD_TIMEOUT" \
                 -o "$output" "$url" 2>&1 | \
                 grep -o "[0-9]*\.[0-9]%" | \
                 while read percent; do
                     printf "\rProgress: ${GREEN}%s${NC}" "$percent"
                 done
            echo
            ;;
        wget)
            wget --progress=bar:force --timeout=30 --tries=3 \
                 -O "$output" "$url" 2>&1 | \
                 grep -o "[0-9]*%" | \
                 while read percent; do
                     printf "\rProgress: ${GREEN}%s${NC}" "$percent"
                 done
            echo
            ;;
        git)
            # Git doesn't provide download progress for archive
            git archive --remote="$url" --format=tar.gz --prefix="${GITHUB_REPO}/" \
                "${GITHUB_BRANCH}" > "$output" 2>/dev/null
            ;;
    esac
    
    return ${PIPESTATUS[0]}
}

# Verify download integrity
verify_download() {
    local file=$1
    
    if [ ! -f "$file" ]; then
        log_error "Download file not found: $file"
        return 1
    fi
    
    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    
    if [ "$size" -lt 1000 ]; then
        log_error "Downloaded file too small (${size} bytes)"
        return 1
    fi
    
    # Verify it's a valid archive
    if command -v file &>/dev/null; then
        local filetype=$(file -b "$file")
        case "$filetype" in
            *gzip*|*zip*|*tar*) 
                log_success "Valid archive detected: $filetype"
                ;;
            *)
                log_error "Invalid file type: $filetype"
                return 1
                ;;
        esac
    fi
    
    # Calculate checksum if available
    if command -v sha256sum &>/dev/null; then
        local checksum=$(sha256sum "$file" | cut -d' ' -f1)
        log_info "SHA256: ${checksum:0:16}..."
    fi
    
    return 0
}

# Download repository with retry logic
download_repo() {
    log_info "Preparing to download repository..."
    
    # Check cache first
    if cached_file=$(check_cache); then
        log_success "Using cached download"
        echo "$cached_file"
        return 0
    fi
    
    mkdir -p "$TEMP_DIR" "$CACHE_DIR"
    local output_file="$TEMP_DIR/repo.archive"
    local success=false
    
    # Try different download methods
    for attempt in $(seq 1 $MAX_RETRIES); do
        log_info "Download attempt $attempt of $MAX_RETRIES"
        
        # Method 1: Git archive (fastest and most reliable)
        if command -v git &>/dev/null && [ $attempt -eq 1 ]; then
            log_info "Trying git archive method..."
            if git archive --remote="${REPO_URL}.git" --format=tar.gz \
                   --prefix="${GITHUB_REPO}-${GITHUB_BRANCH}/" \
                   "${GITHUB_BRANCH}" > "$output_file.tar.gz" 2>/dev/null; then
                if verify_download "$output_file.tar.gz"; then
                    output_file="$output_file.tar.gz"
                    success=true
                    break
                fi
            fi
        fi
        
        # Method 2: Direct ZIP download with curl
        if command -v curl &>/dev/null; then
            log_info "Trying curl download..."
            local zip_url="${REPO_URL}/archive/refs/heads/${GITHUB_BRANCH}.zip"
            if download_with_progress "$zip_url" "$output_file.zip" "curl"; then
                if verify_download "$output_file.zip"; then
                    output_file="$output_file.zip"
                    success=true
                    break
                fi
            fi
        fi
        
        # Method 3: Direct TAR.GZ download with wget
        if command -v wget &>/dev/null; then
            log_info "Trying wget download..."
            local tar_url="${REPO_URL}/archive/refs/heads/${GITHUB_BRANCH}.tar.gz"
            if download_with_progress "$tar_url" "$output_file.tar.gz" "wget"; then
                if verify_download "$output_file.tar.gz"; then
                    output_file="$output_file.tar.gz"
                    success=true
                    break
                fi
            fi
        fi
        
        # Method 4: Git clone (slower but reliable)
        if command -v git &>/dev/null && [ $attempt -eq $MAX_RETRIES ]; then
            log_info "Trying git clone as last resort..."
            if git clone --depth 1 --branch "$GITHUB_BRANCH" "${REPO_URL}.git" "$TEMP_DIR/repo-clone" 2>&1 | \
               while IFS= read -r line; do
                   echo -ne "\r${CYAN}Git:${NC} ${line:0:60}...                    "
               done; then
                echo
                cd "$TEMP_DIR/repo-clone"
                tar czf "$output_file.tar.gz" .
                cd - >/dev/null
                if verify_download "$output_file.tar.gz"; then
                    output_file="$output_file.tar.gz"
                    success=true
                    break
                fi
            fi
        fi
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            log_warning "Download failed, retrying in 5 seconds..."
            sleep 5
        fi
    done
    
    if [ "$success" = false ]; then
        log_error "All download attempts failed"
        return 1
    fi
    
    # Cache the successful download
    cp "$output_file" "$CACHE_DIR/repo-${GITHUB_BRANCH}.tar.gz"
    date +%s > "$CACHE_DIR/repo-${GITHUB_BRANCH}.info"
    
    log_success "Repository downloaded successfully"
    echo "$output_file"
    return 0
}

# Extract archive with proper handling
extract_archive() {
    local archive=$1
    local dest_dir=$2
    
    log_info "Extracting archive..."
    mkdir -p "$dest_dir"
    
    case "$archive" in
        *.zip)
            if command -v unzip &>/dev/null; then
                unzip -q "$archive" -d "$dest_dir"
            else
                log_error "unzip not available for ZIP extraction"
                return 1
            fi
            ;;
        *.tar.gz|*.tgz)
            tar xzf "$archive" -C "$dest_dir"
            ;;
        *)
            log_error "Unknown archive format: $archive"
            return 1
            ;;
    esac
    
    # Find the actual project directory (handle nested structure)
    local project_dir=$(find "$dest_dir" -name "shader-predict-compile" -type d | head -1)
    if [ -z "$project_dir" ]; then
        # Try to find by looking for key files
        project_dir=$(find "$dest_dir" -name "install" -type f -path "*/shader-predict-compile/*" | head -1 | xargs dirname)
    fi
    
    if [ -z "$project_dir" ]; then
        log_warning "Could not find shader-predict-compile directory, using root"
        project_dir="$dest_dir"
    fi
    
    log_success "Extracted to: $project_dir"
    echo "$project_dir"
    return 0
}

# Fix all GitHub-related issues
fix_github_issues() {
    local dir=$1
    
    log_info "Fixing GitHub download issues..."
    cd "$dir"
    
    # Fix permissions with progress
    log_info "Fixing file permissions..."
    local total_files=$(find . -name "*.sh" -o -name "*.py" -o -name "install*" | wc -l)
    local current=0
    
    find . -name "*.sh" -o -name "*.py" -o -name "install*" | while read file; do
        chmod +x "$file" 2>/dev/null || true
        current=$((current + 1))
        show_progress $current $total_files
    done
    echo
    
    # Fix line endings
    if command -v dos2unix &>/dev/null; then
        log_info "Converting line endings with dos2unix..."
        find . \( -name "*.sh" -o -name "*.py" -o -name "install*" \) \
             -exec dos2unix -q {} \; 2>/dev/null || true
    else
        log_info "Converting line endings with sed..."
        find . \( -name "*.sh" -o -name "*.py" -o -name "install*" \) -type f | \
        while read file; do
            sed -i 's/\r$//' "$file" 2>/dev/null || \
            sed -i '' 's/\r$//' "$file" 2>/dev/null || true
        done
    fi
    
    # Create missing directories
    mkdir -p logs cache 2>/dev/null || true
    
    log_success "GitHub issues fixed"
}

# Run installation
run_installation() {
    local install_dir=$1
    shift  # Remove first argument to pass remaining to installer
    
    log_info "Starting installation..."
    cd "$install_dir"
    
    # Find and run appropriate installer
    if [ -x "./install" ]; then
        log_info "Running main installer..."
        ./install "$@"
    elif [ -f "./install" ]; then
        log_info "Running installer with bash..."
        bash ./install "$@"
    elif [ -x "./INSTALL.sh" ]; then
        log_info "Running INSTALL.sh..."
        ./INSTALL.sh "$@"
    else
        log_error "No installer found in $install_dir"
        log_info "Contents:"
        ls -la
        return 1
    fi
}

# Main function
main() {
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}     🚀 Shader Predictive Compiler - Enhanced Installer 🚀${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    # Check requirements
    if ! check_requirements; then
        exit 1
    fi
    
    # Download repository
    if ! archive_file=$(download_repo); then
        log_error "Failed to download repository"
        exit 1
    fi
    
    # Extract archive
    if ! project_dir=$(extract_archive "$archive_file" "$TEMP_DIR/extracted"); then
        log_error "Failed to extract archive"
        exit 1
    fi
    
    # Fix GitHub issues
    fix_github_issues "$project_dir"
    
    # Run installation
    if ! run_installation "$project_dir" "$@"; then
        log_error "Installation failed"
        exit 1
    fi
    
    echo
    log_success "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_success "Installation completed successfully! 🎉"
    log_success "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    log_info "Shader Predictive Compiler is now installed!"
    log_info "You can find it in:"
    echo "  • Gaming Mode: Library → Non-Steam → Shader Predictive Compiler"
    echo "  • Desktop Mode: Applications → Games → Shader Predictive Compiler"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG=1
            shift
            ;;
        --no-cache)
            rm -rf "$CACHE_DIR"
            shift
            ;;
        --branch)
            GITHUB_BRANCH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --debug       Enable debug output"
            echo "  --no-cache    Clear cache and force fresh download"
            echo "  --branch NAME Use specific branch (default: main)"
            echo "  --help        Show this help message"
            echo
            echo "All other arguments are passed to the installer"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Run main function with remaining arguments
main "$@"