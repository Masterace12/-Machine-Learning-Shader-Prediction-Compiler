#!/bin/bash
# Steam Deck QA Framework Environment Setup Script

set -e

echo "🚀 Setting up Steam Deck QA Framework Environment"
echo "=================================================="

# Check if running on Steam Deck
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    if [[ "$ID" == "steamos" ]]; then
        echo "✅ Detected Steam Deck environment"
        STEAM_DECK=true
    else
        echo "⚠️  Not running on Steam Deck, but continuing setup"
        STEAM_DECK=false
    fi
else
    echo "⚠️  Could not detect OS, assuming generic Linux"
    STEAM_DECK=false
fi

# Check Python version
echo "🐍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "   Found Python $PYTHON_VERSION"

# Check if Python version is sufficient
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "   ✅ Python version is sufficient"
else
    echo "   ❌ Python 3.9 or higher is required"
    exit 1
fi

# Install system dependencies for Steam Deck
if [[ "$STEAM_DECK" == true ]]; then
    echo "🔧 Installing Steam Deck specific dependencies..."
    
    # Switch to writable root filesystem (if needed)
    sudo steamos-readonly disable || true
    
    # Update package database
    sudo pacman -Sy
    
    # Install required packages
    sudo pacman -S --noconfirm \
        python-pip \
        python-virtualenv \
        git \
        htop \
        glxinfo \
        radeontop \
        mangohud \
        vulkan-tools || echo "Some packages may already be installed"
    
    # Re-enable read-only root filesystem
    sudo steamos-readonly enable || true
    
else
    echo "🔧 Installing generic Linux dependencies..."
    
    # Try different package managers
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            python3-pip \
            python3-venv \
            git \
            htop \
            mesa-utils \
            radeontop \
            vulkan-tools
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y \
            python3-pip \
            python3-virtualenv \
            git \
            htop \
            mesa-utils \
            radeontop \
            vulkan-tools
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm \
            python-pip \
            python-virtualenv \
            git \
            htop \
            glxinfo \
            radeontop \
            vulkan-tools
    else
        echo "⚠️  Unknown package manager. Please install dependencies manually:"
        echo "   - python3-pip"
        echo "   - python3-virtualenv"
        echo "   - git"
        echo "   - mesa-utils/glxinfo"
        echo "   - radeontop"
        echo "   - vulkan-tools"
    fi
fi

# Create Python virtual environment
echo "🐍 Setting up Python virtual environment..."
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "📦 Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
    echo "   ✅ Python dependencies installed"
else
    echo "   ⚠️  requirements.txt not found, installing basic dependencies"
    pip install numpy matplotlib seaborn psutil jinja2 aiohttp
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/{logs,reports,results,telemetry,baselines}
mkdir -p config/templates
echo "   ✅ Directory structure created"

# Check Steam installation
echo "🎮 Checking Steam installation..."
STEAM_PATHS=(
    "$HOME/.steam/steam"
    "$HOME/.local/share/Steam"
    "/usr/bin/steam"
    "/usr/games/steam"
)

STEAM_PATH=""
for path in "${STEAM_PATHS[@]}"; do
    if [[ -f "$path" ]] || [[ -x "$path" ]]; then
        STEAM_PATH="$path"
        break
    fi
done

if [[ -n "$STEAM_PATH" ]]; then
    echo "   ✅ Steam found at: $STEAM_PATH"
else
    echo "   ⚠️  Steam not found. Please install Steam and update config/qa_config.json"
fi

# Check MangoHUD installation
echo "📊 Checking performance monitoring tools..."
if command -v mangohud &> /dev/null; then
    echo "   ✅ MangoHUD found"
    
    # Create MangoHUD config directory
    mkdir -p "$HOME/.config/MangoHud"
    
    # Create basic MangoHUD config if it doesn't exist
    if [[ ! -f "$HOME/.config/MangoHud/MangoHud.conf" ]]; then
        cat > "$HOME/.config/MangoHud/MangoHud.conf" << EOF
fps_limit=0
fps_sampling_period=500
output_folder=/tmp/mangohud_logs
log_duration=300
cpu_stats=1
gpu_stats=1
ram=1
vram=1
frame_timing=1
EOF
        echo "   ✅ MangoHUD config created"
    fi
else
    echo "   ⚠️  MangoHUD not found. Performance monitoring may be limited"
fi

# Check GPU monitoring tools
if command -v radeontop &> /dev/null; then
    echo "   ✅ radeontop found (AMD GPU monitoring)"
elif command -v nvidia-smi &> /dev/null; then
    echo "   ✅ nvidia-smi found (NVIDIA GPU monitoring)"
else
    echo "   ⚠️  No GPU monitoring tools found"
fi

# Validate configuration
echo "⚙️  Validating configuration..."
if [[ -f "config/qa_config.json" ]]; then
    if python3 -c "import json; json.load(open('config/qa_config.json'))" 2>/dev/null; then
        echo "   ✅ Configuration file is valid JSON"
    else
        echo "   ❌ Configuration file has JSON errors"
        exit 1
    fi
else
    echo "   ❌ Configuration file not found"
    exit 1
fi

# Test basic functionality
echo "🧪 Testing basic functionality..."
if python3 -c "
import sys
sys.path.insert(0, 'src')
from core.steam_deck_qa_framework import SteamDeckQAFramework
print('✅ Framework imports successfully')
"; then
    echo "   ✅ Framework can be imported successfully"
else
    echo "   ❌ Framework import failed"
    exit 1
fi

# Create example scripts
echo "📝 Creating example scripts..."
mkdir -p examples

cat > examples/run_single_game.sh << 'EOF'
#!/bin/bash
# Example script to run tests for a single game

cd "$(dirname "$0")/.."
source venv/bin/activate

# Run test for Cyberpunk 2077
python3 main.py --game cyberpunk_2077 --debug

echo "Test completed. Check data/reports/ for results."
EOF

cat > examples/run_regression_test.sh << 'EOF'
#!/bin/bash
# Example script to run regression tests

cd "$(dirname "$0")/.."
source venv/bin/activate

# First, run a full test suite to create a baseline
echo "Creating baseline..."
python3 main.py --full --debug

# Get the session ID from the last run
BASELINE=$(ls -t data/results/qa_results_*.json | head -1 | grep -o '[0-9]\{8\}_[0-9]\{6\}')

if [[ -n "$BASELINE" ]]; then
    echo "Using baseline: $BASELINE"
    
    # Wait a moment, then run regression test
    sleep 5
    python3 main.py --regression "$BASELINE" --debug
else
    echo "Could not determine baseline session ID"
fi
EOF

chmod +x examples/*.sh
echo "   ✅ Example scripts created"

# Final validation
echo "✅ Running final validation..."
python3 main.py --validate-config

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Review and customize config/qa_config.json if needed"
echo "3. Run your first test: python3 main.py --list-games"
echo "4. Start testing: python3 main.py --full"
echo ""
echo "For more information, see README.md"

deactivate