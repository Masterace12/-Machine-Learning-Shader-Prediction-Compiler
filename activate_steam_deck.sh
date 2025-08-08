#!/bin/bash
# Steam Deck ML Environment Activation Script
# Source this script to activate the optimized Python environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/ml_env"

echo "=== Steam Deck ML Shader Prediction Compiler Environment ==="
echo "Project directory: $SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run the installation script first."
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check if activation was successful
if [ "$VIRTUAL_ENV" != "$VENV_PATH" ]; then
    echo "❌ Failed to activate virtual environment"
    return 1 2>/dev/null || exit 1
fi

# Set Steam Deck specific environment variables
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export SKLEARN_N_JOBS=2

# Gaming mode detection
if pgrep -f "gamescope" > /dev/null 2>&1; then
    echo "🎮 Gaming mode detected - applying performance optimizations"
    export STEAM_DECK_GAMING_MODE=1
    # Lower priority for background ML processing
    renice -n 10 $$ > /dev/null 2>&1
else
    echo "🖥️  Desktop mode detected"
    export STEAM_DECK_GAMING_MODE=0
fi

# Configure Python path
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Set memory optimization flags
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=65536

echo "✅ Environment activated successfully!"
echo
echo "Python executable: $(which python3)"
echo "Virtual environment: $VIRTUAL_ENV"
echo
echo "Available commands:"
echo "  python3 steam_deck_env.py           # Test and configure environment"
echo "  python3 steam_deck_env.py --gaming-mode  # Configure for gaming mode"
echo "  python3 -c 'import numpy; print(numpy.__version__)'  # Test NumPy"
echo "  deactivate                          # Deactivate environment"
echo

# Run the Steam Deck configuration
echo "🔧 Configuring Steam Deck optimizations..."
python3 "$SCRIPT_DIR/steam_deck_env.py" $([ "$STEAM_DECK_GAMING_MODE" = "1" ] && echo "--gaming-mode")

echo
echo "🚀 Ready to run ML Shader Prediction Compiler!"
echo "To deactivate this environment later, run: deactivate"