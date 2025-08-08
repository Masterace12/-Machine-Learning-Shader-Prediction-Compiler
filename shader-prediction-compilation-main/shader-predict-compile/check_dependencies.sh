#!/bin/bash
# Check all dependencies for Shader Predictive Compiler

echo "Checking dependencies..."

missing=()

# Check Python
if ! command -v python3 &>/dev/null; then
    missing+=("python3")
fi

# Check GTK
if ! python3 -c "import gi; gi.require_version('Gtk', '3.0')" &>/dev/null 2>&1; then
    missing+=("python3-gi")
fi

if [ ${#missing[@]} -eq 0 ]; then
    echo "✓ All dependencies satisfied"
    exit 0
else
    echo "✗ Missing dependencies: ${missing[*]}"
    echo "Run: sudo pacman -S ${missing[*]}"
    exit 1
fi
