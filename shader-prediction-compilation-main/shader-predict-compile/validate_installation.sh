#!/bin/bash
# Validate Shader Predictive Compiler installation

echo "Validating installation..."

required_files=(
    "src/background_service.py"
    "ui/main_window.py"
    "config/default_settings.json"
    "requirements.txt"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✓ All required files present"
    exit 0
else
    echo "✗ Missing files: ${missing_files[*]}"
    exit 1
fi
