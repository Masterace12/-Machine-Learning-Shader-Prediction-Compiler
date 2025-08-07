# Makefile for Shader Predictive Compiler

# Installation directories
PREFIX ?= /opt
INSTALL_DIR = $(PREFIX)/shader-predict-compile
DESKTOP_DIR = /usr/share/applications
SYSTEMD_DIR = /etc/systemd/system

# Application information
APP_NAME = shader-predict-compile
SERVICE_NAME = $(APP_NAME).service
DESKTOP_FILE = $(APP_NAME).desktop

.PHONY: all install uninstall clean test help

all: help

help:
	@echo "Shader Predictive Compiler for Steam Deck"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install the application (requires sudo)"
	@echo "  uninstall   - Remove the application (requires sudo)"  
	@echo "  test        - Run compatibility tests"
	@echo "  clean       - Clean temporary files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Note: Use './scripts/install.sh' for guided installation"

install:
	@echo "Installing Shader Predictive Compiler..."
	@./scripts/install.sh

uninstall:
	@echo "Uninstalling Shader Predictive Compiler..."
	@./scripts/install.sh --uninstall

test:
	@echo "Running compatibility tests..."
	@python3 -c "import sys; print(f'Python version: {sys.version}')"
	@python3 -c "import gi; gi.require_version('Gtk', '3.0'); print('GTK bindings: OK')"
	@python3 -c "import psutil; print(f'psutil version: {psutil.__version__}')"
	@python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')"
	@echo "Testing Steam Deck detection..."
	@python3 -c "from src.steam_deck_compat import SteamDeckCompatibility; compat = SteamDeckCompatibility(); print(f'Model: {compat.model}, SteamOS: {compat.steamos_version}')"

clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@rm -f ./*.log

# Development targets (not for production use)
dev-install: 
	@echo "Installing in development mode..."
	@pip3 install --user -r requirements.txt

dev-run:
	@echo "Running in development mode..."
	@export PYTHONPATH="$(PWD)/src:$$PYTHONPATH" && python3 ui/main_window.py

dev-service:
	@echo "Running background service in development mode..."
	@export PYTHONPATH="$(PWD)/src:$$PYTHONPATH" && python3 src/background_service.py

# Packaging targets
package:
	@echo "Creating package..."
	@mkdir -p dist
	@tar -czf dist/shader-predict-compile.tar.gz \
		--exclude='dist' \
		--exclude='.git' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		.

install-deps:
	@echo "Installing dependencies via pacman..."
	@sudo pacman -S --needed python python-gobject python-psutil python-numpy