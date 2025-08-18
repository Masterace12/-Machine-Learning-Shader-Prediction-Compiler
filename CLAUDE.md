# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Machine Learning Shader Prediction Compiler designed to eliminate shader compilation stutters on Steam Deck by predicting and pre-compiling shaders before they're needed. It's a hybrid Python/Rust system optimized for Steam Deck hardware (both LCD and OLED models).

## Key Commands

### Build and Setup

```bash
# Build Rust components (checks syntax, doesn't fully compile without dependencies)
./build_rust_components.sh

# Install on Steam Deck (one-command installation)
curl -fsSL https://raw.githubusercontent.com/Masterace12/-Machine-Learning-Shader-Prediction-Compiler/main/install_fixed.sh | bash -s -- --user-space --enable-autostart

# Update existing installation
./update_fixed.sh

# Uninstall
~/.local/share/shader-predict-compile/uninstall.sh
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests
pytest -m "not slow"     # Skip slow tests
pytest -m steamdeck      # Steam Deck specific tests

# Run with coverage
pytest --cov=src --cov-report=html

# Run benchmarks
pytest -m benchmark --benchmark-only
```

### Linting and Code Quality

```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Run linter
ruff check src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Development Commands

```bash
# Check system status
shader-predict-status

# Run system test
shader-predict-test

# View statistics
shader-predict-compile --stats

# Monitor services
systemctl --user status shader-predict-compile.service
journalctl --user -f -u shader-predict-compile.service
```

### Rust Development

```bash
# Navigate to Rust workspace
cd rust-core/

# Check all Rust components
cargo check --workspace

# Build specific component (requires system dependencies)
cargo build --release -p ml-engine

# Run Rust tests
cargo test --workspace

# Benchmark Rust code
cargo bench
```

## Architecture

### Hybrid Python/Rust System

The project uses a hybrid architecture where performance-critical components are implemented in Rust with Python fallbacks:

1. **Python Layer** (`src/`)
   - `core/unified_ml_predictor.py`: Main ML prediction logic with LightGBM
   - `core/optimized_shader_cache.py`: Shader cache management
   - `optimization/thermal_manager.py`: Thermal/power management for Steam Deck
   - `monitoring/performance_monitor.py`: System performance monitoring
   - `rust_integration.py`: Bridge between Python and Rust components

2. **Rust Components** (`rust-core/`)
   - `ml-engine/`: ONNX-based ML inference with SIMD optimizations
   - `vulkan-cache/`: Memory-mapped Vulkan shader cache
   - `steamdeck-optimizer/`: Hardware-specific optimizations
   - `security-analyzer/`: SPIR-V bytecode validation
   - `system-monitor/`: Low-level system metrics
   - `p2p-network/`: Peer-to-peer shader distribution
   - `python-bindings/`: PyO3 bindings for Python integration

3. **Graceful Degradation**
   - System automatically falls back to Python implementations if Rust components aren't available
   - `HybridMLPredictor` in `rust_integration.py` manages this fallback logic

### Key Design Patterns

1. **Adaptive Performance**: System monitors thermal state and adjusts compilation scheduling
2. **Memory-Mapped Caching**: Uses mmap for efficient large cache management
3. **Background Compilation**: Predicts and compiles shaders during idle moments
4. **Hardware Detection**: Automatically detects Steam Deck model (LCD/OLED) and adjusts parameters

### Steam Deck Integration

- **Gaming Mode**: Integrates via D-Bus for automatic game detection
- **Thermal Management**: Monitors `/sys/class/thermal/` and adjusts behavior
- **Power Awareness**: Adapts based on battery/AC power state
- **User-space Installation**: Works without root access in `~/.local/`

### ML Pipeline

1. **Feature Extraction**: Analyzes shader complexity (instruction count, register usage, etc.)
2. **Prediction**: LightGBM model predicts compilation time
3. **Scheduling**: Thermal-aware scheduler decides when to compile
4. **Caching**: Stores compiled shaders with LRU eviction

## Testing Strategy

- **Unit Tests**: Test individual components (`test_ml_predictor.py`, `test_optimized_cache.py`)
- **Integration Tests**: Test component interactions (`test_main_integration.py`)
- **Performance Tests**: Benchmark critical paths (use `pytest -m benchmark`)
- **Hardware Tests**: Steam Deck specific tests (marked with `@pytest.mark.steamdeck`)

## Dependencies

### Python (Core)
- `lightgbm`: ML predictions
- `numpy`, `scikit-learn`: ML support
- `psutil`: System monitoring
- `pydantic`: Configuration validation

### Python (Optional - gracefully degrades)
- `onnxruntime`: For Rust ML integration
- `dbus-python`: Steam integration
- `PyQt5`: GUI components

### Rust (when building from source)
- ONNX Runtime C API
- Vulkan SDK
- System libraries: `libc`, `libm`

## Important Files and Locations

- **Main entry point**: `main.py`
- **CLI tools**: Installed to `~/.local/bin/shader-predict-*`
- **Configuration**: `~/.config/shader-predict-compile/`
- **Cache storage**: `~/.cache/shader-predict-compile/`
- **Logs**: Available via `journalctl --user -u shader-predict-compile.service`
- **Steam Deck configs**: `config/steamdeck_lcd_config.json`, `config/steamdeck_oled_config.json`

## Performance Considerations

- Python fallback: ~1.6ms per prediction, 71MB memory
- Rust compiled: ~0.3ms per prediction, 15-20MB memory
- Cache lookups: 50μs (Python) vs 5μs (Rust)
- Thermal throttling: Automatically reduces activity above 70°C

## Security Notes

- Sandboxed shader validation in `security/sandbox_executor.py`
- Hardware fingerprinting for anti-cheat compatibility
- No network communication except updates
- All shader operations are read-only to game files