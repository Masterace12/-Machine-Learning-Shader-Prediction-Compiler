# Contributing to ML Shader Prediction Compiler

Thank you for your interest in contributing to this project! This guide will help you get started with development and ensure your contributions align with our project standards.

## Table of Contents
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Performance Guidelines](#performance-guidelines)
- [Pull Request Process](#pull-request-process)
- [Architecture Guidelines](#architecture-guidelines)
- [Steam Deck Testing](#steam-deck-testing)
- [Security Considerations](#security-considerations)

## Development Setup

### Prerequisites
- Python 3.8+ (Python 3.10+ recommended)
- Git with LFS support
- Steam Deck hardware (preferred) or Linux desktop

### Initial Setup
```bash
# Clone with full history for ML model development
git clone --recursive https://github.com/user/shader-predict-compile.git
cd shader-predict-compile

# Development installation
./install.sh --dev --monitoring

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### Development Environment
```bash
# Create isolated development environment
python -m venv venv-dev
source venv-dev/bin/activate

# Install with editable mode
pip install -e .

# Verify installation
shader-predict-test --dev-check
```

## Code Standards

### Formatting & Linting
We use a comprehensive toolchain for code quality:

```bash
# Auto-formatting (required before commit)
black src/ tests/
isort src/ tests/

# Linting (must pass)
ruff src/ tests/
pylint src/

# Type checking (must pass)
mypy src/

# Security scanning
bandit -r src/
```

### Python Code Style

**File Headers**:
```python
#!/usr/bin/env python3
"""
Module description here.

This module implements [specific functionality] for the ML Shader Prediction
Compiler system, optimized for Steam Deck performance.
"""

import asyncio
from typing import Dict, List, Optional, Union
```

**Function Documentation**:
```python
async def predict_shader_compilation(
    shader_hash: str,
    game_id: int,
    hardware_config: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Predict shader compilation time and confidence.
    
    Args:
        shader_hash: SHA-256 hash of shader bytecode
        game_id: Steam application ID
        hardware_config: Steam Deck hardware configuration
        
    Returns:
        Tuple of (predicted_time_ms, confidence_score)
        
    Raises:
        PredictionError: If model inference fails
        ValidationError: If input parameters invalid
        
    Example:
        >>> time_ms, confidence = await predict_shader_compilation(
        ...     "abc123...", 1091500, steamdeck_oled_config
        ... )
        >>> print(f"Predicted: {time_ms:.1f}ms (confidence: {confidence:.2f})")
    """
```

**Error Handling**:
```python
# Proper exception handling with context
try:
    result = await expensive_ml_operation()
except PredictionModelError as e:
    logger.error(f"ML model failed: {e}", exc_info=True)
    # Graceful fallback to heuristic predictor
    result = await heuristic_fallback_prediction()
except Exception as e:
    logger.critical(f"Unexpected error in prediction: {e}", exc_info=True)
    raise PredictionSystemError(f"System failure: {e}") from e
```

### Performance-Critical Code
```python
# Use object pooling for frequently allocated objects
from src.cache.object_pool import ObjectPool

class ShaderPredictor:
    def __init__(self):
        self._feature_pool = ObjectPool(FeatureVector, initial_size=100)
    
    async def predict(self, shader_data: bytes) -> float:
        # Reuse objects to avoid allocation overhead
        features = self._feature_pool.acquire()
        try:
            features.extract_from_bytecode(shader_data)
            return await self._model.predict_async(features)
        finally:
            self._feature_pool.release(features)
```

## Testing Requirements

### Test Structure
```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_ml_predictor.py
│   ├── test_thermal_manager.py
│   └── test_cache_system.py
├── integration/             # Component interaction tests
│   ├── test_steam_integration.py
│   └── test_end_to_end.py
├── performance/             # Benchmark tests
│   ├── test_prediction_speed.py
│   └── test_memory_usage.py
└── fixtures/                # Test data and mocks
    ├── shader_samples/
    └── mock_hardware/
```

### Writing Tests
```python
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.ml.optimized_ml_predictor import MLShaderPredictor

class TestMLShaderPredictor:
    @pytest.fixture
    async def predictor(self):
        """Create test predictor with mocked dependencies."""
        predictor = MLShaderPredictor()
        await predictor.initialize()
        return predictor
    
    @pytest.mark.asyncio
    async def test_prediction_accuracy(self, predictor):
        """Test prediction accuracy with known shader samples."""
        # Test with validated shader data
        shader_hash = "test_shader_cyberpunk2077_main"
        game_id = 1091500
        
        prediction_ms, confidence = await predictor.predict(
            shader_hash, game_id, hardware_config
        )
        
        # Validate prediction range
        assert 0.1 <= prediction_ms <= 5000.0
        assert 0.0 <= confidence <= 1.0
        
        # Check performance requirement
        start_time = time.perf_counter()
        await predictor.predict(shader_hash, game_id, hardware_config)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        assert elapsed_ms < 5.0, f"Prediction too slow: {elapsed_ms:.2f}ms"
    
    @pytest.mark.steamdeck
    async def test_thermal_integration(self, predictor):
        """Test thermal management integration (Steam Deck only)."""
        if not is_steam_deck():
            pytest.skip("Steam Deck hardware required")
        
        # Test thermal state transitions
        with patch('src.thermal.get_current_temperature') as mock_temp:
            mock_temp.return_value = 85.0  # Hot state
            
            prediction = await predictor.predict_with_thermal("test_hash")
            
            # Should reduce threads in hot state
            assert prediction.threads_used <= 2
```

### Test Coverage Requirements
- **Minimum**: 80% line coverage
- **Target**: 90% branch coverage
- **Critical paths**: 100% coverage (ML inference, thermal management)
- **Performance tests**: Must validate <2ms prediction latency

### Running Tests
```bash
# Full test suite
pytest tests/ --cov=src --cov-report=html --cov-fail-under=80

# Fast tests only
pytest tests/unit/ -x

# Steam Deck hardware tests
pytest tests/ -m steamdeck

# Performance benchmarks  
pytest tests/performance/ --benchmark-only

# Memory leak detection
pytest tests/ --memray
```

## Performance Guidelines

### Measurement Requirements
Every performance-sensitive change must include:

1. **Latency benchmarks**:
```python
@pytest.mark.benchmark
def test_prediction_latency(benchmark):
    """Benchmark ML prediction latency."""
    predictor = setup_predictor()
    
    result = benchmark(predictor.predict, test_shader_hash)
    
    # Must meet performance requirement
    assert benchmark.stats['mean'] < 0.002  # 2ms max
```

2. **Memory profiling**:
```bash
# Profile memory usage
python -m memray run --live src/main.py --profile-mode
python -m memray flamegraph profile.bin
```

3. **Steam Deck validation**:
```bash
# Test on actual hardware
shader-predict-test --performance --steam-deck
```

### Performance Requirements
- **ML Inference**: <2ms per prediction
- **Memory Usage**: <100MB total footprint
- **CPU Impact**: <2% during gaming
- **Cache Performance**: >90% hit rate for hot cache
- **Thermal Response**: <1s adaptation to temperature changes

## Pull Request Process

### Before Submitting
```bash
# Pre-submission checklist
pre-commit run --all-files
pytest tests/ --cov=src --cov-fail-under=80
mypy src/
bandit -r src/

# Performance validation
pytest tests/performance/ --benchmark-only
```

### PR Description Template
```markdown
## Summary
Brief description of changes and motivation.

## Changes Made
- [ ] Core functionality changes
- [ ] Performance optimizations  
- [ ] Documentation updates
- [ ] Test additions/modifications

## Testing
- [ ] All tests pass locally
- [ ] Performance benchmarks included
- [ ] Steam Deck hardware tested (if applicable)
- [ ] Memory usage validated

## Benchmarks
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Prediction latency | X.Xms | X.Xms | ±X.X% |
| Memory usage | XMB | XMB | ±X.X% |

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Security implications considered
- [ ] Breaking changes documented
```

### Review Process
1. **Automated checks**: All CI checks must pass
2. **Code review**: At least one maintainer approval required
3. **Performance review**: Benchmark results must meet standards
4. **Security review**: Required for security-sensitive changes
5. **Steam Deck testing**: Hardware validation for core changes

## Architecture Guidelines

### Async/Await Patterns
```python
# Correct async usage
class ThermalManager:
    async def monitor_temperature(self) -> None:
        """Monitor temperature with proper async patterns."""
        while self._running:
            try:
                temp = await self._read_temperature()
                await self._update_thermal_state(temp)
                
                # Non-blocking delay
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Thermal monitoring error: {e}")
                await asyncio.sleep(5.0)  # Back off on errors
```

### Object Pooling
```python
# Memory-efficient object reuse
class CacheManager:
    def __init__(self):
        self._cache_entry_pool = ObjectPool(
            CacheEntry, 
            initial_size=1000,
            max_size=5000
        )
    
    def create_cache_entry(self, shader_hash: str) -> CacheEntry:
        entry = self._cache_entry_pool.acquire()
        entry.reset()  # Clear previous state
        entry.shader_hash = shader_hash
        return entry
```

### Error Handling
```python
# Comprehensive error handling
class ShaderCompiler:
    async def compile_shader(self, source: str) -> bytes:
        """Compile shader with proper error handling."""
        try:
            # Attempt compilation
            return await self._compile_internal(source)
        except CompilerTimeout:
            logger.warning("Shader compilation timeout, using cache")
            return await self._load_from_cache(source)
        except CompilerError as e:
            logger.error(f"Compilation failed: {e}")
            raise ShaderCompilationError(f"Failed to compile shader: {e}")
        except Exception as e:
            logger.critical(f"Unexpected compilation error: {e}", exc_info=True)
            # System-level failure, propagate up
            raise
```

## Steam Deck Testing

### Hardware Requirements
- Steam Deck (LCD or OLED model)
- Developer mode enabled
- SSH access configured
- Test games installed

### Testing Procedures
```bash
# Comprehensive Steam Deck validation
shader-predict-test --steam-deck --full

# Thermal stress testing
shader-predict-test --thermal-stress --duration=3600

# Gaming integration test
shader-predict-test --game-integration --game-id=1091500

# Battery impact testing
shader-predict-test --battery-impact --duration=1800
```

### Performance Validation
```python
# Steam Deck specific performance tests
@pytest.mark.steamdeck_only
def test_steamdeck_performance():
    """Validate Steam Deck specific performance characteristics."""
    # Test thermal adaptation
    with ThermalSimulator(target_temp=90.0):
        performance = measure_performance_under_thermal_load()
        assert performance.threads_reduced > 0
        assert performance.compilation_rate_reduced > 50
    
    # Test memory constraints
    memory_usage = measure_memory_usage_during_gaming()
    assert memory_usage.peak_mb < 100
    assert memory_usage.gaming_impact_percent < 2
```

## Security Considerations

### Code Security
```python
# Input validation for security
def validate_shader_input(shader_data: bytes) -> bool:
    """Validate shader input for security."""
    # Size limits
    if len(shader_data) > MAX_SHADER_SIZE:
        raise ValidationError("Shader too large")
    
    # Format validation  
    if not is_valid_spv_bytecode(shader_data):
        raise ValidationError("Invalid SPIR-V format")
    
    # Static analysis for malicious patterns
    if detect_malicious_patterns(shader_data):
        raise SecurityError("Potentially malicious shader detected")
    
    return True
```

### Sensitive Data Handling
```python
# Proper handling of sensitive data
def process_game_data(game_info: dict) -> dict:
    """Process game data with privacy protection."""
    # Hash sensitive identifiers
    processed = {
        'game_id_hash': hash_game_id(game_info['app_id']),
        'performance_metrics': game_info['metrics'],
        # Remove any personally identifiable information
    }
    
    # Ensure no PII leakage
    assert 'user_id' not in processed
    assert 'steam_id' not in processed
    
    return processed
```

## Community & Communication

### Getting Help
- **Discord**: Join our development channel
- **GitHub Discussions**: For design discussions
- **Issues**: For bug reports and feature requests

### Reporting Security Issues
- **Email**: security@shader-predict-compile.org
- **PGP**: Use our public key for sensitive reports
- **Timeline**: We aim for 48-hour response time

### Recognition
Contributors are recognized in:
- Release notes for significant contributions
- Contributors section of README.md
- Annual contributor appreciation posts

---

Thank you for contributing to the ML Shader Prediction Compiler! Your contributions help improve gaming performance for the entire Steam Deck community.
