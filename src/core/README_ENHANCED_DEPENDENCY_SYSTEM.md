# Enhanced Dependency Management System for ML Shader Prediction Compiler

## Overview

This enhanced dependency management system provides comprehensive, intelligent dependency coordination with graceful fallback mechanisms specifically optimized for the Steam Deck and general Linux environments. The system ensures **100% compatibility** with any Python 3.8+ environment, even with zero external dependencies installed.

## Key Features

### 🚀 Zero Compilation Requirements
- **Pure Python fallbacks** for all dependencies
- **No C/C++ compilation** required during installation
- **Reliable binary wheels** only where proven stable
- **Automatic fallback detection** when dependencies fail

### 🎮 Steam Deck Optimization
- **Hardware detection** and model identification (LCD vs OLED)
- **Thermal-aware performance scaling** 
- **Gaming Mode integration** with background operation
- **Power-efficient dependency selection**
- **Battery life optimization**
- **Docked vs handheld mode optimization**

### ⚡ Intelligent Performance Management
- **Runtime dependency switching** based on system conditions
- **Performance benchmarking** of dependency combinations
- **Adaptive optimization profiles** for different scenarios
- **Real-time system monitoring** and automatic adjustments

### 🔄 Graceful Degradation
- **Seamless fallback** from optimal to pure Python implementations
- **Transparent API compatibility** regardless of backend
- **Error recovery** and automatic dependency switching
- **Health monitoring** with self-repair capabilities

## System Components

### 1. Dependency Coordinator (`dependency_coordinator.py`)
**Central coordination of all dependency management**

```python
from core.dependency_coordinator import get_coordinator

coordinator = get_coordinator()
coordinator.detect_all_dependencies()
profiles = coordinator.benchmark_performance_combinations()
```

**Features:**
- Intelligent dependency detection and validation
- Performance benchmarking of different combinations
- Version compatibility management
- Platform-specific filtering (Steam Deck, ARM64, etc.)

### 2. Installation Validator (`installation_validator.py`)
**Deep validation that dependencies actually work**

```python
from core.installation_validator import quick_validate_installation

summary = quick_validate_installation()
print(f"Health: {summary['average_health']:.1%}")
```

**Features:**
- Functional testing beyond import verification
- Stress testing under various conditions
- Performance validation and benchmarking
- Installation repair recommendations

### 3. Runtime Dependency Manager (`runtime_dependency_manager.py`)
**Dynamic dependency switching at runtime**

```python
from core.runtime_dependency_manager import get_runtime_manager

manager = get_runtime_manager()
manager.start_monitoring()  # Automatic optimization

# Context managers for temporary profiles
with performance_profile('maximum_performance'):
    run_ml_training()
```

**Features:**
- Real-time performance monitoring
- Automatic profile switching based on conditions
- Context managers for temporary optimization
- Thermal and memory-aware dependency selection

### 4. Steam Deck Optimizer (`steam_deck_optimizer.py`)
**Steam Deck specific optimizations and thermal management**

```python
from core.steam_deck_optimizer import get_steam_deck_optimizer

optimizer = get_steam_deck_optimizer()
if optimizer.is_steam_deck:
    optimizer.start_adaptive_optimization()
```

**Features:**
- Hardware detection and profiling
- Thermal-aware performance scaling
- Gaming Mode integration
- Power-efficient dependency selection
- Battery life optimization

### 5. Pure Python Fallbacks (`pure_python_fallbacks.py`)
**Complete pure Python implementations for all dependencies**

```python
from core.pure_python_fallbacks import ArrayMath, MLRegressor, Serializer

# Automatically uses numpy or pure Python fallback
data = ArrayMath.array([1, 2, 3, 4, 5])
mean = ArrayMath.mean(data)
```

**Features:**
- Pure Python implementations for all ML operations
- Transparent API compatibility with full libraries
- Steam Deck hardware detection
- Thermal monitoring and system integration

### 6. Enhanced Dependency System (`enhanced_dependency_system.py`)
**Unified interface coordinating all components**

```python
from core.enhanced_dependency_system import get_enhanced_system

system = get_enhanced_system()
system.start_monitoring()

# Get comprehensive health report
health = system.get_system_health()
print(f"Overall health: {health.overall_health:.1%}")

# Automatic optimization
result = system.optimize_system()
```

**Features:**
- Unified interface for all components
- Comprehensive health monitoring
- Automatic system optimization
- System repair and recovery

## Dependency Categories and Fallbacks

### Core ML Dependencies
| Dependency | Purpose | Fallback Available | Steam Deck Compatible |
|------------|---------|-------------------|----------------------|
| **numpy** | Array operations | ✅ Pure Python | ✅ |
| **scikit-learn** | ML algorithms | ✅ Linear regression | ✅ |
| **lightgbm** | Advanced ML | ✅ Linear regression | ⚠️ Limited |

### Performance Dependencies  
| Dependency | Purpose | Fallback Available | Steam Deck Compatible |
|------------|---------|-------------------|----------------------|
| **numba** | JIT compilation | ✅ No-op decorators | ⚠️ Thermal aware |
| **numexpr** | Fast evaluation | ✅ Pure Python | ✅ |
| **bottleneck** | Fast aggregations | ✅ Pure Python | ✅ |

### System Integration
| Dependency | Purpose | Fallback Available | Steam Deck Compatible |
|------------|---------|-------------------|----------------------|
| **psutil** | System monitoring | ✅ /proc parsing | ✅ |
| **msgpack** | Serialization | ✅ JSON fallback | ✅ |
| **zstandard** | Compression | ✅ gzip fallback | ✅ |

## Usage Examples

### Basic Usage with Automatic Optimization

```python
from core.enhanced_dependency_system import get_enhanced_system

# Initialize and start automatic optimization
system = get_enhanced_system()
system.start_monitoring()

# The system automatically:
# 1. Detects available dependencies
# 2. Benchmarks performance combinations  
# 3. Monitors system conditions
# 4. Switches profiles as needed
# 5. Provides health monitoring
```

### Manual Profile Management

```python
from core.runtime_dependency_manager import get_runtime_manager

manager = get_runtime_manager()

# Switch to specific profile
manager.manual_switch_profile('maximum_performance')

# Temporary profile for specific operations
with manager.temporary_profile('gaming_mode'):
    run_background_ml_task()
```

### Steam Deck Specific Usage

```python
from core.steam_deck_optimizer import get_steam_deck_optimizer

optimizer = get_steam_deck_optimizer()

if optimizer.is_steam_deck:
    # Get current Steam Deck state
    state = optimizer.get_current_state()
    print(f"Temperature: {state.cpu_temperature_celsius}°C")
    print(f"Gaming Mode: {state.gaming_mode_active}")
    
    # Apply Steam Deck optimizations
    optimizer.start_adaptive_optimization()
    
    # Context managers for specific scenarios
    with gaming_mode_optimization():
        run_ml_in_background()
    
    with battery_saving():
        run_power_efficient_ml()
```

### Adaptive Dependency Usage

```python
from core.dependency_coordinator import adaptive_dependency

@adaptive_dependency('array_math')
def process_data(data, array_math_backend=None):
    # Automatically uses numpy or pure Python fallback
    arr = array_math_backend.array(data)
    return array_math_backend.mean(arr)

@adaptive_dependency('ml_backend')  
def train_model(X, y, ml_backend=None):
    # Automatically uses LightGBM, scikit-learn, or pure Python
    model = ml_backend()
    model.fit(X, y)
    return model
```

## Configuration and Optimization Profiles

### Runtime Profiles

| Profile | Description | Use Case | Dependencies |
|---------|-------------|----------|--------------|
| **maximum_performance** | Full optimization | Docked, AC power | All available |
| **balanced** | Balance performance/efficiency | General use | Core dependencies |
| **conservative** | Reduced resource usage | Constrained environments | Minimal set |
| **steam_deck_optimized** | Steam Deck specific | Handheld gaming | Steam Deck friendly |
| **gaming_mode** | Background operation | Active gaming | Minimal impact |
| **battery_saving** | Maximum battery life | Low power | Pure Python only |
| **thermal_emergency** | Emergency cooling | High temperature | Minimal processing |

### Steam Deck Profiles

| Profile | CPU Limit | Memory Limit | Thermal Limit | Use Case |
|---------|-----------|--------------|---------------|----------|
| **docked_performance** | 100% | 1GB | 85°C | External display |
| **handheld_balanced** | 80% | 512MB | 80°C | Portable gaming |
| **gaming_background** | 50% | 256MB | 90°C | During gameplay |
| **battery_conservation** | 40% | 128MB | 70°C | Extended battery |
| **thermal_protection** | 25% | 64MB | 95°C | Emergency cooling |

## Installation and Setup

### Standard Installation

```bash
# Install with all dependencies (recommended)
pip install -r requirements-pure-python.txt

# Minimal installation (core features only)
pip install numpy psutil requests

# Steam Deck installation
pip install --user -r requirements-pure-python.txt
```

### Zero Dependency Installation

```python
# The system works with ZERO external dependencies
# All functionality available through pure Python fallbacks
python enhanced_dependency_system.py  # Works immediately
```

### Development Installation

```bash
# Development with all testing tools
pip install -r requirements-pure-python.txt[dev]

# Run comprehensive validation
python installation_validator.py

# Run performance benchmarks  
python dependency_coordinator.py

# Test Steam Deck optimization
python steam_deck_optimizer.py
```

## System Health and Monitoring

### Health Metrics

The system tracks multiple health dimensions:

- **Overall Health**: Composite score (0-100%)
- **Dependency Health**: Available/working dependencies
- **Performance Health**: Current performance capability  
- **Thermal Health**: Temperature safety margins
- **Memory Health**: Memory usage efficiency
- **Steam Deck Health**: Platform-specific optimization

### Monitoring and Alerts

```python
# Real-time health monitoring
health = system.get_system_health()

if health.overall_health < 0.7:
    print("System health low!")
    for issue in health.critical_issues:
        print(f"Critical: {issue}")
    
    for rec in health.recommendations:
        print(f"Recommendation: {rec}")

# Automatic repair
repair_result = system.repair_system()
if repair_result['success']:
    print("System automatically repaired!")
```

## Performance Benchmarking

### Automatic Benchmarking

```python
coordinator = get_coordinator()

# Benchmark all available combinations
profiles = coordinator.benchmark_performance_combinations()

for profile in profiles:
    print(f"{profile.combination_id}: {profile.overall_score:.2f}")
    print(f"  Execution time: {profile.execution_time:.3f}s")
    print(f"  Memory usage: {profile.memory_usage/1024/1024:.1f}MB")
```

### Custom Performance Testing

```python
validator = InstallationValidator()

# Test specific dependency thoroughly  
result = validator.validate_dependency('numpy', thorough=True)
print(f"Health: {result.installation_health:.1%}")
print(f"Performance: {result.performance_score:.1f}")
```

## Steam Deck Specific Features

### Hardware Detection

```python
from core.steam_deck_optimizer import SteamDeckOptimizer

optimizer = SteamDeckOptimizer()

if optimizer.is_steam_deck:
    profile = optimizer.hardware_profile
    print(f"Model: {profile.model}")  # 'lcd' or 'oled'
    print(f"CPU cores: {profile.cpu_cores}")
    print(f"Memory: {profile.memory_gb}GB")
    print(f"Storage: {profile.storage_type}")
```

### Thermal Management

```python
# Automatic thermal protection
state = optimizer.get_current_state()

if state.thermal_throttling:
    # System automatically switches to thermal_emergency profile
    print("Thermal throttling detected - reducing load")

if state.cpu_temperature_celsius > 85:
    # Automatic emergency measures
    optimizer.apply_optimization_profile('thermal_emergency')
```

### Gaming Mode Integration

```python
# Detect and adapt to Gaming Mode
if state.gaming_mode_active:
    # Automatically switch to minimal impact profile
    optimizer.apply_optimization_profile('gaming_background')
    print("Gaming Mode detected - running in background")
```

## Error Handling and Recovery

### Automatic Fallback

```python
# System automatically falls back when dependencies fail
try:
    import numpy as np
    backend = np  # Use numpy
except ImportError:
    from pure_python_fallbacks import PureArrayMath
    backend = PureArrayMath()  # Automatic fallback

# Transparent usage regardless of backend
result = backend.mean([1, 2, 3, 4, 5])
```

### Error Recovery

```python
# Automatic error recovery and dependency switching
@adaptive_dependency('ml_backend')
def train_model(X, y, ml_backend=None):
    try:
        model = ml_backend()
        model.fit(X, y)
        return model
    except Exception as e:
        # System automatically tries fallback backend
        logger.warning(f"Primary ML backend failed: {e}")
        # Fallback already injected by decorator
        raise
```

## Validation and Testing

### Comprehensive Validation

```python
# Validate entire system
validation = system.validate_system(thorough=True)

print(f"Success: {validation['overall_success']}")
print(f"Health: {validation['summary']['overall_health']:.1%}")

for component, results in validation['components'].items():
    print(f"{component}: {results}")
```

### Continuous Health Monitoring

```python
# Start continuous monitoring
system.start_monitoring()

# Register callback for health changes
def on_health_change(event_type, data):
    if event_type == 'profile_switch':
        print(f"Switched to: {data['new_profile']}")
    elif data.get('health', 1.0) < 0.5:
        print("Critical health issue detected!")

system.add_system_callback(on_health_change)
```

## Configuration Export and Import

### Export System State

```python
# Export comprehensive system report
report_path = Path("system_report.json")
system.export_system_report(report_path, include_history=True)

# Export component-specific configurations
coordinator.export_configuration(Path("dependencies.json"))
optimizer.export_optimization_report(Path("steam_deck.json"))
```

### Configuration Analysis

```python
# Analyze exported configuration
with open("system_report.json") as f:
    config = json.load(f)

print(f"System health: {config['system_health']['overall_health']:.1%}")
print(f"Active profile: {config['configuration']['current_profile']}")
print(f"Available backends: {config['configuration']['active_backends']}")
```

## Best Practices

### 1. Initialize Early
```python
# Initialize system at application startup
system = get_enhanced_system()
system.start_monitoring()  # Start adaptive optimization
```

### 2. Use Context Managers
```python
# Temporary optimization for specific operations
with high_performance_mode():
    train_ml_model()

with battery_saving_mode():
    run_background_tasks()
```

### 3. Monitor Health
```python
# Regular health checks
if not is_system_healthy():
    optimize_for_current_environment()
```

### 4. Steam Deck Awareness
```python
# Check for Steam Deck and adapt accordingly
if is_steam_deck():
    # Use Steam Deck optimized code paths
    with gaming_mode_optimization():
        run_ml_inference()
```

### 5. Graceful Degradation
```python
# Always provide fallback options
@adaptive_dependency('array_math')
def process_array(data, array_math_backend=None):
    # Function works with numpy, cupy, or pure Python
    return array_math_backend.mean(data)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: System automatically falls back to pure Python
2. **Performance Issues**: System automatically switches to optimal profile
3. **Memory Issues**: System switches to memory-efficient profile
4. **Thermal Issues**: System applies thermal protection automatically

### Manual Intervention

```python
# Force specific profile
system.runtime_manager.manual_switch_profile('conservative', force=True)

# Repair system issues
repair_result = system.repair_system()

# Reset to defaults
system.optimize_system(force=True)
```

### Debug Information

```python
# Get detailed system information
health = system.get_system_health()
print(f"Dependencies: {health.dependencies_available}/{health.dependencies_total}")
print(f"Active backends: {health.active_backends}")
print(f"Current profile: {health.current_profile}")
print(f"Critical issues: {health.critical_issues}")
```

## Conclusion

This enhanced dependency management system provides a robust, intelligent foundation for ML applications that need to work reliably across diverse environments, from high-performance desktop systems to resource-constrained Steam Deck handheld gaming devices. The system's graceful degradation ensures that functionality is never lost, while its adaptive optimization maximizes performance when resources allow.

Key benefits:
- **100% Compatibility**: Works in any Python 3.8+ environment
- **Zero Compilation**: Pure Python fallbacks for everything
- **Steam Deck Optimized**: Thermal and power aware
- **Self-Monitoring**: Automatic health checks and optimization
- **Self-Repairing**: Automatic issue detection and resolution
- **Performance Adaptive**: Uses best available dependencies dynamically

The system is designed to be "set it and forget it" - once initialized, it continuously monitors and optimizes itself without requiring manual intervention, while providing detailed visibility when needed for debugging or performance analysis.