# System Architecture

## Overview
The ML Shader Prediction Compiler uses a modular architecture with the following components:

## Core Modules

### ML Prediction Engine (`src/ml/`)
- Optimized machine learning models for shader compilation time prediction
- LightGBM backend for high-performance inference
- Memory pooling and caching for efficiency

### Cache System (`src/cache/`)
- Multi-tier caching (hot/warm/cold storage)
- LZ4 compression for space efficiency
- Async I/O operations for performance

### Thermal Management (`src/thermal/`)
- Predictive thermal modeling
- Game-specific thermal profiles
- Hardware-aware throttling

### Performance Monitoring (`src/monitoring/`)
- Real-time metrics collection
- Health scoring and alerting
- System optimization recommendations

## Data Flow
1. Steam launches detected via D-Bus
2. Shader features extracted and cached
3. ML models predict compilation requirements
4. Thermal manager adjusts compilation strategy
5. Shaders pre-compiled based on predictions
6. Performance metrics collected and analyzed

## Integration Points
- Steam client integration via D-Bus
- Vulkan layer interception
- System thermal sensors
- GPU driver optimization hooks
