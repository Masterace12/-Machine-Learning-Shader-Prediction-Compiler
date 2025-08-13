# API Reference

## Command Line Interface

### Main Commands
- `shader-predict-compile` - Start the prediction system
- `shader-predict-status` - Show system status  
- `shader-predict-test` - Run system diagnostics

### Options
- `--config` - Show configuration
- `--stats` - Performance statistics
- `--export-metrics` - Export performance data
- `--test` - Run comprehensive tests

## Python API

### ML Predictor
```python
from src.ml.optimized_ml_predictor import get_optimized_predictor

predictor = get_optimized_predictor()
prediction = predictor.predict_compilation_time(shader_features)
```

### Cache Manager
```python
from src.cache.optimized_shader_cache import get_optimized_cache

cache = get_optimized_cache()
cache.put(shader_entry)
entry = cache.get(shader_hash)
```

### Thermal Manager
```python
from src.thermal.optimized_thermal_manager import get_thermal_manager

thermal = get_thermal_manager()
thermal.start_monitoring()
status = thermal.get_status()
```
