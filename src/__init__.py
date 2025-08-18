"""
ML Shader Prediction Compiler - Main Package

A high-performance shader compilation prediction system optimized for Steam Deck
with Rust acceleration and comprehensive optimization features.
"""

# Version information
__version__ = "3.0.0"
__author__ = "ML Shader Prediction Compiler Team"

# Import main components
try:
    from .core import (
        OptimizedMLPredictor,
        OptimizedShaderCache,
        UnifiedMLPredictor,
        HybridMLPredictor,
        HybridVulkanCache
    )
except ImportError:
    pass

try:
    from .optimization import (
        OptimizedThermalManager,
        ThermalManager
    )
except ImportError:
    pass

try:
    from .security import SecurityIntegration
except ImportError:
    pass

try:
    from .monitoring import PerformanceMonitor
except ImportError:
    pass

# Rust integration
try:
    from .rust_integration import (
        HybridMLPredictor,
        HybridVulkanCache,
        get_system_info,
        is_steam_deck
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

__all__ = [
    'OptimizedMLPredictor',
    'OptimizedShaderCache',
    'UnifiedMLPredictor',
    'HybridMLPredictor',
    'HybridVulkanCache',
    'OptimizedThermalManager',
    'ThermalManager',
    'SecurityIntegration',
    'PerformanceMonitor',
    'get_system_info',
    'is_steam_deck',
    'RUST_AVAILABLE'
]
