"""
Core ML and caching components for shader prediction.

This module contains the consolidated ML prediction and caching functionality,
including both the optimized Python implementations and Rust integration.
"""

# Import main components
try:
    from .optimized_ml_predictor import OptimizedMLPredictor
    from .optimized_shader_cache import OptimizedShaderCache
    from .unified_ml_predictor import UnifiedMLPredictor
except ImportError as e:
    # Handle missing modules gracefully
    pass

# Import Rust integration
try:
    from ..rust_integration import HybridMLPredictor, HybridVulkanCache
except ImportError:
    # Rust components not available
    pass

__all__ = [
    'OptimizedMLPredictor',
    'OptimizedShaderCache', 
    'UnifiedMLPredictor',
    'HybridMLPredictor',
    'HybridVulkanCache'
]