"""
Core ML and caching components for shader prediction.

This module contains the consolidated ML prediction and caching functionality,
including both the optimized Python implementations and Rust integration.
"""

# Import main components
try:
    from .optimized_ml_predictor import OptimizedMLPredictor, get_optimized_predictor
    HAS_ML_PREDICTOR = True
except ImportError as e:
    # Handle missing modules gracefully
    HAS_ML_PREDICTOR = False
    OptimizedMLPredictor = None
    get_optimized_predictor = None

try:
    from .optimized_shader_cache import OptimizedShaderCache, get_optimized_cache
    HAS_SHADER_CACHE = True
except ImportError as e:
    HAS_SHADER_CACHE = False
    OptimizedShaderCache = None
    get_optimized_cache = None

try:
    from .unified_ml_predictor import UnifiedMLPredictor
    HAS_UNIFIED_PREDICTOR = True
except ImportError as e:
    HAS_UNIFIED_PREDICTOR = False
    UnifiedMLPredictor = None

# Import Rust integration
try:
    from ..rust_integration import HybridMLPredictor, HybridVulkanCache
    HAS_RUST_INTEGRATION = True
except ImportError:
    # Rust components not available
    HAS_RUST_INTEGRATION = False
    HybridMLPredictor = None
    HybridVulkanCache = None

# Fallback factory functions
def get_optimized_predictor_safe():
    """Get optimized predictor with fallback handling"""
    if HAS_ML_PREDICTOR and get_optimized_predictor is not None:
        try:
            return get_optimized_predictor()
        except Exception as e:
            import logging
            logging.warning(f"Failed to create optimized predictor: {e}")
    
    if HAS_RUST_INTEGRATION and HybridMLPredictor is not None:
        try:
            return HybridMLPredictor()
        except Exception as e:
            import logging
            logging.warning(f"Failed to create hybrid predictor: {e}")
    
    # Return None if nothing works
    return None

def get_optimized_cache_safe():
    """Get optimized cache with fallback handling"""
    if HAS_SHADER_CACHE and get_optimized_cache is not None:
        try:
            return get_optimized_cache()
        except Exception as e:
            import logging
            logging.warning(f"Failed to create optimized cache: {e}")
    
    if HAS_RUST_INTEGRATION and HybridVulkanCache is not None:
        try:
            return HybridVulkanCache()
        except Exception as e:
            import logging
            logging.warning(f"Failed to create hybrid cache: {e}")
    
    # Return None if nothing works
    return None

__all__ = [
    'OptimizedMLPredictor',
    'OptimizedShaderCache', 
    'UnifiedMLPredictor',
    'HybridMLPredictor',
    'HybridVulkanCache',
    'get_optimized_predictor',
    'get_optimized_cache',
    'get_optimized_predictor_safe',
    'get_optimized_cache_safe',
    'HAS_ML_PREDICTOR',
    'HAS_SHADER_CACHE',
    'HAS_UNIFIED_PREDICTOR',
    'HAS_RUST_INTEGRATION'
]