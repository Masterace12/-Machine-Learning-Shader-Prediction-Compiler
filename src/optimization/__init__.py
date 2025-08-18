"""
Steam Deck optimization components.

This module contains thermal management, power optimization, and resource
scheduling specifically designed for Steam Deck hardware.
"""

# Import thermal management
try:
    from .optimized_thermal_manager import OptimizedThermalManager
    HAS_OPTIMIZED_THERMAL = True
except ImportError:
    # Handle missing modules gracefully
    HAS_OPTIMIZED_THERMAL = False
    OptimizedThermalManager = None

try:
    from .thermal_manager import ThermalManager
    HAS_THERMAL_MANAGER = True
except ImportError:
    HAS_THERMAL_MANAGER = False
    ThermalManager = None

def get_thermal_manager():
    """Get thermal manager with fallback handling"""
    if HAS_OPTIMIZED_THERMAL and OptimizedThermalManager is not None:
        try:
            return OptimizedThermalManager()
        except Exception as e:
            import logging
            logging.warning(f"Failed to create optimized thermal manager: {e}")
    
    if HAS_THERMAL_MANAGER and ThermalManager is not None:
        try:
            return ThermalManager()
        except Exception as e:
            import logging
            logging.warning(f"Failed to create thermal manager: {e}")
    
    return None

__all__ = [
    'OptimizedThermalManager',
    'ThermalManager',
    'get_thermal_manager',
    'HAS_OPTIMIZED_THERMAL',
    'HAS_THERMAL_MANAGER'
]