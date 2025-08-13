"""
Steam Deck optimization components.

This module contains thermal management, power optimization, and resource
scheduling specifically designed for Steam Deck hardware.
"""

# Import thermal management
try:
    from .optimized_thermal_manager import OptimizedThermalManager
    from .thermal_manager import ThermalManager
except ImportError:
    # Handle missing modules gracefully
    pass

__all__ = [
    'OptimizedThermalManager',
    'ThermalManager'
]