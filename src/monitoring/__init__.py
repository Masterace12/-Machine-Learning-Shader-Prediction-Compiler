"""
Monitoring module for system performance and health tracking.

This module provides performance monitoring, resource tracking, and system
health assessment for the shader prediction system.
"""

# Import performance monitor
try:
    from .performance_monitor import PerformanceMonitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError as e:
    # Handle missing modules gracefully
    HAS_PERFORMANCE_MONITOR = False
    PerformanceMonitor = None

def get_performance_monitor():
    """Get performance monitor with fallback handling"""
    if HAS_PERFORMANCE_MONITOR and PerformanceMonitor is not None:
        try:
            return PerformanceMonitor()
        except Exception as e:
            import logging
            logging.warning(f"Failed to create performance monitor: {e}")
            return None
    
    return None

__all__ = [
    'PerformanceMonitor',
    'get_performance_monitor',
    'HAS_PERFORMANCE_MONITOR'
]
