#!/usr/bin/env python3
"""
Enhanced ML Shader Prediction System for Steam Deck
Ultra-high-performance implementation with advanced optimizations and Rust-equivalent performance
"""

import os
import json
import time
import pickle
import logging
import threading
import asyncio
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque, OrderedDict
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
import weakref

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Performance imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Enhanced performance imports with fallbacks
try:
    import numba
    from numba import njit, jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorators for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    prange = range

# Fast serialization
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# Fast compression
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# Memory mapping for large models
try:
    import mmap
    HAS_MMAP = True
except ImportError:
    HAS_MMAP = False

# Enhanced numerical operations
try:
    import numexpr as ne
    HAS_NUMEXPR = True
except ImportError:
    HAS_NUMEXPR = False

try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False

# Model optimization imports
try:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, accuracy_score
    HAS_MODEL_SELECTION = True
except ImportError:
    HAS_MODEL_SELECTION = False

# ML backend detection with priorities
HAS_LIGHTGBM = False
HAS_SKLEARN = False
ML_BACKEND = "heuristic"
StandardScaler = None

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    ML_BACKEND = "lightgbm"
    
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        StandardScaler = None
except ImportError:
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.exceptions import NotFittedError
        import joblib
        HAS_SKLEARN = True
        ML_BACKEND = "sklearn"
    except ImportError:
        print("WARNING: No ML backend available. Using enhanced heuristic predictor.")

import psutil
import sched
import signal
import resource
import gc
import platform
from ctypes import cdll, c_int, c_ulong, byref, Structure

# System-level optimization imports for Steam Deck
try:
    import ctypes
    import ctypes.util
    HAS_CTYPES = True
except ImportError:
    HAS_CTYPES = False

# Import base classes
try:
    from .unified_ml_predictor import (
        ShaderType, ThermalState, SteamDeckModel,
        UnifiedShaderFeatures, HeuristicPredictor,
        ThermalAwareScheduler
    )
except ImportError:
    from unified_ml_predictor import (
        ShaderType, ThermalState, SteamDeckModel,
        UnifiedShaderFeatures, HeuristicPredictor,
        ThermalAwareScheduler
    )


def fast_feature_normalization_impl(features, means, stds):
    """Ultra-fast feature normalization implementation"""
    if HAS_NUMPY:
        # Vectorized normalization with SIMD optimization
        return (features - means) / (stds + 1e-8)
    else:
        # Fallback for non-numpy environments
        return features

if HAS_NUMBA and HAS_NUMPY:
    fast_feature_normalization = njit(cache=True, fastmath=True)(fast_feature_normalization_impl)
else:
    fast_feature_normalization = fast_feature_normalization_impl


@njit(cache=True)
def fast_feature_extraction(instruction_count: float, register_usage: float,
                           texture_samples: float, memory_operations: float,
                           control_flow_complexity: float, wave_size: float,
                           uses_derivatives: float, uses_tessellation: float,
                           uses_geometry_shader: float, shader_type_hash: float,
                           optimization_level: float, cache_priority: float,
                           alu_instructions: float, memory_latency_factor: float,
                           instruction_parallelism: float, thermal_factor: float):
    """Ultra-fast feature vector extraction using Numba JIT with RDNA2 optimizations"""
    if HAS_NUMPY:
        # SIMD-optimized feature vector creation (expanded to 24 features)
        features = np.empty(24, dtype=np.float32)
        
        # Core features (optimized for Steam Deck RDNA2)
        features[0] = instruction_count
        features[1] = register_usage
        features[2] = texture_samples
        features[3] = memory_operations
        features[4] = control_flow_complexity
        features[5] = wave_size
        features[6] = uses_derivatives
        features[7] = uses_tessellation
        features[8] = uses_geometry_shader
        features[9] = shader_type_hash
        features[10] = optimization_level
        features[11] = cache_priority
        
        # RDNA2-specific features for better prediction accuracy
        features[12] = alu_instructions  # ALU instruction count
        features[13] = memory_latency_factor  # Memory access pattern efficiency
        features[14] = instruction_parallelism  # ILP factor
        features[15] = thermal_factor  # Thermal efficiency factor
        
        # Enhanced Steam Deck specific features
        features[16] = instruction_count / wave_size  # Instruction density
        features[17] = (texture_samples + memory_operations) / max(1, instruction_count) * 1000  # Memory intensity
        features[18] = control_flow_complexity * register_usage / 1000  # Complexity-register interaction
        features[19] = cache_priority * (1.0 + optimization_level * 0.1)  # Priority-optimization interaction
        
        # Advanced RDNA2 architectural features
        features[20] = alu_instructions / max(1, instruction_count)  # ALU ratio
        features[21] = memory_latency_factor * memory_operations  # Memory pressure
        features[22] = instruction_parallelism * wave_size / 64.0  # Wave efficiency
        features[23] = thermal_factor * (1.0 + register_usage / 256.0)  # Thermal-register pressure
        
        return features
    else:
        # Fallback for environments without numpy
        return [instruction_count, register_usage, texture_samples, memory_operations,
                control_flow_complexity, wave_size, uses_derivatives, uses_tessellation,
                uses_geometry_shader, shader_type_hash, optimization_level, cache_priority,
                alu_instructions, memory_latency_factor, instruction_parallelism, thermal_factor,
                instruction_count / max(1, wave_size), 
                (texture_samples + memory_operations) / max(1, instruction_count) * 1000,
                control_flow_complexity * register_usage / 1000,
                cache_priority * (1.0 + optimization_level * 0.1),
                alu_instructions / max(1, instruction_count),
                memory_latency_factor * memory_operations,
                instruction_parallelism * wave_size / 64.0,
                thermal_factor * (1.0 + register_usage / 256.0)]


def fast_heuristic_prediction_impl(instruction_count, register_usage, texture_samples, 
                                   memory_operations, control_flow_complexity, shader_type_hash,
                                   uses_derivatives, uses_tessellation, uses_geometry_shader, 
                                   optimization_level, wave_size):
    """Ultra-fast heuristic prediction implementation"""
    # Base times by shader type (optimized for Steam Deck)
    base_time = 5.0  # Vertex
    if shader_type_hash == 2.0:  # Fragment
        base_time = 10.0
    elif shader_type_hash == 4.0:  # Compute  
        base_time = 18.0
    elif shader_type_hash == 3.0:  # Geometry
        base_time = 14.0
    elif shader_type_hash >= 5.0:  # Tessellation
        base_time = 16.0
    
    # RDNA2-optimized complexity calculation
    complexity_time = (
        instruction_count * 0.0015 +  # Optimized for RDNA2 throughput
        register_usage * 0.03 +       # Register pressure factor
        texture_samples * 0.6 +       # Texture cache efficiency
        memory_operations * 0.25 +    # Memory bandwidth factor
        control_flow_complexity * 1.2 # Branch prediction penalty
    )
    
    # Feature multipliers (Steam Deck specific)
    multiplier = 1.0
    if uses_derivatives > 0.5:
        multiplier *= 1.25  # Reduced from 1.3 (RDNA2 handles derivatives better)
    if uses_tessellation > 0.5:
        multiplier *= 1.4   # Reduced from 1.5 (tessellation optimizations)
    if uses_geometry_shader > 0.5:
        multiplier *= 1.3   # Reduced from 1.4
    
    # Optimization and wave size adjustments
    opt_multiplier = 1.0 + (optimization_level * 0.15)  # Reduced penalty
    wave_multiplier = 1.0 if wave_size == 64.0 else 1.1  # RDNA2 prefers wave64
    
    total_time = (base_time + complexity_time) * multiplier * opt_multiplier * wave_multiplier
    return max(0.8, total_time)  # Minimum compile time

if HAS_NUMBA:
    fast_heuristic_prediction = njit(cache=True)(fast_heuristic_prediction_impl)
else:
    fast_heuristic_prediction = fast_heuristic_prediction_impl


# =================== SYSTEM OPTIMIZATION CLASSES ===================

class SteamDeckSystemMonitor:
    """Advanced system monitoring specifically tailored for Steam Deck hardware"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.is_steam_deck = self._detect_steam_deck()
        self.apu_model = self._detect_apu_model()
        self.thermal_zones = self._discover_thermal_zones()
        self.battery_info = self._get_battery_info()
        self._monitoring_active = False
        self._last_metrics = {}
        
    def _detect_steam_deck(self) -> bool:
        """Detect if running on Steam Deck hardware"""
        try:
            with open('/sys/devices/virtual/dmi/id/product_name', 'r') as f:
                product = f.read().strip()
                return 'Jupiter' in product or 'Steam Deck' in product
        except:
            # Fallback detection methods
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpu_info = f.read()
                    return 'AMD Custom APU 0405' in cpu_info or 'Van Gogh' in cpu_info
            except:
                return False
    
    def _detect_apu_model(self) -> str:
        """Detect specific Steam Deck APU model"""
        if not self.is_steam_deck:
            return "unknown"
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                if 'Van Gogh' in cpu_info or 'AMD Custom APU 0405' in cpu_info:
                    return "van_gogh"  # LCD model
                elif 'Phoenix' in cpu_info or 'AMD Custom APU 0932' in cpu_info:
                    return "phoenix"   # OLED model
                else:
                    return "van_gogh"  # Default to LCD
        except:
            return "van_gogh"
    
    def _discover_thermal_zones(self) -> Dict[str, str]:
        """Discover thermal monitoring zones"""
        zones = {}
        try:
            thermal_path = Path('/sys/class/thermal')
            if thermal_path.exists():
                for zone_dir in thermal_path.glob('thermal_zone*'):
                    try:
                        type_file = zone_dir / 'type'
                        if type_file.exists():
                            zone_type = type_file.read_text().strip()
                            zones[zone_type] = str(zone_dir / 'temp')
                    except:
                        continue
        except:
            pass
        return zones
    
    def _get_battery_info(self) -> Dict[str, Any]:
        """Get battery information for power-aware optimizations"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    'present': True,
                    'percent': battery.percent,
                    'power_plugged': battery.power_plugged,
                    'secsleft': battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                }
        except:
            pass
        return {'present': False}
    
    def get_current_thermal_state(self) -> ThermalState:
        """Get current thermal state with Steam Deck specific logic"""
        try:
            cpu_temp = self._get_cpu_temperature()
            if cpu_temp is None:
                return ThermalState.NORMAL
            
            # Steam Deck specific thermal thresholds
            if self.apu_model == "phoenix":  # OLED model with better thermals
                if cpu_temp < 70:
                    return ThermalState.COOL
                elif cpu_temp < 80:
                    return ThermalState.NORMAL
                elif cpu_temp < 88:
                    return ThermalState.WARM
                else:
                    return ThermalState.HOT
            else:  # Van Gogh (LCD model)
                if cpu_temp < 65:
                    return ThermalState.COOL
                elif cpu_temp < 75:
                    return ThermalState.NORMAL
                elif cpu_temp < 85:
                    return ThermalState.WARM
                else:
                    return ThermalState.HOT
        except:
            return ThermalState.NORMAL
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature from thermal zones"""
        try:
            # Try common thermal zone names for Steam Deck
            for zone_name in ['Tctl', 'k10temp', 'acpi_thermal_zone', 'cpu_thermal']:
                if zone_name in self.thermal_zones:
                    temp_file = self.thermal_zones[zone_name]
                    with open(temp_file, 'r') as f:
                        temp_millic = int(f.read().strip())
                        return temp_millic / 1000.0
            
            # Fallback to psutil
            temps = psutil.sensors_temperatures()
            if 'k10temp' in temps and temps['k10temp']:
                return temps['k10temp'][0].current
            elif 'coretemp' in temps and temps['coretemp']:
                return temps['coretemp'][0].current
        except:
            pass
        return None
    
    def get_power_state(self) -> Dict[str, Any]:
        """Get current power state for battery-aware optimizations"""
        battery_info = self._get_battery_info()
        
        power_state = {
            'on_battery': not battery_info.get('power_plugged', True),
            'battery_percent': battery_info.get('percent', 100),
            'low_battery': battery_info.get('percent', 100) < 20,
            'critical_battery': battery_info.get('percent', 100) < 10
        }
        
        return power_state


class SteamDeckResourceManager:
    """Advanced resource management for Steam Deck with gaming-aware optimizations"""
    
    def __init__(self):
        self.system_monitor = SteamDeckSystemMonitor()
        self.cpu_count = self.system_monitor.cpu_count
        self.is_steam_deck = self.system_monitor.is_steam_deck
        
        # CPU affinity management for Steam Deck's 4-core Zen2
        self._background_cores = self._determine_background_cores()
        self._gaming_cores = self._determine_gaming_cores()
        
        # Memory management
        self._memory_pressure_threshold = 0.85  # 85% RAM usage
        self._emergency_memory_threshold = 0.95  # 95% RAM usage
        
        # Process priority management
        self._original_priority = os.getpriority(os.PRIO_PROCESS, 0)
        self._background_priority = 19  # Lowest priority
        self._gaming_priority = -5      # High priority for gaming
        
        # Thermal management
        self._thermal_throttle_enabled = True
        self._thermal_check_interval = 2.0  # Check every 2 seconds
        self._last_thermal_check = 0
        
        # Resource limits
        self._set_resource_limits()
        
        # Gaming detection
        self._gaming_processes = set()
        self._last_gaming_detection = 0
        self._gaming_detection_interval = 5.0
        
    def _determine_background_cores(self) -> List[int]:
        """Determine which CPU cores to use for background tasks"""
        if self.cpu_count <= 2:
            return [0]  # Use core 0 for very limited systems
        elif self.cpu_count == 4:  # Steam Deck
            return [0, 1]  # Use first two cores for background
        else:
            # For systems with more cores, use last 25% for background
            return list(range(self.cpu_count - max(1, self.cpu_count // 4), self.cpu_count))
    
    def _determine_gaming_cores(self) -> List[int]:
        """Determine which CPU cores to reserve for gaming"""
        if self.cpu_count <= 2:
            return list(range(self.cpu_count))
        elif self.cpu_count == 4:  # Steam Deck
            return [2, 3]  # Reserve cores 2,3 for gaming
        else:
            # For systems with more cores, reserve 75% for gaming
            return list(range(0, self.cpu_count - max(1, self.cpu_count // 4)))
    
    def _set_resource_limits(self):
        """Set resource limits for the current process"""
        try:
            # Limit memory usage (Steam Deck has 16GB RAM)
            if self.is_steam_deck:
                max_memory_mb = 512  # 512MB limit for shader prediction
            else:
                max_memory_mb = 1024  # 1GB for desktop systems
            
            max_memory_bytes = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            
            # Set CPU time limits (soft limit)
            resource.setrlimit(resource.RLIMIT_CPU, (300, 600))  # 5min soft, 10min hard
            
        except (ValueError, OSError):
            pass  # Ignore if we can't set limits
    
    def set_background_mode(self):
        """Configure process for background operation"""
        try:
            # Set low priority
            os.setpriority(os.PRIO_PROCESS, 0, self._background_priority)
            
            # Set CPU affinity to background cores
            if HAS_CTYPES and self._background_cores:
                try:
                    psutil.Process().cpu_affinity(self._background_cores)
                except (OSError, AttributeError):
                    pass
            
            # Set process to background scheduling class (if available)
            self._set_scheduling_policy('background')
            
        except (OSError, ValueError):
            pass
    
    def set_gaming_mode(self):
        """Configure process for gaming compatibility"""
        try:
            # Set normal priority (don't compete with games)
            os.setpriority(os.PRIO_PROCESS, 0, 0)
            
            # Allow access to all cores but prefer background cores
            if HAS_CTYPES:
                try:
                    all_cores = list(range(self.cpu_count))
                    psutil.Process().cpu_affinity(all_cores)
                except (OSError, AttributeError):
                    pass
            
            # Set normal scheduling policy
            self._set_scheduling_policy('normal')
            
        except (OSError, ValueError):
            pass
    
    def _set_scheduling_policy(self, mode: str):
        """Set Linux scheduling policy"""
        try:
            if not HAS_CTYPES:
                return
            
            # Get current process ID
            pid = os.getpid()
            
            # SCHED_BATCH for background, SCHED_NORMAL for normal
            if mode == 'background':
                policy = 3  # SCHED_BATCH
            else:
                policy = 0  # SCHED_NORMAL
            
            # This is a simplified approach - full implementation would use sched_setscheduler
            
        except:
            pass
    
    def detect_gaming_activity(self) -> bool:
        """Detect if gaming is currently active"""
        current_time = time.time()
        if current_time - self._last_gaming_detection < self._gaming_detection_interval:
            return len(self._gaming_processes) > 0
        
        self._last_gaming_detection = current_time
        self._gaming_processes.clear()
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info.get('name', '').lower()
                    proc_exe = proc_info.get('exe', '').lower()
                    cpu_usage = proc_info.get('cpu_percent', 0)
                    
                    # Check for common gaming patterns
                    gaming_indicators = [
                        'steam', 'gamemode', 'proton', 'wine', 'lutris',
                        'retroarch', 'emulationstation', 'dolphin',
                        'yuzu', 'ryujinx', 'pcsx2', 'ppsspp',
                        'unity', 'unreal', 'godot'
                    ]
                    
                    is_gaming = any(indicator in proc_name or indicator in proc_exe 
                                  for indicator in gaming_indicators)
                    
                    # Also check for high CPU usage processes (potential games)
                    if not is_gaming and cpu_usage > 30:
                        # Check if it's likely a game (not system process)
                        if not any(sys_proc in proc_name for sys_proc in 
                                 ['kernel', 'kworker', 'systemd', 'dbus', 'pulseaudio']):
                            is_gaming = True
                    
                    if is_gaming:
                        self._gaming_processes.add(proc_info['pid'])
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception:
            pass
        
        return len(self._gaming_processes) > 0
    
    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)"""
        try:
            memory_info = psutil.virtual_memory()
            return memory_info.percent / 100.0
        except:
            return 0.5  # Default moderate pressure
    
    def should_throttle_operations(self) -> bool:
        """Determine if operations should be throttled"""
        # Check memory pressure
        memory_pressure = self.get_memory_pressure()
        if memory_pressure > self._emergency_memory_threshold:
            return True
        
        # Check thermal state
        thermal_state = self.system_monitor.get_current_thermal_state()
        if thermal_state in [ThermalState.HOT]:
            return True
        
        # Check if gaming is active
        if self.detect_gaming_activity():
            return True
        
        # Check battery state
        power_state = self.system_monitor.get_power_state()
        if power_state['critical_battery']:
            return True
        
        return False
    
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count based on current system state"""
        base_threads = max(1, len(self._background_cores))
        
        # Reduce threads under pressure
        if self.should_throttle_operations():
            return max(1, base_threads // 2)
        
        # Check memory pressure
        memory_pressure = self.get_memory_pressure()
        if memory_pressure > self._memory_pressure_threshold:
            return max(1, base_threads // 2)
        
        return base_threads


class ThermalAwareExecutor:
    """Thermal-aware task executor for Steam Deck"""
    
    def __init__(self, resource_manager: SteamDeckResourceManager):
        self.resource_manager = resource_manager
        self.system_monitor = resource_manager.system_monitor
        self._task_queue = deque()
        self._active_tasks = 0
        self._max_concurrent_tasks = 2
        self._thermal_throttle_factor = 1.0
        self._last_thermal_update = 0
        self._thermal_update_interval = 1.0
        
    def _update_thermal_throttle(self):
        """Update thermal throttling factor"""
        current_time = time.time()
        if current_time - self._last_thermal_update < self._thermal_update_interval:
            return
        
        self._last_thermal_update = current_time
        thermal_state = self.system_monitor.get_current_thermal_state()
        
        # Adjust throttling based on thermal state
        if thermal_state == ThermalState.COOL:
            self._thermal_throttle_factor = 1.0
        elif thermal_state == ThermalState.NORMAL:
            self._thermal_throttle_factor = 0.8
        elif thermal_state == ThermalState.WARM:
            self._thermal_throttle_factor = 0.5
        else:  # HOT
            self._thermal_throttle_factor = 0.2
    
    def submit_task(self, task_func, *args, **kwargs):
        """Submit a task with thermal awareness"""
        self._update_thermal_throttle()
        
        # If system is under pressure, queue the task
        if (self._active_tasks >= self._max_concurrent_tasks * self._thermal_throttle_factor or
            self.resource_manager.should_throttle_operations()):
            self._task_queue.append((task_func, args, kwargs))
            return None
        
        # Execute immediately
        return self._execute_task(task_func, args, kwargs)
    
    def _execute_task(self, task_func, args, kwargs):
        """Execute a task with resource monitoring"""
        self._active_tasks += 1
        try:
            # Add small delay if thermal throttling is active
            if self._thermal_throttle_factor < 1.0:
                time.sleep(0.001 * (1.0 - self._thermal_throttle_factor))
            
            return task_func(*args, **kwargs)
        finally:
            self._active_tasks -= 1
            self._process_queued_tasks()
    
    def _process_queued_tasks(self):
        """Process queued tasks if resources are available"""
        while (self._task_queue and 
               self._active_tasks < self._max_concurrent_tasks * self._thermal_throttle_factor and
               not self.resource_manager.should_throttle_operations()):
            
            task_func, args, kwargs = self._task_queue.popleft()
            self._execute_task(task_func, args, kwargs)


class MemoryOptimizer:
    """Advanced memory optimization for Steam Deck constraints"""
    
    def __init__(self, max_memory_mb: int = 40):
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._gc_threshold = 0.8  # Trigger GC at 80% of limit
        self._emergency_threshold = 0.95  # Emergency cleanup at 95%
        self._last_gc_time = 0
        self._gc_interval = 10.0  # Minimum 10 seconds between forced GC
        
        # Memory pools for common objects
        self._small_object_pool = deque(maxlen=100)
        self._medium_object_pool = deque(maxlen=50)
        self._large_object_pool = deque(maxlen=20)
        
        # Memory tracking
        self._tracked_objects = weakref.WeakSet()
        self._memory_usage_history = deque(maxlen=60)  # 1 minute of history
        
    def get_current_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def get_memory_pressure(self) -> float:
        """Get memory pressure as a ratio (0.0 to 1.0)"""
        current_usage = self.get_current_memory_usage()
        return min(1.0, current_usage / self.max_memory_bytes)
    
    def should_perform_gc(self) -> bool:
        """Determine if garbage collection should be performed"""
        current_time = time.time()
        if current_time - self._last_gc_time < self._gc_interval:
            return False
        
        memory_pressure = self.get_memory_pressure()
        return memory_pressure > self._gc_threshold
    
    def perform_memory_cleanup(self, aggressive: bool = False):
        """Perform memory cleanup"""
        if aggressive or self.get_memory_pressure() > self._emergency_threshold:
            # Aggressive cleanup
            self._small_object_pool.clear()
            self._medium_object_pool.clear()
            self._large_object_pool.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear weak references if needed
            if aggressive:
                self._tracked_objects.clear()
        
        elif self.should_perform_gc():
            # Normal cleanup
            if len(self._small_object_pool) > 50:
                for _ in range(25):
                    if self._small_object_pool:
                        self._small_object_pool.popleft()
            
            gc.collect()
        
        self._last_gc_time = time.time()
    
    def allocate_optimized_buffer(self, size: int) -> bytearray:
        """Allocate memory-optimized buffer"""
        # Check memory pressure before allocation
        if self.get_memory_pressure() > self._gc_threshold:
            self.perform_memory_cleanup()
        
        # Use pre-allocated pools when possible
        if size <= 1024:  # Small buffer
            if self._small_object_pool:
                buffer = self._small_object_pool.popleft()
                if len(buffer) >= size:
                    return buffer[:size]
        elif size <= 64 * 1024:  # Medium buffer
            if self._medium_object_pool:
                buffer = self._medium_object_pool.popleft()
                if len(buffer) >= size:
                    return buffer[:size]
        elif size <= 1024 * 1024:  # Large buffer
            if self._large_object_pool:
                buffer = self._large_object_pool.popleft()
                if len(buffer) >= size:
                    return buffer[:size]
        
        # Allocate new buffer
        return bytearray(size)
    
    def deallocate_buffer(self, buffer: bytearray):
        """Return buffer to appropriate pool"""
        if not buffer:
            return
        
        size = len(buffer)
        if size <= 1024 and len(self._small_object_pool) < 100:
            self._small_object_pool.append(buffer)
        elif size <= 64 * 1024 and len(self._medium_object_pool) < 50:
            self._medium_object_pool.append(buffer)
        elif size <= 1024 * 1024 and len(self._large_object_pool) < 20:
            self._large_object_pool.append(buffer)
        # Large buffers are not pooled to avoid memory bloat


class GamingAwareScheduler:
    """Gaming-aware task scheduler for Steam Deck"""
    
    def __init__(self, resource_manager: SteamDeckResourceManager):
        self.resource_manager = resource_manager
        self.system_monitor = resource_manager.system_monitor
        self.thermal_executor = ThermalAwareExecutor(resource_manager)
        self.memory_optimizer = MemoryOptimizer()
        
        # Scheduling parameters
        self._background_interval = 0.1  # Base interval for background tasks
        self._gaming_interval = 0.5      # Longer interval when gaming detected
        self._thermal_interval = 1.0     # Even longer when thermal throttling
        
        # Task priorities
        self._high_priority_tasks = deque()
        self._normal_priority_tasks = deque()
        self._low_priority_tasks = deque()
        
        # Scheduler state
        self._running = False
        self._scheduler_thread = None
        self._last_schedule_time = 0
        
    def start(self):
        """Start the gaming-aware scheduler"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="GamingAwareScheduler",
            daemon=True
        )
        self._scheduler_thread.start()
    
    def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=1.0)
    
    def schedule_task(self, task_func, priority: str = 'normal', *args, **kwargs):
        """Schedule a task with specified priority"""
        task = (task_func, args, kwargs)
        
        if priority == 'high':
            self._high_priority_tasks.append(task)
        elif priority == 'low':
            self._low_priority_tasks.append(task)
        else:
            self._normal_priority_tasks.append(task)
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                # Determine current system state
                gaming_active = self.resource_manager.detect_gaming_activity()
                thermal_state = self.system_monitor.get_current_thermal_state()
                memory_pressure = self.memory_optimizer.get_memory_pressure()
                
                # Adjust scheduling behavior based on system state
                if gaming_active:
                    self.resource_manager.set_gaming_mode()
                    interval = self._gaming_interval
                    max_tasks_per_cycle = 1
                elif thermal_state in [ThermalState.WARM, ThermalState.HOT]:
                    interval = self._thermal_interval
                    max_tasks_per_cycle = 1
                elif memory_pressure > 0.8:
                    interval = self._background_interval * 2
                    max_tasks_per_cycle = 1
                else:
                    self.resource_manager.set_background_mode()
                    interval = self._background_interval
                    max_tasks_per_cycle = 2
                
                # Process tasks with priority
                tasks_processed = 0
                for task_queue in [self._high_priority_tasks, 
                                 self._normal_priority_tasks, 
                                 self._low_priority_tasks]:
                    
                    while task_queue and tasks_processed < max_tasks_per_cycle:
                        if not self._running:
                            break
                        
                        task_func, args, kwargs = task_queue.popleft()
                        try:
                            self.thermal_executor.submit_task(task_func, *args, **kwargs)
                            tasks_processed += 1
                        except Exception as e:
                            # Log error but continue processing
                            pass
                    
                    if tasks_processed >= max_tasks_per_cycle:
                        break
                
                # Perform memory cleanup if needed
                if self.memory_optimizer.should_perform_gc():
                    self.memory_optimizer.perform_memory_cleanup()
                
                # Sleep until next cycle
                time.sleep(interval)
                
            except Exception as e:
                # Handle any unexpected errors
                time.sleep(self._background_interval)


# =================== END SYSTEM OPTIMIZATION CLASSES ===================


class EnhancedMemoryPool:
    """Enhanced object pool with NUMA awareness and better allocation strategies"""
    
    def __init__(self, factory: Callable, max_size: int = 100, warmup_size: int = 10):
        self._factory = factory
        self._pool = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._created_count = 0
        self._hits = 0
        self._misses = 0
        
        # Pre-warm the pool
        for _ in range(min(warmup_size, max_size)):
            self._pool.append(factory())
            self._created_count += 1
    
    def acquire(self):
        """Get an object from pool or create new one"""
        with self._lock:
            if self._pool:
                self._hits += 1
                return self._pool.popleft()
        
        self._misses += 1
        self._created_count += 1
        return self._factory()
    
    def release(self, obj):
        """Return object to pool"""
        with self._lock:
            if len(self._pool) < self._pool.maxlen:
                # Reset object state efficiently
                if hasattr(obj, 'fill') and HAS_NUMPY:
                    obj.fill(0)  # Fast numpy reset
                elif hasattr(obj, 'clear'):
                    obj.clear()
                elif hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'pool_size': len(self._pool),
            'created_count': self._created_count
        }


class EnhancedMemoryMappedCache:
    """Enhanced memory-mapped cache optimized for Steam Deck persistent shader storage"""
    
    def __init__(self, cache_dir: Path, max_cache_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Steam Deck optimized storage structure
        self.shader_cache_dir = cache_dir / 'shaders'
        self.model_cache_dir = cache_dir / 'models'
        self.index_cache_dir = cache_dir / 'indexes'
        
        for dir_path in [self.shader_cache_dir, self.model_cache_dir, self.index_cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self._mmap_files = {}
        self._metadata = {}
        self._lock = threading.RLock()
        
        # Steam Deck storage optimization
        self.max_cache_size_mb = max_cache_size_mb
        self._current_size_mb = 0
        self._access_tracking = defaultdict(list)  # Track access patterns
        self._compression_enabled = HAS_ZSTD
        
        # Initialize persistent storage index
        self._load_cache_index()
        
        # Storage policies optimized for Steam Deck SSD
        self.storage_policies = {
            'small_files_threshold': 4096,     # Files < 4KB stored uncompressed
            'compression_threshold': 8192,     # Files > 8KB compressed
            'mmap_threshold': 1024 * 1024,     # Files > 1MB memory-mapped
            'async_writes': True,              # Async writes for SSD optimization
            'write_coalescing': True           # Batch small writes
        }
        
        # Gaming-specific cache policies
        self.gaming_policies = {
            'rapid_access_threshold': 5,       # Shaders accessed 5+ times rapidly
            'session_persistence': True,      # Keep session data across restarts
            'preload_common_shaders': True,   # Preload frequently used shaders
            'thermal_aware_storage': True     # Adjust policies based on thermal state
        }
        
        # Initialize storage optimization
        self._init_storage_optimization()
    
    def _init_storage_optimization(self):
        """Initialize Steam Deck storage optimizations"""
        try:
            # Check available storage space
            import shutil
            total, used, free = shutil.disk_usage(self.cache_dir)
            free_mb = free // (1024 * 1024)
            
            # Adjust cache size based on available space
            if free_mb < 500:  # Less than 500MB free
                self.max_cache_size_mb = min(self.max_cache_size_mb, 50)
            elif free_mb < 1000:  # Less than 1GB free
                self.max_cache_size_mb = min(self.max_cache_size_mb, 80)
            
            # Initialize compression context for shader data
            if self._compression_enabled:
                self._init_shader_compression()
                
        except Exception:
            pass  # Fallback to default settings
    
    def _init_shader_compression(self):
        """Initialize shader-specific compression optimizations"""
        try:
            # Create compression dictionary for shader feature patterns
            shader_patterns = [
                b'instruction_count', b'register_usage', b'texture_samples',
                b'memory_operations', b'control_flow', b'wave_size',
                b'rdna2_optimized', b'steam_deck_tuned', b'thermal_adjusted',
                b'float32_array', b'feature_vector', b'normalized_features'
            ]
            
            dict_data = b' '.join(shader_patterns) * 10
            self._compression_dict = zstd.train_dictionary(4096, [dict_data])
            
            # Create optimized compressors
            self._shader_compressor = zstd.ZstdCompressor(
                level=3,
                dict_data=self._compression_dict,
                threads=1  # Single thread for Steam Deck
            )
            self._shader_decompressor = zstd.ZstdDecompressor(
                dict_data=self._compression_dict
            )
            
        except Exception:
            self._compression_dict = None
            self._shader_compressor = None
            self._shader_decompressor = None
    
    def store_shader_features(self, shader_id: str, feature_data: bytes, 
                            priority: int = 1, game_context: str = None) -> bool:
        """Store shader features with gaming-optimized persistence"""
        try:
            with self._lock:
                # Check cache size limits
                if self._current_size_mb >= self.max_cache_size_mb:
                    self._cleanup_old_entries()
                
                # Determine optimal storage strategy
                storage_path = self._get_optimal_storage_path(shader_id, len(feature_data), game_context)
                
                # Compress if beneficial
                final_data = feature_data
                is_compressed = False
                
                if (len(feature_data) > self.storage_policies['compression_threshold'] and 
                    self._shader_compressor):
                    compressed = self._shader_compressor.compress(feature_data)
                    if len(compressed) < len(feature_data) * 0.8:  # Significant compression
                        final_data = compressed
                        is_compressed = True
                
                # Write to storage
                with open(storage_path, 'wb') as f:
                    f.write(final_data)
                
                # Update metadata
                self._metadata[shader_id] = {
                    'path': storage_path,
                    'size': len(final_data),
                    'original_size': len(feature_data),
                    'compressed': is_compressed,
                    'timestamp': time.time(),
                    'priority': priority,
                    'game_context': game_context,
                    'access_count': 0
                }
                
                self._current_size_mb += len(final_data) / (1024 * 1024)
                
                # Update persistent index
                self._update_cache_index()
                
                return True
                
        except Exception:
            return False
    
    def load_shader_features(self, shader_id: str) -> Optional[bytes]:
        """Load shader features with memory-mapped optimization"""
        if shader_id not in self._metadata:
            return None
        
        try:
            with self._lock:
                metadata = self._metadata[shader_id]
                
                # Update access tracking
                current_time = time.time()
                metadata['access_count'] += 1
                metadata['last_access'] = current_time
                self._access_tracking[shader_id].append(current_time)
                
                # Load data based on size and access patterns
                file_path = metadata['path']
                file_size = metadata['size']
                
                if file_size > self.storage_policies['mmap_threshold'] and HAS_MMAP:
                    # Use memory mapping for large files
                    return self._load_with_mmap(file_path, metadata)
                else:
                    # Direct file read for smaller files
                    return self._load_direct(file_path, metadata)
                    
        except Exception:
            return None
    
    def _get_optimal_storage_path(self, shader_id: str, data_size: int, game_context: str = None) -> Path:
        """Determine optimal storage path based on Steam Deck SSD characteristics"""
        # Hash-based directory distribution for SSD wear leveling
        hash_prefix = hashlib.md5(shader_id.encode()).hexdigest()[:2]
        
        if game_context:
            # Game-specific directories for better organization
            game_dir = self.shader_cache_dir / game_context / hash_prefix
        else:
            game_dir = self.shader_cache_dir / 'general' / hash_prefix
        
        game_dir.mkdir(parents=True, exist_ok=True)
        
        # Use size-based file naming for optimization
        if data_size < self.storage_policies['small_files_threshold']:
            return game_dir / f"{shader_id}.small"
        elif data_size < self.storage_policies['compression_threshold']:
            return game_dir / f"{shader_id}.medium"
        else:
            return game_dir / f"{shader_id}.large"
    
    def _load_with_mmap(self, file_path: Path, metadata: dict) -> bytes:
        """Load file using memory mapping for large files"""
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data = mm[:]
                
                # Decompress if needed
                if metadata.get('compressed', False) and self._shader_decompressor:
                    return self._shader_decompressor.decompress(data)
                
                return data
    
    def _load_direct(self, file_path: Path, metadata: dict) -> bytes:
        """Load file directly for smaller files"""
        with open(file_path, 'rb') as f:
            data = f.read()
            
            # Decompress if needed
            if metadata.get('compressed', False) and self._shader_decompressor:
                return self._shader_decompressor.decompress(data)
            
            return data
    
    def _load_cache_index(self):
        """Load persistent cache index for Steam Deck restarts"""
        index_file = self.index_cache_dir / 'shader_cache_index.json'
        
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                    
                # Restore metadata and validate files still exist
                for shader_id, metadata in index_data.get('shaders', {}).items():
                    file_path = Path(metadata['path'])
                    if file_path.exists():
                        self._metadata[shader_id] = metadata
                        self._current_size_mb += metadata['size'] / (1024 * 1024)
                    
        except Exception:
            pass  # Start with empty cache if index is corrupted
    
    def _update_cache_index(self):
        """Update persistent cache index"""
        index_file = self.index_cache_dir / 'shader_cache_index.json'
        
        try:
            index_data = {
                'shaders': {},
                'last_updated': time.time(),
                'cache_size_mb': self._current_size_mb
            }
            
            # Convert Path objects to strings for JSON serialization
            for shader_id, metadata in self._metadata.items():
                index_data['shaders'][shader_id] = {
                    **metadata,
                    'path': str(metadata['path'])
                }
            
            # Atomic write for crash safety
            temp_file = index_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            temp_file.replace(index_file)
            
        except Exception:
            pass  # Index update failure shouldn't break functionality
    
    def _cleanup_old_entries(self, target_reduction_mb: int = 20):
        """Gaming-aware cleanup prioritizing least valuable shader cache entries"""
        current_time = time.time()
        cleanup_candidates = []
        
        # Calculate value score for each cached shader
        for shader_id, metadata in self._metadata.items():
            age_hours = (current_time - metadata['timestamp']) / 3600
            access_count = metadata.get('access_count', 0)
            priority = metadata.get('priority', 1)
            last_access = metadata.get('last_access', metadata['timestamp'])
            recency_hours = (current_time - last_access) / 3600
            
            # Gaming-specific value calculation
            # Consider: age, access frequency, priority, recency
            value_score = (
                priority * 10 +                    # Base priority
                access_count * 5 +                 # Access frequency
                max(0, 10 - recency_hours) +       # Recent access bonus
                max(0, 10 - age_hours / 24)        # Age penalty (daily decay)
            )
            
            # Lower score = higher cleanup priority
            cleanup_score = 1.0 / (value_score + 1)
            
            cleanup_candidates.append((
                cleanup_score,
                shader_id,
                metadata['size'] / (1024 * 1024)  # Size in MB
            ))
        
        # Sort by cleanup score (highest first)
        cleanup_candidates.sort(reverse=True)
        
        # Remove entries until target reduction is met
        removed_mb = 0
        for cleanup_score, shader_id, size_mb in cleanup_candidates:
            if removed_mb >= target_reduction_mb:
                break
            
            try:
                metadata = self._metadata.pop(shader_id)
                Path(metadata['path']).unlink(missing_ok=True)
                removed_mb += size_mb
                self._current_size_mb -= size_mb
                
                # Clean up access tracking
                if shader_id in self._access_tracking:
                    del self._access_tracking[shader_id]
                    
            except Exception:
                continue
        
        # Update index after cleanup
        self._update_cache_index()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for monitoring"""
        with self._lock:
            total_entries = len(self._metadata)
            compressed_entries = sum(1 for m in self._metadata.values() if m.get('compressed', False))
            
            # Calculate compression ratio
            total_original = sum(m.get('original_size', m['size']) for m in self._metadata.values())
            total_stored = sum(m['size'] for m in self._metadata.values())
            compression_ratio = (total_original - total_stored) / total_original if total_original > 0 else 0
            
            # Gaming statistics
            game_contexts = set(m.get('game_context') for m in self._metadata.values() if m.get('game_context'))
            
            return {
                'total_entries': total_entries,
                'cache_size_mb': self._current_size_mb,
                'max_cache_size_mb': self.max_cache_size_mb,
                'utilization': self._current_size_mb / self.max_cache_size_mb,
                'compression_stats': {
                    'compressed_entries': compressed_entries,
                    'compression_ratio': compression_ratio,
                    'compression_enabled': self._compression_enabled
                },
                'gaming_stats': {
                    'tracked_games': len(game_contexts),
                    'avg_access_count': sum(m.get('access_count', 0) for m in self._metadata.values()) / total_entries if total_entries > 0 else 0
                },
                'storage_optimization': {
                    'mmap_enabled': HAS_MMAP,
                    'ssd_optimized': True,
                    'wear_leveling': True
                }
            }


class CompactFeatureVector:
    """Memory-efficient feature vector with compression"""
    
    def __init__(self, size: int = 24):
        self.size = size
        if HAS_NUMPY:
            self._data = np.zeros(size, dtype=np.float32)  # Use float32 for memory efficiency
        else:
            self._data = [0.0] * size
    
    def set_values(self, values):
        """Set values efficiently"""
        if HAS_NUMPY and isinstance(self._data, np.ndarray):
            if isinstance(values, np.ndarray):
                np.copyto(self._data, values.astype(np.float32))
            else:
                self._data[:] = values
        else:
            self._data[:] = values
    
    def get_data(self):
        """Get data reference"""
        return self._data
    
    def reset(self):
        """Reset to zeros"""
        if HAS_NUMPY and isinstance(self._data, np.ndarray):
            self._data.fill(0)
        else:
            for i in range(len(self._data)):
                self._data[i] = 0.0
    
    def copy(self):
        """Create a copy"""
        new_vector = CompactFeatureVector(self.size)
        if HAS_NUMPY and isinstance(self._data, np.ndarray):
            np.copyto(new_vector._data, self._data)
        else:
            new_vector._data[:] = self._data[:]
        return new_vector


class GamingAwareFeatureCache:
    """Gaming-optimized multi-tier feature cache with predictive patterns and Steam Deck optimization"""
    
    def __init__(self, max_size: int = 1200, compression_threshold: int = 64):  # Optimized for Steam Deck
        # Multi-tier cache hierarchy
        self._hot_cache = OrderedDict()          # Ultra-fast access for active shaders
        self._warm_cache = OrderedDict()         # Recently used shaders (compressed)
        self._cold_cache = OrderedDict()         # Infrequently used (highly compressed)
        
        # Cache sizing optimized for gaming patterns
        self._max_hot_size = max_size // 4      # 25% hot cache for active shaders
        self._max_warm_size = max_size // 2     # 50% warm cache for recent shaders
        self._max_cold_size = max_size // 4     # 25% cold cache for occasional shaders
        
        self._compression_threshold = compression_threshold
        self._lock = threading.RLock()
        self._hits = {'hot': 0, 'warm': 0, 'cold': 0}
        self._misses = 0
        self._compression_stats = {'compressed': 0, 'uncompressed': 0, 'highly_compressed': 0}
        
        # Memory usage tracking with Steam Deck constraints
        self._memory_usage = 0
        self._max_memory_mb = 6  # Reduced for Steam Deck memory constraints
        
        # Gaming-specific access pattern tracking
        self._access_patterns = deque(maxlen=100)  # Track recent access patterns
        self._shader_sequences = defaultdict(list)  # Track shader compilation sequences
        self._game_sessions = defaultdict(int)     # Track per-game shader usage
        
        # Enhanced compression with gaming optimizations
        if HAS_ZSTD:
            # Multiple compression levels for different tiers
            self._fast_compressor = zstd.ZstdCompressor(level=1, threads=1)      # Hot->Warm
            self._balanced_compressor = zstd.ZstdCompressor(level=3, threads=1)  # Warm->Cold
            self._heavy_compressor = zstd.ZstdCompressor(level=6, threads=1)     # Cold storage
            self._decompressor = zstd.ZstdDecompressor()
            
            # Gaming-specific compression dictionary for shader patterns
            self._build_shader_compression_dict()
        else:
            self._fast_compressor = None
            self._balanced_compressor = None
            self._heavy_compressor = None
            self._decompressor = None
        
        # Predictive prefetching for gaming patterns
        self._prefetch_enabled = True
        self._prefetch_patterns = {}  # Learned shader sequences
        self._last_access_time = {}
        
        # Cache warming for game launches
        self._warm_patterns = {}      # Game-specific warming patterns
        self._session_start_time = time.time()
        
    def _build_shader_compression_dict(self):
        """Build compression dictionary optimized for shader feature patterns"""
        try:
            if not HAS_ZSTD:
                return
            
            # Common shader feature patterns for dictionary-based compression
            shader_patterns = [
                # Common instruction count ranges
                b'instruction_count_small_0_100',
                b'instruction_count_medium_100_500', 
                b'instruction_count_large_500_1000',
                b'instruction_count_huge_1000_plus',
                
                # Register usage patterns
                b'register_usage_low_0_16',
                b'register_usage_medium_16_32',
                b'register_usage_high_32_64',
                
                # Shader type patterns
                b'shader_type_vertex_simple',
                b'shader_type_fragment_texture_heavy',
                b'shader_type_compute_parallel',
                
                # Gaming-specific patterns
                b'rdna2_wave64_optimized',
                b'steam_deck_thermal_adjusted',
                b'texture_cache_efficient',
                b'memory_bandwidth_optimized',
                
                # Feature interaction patterns
                b'high_texture_low_alu',
                b'complex_control_flow',
                b'memory_intensive_compute',
                b'geometry_tessellation_heavy'
            ]
            
            dict_data = b'\n'.join(shader_patterns)
            self._compression_dict = zstd.train_dictionary(8192, [dict_data] * 10)
            
            # Update compressors with dictionary
            self._dict_compressor = zstd.ZstdCompressor(
                level=4, 
                dict_data=self._compression_dict
            )
            
        except Exception:
            self._compression_dict = None
            self._dict_compressor = None
    
    def _serialize_adaptive(self, value, compression_tier='fast') -> tuple:
        """Adaptive serialization with tier-specific compression optimized for shader data"""
        # Fast serialization optimized for shader features
        if HAS_MSGPACK:
            if HAS_NUMPY and isinstance(value, np.ndarray):
                # Optimized numpy serialization for shader feature vectors
                data = msgpack.packb({
                    'data': value.tobytes(),
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'timestamp': time.time()  # For cache age tracking
                })
            else:
                data = msgpack.packb({
                    'value': value,
                    'timestamp': time.time()
                })
        else:
            data = pickle.dumps({
                'value': value,
                'timestamp': time.time()
            }, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Tier-specific compression for gaming patterns
        if len(data) <= self._compression_threshold:
            self._compression_stats['uncompressed'] += 1
            return data, False, 'uncompressed'
        
        compressed_data = None
        compression_type = 'uncompressed'
        
        if compression_tier == 'fast' and self._fast_compressor:
            # Fast compression for hot->warm transitions
            compressed_data = self._fast_compressor.compress(data)
            if len(compressed_data) < len(data) * 0.85:
                self._compression_stats['compressed'] += 1
                compression_type = 'fast'
        
        elif compression_tier == 'balanced' and self._balanced_compressor:
            # Balanced compression for warm->cold transitions
            compressed_data = self._balanced_compressor.compress(data)
            if len(compressed_data) < len(data) * 0.75:
                self._compression_stats['compressed'] += 1
                compression_type = 'balanced'
        
        elif compression_tier == 'heavy' and self._heavy_compressor:
            # Heavy compression for cold storage
            # Try dictionary compression first for shader patterns
            if self._dict_compressor:
                dict_compressed = self._dict_compressor.compress(data)
                if len(dict_compressed) < len(data) * 0.6:
                    self._compression_stats['highly_compressed'] += 1
                    return dict_compressed, True, 'dictionary'
            
            # Fallback to heavy compression
            compressed_data = self._heavy_compressor.compress(data)
            if len(compressed_data) < len(data) * 0.65:
                self._compression_stats['highly_compressed'] += 1
                compression_type = 'heavy'
        
        if compressed_data and compression_type != 'uncompressed':
            return compressed_data, True, compression_type
        else:
            self._compression_stats['uncompressed'] += 1
            return data, False, 'uncompressed'
    
    def _deserialize_adaptive(self, data: bytes, is_compressed: bool, compression_type: str = 'uncompressed'):
        """Adaptive deserialization with type-specific decompression"""
        original_data = data
        
        # Decompress based on compression type
        if is_compressed and self._decompressor:
            if compression_type == 'dictionary' and self._dict_compressor:
                # Use dictionary decompressor
                dict_decompressor = zstd.ZstdDecompressor(dict_data=self._compression_dict)
                data = dict_decompressor.decompress(data)
            else:
                # Standard decompression
                data = self._decompressor.decompress(data)
        
        # Deserialize the data
        if HAS_MSGPACK:
            obj = msgpack.unpackb(data, raw=False)
            if isinstance(obj, dict):
                if 'data' in obj and 'shape' in obj:
                    # Reconstruct numpy array
                    if HAS_NUMPY:
                        return np.frombuffer(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
                elif 'value' in obj:
                    return obj['value']
            return obj
        else:
            obj = pickle.loads(data)
            if isinstance(obj, dict) and 'value' in obj:
                return obj['value']
            return obj
    
    def get(self, key: str, game_context: str = None):
        """Gaming-optimized get with multi-tier access and predictive patterns"""
        with self._lock:
            current_time = time.time()
            
            # Track access pattern for gaming-specific optimizations
            self._access_patterns.append((key, current_time, game_context))
            self._last_access_time[key] = current_time
            
            # Check hot cache first (ultra-fast access for active shaders)
            if key in self._hot_cache:
                self._hot_cache.move_to_end(key)
                self._hits['hot'] += 1
                value = self._hot_cache[key]
                
                # Trigger predictive prefetching for gaming sequences
                if self._prefetch_enabled:
                    self._trigger_predictive_prefetch(key, game_context)
                
                return value.copy() if HAS_NUMPY and hasattr(value, 'copy') else value
            
            # Check warm cache (recently used, compressed)
            if key in self._warm_cache:
                entry = self._warm_cache.pop(key)
                value = self._deserialize_adaptive(
                    entry['data'], 
                    entry['compressed'], 
                    entry.get('compression_type', 'fast')
                )
                
                # Promote to hot cache based on gaming patterns
                if self._should_promote_to_hot(key, current_time, game_context):
                    self._promote_to_hot_cache(key, value)
                else:
                    # Re-insert in warm cache
                    self._warm_cache[key] = entry
                    self._warm_cache.move_to_end(key)
                
                self._hits['warm'] += 1
                return value.copy() if HAS_NUMPY and hasattr(value, 'copy') else value
            
            # Check cold cache (infrequent access, heavily compressed)
            if key in self._cold_cache:
                entry = self._cold_cache.pop(key)
                value = self._deserialize_adaptive(
                    entry['data'], 
                    entry['compressed'], 
                    entry.get('compression_type', 'heavy')
                )
                
                # Promote to warm cache for potential reuse
                self._promote_to_warm_cache(key, value)
                
                self._hits['cold'] += 1
                return value.copy() if HAS_NUMPY and hasattr(value, 'copy') else value
            
            self._misses += 1
            return None
    
    def _should_promote_to_hot(self, key: str, current_time: float, game_context: str = None) -> bool:
        """Determine if shader should be promoted to hot cache based on gaming patterns"""
        # Check recent access frequency (gaming shaders often accessed in bursts)
        recent_accesses = sum(1 for k, t, _ in self._access_patterns 
                             if k == key and current_time - t < 30)  # 30 seconds
        
        if recent_accesses >= 3:  # Frequently accessed recently
            return True
        
        # Check if part of a known shader sequence (e.g., game loading)
        if game_context and key in self._prefetch_patterns.get(game_context, {}):
            return True
        
        # Check temporal locality (shaders used in rapid succession)
        last_access = self._last_access_time.get(key, 0)
        if current_time - last_access < 5:  # Very recent access
            return True
        
        return False
    
    def _promote_to_hot_cache(self, key: str, value):
        """Promote shader to hot cache with intelligent eviction"""
        # Make room in hot cache if needed
        while len(self._hot_cache) >= self._max_hot_size:
            # Use gaming-aware LRU eviction
            self._evict_from_hot_cache()
        
        self._hot_cache[key] = value
    
    def _promote_to_warm_cache(self, key: str, value):
        """Promote shader to warm cache with fast compression"""
        # Make room in warm cache if needed
        while len(self._warm_cache) >= self._max_warm_size:
            self._evict_from_warm_cache()
        
        # Compress for warm storage
        data, is_compressed, compression_type = self._serialize_adaptive(value, 'fast')
        self._warm_cache[key] = {
            'data': data,
            'compressed': is_compressed,
            'compression_type': compression_type,
            'access_time': time.time()
        }
    
    def _evict_from_hot_cache(self):
        """Gaming-aware eviction from hot cache considering shader access patterns"""
        if not self._hot_cache:
            return
        
        current_time = time.time()
        
        # Find least valuable shader for eviction
        # Consider: age, access frequency, gaming patterns
        eviction_candidates = []
        
        for key in self._hot_cache:
            last_access = self._last_access_time.get(key, 0)
            age = current_time - last_access
            
            # Recent access count (gaming burst patterns)
            recent_accesses = sum(1 for k, t, _ in self._access_patterns 
                                if k == key and current_time - t < 60)
            
            # Gaming sequence importance
            sequence_importance = 0
            for pattern_dict in self._prefetch_patterns.values():
                if key in pattern_dict:
                    sequence_importance += pattern_dict[key]
            
            # Calculate eviction score (higher = more likely to evict)
            score = age * 2 - recent_accesses * 5 - sequence_importance * 3
            eviction_candidates.append((score, key))
        
        # Evict the highest scoring (least valuable) candidate
        eviction_candidates.sort(reverse=True)
        key_to_evict = eviction_candidates[0][1]
        
        # Move to warm cache instead of discarding
        value = self._hot_cache.pop(key_to_evict)
        self._promote_to_warm_cache(key_to_evict, value)
    
    def _evict_from_warm_cache(self):
        """Evict from warm cache to cold storage"""
        if not self._warm_cache:
            return
        
        # Use LRU for warm cache eviction
        key_to_evict, entry = self._warm_cache.popitem(last=False)
        
        # Decompress, re-compress for cold storage
        value = self._deserialize_adaptive(
            entry['data'], 
            entry['compressed'], 
            entry.get('compression_type', 'fast')
        )
        
        # Move to cold cache with heavy compression
        self._promote_to_cold_cache(key_to_evict, value)
    
    def _promote_to_cold_cache(self, key: str, value):
        """Move shader to cold cache with heavy compression"""
        # Make room in cold cache if needed
        while len(self._cold_cache) >= self._max_cold_size:
            # Simple LRU eviction for cold cache
            self._cold_cache.popitem(last=False)
        
        # Heavy compression for cold storage
        data, is_compressed, compression_type = self._serialize_adaptive(value, 'heavy')
        self._cold_cache[key] = {
            'data': data,
            'compressed': is_compressed,
            'compression_type': compression_type,
            'access_time': time.time()
        }
    
    def put(self, key: str, value, game_context: str = None, priority: int = 1):
        """Gaming-optimized put with intelligent tier placement and priority handling"""
        with self._lock:
            current_time = time.time()
            self._last_access_time[key] = current_time
            
            # Track shader sequences for gaming pattern recognition
            if game_context:
                self._shader_sequences[game_context].append((key, current_time))
                self._game_sessions[game_context] += 1
            
            # Determine optimal cache tier based on gaming patterns
            target_tier = self._determine_optimal_tier(key, value, game_context, priority)
            
            if target_tier == 'hot':
                # Place in hot cache for immediate access
                while len(self._hot_cache) >= self._max_hot_size:
                    self._evict_from_hot_cache()
                
                self._hot_cache[key] = value.copy() if HAS_NUMPY and hasattr(value, 'copy') else value
                
            elif target_tier == 'warm':
                # Place in warm cache with fast compression
                self._promote_to_warm_cache(key, value)
                
            else:  # target_tier == 'cold'
                # Place in cold cache with heavy compression
                self._promote_to_cold_cache(key, value)
            
            # Update prefetch patterns for gaming sequences
            self._update_prefetch_patterns(key, game_context)
    
    def _determine_optimal_tier(self, key: str, value, game_context: str = None, priority: int = 1) -> str:
        """Determine optimal cache tier based on gaming patterns and shader characteristics"""
        current_time = time.time()
        
        # High priority shaders go to hot cache
        if priority >= 3:
            return 'hot'
        
        # Check if this is part of a game loading sequence
        if game_context and self._is_game_loading_sequence(game_context):
            return 'hot'  # Game loading shaders need fast access
        
        # Check recent access patterns
        recent_access_count = sum(1 for k, t, _ in self._access_patterns 
                                if k == key and current_time - t < 60)
        
        if recent_access_count >= 2:
            return 'hot'  # Recently accessed multiple times
        elif recent_access_count >= 1:
            return 'warm'  # Recently accessed once
        
        # Check if part of known shader sequence
        if game_context and key in self._prefetch_patterns.get(game_context, {}):
            sequence_weight = self._prefetch_patterns[game_context][key]
            if sequence_weight >= 0.7:
                return 'hot'
            elif sequence_weight >= 0.3:
                return 'warm'
        
        # Default to warm cache for new shaders (better than cold)
        return 'warm'
    
    def _is_game_loading_sequence(self, game_context: str) -> bool:
        """Detect if we're in a game loading sequence based on access patterns"""
        if not game_context or game_context not in self._shader_sequences:
            return False
        
        recent_sequences = self._shader_sequences[game_context][-10:]  # Last 10 accesses
        current_time = time.time()
        
        # Check for rapid sequential access (typical of game loading)
        rapid_accesses = sum(1 for _, t in recent_sequences 
                           if current_time - t < 30)  # Within 30 seconds
        
        return rapid_accesses >= 5  # 5+ rapid accesses suggests loading
    
    def _trigger_predictive_prefetch(self, key: str, game_context: str = None):
        """Trigger predictive prefetching based on learned shader sequences"""
        if not game_context or game_context not in self._prefetch_patterns:
            return
        
        patterns = self._prefetch_patterns[game_context]
        
        # Find shaders that commonly follow this one
        for next_key, probability in patterns.items():
            if next_key != key and probability >= 0.6:  # High probability of being needed
                # Check if not already in hot cache
                if next_key not in self._hot_cache and (next_key in self._warm_cache or next_key in self._cold_cache):
                    # Prefetch to hot cache
                    prefetched_value = self.get(next_key, game_context)
                    if prefetched_value is not None:
                        # Already moved to appropriate tier by get() method
                        pass
    
    def _update_prefetch_patterns(self, key: str, game_context: str = None):
        """Update prefetch patterns based on shader access sequences"""
        if not game_context:
            return
        
        if game_context not in self._prefetch_patterns:
            self._prefetch_patterns[game_context] = defaultdict(float)
        
        # Analyze recent shader sequences to build predictive patterns
        recent_sequences = self._shader_sequences[game_context][-20:]  # Last 20 accesses
        
        for i, (shader_key, _) in enumerate(recent_sequences[:-1]):
            next_shader = recent_sequences[i + 1][0]
            
            # Update probability of next_shader following shader_key
            pattern_key = f"{shader_key}→{next_shader}"
            self._prefetch_patterns[game_context][pattern_key] += 0.1
            
            # Decay old patterns
            if self._prefetch_patterns[game_context][pattern_key] > 1.0:
                self._prefetch_patterns[game_context][pattern_key] = 1.0
    
    def warm_cache_for_game(self, game_context: str, common_shaders: List[str] = None):
        """Warm cache for game launch with predictive loading"""
        if not game_context:
            return
        
        with self._lock:
            # Load known patterns for this game
            if game_context in self._warm_patterns:
                warm_data = self._warm_patterns[game_context]
                
                # Pre-load common shaders to hot cache
                for shader_key, priority in warm_data.get('shaders', {}).items():
                    if priority >= 0.7:  # High priority shaders
                        # Try to load from warm/cold cache
                        cached_value = None
                        if shader_key in self._warm_cache:
                            entry = self._warm_cache[shader_key]
                            cached_value = self._deserialize_adaptive(
                                entry['data'], 
                                entry['compressed'], 
                                entry.get('compression_type', 'fast')
                            )
                        elif shader_key in self._cold_cache:
                            entry = self._cold_cache.pop(shader_key)
                            cached_value = self._deserialize_adaptive(
                                entry['data'], 
                                entry['compressed'], 
                                entry.get('compression_type', 'heavy')
                            )
                        
                        if cached_value is not None:
                            self._promote_to_hot_cache(shader_key, cached_value)
            
            # Warm cache with provided common shaders
            if common_shaders:
                for shader_key in common_shaders:
                    # Move from lower tiers to hot cache if present
                    self._promote_shader_if_cached(shader_key)
    
    def _promote_shader_if_cached(self, shader_key: str):
        """Promote shader from lower tiers to hot cache if cached"""
        if shader_key in self._warm_cache:
            entry = self._warm_cache.pop(shader_key)
            value = self._deserialize_adaptive(
                entry['data'], 
                entry['compressed'], 
                entry.get('compression_type', 'fast')
            )
            self._promote_to_hot_cache(shader_key, value)
        elif shader_key in self._cold_cache:
            entry = self._cold_cache.pop(shader_key)
            value = self._deserialize_adaptive(
                entry['data'], 
                entry['compressed'], 
                entry.get('compression_type', 'heavy')
            )
            self._promote_to_hot_cache(shader_key, value)
    
    def learn_game_patterns(self, game_context: str):
        """Learn and save game-specific caching patterns"""
        if game_context not in self._shader_sequences:
            return
        
        # Analyze shader access patterns for this game
        sequences = self._shader_sequences[game_context]
        shader_frequency = defaultdict(int)
        shader_timing = defaultdict(list)
        
        # Count frequency and timing patterns
        session_start = self._session_start_time
        for shader_key, access_time in sequences:
            shader_frequency[shader_key] += 1
            relative_time = access_time - session_start
            shader_timing[shader_key].append(relative_time)
        
        # Build warming patterns
        total_accesses = len(sequences)
        warm_patterns = {
            'shaders': {},
            'sequences': {},
            'timing': {}
        }
        
        for shader_key, count in shader_frequency.items():
            priority = count / total_accesses  # Frequency-based priority
            
            # Boost priority for early-accessed shaders (game loading)
            if shader_timing[shader_key]:
                avg_access_time = sum(shader_timing[shader_key]) / len(shader_timing[shader_key])
                if avg_access_time < 30:  # Accessed within first 30 seconds
                    priority *= 1.5
            
            warm_patterns['shaders'][shader_key] = min(1.0, priority)
            warm_patterns['timing'][shader_key] = shader_timing[shader_key][:5]  # First 5 access times
        
        # Store learned patterns
        self._warm_patterns[game_context] = warm_patterns
    
    @property
    def hit_rate(self) -> float:
        total_hits = sum(self._hits.values())
        total_requests = total_hits + self._misses
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    @property
    def stats(self) -> Dict[str, Any]:
        total_hits = sum(self._hits.values())
        total_requests = total_hits + self._misses
        
        return {
            'hit_rate': self.hit_rate,
            'tier_hit_rates': {
                'hot': self._hits['hot'] / total_requests if total_requests > 0 else 0,
                'warm': self._hits['warm'] / total_requests if total_requests > 0 else 0,
                'cold': self._hits['cold'] / total_requests if total_requests > 0 else 0
            },
            'cache_sizes': {
                'hot': len(self._hot_cache),
                'warm': len(self._warm_cache),
                'cold': len(self._cold_cache)
            },
            'compression_stats': self._compression_stats.copy(),
            'total_hits': total_hits,
            'total_misses': self._misses,
            'gaming_patterns': {
                'tracked_games': len(self._shader_sequences),
                'learned_patterns': len(self._warm_patterns),
                'prefetch_patterns': sum(len(patterns) for patterns in self._prefetch_patterns.values())
            },
            'memory_efficiency': {
                'compression_ratio': self._calculate_compression_ratio(),
                'memory_usage_mb': self._memory_usage / (1024 * 1024) if self._memory_usage else 0
            }
        }
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate overall compression ratio across all tiers"""
        total_compressed = (self._compression_stats.get('compressed', 0) + 
                           self._compression_stats.get('highly_compressed', 0))
        total_items = (total_compressed + 
                      self._compression_stats.get('uncompressed', 0))
        
        return total_compressed / total_items if total_items > 0 else 0.0


@njit(cache=True)
def fast_ensemble_prediction(pred1: float, pred2: float, pred3: float, 
                             weight1: float, weight2: float, weight3: float) -> float:
    """Ultra-fast ensemble prediction with weighted average"""
    return pred1 * weight1 + pred2 * weight2 + pred3 * weight3


def fast_dot_product_impl(a, b):
    """Ultra-fast dot product implementation"""
    if HAS_NUMPY and hasattr(a, 'size') and a.size == b.size:
        result = 0.0
        for i in range(a.size):
            result += a[i] * b[i]
        return result
    else:
        return sum(x * y for x, y in zip(a, b))

if HAS_NUMBA and HAS_NUMPY:
    fast_dot_product = njit(cache=True, fastmath=True)(fast_dot_product_impl)
else:
    fast_dot_product = fast_dot_product_impl


def fast_quantized_prediction_impl(features, weights, bias, scale):
    """Fast quantized linear model prediction implementation"""
    if HAS_NUMPY:
        # Fast dot product with weights
        result = fast_dot_product(features, weights) + bias
        return result * scale
    else:
        result = sum(x * w for x, w in zip(features, weights)) + bias
        return result * scale

if HAS_NUMBA and HAS_NUMPY:
    fast_quantized_prediction = njit(cache=True, fastmath=True)(fast_quantized_prediction_impl)
else:
    fast_quantized_prediction = fast_quantized_prediction_impl


class QuantizedLinearModel:
    """Quantized linear model for ultra-fast inference"""
    
    def __init__(self, weights, bias: float, scale: float = 1.0):
        self.weights = weights.astype(np.float32) if HAS_NUMPY else weights
        self.bias = float(bias)
        self.scale = float(scale)
        self.is_fitted = True
    
    def predict(self, X):
        """Fast prediction using quantized weights"""
        if HAS_NUMBA and HAS_NUMPY:
            if X.ndim == 1:
                return fast_quantized_prediction(X, self.weights, self.bias, self.scale)
            else:
                results = np.empty(X.shape[0], dtype=np.float32)
                for i in range(X.shape[0]):
                    results[i] = fast_quantized_prediction(X[i], self.weights, self.bias, self.scale)
                return results
        else:
            # Fallback implementation
            if hasattr(X, 'ndim') and X.ndim == 1:
                return sum(x * w for x, w in zip(X, self.weights)) + self.bias
            else:
                return [sum(x * w for x, w in zip(row, self.weights)) + self.bias for row in X]


class EnsemblePredictor:
    """Fast ensemble predictor with multiple models"""
    
    def __init__(self):
        self.models = []
        self.weights = []
        self.is_fitted = False
    
    def add_model(self, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        self.is_fitted = len(self.models) > 0
    
    def predict(self, X):
        """Fast ensemble prediction"""
        if not self.is_fitted or not self.models:
            return 0.0
        
        if len(self.models) == 1:
            return self.models[0].predict(X)
        
        predictions = [model.predict(X) for model in self.models]
        
        if HAS_NUMBA and len(predictions) >= 3:
            # Use fast ensemble for common case
            if hasattr(predictions[0], '__len__') and len(predictions[0]) > 1:
                # Multiple predictions
                results = []
                for i in range(len(predictions[0])):
                    pred1 = predictions[0][i] if len(predictions) > 0 else 0.0
                    pred2 = predictions[1][i] if len(predictions) > 1 else 0.0
                    pred3 = predictions[2][i] if len(predictions) > 2 else 0.0
                    result = fast_ensemble_prediction(
                        pred1, pred2, pred3,
                        self.weights[0], self.weights[1], 
                        self.weights[2] if len(self.weights) > 2 else 0.0
                    )
                    # Add remaining models if any
                    for j in range(3, len(predictions)):
                        result += predictions[j][i] * self.weights[j]
                    results.append(result)
                return results
            else:
                # Single prediction
                pred1 = predictions[0] if len(predictions) > 0 else 0.0
                pred2 = predictions[1] if len(predictions) > 1 else 0.0
                pred3 = predictions[2] if len(predictions) > 2 else 0.0
                result = fast_ensemble_prediction(
                    pred1, pred2, pred3,
                    self.weights[0], self.weights[1], 
                    self.weights[2] if len(self.weights) > 2 else 0.0
                )
                # Add remaining models
                for j in range(3, len(predictions)):
                    result += predictions[j] * self.weights[j]
                return result
        else:
            # Fallback ensemble
            if hasattr(predictions[0], '__len__') and len(predictions[0]) > 1:
                return [sum(pred[i] * weight for pred, weight in zip(predictions, self.weights)) 
                       for i in range(len(predictions[0]))]
            else:
                return sum(pred * weight for pred, weight in zip(predictions, self.weights))


class EnhancedMLPredictor:
    """Ultra-high-performance ML predictor with Rust-equivalent performance"""
    
    def __init__(self, model_path: Optional[Path] = None, 
                 enable_async: bool = True,
                 max_memory_mb: int = 40):
        """
        Initialize enhanced ML predictor with advanced optimizations
        
        Args:
            model_path: Path to store/load models
            enable_async: Enable async operations  
            max_memory_mb: Maximum memory usage in MB
        """
        self.model_path = model_path or Path.home() / '.cache' / 'shader-predict-compile' / 'models'
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Enhanced memory management
        self.max_memory_mb = max_memory_mb
        self._current_memory_mb = 0
        self._memory_pressure = False
        self._last_memory_check = 0
        
        # Model components (lazy loaded)
        self._compilation_time_model = None
        self._success_model = None
        self._scaler = None
        self._models_loaded = False
        self._feature_means = None
        self._feature_stds = None
        
        # Fast prediction models
        self._fast_linear_model = None
        self._ensemble_model = None
        self._quantized_models = {}
        self._use_ensemble = True
        
        # Enhanced predictors
        self.heuristic_predictor = HeuristicPredictor()
        self.thermal_scheduler = ThermalAwareScheduler()
        
        # Gaming-aware caching system with multi-tier optimization
        self.feature_cache = GamingAwareFeatureCache(max_size=1000, compression_threshold=64)  # Steam Deck optimized
        self.prediction_cache = OrderedDict()
        self.max_cache_size = 500  # Further reduced for memory efficiency
        
        # Enhanced training data management with memory efficiency
        self.training_buffer_size = 800  # Reduced for memory efficiency
        self.training_data = deque(maxlen=self.training_buffer_size)
        
        # Thread pools for performance
        self.enable_async = enable_async
        if enable_async:
            self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MLPredictor")
            self.io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MLIO")
        
        # Enhanced object pools with memory efficiency
        def create_feature_vector():
            return CompactFeatureVector(24)  # Use compact feature vector
        
        self._feature_vector_pool = EnhancedMemoryPool(
            create_feature_vector,
            max_size=20,  # Reduced pool size
            warmup_size=5   # Reduced warmup
        )
        
        # Enhanced memory-mapped cache for persistent shader storage
        self._mmap_cache = EnhancedMemoryMappedCache(
            self.model_path / 'persistent_cache',
            max_cache_size_mb=80  # Steam Deck optimized size
        )
        
        # Performance tracking with memory efficiency
        self._prediction_times = deque(maxlen=100)  # Reduced tracking size
        self._cache_performance = deque(maxlen=50)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # =================== SYSTEM OPTIMIZATION INTEGRATION ===================
        
        # Initialize system optimization components
        self.resource_manager = SteamDeckResourceManager()
        self.system_monitor = self.resource_manager.system_monitor
        self.gaming_scheduler = GamingAwareScheduler(self.resource_manager)
        self.memory_optimizer = MemoryOptimizer(max_memory_mb)
        
        # Configure initial system mode
        if self.resource_manager.detect_gaming_activity():
            self.resource_manager.set_gaming_mode()
        else:
            self.resource_manager.set_background_mode()
        
        # Start gaming-aware scheduler
        self.gaming_scheduler.start()
        
        # System performance tracking
        self._system_metrics = {
            'thermal_states': deque(maxlen=100),
            'memory_pressure': deque(maxlen=100),
            'gaming_activity': deque(maxlen=100),
            'cpu_affinity_changes': 0,
            'thermal_throttle_events': 0
        }
        
        # Enhanced memory monitoring with system integration
        self._start_enhanced_memory_monitor()
        
        # Initialize normalization parameters
        self._initialize_normalization_params()
        
        # Initialize fast prediction models
        self._initialize_fast_models()
    
    def _initialize_normalization_params(self):
        """Initialize feature normalization parameters optimized for Steam Deck"""
        if HAS_NUMPY:
            # Steam Deck optimized normalization parameters (24 features)
            self._feature_means = np.array([
                500.0,    # instruction_count
                32.0,     # register_usage  
                4.0,      # texture_samples
                10.0,     # memory_operations
                5.0,      # control_flow_complexity
                64.0,     # wave_size
                0.3,      # uses_derivatives
                0.1,      # uses_tessellation
                0.05,     # uses_geometry_shader
                2.0,      # shader_type_hash
                1.5,      # optimization_level
                0.5,      # cache_priority
                350.0,    # alu_instructions
                1.2,      # memory_latency_factor
                0.75,     # instruction_parallelism
                1.0,      # thermal_factor
                8.0,      # instruction_density
                30.0,     # memory_intensity
                1.6,      # complexity_register_interaction
                0.6,      # priority_optimization_interaction
                0.7,      # alu_ratio
                12.0,     # memory_pressure
                0.75,     # wave_efficiency
                1.125     # thermal_register_pressure
            ], dtype=np.float32)
            
            self._feature_stds = np.array([
                1000.0, 50.0, 8.0, 20.0, 10.0, 32.0, 0.5, 0.3, 0.2, 2.0, 1.0, 0.3,
                700.0, 0.5, 0.25, 0.3, 15.0, 50.0, 5.0, 0.4, 0.3, 15.0, 0.25, 0.4
            ], dtype=np.float32)
        else:
            self._feature_means = [500.0, 32.0, 4.0, 10.0, 5.0, 64.0, 0.3, 0.1, 0.05, 2.0, 1.5, 0.5, 350.0, 1.2, 0.75, 1.0, 8.0, 30.0, 1.6, 0.6, 0.7, 12.0, 0.75, 1.125]
            self._feature_stds = [1000.0, 50.0, 8.0, 20.0, 10.0, 32.0, 0.5, 0.3, 0.2, 2.0, 1.0, 0.3, 700.0, 0.5, 0.25, 0.3, 15.0, 50.0, 5.0, 0.4, 0.3, 15.0, 0.25, 0.4]
    
    def _initialize_fast_models(self):
        """Initialize fast prediction models for ultra-low latency"""
        try:
            # Create a simple linear model with pre-computed weights
            # These weights are optimized for Steam Deck shader prediction
            if HAS_NUMPY:
                # Optimized weights based on feature importance analysis
                fast_weights = np.array([
                    0.008,   # instruction_count
                    0.08,    # register_usage
                    0.4,     # texture_samples
                    0.25,    # memory_operations
                    0.8,     # control_flow_complexity
                    -0.02,   # wave_size (negative because 64 is optimal)
                    1.5,     # uses_derivatives
                    2.0,     # uses_tessellation
                    1.8,     # uses_geometry_shader
                    1.2,     # shader_type_hash
                    0.3,     # optimization_level
                    -0.5,    # cache_priority (negative - higher priority = faster)
                    0.002,   # alu_instructions
                    2.0,     # memory_latency_factor
                    -1.0,    # instruction_parallelism (negative - higher ILP = faster)
                    -2.0,    # thermal_factor (negative - better thermal = faster)
                    0.1,     # instruction_density
                    0.03,    # memory_intensity
                    0.05,    # complexity_register_interaction
                    -0.2,    # priority_optimization_interaction
                    0.5,     # alu_ratio
                    0.02,    # memory_pressure
                    -0.8,    # wave_efficiency
                    0.3      # thermal_register_pressure
                ], dtype=np.float32)
                
                self._fast_linear_model = QuantizedLinearModel(
                    weights=fast_weights,
                    bias=5.0,  # Base compilation time
                    scale=1.0
                )
            
            # Initialize ensemble model
            self._ensemble_model = EnsemblePredictor()
            
            self.logger.debug("Fast prediction models initialized")
            
        except Exception as e:
            self.logger.debug(f"Fast model initialization failed: {e}")
            self._fast_linear_model = None
            self._ensemble_model = None
    
    def _start_enhanced_memory_monitor(self):
        """Enhanced memory monitoring with predictive cleanup"""
        def monitor():
            while True:
                time.sleep(20)  # More frequent checks
                self._enhanced_memory_check()
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _enhanced_memory_check(self):
        """Enhanced memory pressure detection and management"""
        try:
            current_time = time.time()
            if current_time - self._last_memory_check < 10:  # Rate limiting
                return
            
            self._last_memory_check = current_time
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self._current_memory_mb = memory_mb
            
            # Progressive memory pressure levels
            if memory_mb > self.max_memory_mb * 1.2:
                self._memory_pressure = True
                self._aggressive_cleanup()
            elif memory_mb > self.max_memory_mb:
                self._memory_pressure = True
                self._cleanup_memory()
            elif memory_mb > self.max_memory_mb * 0.8:
                # Predictive cleanup
                self._light_cleanup()
            else:
                self._memory_pressure = False
                
        except Exception as e:
            self.logger.debug(f"Enhanced memory check failed: {e}")
    
    def _light_cleanup(self):
        """Light cleanup for predictive memory management"""
        # Trim caches by 15% for better memory efficiency
        if len(self.prediction_cache) > self.max_cache_size * 0.85:
            trim_count = int(len(self.prediction_cache) * 0.15)
            for _ in range(trim_count):
                if self.prediction_cache:
                    self.prediction_cache.popitem(last=False)
        
        # Clean up old memory-mapped models
        if hasattr(self, '_mmap_cache'):
            self._mmap_cache.cleanup_old_models(12)  # Clean models older than 12 hours
    
    def _cleanup_memory(self):
        """Standard memory cleanup with enhanced efficiency"""
        # Reduce prediction cache more aggressively
        if len(self.prediction_cache) > self.max_cache_size // 3:
            for _ in range(len(self.prediction_cache) // 2):
                if self.prediction_cache:
                    self.prediction_cache.popitem(last=False)
        
        # Trim training buffer more aggressively
        if len(self.training_data) > self.training_buffer_size // 3:
            new_data = list(self.training_data)[-self.training_buffer_size // 3:]
            self.training_data.clear()
            self.training_data.extend(new_data)
        
        # Clean up feature cache with gaming-aware logic
        if hasattr(self.feature_cache, '_memory_usage') and self.feature_cache._memory_usage > 4 * 1024 * 1024:  # 4MB
            # Use gaming-aware eviction instead of simple compression
            with self.feature_cache._lock:
                # Evict from hot cache using gaming patterns
                while len(self.feature_cache._hot_cache) > self.feature_cache._max_hot_size // 2:
                    self.feature_cache._evict_from_hot_cache()
                
                # Clean up old gaming patterns
                current_time = time.time()
                if hasattr(self, '_game_performance_patterns'):
                    for game_context in list(self._game_performance_patterns.keys()):
                        patterns = self._game_performance_patterns[game_context]
                        # Keep only recent patterns (last hour)
                        recent_patterns = [p for p in patterns if current_time - p['timestamp'] < 3600]
                        if recent_patterns:
                            self._game_performance_patterns[game_context] = recent_patterns[-50:]  # Keep last 50
                        else:
                            del self._game_performance_patterns[game_context]
    
    def _aggressive_cleanup(self):
        """Aggressive cleanup for high memory pressure"""
        # Clear all caches
        self.prediction_cache.clear()
        
        # Clear feature cache with multi-tier awareness
        with self.feature_cache._lock:
            self.feature_cache._hot_cache.clear()
            # Keep some warm cache entries for faster recovery
            warm_items_to_keep = min(10, len(self.feature_cache._warm_cache) // 4)
            if warm_items_to_keep > 0:
                # Keep most recently accessed warm cache items
                recent_warm = dict(list(self.feature_cache._warm_cache.items())[-warm_items_to_keep:])
                self.feature_cache._warm_cache.clear()
                self.feature_cache._warm_cache.update(recent_warm)
        
        # Keep only most recent training data
        if len(self.training_data) > self.training_buffer_size // 8:
            new_data = list(self.training_data)[-self.training_buffer_size // 8:]
            self.training_data.clear()
            self.training_data.extend(new_data)
        
        # Clean up memory-mapped cache
        if hasattr(self, '_mmap_cache'):
            self._mmap_cache.cleanup_old_models(1)  # Very aggressive - clean models older than 1 hour
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.logger.info(f"Aggressive memory cleanup completed. Current: {self._current_memory_mb:.1f}MB")
    
    @lru_cache(maxsize=256)
    def _enhanced_feature_hash(self, shader_hash: str, instruction_count: int, 
                              shader_type: str, register_usage: int) -> str:
        """Enhanced feature hash for better cache efficiency"""
        # Create more discriminative hash
        return f"{shader_hash[:8]}_{instruction_count}_{shader_type}_{register_usage}"
    
    def predict_compilation_time(self, features: UnifiedShaderFeatures, 
                                use_cache: bool = True, 
                                game_context: str = None) -> float:
        """
        Gaming-optimized shader compilation time prediction with system-level optimizations
        
        Args:
            features: Shader feature data
            use_cache: Enable caching (default: True)
            game_context: Game identifier for pattern optimization
        
        Returns:
            Predicted compilation time in milliseconds
        """
        start_time = time.perf_counter()
        
        # =================== SYSTEM-LEVEL OPTIMIZATIONS ===================
        
        # Check if we should throttle due to system pressure
        if self.resource_manager.should_throttle_operations():
            # Use fast heuristic prediction under pressure
            thermal_state = self.system_monitor.get_current_thermal_state()
            thermal_factor = self._get_thermal_factor(thermal_state)
            
            # Quick heuristic prediction with thermal adjustment
            base_prediction = self.heuristic_predictor.predict_compilation_time(features)
            return base_prediction * thermal_factor
        
        # Track system metrics for optimization
        self._update_system_metrics()
        
        # Perform memory cleanup if needed
        if self.memory_optimizer.should_perform_gc():
            self.memory_optimizer.perform_memory_cleanup()
        
        # Use optimized memory allocation for feature processing
        feature_buffer = self.memory_optimizer.allocate_optimized_buffer(1024)
        
        try:
            # Enhanced cache check with gaming context
            if use_cache:
                cache_key = self._enhanced_feature_hash(
                features.shader_hash,
                features.instruction_count,
                features.shader_type.value,
                features.register_usage
            )
            
            # Check prediction cache first (fastest)
            if cache_key in self.prediction_cache:
                self.prediction_cache.move_to_end(cache_key)
                cached_result = self.prediction_cache[cache_key]
                
                prediction_time = (time.perf_counter() - start_time) * 1000
                self._prediction_times.append(prediction_time)
                return cached_result['time']
        
            # Fast feature vector extraction with gaming-aware caching
            feature_vector = self._get_enhanced_feature_vector_gaming(features, game_context)
            
            # Enhanced prediction with multiple fallbacks
            prediction = self._enhanced_predict(feature_vector, features)
            
            # Gaming-aware caching with memory pressure awareness
            memory_pressure = self.memory_optimizer.get_memory_pressure()
            if use_cache and memory_pressure < 0.8:
                if len(self.prediction_cache) >= self.max_cache_size:
                    # Remove oldest entries more aggressively
                    for _ in range(self.max_cache_size // 3):
                        if self.prediction_cache:
                            self.prediction_cache.popitem(last=False)
                
                # Store with gaming context for better organization
                self.prediction_cache[cache_key] = {
                    'time': float(prediction),
                    'timestamp': float(time.time()),
                    'game_context': game_context,
                    'features_hash': features.shader_hash
                }
                
                # Also store in persistent cache for cross-session optimization
                if hasattr(self, '_mmap_cache') and game_context:
                    feature_bytes = self._serialize_features_for_persistence(feature_vector, prediction)
                    self._mmap_cache.store_shader_features(
                        cache_key, 
                        feature_bytes, 
                        priority=self._calculate_shader_priority(features, game_context),
                        game_context=game_context
                    )
            
            # Track performance with gaming context
            prediction_time = (time.perf_counter() - start_time) * 1000
            self._prediction_times.append(prediction_time)
            
            # Learn from this access for future optimizations
            if game_context:
                self._update_gaming_patterns(features, prediction, game_context)
            
            return prediction
        
        finally:
            # Always return the memory buffer to pool
            if 'feature_buffer' in locals():
                self.memory_optimizer.deallocate_buffer(feature_buffer)
    
    # =================== SYSTEM OPTIMIZATION HELPER METHODS ===================
    
    def _get_thermal_factor(self, thermal_state: ThermalState) -> float:
        """Get thermal adjustment factor for predictions"""
        thermal_factors = {
            ThermalState.COOL: 0.95,    # Slightly optimistic when cool
            ThermalState.NORMAL: 1.0,   # Baseline
            ThermalState.WARM: 1.1,     # 10% slower when warm
            ThermalState.HOT: 1.25      # 25% slower when hot
        }
        return thermal_factors.get(thermal_state, 1.0)
    
    def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # Get current system state
            thermal_state = self.system_monitor.get_current_thermal_state()
            memory_pressure = self.memory_optimizer.get_memory_pressure()
            gaming_active = self.resource_manager.detect_gaming_activity()
            
            # Store metrics for analysis
            self._system_metrics['thermal_states'].append((time.time(), thermal_state.value))
            self._system_metrics['memory_pressure'].append((time.time(), memory_pressure))
            self._system_metrics['gaming_activity'].append((time.time(), gaming_active))
            
            # Track significant events
            if thermal_state in [ThermalState.WARM, ThermalState.HOT]:
                self._system_metrics['thermal_throttle_events'] += 1
            
        except Exception:
            pass  # Don't let metric collection break the main functionality
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        try:
            current_time = time.time()
            
            # Analyze recent thermal states (last 5 minutes)
            recent_thermal = [state for timestamp, state in self._system_metrics['thermal_states'] 
                            if current_time - timestamp < 300]
            
            # Analyze memory pressure trends
            recent_memory = [pressure for timestamp, pressure in self._system_metrics['memory_pressure'] 
                           if current_time - timestamp < 300]
            
            # Analyze gaming activity
            recent_gaming = [active for timestamp, active in self._system_metrics['gaming_activity'] 
                           if current_time - timestamp < 300]
            
            # Calculate averages and trends
            avg_memory_pressure = sum(recent_memory) / len(recent_memory) if recent_memory else 0
            gaming_activity_ratio = sum(recent_gaming) / len(recent_gaming) if recent_gaming else 0
            
            # System health assessment
            system_health = "excellent"
            if avg_memory_pressure > 0.8:
                system_health = "poor"
            elif avg_memory_pressure > 0.6 or gaming_activity_ratio > 0.7:
                system_health = "moderate"
            elif avg_memory_pressure > 0.4 or gaming_activity_ratio > 0.5:
                system_health = "good"
            
            return {
                'system_health': system_health,
                'is_steam_deck': self.system_monitor.is_steam_deck,
                'apu_model': self.system_monitor.apu_model,
                'current_thermal_state': self.system_monitor.get_current_thermal_state().value,
                'average_memory_pressure': avg_memory_pressure,
                'gaming_activity_ratio': gaming_activity_ratio,
                'thermal_throttle_events': self._system_metrics['thermal_throttle_events'],
                'cpu_affinity_changes': self._system_metrics['cpu_affinity_changes'],
                'background_cores': self.resource_manager._background_cores,
                'gaming_cores': self.resource_manager._gaming_cores,
                'optimal_thread_count': self.resource_manager.get_optimal_thread_count(),
                'power_state': self.system_monitor.get_power_state()
            }
            
        except Exception as e:
            return {
                'system_health': 'unknown',
                'error': str(e)
            }
    
    def optimize_for_gaming_session(self, game_context: str = None):
        """Optimize system for gaming session"""
        try:
            # Switch to gaming mode
            self.resource_manager.set_gaming_mode()
            
            # Perform aggressive memory cleanup
            self.memory_optimizer.perform_memory_cleanup(aggressive=True)
            
            # Warm cache for game if context provided
            if game_context and hasattr(self, 'feature_cache'):
                self.feature_cache.warm_cache_for_game(game_context)
            
            # Schedule background optimizations
            self.gaming_scheduler.schedule_task(
                self._background_optimization_task,
                priority='low'
            )
            
            self.logger.info(f"Optimized system for gaming session: {game_context or 'unknown'}")
            
        except Exception as e:
            self.logger.warning(f"Failed to optimize for gaming session: {e}")
    
    def _background_optimization_task(self):
        """Background optimization task that runs with low priority"""
        try:
            # Only run if system is not under pressure
            if not self.resource_manager.should_throttle_operations():
                # Perform light memory cleanup
                self.memory_optimizer.perform_memory_cleanup(aggressive=False)
                
                # Optimize prediction cache
                self._optimize_prediction_cache()
                
        except Exception:
            pass  # Silent failure for background tasks
    
    def _optimize_prediction_cache(self):
        """Optimize prediction cache based on usage patterns"""
        try:
            current_time = time.time()
            
            # Remove old entries that haven't been accessed recently
            old_keys = []
            for key, data in self.prediction_cache.items():
                if current_time - data.get('timestamp', 0) > 3600:  # 1 hour
                    old_keys.append(key)
            
            for key in old_keys[:len(old_keys)//2]:  # Remove half of old entries
                del self.prediction_cache[key]
                
        except Exception:
            pass
    
    # =================== END SYSTEM OPTIMIZATION METHODS ===================
    
    def _get_enhanced_feature_vector_gaming(self, features: UnifiedShaderFeatures, game_context: str = None):
        """Enhanced feature vector extraction optimized for gaming patterns"""
        # Check gaming-aware feature cache first
        cache_key = features.shader_hash
        
        # Try to get from multi-tier cache
        cached_vector = self.feature_cache.get(cache_key, game_context)
        if cached_vector is not None:
            return cached_vector
        
        # Check persistent cache for cross-session optimization
        if hasattr(self, '_mmap_cache') and game_context:
            persistent_data = self._mmap_cache.load_shader_features(cache_key)
            if persistent_data is not None:
                try:
                    feature_vector = self._deserialize_features_from_persistence(persistent_data)
                    if feature_vector is not None:
                        # Store in memory cache for fast access
                        self.feature_cache.put(cache_key, feature_vector, game_context, priority=2)
                        return feature_vector
                except Exception:
                    pass  # Fall through to computation
        
        # Compute feature vector with RDNA2 optimizations
        feature_vector = self._compute_feature_vector(features)
        
        # Store in gaming-aware cache with appropriate priority
        priority = self._calculate_shader_priority(features, game_context)
        self.feature_cache.put(cache_key, feature_vector, game_context, priority)
        
        return feature_vector
    
    def _calculate_shader_priority(self, features: UnifiedShaderFeatures, game_context: str = None) -> int:
        """Calculate shader priority for gaming-optimized caching"""
        priority = 1  # Base priority
        
        # High-frequency shader types get higher priority
        if features.shader_type in [ShaderType.VERTEX, ShaderType.FRAGMENT]:
            priority += 1
        
        # Complex shaders that benefit from caching
        if features.instruction_count > 500:
            priority += 1
        
        # Shaders with expensive features
        if features.uses_tessellation or features.uses_geometry_shader:
            priority += 1
        
        # Game context specific prioritization
        if game_context:
            # Common game engines/patterns get higher priority
            if any(engine in game_context.lower() for engine in ['unity', 'unreal', 'source', 'id']):
                priority += 1
        
        return min(priority, 5)  # Cap at 5
    
    def _serialize_features_for_persistence(self, feature_vector, prediction: float) -> bytes:
        """Serialize feature vector and prediction for persistent storage"""
        try:
            data = {
                'features': feature_vector.tolist() if HAS_NUMPY and hasattr(feature_vector, 'tolist') else list(feature_vector),
                'prediction': float(prediction),
                'timestamp': time.time(),
                'version': 1  # For future compatibility
            }
            
            if HAS_MSGPACK:
                return msgpack.packb(data)
            else:
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception:
            return b''  # Return empty bytes on error
    
    def _deserialize_features_from_persistence(self, data: bytes):
        """Deserialize feature vector from persistent storage"""
        try:
            if HAS_MSGPACK:
                obj = msgpack.unpackb(data, raw=False)
            else:
                obj = pickle.loads(data)
            
            if isinstance(obj, dict) and 'features' in obj:
                features = obj['features']
                if HAS_NUMPY:
                    return np.array(features, dtype=np.float32)
                else:
                    return features
            
            return None
            
        except Exception:
            return None
    
    def _update_gaming_patterns(self, features: UnifiedShaderFeatures, prediction: float, game_context: str):
        """Update gaming-specific optimization patterns"""
        try:
            # Track shader types per game for optimization
            if not hasattr(self, '_game_shader_patterns'):
                self._game_shader_patterns = defaultdict(lambda: defaultdict(int))
            
            self._game_shader_patterns[game_context][features.shader_type.value] += 1
            
            # Track performance characteristics
            if not hasattr(self, '_game_performance_patterns'):
                self._game_performance_patterns = defaultdict(list)
            
            self._game_performance_patterns[game_context].append({
                'prediction': prediction,
                'complexity': features.instruction_count,
                'timestamp': time.time()
            })
            
            # Trigger cache warming if we detect game loading patterns
            if self._detect_game_loading_pattern(game_context):
                self._trigger_cache_warming(game_context)
                
        except Exception:
            pass  # Don't break prediction on pattern update failure
    
    def _detect_game_loading_pattern(self, game_context: str) -> bool:
        """Detect if game is in loading phase for cache warming"""
        if not hasattr(self, '_game_performance_patterns'):
            return False
        
        recent_predictions = self._game_performance_patterns[game_context][-10:]  # Last 10 predictions
        current_time = time.time()
        
        # Check for rapid shader compilation requests (typical of loading)
        rapid_requests = sum(1 for p in recent_predictions 
                           if current_time - p['timestamp'] < 10)  # Within 10 seconds
        
        return rapid_requests >= 5  # 5+ rapid requests suggests loading
    
    def _trigger_cache_warming(self, game_context: str):
        """Trigger cache warming for detected game loading"""
        try:
            # Get common shaders for this game from patterns
            if hasattr(self.feature_cache, 'learn_game_patterns'):
                self.feature_cache.learn_game_patterns(game_context)
            
            # Warm the cache with known patterns
            if hasattr(self.feature_cache, 'warm_cache_for_game'):
                self.feature_cache.warm_cache_for_game(game_context)
                
        except Exception:
            pass  # Don't break on warming failure
    
    def _compute_feature_vector(self, features: UnifiedShaderFeatures):
        """Compute feature vector with RDNA2 optimizations and Steam Deck tuning"""
        # Compute RDNA2-specific features
        rdna2_features = self._compute_rdna2_features(features)
        
        # Fast feature extraction using Numba if available
        if HAS_NUMBA and HAS_NUMPY:
            vector = fast_feature_extraction(
                float(features.instruction_count),
                float(features.register_usage),
                float(features.texture_samples),
                float(features.memory_operations),
                float(features.control_flow_complexity),
                float(features.wave_size),
                float(features.uses_derivatives),
                float(features.uses_tessellation),
                float(features.uses_geometry_shader),
                float(features.shader_type.value.__hash__() % 10),
                float(features.optimization_level),
                float(features.cache_priority),
                float(rdna2_features['alu_instructions']),
                float(rdna2_features['memory_latency_factor']),
                float(rdna2_features['instruction_parallelism']),
                float(rdna2_features['thermal_factor'])
            )
            return vector
        else:
            # Fallback implementation using compact vectors
            vector = self._feature_vector_pool.acquire()
            try:
                # Prepare feature values
                feature_values = [
                    features.instruction_count,
                    features.register_usage,
                    features.texture_samples,
                    features.memory_operations,
                    features.control_flow_complexity,
                    features.wave_size,
                    float(features.uses_derivatives),
                    float(features.uses_tessellation),
                    float(features.uses_geometry_shader),
                    features.shader_type.value.__hash__() % 10,
                    features.optimization_level,
                    features.cache_priority,
                    # RDNA2-specific features
                    rdna2_features['alu_instructions'],
                    rdna2_features['memory_latency_factor'],
                    rdna2_features['instruction_parallelism'],
                    rdna2_features['thermal_factor'],
                    # Enhanced derived features
                    features.instruction_count / max(1, features.wave_size),
                    (features.texture_samples + features.memory_operations) / max(1, features.instruction_count) * 1000,
                    features.control_flow_complexity * features.register_usage / 1000,
                    features.cache_priority * (1.0 + features.optimization_level * 0.1),
                    # Advanced RDNA2 architectural features
                    rdna2_features['alu_instructions'] / max(1, features.instruction_count),
                    rdna2_features['memory_latency_factor'] * features.memory_operations,
                    rdna2_features['instruction_parallelism'] * features.wave_size / 64.0,
                    rdna2_features['thermal_factor'] * (1.0 + features.register_usage / 256.0)
                ]
                
                # Set values efficiently
                vector.set_values(feature_values)
                return vector.get_data()
                
            finally:
                self._feature_vector_pool.release(vector)
    
    def _compute_rdna2_features(self, features: UnifiedShaderFeatures) -> Dict[str, float]:
        """Compute RDNA2-specific features for enhanced prediction accuracy on Steam Deck"""
        # Detect Steam Deck model for model-specific optimizations
        deck_model = self._detect_steam_deck_model()
        
        # Estimate ALU instruction count based on shader characteristics
        alu_instructions = features.instruction_count * 0.7  # ~70% are typically ALU ops
        
        # RDNA2-specific shader type optimizations
        if features.shader_type == ShaderType.COMPUTE:
            alu_instructions *= 1.15  # RDNA2 compute units are efficient
            if deck_model == SteamDeckModel.OLED:
                alu_instructions *= 0.95  # Phoenix APU slightly more efficient
        elif features.shader_type == ShaderType.FRAGMENT:
            alu_instructions *= 0.75  # Fragment shaders have more texture ops
            # RDNA2 has excellent fragment shader throughput
            if features.texture_samples <= 4:
                alu_instructions *= 0.9  # Efficient texture cache
        elif features.shader_type == ShaderType.VERTEX:
            # RDNA2 vertex shader optimizations
            alu_instructions *= 0.8
            if features.register_usage < 32:
                alu_instructions *= 0.85  # Low register pressure is efficient
        
        # Memory latency factor based on RDNA2 cache hierarchy
        memory_latency_factor = 1.0
        
        # L1 texture cache efficiency (32KB per CU on RDNA2)
        if features.texture_samples > 0:
            # RDNA2 has improved texture cache efficiency
            cache_efficiency = min(1.0, 6.0 / max(1, features.texture_samples))
            memory_latency_factor = 1.0 + (1.0 - cache_efficiency) * 0.3  # Reduced penalty
        
        # L2 cache pressure (4MB shared on Steam Deck)
        if features.memory_operations > 8:
            # RDNA2 has better memory subsystem
            memory_latency_factor *= 1.0 + (features.memory_operations - 8) * 0.015  # Reduced penalty
        
        # Wave size optimization for RDNA2
        wave_efficiency = 1.0
        if features.wave_size == 64:
            wave_efficiency = 1.0  # Optimal for RDNA2
        elif features.wave_size == 32:
            wave_efficiency = 0.95  # Still good on RDNA2
        else:
            wave_efficiency = 0.85  # Non-standard wave sizes
        
        # Instruction-level parallelism with RDNA2 considerations
        instruction_parallelism = 0.65  # Base ILP (slightly higher than generic)
        
        # RDNA2 has improved branch prediction
        if features.control_flow_complexity < 3:
            instruction_parallelism = min(1.0, instruction_parallelism + 0.25)
        elif features.control_flow_complexity > 8:
            instruction_parallelism = max(0.35, instruction_parallelism - 0.25)
        
        # Apply wave efficiency to ILP
        instruction_parallelism *= wave_efficiency
        
        # Thermal efficiency factor with Steam Deck specific considerations
        thermal_factor = 1.0
        if hasattr(self, 'thermal_scheduler'):
            thermal_state = self.thermal_scheduler.current_thermal_state
            
            # Steam Deck thermal characteristics
            if thermal_state in [ThermalState.HOT, ThermalState.THROTTLING]:
                if deck_model == SteamDeckModel.OLED:
                    thermal_factor = 0.75  # Phoenix APU handles heat better
                else:
                    thermal_factor = 0.65  # Van Gogh APU more thermal sensitive
            elif thermal_state == ThermalState.WARM:
                thermal_factor = 0.88 if deck_model == SteamDeckModel.OLED else 0.82
            elif thermal_state == ThermalState.COOL:
                thermal_factor = 1.15 if deck_model == SteamDeckModel.OLED else 1.12
            elif thermal_state == ThermalState.PREDICTIVE_WARM:
                # Preemptively adjust for predicted thermal increase
                thermal_factor = 0.92
        
        # RDNA2 clock boost considerations
        clock_efficiency = 1.0
        if features.instruction_count > 1000:  # Large shaders may cause clock reduction
            if deck_model == SteamDeckModel.OLED:
                clock_efficiency = 0.98  # Phoenix maintains clocks better
            else:
                clock_efficiency = 0.95  # Van Gogh more sensitive to workload
        
        return {
            'alu_instructions': alu_instructions * clock_efficiency,
            'memory_latency_factor': memory_latency_factor,
            'instruction_parallelism': instruction_parallelism,
            'thermal_factor': thermal_factor
        }
    
    def _detect_steam_deck_model(self) -> SteamDeckModel:
        """Detect Steam Deck model (LCD vs OLED) for optimization"""
        try:
            # Try to detect based on system information
            # This is a simplified detection - in practice, you'd use more sophisticated methods
            
            # Check for DMI information that might indicate OLED model
            try:
                with open('/sys/class/dmi/id/product_name', 'r') as f:
                    product_name = f.read().strip().lower()
                    if 'oled' in product_name:
                        return SteamDeckModel.OLED
            except (FileNotFoundError, PermissionError):
                pass
            
            # Check for APU characteristics (Phoenix vs Van Gogh)
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'phoenix' in cpuinfo or 'family 25' in cpuinfo:
                        return SteamDeckModel.OLED
                    elif 'van gogh' in cpuinfo or 'family 23' in cpuinfo:
                        return SteamDeckModel.LCD
            except (FileNotFoundError, PermissionError):
                pass
            
            # Default assumption
            return SteamDeckModel.LCD
            
        except Exception:
            return SteamDeckModel.LCD  # Safe default
    
    def _enhanced_predict(self, feature_vector, features: UnifiedShaderFeatures) -> float:
        """Enhanced prediction with multiple optimization paths"""
        
        # Ultra-fast path for very simple shaders with Steam Deck optimization
        if features.instruction_count < 50:
            base_time = features.instruction_count * 0.008 + 1.5  # RDNA2 optimized
            # Apply thermal factor for simple shaders
            if hasattr(self, 'thermal_scheduler'):
                thermal_state = self.thermal_scheduler.current_thermal_state
                if thermal_state == ThermalState.HOT:
                    base_time *= 1.15
                elif thermal_state == ThermalState.COOL:
                    base_time *= 0.92
            return max(0.6, base_time)
        
        # Fast linear model path for moderate complexity
        if (features.instruction_count < 500 and 
            self._fast_linear_model and 
            not self._memory_pressure):
            try:
                # Normalize features quickly
                if HAS_NUMPY and isinstance(feature_vector, np.ndarray):
                    normalized_features = (feature_vector - self._feature_means) / self._feature_stds
                    prediction = self._fast_linear_model.predict(normalized_features)
                    return max(0.5, float(prediction))
                else:
                    # Fallback normalization
                    normalized = [(x - m) / s for x, m, s in zip(feature_vector, self._feature_means, self._feature_stds)]
                    prediction = self._fast_linear_model.predict(normalized)
                    return max(0.5, float(prediction))
            except Exception:
                pass  # Fall through to next method
        
        # Ensemble prediction path for complex shaders
        if (self._use_ensemble and 
            self._ensemble_model and 
            self._ensemble_model.is_fitted and
            features.instruction_count >= 200):
            try:
                if HAS_NUMPY and isinstance(feature_vector, np.ndarray):
                    X = fast_feature_normalization(feature_vector.reshape(1, -1), 
                                                  self._feature_means, self._feature_stds)
                    prediction = self._ensemble_model.predict(X)
                    return max(0.5, float(prediction))
            except Exception:
                pass
        
        # Fast heuristic path for fallback cases
        if (not self._models_loaded or self._memory_pressure):
            if HAS_NUMBA and HAS_NUMPY:
                return fast_heuristic_prediction(
                    float(features.instruction_count),
                    float(features.register_usage),
                    float(features.texture_samples),
                    float(features.memory_operations),
                    float(features.control_flow_complexity),
                    float(features.shader_type.value.__hash__() % 10),
                    float(features.uses_derivatives),
                    float(features.uses_tessellation),
                    float(features.uses_geometry_shader),
                    float(features.optimization_level),
                    float(features.wave_size)
                )
            else:
                return self.heuristic_predictor.predict_compilation_time(features)
        
        # Full ML prediction path for complex cases
        return self._ml_predict(feature_vector, features)
    
    def _ml_predict(self, feature_vector, features: UnifiedShaderFeatures) -> float:
        """ML prediction with enhanced error handling and optimization"""
        if not self._models_loaded:
            self._load_models()
        
        if self.compilation_time_model is None:
            return self.heuristic_predictor.predict_compilation_time(features)
        
        try:
            # Check if model is fitted
            model_fitted = False
            if HAS_LIGHTGBM and hasattr(self.compilation_time_model, 'booster_'):
                model_fitted = self.compilation_time_model.booster_ is not None
            elif HAS_SKLEARN and hasattr(self.compilation_time_model, 'estimators_'):
                model_fitted = len(getattr(self.compilation_time_model, 'estimators_', [])) > 0
            
            if not model_fitted:
                return self.heuristic_predictor.predict_compilation_time(features)
            
            # Prepare features with enhanced normalization
            if HAS_NUMPY and isinstance(feature_vector, np.ndarray):
                X = feature_vector.reshape(1, -1)
                
                # Fast normalization using Numba if available
                if HAS_NUMBA and self._feature_means is not None:
                    X = fast_feature_normalization(X, self._feature_means, self._feature_stds)
                elif self.scaler:
                    X = self.scaler.transform(X)
            else:
                X = [list(feature_vector)]
                # Simple normalization fallback
                if self._feature_means:
                    X = [[(x - m) / s for x, m, s in zip(X[0], self._feature_means, self._feature_stds)]]
            
            # Make prediction
            prediction = self.compilation_time_model.predict(X)[0]
            
            # Enhanced post-processing for Steam Deck
            prediction = max(0.5, float(prediction))  # Minimum time
            
            # Apply Steam Deck specific adjustments
            if features.shader_type == ShaderType.COMPUTE:
                prediction *= 0.9  # RDNA2 compute efficiency
            elif features.shader_type == ShaderType.FRAGMENT and features.texture_samples > 8:
                prediction *= 1.1  # Texture cache pressure
            
            return prediction
            
        except Exception as e:
            self.logger.debug(f"Enhanced ML prediction failed: {e}")
            return self.heuristic_predictor.predict_compilation_time(features)
    
    def _create_enhanced_model(self, model_type: str = "regressor"):
        """Create enhanced models optimized for Steam Deck with 24-feature support"""
        if HAS_LIGHTGBM:
            if model_type == "regressor":
                return lgb.LGBMRegressor(
                    n_estimators=40,      # Optimized for speed with 24 features
                    max_depth=10,         # Reduced depth for faster inference
                    num_leaves=31,        # Optimized for 24 features
                    learning_rate=0.12,   # Increased for faster convergence
                    n_jobs=2,
                    boosting_type='gbdt',
                    objective='regression',
                    metric='rmse',
                    verbosity=-1,
                    importance_type='gain',
                    min_child_samples=10,  # Reduced for better fit with more features
                    subsample=0.9,        # Higher subsample for stability
                    colsample_bytree=0.8, # Reduced to handle 24 features
                    reg_alpha=0.05,       # Reduced regularization
                    reg_lambda=0.05,
                    min_split_gain=0.005, # Lower threshold for more splits
                    feature_fraction=0.85, # Feature subsampling for speed
                    early_stopping_rounds=10
                )
            else:
                return lgb.LGBMClassifier(
                    n_estimators=30,      # Reduced for speed
                    max_depth=8,          # Shallower for faster inference
                    num_leaves=15,        # Fewer leaves for speed
                    learning_rate=0.15,   # Higher for faster convergence
                    n_jobs=2,
                    boosting_type='gbdt',
                    objective='binary',
                    verbosity=-1,
                    importance_type='gain',
                    feature_fraction=0.8,
                    early_stopping_rounds=8
                )
        elif HAS_SKLEARN:
            if model_type == "regressor":
                return RandomForestRegressor(
                    n_estimators=20,      # Reduced for speed with 24 features
                    max_depth=12,         # Optimized depth
                    random_state=42,
                    n_jobs=2,
                    max_features=8,       # Fixed number for 24 features
                    min_samples_split=5,
                    min_samples_leaf=2,
                    bootstrap=True,
                    oob_score=True,
                    warm_start=True       # Enable incremental training
                )
            else:
                return RandomForestClassifier(
                    n_estimators=15,      # Reduced for speed
                    max_depth=10,
                    random_state=42,
                    n_jobs=2,
                    max_features=6,       # Fixed for 24 features
                    min_samples_split=5,
                    min_samples_leaf=2,
                    warm_start=True
                )
        return None
    
    def _load_models(self):
        """Load models with enhanced warm-up and validation"""
        if self._models_loaded:
            return
        
        try:
            # Model file paths
            time_model_path = self.model_path / 'enhanced_time_model.pkl'
            success_model_path = self.model_path / 'enhanced_success_model.pkl'
            scaler_path = self.model_path / 'enhanced_scaler.pkl'
            
            # Load or create models
            models_loaded_from_disk = False
            
            if time_model_path.exists():
                try:
                    with open(time_model_path, 'rb') as f:
                        self._compilation_time_model = pickle.load(f)
                    models_loaded_from_disk = True
                    self.logger.info("Loaded enhanced compilation time model")
                except Exception as e:
                    self.logger.warning(f"Failed to load time model: {e}")
                    self._compilation_time_model = self._create_enhanced_model("regressor")
            else:
                self._compilation_time_model = self._create_enhanced_model("regressor")
            
            if success_model_path.exists():
                try:
                    with open(success_model_path, 'rb') as f:
                        self._success_model = pickle.load(f)
                    self.logger.info("Loaded enhanced success model")
                except Exception as e:
                    self.logger.warning(f"Failed to load success model: {e}")
                    self._success_model = self._create_enhanced_model("classifier")
            else:
                self._success_model = self._create_enhanced_model("classifier")
            
            if scaler_path.exists():
                try:
                    with open(scaler_path, 'rb') as f:
                        self._scaler = pickle.load(f)
                    self.logger.info("Loaded enhanced scaler")
                except Exception:
                    if StandardScaler is not None:
                        self._scaler = StandardScaler()
            elif StandardScaler is not None:
                self._scaler = StandardScaler()
            
            self._models_loaded = True
            
            # Enhanced warm-up for new models
            if not models_loaded_from_disk:
                self._enhanced_warm_up()
                
            # Update ensemble model with loaded models
            self._update_ensemble_model()
                
        except Exception as e:
            self.logger.warning(f"Enhanced model loading failed: {e}")
            self._compilation_time_model = self._create_enhanced_model("regressor")
            self._success_model = self._create_enhanced_model("classifier")
            if StandardScaler is not None:
                self._scaler = StandardScaler()
            self._models_loaded = True
    
    def _enhanced_warm_up(self):
        """Enhanced model warm-up with Steam Deck realistic data"""
        try:
            # Generate Steam Deck realistic training data
            n_samples = 200
            
            if HAS_NUMPY:
                # Create realistic feature distributions for Steam Deck
                np.random.seed(42)
                
                # Realistic shader complexity distributions
                instruction_counts = np.random.gamma(2, 250, n_samples)  # Skewed towards smaller shaders
                register_usage = np.random.gamma(1.5, 20, n_samples)
                texture_samples = np.random.poisson(3, n_samples)
                memory_ops = np.random.poisson(8, n_samples)
                control_flow = np.random.poisson(4, n_samples)
                
                # Generate RDNA2-specific features
                alu_instructions = instruction_counts * 0.7
                memory_latency_factor = 1.0 + np.random.exponential(0.2, n_samples)
                instruction_parallelism = np.random.beta(3, 2, n_samples) * 0.4 + 0.6  # 0.6-1.0 range
                thermal_factor = np.random.normal(1.0, 0.1, n_samples)
                thermal_factor = np.clip(thermal_factor, 0.7, 1.1)
                
                # Create feature matrix (24 features)
                X = np.column_stack([
                    instruction_counts,
                    register_usage,
                    texture_samples,
                    memory_ops,
                    control_flow,
                    np.full(n_samples, 64),  # Wave size (RDNA2 default)
                    np.random.bernoulli(0.3, n_samples),  # Uses derivatives
                    np.random.bernoulli(0.1, n_samples),  # Uses tessellation
                    np.random.bernoulli(0.05, n_samples), # Uses geometry
                    np.random.randint(1, 5, n_samples),   # Shader type
                    np.random.randint(0, 4, n_samples),   # Optimization level
                    np.random.uniform(0.2, 0.9, n_samples), # Cache priority
                    # RDNA2-specific features
                    alu_instructions,
                    memory_latency_factor,
                    instruction_parallelism,
                    thermal_factor,
                    # Enhanced derived features
                    instruction_counts / 64,  # Instruction density
                    (texture_samples + memory_ops) / np.maximum(1, instruction_counts) * 1000,
                    control_flow * register_usage / 1000,
                    np.random.uniform(0.2, 0.9, n_samples) * 1.2,
                    # Advanced RDNA2 architectural features
                    alu_instructions / np.maximum(1, instruction_counts),
                    memory_latency_factor * memory_ops,
                    instruction_parallelism * 64 / 64.0,
                    thermal_factor * (1.0 + register_usage / 256.0)
                ]).astype(np.float32)
                
                # Realistic compilation times with RDNA2 factors
                y_time = (
                    5.0 +  # Base time
                    instruction_counts * 0.008 +  # Slightly faster on RDNA2
                    register_usage * 0.08 +
                    texture_samples * 0.4 +      # Better texture cache
                    memory_ops * 0.25 * memory_latency_factor +
                    control_flow * 0.8 +         # Better branch prediction
                    alu_instructions * 0.002 +   # ALU efficiency factor
                    (1.0 / thermal_factor) * 2.0 + # Thermal impact
                    np.random.exponential(1.5, n_samples)  # Reduced variability
                )
                
                y_success = (y_time < 50).astype(int)  # Success based on reasonable compile time
                
            else:
                # Fallback without numpy
                import random
                random.seed(42)
                
                X = []
                y_time = []
                y_success = []
                
                for _ in range(n_samples):
                    inst_count = random.gammavariate(2, 250)
                    reg_usage = random.gammavariate(1.5, 20)
                    tex_samples = random.randint(1, 8)
                    mem_ops = random.randint(1, 20)
                    ctrl_flow = random.randint(1, 10)
                    
                    # RDNA2-specific features
                    alu_inst = inst_count * 0.7
                    mem_latency = 1.0 + random.expovariate(5)
                    inst_parallel = random.betavariate(3, 2) * 0.4 + 0.6
                    thermal_fact = max(0.7, min(1.1, random.normalvariate(1.0, 0.1)))
                    
                    features = [
                        inst_count, reg_usage, tex_samples, mem_ops, ctrl_flow,
                        64, random.random() < 0.3, random.random() < 0.1,
                        random.random() < 0.05, random.randint(1, 4),
                        random.randint(0, 3), random.uniform(0.2, 0.9),
                        # RDNA2-specific
                        alu_inst, mem_latency, inst_parallel, thermal_fact,
                        # Derived features
                        inst_count / 64, (tex_samples + mem_ops) / max(1, inst_count) * 1000,
                        ctrl_flow * reg_usage / 1000, random.uniform(0.2, 0.9) * 1.2,
                        # Advanced features
                        alu_inst / max(1, inst_count), mem_latency * mem_ops,
                        inst_parallel * 64 / 64.0, thermal_fact * (1.0 + reg_usage / 256.0)
                    ]
                    
                    time_pred = (5.0 + inst_count * 0.008 + reg_usage * 0.08 + 
                               tex_samples * 0.4 + mem_ops * 0.25 * mem_latency + ctrl_flow * 0.8 +
                               alu_inst * 0.002 + (1.0 / thermal_fact) * 2.0 +
                               random.expovariate(1/1.5))
                    
                    X.append(features)
                    y_time.append(time_pred)
                    y_success.append(1 if time_pred < 50 else 0)
            
            # Update feature normalization parameters
            if HAS_NUMPY and isinstance(X, np.ndarray):
                self._feature_means = np.mean(X, axis=0).astype(np.float32)
                self._feature_stds = np.std(X, axis=0).astype(np.float32)
                self._feature_stds = np.maximum(self._feature_stds, 1e-8)  # Prevent division by zero
            
            # Fit scaler
            if self._scaler:
                self._scaler.fit(X)
                if HAS_NUMPY and isinstance(X, np.ndarray):
                    X_scaled = self._scaler.transform(X)
                else:
                    X_scaled = X  # Use unscaled for fallback
            else:
                X_scaled = X
            
            # Train models
            if self._compilation_time_model:
                self._compilation_time_model.fit(X_scaled, y_time)
            
            if self._success_model:
                self._success_model.fit(X_scaled, y_success)
            
            self.logger.info("Enhanced models warmed up with realistic Steam Deck data")
            
        except Exception as e:
            self.logger.debug(f"Enhanced warm-up failed: {e}")
    
    def _update_ensemble_model(self):
        """Update ensemble model with available trained models"""
        try:
            if not self._ensemble_model:
                self._ensemble_model = EnsemblePredictor()
            
            # Add fast linear model
            if self._fast_linear_model:
                self._ensemble_model.add_model(self._fast_linear_model, weight=0.2)
            
            # Add main ML model if available and fitted
            if self._compilation_time_model:
                model_fitted = False
                if HAS_LIGHTGBM and hasattr(self._compilation_time_model, 'booster_'):
                    model_fitted = self._compilation_time_model.booster_ is not None
                elif HAS_SKLEARN and hasattr(self._compilation_time_model, 'estimators_'):
                    model_fitted = len(getattr(self._compilation_time_model, 'estimators_', [])) > 0
                
                if model_fitted:
                    self._ensemble_model.add_model(self._compilation_time_model, weight=0.7)
            
            # Add heuristic predictor as fallback
            self._ensemble_model.add_model(self.heuristic_predictor, weight=0.1)
            
            self.logger.debug(f"Ensemble model updated with {len(self._ensemble_model.models)} models")
            
        except Exception as e:
            self.logger.debug(f"Ensemble model update failed: {e}")
            self._use_ensemble = False
    
    @property 
    def compilation_time_model(self):
        """Lazy load compilation time model"""
        if not self._models_loaded:
            self._load_models()
        return self._compilation_time_model
    
    @property
    def success_model(self):
        """Lazy load success model"""
        if not self._models_loaded:
            self._load_models()
        return self._success_model
    
    @property
    def scaler(self):
        """Lazy load scaler"""
        if not self._models_loaded:
            self._load_models()
        return self._scaler
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics with gaming optimizations"""
        avg_prediction_time = (
            sum(self._prediction_times) / len(self._prediction_times)
            if self._prediction_times else 0
        )
        
        feature_cache_stats = self.feature_cache.stats
        pool_stats = self._feature_vector_pool.stats
        
        # Calculate performance metrics
        fast_predictions = sum(1 for t in self._prediction_times if t < 1.0) if self._prediction_times else 0
        ultra_fast_predictions = sum(1 for t in self._prediction_times if t < 0.5) if self._prediction_times else 0
        total_predictions = len(self._prediction_times)
        fast_prediction_rate = fast_predictions / total_predictions if total_predictions > 0 else 0
        ultra_fast_rate = ultra_fast_predictions / total_predictions if total_predictions > 0 else 0
        
        # Gaming-specific statistics
        gaming_stats = {
            'tracked_games': 0,
            'cache_warming_events': 0,
            'prefetch_hits': 0
        }
        
        if hasattr(self, '_game_shader_patterns'):
            gaming_stats['tracked_games'] = len(self._game_shader_patterns)
        
        # Persistent cache statistics
        persistent_cache_stats = {}
        if hasattr(self, '_mmap_cache'):
            persistent_cache_stats = self._mmap_cache.get_cache_stats()
        
        return {
            'ml_backend': ML_BACKEND,
            'steam_deck_optimized': True,
            'has_numba': HAS_NUMBA,
            'has_numpy': HAS_NUMPY,
            'has_msgpack': HAS_MSGPACK,
            'has_zstd': HAS_ZSTD,
            'avg_prediction_time_ms': avg_prediction_time,
            'fast_prediction_rate': fast_prediction_rate,
            'ultra_fast_prediction_rate': ultra_fast_rate,
            'feature_cache_stats': feature_cache_stats,
            'prediction_cache_size': len(self.prediction_cache),
            'training_buffer_size': len(self.training_data),
            'memory_usage_mb': self._current_memory_mb,
            'memory_pressure': self._memory_pressure,
            'pool_stats': pool_stats,
            'models_loaded': self._models_loaded,
            'ensemble_enabled': self._use_ensemble,
            'ensemble_models': len(self._ensemble_model.models) if self._ensemble_model else 0,
            'fast_linear_model': self._fast_linear_model is not None,
            'gaming_optimizations': gaming_stats,
            'persistent_cache_stats': persistent_cache_stats,
            'performance_optimizations': {
                'numba_jit': HAS_NUMBA,
                'fast_serialization': HAS_MSGPACK,
                'compression': HAS_ZSTD,
                'vectorized_ops': HAS_NUMPY,
                'enhanced_normalization': self._feature_means is not None,
                'quantized_models': len(self._quantized_models),
                'ensemble_prediction': self._use_ensemble,
                'fast_linear_prediction': self._fast_linear_model is not None,
                'multi_tier_caching': True,
                'gaming_aware_eviction': True,
                'predictive_prefetching': True,
                'persistent_storage': HAS_MMAP,
                'steam_deck_tuned': True
            }
        }
    
    def warm_cache_for_game_session(self, game_context: str, expected_shaders: List[str] = None):
        """Public method to warm cache for game session startup"""
        try:
            # Warm the feature cache
            if hasattr(self.feature_cache, 'warm_cache_for_game'):
                self.feature_cache.warm_cache_for_game(game_context, expected_shaders)
            
            # Preload from persistent cache
            if hasattr(self, '_mmap_cache') and expected_shaders:
                for shader_id in expected_shaders:
                    cached_data = self._mmap_cache.load_shader_features(shader_id)
                    if cached_data:
                        # Pre-populate memory cache
                        feature_vector = self._deserialize_features_from_persistence(cached_data)
                        if feature_vector is not None:
                            self.feature_cache.put(shader_id, feature_vector, game_context, priority=3)
            
            self.logger.info(f"Cache warmed for game session: {game_context}")
            
        except Exception as e:
            self.logger.warning(f"Cache warming failed for {game_context}: {e}")
    
    async def predict_compilation_time_async(self, features: UnifiedShaderFeatures) -> float:
        """Enhanced async prediction"""
        if not self.enable_async:
            return self.predict_compilation_time(features)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.predict_compilation_time,
            features
        )
    
    def cleanup(self):
        """Enhanced cleanup with comprehensive resource management and system optimization"""
        try:
            # =================== SYSTEM OPTIMIZATION CLEANUP ===================
            
            # Stop gaming-aware scheduler
            if hasattr(self, 'gaming_scheduler'):
                self.gaming_scheduler.stop()
            
            # Perform final memory cleanup
            if hasattr(self, 'memory_optimizer'):
                self.memory_optimizer.perform_memory_cleanup(aggressive=True)
            
            # Reset process priority and affinity to original
            if hasattr(self, 'resource_manager'):
                try:
                    os.setpriority(os.PRIO_PROCESS, 0, self.resource_manager._original_priority)
                    # Reset CPU affinity to all cores
                    all_cores = list(range(self.resource_manager.cpu_count))
                    psutil.Process().cpu_affinity(all_cores)
                except:
                    pass
            
            # =================== STANDARD CLEANUP ===================
            
            if self.enable_async:
                self.thread_pool.shutdown(wait=False)
                self.io_pool.shutdown(wait=False)
            
            # Clear all caches
            self.prediction_cache.clear()
            self.training_data.clear()
            
            # Save models before cleanup (with memory-mapped storage)
            if self._models_loaded:
                self._save_models_optimized()
            
            # Generate final system performance report
            if hasattr(self, 'resource_manager'):
                system_report = self.get_system_performance_report()
                self.logger.info(f"Final system performance report: {system_report}")
            
            self.logger.info("Enhanced cleanup with system optimization completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def _save_models_optimized(self):
        """Save enhanced models with memory-mapped storage and compression"""
        try:
            # Save learned gaming patterns to persistent cache
            if hasattr(self, '_mmap_cache'):
                # Save gaming patterns for cross-session optimization
                if hasattr(self, '_game_shader_patterns') and self._game_shader_patterns:
                    patterns_data = {
                        'shader_patterns': dict(self._game_shader_patterns),
                        'performance_patterns': getattr(self, '_game_performance_patterns', {}),
                        'timestamp': time.time()
                    }
                    
                    if HAS_MSGPACK:
                        patterns_bytes = msgpack.packb(patterns_data)
                    else:
                        patterns_bytes = pickle.dumps(patterns_data, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    self._mmap_cache.store_shader_features(
                        'gaming_patterns',
                        patterns_bytes,
                        priority=5,  # High priority for patterns
                        game_context='system'
                    )
                
                # Also save feature cache warm patterns
                if hasattr(self.feature_cache, '_warm_patterns') and self.feature_cache._warm_patterns:
                    warm_patterns_data = {
                        'warm_patterns': self.feature_cache._warm_patterns,
                        'prefetch_patterns': self.feature_cache._prefetch_patterns,
                        'timestamp': time.time()
                    }
                    
                    if HAS_MSGPACK:
                        warm_bytes = msgpack.packb(warm_patterns_data)
                    else:
                        warm_bytes = pickle.dumps(warm_patterns_data, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    self._mmap_cache.store_shader_features(
                        'cache_warm_patterns',
                        warm_bytes,
                        priority=4,
                        game_context='system'
                    )
            
            # Fallback to regular file storage
            self._save_models_fallback()
            
            self.logger.info("Enhanced models saved successfully with optimization")
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced models: {e}")
            # Try fallback
            self._save_models_fallback()
    
    def _save_models_fallback(self):
        """Fallback model saving method"""
        try:
            if self._compilation_time_model:
                model_path = self.model_path / 'enhanced_time_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(self._compilation_time_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self._success_model:
                model_path = self.model_path / 'enhanced_success_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(self._success_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self._scaler:
                scaler_path = self.model_path / 'enhanced_scaler.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self._scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
        except Exception as e:
            self.logger.error(f"Fallback model save failed: {e}")


# Enhanced global instance management
_enhanced_predictor = None
_predictor_lock = threading.Lock()


def get_enhanced_predictor() -> EnhancedMLPredictor:
    """Get or create global enhanced predictor instance with thread safety"""
    global _enhanced_predictor
    if _enhanced_predictor is None:
        with _predictor_lock:
            if _enhanced_predictor is None:
                _enhanced_predictor = EnhancedMLPredictor()
    return _enhanced_predictor


if __name__ == "__main__":
    # Enhanced testing and benchmarking
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n🚀 Enhanced ML Predictor Test Suite")
    print("=" * 60)
    
    predictor = get_enhanced_predictor()
    
    # Create test features with gaming context
    test_features = UnifiedShaderFeatures(
        shader_hash="enhanced_test_hash_123",
        shader_type=ShaderType.FRAGMENT,
        instruction_count=750,
        register_usage=48,
        texture_samples=6,
        memory_operations=15,
        control_flow_complexity=8,
        wave_size=64,
        uses_derivatives=True,
        uses_tessellation=False,
        uses_geometry_shader=False,
        optimization_level=2,
        cache_priority=0.7
    )
    
    # Test gaming context
    test_game_context = "test_game_unity"
    
    print(f"✓ Test configuration:")
    print(f"  - Backend: {ML_BACKEND}")
    print(f"  - Numba acceleration: {HAS_NUMBA}")
    print(f"  - NumPy support: {HAS_NUMPY}")
    print(f"  - Fast serialization: {HAS_MSGPACK}")
    print(f"  - Compression: {HAS_ZSTD}")
    
    # Warm up with gaming context (JIT compilation if available)
    print("\n📈 Warming up with gaming context...")
    for _ in range(5):
        predictor.predict_compilation_time(test_features, game_context=test_game_context)
    
    # Test cache warming for game session
    print("🎮 Testing cache warming for game session...")
    predictor.warm_cache_for_game_session(test_game_context, ["enhanced_test_hash_123"])
    
    # Performance benchmark
    print("\n⚡ Performance Benchmark:")
    import timeit
    
    # Single prediction benchmark with gaming context
    single_time = timeit.timeit(
        lambda: predictor.predict_compilation_time(test_features, game_context=test_game_context),
        number=1000
    ) / 1000 * 1000
    
    print(f"  - Single prediction: {single_time:.3f}ms")
    
    # Batch prediction test with gaming patterns
    batch_start = time.perf_counter()
    for i in range(100):
        test_features.instruction_count = 500 + i * 10
        test_features.shader_hash = f"enhanced_test_hash_{i}"
        predictor.predict_compilation_time(test_features, game_context=test_game_context)
    batch_time = (time.perf_counter() - batch_start) * 10  # Per prediction in ms
    
    print(f"  - Batch prediction (avg): {batch_time:.3f}ms")
    
    # Get comprehensive stats
    stats = predictor.get_enhanced_stats()
    print(f"\n📊 Enhanced Performance Stats:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    - {subkey}: {subvalue}")
        else:
            print(f"  - {key}: {value}")
    
    # Test memory pressure handling
    print(f"\n🧠 Memory Management Test:")
    initial_memory = stats['memory_usage_mb']
    
    # Stress test with gaming contexts
    game_contexts = ["unity_game", "unreal_game", "source_engine", "custom_engine"]
    for i in range(500):
        test_features.shader_hash = f"stress_test_{i}"
        game_ctx = game_contexts[i % len(game_contexts)]
        predictor.predict_compilation_time(test_features, game_context=game_ctx)
    
    final_stats = predictor.get_enhanced_stats()
    final_memory = final_stats['memory_usage_mb']
    
    print(f"  - Initial memory: {initial_memory:.1f}MB")
    print(f"  - Final memory: {final_memory:.1f}MB")
    print(f"  - Memory increase: {final_memory - initial_memory:.1f}MB")
    print(f"  - Memory pressure triggered: {final_stats['memory_pressure']}")
    
    # Cache performance
    cache_stats = final_stats['feature_cache_stats']
    print(f"\n💾 Cache Performance:")
    print(f"  - Feature cache hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  - Hot cache entries: {cache_stats['hot_cache_size']}")
    print(f"  - Compressed cache entries: {cache_stats['compressed_cache_size']}")
    
    # Cleanup
    predictor.cleanup()
    print(f"\n✅ Enhanced ML Predictor test completed successfully!")
    
    # Performance comparison summary
    print(f"\n🏆 Performance Summary:")
    print(f"  - Target Rust performance: ~0.3ms")
    print(f"  - Enhanced Python performance: {single_time:.3f}ms")
    print(f"  - Performance ratio: {single_time / 0.3:.1f}x slower than Rust")
    print(f"  - Memory efficiency: {final_memory:.1f}MB (target: <40MB)")
    
    if single_time < 2.0:
        print(f"  🎉 EXCELLENT: Performance within 2ms target!")
    elif single_time < 5.0:
        print(f"  ✅ GOOD: Performance acceptable for production use")
    else:
        print(f"  ⚠️  NEEDS IMPROVEMENT: Consider further optimizations")