#!/usr/bin/env python3
"""
Runtime Dependency Manager for ML Shader Prediction Compiler

This module provides dynamic dependency switching and runtime optimization
for the Enhanced ML Predictor, allowing seamless transitions between different
dependency configurations based on current system conditions.

Features:
- Dynamic dependency backend switching
- Runtime performance monitoring
- Thermal-aware dependency management
- Steam Deck adaptive optimization
- Graceful degradation under system stress
- Memory-aware dependency selection
- Context-aware optimization profiles
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import weakref

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# RUNTIME CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class RuntimeProfile:
    """Runtime dependency profile configuration"""
    profile_id: str
    description: str
    dependencies: List[str]
    priority: int = 5  # 1-10, higher is more preferred
    conditions: Dict[str, Any] = field(default_factory=dict)
    performance_target: float = 1.0  # Expected performance multiplier
    memory_limit_mb: Optional[int] = None
    thermal_threshold: Optional[float] = None
    steam_deck_optimized: bool = False
    fallback_profile: Optional[str] = None

@dataclass
class RuntimeConditions:
    """Current system runtime conditions"""
    cpu_temperature: float
    memory_usage_mb: float
    cpu_usage_percent: float
    battery_percent: Optional[float]
    is_gaming_mode: bool
    thermal_state: str  # 'cool', 'normal', 'warm', 'hot', 'critical'
    memory_pressure: str  # 'low', 'medium', 'high', 'critical'
    power_state: str  # 'ac', 'battery', 'low_battery'
    timestamp: float

@dataclass
class DependencyBackend:
    """Wrapper for a dependency backend implementation"""
    name: str
    implementation: Any
    performance_score: float
    memory_footprint_mb: float
    initialization_time: float
    is_active: bool = False
    last_used: float = 0.0
    error_count: int = 0
    success_count: int = 0

# =============================================================================
# RUNTIME DEPENDENCY MANAGER
# =============================================================================

class RuntimeDependencyManager:
    """
    Dynamic dependency manager with runtime switching capabilities
    """
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.current_profile: Optional[RuntimeProfile] = None
        self.available_profiles: Dict[str, RuntimeProfile] = {}
        self.dependency_backends: Dict[str, Dict[str, DependencyBackend]] = {}
        self.conditions_history: List[RuntimeConditions] = []
        self.switch_lock = threading.RLock()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.condition_queue = Queue(maxsize=100)
        self.switch_callbacks: List[Callable] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}
        self.switch_count = 0
        self.optimization_decisions: List[Dict[str, Any]] = []
        
        # Initialize system monitors
        self._initialize_monitors()
        
        # Setup default profiles
        self._setup_default_profiles()
        
        # Setup default backends
        self._setup_dependency_backends()
        
        logger.info("RuntimeDependencyManager initialized")
    
    def _initialize_monitors(self) -> None:
        """Initialize system monitoring capabilities"""
        try:
            from .pure_python_fallbacks import PureThermalMonitor, PureSteamDeckDetector
            self.thermal_monitor = PureThermalMonitor()
            self.steam_deck_detector = PureSteamDeckDetector()
            self.is_steam_deck = self.steam_deck_detector.is_steam_deck()
        except ImportError:
            logger.warning("Pure Python fallbacks not available")
            self.thermal_monitor = None
            self.steam_deck_detector = None
            self.is_steam_deck = False
        
        # Initialize system monitor
        try:
            import psutil
            self.system_monitor = psutil
        except ImportError:
            try:
                from .pure_python_fallbacks import PureSystemMonitor
                self.system_monitor = PureSystemMonitor()
            except ImportError:
                self.system_monitor = None
                logger.warning("No system monitor available")
    
    def _setup_default_profiles(self) -> None:
        """Setup default runtime profiles"""
        
        # Maximum performance profile
        self.available_profiles['maximum_performance'] = RuntimeProfile(
            profile_id='maximum_performance',
            description='Maximum performance with all optimizations',
            dependencies=['numpy', 'numba', 'lightgbm', 'numexpr', 'bottleneck', 'msgpack', 'zstandard'],
            priority=10,
            conditions={
                'thermal_state': ['cool', 'normal'],
                'memory_pressure': ['low', 'medium'],
                'power_state': ['ac']
            },
            performance_target=3.0,
            memory_limit_mb=1024,
            thermal_threshold=70.0,
            steam_deck_optimized=False,
            fallback_profile='balanced'
        )
        
        # Balanced profile
        self.available_profiles['balanced'] = RuntimeProfile(
            profile_id='balanced',
            description='Balanced performance and compatibility',
            dependencies=['numpy', 'scikit-learn', 'msgpack', 'psutil'],
            priority=8,
            conditions={
                'thermal_state': ['cool', 'normal', 'warm'],
                'memory_pressure': ['low', 'medium', 'high']
            },
            performance_target=2.0,
            memory_limit_mb=512,
            thermal_threshold=80.0,
            steam_deck_optimized=True,
            fallback_profile='conservative'
        )
        
        # Conservative profile
        self.available_profiles['conservative'] = RuntimeProfile(
            profile_id='conservative',
            description='Conservative profile for constrained environments',
            dependencies=['numpy', 'psutil'],
            priority=6,
            conditions={
                'thermal_state': ['cool', 'normal', 'warm', 'hot'],
                'memory_pressure': ['low', 'medium', 'high', 'critical'],
                'power_state': ['ac', 'battery']
            },
            performance_target=1.5,
            memory_limit_mb=256,
            thermal_threshold=85.0,
            steam_deck_optimized=True,
            fallback_profile='minimal'
        )
        
        # Steam Deck optimized profile
        self.available_profiles['steam_deck_optimized'] = RuntimeProfile(
            profile_id='steam_deck_optimized',
            description='Optimized specifically for Steam Deck',
            dependencies=['numpy', 'scikit-learn', 'psutil', 'msgpack'],
            priority=9,
            conditions={
                'thermal_state': ['cool', 'normal', 'warm'],
                'memory_pressure': ['low', 'medium'],
                'power_state': ['ac', 'battery']
            },
            performance_target=2.5,
            memory_limit_mb=384,
            thermal_threshold=75.0,
            steam_deck_optimized=True,
            fallback_profile='conservative'
        )
        
        # Gaming mode profile
        self.available_profiles['gaming_mode'] = RuntimeProfile(
            profile_id='gaming_mode',
            description='Optimized for background operation during gaming',
            dependencies=['psutil'],
            priority=7,
            conditions={
                'is_gaming_mode': True,
                'thermal_state': ['cool', 'normal', 'warm', 'hot']
            },
            performance_target=1.2,
            memory_limit_mb=128,
            thermal_threshold=90.0,
            steam_deck_optimized=True,
            fallback_profile='minimal'
        )
        
        # Minimal fallback profile
        self.available_profiles['minimal'] = RuntimeProfile(
            profile_id='minimal',
            description='Pure Python fallback with minimal dependencies',
            dependencies=[],
            priority=1,
            conditions={},  # Always available
            performance_target=1.0,
            memory_limit_mb=64,
            thermal_threshold=95.0,
            steam_deck_optimized=True,
            fallback_profile=None
        )
        
        # Power saving profile
        self.available_profiles['power_saving'] = RuntimeProfile(
            profile_id='power_saving',
            description='Power-efficient profile for battery operation',
            dependencies=['psutil'],
            priority=5,
            conditions={
                'power_state': ['battery', 'low_battery'],
                'thermal_state': ['cool', 'normal', 'warm', 'hot']
            },
            performance_target=1.1,
            memory_limit_mb=96,
            thermal_threshold=85.0,
            steam_deck_optimized=True,
            fallback_profile='minimal'
        )
        
        logger.info(f"Initialized {len(self.available_profiles)} runtime profiles")
    
    def _setup_dependency_backends(self) -> None:
        """Setup available dependency backends"""
        
        # Array math backends
        self.dependency_backends['array_math'] = {}
        
        # NumPy backend
        try:
            import numpy as np
            self.dependency_backends['array_math']['numpy'] = DependencyBackend(
                name='numpy',
                implementation=np,
                performance_score=5.0,
                memory_footprint_mb=50,
                initialization_time=0.1
            )
        except ImportError:
            pass
        
        # Pure Python fallback
        try:
            from .pure_python_fallbacks import PureArrayMath
            self.dependency_backends['array_math']['pure_python'] = DependencyBackend(
                name='pure_python',
                implementation=PureArrayMath(),
                performance_score=1.0,
                memory_footprint_mb=5,
                initialization_time=0.01
            )
        except ImportError:
            pass
        
        # ML backends
        self.dependency_backends['ml_backend'] = {}
        
        # LightGBM backend
        try:
            import lightgbm as lgb
            self.dependency_backends['ml_backend']['lightgbm'] = DependencyBackend(
                name='lightgbm',
                implementation=lgb.LGBMRegressor,
                performance_score=4.0,
                memory_footprint_mb=100,
                initialization_time=0.5
            )
        except ImportError:
            pass
        
        # Scikit-learn backend
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.dependency_backends['ml_backend']['sklearn'] = DependencyBackend(
                name='sklearn',
                implementation=RandomForestRegressor,
                performance_score=3.0,
                memory_footprint_mb=80,
                initialization_time=0.2
            )
        except ImportError:
            pass
        
        # Pure Python ML backend
        try:
            from .pure_python_fallbacks import PureLinearRegressor
            self.dependency_backends['ml_backend']['pure_python'] = DependencyBackend(
                name='pure_python',
                implementation=PureLinearRegressor,
                performance_score=1.0,
                memory_footprint_mb=10,
                initialization_time=0.01
            )
        except ImportError:
            pass
        
        # Serialization backends
        self.dependency_backends['serialization'] = {}
        
        # MessagePack backend
        try:
            import msgpack
            self.dependency_backends['serialization']['msgpack'] = DependencyBackend(
                name='msgpack',
                implementation=msgpack,
                performance_score=2.0,
                memory_footprint_mb=15,
                initialization_time=0.05
            )
        except ImportError:
            pass
        
        # Pure Python serialization backend
        try:
            from .pure_python_fallbacks import PureSerializer
            self.dependency_backends['serialization']['pure_python'] = DependencyBackend(
                name='pure_python',
                implementation=PureSerializer(),
                performance_score=1.0,
                memory_footprint_mb=5,
                initialization_time=0.01
            )
        except ImportError:
            pass
        
        # Compression backends
        self.dependency_backends['compression'] = {}
        
        # Zstandard backend
        try:
            import zstandard as zstd
            self.dependency_backends['compression']['zstandard'] = DependencyBackend(
                name='zstandard',
                implementation=zstd,
                performance_score=3.0,
                memory_footprint_mb=25,
                initialization_time=0.1
            )
        except ImportError:
            pass
        
        # Pure Python compression backend
        try:
            from .pure_python_fallbacks import PureCompression
            self.dependency_backends['compression']['pure_python'] = DependencyBackend(
                name='pure_python',
                implementation=PureCompression(),
                performance_score=1.0,
                memory_footprint_mb=5,
                initialization_time=0.01
            )
        except ImportError:
            pass
        
        logger.info(f"Initialized {sum(len(backends) for backends in self.dependency_backends.values())} dependency backends")
    
    def get_current_conditions(self) -> RuntimeConditions:
        """Get current system runtime conditions"""
        current_time = time.time()
        
        # Get temperature
        if self.thermal_monitor:
            cpu_temp = self.thermal_monitor.get_cpu_temperature()
            thermal_state = self.thermal_monitor.get_thermal_state()
        else:
            cpu_temp = 50.0  # Default safe temperature
            thermal_state = 'normal'
        
        # Get memory usage
        if self.system_monitor:
            try:
                if hasattr(self.system_monitor, 'virtual_memory'):
                    memory_info = self.system_monitor.virtual_memory()
                    memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
                    memory_percent = memory_info.percent
                else:
                    memory_info = self.system_monitor.memory_info()
                    memory_usage_mb = memory_info.rss / (1024 * 1024)
                    memory_percent = 50.0  # Fallback estimate
            except Exception:
                memory_usage_mb = 256.0  # Fallback estimate
                memory_percent = 50.0
        else:
            memory_usage_mb = 256.0
            memory_percent = 50.0
        
        # Get CPU usage
        if self.system_monitor and hasattr(self.system_monitor, 'cpu_percent'):
            try:
                cpu_usage = self.system_monitor.cpu_percent(interval=None)
            except Exception:
                cpu_usage = 25.0  # Fallback estimate
        else:
            cpu_usage = 25.0
        
        # Get battery info
        battery_percent = None
        if self.system_monitor and hasattr(self.system_monitor, 'sensors_battery'):
            try:
                battery = self.system_monitor.sensors_battery()
                if battery:
                    battery_percent = battery.percent
            except Exception:
                pass
        
        # Determine memory pressure
        if memory_percent < 60:
            memory_pressure = 'low'
        elif memory_percent < 75:
            memory_pressure = 'medium'
        elif memory_percent < 90:
            memory_pressure = 'high'
        else:
            memory_pressure = 'critical'
        
        # Determine power state
        if battery_percent is None:
            power_state = 'ac'
        elif battery_percent > 20:
            power_state = 'battery'
        else:
            power_state = 'low_battery'
        
        # Detect gaming mode (Steam Deck specific)
        is_gaming_mode = False
        if self.is_steam_deck:
            try:
                # Check for gamescope process (Gaming Mode indicator)
                import subprocess
                result = subprocess.run(['pgrep', '-f', 'gamescope'], 
                                      capture_output=True, text=True, timeout=2)
                is_gaming_mode = result.returncode == 0
            except Exception:
                pass
        
        return RuntimeConditions(
            cpu_temperature=cpu_temp,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage,
            battery_percent=battery_percent,
            is_gaming_mode=is_gaming_mode,
            thermal_state=thermal_state,
            memory_pressure=memory_pressure,
            power_state=power_state,
            timestamp=current_time
        )
    
    def start_monitoring(self) -> None:
        """Start runtime monitoring and adaptive switching"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="RuntimeDependencyMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Runtime monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop runtime monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=self.monitoring_interval * 2)
        logger.info("Runtime monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        logger.info("Runtime monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Get current conditions
                conditions = self.get_current_conditions()
                
                # Store conditions history
                self.conditions_history.append(conditions)
                if len(self.conditions_history) > 100:  # Keep last 100 readings
                    self.conditions_history.pop(0)
                
                # Add to condition queue for other consumers
                try:
                    self.condition_queue.put_nowait(conditions)
                except:
                    pass  # Queue full, skip
                
                # Check if profile switch is needed
                optimal_profile = self._select_optimal_profile(conditions)
                if optimal_profile and optimal_profile != self.current_profile:
                    logger.info(f"Conditions changed, switching to profile: {optimal_profile.profile_id}")
                    self._switch_to_profile(optimal_profile, conditions)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
        
        logger.info("Runtime monitoring loop ended")
    
    def _select_optimal_profile(self, conditions: RuntimeConditions) -> Optional[RuntimeProfile]:
        """Select the optimal profile for current conditions"""
        compatible_profiles = []
        
        for profile in self.available_profiles.values():
            if self._is_profile_compatible(profile, conditions):
                compatibility_score = self._calculate_compatibility_score(profile, conditions)
                compatible_profiles.append((profile, compatibility_score))
        
        if not compatible_profiles:
            # Fallback to minimal profile
            return self.available_profiles.get('minimal')
        
        # Sort by compatibility score and priority
        compatible_profiles.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        
        return compatible_profiles[0][0]
    
    def _is_profile_compatible(self, profile: RuntimeProfile, conditions: RuntimeConditions) -> bool:
        """Check if a profile is compatible with current conditions"""
        # Check thermal constraints
        if profile.thermal_threshold and conditions.cpu_temperature > profile.thermal_threshold:
            return False
        
        # Check memory constraints
        if profile.memory_limit_mb and conditions.memory_usage_mb > profile.memory_limit_mb:
            return False
        
        # Check profile-specific conditions
        for condition_key, condition_values in profile.conditions.items():
            if condition_key == 'thermal_state':
                if conditions.thermal_state not in condition_values:
                    return False
            elif condition_key == 'memory_pressure':
                if conditions.memory_pressure not in condition_values:
                    return False
            elif condition_key == 'power_state':
                if conditions.power_state not in condition_values:
                    return False
            elif condition_key == 'is_gaming_mode':
                if conditions.is_gaming_mode != condition_values:
                    return False
        
        return True
    
    def _calculate_compatibility_score(self, profile: RuntimeProfile, conditions: RuntimeConditions) -> float:
        """Calculate compatibility score for a profile"""
        score = 0.0
        
        # Base score from profile priority
        score += profile.priority
        
        # Thermal efficiency bonus
        if profile.thermal_threshold:
            thermal_margin = profile.thermal_threshold - conditions.cpu_temperature
            score += max(0, thermal_margin / 10.0)
        
        # Memory efficiency bonus
        if profile.memory_limit_mb:
            memory_margin = profile.memory_limit_mb - conditions.memory_usage_mb
            score += max(0, memory_margin / 100.0)
        
        # Steam Deck optimization bonus
        if self.is_steam_deck and profile.steam_deck_optimized:
            score += 2.0
        
        # Gaming mode compatibility
        if conditions.is_gaming_mode:
            if 'is_gaming_mode' in profile.conditions:
                score += 3.0
            else:
                score -= 1.0  # Penalize non-gaming profiles during gaming
        
        # Performance target bonus (higher performance preferred when conditions allow)
        if conditions.thermal_state in ['cool', 'normal'] and conditions.memory_pressure in ['low', 'medium']:
            score += profile.performance_target
        
        return score
    
    def _switch_to_profile(self, new_profile: RuntimeProfile, conditions: RuntimeConditions) -> bool:
        """Switch to a new runtime profile"""
        with self.switch_lock:
            old_profile = self.current_profile
            
            try:
                # Record the switch decision
                decision_record = {
                    'timestamp': time.time(),
                    'old_profile': old_profile.profile_id if old_profile else None,
                    'new_profile': new_profile.profile_id,
                    'conditions': {
                        'thermal_state': conditions.thermal_state,
                        'memory_pressure': conditions.memory_pressure,
                        'cpu_temperature': conditions.cpu_temperature,
                        'memory_usage_mb': conditions.memory_usage_mb,
                        'is_gaming_mode': conditions.is_gaming_mode,
                        'power_state': conditions.power_state
                    },
                    'reason': 'adaptive_optimization'
                }
                
                # Switch dependency backends based on new profile
                success = self._activate_profile_backends(new_profile)
                
                if success:
                    self.current_profile = new_profile
                    self.switch_count += 1
                    self.optimization_decisions.append(decision_record)
                    
                    # Notify callbacks
                    for callback in self.switch_callbacks:
                        try:
                            callback(old_profile, new_profile, conditions)
                        except Exception as e:
                            logger.error(f"Error in switch callback: {e}")
                    
                    logger.info(f"Successfully switched to profile '{new_profile.profile_id}'")
                    return True
                else:
                    logger.error(f"Failed to activate backends for profile '{new_profile.profile_id}'")
                    decision_record['success'] = False
                    self.optimization_decisions.append(decision_record)
                    return False
                
            except Exception as e:
                logger.error(f"Error switching to profile '{new_profile.profile_id}': {e}")
                return False
    
    def _activate_profile_backends(self, profile: RuntimeProfile) -> bool:
        """Activate dependency backends for a profile"""
        try:
            activated_backends = {}
            
            # For each backend category, select the best available backend
            for category, backends in self.dependency_backends.items():
                best_backend = None
                best_score = -1
                
                for backend_name, backend in backends.items():
                    # Check if this backend is available for the profile
                    if self._is_backend_suitable_for_profile(backend, profile):
                        # Calculate suitability score
                        score = self._calculate_backend_score(backend, profile)
                        if score > best_score:
                            best_backend = backend
                            best_score = score
                
                if best_backend:
                    # Deactivate current backend if different
                    for backend in backends.values():
                        if backend.is_active and backend != best_backend:
                            backend.is_active = False
                    
                    # Activate best backend
                    best_backend.is_active = True
                    best_backend.last_used = time.time()
                    activated_backends[category] = best_backend
                    
                    logger.debug(f"Activated {best_backend.name} backend for {category}")
            
            return len(activated_backends) > 0
            
        except Exception as e:
            logger.error(f"Error activating profile backends: {e}")
            return False
    
    def _is_backend_suitable_for_profile(self, backend: DependencyBackend, profile: RuntimeProfile) -> bool:
        """Check if a backend is suitable for a profile"""
        # Check if backend is in profile dependencies or is pure Python fallback
        if backend.name == 'pure_python':
            return True  # Pure Python is always suitable
        
        # Extract module name from backend
        backend_module = backend.name.split('.')[0] if '.' in backend.name else backend.name
        
        return backend_module in profile.dependencies
    
    def _calculate_backend_score(self, backend: DependencyBackend, profile: RuntimeProfile) -> float:
        """Calculate suitability score for a backend given a profile"""
        score = 0.0
        
        # Base performance score
        score += backend.performance_score
        
        # Memory efficiency (prefer lower memory footprint for constrained profiles)
        if profile.memory_limit_mb:
            memory_efficiency = max(0, 1.0 - (backend.memory_footprint_mb / profile.memory_limit_mb))
            score += memory_efficiency * 2.0
        
        # Reliability score (based on success/error ratio)
        total_uses = backend.success_count + backend.error_count
        if total_uses > 0:
            reliability = backend.success_count / total_uses
            score += reliability * 2.0
        
        # Initialization speed bonus (prefer faster initialization)
        score += max(0, 1.0 - backend.initialization_time)
        
        # Recent usage bonus
        time_since_use = time.time() - backend.last_used
        if time_since_use < 300:  # Used in last 5 minutes
            score += 0.5
        
        return score
    
    def get_active_backends(self) -> Dict[str, DependencyBackend]:
        """Get currently active dependency backends"""
        active_backends = {}
        
        for category, backends in self.dependency_backends.items():
            for backend_name, backend in backends.items():
                if backend.is_active:
                    active_backends[category] = backend
                    break
        
        return active_backends
    
    def get_backend_for_category(self, category: str) -> Optional[DependencyBackend]:
        """Get the active backend for a specific category"""
        if category not in self.dependency_backends:
            return None
        
        for backend in self.dependency_backends[category].values():
            if backend.is_active:
                return backend
        
        # Return pure Python fallback if available
        return self.dependency_backends[category].get('pure_python')
    
    def manual_switch_profile(self, profile_id: str, force: bool = False) -> bool:
        """Manually switch to a specific profile"""
        if profile_id not in self.available_profiles:
            logger.error(f"Profile '{profile_id}' not found")
            return False
        
        target_profile = self.available_profiles[profile_id]
        current_conditions = self.get_current_conditions()
        
        # Check compatibility unless forced
        if not force and not self._is_profile_compatible(target_profile, current_conditions):
            logger.warning(f"Profile '{profile_id}' not compatible with current conditions")
            return False
        
        return self._switch_to_profile(target_profile, current_conditions)
    
    def add_switch_callback(self, callback: Callable) -> None:
        """Add a callback for profile switches"""
        self.switch_callbacks.append(callback)
    
    def remove_switch_callback(self, callback: Callable) -> None:
        """Remove a profile switch callback"""
        if callback in self.switch_callbacks:
            self.switch_callbacks.remove(callback)
    
    @contextmanager
    def temporary_profile(self, profile_id: str):
        """Context manager for temporary profile switching"""
        original_profile = self.current_profile
        
        try:
            if self.manual_switch_profile(profile_id, force=True):
                yield self.current_profile
            else:
                yield original_profile
        finally:
            if original_profile and original_profile != self.current_profile:
                self.manual_switch_profile(original_profile.profile_id, force=True)
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization decisions"""
        return self.optimization_decisions.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the runtime manager"""
        return {
            'switch_count': self.switch_count,
            'current_profile': self.current_profile.profile_id if self.current_profile else None,
            'monitoring_active': self.monitoring_active,
            'available_profiles': list(self.available_profiles.keys()),
            'active_backends': {cat: backend.name for cat, backend in self.get_active_backends().items()},
            'conditions_history_length': len(self.conditions_history),
            'optimization_decisions_count': len(self.optimization_decisions)
        }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        conditions = self.get_current_conditions()
        active_backends = self.get_active_backends()
        
        # Calculate health scores
        thermal_health = max(0, 1.0 - (conditions.cpu_temperature - 40) / 50)  # 40-90°C range
        memory_health = 1.0 - (conditions.memory_usage_mb / 2048)  # Assume 2GB reasonable limit
        
        backend_health = 0.0
        if active_backends:
            total_reliability = 0.0
            for backend in active_backends.values():
                total_uses = backend.success_count + backend.error_count
                if total_uses > 0:
                    total_reliability += backend.success_count / total_uses
            backend_health = total_reliability / len(active_backends)
        
        overall_health = (thermal_health + memory_health + backend_health) / 3.0
        
        return {
            'timestamp': time.time(),
            'overall_health': overall_health,
            'thermal_health': thermal_health,
            'memory_health': memory_health,
            'backend_health': backend_health,
            'current_conditions': {
                'cpu_temperature': conditions.cpu_temperature,
                'thermal_state': conditions.thermal_state,
                'memory_usage_mb': conditions.memory_usage_mb,
                'memory_pressure': conditions.memory_pressure,
                'cpu_usage': conditions.cpu_usage_percent,
                'power_state': conditions.power_state,
                'is_gaming_mode': conditions.is_gaming_mode
            },
            'active_profile': self.current_profile.profile_id if self.current_profile else None,
            'profile_performance_target': self.current_profile.performance_target if self.current_profile else 1.0,
            'recommendations': self._generate_health_recommendations(conditions, overall_health)
        }
    
    def _generate_health_recommendations(self, conditions: RuntimeConditions, health: float) -> List[str]:
        """Generate health-based recommendations"""
        recommendations = []
        
        if health < 0.5:
            recommendations.append("System health is poor - consider switching to conservative profile")
        
        if conditions.thermal_state in ['hot', 'critical']:
            recommendations.append("High temperature detected - reduce computational load")
        
        if conditions.memory_pressure in ['high', 'critical']:
            recommendations.append("High memory pressure - consider freeing memory or switching to minimal profile")
        
        if conditions.power_state == 'low_battery':
            recommendations.append("Low battery - switch to power saving profile")
        
        if conditions.is_gaming_mode and self.current_profile and self.current_profile.profile_id != 'gaming_mode':
            recommendations.append("Gaming mode detected - consider switching to gaming profile")
        
        return recommendations
    
    def export_configuration(self, path: Path) -> None:
        """Export current configuration and state"""
        config = {
            'current_profile': self.current_profile.profile_id if self.current_profile else None,
            'available_profiles': {
                pid: {
                    'profile_id': profile.profile_id,
                    'description': profile.description,
                    'dependencies': profile.dependencies,
                    'priority': profile.priority,
                    'performance_target': profile.performance_target,
                    'steam_deck_optimized': profile.steam_deck_optimized
                }
                for pid, profile in self.available_profiles.items()
            },
            'active_backends': {
                category: backend.name
                for category, backend in self.get_active_backends().items()
            },
            'performance_metrics': self.get_performance_metrics(),
            'optimization_history': self.optimization_decisions[-10:],  # Last 10 decisions
            'system_info': {
                'is_steam_deck': self.is_steam_deck,
                'monitoring_active': self.monitoring_active,
                'switch_count': self.switch_count
            },
            'timestamp': time.time()
        }
        
        import json
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration exported to {path}")


# =============================================================================
# CONVENIENCE FUNCTIONS AND DECORATORS
# =============================================================================

# Global runtime manager instance
_runtime_manager: Optional[RuntimeDependencyManager] = None

def get_runtime_manager() -> RuntimeDependencyManager:
    """Get or create the global runtime manager"""
    global _runtime_manager
    if _runtime_manager is None:
        _runtime_manager = RuntimeDependencyManager()
    return _runtime_manager

def adaptive_dependency(category: str):
    """Decorator for functions that use adaptive dependencies"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_runtime_manager()
            backend = manager.get_backend_for_category(category)
            
            if backend:
                # Inject the appropriate implementation
                kwargs[f'{category}_backend'] = backend.implementation
                
                try:
                    result = func(*args, **kwargs)
                    backend.success_count += 1
                    return result
                except Exception as e:
                    backend.error_count += 1
                    logger.error(f"Error using {backend.name} backend: {e}")
                    
                    # Try fallback if available
                    fallback_backend = manager.dependency_backends[category].get('pure_python')
                    if fallback_backend and fallback_backend != backend:
                        kwargs[f'{category}_backend'] = fallback_backend.implementation
                        return func(*args, **kwargs)
                    else:
                        raise
            else:
                # No backend available
                raise RuntimeError(f"No backend available for category '{category}'")
        
        return wrapper
    return decorator

@contextmanager
def performance_profile(profile_id: str):
    """Context manager for temporary performance profile"""
    manager = get_runtime_manager()
    with manager.temporary_profile(profile_id):
        yield

def optimize_for_gaming():
    """Quick optimization for gaming mode"""
    manager = get_runtime_manager()
    return manager.manual_switch_profile('gaming_mode')

def optimize_for_performance():
    """Quick optimization for maximum performance"""
    manager = get_runtime_manager()
    return manager.manual_switch_profile('maximum_performance')

def optimize_for_battery():
    """Quick optimization for battery saving"""
    manager = get_runtime_manager()
    return manager.manual_switch_profile('power_saving')


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n⚡ Runtime Dependency Manager Test Suite")
    print("=" * 55)
    
    # Create runtime manager
    manager = RuntimeDependencyManager()
    
    # Get current conditions
    print("\n📊 Current System Conditions:")
    conditions = manager.get_current_conditions()
    print(f"  Temperature: {conditions.cpu_temperature:.1f}°C ({conditions.thermal_state})")
    print(f"  Memory: {conditions.memory_usage_mb:.0f}MB ({conditions.memory_pressure})")
    print(f"  CPU Usage: {conditions.cpu_usage_percent:.1f}%")
    print(f"  Power: {conditions.power_state}")
    print(f"  Gaming Mode: {conditions.is_gaming_mode}")
    
    # Show available profiles
    print(f"\n🎯 Available Profiles:")
    for profile_id, profile in manager.available_profiles.items():
        print(f"  • {profile_id}: {profile.description} (priority: {profile.priority})")
    
    # Select optimal profile
    optimal_profile = manager._select_optimal_profile(conditions)
    print(f"\n🚀 Optimal Profile: {optimal_profile.profile_id if optimal_profile else 'None'}")
    
    # Show active backends
    print(f"\n🔧 Available Backends:")
    for category, backends in manager.dependency_backends.items():
        print(f"  {category}:")
        for name, backend in backends.items():
            status = "🟢" if backend.is_active else "⚪"
            print(f"    {status} {name} (score: {backend.performance_score:.1f}, mem: {backend.memory_footprint_mb}MB)")
    
    # Test manual profile switching
    print(f"\n🔄 Testing Manual Profile Switching:")
    
    test_profiles = ['balanced', 'conservative', 'minimal']
    for profile_id in test_profiles:
        if profile_id in manager.available_profiles:
            success = manager.manual_switch_profile(profile_id, force=True)
            print(f"  Switch to {profile_id}: {'✅' if success else '❌'}")
            
            if success:
                active_backends = manager.get_active_backends()
                backend_names = [backend.name for backend in active_backends.values()]
                print(f"    Active backends: {', '.join(backend_names)}")
    
    # Test context manager
    print(f"\n🎭 Testing Temporary Profile Context:")
    original_profile = manager.current_profile.profile_id if manager.current_profile else 'None'
    
    with manager.temporary_profile('gaming_mode'):
        current_in_context = manager.current_profile.profile_id if manager.current_profile else 'None'
        print(f"  In context: {current_in_context}")
    
    final_profile = manager.current_profile.profile_id if manager.current_profile else 'None'
    print(f"  After context: {final_profile}")
    print(f"  Context worked: {'✅' if final_profile == original_profile else '❌'}")
    
    # Test adaptive dependency decorator
    print(f"\n🧪 Testing Adaptive Dependency Decorator:")
    
    @adaptive_dependency('array_math')
    def test_array_function(data, array_math_backend=None):
        """Test function using adaptive array math backend"""
        if hasattr(array_math_backend, 'array'):
            # NumPy-like interface
            arr = array_math_backend.array(data)
            return array_math_backend.mean(arr) if hasattr(array_math_backend, 'mean') else sum(data) / len(data)
        else:
            # Pure Python fallback
            return sum(data) / len(data)
    
    test_data = [1, 2, 3, 4, 5]
    result = test_array_function(test_data)
    print(f"  Array mean result: {result}")
    
    # Get health report
    print(f"\n🏥 System Health Report:")
    health_report = manager.get_system_health_report()
    print(f"  Overall Health: {health_report['overall_health']:.1%}")
    print(f"  Thermal Health: {health_report['thermal_health']:.1%}")
    print(f"  Memory Health: {health_report['memory_health']:.1%}")
    print(f"  Backend Health: {health_report['backend_health']:.1%}")
    
    if health_report['recommendations']:
        print(f"  Recommendations:")
        for rec in health_report['recommendations']:
            print(f"    💡 {rec}")
    
    # Export configuration
    config_path = Path("/tmp/runtime_dependency_config.json")
    manager.export_configuration(config_path)
    print(f"\n💾 Configuration exported to {config_path}")
    
    print(f"\n✅ Runtime Dependency Manager test completed successfully!")
    print(f"🎯 Final profile: {manager.current_profile.profile_id if manager.current_profile else 'None'}")