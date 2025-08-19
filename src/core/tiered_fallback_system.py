#!/usr/bin/env python3
"""
Tiered Fallback System for ML Shader Prediction Compiler

This module implements a sophisticated tiered fallback system that provides
graceful degradation from high-performance ML libraries to pure Python
implementations while maintaining API compatibility.

Features:
- Multi-tier fallback chains (Tier 1: Optimal -> Tier 2: Compatible -> Tier 3: Pure Python)
- Dynamic fallback switching based on runtime conditions
- Performance-aware fallback selection
- Steam Deck optimized fallback paths
- Memory-efficient fallback implementations
- Comprehensive error recovery and logging
- API compatibility layer for seamless switching
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
import weakref
from typing import Dict, List, Any, Optional, Union, Callable, Type, Protocol
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps, lru_cache
from collections import defaultdict, deque
from enum import Enum, auto
import gc

# Import our components
from .enhanced_dependency_detector import get_detector, EnhancedDependencyDetector
from .dependency_version_manager import get_version_manager

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# FALLBACK TIER DEFINITIONS
# =============================================================================

class FallbackTier(Enum):
    """Fallback tier levels"""
    OPTIMAL = 1      # High-performance libraries (LightGBM, NumPy + Numba)
    COMPATIBLE = 2   # Standard libraries (scikit-learn, NumPy)
    EFFICIENT = 3    # Lightweight libraries (basic NumPy, psutil)
    PURE_PYTHON = 4  # Pure Python implementations

class FallbackReason(Enum):
    """Reasons for fallback activation"""
    DEPENDENCY_MISSING = auto()
    VERSION_INCOMPATIBLE = auto()
    IMPORT_ERROR = auto()
    RUNTIME_ERROR = auto()
    PERFORMANCE_DEGRADATION = auto()
    MEMORY_PRESSURE = auto()
    THERMAL_THROTTLING = auto()
    STEAM_DECK_OPTIMIZATION = auto()
    USER_PREFERENCE = auto()

@dataclass
class FallbackImplementation:
    """Implementation details for a fallback"""
    name: str
    tier: FallbackTier
    dependencies: List[str]
    performance_multiplier: float
    memory_footprint_mb: int
    initialization_time: float
    stability_score: float  # 0.0 - 1.0
    steam_deck_optimized: bool
    implementation_class: Optional[Type] = None
    factory_function: Optional[Callable] = None
    test_function: Optional[Callable] = None

@dataclass
class FallbackStatus:
    """Current fallback status"""
    active_tier: FallbackTier
    active_implementation: str
    reason: FallbackReason
    performance_impact: float
    fallback_chain: List[str]
    last_switch_time: float
    switch_count: int
    error_count: int
    is_stable: bool

class FallbackCapability(Protocol):
    """Protocol for fallback-capable components"""
    
    def supports_fallback(self) -> bool:
        """Check if component supports fallback"""
        ...
    
    def get_current_tier(self) -> FallbackTier:
        """Get current fallback tier"""
        ...
    
    def switch_to_tier(self, tier: FallbackTier) -> bool:
        """Switch to specific fallback tier"""
        ...

# =============================================================================
# TIERED FALLBACK SYSTEM
# =============================================================================

class TieredFallbackSystem:
    """
    Comprehensive tiered fallback system with intelligent switching
    """
    
    def __init__(self, 
                 auto_fallback: bool = True,
                 performance_threshold: float = 0.5,
                 memory_limit_mb: int = 1024):
        
        self.auto_fallback = auto_fallback
        self.performance_threshold = performance_threshold
        self.memory_limit_mb = memory_limit_mb
        
        # System state
        self.detector = get_detector()
        self.version_manager = get_version_manager()
        self.is_steam_deck = self.version_manager.system_info['is_steam_deck']
        
        # Fallback state
        self.fallback_implementations: Dict[str, List[FallbackImplementation]] = {}
        self.active_fallbacks: Dict[str, FallbackStatus] = {}
        self.fallback_cache: Dict[str, Any] = {}
        self.fallback_lock = threading.RLock()
        
        # Performance monitoring
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.switch_history: List[Dict[str, Any]] = []
        
        # Initialize fallback implementations
        self._initialize_fallback_implementations()
        
        # Perform initial fallback detection and setup
        self._setup_initial_fallbacks()
        
        logger.info(f"TieredFallbackSystem initialized with {len(self.fallback_implementations)} component types")
        logger.info(f"Auto-fallback: {auto_fallback}, Steam Deck: {self.is_steam_deck}")

    def _initialize_fallback_implementations(self) -> None:
        """Initialize all fallback implementation definitions"""
        
        # Array Math Fallbacks
        self.fallback_implementations['array_math'] = [
            FallbackImplementation(
                name='numpy_optimized',
                tier=FallbackTier.OPTIMAL,
                dependencies=['numpy', 'numba'],
                performance_multiplier=5.0,
                memory_footprint_mb=80,
                initialization_time=0.2,
                stability_score=0.95,
                steam_deck_optimized=True,
                factory_function=self._create_numpy_optimized_math
            ),
            FallbackImplementation(
                name='numpy_standard',
                tier=FallbackTier.COMPATIBLE,
                dependencies=['numpy'],
                performance_multiplier=3.0,
                memory_footprint_mb=50,
                initialization_time=0.1,
                stability_score=0.98,
                steam_deck_optimized=True,
                factory_function=self._create_numpy_standard_math
            ),
            FallbackImplementation(
                name='numpy_minimal',
                tier=FallbackTier.EFFICIENT,
                dependencies=['numpy'],
                performance_multiplier=2.0,
                memory_footprint_mb=30,
                initialization_time=0.05,
                stability_score=0.99,
                steam_deck_optimized=True,
                factory_function=self._create_numpy_minimal_math
            ),
            FallbackImplementation(
                name='pure_python_math',
                tier=FallbackTier.PURE_PYTHON,
                dependencies=[],
                performance_multiplier=1.0,
                memory_footprint_mb=5,
                initialization_time=0.01,
                stability_score=1.0,
                steam_deck_optimized=True,
                factory_function=self._create_pure_python_math
            )
        ]
        
        # ML Algorithm Fallbacks
        self.fallback_implementations['ml_algorithms'] = [
            FallbackImplementation(
                name='lightgbm_optimized',
                tier=FallbackTier.OPTIMAL,
                dependencies=['lightgbm', 'numpy'],
                performance_multiplier=6.0,
                memory_footprint_mb=150,
                initialization_time=0.5,
                stability_score=0.92,
                steam_deck_optimized=True,
                factory_function=self._create_lightgbm_optimized
            ),
            FallbackImplementation(
                name='sklearn_randomforest',
                tier=FallbackTier.COMPATIBLE,
                dependencies=['scikit-learn', 'numpy'],
                performance_multiplier=3.5,
                memory_footprint_mb=100,
                initialization_time=0.2,
                stability_score=0.96,
                steam_deck_optimized=True,
                factory_function=self._create_sklearn_randomforest
            ),
            FallbackImplementation(
                name='sklearn_linear',
                tier=FallbackTier.EFFICIENT,
                dependencies=['scikit-learn'],
                performance_multiplier=2.0,
                memory_footprint_mb=60,
                initialization_time=0.1,
                stability_score=0.98,
                steam_deck_optimized=True,
                factory_function=self._create_sklearn_linear
            ),
            FallbackImplementation(
                name='pure_python_ml',
                tier=FallbackTier.PURE_PYTHON,
                dependencies=[],
                performance_multiplier=1.0,
                memory_footprint_mb=10,
                initialization_time=0.02,
                stability_score=1.0,
                steam_deck_optimized=True,
                factory_function=self._create_pure_python_ml
            )
        ]
        
        # System Monitoring Fallbacks
        self.fallback_implementations['system_monitor'] = [
            FallbackImplementation(
                name='psutil_full',
                tier=FallbackTier.OPTIMAL,
                dependencies=['psutil'],
                performance_multiplier=3.0,
                memory_footprint_mb=25,
                initialization_time=0.05,
                stability_score=0.95,
                steam_deck_optimized=True,
                factory_function=self._create_psutil_full
            ),
            FallbackImplementation(
                name='psutil_minimal',
                tier=FallbackTier.COMPATIBLE,
                dependencies=['psutil'],
                performance_multiplier=2.0,
                memory_footprint_mb=15,
                initialization_time=0.03,
                stability_score=0.98,
                steam_deck_optimized=True,
                factory_function=self._create_psutil_minimal
            ),
            FallbackImplementation(
                name='system_proc',
                tier=FallbackTier.EFFICIENT,
                dependencies=[],
                performance_multiplier=1.5,
                memory_footprint_mb=8,
                initialization_time=0.02,
                stability_score=0.92,
                steam_deck_optimized=True,
                factory_function=self._create_system_proc
            ),
            FallbackImplementation(
                name='pure_system_monitor',
                tier=FallbackTier.PURE_PYTHON,
                dependencies=[],
                performance_multiplier=1.0,
                memory_footprint_mb=5,
                initialization_time=0.01,
                stability_score=1.0,
                steam_deck_optimized=True,
                factory_function=self._create_pure_system_monitor
            )
        ]
        
        # Serialization Fallbacks
        self.fallback_implementations['serialization'] = [
            FallbackImplementation(
                name='msgpack_optimized',
                tier=FallbackTier.OPTIMAL,
                dependencies=['msgpack'],
                performance_multiplier=3.0,
                memory_footprint_mb=20,
                initialization_time=0.05,
                stability_score=0.96,
                steam_deck_optimized=True,
                factory_function=self._create_msgpack_optimized
            ),
            FallbackImplementation(
                name='msgpack_standard',
                tier=FallbackTier.COMPATIBLE,
                dependencies=['msgpack'],
                performance_multiplier=2.5,
                memory_footprint_mb=15,
                initialization_time=0.03,
                stability_score=0.98,
                steam_deck_optimized=True,
                factory_function=self._create_msgpack_standard
            ),
            FallbackImplementation(
                name='json_compressed',
                tier=FallbackTier.EFFICIENT,
                dependencies=[],
                performance_multiplier=1.5,
                memory_footprint_mb=8,
                initialization_time=0.02,
                stability_score=0.99,
                steam_deck_optimized=True,
                factory_function=self._create_json_compressed
            ),
            FallbackImplementation(
                name='pure_json',
                tier=FallbackTier.PURE_PYTHON,
                dependencies=[],
                performance_multiplier=1.0,
                memory_footprint_mb=5,
                initialization_time=0.01,
                stability_score=1.0,
                steam_deck_optimized=True,
                factory_function=self._create_pure_json
            )
        ]
        
        # Compression Fallbacks
        self.fallback_implementations['compression'] = [
            FallbackImplementation(
                name='zstandard_fast',
                tier=FallbackTier.OPTIMAL,
                dependencies=['zstandard'],
                performance_multiplier=4.0,
                memory_footprint_mb=40,
                initialization_time=0.1,
                stability_score=0.93,
                steam_deck_optimized=False,  # Compilation issues on Steam Deck
                factory_function=self._create_zstandard_fast
            ),
            FallbackImplementation(
                name='zstandard_compatible',
                tier=FallbackTier.COMPATIBLE,
                dependencies=['zstandard'],
                performance_multiplier=3.0,
                memory_footprint_mb=30,
                initialization_time=0.08,
                stability_score=0.95,
                steam_deck_optimized=False,
                factory_function=self._create_zstandard_compatible
            ),
            FallbackImplementation(
                name='gzip_optimized',
                tier=FallbackTier.EFFICIENT,
                dependencies=[],
                performance_multiplier=2.0,
                memory_footprint_mb=15,
                initialization_time=0.02,
                stability_score=0.99,
                steam_deck_optimized=True,
                factory_function=self._create_gzip_optimized
            ),
            FallbackImplementation(
                name='gzip_standard',
                tier=FallbackTier.PURE_PYTHON,
                dependencies=[],
                performance_multiplier=1.0,
                memory_footprint_mb=10,
                initialization_time=0.01,
                stability_score=1.0,
                steam_deck_optimized=True,
                factory_function=self._create_gzip_standard
            )
        ]

    def _setup_initial_fallbacks(self) -> None:
        """Setup initial fallback configurations based on available dependencies"""
        logger.info("Setting up initial fallback configurations...")
        
        for component_name, implementations in self.fallback_implementations.items():
            try:
                best_impl = self._select_best_implementation(component_name, implementations)
                if best_impl:
                    success = self._activate_implementation(component_name, best_impl)
                    if success:
                        logger.info(f"Activated {best_impl.name} for {component_name} (Tier {best_impl.tier.value})")
                    else:
                        logger.warning(f"Failed to activate {best_impl.name} for {component_name}")
                        # Try next best implementation
                        self._try_fallback_implementation(component_name, implementations)
                else:
                    logger.error(f"No suitable implementation found for {component_name}")
            except Exception as e:
                logger.error(f"Error setting up fallback for {component_name}: {e}")

    def _select_best_implementation(self, 
                                  component_name: str, 
                                  implementations: List[FallbackImplementation]) -> Optional[FallbackImplementation]:
        """Select the best available implementation for a component"""
        
        # Sort by tier (optimal first) then by performance multiplier
        candidates = sorted(implementations, key=lambda x: (x.tier.value, -x.performance_multiplier))
        
        for impl in candidates:
            try:
                # Check dependency availability
                if self._check_implementation_dependencies(impl):
                    # Check Steam Deck compatibility if applicable
                    if self.is_steam_deck and not impl.steam_deck_optimized:
                        continue
                    
                    # Check memory constraints
                    if impl.memory_footprint_mb > self.memory_limit_mb:
                        continue
                    
                    # Check if implementation can be created
                    if self._can_create_implementation(impl):
                        return impl
                        
            except Exception as e:
                logger.debug(f"Implementation {impl.name} failed check: {e}")
                continue
        
        return None

    def _check_implementation_dependencies(self, impl: FallbackImplementation) -> bool:
        """Check if implementation dependencies are available"""
        if not impl.dependencies:
            return True  # Pure Python implementation
        
        for dep in impl.dependencies:
            # Check if dependency is available in detector results
            if dep in self.detector.detection_results:
                result = self.detector.detection_results[dep]
                if not (result.status.available or result.status.fallback_active):
                    return False
            else:
                # Try quick detection
                try:
                    results = self.detector.detect_all_dependencies(dependencies=[dep])
                    if dep in results:
                        result = results[dep]
                        if not (result.status.available or result.status.fallback_active):
                            return False
                    else:
                        return False
                except Exception:
                    return False
        
        return True

    def _can_create_implementation(self, impl: FallbackImplementation) -> bool:
        """Check if implementation can be instantiated"""
        if not impl.factory_function:
            return False
        
        try:
            # Try creating the implementation (with timeout)
            test_instance = impl.factory_function()
            if test_instance is None:
                return False
            
            # Basic functionality test if available
            if impl.test_function:
                return impl.test_function(test_instance)
            
            return True
            
        except Exception as e:
            logger.debug(f"Implementation {impl.name} creation test failed: {e}")
            return False

    def _activate_implementation(self, component_name: str, impl: FallbackImplementation) -> bool:
        """Activate a specific implementation for a component"""
        with self.fallback_lock:
            try:
                # Create the implementation instance
                start_time = time.time()
                instance = impl.factory_function()
                if instance is None:
                    return False
                
                initialization_time = time.time() - start_time
                
                # Store in cache
                self.fallback_cache[component_name] = instance
                
                # Update status
                self.active_fallbacks[component_name] = FallbackStatus(
                    active_tier=impl.tier,
                    active_implementation=impl.name,
                    reason=FallbackReason.DEPENDENCY_MISSING if impl.tier != FallbackTier.OPTIMAL else FallbackReason.USER_PREFERENCE,
                    performance_impact=1.0 / impl.performance_multiplier,
                    fallback_chain=[impl.name],
                    last_switch_time=time.time(),
                    switch_count=1,
                    error_count=0,
                    is_stable=True
                )
                
                # Record switch
                self.switch_history.append({
                    'timestamp': time.time(),
                    'component': component_name,
                    'from_implementation': None,
                    'to_implementation': impl.name,
                    'tier': impl.tier.value,
                    'reason': 'initial_setup',
                    'initialization_time': initialization_time
                })
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to activate {impl.name} for {component_name}: {e}")
                return False

    def _try_fallback_implementation(self, component_name: str, implementations: List[FallbackImplementation]) -> bool:
        """Try to activate the next best fallback implementation"""
        current_status = self.active_fallbacks.get(component_name)
        current_tier = current_status.active_tier if current_status else FallbackTier.OPTIMAL
        
        # Find implementations with lower tier (higher number = lower tier)
        fallback_candidates = [
            impl for impl in implementations
            if impl.tier.value > current_tier.value
        ]
        
        # Sort by tier and performance
        fallback_candidates.sort(key=lambda x: (x.tier.value, -x.performance_multiplier))
        
        for impl in fallback_candidates:
            if self._check_implementation_dependencies(impl):
                if self._activate_implementation(component_name, impl):
                    logger.info(f"Successfully fell back to {impl.name} for {component_name}")
                    return True
        
        logger.error(f"No fallback implementation available for {component_name}")
        return False

    def get_implementation(self, component_name: str, component_type: Optional[str] = None) -> Any:
        """Get the active implementation for a component"""
        if component_name in self.fallback_cache:
            return self.fallback_cache[component_name]
        
        # Try to activate an implementation
        if component_name in self.fallback_implementations:
            implementations = self.fallback_implementations[component_name]
            best_impl = self._select_best_implementation(component_name, implementations)
            if best_impl and self._activate_implementation(component_name, best_impl):
                return self.fallback_cache[component_name]
        
        # Return None if no implementation available
        logger.warning(f"No implementation available for {component_name}")
        return None

    def switch_to_tier(self, component_name: str, target_tier: FallbackTier, reason: FallbackReason = FallbackReason.USER_PREFERENCE) -> bool:
        """Switch a component to a specific fallback tier"""
        if component_name not in self.fallback_implementations:
            logger.error(f"Unknown component: {component_name}")
            return False
        
        implementations = self.fallback_implementations[component_name]
        target_impl = None
        
        # Find implementation for target tier
        for impl in implementations:
            if impl.tier == target_tier:
                if self._check_implementation_dependencies(impl):
                    target_impl = impl
                    break
        
        if not target_impl:
            logger.error(f"No available implementation for tier {target_tier} in {component_name}")
            return False
        
        # Record current state for rollback if needed
        current_status = self.active_fallbacks.get(component_name)
        current_implementation = self.fallback_cache.get(component_name)
        
        try:
            # Activate new implementation
            success = self._activate_implementation(component_name, target_impl)
            if success:
                # Update fallback reason
                if component_name in self.active_fallbacks:
                    self.active_fallbacks[component_name].reason = reason
                
                logger.info(f"Successfully switched {component_name} to tier {target_tier} ({target_impl.name})")
                return True
            else:
                logger.error(f"Failed to activate {target_impl.name} for {component_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching {component_name} to tier {target_tier}: {e}")
            
            # Attempt rollback
            if current_status and current_implementation:
                try:
                    self.active_fallbacks[component_name] = current_status
                    self.fallback_cache[component_name] = current_implementation
                    logger.info(f"Rolled back {component_name} to previous implementation")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed for {component_name}: {rollback_error}")
            
            return False

    def auto_optimize_fallbacks(self) -> Dict[str, Any]:
        """Automatically optimize fallback configurations based on current conditions"""
        optimization_results = {
            'optimizations_applied': [],
            'performance_improvements': {},
            'warnings': [],
            'recommendations': []
        }
        
        logger.info("Starting automatic fallback optimization...")
        
        # Check system conditions
        memory_usage = self._get_memory_usage()
        cpu_temp = self._get_cpu_temperature()
        
        for component_name, status in self.active_fallbacks.items():
            try:
                # Check if we can upgrade to a better tier
                if status.active_tier.value > 1:  # Not already optimal
                    better_impl = self._find_better_implementation(component_name, status.active_tier)
                    if better_impl:
                        # Check if system can handle the upgrade
                        if memory_usage + better_impl.memory_footprint_mb < self.memory_limit_mb:
                            if self.switch_to_tier(component_name, better_impl.tier, FallbackReason.PERFORMANCE_DEGRADATION):
                                optimization_results['optimizations_applied'].append(
                                    f"Upgraded {component_name} to {better_impl.name}"
                                )
                                performance_gain = better_impl.performance_multiplier / status.performance_impact
                                optimization_results['performance_improvements'][component_name] = performance_gain
                
                # Check if we need to downgrade due to system constraints
                elif memory_usage > self.memory_limit_mb * 0.8:  # High memory usage
                    worse_impl = self._find_lower_tier_implementation(component_name, status.active_tier)
                    if worse_impl:
                        if self.switch_to_tier(component_name, worse_impl.tier, FallbackReason.MEMORY_PRESSURE):
                            optimization_results['optimizations_applied'].append(
                                f"Downgraded {component_name} to {worse_impl.name} due to memory pressure"
                            )
                
                # Steam Deck specific optimizations
                elif self.is_steam_deck and cpu_temp > 70.0:  # High temperature
                    if not self._is_steam_deck_optimized(component_name):
                        steam_impl = self._find_steam_deck_implementation(component_name)
                        if steam_impl:
                            if self.switch_to_tier(component_name, steam_impl.tier, FallbackReason.THERMAL_THROTTLING):
                                optimization_results['optimizations_applied'].append(
                                    f"Switched {component_name} to Steam Deck optimized implementation"
                                )
                
            except Exception as e:
                optimization_results['warnings'].append(f"Failed to optimize {component_name}: {e}")
        
        # Generate general recommendations
        if memory_usage > self.memory_limit_mb * 0.9:
            optimization_results['recommendations'].append(
                "High memory usage detected - consider switching to more efficient implementations"
            )
        
        if self.is_steam_deck and cpu_temp > 80.0:
            optimization_results['recommendations'].append(
                "High temperature on Steam Deck - consider reducing computational load"
            )
        
        logger.info(f"Auto-optimization completed: {len(optimization_results['optimizations_applied'])} changes applied")
        return optimization_results

    def _find_better_implementation(self, component_name: str, current_tier: FallbackTier) -> Optional[FallbackImplementation]:
        """Find a better implementation than current tier"""
        if component_name not in self.fallback_implementations:
            return None
        
        implementations = self.fallback_implementations[component_name]
        
        # Look for implementations with better (lower) tier number
        for impl in implementations:
            if (impl.tier.value < current_tier.value and 
                self._check_implementation_dependencies(impl)):
                return impl
        
        return None

    def _find_lower_tier_implementation(self, component_name: str, current_tier: FallbackTier) -> Optional[FallbackImplementation]:
        """Find a lower tier implementation for resource conservation"""
        if component_name not in self.fallback_implementations:
            return None
        
        implementations = self.fallback_implementations[component_name]
        
        # Look for implementations with higher tier number (lower performance)
        for impl in implementations:
            if (impl.tier.value > current_tier.value and
                self._check_implementation_dependencies(impl)):
                return impl
        
        return None

    def _find_steam_deck_implementation(self, component_name: str) -> Optional[FallbackImplementation]:
        """Find Steam Deck optimized implementation"""
        if component_name not in self.fallback_implementations:
            return None
        
        implementations = self.fallback_implementations[component_name]
        
        # Look for Steam Deck optimized implementations
        for impl in implementations:
            if (impl.steam_deck_optimized and
                self._check_implementation_dependencies(impl)):
                return impl
        
        return None

    def _is_steam_deck_optimized(self, component_name: str) -> bool:
        """Check if current implementation is Steam Deck optimized"""
        if component_name not in self.active_fallbacks:
            return False
        
        status = self.active_fallbacks[component_name]
        implementations = self.fallback_implementations[component_name]
        
        for impl in implementations:
            if impl.name == status.active_implementation:
                return impl.steam_deck_optimized
        
        return False

    def get_fallback_status(self) -> Dict[str, Any]:
        """Get comprehensive fallback status report"""
        total_components = len(self.fallback_implementations)
        active_components = len(self.active_fallbacks)
        
        tier_distribution = defaultdict(int)
        performance_scores = []
        memory_usage = 0
        
        for component_name, status in self.active_fallbacks.items():
            tier_distribution[status.active_tier.name] += 1
            performance_scores.append(1.0 / status.performance_impact)
            
            # Add memory usage
            implementations = self.fallback_implementations[component_name]
            for impl in implementations:
                if impl.name == status.active_implementation:
                    memory_usage += impl.memory_footprint_mb
                    break
        
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
        
        return {
            'total_components': total_components,
            'active_components': active_components,
            'tier_distribution': dict(tier_distribution),
            'average_performance_multiplier': avg_performance,
            'total_memory_usage_mb': memory_usage,
            'system_health': {
                'memory_pressure': memory_usage / self.memory_limit_mb,
                'fallback_stability': self._calculate_fallback_stability(),
                'steam_deck_optimization': self._calculate_steam_deck_optimization()
            },
            'recent_switches': len([s for s in self.switch_history if time.time() - s['timestamp'] < 300]),
            'total_switches': len(self.switch_history)
        }

    def _calculate_fallback_stability(self) -> float:
        """Calculate overall fallback stability score"""
        if not self.active_fallbacks:
            return 1.0
        
        stability_scores = []
        for component_name, status in self.active_fallbacks.items():
            # Factor in error count and switch frequency
            error_penalty = min(status.error_count * 0.1, 0.5)
            switch_penalty = min(status.switch_count * 0.05, 0.3)
            
            component_stability = 1.0 - error_penalty - switch_penalty
            stability_scores.append(max(0.0, component_stability))
        
        return sum(stability_scores) / len(stability_scores)

    def _calculate_steam_deck_optimization(self) -> float:
        """Calculate Steam Deck optimization score"""
        if not self.is_steam_deck:
            return 1.0  # N/A for non-Steam Deck systems
        
        if not self.active_fallbacks:
            return 0.0
        
        optimized_count = sum(
            1 for component_name in self.active_fallbacks.keys()
            if self._is_steam_deck_optimized(component_name)
        )
        
        return optimized_count / len(self.active_fallbacks)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature"""
        try:
            from .pure_python_fallbacks import PureThermalMonitor
            monitor = PureThermalMonitor()
            return monitor.get_cpu_temperature()
        except Exception:
            return 50.0  # Default safe temperature

    def export_fallback_configuration(self, filepath: Path) -> None:
        """Export current fallback configuration"""
        config = {
            'system_info': self.version_manager.system_info,
            'fallback_status': self.get_fallback_status(),
            'active_configurations': {
                name: {
                    'active_tier': status.active_tier.name,
                    'active_implementation': status.active_implementation,
                    'reason': status.reason.name,
                    'performance_impact': status.performance_impact,
                    'switch_count': status.switch_count,
                    'is_stable': status.is_stable
                }
                for name, status in self.active_fallbacks.items()
            },
            'switch_history': self.switch_history[-20:],  # Last 20 switches
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Fallback configuration exported to {filepath}")

    # =============================================================================
    # IMPLEMENTATION FACTORY FUNCTIONS
    # =============================================================================
    
    def _create_numpy_optimized_math(self):
        """Create NumPy + Numba optimized math implementation"""
        try:
            import numpy as np
            from numba import njit
            
            class NumPyOptimizedMath:
                def __init__(self):
                    self.np = np
                    self.njit = njit
                
                @njit
                def fast_mean(self, arr):
                    return np.mean(arr)
                
                @njit  
                def fast_std(self, arr):
                    return np.std(arr)
                
                def array(self, data):
                    return self.np.array(data)
                
                def mean(self, arr):
                    if hasattr(arr, '__array__'):
                        return self.fast_mean(arr)
                    return self.np.mean(arr)
                
                def std(self, arr):
                    if hasattr(arr, '__array__'):
                        return self.fast_std(arr)
                    return self.np.std(arr)
            
            return NumPyOptimizedMath()
            
        except ImportError:
            return None

    def _create_numpy_standard_math(self):
        """Create standard NumPy math implementation"""
        try:
            import numpy as np
            
            class NumPyStandardMath:
                def __init__(self):
                    self.np = np
                
                def array(self, data):
                    return self.np.array(data)
                
                def mean(self, arr):
                    return self.np.mean(arr)
                
                def std(self, arr):
                    return self.np.std(arr)
                
                def zeros(self, shape):
                    return self.np.zeros(shape)
                
                def maximum(self, a, b):
                    return self.np.maximum(a, b)
            
            return NumPyStandardMath()
            
        except ImportError:
            return None

    def _create_numpy_minimal_math(self):
        """Create minimal NumPy math implementation"""
        try:
            import numpy as np
            
            class NumPyMinimalMath:
                def __init__(self):
                    self.np = np
                
                def array(self, data):
                    return self.np.asarray(data)  # More memory efficient
                
                def mean(self, arr):
                    return float(self.np.mean(arr))
                
                def std(self, arr):
                    return float(self.np.std(arr))
            
            return NumPyMinimalMath()
            
        except ImportError:
            return None

    def _create_pure_python_math(self):
        """Create pure Python math implementation"""
        from .pure_python_fallbacks import PureArrayMath
        return PureArrayMath()

    def _create_lightgbm_optimized(self):
        """Create optimized LightGBM implementation"""
        try:
            import lightgbm as lgb
            import numpy as np
            
            class LightGBMOptimized:
                def __init__(self):
                    self.lgb = lgb
                    self.np = np
                
                def create_regressor(self, **kwargs):
                    # Optimized parameters for Steam Deck
                    params = {
                        'n_estimators': kwargs.get('n_estimators', 100),
                        'num_leaves': kwargs.get('num_leaves', 31),
                        'learning_rate': kwargs.get('learning_rate', 0.1),
                        'feature_fraction': kwargs.get('feature_fraction', 0.8),
                        'bagging_fraction': kwargs.get('bagging_fraction', 0.8),
                        'bagging_freq': kwargs.get('bagging_freq', 5),
                        'verbose': -1,
                        'num_threads': min(4, os.cpu_count() or 4)
                    }
                    return self.lgb.LGBMRegressor(**params)
            
            return LightGBMOptimized()
            
        except ImportError:
            return None

    def _create_sklearn_randomforest(self):
        """Create scikit-learn RandomForest implementation"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            class SklearnRandomForest:
                def __init__(self):
                    self.RandomForestRegressor = RandomForestRegressor
                
                def create_regressor(self, **kwargs):
                    params = {
                        'n_estimators': kwargs.get('n_estimators', 50),
                        'max_depth': kwargs.get('max_depth', 10),
                        'random_state': kwargs.get('random_state', 42),
                        'n_jobs': min(2, os.cpu_count() or 2)  # Conservative for Steam Deck
                    }
                    return self.RandomForestRegressor(**params)
            
            return SklearnRandomForest()
            
        except ImportError:
            return None

    def _create_sklearn_linear(self):
        """Create scikit-learn linear model implementation"""
        try:
            from sklearn.linear_model import LinearRegression
            
            class SklearnLinear:
                def __init__(self):
                    self.LinearRegression = LinearRegression
                
                def create_regressor(self, **kwargs):
                    return self.LinearRegression()
            
            return SklearnLinear()
            
        except ImportError:
            return None

    def _create_pure_python_ml(self):
        """Create pure Python ML implementation"""
        from .pure_python_fallbacks import PureLinearRegressor
        return PureLinearRegressor()

    def _create_psutil_full(self):
        """Create full psutil implementation"""
        try:
            import psutil
            return psutil
        except ImportError:
            return None

    def _create_psutil_minimal(self):
        """Create minimal psutil wrapper"""
        try:
            import psutil
            
            class PsutilMinimal:
                def __init__(self):
                    self.psutil = psutil
                
                def cpu_count(self):
                    return self.psutil.cpu_count()
                
                def memory_info(self):
                    return self.psutil.virtual_memory()
                
                def Process(self, pid=None):
                    return self.psutil.Process(pid)
            
            return PsutilMinimal()
            
        except ImportError:
            return None

    def _create_system_proc(self):
        """Create /proc-based system monitor"""
        class SystemProcMonitor:
            def cpu_count(self):
                return os.cpu_count() or 4
            
            def memory_info(self):
                try:
                    with open('/proc/meminfo', 'r') as f:
                        lines = f.readlines()
                    
                    mem_total = 0
                    mem_available = 0
                    
                    for line in lines:
                        if line.startswith('MemTotal:'):
                            mem_total = int(line.split()[1]) * 1024
                        elif line.startswith('MemAvailable:'):
                            mem_available = int(line.split()[1]) * 1024
                    
                    class MemInfo:
                        def __init__(self):
                            self.total = mem_total
                            self.available = mem_available
                            self.percent = ((mem_total - mem_available) / mem_total) * 100
                    
                    return MemInfo()
                    
                except Exception:
                    class FallbackMemInfo:
                        def __init__(self):
                            self.total = 8 * 1024 * 1024 * 1024  # 8GB assumption
                            self.available = self.total // 2
                            self.percent = 50.0
                    
                    return FallbackMemInfo()
        
        return SystemProcMonitor()

    def _create_pure_system_monitor(self):
        """Create pure Python system monitor"""
        from .pure_python_fallbacks import PureSystemMonitor
        return PureSystemMonitor()

    def _create_msgpack_optimized(self):
        """Create optimized msgpack implementation"""
        try:
            import msgpack
            
            class MsgPackOptimized:
                def __init__(self):
                    self.msgpack = msgpack
                
                def packb(self, obj):
                    return self.msgpack.packb(obj, use_bin_type=True)
                
                def unpackb(self, data):
                    return self.msgpack.unpackb(data, raw=False)
            
            return MsgPackOptimized()
            
        except ImportError:
            return None

    def _create_msgpack_standard(self):
        """Create standard msgpack implementation"""
        try:
            import msgpack
            return msgpack
        except ImportError:
            return None

    def _create_json_compressed(self):
        """Create compressed JSON implementation"""
        import json
        import gzip
        
        class JsonCompressed:
            def packb(self, obj):
                json_str = json.dumps(obj, separators=(',', ':'))
                return gzip.compress(json_str.encode('utf-8'))
            
            def unpackb(self, data):
                json_str = gzip.decompress(data).decode('utf-8')
                return json.loads(json_str)
        
        return JsonCompressed()

    def _create_pure_json(self):
        """Create pure JSON implementation"""
        from .pure_python_fallbacks import PureSerializer
        return PureSerializer()

    def _create_zstandard_fast(self):
        """Create fast zstandard implementation"""
        try:
            import zstandard as zstd
            
            class ZstandardFast:
                def __init__(self):
                    self.compressor = zstd.ZstdCompressor(level=1)
                    self.decompressor = zstd.ZstdDecompressor()
                
                def compress(self, data):
                    return self.compressor.compress(data)
                
                def decompress(self, data):
                    return self.decompressor.decompress(data)
            
            return ZstandardFast()
            
        except ImportError:
            return None

    def _create_zstandard_compatible(self):
        """Create compatible zstandard implementation"""
        try:
            import zstandard as zstd
            
            class ZstandardCompatible:
                def __init__(self):
                    self.compressor = zstd.ZstdCompressor(level=3)
                    self.decompressor = zstd.ZstdDecompressor()
                
                def compress(self, data):
                    return self.compressor.compress(data)
                
                def decompress(self, data):
                    return self.decompressor.decompress(data)
            
            return ZstandardCompatible()
            
        except ImportError:
            return None

    def _create_gzip_optimized(self):
        """Create optimized gzip implementation"""
        import gzip
        
        class GzipOptimized:
            def compress(self, data, compresslevel=6):
                return gzip.compress(data, compresslevel=compresslevel)
            
            def decompress(self, data):
                return gzip.decompress(data)
        
        return GzipOptimized()

    def _create_gzip_standard(self):
        """Create standard gzip implementation"""
        from .pure_python_fallbacks import PureCompression
        return PureCompression()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_fallback_system: Optional[TieredFallbackSystem] = None

def get_fallback_system() -> TieredFallbackSystem:
    """Get or create global fallback system instance"""
    global _fallback_system
    if _fallback_system is None:
        _fallback_system = TieredFallbackSystem()
    return _fallback_system

def get_implementation(component_name: str) -> Any:
    """Get active implementation for a component"""
    system = get_fallback_system()
    return system.get_implementation(component_name)

def switch_to_performance_mode() -> Dict[str, bool]:
    """Switch all components to highest performance tier"""
    system = get_fallback_system()
    results = {}
    
    for component_name in system.fallback_implementations.keys():
        results[component_name] = system.switch_to_tier(
            component_name, 
            FallbackTier.OPTIMAL,
            FallbackReason.USER_PREFERENCE
        )
    
    return results

def switch_to_efficiency_mode() -> Dict[str, bool]:
    """Switch all components to efficient tier"""
    system = get_fallback_system()
    results = {}
    
    for component_name in system.fallback_implementations.keys():
        results[component_name] = system.switch_to_tier(
            component_name, 
            FallbackTier.EFFICIENT,
            FallbackReason.USER_PREFERENCE
        )
    
    return results

def optimize_for_steam_deck() -> Dict[str, Any]:
    """Optimize all fallbacks for Steam Deck"""
    system = get_fallback_system()
    return system.auto_optimize_fallbacks()


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🔄 Tiered Fallback System Test Suite")
    print("=" * 55)
    
    # Initialize system
    system = TieredFallbackSystem()
    
    print("\n📊 System Configuration:")
    print(f"  Steam Deck: {system.is_steam_deck}")
    print(f"  Auto Fallback: {system.auto_fallback}")
    print(f"  Memory Limit: {system.memory_limit_mb}MB")
    
    # Show available implementations
    print(f"\n🔧 Available Component Types:")
    for component_name, implementations in system.fallback_implementations.items():
        print(f"  {component_name}: {len(implementations)} implementations")
        for impl in implementations:
            tier_emoji = {1: "🚀", 2: "⚡", 3: "💡", 4: "🐌"}
            print(f"    {tier_emoji.get(impl.tier.value, '❓')} {impl.name} (Tier {impl.tier.value}, {impl.performance_multiplier}x)")
    
    # Test getting implementations
    print(f"\n🧪 Testing Implementation Retrieval:")
    test_components = ['array_math', 'ml_algorithms', 'system_monitor']
    
    for component in test_components:
        impl = system.get_implementation(component)
        if impl:
            status = system.active_fallbacks.get(component)
            tier_name = status.active_tier.name if status else "Unknown"
            impl_name = status.active_implementation if status else "Unknown"
            print(f"  ✅ {component}: {impl_name} (Tier: {tier_name})")
        else:
            print(f"  ❌ {component}: No implementation available")
    
    # Test tier switching
    print(f"\n🔄 Testing Tier Switching:")
    if 'array_math' in system.active_fallbacks:
        original_tier = system.active_fallbacks['array_math'].active_tier
        print(f"  Original tier: {original_tier.name}")
        
        # Try switching to pure Python
        success = system.switch_to_tier('array_math', FallbackTier.PURE_PYTHON)
        print(f"  Switch to Pure Python: {'✅' if success else '❌'}")
        
        if success:
            current_tier = system.active_fallbacks['array_math'].active_tier
            print(f"  New tier: {current_tier.name}")
    
    # Test auto optimization
    print(f"\n⚡ Testing Auto Optimization:")
    optimization_result = system.auto_optimize_fallbacks()
    print(f"  Optimizations applied: {len(optimization_result['optimizations_applied'])}")
    
    for opt in optimization_result['optimizations_applied']:
        print(f"    ✅ {opt}")
    
    if optimization_result['warnings']:
        print("  Warnings:")
        for warning in optimization_result['warnings']:
            print(f"    ⚠️  {warning}")
    
    # Get status report
    print(f"\n📈 Fallback Status Report:")
    status = system.get_fallback_status()
    print(f"  Active Components: {status['active_components']}/{status['total_components']}")
    print(f"  Average Performance: {status['average_performance_multiplier']:.1f}x")
    print(f"  Memory Usage: {status['total_memory_usage_mb']}MB")
    print(f"  System Health: {status['system_health']['fallback_stability']:.1%}")
    
    if system.is_steam_deck:
        print(f"  Steam Deck Optimization: {status['system_health']['steam_deck_optimization']:.1%}")
    
    # Export configuration
    config_path = Path("/tmp/tiered_fallback_config.json")
    system.export_fallback_configuration(config_path)
    print(f"\n💾 Configuration exported to {config_path}")
    
    print(f"\n✅ Tiered Fallback System test completed!")
    print(f"🎯 System ready with {len(system.active_fallbacks)} active fallbacks")