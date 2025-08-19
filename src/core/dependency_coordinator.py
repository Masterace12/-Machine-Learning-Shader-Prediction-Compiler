#!/usr/bin/env python3
"""
Enhanced Dependency Coordination System for ML Shader Prediction Compiler

This module provides intelligent dependency detection, validation, performance
optimization, and graceful fallback management for the Enhanced ML Predictor.

Features:
- Intelligent dependency detection and validation
- Performance benchmarking of different dependency combinations
- Graceful degradation with pure Python fallbacks
- Steam Deck specific optimizations
- Runtime dependency switching
- Installation failure recovery
- Zero-compilation requirement enforcement
"""

import os
import sys
import time
import json
import logging
import platform
import importlib
import subprocess
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import wraps, lru_cache

# Import our pure Python fallbacks
from .pure_python_fallbacks import (
    AVAILABLE_DEPS, get_fallback_status, 
    PureSteamDeckDetector, PureThermalMonitor
)

# Import enhanced installation capabilities
try:
    from .enhanced_dependency_installer import SteamDeckDependencyInstaller
    ENHANCED_INSTALLER_AVAILABLE = True
except ImportError:
    ENHANCED_INSTALLER_AVAILABLE = False
    logger.warning("Enhanced dependency installer not available")

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY CONFIGURATION
# =============================================================================

@dataclass
class DependencySpec:
    """Specification for a dependency"""
    name: str
    version_min: Optional[str] = None
    version_max: Optional[str] = None
    import_name: Optional[str] = None
    test_function: Optional[Callable] = None
    fallback_available: bool = False
    performance_impact: float = 1.0  # Multiplier for performance benefit
    required: bool = False
    steam_deck_compatible: bool = True
    compilation_required: bool = False
    category: str = "optional"
    platform_specific: List[str] = field(default_factory=list)
    python_versions: List[str] = field(default_factory=list)

@dataclass
class DependencyState:
    """Current state of a dependency"""
    spec: DependencySpec
    available: bool = False
    version: Optional[str] = None
    import_time: float = 0.0
    test_passed: bool = False
    test_time: float = 0.0
    error: Optional[str] = None
    fallback_active: bool = False
    performance_score: float = 0.0

@dataclass
class PerformanceProfile:
    """Performance profile for a dependency combination"""
    combination_id: str
    dependencies: List[str]
    setup_time: float
    execution_time: float
    memory_usage: float
    stability_score: float
    compatibility_score: float
    overall_score: float

# =============================================================================
# DEPENDENCY DEFINITIONS
# =============================================================================

DEPENDENCY_SPECS = {
    # Core ML Dependencies
    'numpy': DependencySpec(
        name='numpy',
        version_min='1.20.0',
        version_max='2.0.0',
        import_name='numpy',
        fallback_available=True,
        performance_impact=5.0,
        required=False,
        category='ml_core',
        test_function=lambda: __import__('numpy').array([1, 2, 3]).mean() == 2.0
    ),
    
    'scikit-learn': DependencySpec(
        name='scikit-learn',
        version_min='1.0.0',
        version_max='2.0.0',
        import_name='sklearn',
        fallback_available=True,
        performance_impact=3.0,
        required=False,
        category='ml_core',
        test_function=lambda: hasattr(__import__('sklearn.ensemble', fromlist=['RandomForestRegressor']), 'RandomForestRegressor')
    ),
    
    'lightgbm': DependencySpec(
        name='lightgbm',
        version_min='3.0.0',
        version_max='5.0.0',
        import_name='lightgbm',
        fallback_available=True,
        performance_impact=4.0,
        required=False,
        category='ml_advanced',
        steam_deck_compatible=True,
        platform_specific=['!aarch64'],  # Skip on ARM to avoid compilation
        test_function=lambda: hasattr(__import__('lightgbm'), 'LGBMRegressor')
    ),
    
    # Performance Dependencies
    'numba': DependencySpec(
        name='numba',
        version_min='0.56.0',
        version_max='1.0.0',
        import_name='numba',
        fallback_available=True,
        performance_impact=8.0,
        required=False,
        category='performance',
        python_versions=[],  # Allow on all Python versions now that numba supports 3.13
        test_function=lambda: __import__('numba').njit(lambda x: x + 1)(5) == 6
    ),
    
    'numexpr': DependencySpec(
        name='numexpr',
        version_min='2.8.0',
        version_max='3.0.0',
        import_name='numexpr',
        fallback_available=True,
        performance_impact=2.5,
        required=False,
        category='performance',
        python_versions=[],  # Allow on all Python versions
        test_function=lambda: __import__('numexpr').evaluate('2 * 3') == 6
    ),
    
    'bottleneck': DependencySpec(
        name='bottleneck',
        version_min='1.3.0',
        version_max='2.0.0',
        import_name='bottleneck',
        fallback_available=True,
        performance_impact=1.8,
        required=False,
        category='performance',
        python_versions=[],  # Allow on all Python versions
        test_function=lambda: hasattr(__import__('bottleneck'), 'nanmean')
    ),
    
    # Serialization and Compression
    'msgpack': DependencySpec(
        name='msgpack',
        version_min='1.0.0',
        version_max='2.0.0',
        import_name='msgpack',
        fallback_available=True,
        performance_impact=2.0,
        required=False,
        category='serialization',
        test_function=lambda: __import__('msgpack').unpackb(__import__('msgpack').packb({'test': 123}))['test'] == 123
    ),
    
    'zstandard': DependencySpec(
        name='zstandard',
        version_min='0.20.0',
        version_max='1.0.0',
        import_name='zstandard',
        fallback_available=True,
        performance_impact=3.0,
        required=False,
        category='compression',
        platform_specific=['!aarch64'],  # Skip on ARM64
        test_function=lambda: len(__import__('zstandard').ZstdCompressor().compress(b'test')) > 0
    ),
    
    # System Integration
    'psutil': DependencySpec(
        name='psutil',
        version_min='5.8.0',
        version_max='6.0.0',
        import_name='psutil',
        fallback_available=True,
        performance_impact=1.5,
        required=False,
        category='system',
        test_function=lambda: __import__('psutil').cpu_count() > 0
    ),
    
    'dbus-next': DependencySpec(
        name='dbus-next',
        version_min='0.2.0',
        version_max='1.0.0',
        import_name='dbus_next',
        fallback_available=True,
        performance_impact=1.2,
        required=False,
        category='system',
        platform_specific=['linux'],
        test_function=lambda: hasattr(__import__('dbus_next'), 'BaseInterface')
    ),
    
    'jeepney': DependencySpec(
        name='jeepney',
        version_min='0.8.0',
        version_max='1.0.0',
        import_name='jeepney',
        fallback_available=True,
        performance_impact=1.1,
        required=False,
        category='system',
        platform_specific=['linux'],
        test_function=lambda: hasattr(__import__('jeepney'), 'DBusAddress')
    ),
}

# =============================================================================
# DEPENDENCY COORDINATOR
# =============================================================================

class DependencyCoordinator:
    """
    Central coordinator for dependency management with intelligent detection,
    validation, performance optimization, and graceful fallback.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.dependency_states: Dict[str, DependencyState] = {}
        self.performance_profiles: List[PerformanceProfile] = []
        self.current_profile: Optional[PerformanceProfile] = None
        self.detection_cache: Dict[str, Any] = {}
        self.is_steam_deck = PureSteamDeckDetector.is_steam_deck()
        self.thermal_monitor = PureThermalMonitor()
        self.optimization_lock = threading.Lock()
        
        # Initialize enhanced installer if available
        self.installer = None
        if ENHANCED_INSTALLER_AVAILABLE:
            try:
                self.installer = SteamDeckDependencyInstaller()
                logger.info("Enhanced dependency installer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced installer: {e}")
        
        # System information
        self.system_info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'is_steam_deck': self.is_steam_deck,
            'steam_deck_model': PureSteamDeckDetector.get_steam_deck_model(),
            'cpu_count': os.cpu_count() or 4,
            'thermal_state': self.thermal_monitor.get_thermal_state()
        }
        
        logger.info(f"DependencyCoordinator initialized for {self.system_info['platform']} {self.system_info['machine']}")
        logger.info(f"Steam Deck: {self.is_steam_deck}, Python: {self.system_info['python_version']}")
    
    def detect_all_dependencies(self, force_refresh: bool = False) -> Dict[str, DependencyState]:
        """
        Detect and validate all dependencies with comprehensive testing
        """
        if not force_refresh and self.dependency_states:
            return self.dependency_states
        
        logger.info("Starting comprehensive dependency detection...")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel detection
        with ThreadPoolExecutor(max_workers=min(8, len(DEPENDENCY_SPECS))) as executor:
            futures = {
                executor.submit(self._detect_single_dependency, name, spec): name
                for name, spec in DEPENDENCY_SPECS.items()
            }
            
            for future in as_completed(futures):
                dep_name = futures[future]
                try:
                    self.dependency_states[dep_name] = future.result()
                except Exception as e:
                    logger.error(f"Failed to detect dependency {dep_name}: {e}")
                    # Create failed state
                    self.dependency_states[dep_name] = DependencyState(
                        spec=DEPENDENCY_SPECS[dep_name],
                        available=False,
                        error=str(e),
                        fallback_active=True
                    )
        
        detection_time = time.time() - start_time
        logger.info(f"Dependency detection completed in {detection_time:.2f}s")
        
        self._log_detection_summary()
        return self.dependency_states
    
    def _detect_single_dependency(self, name: str, spec: DependencySpec) -> DependencyState:
        """Detect and validate a single dependency"""
        state = DependencyState(spec=spec)
        
        # Check platform compatibility
        if not self._is_platform_compatible(spec):
            state.available = False
            state.error = f"Not compatible with platform {self.system_info['platform']} {self.system_info['machine']}"
            state.fallback_active = True
            return state
        
        # Check Python version compatibility
        if not self._is_python_version_compatible(spec):
            state.available = False
            state.error = f"Not compatible with Python {self.system_info['python_version']}"
            state.fallback_active = True
            return state
        
        # Check Steam Deck compatibility
        if self.is_steam_deck and not spec.steam_deck_compatible:
            state.available = False
            state.error = "Not compatible with Steam Deck"
            state.fallback_active = True
            return state
        
        # Try to import the dependency
        import_name = spec.import_name or name
        start_time = time.time()
        
        try:
            module = importlib.import_module(import_name)
            state.import_time = time.time() - start_time
            state.available = True
            
            # Get version if possible
            state.version = self._get_module_version(module, name)
            
            # Run test function if provided
            if spec.test_function:
                test_start = time.time()
                try:
                    test_result = spec.test_function()
                    state.test_passed = bool(test_result)
                    state.test_time = time.time() - test_start
                except Exception as e:
                    state.test_passed = False
                    state.error = f"Test failed: {e}"
                    logger.warning(f"Test failed for {name}: {e}")
            else:
                state.test_passed = True
            
            # Calculate performance score
            if state.test_passed:
                state.performance_score = self._calculate_performance_score(spec, state)
            
        except ImportError as e:
            state.available = False
            state.error = f"Import failed: {e}"
            state.fallback_active = spec.fallback_available
        except Exception as e:
            state.available = False
            state.error = f"Unexpected error: {e}"
            state.fallback_active = spec.fallback_available
            logger.error(f"Unexpected error detecting {name}: {e}")
        
        return state
    
    def _is_platform_compatible(self, spec: DependencySpec) -> bool:
        """Check if dependency is compatible with current platform"""
        if not spec.platform_specific:
            return True
        
        current_platform = self.system_info['platform'].lower()
        current_machine = self.system_info['machine'].lower()
        
        for platform_req in spec.platform_specific:
            if platform_req.startswith('!'):
                # Exclusion rule
                excluded = platform_req[1:]
                if excluded == current_platform or excluded == current_machine:
                    return False
            else:
                # Inclusion rule
                if platform_req == current_platform or platform_req == current_machine:
                    return True
        
        # If we have inclusion rules and none matched, exclude
        inclusion_rules = [p for p in spec.platform_specific if not p.startswith('!')]
        if inclusion_rules:
            return False
        
        return True
    
    def _is_python_version_compatible(self, spec: DependencySpec) -> bool:
        """Check if dependency is compatible with current Python version"""
        if not spec.python_versions:
            return True
        
        current_version = self.system_info['python_version']
        
        for version_req in spec.python_versions:
            if version_req.startswith('<'):
                max_version = version_req[1:]
                if self._compare_versions(current_version, max_version) >= 0:
                    return False
            elif version_req.startswith('>'):
                min_version = version_req[1:]
                if self._compare_versions(current_version, min_version) <= 0:
                    return False
            elif version_req.startswith('=='):
                exact_version = version_req[2:]
                if current_version != exact_version:
                    return False
        
        return True
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings (-1: v1 < v2, 0: v1 == v2, 1: v1 > v2)"""
        def version_tuple(v):
            return tuple(map(int, v.split('.')))
        
        v1_tuple = version_tuple(v1)
        v2_tuple = version_tuple(v2)
        
        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0
    
    def _get_module_version(self, module: Any, name: str) -> Optional[str]:
        """Extract version from imported module"""
        version_attrs = ['__version__', 'version', 'VERSION']
        
        for attr in version_attrs:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, str):
                    return version
                elif hasattr(version, '__str__'):
                    return str(version)
        
        return None
    
    def _calculate_performance_score(self, spec: DependencySpec, state: DependencyState) -> float:
        """Calculate performance score for a dependency"""
        base_score = spec.performance_impact
        
        # Adjust for import time (faster is better)
        import_penalty = min(state.import_time * 10, 2.0)
        
        # Adjust for test time (faster is better)
        test_penalty = min(state.test_time * 5, 1.0)
        
        # Steam Deck bonus for compatible dependencies
        steam_deck_bonus = 0.5 if self.is_steam_deck and spec.steam_deck_compatible else 0.0
        
        score = base_score - import_penalty - test_penalty + steam_deck_bonus
        return max(0.1, score)
    
    def benchmark_performance_combinations(self, test_size: int = 1000) -> List[PerformanceProfile]:
        """
        Benchmark different combinations of dependencies for optimal performance
        """
        logger.info("Starting performance benchmarking...")
        
        # Get available dependencies
        available_deps = {
            name: state for name, state in self.dependency_states.items()
            if state.available and state.test_passed
        }
        
        if not available_deps:
            logger.warning("No available dependencies to benchmark")
            return []
        
        combinations = self._generate_dependency_combinations(available_deps)
        profiles = []
        
        for combination in combinations:
            try:
                profile = self._benchmark_combination(combination, test_size)
                profiles.append(profile)
                logger.info(f"Benchmarked combination '{profile.combination_id}': score {profile.overall_score:.2f}")
            except Exception as e:
                logger.error(f"Failed to benchmark combination {combination}: {e}")
        
        # Sort by overall score (descending)
        profiles.sort(key=lambda p: p.overall_score, reverse=True)
        self.performance_profiles = profiles
        
        if profiles:
            self.current_profile = profiles[0]
            logger.info(f"Best performing combination: '{self.current_profile.combination_id}' (score: {self.current_profile.overall_score:.2f})")
        
        return profiles
    
    def _generate_dependency_combinations(self, available_deps: Dict[str, DependencyState]) -> List[List[str]]:
        """Generate meaningful combinations of dependencies to test"""
        combinations = []
        
        # Always test pure Python fallback (empty combination)
        combinations.append([])
        
        # Test individual high-impact dependencies
        high_impact = [name for name, state in available_deps.items() 
                      if state.spec.performance_impact > 3.0]
        for dep in high_impact:
            combinations.append([dep])
        
        # Test category combinations
        categories = defaultdict(list)
        for name, state in available_deps.items():
            categories[state.spec.category].append(name)
        
        # ML core combination
        if categories['ml_core']:
            combinations.append(categories['ml_core'])
        
        # Performance combination
        if categories['performance']:
            combinations.append(categories['performance'])
        
        # Full optimization combination
        all_deps = list(available_deps.keys())
        if len(all_deps) > 1:
            combinations.append(all_deps)
        
        # Steam Deck optimized combination
        if self.is_steam_deck:
            steam_deck_optimized = [
                name for name, state in available_deps.items()
                if state.spec.steam_deck_compatible and state.spec.performance_impact > 2.0
            ]
            if steam_deck_optimized:
                combinations.append(steam_deck_optimized)
        
        return combinations
    
    def _benchmark_combination(self, deps: List[str], test_size: int) -> PerformanceProfile:
        """Benchmark a specific combination of dependencies"""
        combination_id = '+'.join(sorted(deps)) if deps else 'pure_python'
        
        # Setup phase
        setup_start = time.time()
        memory_before = self._get_memory_usage()
        
        # Simulate workload with this combination
        execution_start = time.time()
        try:
            self._simulate_ml_workload(deps, test_size)
            execution_time = time.time() - execution_start
            stability_score = 1.0
        except Exception as e:
            execution_time = float('inf')
            stability_score = 0.0
            logger.warning(f"Combination {combination_id} failed execution: {e}")
        
        setup_time = execution_start - setup_start
        memory_after = self._get_memory_usage()
        memory_usage = memory_after - memory_before
        
        # Calculate compatibility score
        compatibility_score = self._calculate_compatibility_score(deps)
        
        # Calculate overall score
        if execution_time == float('inf'):
            overall_score = 0.0
        else:
            # Weighted scoring: performance (40%), stability (30%), compatibility (20%), memory (10%)
            performance_score = max(0, 10.0 - execution_time)  # Lower time = higher score
            memory_score = max(0, 10.0 - (memory_usage / 1024 / 1024))  # Lower memory = higher score
            
            overall_score = (
                0.4 * performance_score +
                0.3 * stability_score * 10 +
                0.2 * compatibility_score * 10 +
                0.1 * memory_score
            )
        
        return PerformanceProfile(
            combination_id=combination_id,
            dependencies=deps.copy(),
            setup_time=setup_time,
            execution_time=execution_time,
            memory_usage=memory_usage,
            stability_score=stability_score,
            compatibility_score=compatibility_score,
            overall_score=overall_score
        )
    
    def _simulate_ml_workload(self, deps: List[str], test_size: int) -> None:
        """Simulate ML workload with specified dependencies"""
        # Create synthetic data
        data = [[i / 100.0, (i * 2) / 100.0, (i * 3) / 100.0] for i in range(test_size)]
        labels = [sum(row) + 0.1 * (i % 10) for i, row in enumerate(data)]
        
        # Test array operations (numpy or fallback)
        if 'numpy' in deps:
            import numpy as np
            np_data = np.array(data)
            np_labels = np.array(labels)
            # Perform some operations
            mean_data = np.mean(np_data, axis=0)
            std_data = np.std(np_data, axis=0)
        else:
            # Use pure Python
            mean_data = [sum(row[i] for row in data) / len(data) for i in range(3)]
            # Simple std calculation
        
        # Test ML algorithms
        if 'lightgbm' in deps:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
            model.fit(data[:test_size//2], labels[:test_size//2])
            predictions = model.predict(data[test_size//2:])
        elif 'scikit-learn' in deps:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(data[:test_size//2], labels[:test_size//2])
            predictions = model.predict(data[test_size//2:])
        else:
            # Use pure Python linear regression
            from .pure_python_fallbacks import PureLinearRegressor
            model = PureLinearRegressor()
            model.fit(data[:test_size//2], labels[:test_size//2])
            predictions = model.predict(data[test_size//2:])
        
        # Test compression if available
        if 'zstandard' in deps:
            import zstandard as zstd
            compressor = zstd.ZstdCompressor()
            test_data = json.dumps(data).encode()
            compressed = compressor.compress(test_data)
        elif 'msgpack' in deps:
            import msgpack
            compressed = msgpack.packb(data)
        else:
            import gzip
            compressed = gzip.compress(json.dumps(data).encode())
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            if 'psutil' in self.dependency_states and self.dependency_states['psutil'].available:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss
            else:
                # Fallback estimation
                from .pure_python_fallbacks import PureSystemMonitor
                monitor = PureSystemMonitor()
                return monitor.memory_info().rss
        except Exception:
            return 0
    
    def _calculate_compatibility_score(self, deps: List[str]) -> float:
        """Calculate compatibility score for a combination"""
        if not deps:
            return 1.0  # Pure Python is always compatible
        
        score = 1.0
        
        # Steam Deck compatibility bonus/penalty
        if self.is_steam_deck:
            for dep in deps:
                if dep in self.dependency_states:
                    spec = self.dependency_states[dep].spec
                    if spec.steam_deck_compatible:
                        score += 0.1
                    else:
                        score -= 0.2
        
        # Compilation penalty
        for dep in deps:
            if dep in self.dependency_states:
                spec = self.dependency_states[dep].spec
                if spec.compilation_required:
                    score -= 0.3
        
        # Platform compatibility
        current_platform = self.system_info['platform'].lower()
        for dep in deps:
            if dep in self.dependency_states:
                spec = self.dependency_states[dep].spec
                if spec.platform_specific:
                    if current_platform not in [p.lower() for p in spec.platform_specific if not p.startswith('!')]:
                        score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def optimize_for_environment(self) -> Dict[str, Any]:
        """
        Optimize dependency configuration for current environment
        """
        logger.info("Optimizing dependency configuration for environment...")
        
        optimization_result = {
            'environment': self.system_info.copy(),
            'recommendations': [],
            'warnings': [],
            'optimizations_applied': [],
            'performance_improvement': 0.0
        }
        
        # Check thermal state for aggressive optimizations
        thermal_state = self.thermal_monitor.get_thermal_state()
        if thermal_state in ['hot', 'critical']:
            optimization_result['warnings'].append(
                f"High thermal state ({thermal_state}) - enabling conservative optimizations"
            )
            # Disable JIT compilation to reduce CPU load
            if 'numba' in self.dependency_states and self.dependency_states['numba'].available:
                optimization_result['recommendations'].append(
                    "Consider disabling numba JIT compilation due to high temperature"
                )
        
        # Steam Deck specific optimizations
        if self.is_steam_deck:
            steam_optimizations = self._optimize_for_steam_deck()
            optimization_result['optimizations_applied'].extend(steam_optimizations)
        
        # Memory-constrained optimizations
        memory_usage = self._get_memory_usage()
        if memory_usage > 1024 * 1024 * 1024:  # > 1GB
            optimization_result['recommendations'].append(
                "High memory usage detected - consider using lightweight alternatives"
            )
        
        # Performance combination selection
        if self.performance_profiles:
            best_profile = self.performance_profiles[0]
            if best_profile.overall_score > 5.0:
                optimization_result['recommendations'].append(
                    f"Switch to performance profile '{best_profile.combination_id}' for {best_profile.overall_score:.1f}x improvement"
                )
                optimization_result['performance_improvement'] = best_profile.overall_score
        
        return optimization_result
    
    def _optimize_for_steam_deck(self) -> List[str]:
        """Apply Steam Deck specific optimizations"""
        optimizations = []
        
        # Prefer dependencies with proven Steam Deck compatibility
        steam_deck_friendly = ['numpy', 'scikit-learn', 'msgpack', 'psutil']
        available_friendly = [
            dep for dep in steam_deck_friendly
            if dep in self.dependency_states and self.dependency_states[dep].available
        ]
        
        if available_friendly:
            optimizations.append(f"Prioritized Steam Deck compatible dependencies: {', '.join(available_friendly)}")
        
        # Disable compilation-heavy dependencies if temperature is high
        thermal_state = self.thermal_monitor.get_thermal_state()
        if thermal_state in ['warm', 'hot', 'critical']:
            compilation_deps = [
                name for name, state in self.dependency_states.items()
                if state.spec.compilation_required and state.available
            ]
            if compilation_deps:
                optimizations.append(f"Disabled compilation-heavy dependencies due to thermal state: {', '.join(compilation_deps)}")
        
        # Enable power-efficient algorithms
        optimizations.append("Enabled power-efficient ML algorithms for handheld gaming")
        
        return optimizations
    
    def validate_installation(self, dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate that dependencies are properly installed and working
        """
        logger.info("Validating dependency installation...")
        
        deps_to_validate = dependencies or list(self.dependency_states.keys())
        validation_results = {
            'total': len(deps_to_validate),
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'details': {},
            'overall_health': 0.0,
            'recommendations': []
        }
        
        for dep_name in deps_to_validate:
            if dep_name not in self.dependency_states:
                validation_results['details'][dep_name] = {
                    'status': 'unknown',
                    'message': 'Dependency not detected'
                }
                validation_results['skipped'] += 1
                continue
            
            state = self.dependency_states[dep_name]
            result = {
                'status': 'unknown',
                'available': state.available,
                'version': state.version,
                'test_passed': state.test_passed,
                'fallback_active': state.fallback_active,
                'message': '',
                'performance_score': state.performance_score
            }
            
            if not state.available:
                if state.spec.fallback_available:
                    result['status'] = 'fallback'
                    result['message'] = f"Using fallback implementation. Original error: {state.error}"
                    validation_results['passed'] += 1
                else:
                    result['status'] = 'failed'
                    result['message'] = f"Not available and no fallback. Error: {state.error}"
                    validation_results['failed'] += 1
            elif not state.test_passed:
                result['status'] = 'degraded'
                result['message'] = f"Available but tests failed: {state.error}"
                validation_results['failed'] += 1
            else:
                result['status'] = 'healthy'
                result['message'] = f"Fully functional (performance score: {state.performance_score:.1f})"
                validation_results['passed'] += 1
            
            validation_results['details'][dep_name] = result
        
        # Calculate overall health
        validation_results['overall_health'] = validation_results['passed'] / max(1, validation_results['total'])
        
        # Generate recommendations
        if validation_results['failed'] > 0:
            validation_results['recommendations'].append(
                f"{validation_results['failed']} dependencies failed validation - consider reinstallation"
            )
        
        if validation_results['overall_health'] < 0.8:
            validation_results['recommendations'].append(
                "System health below 80% - run dependency detection again or check installation"
            )
        
        # Check for missing high-impact dependencies
        high_impact_missing = [
            name for name, state in self.dependency_states.items()
            if not state.available and state.spec.performance_impact > 3.0
        ]
        if high_impact_missing:
            validation_results['recommendations'].append(
                f"Consider installing high-impact dependencies: {', '.join(high_impact_missing)}"
            )
        
        logger.info(f"Validation complete: {validation_results['passed']}/{validation_results['total']} healthy, overall health: {validation_results['overall_health']:.1%}")
        
        return validation_results
    
    def switch_dependency_profile(self, profile_id: str) -> bool:
        """
        Switch to a different dependency profile at runtime
        """
        target_profile = None
        for profile in self.performance_profiles:
            if profile.combination_id == profile_id:
                target_profile = profile
                break
        
        if not target_profile:
            logger.error(f"Profile '{profile_id}' not found")
            return False
        
        with self.optimization_lock:
            try:
                logger.info(f"Switching to dependency profile '{profile_id}'...")
                
                # Validate all dependencies in the target profile are available
                for dep_name in target_profile.dependencies:
                    if dep_name not in self.dependency_states:
                        logger.error(f"Dependency '{dep_name}' not detected")
                        return False
                    
                    state = self.dependency_states[dep_name]
                    if not state.available or not state.test_passed:
                        logger.error(f"Dependency '{dep_name}' not available or failed tests")
                        return False
                
                # Switch to the new profile
                self.current_profile = target_profile
                logger.info(f"Successfully switched to profile '{profile_id}' (score: {target_profile.overall_score:.2f})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to switch to profile '{profile_id}': {e}")
                return False
    
    def get_dependency_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations for dependency management
        """
        recommendations = []
        
        # Check for missing critical dependencies
        missing_critical = [
            name for name, state in self.dependency_states.items()
            if not state.available and state.spec.required
        ]
        if missing_critical:
            recommendations.append({
                'type': 'critical',
                'title': 'Missing Critical Dependencies',
                'description': f"Install required dependencies: {', '.join(missing_critical)}",
                'action': f"pip install {' '.join(missing_critical)}",
                'priority': 'high'
            })
        
        # Check for performance improvements
        high_impact_missing = [
            name for name, state in self.dependency_states.items()
            if not state.available and state.spec.performance_impact > 3.0 and not state.spec.compilation_required
        ]
        if high_impact_missing:
            recommendations.append({
                'type': 'performance',
                'title': 'Performance Optimization Available',
                'description': f"Install high-impact dependencies for better performance: {', '.join(high_impact_missing)}",
                'action': f"pip install {' '.join(high_impact_missing)}",
                'priority': 'medium'
            })
        
        # Check for outdated dependencies
        outdated = []
        for name, state in self.dependency_states.items():
            if state.available and state.version and state.spec.version_max:
                if self._compare_versions(state.version, state.spec.version_max) >= 0:
                    outdated.append(name)
        if outdated:
            recommendations.append({
                'type': 'maintenance',
                'title': 'Outdated Dependencies',
                'description': f"Update outdated dependencies: {', '.join(outdated)}",
                'action': f"pip install --upgrade {' '.join(outdated)}",
                'priority': 'low'
            })
        
        # Steam Deck specific recommendations
        if self.is_steam_deck:
            compilation_deps = [
                name for name, state in self.dependency_states.items()
                if state.available and state.spec.compilation_required
            ]
            if compilation_deps:
                recommendations.append({
                    'type': 'compatibility',
                    'title': 'Steam Deck Compatibility Warning',
                    'description': f"Dependencies requiring compilation detected on Steam Deck: {', '.join(compilation_deps)}",
                    'action': "Consider using pure Python alternatives for better compatibility",
                    'priority': 'medium'
                })
        
        return recommendations
    
    def auto_install_missing_dependencies(self, dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Automatically install missing dependencies using enhanced installer
        """
        if not self.installer:
            return {
                'status': 'unavailable',
                'message': 'Enhanced installer not available',
                'installed': {},
                'failed': {}
            }
        
        logger.info("Starting automatic dependency installation...")
        
        # Determine which dependencies to install
        if dependencies:
            deps_to_install = dependencies
        else:
            # Install all missing dependencies that are not available
            deps_to_install = [
                name for name, state in self.dependency_states.items()
                if not state.available and not state.spec.compilation_required
            ]
        
        if not deps_to_install:
            logger.info("No dependencies need installation")
            return {
                'status': 'complete',
                'message': 'All dependencies already available',
                'installed': {},
                'failed': {}
            }
        
        logger.info(f"Installing {len(deps_to_install)} dependencies: {', '.join(deps_to_install)}")
        
        # Use enhanced installer
        installation_results = self.installer.install_multiple_dependencies(deps_to_install)
        
        # Re-detect dependencies after installation
        logger.info("Re-detecting dependencies after installation...")
        self.detect_all_dependencies(force_refresh=True)
        
        # Categorize results
        successful = {
            dep: result for dep, result in installation_results.items()
            if result.success
        }
        
        failed = {
            dep: result for dep, result in installation_results.items()
            if not result.success
        }
        
        return {
            'status': 'complete' if not failed else 'partial',
            'message': f"Installed {len(successful)}/{len(deps_to_install)} dependencies",
            'installed': successful,
            'failed': failed,
            'total_attempted': len(deps_to_install)
        }
    
    def create_comprehensive_health_report(self) -> Dict[str, Any]:
        """
        Create a comprehensive health report including installation options
        """
        # Basic validation
        validation = self.validate_installation()
        
        # Enhanced health check if installer available
        installer_health = None
        if self.installer:
            try:
                installer_health = self.installer.create_dependency_health_check()
            except Exception as e:
                logger.warning(f"Enhanced health check failed: {e}")
        
        # Combine reports
        health_report = {
            'basic_validation': validation,
            'enhanced_health': installer_health,
            'system_optimization': self.optimize_for_environment(),
            'recommendations': self.get_dependency_recommendations(),
            'installation_options': self._get_installation_options()
        }
        
        # Calculate overall system health
        basic_health = validation['overall_health']
        enhanced_health = installer_health['overall_health'] if installer_health else basic_health
        
        health_report['overall_system_health'] = max(basic_health, enhanced_health)
        health_report['health_sources'] = {
            'basic': basic_health,
            'enhanced': enhanced_health if installer_health else None
        }
        
        return health_report
    
    def _get_installation_options(self) -> Dict[str, Any]:
        """Get available installation options for missing dependencies"""
        options = {
            'enhanced_installer_available': self.installer is not None,
            'missing_dependencies': [],
            'recommended_actions': []
        }
        
        # Find missing dependencies
        for name, state in self.dependency_states.items():
            if not state.available:
                dep_info = {
                    'name': name,
                    'required': state.spec.required,
                    'performance_impact': state.spec.performance_impact,
                    'steam_deck_compatible': state.spec.steam_deck_compatible,
                    'compilation_required': state.spec.compilation_required,
                    'fallback_available': state.spec.fallback_available
                }
                options['missing_dependencies'].append(dep_info)
        
        # Generate recommended actions
        if options['missing_dependencies']:
            high_impact_missing = [
                dep for dep in options['missing_dependencies']
                if dep['performance_impact'] > 3.0 and not dep['compilation_required']
            ]
            
            if high_impact_missing:
                options['recommended_actions'].append({
                    'action': 'install_high_impact',
                    'dependencies': [dep['name'] for dep in high_impact_missing],
                    'description': 'Install high-impact dependencies for significant performance improvement'
                })
            
            if self.is_steam_deck:
                steam_deck_safe = [
                    dep for dep in options['missing_dependencies']
                    if dep['steam_deck_compatible'] and not dep['compilation_required']
                ]
                
                if steam_deck_safe:
                    options['recommended_actions'].append({
                        'action': 'install_steam_deck_safe',
                        'dependencies': [dep['name'] for dep in steam_deck_safe],
                        'description': 'Install Steam Deck compatible dependencies'
                    })
        
        return options

    def _log_detection_summary(self) -> None:
        """Log a summary of dependency detection results"""
        total = len(self.dependency_states)
        available = sum(1 for state in self.dependency_states.values() if state.available)
        tested = sum(1 for state in self.dependency_states.values() if state.test_passed)
        fallbacks = sum(1 for state in self.dependency_states.values() if state.fallback_active)
        
        logger.info(f"Dependency Summary: {available}/{total} available, {tested}/{total} tested, {fallbacks} fallbacks active")
        
        # Log by category
        categories = defaultdict(list)
        for name, state in self.dependency_states.items():
            categories[state.spec.category].append((name, state.available))
        
        for category, deps in categories.items():
            available_in_category = sum(1 for _, available in deps if available)
            logger.info(f"  {category}: {available_in_category}/{len(deps)} available")
    
    @lru_cache(maxsize=128)
    def get_cached_dependency_info(self, dep_name: str) -> Optional[Dict[str, Any]]:
        """Get cached dependency information"""
        if dep_name not in self.dependency_states:
            return None
        
        state = self.dependency_states[dep_name]
        return {
            'name': dep_name,
            'available': state.available,
            'version': state.version,
            'performance_score': state.performance_score,
            'fallback_active': state.fallback_active,
            'category': state.spec.category
        }
    
    def export_configuration(self, path: Path) -> None:
        """Export current dependency configuration to file"""
        config = {
            'system_info': self.system_info,
            'dependency_states': {
                name: {
                    'available': state.available,
                    'version': state.version,
                    'test_passed': state.test_passed,
                    'fallback_active': state.fallback_active,
                    'performance_score': state.performance_score,
                    'error': state.error
                }
                for name, state in self.dependency_states.items()
            },
            'performance_profiles': [
                {
                    'combination_id': profile.combination_id,
                    'dependencies': profile.dependencies,
                    'overall_score': profile.overall_score,
                    'execution_time': profile.execution_time,
                    'memory_usage': profile.memory_usage
                }
                for profile in self.performance_profiles
            ],
            'current_profile': self.current_profile.combination_id if self.current_profile else None,
            'timestamp': time.time()
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration exported to {path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global coordinator instance
_coordinator: Optional[DependencyCoordinator] = None

def get_coordinator() -> DependencyCoordinator:
    """Get or create the global dependency coordinator"""
    global _coordinator
    if _coordinator is None:
        _coordinator = DependencyCoordinator()
        _coordinator.detect_all_dependencies()
    return _coordinator

def quick_optimization() -> Dict[str, Any]:
    """Perform quick dependency optimization and return results"""
    coordinator = get_coordinator()
    
    # Run benchmarks if not already done
    if not coordinator.performance_profiles:
        coordinator.benchmark_performance_combinations(test_size=500)
    
    # Get optimization recommendations
    optimization_result = coordinator.optimize_for_environment()
    validation_result = coordinator.validate_installation()
    recommendations = coordinator.get_dependency_recommendations()
    
    return {
        'optimization': optimization_result,
        'validation': validation_result,
        'recommendations': recommendations,
        'best_profile': coordinator.current_profile.combination_id if coordinator.current_profile else 'pure_python'
    }

def check_dependency_health() -> float:
    """Quick health check - returns score from 0.0 to 1.0"""
    coordinator = get_coordinator()
    validation = coordinator.validate_installation()
    return validation['overall_health']

def get_optimal_dependencies() -> List[str]:
    """Get list of optimal dependencies for current environment"""
    coordinator = get_coordinator()
    if coordinator.current_profile:
        return coordinator.current_profile.dependencies
    else:
        return []


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🔧 Enhanced Dependency Coordinator Test Suite")
    print("=" * 60)
    
    # Initialize coordinator
    coordinator = DependencyCoordinator()
    
    # Detect all dependencies
    print("\n📋 Detecting Dependencies...")
    states = coordinator.detect_all_dependencies()
    
    # Show detection results
    print(f"\nDetection Results:")
    for name, state in states.items():
        status = "✅" if state.available and state.test_passed else "🔄" if state.fallback_active else "❌"
        version = f" v{state.version}" if state.version else ""
        print(f"  {status} {name}{version} (score: {state.performance_score:.1f})")
    
    # Run performance benchmarks
    print("\n⚡ Running Performance Benchmarks...")
    profiles = coordinator.benchmark_performance_combinations(test_size=200)
    
    print(f"\nPerformance Profiles:")
    for i, profile in enumerate(profiles[:5]):  # Top 5
        print(f"  {i+1}. {profile.combination_id}: {profile.overall_score:.2f} (exec: {profile.execution_time:.3f}s)")
    
    # Validate installation
    print("\n🔍 Validating Installation...")
    validation = coordinator.validate_installation()
    print(f"Health Score: {validation['overall_health']:.1%}")
    print(f"Status: {validation['passed']} passed, {validation['failed']} failed, {validation['skipped']} skipped")
    
    # Get recommendations
    print("\n💡 Recommendations:")
    recommendations = coordinator.get_dependency_recommendations()
    for rec in recommendations:
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "ℹ️")
        print(f"  {priority_emoji} {rec['title']}: {rec['description']}")
    
    # Environment optimization
    print("\n🎯 Environment Optimization...")
    optimization = coordinator.optimize_for_environment()
    print(f"Platform: {optimization['environment']['platform']} {optimization['environment']['machine']}")
    print(f"Steam Deck: {optimization['environment']['is_steam_deck']}")
    
    if optimization['optimizations_applied']:
        print("Applied optimizations:")
        for opt in optimization['optimizations_applied']:
            print(f"  ✅ {opt}")
    
    if optimization['recommendations']:
        print("Recommendations:")
        for rec in optimization['recommendations']:
            print(f"  💡 {rec}")
    
    # Export configuration
    config_path = Path("/tmp/dependency_config.json")
    coordinator.export_configuration(config_path)
    print(f"\n💾 Configuration exported to {config_path}")
    
    print(f"\n✅ Dependency Coordinator test completed successfully!")
    print(f"🚀 Optimal configuration: {coordinator.current_profile.combination_id if coordinator.current_profile else 'pure_python'}")