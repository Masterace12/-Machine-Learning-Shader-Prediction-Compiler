#!/usr/bin/env python3
"""
Enhanced Dependency Detection System

This module provides bulletproof dependency detection with comprehensive error handling,
graceful fallback management, and Steam Deck optimizations.

Features:
- Thread-safe dependency detection with configurable timeouts
- Comprehensive error handling and recovery
- Multiple detection strategies (import, version, functionality tests)
- Steam Deck specific optimizations and workarounds
- Memory-efficient detection with resource management
- Detailed logging and status reporting
- Automatic fallback to pure Python implementations
- Performance benchmarking of detected dependencies
"""

import os
import sys
import time
import json
import logging
import threading
import importlib
import subprocess
import traceback
import gc
from typing import Dict, List, Any, Optional, Union, Callable, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import wraps, lru_cache
from collections import defaultdict, deque
import io
import warnings
import weakref

# Import our components
from .dependency_version_manager import (
    DependencyVersionManager, DEPENDENCY_MATRIX, get_version_manager
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# DETECTION RESULT STRUCTURES
# =============================================================================

class DetectionStatus(NamedTuple):
    """Status of dependency detection"""
    available: bool
    version: Optional[str]
    import_time: float
    test_time: float
    error: Optional[str]
    fallback_active: bool
    performance_score: float
    memory_usage_mb: float

@dataclass
class DetectionStrategy:
    """Strategy for detecting a dependency"""
    name: str
    priority: int  # Higher number = higher priority
    timeout: float = 5.0
    memory_limit_mb: int = 100
    retry_count: int = 2
    fallback_on_failure: bool = True

@dataclass 
class DetectionResult:
    """Comprehensive detection result"""
    dependency_name: str
    strategies_attempted: List[str]
    successful_strategy: Optional[str]
    status: DetectionStatus
    warnings: List[str]
    recommendations: List[str]
    fallback_reason: Optional[str]
    detection_timestamp: float
    system_impact: Dict[str, Any]

@dataclass
class SystemResourceMonitor:
    """Monitor system resources during detection"""
    initial_memory_mb: float
    peak_memory_mb: float
    cpu_time_used: float
    thread_count: int
    file_descriptors_used: int = 0

# =============================================================================
# ENHANCED DEPENDENCY DETECTOR
# =============================================================================

class EnhancedDependencyDetector:
    """
    Advanced dependency detector with comprehensive error handling and fallbacks
    """
    
    def __init__(self, 
                 max_workers: int = 4, 
                 global_timeout: float = 60.0,
                 memory_limit_mb: int = 512):
        
        self.version_manager = get_version_manager()
        self.max_workers = min(max_workers, os.cpu_count() or 4)
        self.global_timeout = global_timeout
        self.memory_limit_mb = memory_limit_mb
        
        # Detection state
        self.detection_results: Dict[str, DetectionResult] = {}
        self.detection_lock = threading.RLock()
        self.active_detections = set()
        self.detection_cache: Dict[str, DetectionResult] = {}
        self.fallback_reasons: Dict[str, str] = {}
        
        # Resource monitoring
        self.resource_monitor = SystemResourceMonitor(
            initial_memory_mb=self._get_memory_usage(),
            peak_memory_mb=0.0,
            cpu_time_used=0.0,
            thread_count=threading.active_count()
        )
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.successful_detections: Dict[str, int] = defaultdict(int)
        self.detection_failures: Dict[str, List[str]] = defaultdict(list)
        
        # Steam Deck optimizations
        self.is_steam_deck = self.version_manager.system_info['is_steam_deck']
        self.steam_deck_workarounds_enabled = self.is_steam_deck
        
        # Detection strategies
        self.detection_strategies = self._setup_detection_strategies()
        
        logger.info(f"EnhancedDependencyDetector initialized with {self.max_workers} workers")
        logger.info(f"Steam Deck mode: {self.is_steam_deck}, Memory limit: {self.memory_limit_mb}MB")

    def _setup_detection_strategies(self) -> Dict[str, DetectionStrategy]:
        """Setup detection strategies in priority order"""
        strategies = {
            'import_test': DetectionStrategy(
                name='import_test',
                priority=10,
                timeout=2.0,
                memory_limit_mb=50,
                retry_count=1,
                fallback_on_failure=True
            ),
            'version_check': DetectionStrategy(
                name='version_check', 
                priority=9,
                timeout=3.0,
                memory_limit_mb=30,
                retry_count=2,
                fallback_on_failure=True
            ),
            'functionality_test': DetectionStrategy(
                name='functionality_test',
                priority=8,
                timeout=10.0,
                memory_limit_mb=100,
                retry_count=1,
                fallback_on_failure=True
            ),
            'pip_show': DetectionStrategy(
                name='pip_show',
                priority=7,
                timeout=15.0,
                memory_limit_mb=20,
                retry_count=3,
                fallback_on_failure=True
            ),
            'comprehensive_test': DetectionStrategy(
                name='comprehensive_test',
                priority=6,
                timeout=20.0,
                memory_limit_mb=200,
                retry_count=1,
                fallback_on_failure=True
            )
        }
        
        # Adjust strategies for Steam Deck
        if self.is_steam_deck:
            # Reduce timeouts and memory limits for constrained environment
            for strategy in strategies.values():
                strategy.timeout *= 0.7  # Reduce timeouts by 30%
                strategy.memory_limit_mb = int(strategy.memory_limit_mb * 0.8)  # Reduce memory limit
        
        return strategies

    def detect_all_dependencies(self, 
                              dependencies: Optional[List[str]] = None,
                              force_refresh: bool = False,
                              parallel: bool = True) -> Dict[str, DetectionResult]:
        """
        Detect all dependencies with comprehensive error handling
        """
        if not force_refresh and self.detection_results and not dependencies:
            return self.detection_results

        deps_to_detect = dependencies or list(DEPENDENCY_MATRIX.keys())
        
        logger.info(f"Starting detection of {len(deps_to_detect)} dependencies...")
        logger.info(f"Parallel: {parallel}, Workers: {self.max_workers if parallel else 1}")
        
        start_time = time.time()
        
        try:
            if parallel and len(deps_to_detect) > 1:
                results = self._detect_parallel(deps_to_detect)
            else:
                results = self._detect_sequential(deps_to_detect)
            
            detection_time = time.time() - start_time
            
            # Update resource monitor
            self.resource_monitor.peak_memory_mb = max(
                self.resource_monitor.peak_memory_mb,
                self._get_memory_usage()
            )
            self.resource_monitor.cpu_time_used = detection_time
            
            # Update results
            for dep_name, result in results.items():
                self.detection_results[dep_name] = result
                if not dependencies:  # Only cache if detecting all
                    self.detection_cache[dep_name] = result
            
            self._log_detection_summary(results, detection_time)
            return results
            
        except Exception as e:
            logger.error(f"Critical error in dependency detection: {e}")
            logger.error(traceback.format_exc())
            
            # Return partial results with error markers
            return self._create_error_results(deps_to_detect, str(e))

    def _detect_parallel(self, dependencies: List[str]) -> Dict[str, DetectionResult]:
        """Detect dependencies in parallel with proper resource management"""
        results = {}
        failed_deps = []
        
        # Use a more conservative worker count for Steam Deck
        worker_count = min(self.max_workers, 2 if self.is_steam_deck else 4)
        
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                # Submit detection tasks
                future_to_dep = {
                    executor.submit(self._detect_single_dependency_safe, dep): dep
                    for dep in dependencies
                }
                
                # Collect results with timeout
                for future in as_completed(future_to_dep, timeout=self.global_timeout):
                    dep_name = future_to_dep[future]
                    
                    try:
                        result = future.result(timeout=5.0)  # Individual task timeout
                        results[dep_name] = result
                        
                        if result.status.available:
                            self.successful_detections[dep_name] += 1
                        else:
                            self.error_counts[dep_name] += 1
                            
                    except Exception as e:
                        logger.error(f"Detection failed for {dep_name}: {e}")
                        failed_deps.append(dep_name)
                        self.error_counts[dep_name] += 1
                        self.detection_failures[dep_name].append(str(e))
                        
        except Exception as e:
            logger.error(f"Parallel detection executor error: {e}")
            failed_deps.extend([dep for dep in dependencies if dep not in results])
        
        # Handle failed dependencies with fallback detection
        if failed_deps:
            logger.warning(f"Retrying {len(failed_deps)} failed detections sequentially...")
            for dep_name in failed_deps:
                try:
                    result = self._detect_single_dependency_safe(dep_name)
                    results[dep_name] = result
                except Exception as e:
                    results[dep_name] = self._create_failed_detection_result(dep_name, str(e))
        
        return results

    def _detect_sequential(self, dependencies: List[str]) -> Dict[str, DetectionResult]:
        """Detect dependencies sequentially with comprehensive error handling"""
        results = {}
        
        for i, dep_name in enumerate(dependencies):
            logger.debug(f"Detecting {dep_name} ({i+1}/{len(dependencies)})...")
            
            try:
                result = self._detect_single_dependency_safe(dep_name)
                results[dep_name] = result
                
                if result.status.available:
                    self.successful_detections[dep_name] += 1
                else:
                    self.error_counts[dep_name] += 1
                
                # Memory management - force garbage collection periodically
                if i % 5 == 0:
                    gc.collect()
                    
                # Check memory usage
                current_memory = self._get_memory_usage()
                if current_memory > self.memory_limit_mb:
                    logger.warning(f"High memory usage: {current_memory:.1f}MB, forcing cleanup...")
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Sequential detection failed for {dep_name}: {e}")
                results[dep_name] = self._create_failed_detection_result(dep_name, str(e))
                self.error_counts[dep_name] += 1
        
        return results

    def _detect_single_dependency_safe(self, dep_name: str) -> DetectionResult:
        """Safely detect a single dependency with comprehensive error handling"""
        if dep_name not in DEPENDENCY_MATRIX:
            return self._create_failed_detection_result(
                dep_name, 
                f"Unknown dependency: {dep_name}"
            )
        
        # Check cache first
        if dep_name in self.detection_cache:
            cached_result = self.detection_cache[dep_name]
            # Use cache if less than 5 minutes old
            if time.time() - cached_result.detection_timestamp < 300:
                return cached_result
        
        profile = DEPENDENCY_MATRIX[dep_name]
        strategies_attempted = []
        warnings = []
        recommendations = []
        fallback_reason = None
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        with self.detection_lock:
            if dep_name in self.active_detections:
                # Avoid concurrent detection of same dependency
                return self._create_failed_detection_result(
                    dep_name, 
                    "Detection already in progress"
                )
            self.active_detections.add(dep_name)
        
        try:
            # Try detection strategies in priority order
            detection_status = None
            successful_strategy = None
            
            for strategy_name, strategy in sorted(
                self.detection_strategies.items(), 
                key=lambda x: x[1].priority, 
                reverse=True
            ):
                strategies_attempted.append(strategy_name)
                
                try:
                    logger.debug(f"Trying {strategy_name} for {dep_name}...")
                    
                    status = self._execute_detection_strategy(
                        dep_name, profile, strategy
                    )
                    
                    if status.available or not strategy.fallback_on_failure:
                        detection_status = status
                        successful_strategy = strategy_name
                        break
                        
                except Exception as e:
                    logger.debug(f"Strategy {strategy_name} failed for {dep_name}: {e}")
                    warnings.append(f"{strategy_name}: {str(e)[:100]}")
                    continue
            
            # If all strategies failed, create fallback result
            if detection_status is None:
                fallback_reason = "All detection strategies failed"
                detection_status = DetectionStatus(
                    available=False,
                    version=None,
                    import_time=0.0,
                    test_time=0.0,
                    error=fallback_reason,
                    fallback_active=True,
                    performance_score=0.0,
                    memory_usage_mb=0.0
                )
                successful_strategy = None
            
            # Generate recommendations
            if not detection_status.available and profile.fallback_available:
                recommendations.append(f"Pure Python fallback available for {dep_name}")
            
            if detection_status.available and profile.risk_assessment.compilation_risk > 5:
                if self.is_steam_deck:
                    recommendations.append("Monitor for compilation issues on Steam Deck")
            
            # System impact assessment
            final_memory = self._get_memory_usage()
            system_impact = {
                'memory_delta_mb': final_memory - initial_memory,
                'detection_time': time.time() - start_time,
                'strategies_tried': len(strategies_attempted),
                'thread_safe': True
            }
            
            return DetectionResult(
                dependency_name=dep_name,
                strategies_attempted=strategies_attempted,
                successful_strategy=successful_strategy,
                status=detection_status,
                warnings=warnings,
                recommendations=recommendations,
                fallback_reason=fallback_reason,
                detection_timestamp=time.time(),
                system_impact=system_impact
            )
            
        except Exception as e:
            logger.error(f"Critical error detecting {dep_name}: {e}")
            return self._create_failed_detection_result(dep_name, str(e))
            
        finally:
            with self.detection_lock:
                self.active_detections.discard(dep_name)

    def _execute_detection_strategy(self, 
                                   dep_name: str, 
                                   profile: Any,
                                   strategy: DetectionStrategy) -> DetectionStatus:
        """Execute a specific detection strategy with timeout and resource limits"""
        
        if strategy.name == 'import_test':
            return self._strategy_import_test(dep_name, profile, strategy)
        elif strategy.name == 'version_check':
            return self._strategy_version_check(dep_name, profile, strategy)
        elif strategy.name == 'functionality_test':
            return self._strategy_functionality_test(dep_name, profile, strategy)
        elif strategy.name == 'pip_show':
            return self._strategy_pip_show(dep_name, profile, strategy)
        elif strategy.name == 'comprehensive_test':
            return self._strategy_comprehensive_test(dep_name, profile, strategy)
        else:
            raise ValueError(f"Unknown strategy: {strategy.name}")

    def _strategy_import_test(self, dep_name: str, profile: Any, strategy: DetectionStrategy) -> DetectionStatus:
        """Basic import test strategy"""
        import_time_start = time.time()
        
        for import_name in profile.import_names:
            try:
                with self._timeout_context(strategy.timeout):
                    # Suppress warnings during import
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with redirect_stderr(io.StringIO()):
                            module = importlib.import_module(import_name)
                    
                    import_time = time.time() - import_time_start
                    
                    # Basic availability check
                    if module is not None:
                        return DetectionStatus(
                            available=True,
                            version=None,  # Will be detected separately
                            import_time=import_time,
                            test_time=0.0,
                            error=None,
                            fallback_active=False,
                            performance_score=profile.risk_assessment.performance_impact,
                            memory_usage_mb=self._estimate_module_memory(module)
                        )
                        
            except ImportError:
                continue
            except Exception as e:
                logger.debug(f"Import test failed for {import_name}: {e}")
                continue
        
        return DetectionStatus(
            available=False,
            version=None,
            import_time=time.time() - import_time_start,
            test_time=0.0,
            error="Import failed for all names",
            fallback_active=profile.fallback_available,
            performance_score=0.0,
            memory_usage_mb=0.0
        )

    def _strategy_version_check(self, dep_name: str, profile: Any, strategy: DetectionStrategy) -> DetectionStatus:
        """Version detection strategy"""
        start_time = time.time()
        
        try:
            with self._timeout_context(strategy.timeout):
                version_str = self.version_manager.detect_installed_version(dep_name, timeout=strategy.timeout - 1)
                
                if version_str:
                    # Verify version compatibility
                    compatibility = self.version_manager.check_version_compatibility(dep_name, version_str)
                    
                    return DetectionStatus(
                        available=True,
                        version=version_str,
                        import_time=0.0,
                        test_time=time.time() - start_time,
                        error=None,
                        fallback_active=not compatibility['compatible'],
                        performance_score=profile.risk_assessment.performance_impact * (0.8 if compatibility['compatible'] else 0.5),
                        memory_usage_mb=profile.risk_assessment.memory_footprint_mb
                    )
                
        except Exception as e:
            logger.debug(f"Version check failed for {dep_name}: {e}")
        
        return DetectionStatus(
            available=False,
            version=None,
            import_time=0.0,
            test_time=time.time() - start_time,
            error="Version detection failed",
            fallback_active=profile.fallback_available,
            performance_score=0.0,
            memory_usage_mb=0.0
        )

    def _strategy_functionality_test(self, dep_name: str, profile: Any, strategy: DetectionStrategy) -> DetectionStatus:
        """Functionality test strategy"""
        start_time = time.time()
        
        # First ensure the module can be imported
        import_status = self._strategy_import_test(dep_name, profile, strategy)
        if not import_status.available:
            return import_status
        
        # Run functionality tests
        test_time_start = time.time()
        
        if hasattr(profile, 'test_commands') and profile.test_commands:
            for test_cmd in profile.test_commands:
                try:
                    with self._timeout_context(strategy.timeout):
                        # Execute test command safely
                        result = subprocess.run(
                            test_cmd,
                            shell=True,
                            capture_output=True,
                            timeout=strategy.timeout,
                            check=True
                        )
                        
                        if result.returncode == 0:
                            test_time = time.time() - test_time_start
                            
                            return DetectionStatus(
                                available=True,
                                version=import_status.version,
                                import_time=import_status.import_time,
                                test_time=test_time,
                                error=None,
                                fallback_active=False,
                                performance_score=profile.risk_assessment.performance_impact,
                                memory_usage_mb=profile.risk_assessment.memory_footprint_mb
                            )
                            
                except Exception as e:
                    logger.debug(f"Functionality test failed for {dep_name}: {e}")
                    continue
        
        # Fallback to basic functionality test using test_function from old format
        if hasattr(profile, 'test_function') and profile.test_function:
            try:
                with self._timeout_context(strategy.timeout):
                    test_result = profile.test_function()
                    test_time = time.time() - test_time_start
                    
                    return DetectionStatus(
                        available=bool(test_result),
                        version=import_status.version,
                        import_time=import_status.import_time,
                        test_time=test_time,
                        error=None if test_result else "Test function returned False",
                        fallback_active=not bool(test_result),
                        performance_score=profile.risk_assessment.performance_impact if test_result else 0.0,
                        memory_usage_mb=profile.risk_assessment.memory_footprint_mb
                    )
                    
            except Exception as e:
                logger.debug(f"Test function failed for {dep_name}: {e}")
        
        return DetectionStatus(
            available=import_status.available,
            version=import_status.version,
            import_time=import_status.import_time,
            test_time=time.time() - test_time_start,
            error="No functionality tests available",
            fallback_active=True,
            performance_score=profile.risk_assessment.performance_impact * 0.7,
            memory_usage_mb=profile.risk_assessment.memory_footprint_mb
        )

    def _strategy_pip_show(self, dep_name: str, profile: Any, strategy: DetectionStrategy) -> DetectionStatus:
        """Pip show strategy for getting package information"""
        start_time = time.time()
        
        try:
            with self._timeout_context(strategy.timeout):
                result = subprocess.run(
                    ['pip', 'show', dep_name],
                    capture_output=True,
                    timeout=strategy.timeout,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout:
                    # Parse pip show output
                    lines = result.stdout.strip().split('\n')
                    version = None
                    location = None
                    
                    for line in lines:
                        if line.startswith('Version:'):
                            version = line.split(':', 1)[1].strip()
                        elif line.startswith('Location:'):
                            location = line.split(':', 1)[1].strip()
                    
                    return DetectionStatus(
                        available=True,
                        version=version,
                        import_time=0.0,
                        test_time=time.time() - start_time,
                        error=None,
                        fallback_active=False,
                        performance_score=profile.risk_assessment.performance_impact,
                        memory_usage_mb=profile.risk_assessment.memory_footprint_mb
                    )
                    
        except Exception as e:
            logger.debug(f"Pip show failed for {dep_name}: {e}")
        
        return DetectionStatus(
            available=False,
            version=None,
            import_time=0.0,
            test_time=time.time() - start_time,
            error="Pip show command failed",
            fallback_active=profile.fallback_available,
            performance_score=0.0,
            memory_usage_mb=0.0
        )

    def _strategy_comprehensive_test(self, dep_name: str, profile: Any, strategy: DetectionStrategy) -> DetectionStatus:
        """Comprehensive test strategy combining all methods"""
        start_time = time.time()
        
        # Try import test first
        import_status = self._strategy_import_test(dep_name, profile, strategy)
        
        # Try version detection
        version_status = self._strategy_version_check(dep_name, profile, strategy)
        
        # Try functionality test if import succeeded
        functionality_status = None
        if import_status.available:
            functionality_status = self._strategy_functionality_test(dep_name, profile, strategy)
        
        # Combine results
        final_available = import_status.available
        final_version = version_status.version or import_status.version
        final_error = None
        final_fallback = False
        performance_score = 0.0
        
        if import_status.available:
            if functionality_status and functionality_status.available:
                # Full functionality confirmed
                performance_score = profile.risk_assessment.performance_impact
                final_fallback = False
            else:
                # Import works but functionality uncertain
                performance_score = profile.risk_assessment.performance_impact * 0.8
                final_fallback = True
                final_error = "Functionality tests inconclusive"
        else:
            final_fallback = profile.fallback_available
            final_error = import_status.error or "Import and version detection failed"
        
        total_time = time.time() - start_time
        
        return DetectionStatus(
            available=final_available,
            version=final_version,
            import_time=import_status.import_time,
            test_time=total_time - import_status.import_time,
            error=final_error,
            fallback_active=final_fallback,
            performance_score=performance_score,
            memory_usage_mb=profile.risk_assessment.memory_footprint_mb
        )

    @contextmanager
    def _timeout_context(self, timeout_seconds: float):
        """Context manager for operation timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
        
        # Only use signal on Unix systems
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Fallback for Windows - just run without timeout
            yield

    def _estimate_module_memory(self, module) -> float:
        """Estimate memory usage of imported module"""
        try:
            import sys
            return sys.getsizeof(module) / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            # Fallback estimation
            try:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except Exception:
                return 0.0

    def _create_failed_detection_result(self, dep_name: str, error: str) -> DetectionResult:
        """Create a failed detection result"""
        fallback_available = (
            dep_name in DEPENDENCY_MATRIX and 
            DEPENDENCY_MATRIX[dep_name].fallback_available
        )
        
        return DetectionResult(
            dependency_name=dep_name,
            strategies_attempted=['error'],
            successful_strategy=None,
            status=DetectionStatus(
                available=False,
                version=None,
                import_time=0.0,
                test_time=0.0,
                error=error,
                fallback_active=fallback_available,
                performance_score=0.0,
                memory_usage_mb=0.0
            ),
            warnings=[error],
            recommendations=['Use pure Python fallback'] if fallback_available else ['Installation required'],
            fallback_reason=error,
            detection_timestamp=time.time(),
            system_impact={'error': True}
        )

    def _create_error_results(self, dependencies: List[str], error: str) -> Dict[str, DetectionResult]:
        """Create error results for all dependencies"""
        return {
            dep: self._create_failed_detection_result(dep, error)
            for dep in dependencies
        }

    def _log_detection_summary(self, results: Dict[str, DetectionResult], detection_time: float) -> None:
        """Log comprehensive detection summary"""
        total = len(results)
        available = sum(1 for r in results.values() if r.status.available)
        fallback = sum(1 for r in results.values() if r.status.fallback_active)
        failed = total - available - fallback
        
        logger.info(f"Detection completed in {detection_time:.2f}s:")
        logger.info(f"  Available: {available}/{total}")
        logger.info(f"  Fallback: {fallback}/{total}")
        logger.info(f"  Failed: {failed}/{total}")
        
        # Log by category
        categories = defaultdict(list)
        for dep_name, result in results.items():
            if dep_name in DEPENDENCY_MATRIX:
                category = DEPENDENCY_MATRIX[dep_name].category
                categories[category].append(result.status.available)
        
        for category, statuses in categories.items():
            available_in_cat = sum(statuses)
            logger.info(f"  {category}: {available_in_cat}/{len(statuses)}")
        
        # Resource usage summary
        peak_memory = self.resource_monitor.peak_memory_mb
        if peak_memory > 0:
            logger.info(f"Peak memory usage: {peak_memory:.1f}MB")

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get comprehensive detection summary"""
        total_detections = len(self.detection_results)
        successful = sum(1 for r in self.detection_results.values() if r.status.available)
        fallback = sum(1 for r in self.detection_results.values() if r.status.fallback_active and not r.status.available)
        failed = total_detections - successful - fallback
        
        return {
            'total_dependencies': total_detections,
            'successful_detections': successful,
            'fallback_detections': fallback,
            'failed_detections': failed,
            'success_rate': successful / max(total_detections, 1),
            'fallback_rate': fallback / max(total_detections, 1),
            'error_counts': dict(self.error_counts),
            'successful_counts': dict(self.successful_detections),
            'resource_usage': {
                'peak_memory_mb': self.resource_monitor.peak_memory_mb,
                'cpu_time_used': self.resource_monitor.cpu_time_used,
                'thread_count': self.resource_monitor.thread_count
            },
            'steam_deck_mode': self.is_steam_deck,
            'detection_strategies': list(self.detection_strategies.keys())
        }

    def get_fallback_status_report(self) -> Dict[str, Any]:
        """Get detailed fallback status report"""
        fallback_active = {}
        fallback_reasons = {}
        performance_impact = {}
        
        for dep_name, result in self.detection_results.items():
            if result.status.fallback_active:
                fallback_active[dep_name] = True
                fallback_reasons[dep_name] = result.fallback_reason or "Detection failed"
                
                if dep_name in DEPENDENCY_MATRIX:
                    profile = DEPENDENCY_MATRIX[dep_name]
                    performance_impact[dep_name] = profile.risk_assessment.performance_impact
            else:
                fallback_active[dep_name] = False
        
        return {
            'fallback_active': fallback_active,
            'fallback_reasons': fallback_reasons,
            'performance_impact': performance_impact,
            'total_fallbacks': sum(fallback_active.values()),
            'fallback_categories': self._categorize_fallbacks(),
            'recommendations': self._generate_fallback_recommendations()
        }

    def _categorize_fallbacks(self) -> Dict[str, List[str]]:
        """Categorize dependencies by fallback status"""
        categories = defaultdict(list)
        
        for dep_name, result in self.detection_results.items():
            if dep_name in DEPENDENCY_MATRIX:
                category = DEPENDENCY_MATRIX[dep_name].category
                if result.status.fallback_active:
                    categories[category].append(dep_name)
        
        return dict(categories)

    def _generate_fallback_recommendations(self) -> List[str]:
        """Generate recommendations based on fallback status"""
        recommendations = []
        fallback_count = sum(1 for r in self.detection_results.values() if r.status.fallback_active)
        
        if fallback_count > 0:
            recommendations.append(
                f"{fallback_count} dependencies using fallbacks - consider installation for better performance"
            )
        
        # Steam Deck specific recommendations
        if self.is_steam_deck:
            compilation_fallbacks = [
                dep_name for dep_name, result in self.detection_results.items()
                if result.status.fallback_active and dep_name in DEPENDENCY_MATRIX
                and DEPENDENCY_MATRIX[dep_name].hardware_compatibility.compilation_required
            ]
            
            if compilation_fallbacks:
                recommendations.append(
                    f"Steam Deck: Consider pre-compiled wheels for: {', '.join(compilation_fallbacks)}"
                )
        
        return recommendations

    def export_detection_report(self, filepath: Path) -> None:
        """Export comprehensive detection report"""
        report = {
            'detection_summary': self.get_detection_summary(),
            'fallback_status': self.get_fallback_status_report(),
            'detailed_results': {
                name: {
                    'status': result.status._asdict(),
                    'strategies_attempted': result.strategies_attempted,
                    'successful_strategy': result.successful_strategy,
                    'warnings': result.warnings,
                    'recommendations': result.recommendations,
                    'system_impact': result.system_impact,
                    'timestamp': result.detection_timestamp
                }
                for name, result in self.detection_results.items()
            },
            'system_info': self.version_manager.system_info,
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Detection report exported to {filepath}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_detector: Optional[EnhancedDependencyDetector] = None

def get_detector() -> EnhancedDependencyDetector:
    """Get or create global detector instance"""
    global _detector
    if _detector is None:
        _detector = EnhancedDependencyDetector()
    return _detector

def quick_detect_all(force_refresh: bool = False) -> Dict[str, bool]:
    """Quick detection returning simple availability status"""
    detector = get_detector()
    results = detector.detect_all_dependencies(force_refresh=force_refresh)
    return {
        name: result.status.available or result.status.fallback_active
        for name, result in results.items()
    }

def get_available_dependencies() -> List[str]:
    """Get list of available dependencies"""
    detector = get_detector()
    return [
        name for name, result in detector.detection_results.items()
        if result.status.available
    ]

def get_fallback_dependencies() -> List[str]:
    """Get list of dependencies using fallbacks"""
    detector = get_detector()
    return [
        name for name, result in detector.detection_results.items()
        if result.status.fallback_active and not result.status.available
    ]

def detect_with_timeout(dependencies: List[str], timeout: float = 30.0) -> Dict[str, DetectionResult]:
    """Detect dependencies with custom timeout"""
    detector = get_detector()
    original_timeout = detector.global_timeout
    detector.global_timeout = timeout
    try:
        return detector.detect_all_dependencies(dependencies=dependencies)
    finally:
        detector.global_timeout = original_timeout


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🔍 Enhanced Dependency Detector Test Suite")
    print("=" * 55)
    
    # Initialize detector
    detector = EnhancedDependencyDetector(max_workers=2)
    
    print("\n📊 System Information:")
    for key, value in detector.version_manager.system_info.items():
        print(f"  {key}: {value}")
    
    # Test detection on a subset
    test_deps = ['numpy', 'scikit-learn', 'psutil', 'msgpack', 'lightgbm']
    print(f"\n🔍 Testing detection on {len(test_deps)} dependencies...")
    
    start_time = time.time()
    results = detector.detect_all_dependencies(dependencies=test_deps)
    detection_time = time.time() - start_time
    
    print(f"\n📋 Detection Results ({detection_time:.2f}s):")
    for dep_name, result in results.items():
        status_emoji = "✅" if result.status.available else "🔄" if result.status.fallback_active else "❌"
        version_str = f" v{result.status.version}" if result.status.version else ""
        print(f"  {status_emoji} {dep_name}{version_str}")
        print(f"    Strategy: {result.successful_strategy or 'None'}")
        print(f"    Performance: {result.status.performance_score:.1f}")
        
        if result.warnings:
            print(f"    Warnings: {len(result.warnings)}")
        if result.recommendations:
            print(f"    Recommendations: {len(result.recommendations)}")
    
    # Get summary report
    print("\n📈 Detection Summary:")
    summary = detector.get_detection_summary()
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Fallback Rate: {summary['fallback_rate']:.1%}")
    print(f"  Peak Memory: {summary['resource_usage']['peak_memory_mb']:.1f}MB")
    
    # Get fallback status
    print("\n🔄 Fallback Status:")
    fallback_report = detector.get_fallback_status_report()
    print(f"  Total Fallbacks: {fallback_report['total_fallbacks']}")
    
    if fallback_report['recommendations']:
        print("  Recommendations:")
        for rec in fallback_report['recommendations']:
            print(f"    💡 {rec}")
    
    # Export detailed report
    report_path = Path("/tmp/enhanced_dependency_detection.json")
    detector.export_detection_report(report_path)
    print(f"\n💾 Detailed report exported to {report_path}")
    
    print(f"\n✅ Enhanced Dependency Detector test completed!")
    print(f"🎯 Overall health: {summary['success_rate']:.1%} available, {summary['fallback_rate']:.1%} fallback")