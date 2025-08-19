#!/usr/bin/env python3
"""
Enhanced Dependency Health Checker for ML Shader Prediction Compiler

This module provides comprehensive dependency health checking, validation,
and diagnostic capabilities to ensure all dependencies are correctly installed
and functioning optimally.

Features:
- Complete dependency installation verification
- Version compatibility checking
- Performance benchmarking for each dependency
- Steam Deck specific validation
- Automated issue detection and resolution suggestions
- Runtime capability testing
"""

import os
import sys
import time
import json
import platform
import importlib
import subprocess
import traceback
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# HEALTH CHECK DATA STRUCTURES
# =============================================================================

@dataclass
class DependencyHealth:
    """Health status of a single dependency"""
    name: str
    installed: bool
    version: Optional[str]
    import_success: bool
    test_passed: bool
    performance_score: float  # 0.0 to 1.0
    error_message: Optional[str]
    warnings: List[str]
    suggestions: List[str]
    benchmark_time: Optional[float]

@dataclass
class SystemHealth:
    """Overall system health status"""
    python_version: str
    platform: str
    is_steam_deck: bool
    total_dependencies: int
    installed_dependencies: int
    working_dependencies: int
    missing_dependencies: List[str]
    failed_dependencies: List[str]
    overall_health_score: float  # 0.0 to 1.0
    optimization_available: bool
    warnings: List[str]
    critical_issues: List[str]

# =============================================================================
# DEPENDENCY HEALTH CHECKER
# =============================================================================

class DependencyHealthChecker:
    """
    Comprehensive dependency health checking and validation system
    """
    
    # Core dependencies that must work
    CORE_DEPENDENCIES = [
        'numpy', 'scikit-learn', 'lightgbm', 'psutil'
    ]
    
    # Performance dependencies that enhance operation
    PERFORMANCE_DEPENDENCIES = [
        'numba', 'numexpr', 'bottleneck', 'msgpack', 'zstandard'
    ]
    
    # Optional dependencies
    OPTIONAL_DEPENDENCIES = [
        'pydantic', 'aiofiles', 'dbus-next', 'distro'
    ]
    
    def __init__(self):
        self.health_status: Dict[str, DependencyHealth] = {}
        self.system_health: Optional[SystemHealth] = None
        self.benchmark_results: Dict[str, float] = {}
        self._detect_steam_deck()
        
    def _detect_steam_deck(self) -> None:
        """Detect if running on Steam Deck"""
        self.is_steam_deck = False
        try:
            # Check for Steam Deck specific markers
            if os.path.exists('/etc/steamos-release'):
                self.is_steam_deck = True
            elif 'steamdeck' in platform.node().lower():
                self.is_steam_deck = True
            elif os.path.exists('/home/deck'):
                # Additional check for deck user
                self.is_steam_deck = True
        except Exception:
            pass
    
    def check_dependency(self, name: str, import_name: Optional[str] = None) -> DependencyHealth:
        """Check health of a single dependency"""
        if import_name is None:
            import_name = name.replace('-', '_')
        
        health = DependencyHealth(
            name=name,
            installed=False,
            version=None,
            import_success=False,
            test_passed=False,
            performance_score=0.0,
            error_message=None,
            warnings=[],
            suggestions=[],
            benchmark_time=None
        )
        
        try:
            # Try to import the module
            start_time = time.time()
            module = importlib.import_module(import_name)
            import_time = time.time() - start_time
            
            health.installed = True
            health.import_success = True
            
            # Get version
            for attr in ['__version__', 'VERSION', 'version']:
                if hasattr(module, attr):
                    health.version = str(getattr(module, attr))
                    break
            
            # Run specific tests for each dependency
            health.test_passed = self._run_dependency_test(name, module)
            
            # Benchmark if applicable
            if name in self.PERFORMANCE_DEPENDENCIES:
                health.benchmark_time = self._benchmark_dependency(name, module)
                health.performance_score = self._calculate_performance_score(
                    name, health.benchmark_time
                )
            else:
                health.performance_score = 1.0 if health.test_passed else 0.5
            
            # Check for slow import
            if import_time > 1.0:
                health.warnings.append(f"Slow import time: {import_time:.2f}s")
            
        except ImportError as e:
            health.error_message = str(e)
            health.suggestions.append(f"Install with: pip install {name}")
            
        except Exception as e:
            health.import_success = True  # Module imported but had other issues
            health.installed = True
            health.error_message = f"Runtime error: {str(e)}"
            health.warnings.append("Module imports but has runtime issues")
        
        return health
    
    def _run_dependency_test(self, name: str, module: Any) -> bool:
        """Run specific tests for each dependency"""
        try:
            if name == 'numpy':
                import numpy as np
                arr = np.array([1, 2, 3, 4, 5])
                return np.mean(arr) == 3.0
                
            elif name == 'scikit-learn':
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.datasets import make_regression
                X, y = make_regression(n_samples=10, n_features=4, random_state=42)
                model = RandomForestRegressor(n_estimators=2, random_state=42)
                model.fit(X, y)
                return model.predict(X).shape == y.shape
                
            elif name == 'lightgbm':
                import lightgbm as lgb
                import numpy as np
                X = np.random.rand(10, 5)
                y = np.random.rand(10)
                train_data = lgb.Dataset(X, label=y)
                params = {'objective': 'regression', 'verbose': -1}
                model = lgb.train(params, train_data, num_boost_round=1)
                return model.predict(X).shape == y.shape
                
            elif name == 'numba':
                from numba import njit
                @njit
                def test_func(x):
                    return x * 2
                return test_func(5) == 10
                
            elif name == 'numexpr':
                import numexpr as ne
                import numpy as np
                a = np.array([1, 2, 3])
                b = np.array([4, 5, 6])
                result = ne.evaluate('a + b')
                return np.array_equal(result, np.array([5, 7, 9]))
                
            elif name == 'bottleneck':
                import bottleneck as bn
                import numpy as np
                arr = np.array([1, 2, np.nan, 4, 5])
                return bn.nanmean(arr) == 3.0
                
            elif name == 'msgpack':
                import msgpack
                data = {'test': 123, 'list': [1, 2, 3]}
                packed = msgpack.packb(data)
                unpacked = msgpack.unpackb(packed)
                # Handle both string and bytes keys depending on msgpack version
                return unpacked.get('test', unpacked.get(b'test')) == 123
                
            elif name == 'zstandard':
                import zstandard as zstd
                data = b'test data for compression'
                cctx = zstd.ZstdCompressor()
                compressed = cctx.compress(data)
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(compressed)
                return decompressed == data
                
            elif name == 'psutil':
                import psutil
                return psutil.cpu_count() > 0 and psutil.virtual_memory().total > 0
                
            elif name == 'pydantic':
                from pydantic import BaseModel
                class TestModel(BaseModel):
                    value: int
                model = TestModel(value=42)
                return model.value == 42
                
            elif name == 'aiofiles':
                return hasattr(module, 'open')
                
            elif name == 'dbus-next':
                return hasattr(module, 'Message')
                
            elif name == 'distro':
                return hasattr(module, 'name')
                
            else:
                # Generic test - just check if module has expected attributes
                return True
                
        except Exception as e:
            logger.debug(f"Test failed for {name}: {e}")
            return False
    
    def _benchmark_dependency(self, name: str, module: Any) -> float:
        """Benchmark performance-critical dependencies"""
        try:
            iterations = 1000
            
            if name == 'numpy':
                import numpy as np
                arr = np.random.rand(1000, 100)
                start = time.time()
                for _ in range(iterations):
                    np.mean(arr, axis=1)
                return (time.time() - start) / iterations
                
            elif name == 'numba':
                from numba import njit
                import numpy as np
                @njit
                def compute(arr):
                    return np.sum(arr ** 2)
                arr = np.random.rand(1000)
                # Warm up JIT
                compute(arr)
                start = time.time()
                for _ in range(iterations):
                    compute(arr)
                return (time.time() - start) / iterations
                
            elif name == 'numexpr':
                import numexpr as ne
                import numpy as np
                a = np.random.rand(10000)
                b = np.random.rand(10000)
                start = time.time()
                for _ in range(iterations):
                    ne.evaluate('a * b + a - b')
                return (time.time() - start) / iterations
                
            else:
                return 0.0
                
        except Exception:
            return float('inf')
    
    def _calculate_performance_score(self, name: str, benchmark_time: float) -> float:
        """Calculate performance score based on benchmark"""
        if benchmark_time == float('inf'):
            return 0.0
        
        # Define expected times (in seconds) for good performance
        expected_times = {
            'numpy': 0.0001,
            'numba': 0.00001,
            'numexpr': 0.00005,
            'bottleneck': 0.00005,
        }
        
        if name not in expected_times:
            return 1.0
        
        expected = expected_times[name]
        if benchmark_time <= expected:
            return 1.0
        elif benchmark_time <= expected * 2:
            return 0.8
        elif benchmark_time <= expected * 5:
            return 0.6
        elif benchmark_time <= expected * 10:
            return 0.4
        else:
            return 0.2
    
    def check_all_dependencies(self) -> SystemHealth:
        """Check health of all dependencies"""
        all_deps = (
            self.CORE_DEPENDENCIES + 
            self.PERFORMANCE_DEPENDENCIES + 
            self.OPTIONAL_DEPENDENCIES
        )
        
        # Check each dependency
        for dep in all_deps:
            import_name = dep.replace('-', '_')
            if dep == 'scikit-learn':
                import_name = 'sklearn'
            self.health_status[dep] = self.check_dependency(dep, import_name)
        
        # Calculate system health
        total = len(all_deps)
        installed = sum(1 for h in self.health_status.values() if h.installed)
        working = sum(1 for h in self.health_status.values() if h.test_passed)
        
        missing = [name for name, h in self.health_status.items() if not h.installed]
        failed = [name for name, h in self.health_status.items() 
                 if h.installed and not h.test_passed]
        
        # Check for critical issues
        critical_issues = []
        warnings = []
        
        for dep in self.CORE_DEPENDENCIES:
            if dep in missing:
                critical_issues.append(f"Core dependency '{dep}' is not installed")
            elif dep in failed:
                critical_issues.append(f"Core dependency '{dep}' is not working properly")
        
        # Check for performance issues
        slow_deps = [name for name, h in self.health_status.items() 
                    if h.performance_score < 0.5 and h.installed]
        if slow_deps:
            warnings.append(f"Performance issues detected in: {', '.join(slow_deps)}")
        
        # Calculate overall health score
        core_score = sum(1 for d in self.CORE_DEPENDENCIES 
                        if self.health_status.get(d, DependencyHealth(d, False, None, False, False, 0, None, [], [], None)).test_passed) / len(self.CORE_DEPENDENCIES)
        perf_score = sum(self.health_status.get(d, DependencyHealth(d, False, None, False, False, 0, None, [], [], None)).performance_score 
                        for d in self.PERFORMANCE_DEPENDENCIES) / len(self.PERFORMANCE_DEPENDENCIES)
        
        overall_score = (core_score * 0.6 + perf_score * 0.4)
        
        self.system_health = SystemHealth(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.system(),
            is_steam_deck=self.is_steam_deck,
            total_dependencies=total,
            installed_dependencies=installed,
            working_dependencies=working,
            missing_dependencies=missing,
            failed_dependencies=failed,
            overall_health_score=overall_score,
            optimization_available=len(self.PERFORMANCE_DEPENDENCIES) > 0,
            warnings=warnings,
            critical_issues=critical_issues
        )
        
        return self.system_health
    
    def generate_report(self) -> str:
        """Generate a detailed health report"""
        if not self.system_health:
            self.check_all_dependencies()
        
        report = []
        report.append("=" * 80)
        report.append("ML SHADER PREDICTION COMPILER - DEPENDENCY HEALTH REPORT")
        report.append("=" * 80)
        report.append("")
        
        # System information
        report.append("SYSTEM INFORMATION:")
        report.append(f"  Python Version: {self.system_health.python_version}")
        report.append(f"  Platform: {self.system_health.platform}")
        report.append(f"  Steam Deck: {'Yes' if self.system_health.is_steam_deck else 'No'}")
        report.append("")
        
        # Overall health
        report.append("OVERALL HEALTH:")
        report.append(f"  Health Score: {self.system_health.overall_health_score:.1%}")
        report.append(f"  Dependencies: {self.system_health.working_dependencies}/{self.system_health.total_dependencies} working")
        report.append("")
        
        # Critical issues
        if self.system_health.critical_issues:
            report.append("CRITICAL ISSUES:")
            for issue in self.system_health.critical_issues:
                report.append(f"  ✗ {issue}")
            report.append("")
        
        # Warnings
        if self.system_health.warnings:
            report.append("WARNINGS:")
            for warning in self.system_health.warnings:
                report.append(f"  ⚠ {warning}")
            report.append("")
        
        # Dependency details
        report.append("DEPENDENCY STATUS:")
        report.append("-" * 80)
        
        for category, deps in [
            ("Core Dependencies", self.CORE_DEPENDENCIES),
            ("Performance Dependencies", self.PERFORMANCE_DEPENDENCIES),
            ("Optional Dependencies", self.OPTIONAL_DEPENDENCIES)
        ]:
            report.append(f"\n{category}:")
            for dep in deps:
                health = self.health_status.get(dep)
                if health:
                    status = "✓" if health.test_passed else "✗" if health.installed else "○"
                    version = health.version or "unknown"
                    perf = f" (perf: {health.performance_score:.0%})" if health.benchmark_time is not None else ""
                    report.append(f"  {status} {dep:20} {version:15}{perf}")
                    
                    if health.error_message:
                        report.append(f"     Error: {health.error_message}")
                    for warning in health.warnings:
                        report.append(f"     Warning: {warning}")
                    for suggestion in health.suggestions:
                        report.append(f"     → {suggestion}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing dependencies that should be installed"""
        if not self.system_health:
            self.check_all_dependencies()
        return self.system_health.missing_dependencies
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for optimizing dependency setup"""
        suggestions = []
        
        if not self.system_health:
            self.check_all_dependencies()
        
        # Check for missing performance dependencies
        missing_perf = [d for d in self.PERFORMANCE_DEPENDENCIES 
                       if d in self.system_health.missing_dependencies]
        if missing_perf:
            suggestions.append(f"Install performance dependencies for better speed: {', '.join(missing_perf)}")
        
        # Check for slow dependencies
        slow_deps = [(name, health.performance_score) 
                    for name, health in self.health_status.items() 
                    if health.performance_score < 0.5 and health.installed]
        if slow_deps:
            for dep, score in slow_deps:
                suggestions.append(f"Optimize {dep} configuration (current performance: {score:.0%})")
        
        # Steam Deck specific suggestions
        if self.is_steam_deck:
            if 'numba' in self.system_health.missing_dependencies:
                suggestions.append("Install numba for Steam Deck GPU acceleration support")
        
        return suggestions

# =============================================================================
# QUICK CHECK FUNCTION
# =============================================================================

def quick_health_check() -> Tuple[bool, float, List[str]]:
    """
    Perform a quick health check and return status
    
    Returns:
        Tuple of (all_good, health_score, issues)
    """
    checker = DependencyHealthChecker()
    health = checker.check_all_dependencies()
    
    all_good = len(health.critical_issues) == 0
    return all_good, health.overall_health_score, health.critical_issues

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run health check
    checker = DependencyHealthChecker()
    checker.check_all_dependencies()
    
    # Print report
    print(checker.generate_report())
    
    # Print suggestions if any
    suggestions = checker.get_optimization_suggestions()
    if suggestions:
        print("\nOPTIMIZATION SUGGESTIONS:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
    
    # Exit with appropriate code
    sys.exit(0 if checker.system_health.overall_health_score >= 0.8 else 1)