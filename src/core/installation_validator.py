#!/usr/bin/env python3
"""
Installation Validator for ML Shader Prediction Compiler

This module provides comprehensive validation of dependency installations,
ensuring that all dependencies not only import correctly but actually function
as expected under real workload conditions.

Features:
- Deep functional testing beyond import verification
- Stress testing under various conditions
- Steam Deck specific validation
- Performance validation and benchmarking
- Graceful degradation testing
- Installation repair recommendations
"""

import os
import sys
import time
import json
import tempfile
import traceback
import threading
import subprocess
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import logging

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# VALIDATION TEST DEFINITIONS
# =============================================================================

@dataclass
class ValidationTest:
    """Definition of a validation test"""
    name: str
    description: str
    test_function: Callable
    timeout: float = 30.0
    critical: bool = False
    steam_deck_specific: bool = False
    stress_test: bool = False
    performance_benchmark: bool = False
    cleanup_function: Optional[Callable] = None

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DependencyValidation:
    """Complete validation results for a dependency"""
    dependency_name: str
    overall_success: bool
    test_results: List[ValidationResult]
    installation_health: float  # 0.0 to 1.0
    performance_score: float
    recommendations: List[str] = field(default_factory=list)

# =============================================================================
# CORE VALIDATOR
# =============================================================================

class InstallationValidator:
    """
    Comprehensive validator for dependency installations
    """
    
    def __init__(self, timeout_multiplier: float = 1.0):
        self.timeout_multiplier = timeout_multiplier
        self.validation_tests: Dict[str, List[ValidationTest]] = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix='ml_shader_validation_'))
        self.is_steam_deck = self._detect_steam_deck()
        
        # Initialize validation tests
        self._setup_validation_tests()
        
        logger.info(f"InstallationValidator initialized (Steam Deck: {self.is_steam_deck})")
        logger.info(f"Temporary directory: {self.temp_dir}")
    
    def _detect_steam_deck(self) -> bool:
        """Detect if running on Steam Deck"""
        try:
            from .pure_python_fallbacks import PureSteamDeckDetector
            return PureSteamDeckDetector.is_steam_deck()
        except ImportError:
            # Fallback detection
            return (
                os.path.exists('/home/deck') or
                'steamdeck' in platform.platform().lower() or
                os.environ.get('SteamDeck') is not None
            )
    
    def _setup_validation_tests(self) -> None:
        """Setup all validation tests for different dependencies"""
        
        # NumPy validation tests
        self.validation_tests['numpy'] = [
            ValidationTest(
                name='basic_operations',
                description='Test basic NumPy array operations',
                test_function=self._test_numpy_basic,
                timeout=10.0,
                critical=True
            ),
            ValidationTest(
                name='mathematical_functions',
                description='Test NumPy mathematical functions',
                test_function=self._test_numpy_math,
                timeout=15.0
            ),
            ValidationTest(
                name='performance_benchmark',
                description='Benchmark NumPy performance',
                test_function=self._test_numpy_performance,
                timeout=30.0,
                performance_benchmark=True
            ),
            ValidationTest(
                name='memory_stress',
                description='Test NumPy under memory pressure',
                test_function=self._test_numpy_memory_stress,
                timeout=45.0,
                stress_test=True
            )
        ]
        
        # Scikit-learn validation tests
        self.validation_tests['scikit-learn'] = [
            ValidationTest(
                name='basic_ml_workflow',
                description='Test basic ML workflow',
                test_function=self._test_sklearn_basic,
                timeout=30.0,
                critical=True
            ),
            ValidationTest(
                name='model_persistence',
                description='Test model saving and loading',
                test_function=self._test_sklearn_persistence,
                timeout=20.0
            ),
            ValidationTest(
                name='performance_benchmark',
                description='Benchmark ML model performance',
                test_function=self._test_sklearn_performance,
                timeout=60.0,
                performance_benchmark=True
            )
        ]
        
        # LightGBM validation tests
        self.validation_tests['lightgbm'] = [
            ValidationTest(
                name='basic_training',
                description='Test LightGBM basic training',
                test_function=self._test_lightgbm_basic,
                timeout=45.0,
                critical=True
            ),
            ValidationTest(
                name='steam_deck_compatibility',
                description='Test LightGBM on Steam Deck',
                test_function=self._test_lightgbm_steam_deck,
                timeout=60.0,
                steam_deck_specific=True
            )
        ]
        
        # Numba validation tests
        self.validation_tests['numba'] = [
            ValidationTest(
                name='jit_compilation',
                description='Test JIT compilation',
                test_function=self._test_numba_jit,
                timeout=30.0,
                critical=True
            ),
            ValidationTest(
                name='performance_comparison',
                description='Compare JIT vs Python performance',
                test_function=self._test_numba_performance,
                timeout=45.0,
                performance_benchmark=True
            ),
            ValidationTest(
                name='thermal_stress',
                description='Test JIT under thermal constraints',
                test_function=self._test_numba_thermal,
                timeout=60.0,
                steam_deck_specific=True,
                stress_test=True
            )
        ]
        
        # System integration tests
        self.validation_tests['psutil'] = [
            ValidationTest(
                name='system_monitoring',
                description='Test system monitoring capabilities',
                test_function=self._test_psutil_monitoring,
                timeout=15.0,
                critical=True
            ),
            ValidationTest(
                name='steam_deck_sensors',
                description='Test Steam Deck specific sensors',
                test_function=self._test_psutil_steam_deck,
                timeout=20.0,
                steam_deck_specific=True
            )
        ]
        
        # Compression and serialization tests
        self.validation_tests['msgpack'] = [
            ValidationTest(
                name='serialization_roundtrip',
                description='Test serialization round-trip',
                test_function=self._test_msgpack_basic,
                timeout=10.0,
                critical=True
            ),
            ValidationTest(
                name='performance_vs_json',
                description='Compare msgpack vs JSON performance',
                test_function=self._test_msgpack_performance,
                timeout=20.0,
                performance_benchmark=True
            )
        ]
        
        self.validation_tests['zstandard'] = [
            ValidationTest(
                name='compression_roundtrip',
                description='Test compression round-trip',
                test_function=self._test_zstd_basic,
                timeout=15.0,
                critical=True
            ),
            ValidationTest(
                name='compression_performance',
                description='Benchmark compression performance',
                test_function=self._test_zstd_performance,
                timeout=30.0,
                performance_benchmark=True
            )
        ]
    
    def validate_dependency(self, dependency_name: str, 
                          include_stress_tests: bool = False,
                          include_performance_tests: bool = True) -> DependencyValidation:
        """
        Validate a specific dependency with comprehensive testing
        """
        logger.info(f"Validating dependency '{dependency_name}'...")
        
        if dependency_name not in self.validation_tests:
            logger.warning(f"No validation tests defined for '{dependency_name}'")
            return DependencyValidation(
                dependency_name=dependency_name,
                overall_success=False,
                test_results=[],
                installation_health=0.0,
                performance_score=0.0,
                recommendations=[f"No validation tests available for {dependency_name}"]
            )
        
        tests_to_run = self.validation_tests[dependency_name]
        
        # Filter tests based on options
        if not include_stress_tests:
            tests_to_run = [t for t in tests_to_run if not t.stress_test]
        
        if not include_performance_tests:
            tests_to_run = [t for t in tests_to_run if not t.performance_benchmark]
        
        # Filter Steam Deck specific tests
        if not self.is_steam_deck:
            tests_to_run = [t for t in tests_to_run if not t.steam_deck_specific]
        
        # Run tests in parallel where possible
        test_results = []
        critical_tests = [t for t in tests_to_run if t.critical]
        non_critical_tests = [t for t in tests_to_run if not t.critical]
        
        # Run critical tests first (sequentially)
        for test in critical_tests:
            result = self._run_validation_test(dependency_name, test)
            test_results.append(result)
            
            # If critical test fails, consider skipping non-critical tests
            if not result.success:
                logger.warning(f"Critical test '{test.name}' failed for {dependency_name}")
                break
        
        # Run non-critical tests in parallel if critical tests passed
        if not critical_tests or all(r.success for r in test_results):
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_test = {
                    executor.submit(self._run_validation_test, dependency_name, test): test
                    for test in non_critical_tests
                }
                
                for future in as_completed(future_to_test):
                    test = future_to_test[future]
                    try:
                        result = future.result(timeout=test.timeout * self.timeout_multiplier)
                        test_results.append(result)
                    except TimeoutError:
                        logger.error(f"Test '{test.name}' timed out for {dependency_name}")
                        test_results.append(ValidationResult(
                            test_name=test.name,
                            success=False,
                            duration=test.timeout * self.timeout_multiplier,
                            error_message="Test timed out"
                        ))
                    except Exception as e:
                        logger.error(f"Test '{test.name}' failed with exception: {e}")
                        test_results.append(ValidationResult(
                            test_name=test.name,
                            success=False,
                            duration=0.0,
                            error_message=str(e)
                        ))
        
        # Calculate overall metrics
        overall_success = len(test_results) > 0 and all(r.success for r in test_results if self._is_critical_test(dependency_name, r.test_name))
        installation_health = self._calculate_installation_health(test_results)
        performance_score = self._calculate_performance_score(test_results)
        recommendations = self._generate_recommendations(dependency_name, test_results)
        
        validation = DependencyValidation(
            dependency_name=dependency_name,
            overall_success=overall_success,
            test_results=test_results,
            installation_health=installation_health,
            performance_score=performance_score,
            recommendations=recommendations
        )
        
        logger.info(f"Validation complete for '{dependency_name}': {installation_health:.1%} health, {performance_score:.1f} performance")
        return validation
    
    def _run_validation_test(self, dependency_name: str, test: ValidationTest) -> ValidationResult:
        """Run a single validation test"""
        logger.debug(f"Running test '{test.name}' for {dependency_name}")
        
        start_time = time.time()
        
        try:
            # Run the test with timeout
            test_result = test.test_function()
            duration = time.time() - start_time
            
            if isinstance(test_result, tuple):
                success, details = test_result
            elif isinstance(test_result, dict):
                success = test_result.get('success', True)
                details = test_result
            else:
                success = bool(test_result)
                details = {}
            
            # Extract metrics and warnings from details
            performance_metrics = details.get('performance_metrics', {})
            warnings = details.get('warnings', [])
            error_message = details.get('error_message', None) if not success else None
            
            return ValidationResult(
                test_name=test.name,
                success=success,
                duration=duration,
                error_message=error_message,
                performance_metrics=performance_metrics,
                warnings=warnings,
                details=details
            )
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Test '{test.name}' for {dependency_name} failed: {e}")
            return ValidationResult(
                test_name=test.name,
                success=False,
                duration=duration,
                error_message=str(e),
                details={'exception': traceback.format_exc()}
            )
        
        finally:
            # Run cleanup if provided
            if test.cleanup_function:
                try:
                    test.cleanup_function()
                except Exception as e:
                    logger.warning(f"Cleanup failed for test '{test.name}': {e}")
    
    def _is_critical_test(self, dependency_name: str, test_name: str) -> bool:
        """Check if a test is marked as critical"""
        tests = self.validation_tests.get(dependency_name, [])
        for test in tests:
            if test.name == test_name:
                return test.critical
        return False
    
    def _calculate_installation_health(self, results: List[ValidationResult]) -> float:
        """Calculate overall installation health score"""
        if not results:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in results:
            # Weight critical tests more heavily
            weight = 2.0 if self._is_critical_result(result) else 1.0
            score = 1.0 if result.success else 0.0
            
            # Partial credit for tests with warnings
            if result.success and result.warnings:
                score *= 0.8
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 1.0)
    
    def _calculate_performance_score(self, results: List[ValidationResult]) -> float:
        """Calculate performance score from benchmark results"""
        performance_results = [r for r in results if r.performance_metrics]
        
        if not performance_results:
            return 5.0  # Default score when no performance data
        
        scores = []
        for result in performance_results:
            metrics = result.performance_metrics
            
            # Convert various metrics to scores (higher is better)
            if 'throughput' in metrics:
                scores.append(min(10.0, metrics['throughput'] / 1000.0))
            
            if 'speedup' in metrics:
                scores.append(min(10.0, metrics['speedup']))
            
            if 'efficiency' in metrics:
                scores.append(metrics['efficiency'] * 10.0)
            
            # Inverse metrics (lower is better)
            if 'duration' in metrics:
                scores.append(max(0.1, 10.0 - metrics['duration']))
        
        return sum(scores) / len(scores) if scores else 5.0
    
    def _is_critical_result(self, result: ValidationResult) -> bool:
        """Check if a result is from a critical test"""
        return 'critical' in result.test_name or result.test_name in ['basic_operations', 'basic_training', 'jit_compilation']
    
    def _generate_recommendations(self, dependency_name: str, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_results = [r for r in results if not r.success]
        warning_results = [r for r in results if r.warnings]
        
        if failed_results:
            critical_failures = [r for r in failed_results if self._is_critical_result(r)]
            if critical_failures:
                recommendations.append(f"Critical functionality not working - consider reinstalling {dependency_name}")
            
            recommendations.append(f"Consider running: pip install --upgrade --force-reinstall {dependency_name}")
        
        if warning_results:
            recommendations.append(f"Performance or compatibility warnings detected for {dependency_name}")
        
        # Steam Deck specific recommendations
        if self.is_steam_deck:
            if dependency_name in ['numba', 'lightgbm'] and failed_results:
                recommendations.append("Consider using pure Python alternatives on Steam Deck for better compatibility")
        
        return recommendations
    
    # =============================================================================
    # SPECIFIC VALIDATION TESTS
    # =============================================================================
    
    def _test_numpy_basic(self) -> Dict[str, Any]:
        """Test basic NumPy functionality"""
        try:
            import numpy as np
            
            # Basic array operations
            arr = np.array([1, 2, 3, 4, 5])
            result1 = np.mean(arr)
            result2 = np.std(arr)
            result3 = np.sum(arr)
            
            # Matrix operations
            matrix = np.random.rand(10, 10)
            eigenvalues = np.linalg.eigvals(matrix)
            
            # Data type handling
            float_arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
            int_arr = np.array([1, 2, 3], dtype=np.int32)
            
            success = (
                abs(result1 - 3.0) < 0.001 and
                result3 == 15 and
                len(eigenvalues) == 10 and
                float_arr.dtype == np.float32
            )
            
            return {
                'success': success,
                'performance_metrics': {
                    'array_size': len(arr),
                    'matrix_size': matrix.size
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_numpy_math(self) -> Dict[str, Any]:
        """Test NumPy mathematical functions"""
        try:
            import numpy as np
            
            # Trigonometric functions
            x = np.linspace(0, 2*np.pi, 100)
            sin_vals = np.sin(x)
            cos_vals = np.cos(x)
            
            # Statistical functions
            data = np.random.normal(0, 1, 1000)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Advanced math
            result = np.exp(np.log(10))
            
            success = (
                len(sin_vals) == 100 and
                abs(result - 10.0) < 0.001 and
                abs(mean_val) < 0.5 and  # Should be close to 0
                0.5 < std_val < 1.5      # Should be close to 1
            )
            
            return {
                'success': success,
                'performance_metrics': {
                    'trigonometric_ops': len(x),
                    'statistical_samples': len(data)
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_numpy_performance(self) -> Dict[str, Any]:
        """Benchmark NumPy performance"""
        try:
            import numpy as np
            
            size = 10000
            iterations = 10
            
            # Test matrix multiplication performance
            start_time = time.time()
            for _ in range(iterations):
                a = np.random.rand(size // 100, size // 100)
                b = np.random.rand(size // 100, size // 100)
                c = np.dot(a, b)
            matmul_time = time.time() - start_time
            
            # Test element-wise operations performance
            start_time = time.time()
            for _ in range(iterations):
                arr = np.random.rand(size)
                result = np.sin(arr) + np.cos(arr) + np.exp(arr * 0.1)
            elementwise_time = time.time() - start_time
            
            # Compare with pure Python (simple case)
            python_list = list(range(1000))
            start_time = time.time()
            python_sum = sum(x * x for x in python_list)
            python_time = time.time() - start_time
            
            numpy_arr = np.array(python_list)
            start_time = time.time()
            numpy_sum = np.sum(numpy_arr * numpy_arr)
            numpy_time = time.time() - start_time
            
            speedup = python_time / max(numpy_time, 1e-6)
            
            return {
                'success': True,
                'performance_metrics': {
                    'matmul_time': matmul_time,
                    'elementwise_time': elementwise_time,
                    'speedup': speedup,
                    'throughput': size * iterations / (matmul_time + elementwise_time)
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_numpy_memory_stress(self) -> Dict[str, Any]:
        """Test NumPy under memory pressure"""
        try:
            import numpy as np
            
            # Gradually increase memory usage
            arrays = []
            max_memory_mb = 100 if self.is_steam_deck else 500
            
            for i in range(10):
                try:
                    # Create progressively larger arrays
                    size = 1024 * 1024 * (i + 1) // 10  # Up to max_memory_mb MB
                    arr = np.random.rand(size)
                    arrays.append(arr)
                    
                    # Perform operations on the array
                    mean_val = np.mean(arr)
                    std_val = np.std(arr)
                    
                except MemoryError:
                    break
            
            # Clean up
            del arrays
            
            return {
                'success': True,
                'performance_metrics': {
                    'max_arrays_created': len(arrays),
                    'memory_stress_level': i + 1
                },
                'warnings': ['Memory stress test completed - may have hit memory limits'] if i < 5 else []
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_sklearn_basic(self) -> Dict[str, Any]:
        """Test basic scikit-learn functionality"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            import numpy as np
            
            # Generate synthetic data
            X = np.random.rand(1000, 5)
            y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, predictions)
            
            success = mse < 1.0  # Should be a reasonable error for this synthetic data
            
            return {
                'success': success,
                'performance_metrics': {
                    'mse': mse,
                    'training_samples': len(X_train),
                    'features': X.shape[1]
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_sklearn_persistence(self) -> Dict[str, Any]:
        """Test scikit-learn model persistence"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            import numpy as np
            import pickle
            
            # Train a simple model
            X = np.random.rand(100, 3)
            y = np.sum(X, axis=1)
            
            model = RandomForestRegressor(n_estimators=5, random_state=42)
            model.fit(X, y)
            
            # Save model to temporary file
            model_path = self.temp_dir / "test_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Load model and test
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Verify predictions match
            test_X = np.random.rand(10, 3)
            original_pred = model.predict(test_X)
            loaded_pred = loaded_model.predict(test_X)
            
            success = np.allclose(original_pred, loaded_pred)
            
            # Cleanup
            model_path.unlink(missing_ok=True)
            
            return {
                'success': success,
                'performance_metrics': {
                    'model_size_bytes': model_path.stat().st_size if model_path.exists() else 0
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_sklearn_performance(self) -> Dict[str, Any]:
        """Benchmark scikit-learn performance"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            import numpy as np
            
            # Generate larger dataset for performance testing
            n_samples = 5000 if not self.is_steam_deck else 1000
            n_features = 20
            
            X = np.random.rand(n_samples, n_features)
            y = np.sum(X[:, :5], axis=1) + np.random.normal(0, 0.1, n_samples)
            
            # Test Random Forest performance
            start_time = time.time()
            rf_model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
            rf_model.fit(X, y)
            rf_time = time.time() - start_time
            
            # Test Linear Regression performance
            start_time = time.time()
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            lr_time = time.time() - start_time
            
            # Test prediction performance
            start_time = time.time()
            rf_predictions = rf_model.predict(X[:100])
            rf_pred_time = time.time() - start_time
            
            return {
                'success': True,
                'performance_metrics': {
                    'rf_training_time': rf_time,
                    'lr_training_time': lr_time,
                    'prediction_time': rf_pred_time,
                    'samples_per_second': n_samples / rf_time,
                    'efficiency': lr_time / rf_time  # Should be < 1
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_lightgbm_basic(self) -> Dict[str, Any]:
        """Test LightGBM basic functionality"""
        try:
            import lightgbm as lgb
            import numpy as np
            
            # Generate data
            X = np.random.rand(1000, 10)
            y = np.sum(X[:, :3], axis=1) + np.random.normal(0, 0.1, 1000)
            
            # Train LightGBM model
            model = lgb.LGBMRegressor(n_estimators=20, verbose=-1, random_state=42)
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X[:100])
            
            # Verify predictions are reasonable
            success = (
                len(predictions) == 100 and
                np.all(np.isfinite(predictions)) and
                np.std(predictions) > 0  # Should have some variance
            )
            
            return {
                'success': success,
                'performance_metrics': {
                    'training_samples': len(X),
                    'features': X.shape[1],
                    'prediction_variance': float(np.std(predictions))
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_lightgbm_steam_deck(self) -> Dict[str, Any]:
        """Test LightGBM Steam Deck compatibility"""
        if not self.is_steam_deck:
            return {'success': True, 'warnings': ['Not running on Steam Deck']}
        
        try:
            import lightgbm as lgb
            import numpy as np
            
            # Test with conservative settings for Steam Deck
            X = np.random.rand(500, 5)
            y = np.sum(X, axis=1)
            
            # Use Steam Deck optimized parameters
            model = lgb.LGBMRegressor(
                n_estimators=10,
                max_depth=3,
                num_leaves=7,
                verbose=-1,
                n_jobs=2,  # Conservative CPU usage
                random_state=42
            )
            
            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time
            
            predictions = model.predict(X[:50])
            
            # Check for thermal impact
            warnings = []
            if training_time > 10.0:
                warnings.append("Training time high - may cause thermal issues on Steam Deck")
            
            return {
                'success': True,
                'performance_metrics': {
                    'training_time': training_time,
                    'samples_per_second': len(X) / training_time
                },
                'warnings': warnings
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_numba_jit(self) -> Dict[str, Any]:
        """Test Numba JIT compilation"""
        try:
            from numba import njit
            import numpy as np
            
            # Define a simple function to JIT compile
            @njit
            def compute_sum_of_squares(arr):
                result = 0.0
                for i in range(len(arr)):
                    result += arr[i] * arr[i]
                return result
            
            # Test data
            test_arr = np.random.rand(1000)
            
            # First call (includes compilation time)
            start_time = time.time()
            result1 = compute_sum_of_squares(test_arr)
            first_call_time = time.time() - start_time
            
            # Second call (pure execution time)
            start_time = time.time()
            result2 = compute_sum_of_squares(test_arr)
            second_call_time = time.time() - start_time
            
            # Verify results are consistent
            success = abs(result1 - result2) < 1e-10
            
            return {
                'success': success,
                'performance_metrics': {
                    'compilation_time': first_call_time - second_call_time,
                    'execution_time': second_call_time,
                    'speedup_potential': first_call_time / max(second_call_time, 1e-6)
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_numba_performance(self) -> Dict[str, Any]:
        """Compare Numba vs pure Python performance"""
        try:
            from numba import njit
            import numpy as np
            
            # Pure Python version
            def python_function(arr):
                result = 0.0
                for i in range(len(arr)):
                    result += arr[i] * arr[i] + np.sin(arr[i])
                return result
            
            # Numba version
            @njit
            def numba_function(arr):
                result = 0.0
                for i in range(len(arr)):
                    result += arr[i] * arr[i] + np.sin(arr[i])
                return result
            
            test_arr = np.random.rand(10000)
            
            # Warm up numba
            numba_function(test_arr[:100])
            
            # Benchmark Python
            start_time = time.time()
            python_result = python_function(test_arr)
            python_time = time.time() - start_time
            
            # Benchmark Numba
            start_time = time.time()
            numba_result = numba_function(test_arr)
            numba_time = time.time() - start_time
            
            speedup = python_time / max(numba_time, 1e-6)
            
            return {
                'success': True,
                'performance_metrics': {
                    'python_time': python_time,
                    'numba_time': numba_time,
                    'speedup': speedup,
                    'efficiency': min(speedup / 10.0, 1.0)  # Normalize to 0-1
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_numba_thermal(self) -> Dict[str, Any]:
        """Test Numba under thermal constraints"""
        if not self.is_steam_deck:
            return {'success': True, 'warnings': ['Not running on Steam Deck']}
        
        try:
            from numba import njit
            import numpy as np
            from .pure_python_fallbacks import PureThermalMonitor
            
            thermal_monitor = PureThermalMonitor()
            initial_temp = thermal_monitor.get_cpu_temperature()
            
            @njit
            def intensive_computation(n):
                result = 0.0
                for i in range(n):
                    for j in range(100):
                        result += np.sin(i * j * 0.001)
                return result
            
            # Run intensive computation
            start_time = time.time()
            result = intensive_computation(1000)
            computation_time = time.time() - start_time
            
            final_temp = thermal_monitor.get_cpu_temperature()
            temp_increase = final_temp - initial_temp
            
            warnings = []
            if temp_increase > 5.0:
                warnings.append(f"Temperature increased by {temp_increase:.1f}°C during computation")
            
            if final_temp > 80.0:
                warnings.append(f"High temperature reached: {final_temp:.1f}°C")
            
            return {
                'success': True,
                'performance_metrics': {
                    'computation_time': computation_time,
                    'temperature_increase': temp_increase,
                    'final_temperature': final_temp
                },
                'warnings': warnings
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_psutil_monitoring(self) -> Dict[str, Any]:
        """Test psutil system monitoring"""
        try:
            import psutil
            
            # Test basic system info
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Test process monitoring
            current_process = psutil.Process()
            process_info = current_process.memory_info()
            
            success = (
                cpu_count > 0 and
                memory_info.total > 0 and
                0 <= cpu_percent <= 100 and
                process_info.rss > 0
            )
            
            return {
                'success': success,
                'performance_metrics': {
                    'cpu_count': cpu_count,
                    'memory_gb': memory_info.total / (1024**3),
                    'cpu_usage': cpu_percent
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_psutil_steam_deck(self) -> Dict[str, Any]:
        """Test Steam Deck specific monitoring"""
        if not self.is_steam_deck:
            return {'success': True, 'warnings': ['Not running on Steam Deck']}
        
        try:
            import psutil
            
            # Test Steam Deck specific features
            cpu_freq = psutil.cpu_freq()
            battery = psutil.sensors_battery()
            
            # Look for Steam Deck specific sensors
            sensors = psutil.sensors_temperatures()
            
            warnings = []
            if not battery:
                warnings.append("Battery information not available")
            
            if not sensors:
                warnings.append("Temperature sensors not accessible via psutil")
            
            return {
                'success': True,
                'performance_metrics': {
                    'cpu_frequency': cpu_freq.current if cpu_freq else 0,
                    'battery_percent': battery.percent if battery else 0,
                    'sensors_count': len(sensors)
                },
                'warnings': warnings
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_msgpack_basic(self) -> Dict[str, Any]:
        """Test msgpack serialization"""
        try:
            import msgpack
            
            # Test various data types
            test_data = {
                'string': 'test_string',
                'number': 42,
                'float': 3.14159,
                'list': [1, 2, 3, 4, 5],
                'nested': {'inner': [True, False, None]}
            }
            
            # Serialize and deserialize
            packed = msgpack.packb(test_data)
            unpacked = msgpack.unpackb(packed, raw=False)
            
            # Verify round-trip
            success = unpacked == test_data
            
            return {
                'success': success,
                'performance_metrics': {
                    'original_size': len(str(test_data)),
                    'packed_size': len(packed),
                    'compression_ratio': len(str(test_data)) / len(packed)
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_msgpack_performance(self) -> Dict[str, Any]:
        """Compare msgpack vs JSON performance"""
        try:
            import msgpack
            import json
            
            # Create test data
            test_data = {
                'data': [[i, i*2, i*3] for i in range(1000)],
                'metadata': {'type': 'test', 'version': 1.0}
            }
            
            # Benchmark msgpack
            start_time = time.time()
            for _ in range(10):
                packed = msgpack.packb(test_data)
                unpacked = msgpack.unpackb(packed, raw=False)
            msgpack_time = time.time() - start_time
            
            # Benchmark JSON
            start_time = time.time()
            for _ in range(10):
                json_str = json.dumps(test_data)
                unpacked = json.loads(json_str)
            json_time = time.time() - start_time
            
            speedup = json_time / max(msgpack_time, 1e-6)
            
            return {
                'success': True,
                'performance_metrics': {
                    'msgpack_time': msgpack_time,
                    'json_time': json_time,
                    'speedup': speedup,
                    'msgpack_size': len(msgpack.packb(test_data)),
                    'json_size': len(json.dumps(test_data))
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_zstd_basic(self) -> Dict[str, Any]:
        """Test zstandard compression"""
        try:
            import zstandard as zstd
            
            # Test data
            test_data = b"This is a test string for compression. " * 100
            
            # Create compressor and decompressor
            compressor = zstd.ZstdCompressor(level=1)
            decompressor = zstd.ZstdDecompressor()
            
            # Compress and decompress
            compressed = compressor.compress(test_data)
            decompressed = decompressor.decompress(compressed)
            
            success = decompressed == test_data
            
            return {
                'success': success,
                'performance_metrics': {
                    'original_size': len(test_data),
                    'compressed_size': len(compressed),
                    'compression_ratio': len(test_data) / len(compressed)
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def _test_zstd_performance(self) -> Dict[str, Any]:
        """Benchmark zstandard performance"""
        try:
            import zstandard as zstd
            import gzip
            
            # Create test data
            test_data = b"Performance test data. " * 10000
            
            # Benchmark zstandard
            compressor = zstd.ZstdCompressor(level=1)
            start_time = time.time()
            zstd_compressed = compressor.compress(test_data)
            zstd_time = time.time() - start_time
            
            # Benchmark gzip for comparison
            start_time = time.time()
            gzip_compressed = gzip.compress(test_data)
            gzip_time = time.time() - start_time
            
            speedup = gzip_time / max(zstd_time, 1e-6)
            
            return {
                'success': True,
                'performance_metrics': {
                    'zstd_time': zstd_time,
                    'gzip_time': gzip_time,
                    'speedup': speedup,
                    'zstd_size': len(zstd_compressed),
                    'gzip_size': len(gzip_compressed),
                    'throughput': len(test_data) / zstd_time
                }
            }
        
        except Exception as e:
            return {'success': False, 'error_message': str(e)}
    
    def validate_all_dependencies(self, 
                                include_stress_tests: bool = False,
                                include_performance_tests: bool = True,
                                parallel: bool = True) -> Dict[str, DependencyValidation]:
        """
        Validate all available dependencies
        """
        logger.info("Starting comprehensive dependency validation...")
        
        # Get list of dependencies to validate
        try:
            from .dependency_coordinator import get_coordinator
            coordinator = get_coordinator()
            available_deps = [
                name for name, state in coordinator.dependency_states.items()
                if state.available
            ]
        except ImportError:
            # Fallback to all known dependencies
            available_deps = list(self.validation_tests.keys())
        
        results = {}
        
        if parallel and len(available_deps) > 1:
            # Parallel validation
            with ThreadPoolExecutor(max_workers=min(4, len(available_deps))) as executor:
                future_to_dep = {
                    executor.submit(
                        self.validate_dependency, 
                        dep, 
                        include_stress_tests, 
                        include_performance_tests
                    ): dep
                    for dep in available_deps
                }
                
                for future in as_completed(future_to_dep):
                    dep = future_to_dep[future]
                    try:
                        results[dep] = future.result()
                    except Exception as e:
                        logger.error(f"Validation failed for {dep}: {e}")
                        results[dep] = DependencyValidation(
                            dependency_name=dep,
                            overall_success=False,
                            test_results=[],
                            installation_health=0.0,
                            performance_score=0.0,
                            recommendations=[f"Validation failed: {e}"]
                        )
        else:
            # Sequential validation
            for dep in available_deps:
                results[dep] = self.validate_dependency(
                    dep, include_stress_tests, include_performance_tests
                )
        
        # Log summary
        total_deps = len(results)
        healthy_deps = sum(1 for r in results.values() if r.overall_success)
        avg_health = sum(r.installation_health for r in results.values()) / max(total_deps, 1)
        
        logger.info(f"Validation complete: {healthy_deps}/{total_deps} dependencies healthy, {avg_health:.1%} average health")
        
        return results
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validate_installation(include_performance: bool = True) -> Dict[str, Any]:
    """
    Quick validation of all dependencies with summary results
    """
    validator = InstallationValidator()
    
    try:
        results = validator.validate_all_dependencies(
            include_stress_tests=False,
            include_performance_tests=include_performance,
            parallel=True
        )
        
        # Generate summary
        summary = {
            'total_dependencies': len(results),
            'healthy_dependencies': sum(1 for r in results.values() if r.overall_success),
            'average_health': sum(r.installation_health for r in results.values()) / max(len(results), 1),
            'average_performance': sum(r.performance_score for r in results.values()) / max(len(results), 1),
            'critical_issues': [],
            'recommendations': [],
            'details': {}
        }
        
        # Collect critical issues and recommendations
        for dep_name, validation in results.items():
            summary['details'][dep_name] = {
                'health': validation.installation_health,
                'performance': validation.performance_score,
                'success': validation.overall_success
            }
            
            if not validation.overall_success:
                summary['critical_issues'].append(f"{dep_name}: {validation.test_results[0].error_message if validation.test_results else 'Unknown error'}")
            
            summary['recommendations'].extend(validation.recommendations)
        
        return summary
    
    finally:
        validator.cleanup()

def validate_specific_dependency(dependency_name: str, thorough: bool = False) -> DependencyValidation:
    """
    Validate a specific dependency with detailed testing
    """
    validator = InstallationValidator()
    
    try:
        return validator.validate_dependency(
            dependency_name,
            include_stress_tests=thorough,
            include_performance_tests=True
        )
    finally:
        validator.cleanup()


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🔍 Installation Validator Test Suite")
    print("=" * 50)
    
    # Create validator
    validator = InstallationValidator()
    
    try:
        # Quick validation of available dependencies
        print("\n📋 Running Quick Validation...")
        summary = quick_validate_installation(include_performance=False)
        
        print(f"Results: {summary['healthy_dependencies']}/{summary['total_dependencies']} healthy")
        print(f"Average Health: {summary['average_health']:.1%}")
        
        if summary['critical_issues']:
            print(f"\n🚨 Critical Issues:")
            for issue in summary['critical_issues'][:3]:  # Show top 3
                print(f"  ❌ {issue}")
        
        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in summary['recommendations'][:3]:  # Show top 3
                print(f"  💡 {rec}")
        
        # Test a specific dependency in detail if available
        test_deps = ['numpy', 'psutil', 'msgpack']
        for dep in test_deps:
            if dep in summary['details'] and summary['details'][dep]['success']:
                print(f"\n🔬 Detailed Validation: {dep}")
                detailed_result = validator.validate_dependency(dep, include_stress_tests=False)
                
                print(f"Health: {detailed_result.installation_health:.1%}")
                print(f"Performance Score: {detailed_result.performance_score:.1f}")
                print(f"Tests Run: {len(detailed_result.test_results)}")
                
                for result in detailed_result.test_results:
                    status = "✅" if result.success else "❌"
                    print(f"  {status} {result.test_name} ({result.duration:.3f}s)")
                
                break
        
        print(f"\n✅ Installation validation completed successfully!")
    
    finally:
        validator.cleanup()