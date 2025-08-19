#!/usr/bin/env python3
"""
Enhanced Python integration layer for shader prediction system.

This module provides intelligent hybrid execution between Rust acceleration
and optimized Python implementations with advanced fallback strategies.
"""

import sys
import time
import logging
import importlib.util
import threading
import psutil
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Enhanced imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Performance monitoring
class PerformanceLevel(Enum):
    RUST_OPTIMAL = "rust_optimal"           # <0.5ms predictions
    RUST_GOOD = "rust_good"                 # <1.0ms predictions  
    PYTHON_ENHANCED = "python_enhanced"     # <2.0ms predictions
    PYTHON_STANDARD = "python_standard"     # <5.0ms predictions
    HEURISTIC_FALLBACK = "heuristic"       # Basic fallback

class BackendType(Enum):
    RUST_PRIMARY = "rust_primary"
    PYTHON_ENHANCED = "python_enhanced"
    PYTHON_OPTIMIZED = "python_optimized"
    HEURISTIC = "heuristic"

@dataclass
class PredictionMetrics:
    """Performance metrics for prediction system"""
    average_time_ms: float = 0.0
    success_rate: float = 1.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    backend: str = "unknown"
    prediction_count: int = 0
    performance_level: PerformanceLevel = PerformanceLevel.HEURISTIC_FALLBACK
    last_update: float = field(default_factory=time.time)

class PerformanceMonitor:
    """Monitor performance of different prediction backends"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.prediction_times = deque(maxlen=window_size)
        self.success_count = 0
        self.total_count = 0
        self.lock = threading.Lock()
        
    def record_prediction(self, duration_ms: float, success: bool = True):
        """Record a prediction result"""
        with self.lock:
            self.prediction_times.append(duration_ms)
            self.total_count += 1
            if success:
                self.success_count += 1
    
    def get_metrics(self) -> PredictionMetrics:
        """Get current performance metrics"""
        with self.lock:
            if not self.prediction_times:
                return PredictionMetrics()
            
            avg_time = sum(self.prediction_times) / len(self.prediction_times)
            success_rate = self.success_count / max(1, self.total_count)
            
            # Determine performance level
            if avg_time < 0.5:
                level = PerformanceLevel.RUST_OPTIMAL
            elif avg_time < 1.0:
                level = PerformanceLevel.RUST_GOOD
            elif avg_time < 2.0:
                level = PerformanceLevel.PYTHON_ENHANCED
            elif avg_time < 5.0:
                level = PerformanceLevel.PYTHON_STANDARD
            else:
                level = PerformanceLevel.HEURISTIC_FALLBACK
            
            return PredictionMetrics(
                average_time_ms=avg_time,
                success_rate=success_rate,
                prediction_count=len(self.prediction_times),
                performance_level=level
            )

# Enhanced Rust module detection with performance validation
RUST_AVAILABLE = False
RUST_PERFORMANCE_VALIDATED = False

try:
    import shader_predict_rust
    RUST_AVAILABLE = True
    logging.info("Rust module detected - will validate performance on first use")
except ImportError:
    logging.info("Rust module not available - using optimized Python implementation")


class EnhancedHybridPredictor:
    """
    Advanced hybrid ML predictor with intelligent backend selection,
    performance monitoring, and graceful degradation.
    """
    
    def __init__(self, model_path: Optional[Path] = None, 
                 force_python: bool = False,
                 performance_threshold_ms: float = 2.0):
        """
        Initialize enhanced hybrid predictor
        
        Args:
            model_path: Path to model files
            force_python: Force Python implementation
            performance_threshold_ms: Maximum acceptable prediction time
        """
        self.model_path = model_path
        self.force_python = force_python
        self.performance_threshold_ms = performance_threshold_ms
        
        # Backend management
        self.current_backend = BackendType.HEURISTIC
        self.rust_predictor = None
        self.python_enhanced_predictor = None
        self.python_optimized_predictor = None
        self.heuristic_predictor = None
        
        # Performance monitoring
        self.performance_monitors = {
            BackendType.RUST_PRIMARY: PerformanceMonitor(),
            BackendType.PYTHON_ENHANCED: PerformanceMonitor(),
            BackendType.PYTHON_OPTIMIZED: PerformanceMonitor(),
            BackendType.HEURISTIC: PerformanceMonitor()
        }
        
        # Auto-switching configuration
        self.auto_switch_enabled = True
        self.switch_threshold_failures = 5
        self.backend_failure_counts = {backend: 0 for backend in BackendType}
        self.last_backend_switch = time.time()
        self.min_switch_interval = 30.0  # Minimum seconds between switches
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize backends
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize available prediction backends in order of preference"""
        with self.lock:
            # Try to initialize Rust backend first (highest performance)
            if RUST_AVAILABLE and not self.force_python:
                self._try_initialize_rust()
            
            # Always initialize Python backends as fallbacks
            self._initialize_python_enhanced()
            self._initialize_python_optimized()
            self._initialize_heuristic()
            
            # Set initial backend
            self._select_optimal_backend()
    
    def _try_initialize_rust(self):
        """Attempt to initialize Rust backend with validation"""
        try:
            self.rust_predictor = shader_predict_rust.RustMLPredictor(
                str(self.model_path) if self.model_path else None
            )
            
            # Quick performance validation
            test_features = {
                'instruction_count': 500,
                'register_usage': 32,
                'texture_samples': 4,
                'memory_operations': 10,
                'control_flow_complexity': 5,
                'wave_size': 64,
                'uses_derivatives': False,
                'shader_type_hash': 2.0,
                'optimization_level': 1,
                'cache_priority': 0.5
            }
            
            start_time = time.perf_counter()
            self.rust_predictor.predict_compilation_time(test_features)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if duration_ms < 10.0:  # Reasonable performance threshold
                self.current_backend = BackendType.RUST_PRIMARY
                self.logger.info(f"Rust backend initialized successfully ({duration_ms:.3f}ms validation)")
                return True
            else:
                self.logger.warning(f"Rust backend too slow ({duration_ms:.3f}ms), using Python fallback")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize Rust backend: {e}")
        
        self.rust_predictor = None
        return False
    
    def _initialize_python_enhanced(self):
        """Initialize enhanced Python predictor"""
        try:
            from .core.enhanced_ml_predictor import get_enhanced_predictor
            self.python_enhanced_predictor = get_enhanced_predictor()
            self.logger.info("Enhanced Python predictor initialized")
        except ImportError as e:
            self.logger.warning(f"Enhanced Python predictor not available: {e}")
            self.python_enhanced_predictor = None
    
    def _initialize_python_optimized(self):
        """Initialize optimized Python predictor"""
        try:
            from .core.optimized_ml_predictor import get_optimized_predictor
            self.python_optimized_predictor = get_optimized_predictor()
            self.logger.info("Optimized Python predictor initialized")
        except ImportError as e:
            self.logger.warning(f"Optimized Python predictor not available: {e}")
            self.python_optimized_predictor = None
    
    def _initialize_heuristic(self):
        """Initialize heuristic fallback predictor"""
        try:
            from .core.unified_ml_predictor import HeuristicPredictor
            self.heuristic_predictor = HeuristicPredictor()
            self.logger.info("Heuristic predictor initialized")
        except ImportError as e:
            self.logger.error(f"Cannot initialize heuristic predictor: {e}")
            raise RuntimeError("No prediction backend available")
    
    def _select_optimal_backend(self):
        """Select the optimal backend based on availability and performance"""
        # Priority order: Rust -> Enhanced Python -> Optimized Python -> Heuristic
        if self.rust_predictor and not self.force_python:
            self.current_backend = BackendType.RUST_PRIMARY
        elif self.python_enhanced_predictor:
            self.current_backend = BackendType.PYTHON_ENHANCED
        elif self.python_optimized_predictor:
            self.current_backend = BackendType.PYTHON_OPTIMIZED
        else:
            self.current_backend = BackendType.HEURISTIC
        
        self.logger.info(f"Selected backend: {self.current_backend.value}")
    
    def _should_switch_backend(self, current_backend: BackendType) -> Optional[BackendType]:
        """Determine if we should switch to a different backend"""
        if not self.auto_switch_enabled:
            return None
        
        # Don't switch too frequently
        if time.time() - self.last_backend_switch < self.min_switch_interval:
            return None
        
        # Check if current backend is failing too often
        failure_count = self.backend_failure_counts[current_backend]
        if failure_count >= self.switch_threshold_failures:
            # Find next best backend
            backend_order = [
                BackendType.RUST_PRIMARY,
                BackendType.PYTHON_ENHANCED, 
                BackendType.PYTHON_OPTIMIZED,
                BackendType.HEURISTIC
            ]
            
            current_index = backend_order.index(current_backend)
            for i in range(current_index + 1, len(backend_order)):
                next_backend = backend_order[i]
                if self._is_backend_available(next_backend):
                    return next_backend
        
        return None
    
    def _is_backend_available(self, backend: BackendType) -> bool:
        """Check if a backend is available"""
        if backend == BackendType.RUST_PRIMARY:
            return self.rust_predictor is not None
        elif backend == BackendType.PYTHON_ENHANCED:
            return self.python_enhanced_predictor is not None
        elif backend == BackendType.PYTHON_OPTIMIZED:
            return self.python_optimized_predictor is not None
        elif backend == BackendType.HEURISTIC:
            return self.heuristic_predictor is not None
        return False
    
    def _switch_backend(self, new_backend: BackendType):
        """Switch to a different backend"""
        with self.lock:
            if new_backend != self.current_backend and self._is_backend_available(new_backend):
                old_backend = self.current_backend
                self.current_backend = new_backend
                self.last_backend_switch = time.time()
                
                # Reset failure counts
                self.backend_failure_counts[new_backend] = 0
                
                self.logger.info(f"Switched backend: {old_backend.value} -> {new_backend.value}")
    
    def predict_compilation_time(self, features: Union[Dict[str, Any], Any], 
                                game_context: Optional[str] = None) -> float:
        """
        Predict shader compilation time with intelligent backend selection
        
        Args:
            features: Shader features (dict or UnifiedShaderFeatures object)
            game_context: Optional game context for optimization
            
        Returns:
            Predicted compilation time in milliseconds
        """
        start_time = time.perf_counter()
        
        with self.lock:
            # Check if we should switch backends
            new_backend = self._should_switch_backend(self.current_backend)
            if new_backend:
                self._switch_backend(new_backend)
            
            # Make prediction with current backend
            prediction = self._predict_with_backend(self.current_backend, features, game_context)
            
            # Record performance
            duration_ms = (time.perf_counter() - start_time) * 1000
            success = prediction is not None and prediction > 0
            
            if success:
                self.performance_monitors[self.current_backend].record_prediction(duration_ms, True)
                self.backend_failure_counts[self.current_backend] = 0
            else:
                self.performance_monitors[self.current_backend].record_prediction(duration_ms, False)
                self.backend_failure_counts[self.current_backend] += 1
                self.logger.warning(f"Backend {self.current_backend.value} failed prediction")
            
            return prediction if prediction is not None else 10.0  # Safe fallback
    
    def _predict_with_backend(self, backend: BackendType, features: Any, 
                             game_context: Optional[str] = None) -> Optional[float]:
        """Make prediction with specified backend"""
        try:
            if backend == BackendType.RUST_PRIMARY and self.rust_predictor:
                return self._predict_rust(features)
            elif backend == BackendType.PYTHON_ENHANCED and self.python_enhanced_predictor:
                return self._predict_python_enhanced(features, game_context)
            elif backend == BackendType.PYTHON_OPTIMIZED and self.python_optimized_predictor:
                return self._predict_python_optimized(features)
            elif backend == BackendType.HEURISTIC and self.heuristic_predictor:
                return self._predict_heuristic(features)
            else:
                raise ValueError(f"Backend {backend.value} not available")
                
        except Exception as e:
            self.logger.error(f"Prediction failed with {backend.value}: {e}")
            return None
    
    def _predict_rust(self, features: Any) -> float:
        """Predict using Rust backend"""
        if isinstance(features, dict):
            return self.rust_predictor.predict_compilation_time(features)
        else:
            # Convert UnifiedShaderFeatures to dict
            feature_dict = {
                'instruction_count': features.instruction_count,
                'register_usage': features.register_usage,
                'texture_samples': features.texture_samples,
                'memory_operations': features.memory_operations,
                'control_flow_complexity': features.control_flow_complexity,
                'wave_size': features.wave_size,
                'uses_derivatives': features.uses_derivatives,
                'uses_tessellation': features.uses_tessellation,
                'uses_geometry_shader': features.uses_geometry_shader,
                'shader_type_hash': features.shader_type.value.__hash__() % 10,
                'optimization_level': features.optimization_level,
                'cache_priority': features.cache_priority
            }
            return self.rust_predictor.predict_compilation_time(feature_dict)
    
    def _predict_python_enhanced(self, features: Any, game_context: Optional[str] = None) -> float:
        """Predict using enhanced Python backend"""
        if hasattr(self.python_enhanced_predictor, 'predict_compilation_time'):
            return self.python_enhanced_predictor.predict_compilation_time(features, game_context=game_context)
        else:
            return self.python_enhanced_predictor.predict_compilation_time(features)
    
    def _predict_python_optimized(self, features: Any) -> float:
        """Predict using optimized Python backend"""
        return self.python_optimized_predictor.predict_compilation_time(features)
    
    def _predict_heuristic(self, features: Any) -> float:
        """Predict using heuristic fallback"""
        return self.heuristic_predictor.predict_compilation_time(features)
    
    def predict_batch(self, features_list: List[Any], 
                     game_context: Optional[str] = None) -> List[float]:
        """
        Predict compilation times for multiple shaders
        
        Args:
            features_list: List of feature objects
            game_context: Optional game context
            
        Returns:
            List of predicted compilation times
        """
        if self.current_backend == BackendType.RUST_PRIMARY and self.rust_predictor:
            try:
                # Convert features to dicts if needed
                dict_features = []
                for features in features_list:
                    if isinstance(features, dict):
                        dict_features.append(features)
                    else:
                        dict_features.append({
                            'instruction_count': features.instruction_count,
                            'register_usage': features.register_usage,
                            'texture_samples': features.texture_samples,
                            'memory_operations': features.memory_operations,
                            'control_flow_complexity': features.control_flow_complexity,
                            'wave_size': features.wave_size,
                            'uses_derivatives': features.uses_derivatives,
                            'uses_tessellation': features.uses_tessellation,
                            'uses_geometry_shader': features.uses_geometry_shader,
                            'shader_type_hash': features.shader_type.value.__hash__() % 10,
                            'optimization_level': features.optimization_level,
                            'cache_priority': features.cache_priority
                        })
                
                predictions = self.rust_predictor.predict_batch(dict_features)
                return predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
                
            except Exception as e:
                self.logger.warning(f"Rust batch prediction failed: {e}, falling back to individual predictions")
        
        # Fallback to individual predictions
        return [self.predict_compilation_time(features, game_context) for features in features_list]
    
    def extract_features_from_spirv(self, spirv_data: bytes) -> Dict[str, Any]:
        """Extract features from SPIR-V bytecode"""
        if self.current_backend == BackendType.RUST_PRIMARY and self.rust_predictor:
            try:
                rust_features = self.rust_predictor.extract_features_from_spirv(spirv_data)
                return rust_features.to_dict() if hasattr(rust_features, 'to_dict') else dict(rust_features)
            except Exception as e:
                self.logger.warning(f"Rust SPIR-V extraction failed: {e}")
        
        # Python fallback - basic analysis
        return self._extract_features_python_fallback(spirv_data)
    
    def _extract_features_python_fallback(self, spirv_data: bytes) -> Dict[str, Any]:
        """Basic Python fallback for SPIR-V feature extraction"""
        instruction_count = len(spirv_data) // 4
        
        return {
            "instruction_count": float(instruction_count),
            "register_usage": min(64.0, instruction_count / 10),
            "texture_samples": min(8.0, instruction_count / 100),
            "memory_operations": min(20.0, instruction_count / 50),
            "control_flow_complexity": min(10.0, instruction_count / 80),
            "wave_size": 64.0,
            "uses_derivatives": instruction_count > 200,
            "uses_tessellation": False,
            "uses_geometry_shader": False,
            "shader_type_hash": 2.0,
            "optimization_level": 1.0,
            "cache_priority": 0.5,
            "van_gogh_optimized": True,
            "rdna2_features": True,
            "thermal_state": 0.5,
            "power_mode": 1.0,
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for all backends"""
        metrics = {}
        
        for backend, monitor in self.performance_monitors.items():
            metrics[backend.value] = monitor.get_metrics()
        
        # Add system information
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
        except Exception:
            memory_mb = 0.0
        
        return {
            'current_backend': self.current_backend.value,
            'backend_metrics': metrics,
            'system_memory_mb': memory_mb,
            'rust_available': RUST_AVAILABLE,
            'auto_switch_enabled': self.auto_switch_enabled,
            'backend_failure_counts': dict(self.backend_failure_counts),
            'performance_threshold_ms': self.performance_threshold_ms
        }
    
    def force_backend(self, backend: BackendType):
        """Force use of specific backend (disables auto-switching)"""
        with self.lock:
            if self._is_backend_available(backend):
                self.current_backend = backend
                self.auto_switch_enabled = False
                self.logger.info(f"Forced backend to: {backend.value}")
            else:
                raise ValueError(f"Backend {backend.value} is not available")
    
    def enable_auto_switching(self):
        """Re-enable automatic backend switching"""
        with self.lock:
            self.auto_switch_enabled = True
            self.logger.info("Auto-switching enabled")
    
    def cleanup(self):
        """Cleanup all predictors and resources"""
        with self.lock:
            if self.python_enhanced_predictor and hasattr(self.python_enhanced_predictor, 'cleanup'):
                self.python_enhanced_predictor.cleanup()
            
            if self.python_optimized_predictor and hasattr(self.python_optimized_predictor, 'cleanup'):
                self.python_optimized_predictor.cleanup()
            
            self.logger.info("Enhanced hybrid predictor cleanup completed")


# Convenience functions for backward compatibility and easy usage
_global_predictor = None
_predictor_lock = threading.Lock()


def get_hybrid_predictor(force_python: bool = False) -> EnhancedHybridPredictor:
    """Get or create global hybrid predictor instance"""
    global _global_predictor
    if _global_predictor is None:
        with _predictor_lock:
            if _global_predictor is None:
                _global_predictor = EnhancedHybridPredictor(force_python=force_python)
    return _global_predictor


def predict_shader_compilation_time(features: Union[Dict[str, Any], Any], 
                                   game_context: Optional[str] = None) -> float:
    """Convenience function for shader compilation time prediction"""
    predictor = get_hybrid_predictor()
    return predictor.predict_compilation_time(features, game_context)


def get_system_info() -> Dict[str, Any]:
    """Get system information for optimization decisions"""
    if RUST_AVAILABLE:
        try:
            return shader_predict_rust.get_system_info()
        except Exception as e:
            logging.warning(f"Failed to get Rust system info: {e}")
    
    # Python fallback
    import platform
    import os
    
    info = {
        "is_steam_deck": os.path.exists("/home/deck"),
        "cpu_count": os.cpu_count() or 4,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "optimization_target": "steam_deck" if os.path.exists("/home/deck") else "desktop",
        "backend": "python_fallback"
    }
    
    return info


def is_steam_deck() -> bool:
    """Check if running on Steam Deck"""
    if RUST_AVAILABLE:
        try:
            return shader_predict_rust.is_steam_deck()
        except Exception:
            pass
    
    # Python fallback
    import os
    return os.path.exists("/home/deck")


# Legacy compatibility
HybridMLPredictor = EnhancedHybridPredictor
HybridVulkanCache = None  # Deprecated - use enhanced predictor caching


if __name__ == "__main__":
    # Enhanced testing and demonstration
    logging.basicConfig(level=logging.INFO)
    
    print("\n🔄 Enhanced Hybrid Predictor Test Suite")
    print("=" * 60)
    
    # Test system detection
    print(f"System Info: {get_system_info()}")
    print(f"Is Steam Deck: {is_steam_deck()}")
    print(f"Rust Available: {RUST_AVAILABLE}")
    
    # Test hybrid predictor
    try:
        predictor = get_hybrid_predictor()
        
        print(f"\n✓ Initialized with backend: {predictor.current_backend.value}")
        
        # Test feature extraction and prediction
        from .core.unified_ml_predictor import UnifiedShaderFeatures, ShaderType
        
        test_features = UnifiedShaderFeatures(
            shader_hash="hybrid_test_123",
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
        
        # Performance test
        print(f"\n⚡ Performance Test:")
        predictions = []
        start_time = time.perf_counter()
        
        for i in range(10):
            prediction = predictor.predict_compilation_time(test_features)
            predictions.append(prediction)
        
        avg_time = (time.perf_counter() - start_time) / 10 * 1000
        print(f"  - Average prediction time: {avg_time:.3f}ms")
        print(f"  - Predictions: {predictions[:3]}... (showing first 3)")
        
        # Test batch prediction
        feature_list = [test_features] * 5
        batch_predictions = predictor.predict_batch(feature_list)
        print(f"  - Batch predictions: {len(batch_predictions)} results")
        
        # Test backend switching
        print(f"\n🔄 Backend Management Test:")
        initial_backend = predictor.current_backend
        print(f"  - Initial backend: {initial_backend.value}")
        
        # Get comprehensive metrics
        metrics = predictor.get_comprehensive_metrics()
        print(f"\n📊 Performance Metrics:")
        for backend, metric in metrics['backend_metrics'].items():
            print(f"  - {backend}: {metric.average_time_ms:.3f}ms avg, "
                  f"{metric.success_rate:.1%} success, {metric.prediction_count} predictions")
        
        print(f"  - System Memory: {metrics['system_memory_mb']:.1f}MB")
        print(f"  - Auto-switching: {metrics['auto_switch_enabled']}")
        
        # Cleanup
        predictor.cleanup()
        print(f"\n✅ Enhanced hybrid predictor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()