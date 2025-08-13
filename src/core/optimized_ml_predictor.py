#!/usr/bin/env python3
"""
Optimized ML Shader Prediction System for Steam Deck
High-performance, memory-efficient implementation with advanced caching and async support
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from functools import lru_cache, wraps
import weakref

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core ML imports with optimized loading
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Optimized ML imports - prefer lightweight alternatives
try:
    # Try LightGBM first (fastest, most memory efficient)
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    ML_BACKEND = "lightgbm"
except ImportError:
    HAS_LIGHTGBM = False
    try:
        # Fallback to scikit-learn with optimized imports
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.exceptions import NotFittedError
        import joblib
        HAS_SKLEARN = True
        ML_BACKEND = "sklearn"
    except ImportError:
        HAS_SKLEARN = False
        ML_BACKEND = "heuristic"
        print("WARNING: No ML backend available. Using heuristic predictor only.")

import psutil

# Import base classes from unified implementation
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


class MemoryPool:
    """Object pool for efficient memory management"""
    
    def __init__(self, factory: Callable, max_size: int = 100):
        self._factory = factory
        self._pool = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def acquire(self):
        """Get an object from pool or create new one"""
        with self._lock:
            if self._pool:
                return self._pool.popleft()
        return self._factory()
    
    def release(self, obj):
        """Return object to pool"""
        with self._lock:
            if len(self._pool) < self._pool.maxlen:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)


class FeatureVectorCache:
    """Optimized feature vector caching with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[list]:
        """Get cached feature vector"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key].copy() if HAS_NUMPY and hasattr(self._cache[key], 'copy') else list(self._cache[key])
            self._misses += 1
            return None
    
    def put(self, key: str, value):
        """Cache feature vector"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    # Evict least recently used
                    self._cache.popitem(last=False)
                self._cache[key] = value.copy() if HAS_NUMPY and hasattr(value, 'copy') else list(value) if hasattr(value, '__iter__') else value
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class OptimizedMLPredictor:
    """High-performance ML prediction system with advanced optimizations"""
    
    def __init__(self, model_path: Optional[Path] = None, 
                 enable_async: bool = True,
                 max_memory_mb: int = 50):
        """
        Initialize optimized ML predictor
        
        Args:
            model_path: Path to store/load models
            enable_async: Enable async operations
            max_memory_mb: Maximum memory usage in MB
        """
        self.model_path = model_path or Path.home() / '.cache' / 'shader-predict-compile' / 'models'
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Memory management
        self.max_memory_mb = max_memory_mb
        self._current_memory_mb = 0
        self._memory_pressure = False
        
        # Models (lazy loaded)
        self._compilation_time_model = None
        self._success_model = None
        self._scaler = None
        self._models_loaded = False
        
        # Fallback predictor
        self.heuristic_predictor = HeuristicPredictor()
        
        # Thermal management
        self.thermal_scheduler = ThermalAwareScheduler()
        
        # Optimized caching
        self.feature_cache = FeatureVectorCache(max_size=500)
        self.prediction_cache = OrderedDict()
        self.max_cache_size = 1000
        
        # Training data with smaller buffer
        self.training_buffer_size = 2000  # Reduced from 10000
        self.training_data = deque(maxlen=self.training_buffer_size)
        
        # Thread pools for async operations
        self.enable_async = enable_async
        if enable_async:
            self.thread_pool = ThreadPoolExecutor(max_workers=2)
            self.io_pool = ThreadPoolExecutor(max_workers=1)
        
        # Object pools for memory efficiency
        def create_feature_vector():
            if HAS_NUMPY:
                return np.zeros(12, dtype=np.float32)
            else:
                return [0.0] * 12
        
        self._feature_vector_pool = MemoryPool(
            create_feature_vector,
            max_size=20
        )
        
        # Performance metrics
        self._prediction_times = deque(maxlen=100)
        self._last_memory_check = 0
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Start memory monitoring
        self._start_memory_monitor()
    
    def _start_memory_monitor(self):
        """Monitor memory usage and trigger cleanup when needed"""
        def monitor():
            while True:
                time.sleep(30)  # Check every 30 seconds
                self._check_memory_pressure()
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _check_memory_pressure(self):
        """Check system memory pressure and cleanup if needed"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self._current_memory_mb = memory_mb
            
            if memory_mb > self.max_memory_mb:
                self._memory_pressure = True
                self._cleanup_memory()
            else:
                self._memory_pressure = False
        except Exception as e:
            self.logger.debug(f"Memory check failed: {e}")
    
    def _cleanup_memory(self):
        """Clean up memory when under pressure"""
        # Clear old predictions from cache
        if len(self.prediction_cache) > self.max_cache_size // 2:
            # Remove oldest half
            for _ in range(len(self.prediction_cache) // 2):
                self.prediction_cache.popitem(last=False)
        
        # Reduce training buffer if needed
        if len(self.training_data) > self.training_buffer_size // 2:
            # Keep only recent half
            new_data = list(self.training_data)[-self.training_buffer_size // 2:]
            self.training_data.clear()
            self.training_data.extend(new_data)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.logger.info(f"Memory cleanup completed. Current: {self._current_memory_mb:.1f}MB")
    
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
    
    def _create_lightweight_model(self, model_type: str = "regressor"):
        """Create lightweight model based on available backend"""
        if HAS_LIGHTGBM:
            if model_type == "regressor":
                return lgb.LGBMRegressor(
                    n_estimators=30,  # Reduced for speed
                    max_depth=8,      # Shallow for memory
                    num_leaves=31,
                    learning_rate=0.1,
                    n_jobs=2,
                    silent=True,
                    importance_type='gain',
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8
                )
            else:
                return lgb.LGBMClassifier(
                    n_estimators=20,
                    max_depth=6,
                    num_leaves=31,
                    learning_rate=0.1,
                    n_jobs=2,
                    silent=True,
                    importance_type='gain'
                )
        elif HAS_SKLEARN:
            if model_type == "regressor":
                return RandomForestRegressor(
                    n_estimators=20,  # Reduced from 50
                    max_depth=10,     # Reduced from 15
                    random_state=42,
                    n_jobs=2,
                    max_features='sqrt',
                    min_samples_split=5,
                    min_samples_leaf=2
                )
            else:
                return RandomForestClassifier(
                    n_estimators=15,  # Reduced from 30
                    max_depth=8,      # Reduced from 10
                    random_state=42,
                    n_jobs=2,
                    max_features='sqrt'
                )
        return None
    
    def _load_models(self):
        """Load models with optimization and warm-up"""
        if self._models_loaded:
            return
        
        try:
            # Try to load from disk first
            time_model_path = self.model_path / 'optimized_time_model.pkl'
            success_model_path = self.model_path / 'optimized_success_model.pkl'
            scaler_path = self.model_path / 'optimized_scaler.pkl'
            
            if time_model_path.exists():
                with open(time_model_path, 'rb') as f:
                    self._compilation_time_model = pickle.load(f)
                self.logger.info("Loaded optimized compilation time model")
            else:
                self._compilation_time_model = self._create_lightweight_model("regressor")
            
            if success_model_path.exists():
                with open(success_model_path, 'rb') as f:
                    self._success_model = pickle.load(f)
                self.logger.info("Loaded optimized success model")
            else:
                self._success_model = self._create_lightweight_model("classifier")
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self._scaler = pickle.load(f)
                self.logger.info("Loaded optimized scaler")
            elif HAS_SKLEARN or HAS_LIGHTGBM:
                self._scaler = StandardScaler()
            
            self._models_loaded = True
            
            # Warm up models with synthetic data if newly created
            if self._compilation_time_model and not time_model_path.exists():
                self._warm_up_models()
                
        except Exception as e:
            self.logger.warning(f"Could not load models: {e}")
            # Create fresh models
            self._compilation_time_model = self._create_lightweight_model("regressor")
            self._success_model = self._create_lightweight_model("classifier")
            if HAS_SKLEARN or HAS_LIGHTGBM:
                self._scaler = StandardScaler()
            self._models_loaded = True
    
    def _warm_up_models(self):
        """Warm up models with synthetic data for faster first prediction"""
        try:
            # Generate synthetic training data
            n_samples = 100
            if HAS_NUMPY:
                X = np.random.randn(n_samples, 12).astype(np.float32)
                y_time = np.random.exponential(10, n_samples)
                y_success = np.random.randint(0, 2, n_samples)
            else:
                import random
                X = [[random.gauss(0, 1) for _ in range(12)] for _ in range(n_samples)]
                y_time = [random.expovariate(1/10) for _ in range(n_samples)]
                y_success = [random.randint(0, 1) for _ in range(n_samples)]
            
            # Fit scaler
            if self._scaler:
                self._scaler.fit(X)
                X_scaled = self._scaler.transform(X)
            else:
                X_scaled = X
            
            # Train models
            if self._compilation_time_model:
                self._compilation_time_model.fit(X_scaled, y_time)
            
            if self._success_model:
                self._success_model.fit(X_scaled, y_success)
            
            self.logger.info("Models warmed up with synthetic data")
        except Exception as e:
            self.logger.debug(f"Model warm-up failed: {e}")
    
    @lru_cache(maxsize=128)
    def _hash_features(self, shader_hash: str, instruction_count: int, 
                      shader_type: str) -> str:
        """Create efficient feature hash for caching"""
        return f"{shader_hash[:8]}_{instruction_count}_{shader_type}"
    
    def predict_compilation_time(self, features: UnifiedShaderFeatures, 
                                use_cache: bool = True) -> float:
        """
        Optimized shader compilation time prediction
        
        Returns:
            Predicted compilation time in milliseconds
        """
        start_time = time.perf_counter()
        
        # Quick cache check
        if use_cache:
            cache_key = self._hash_features(
                features.shader_hash,
                features.instruction_count,
                features.shader_type.value
            )
            
            if cache_key in self.prediction_cache:
                # Move to end (LRU)
                self.prediction_cache.move_to_end(cache_key)
                prediction_time = (time.perf_counter() - start_time) * 1000
                self._prediction_times.append(prediction_time)
                return self.prediction_cache[cache_key]['time']
        
        # Get or compute feature vector
        feature_vector = self._get_feature_vector(features)
        
        # ML prediction with fallback
        prediction = self._predict_with_model(feature_vector, features)
        
        # Cache result
        if use_cache:
            # Enforce cache size limit
            if len(self.prediction_cache) >= self.max_cache_size:
                self.prediction_cache.popitem(last=False)
            
            self.prediction_cache[cache_key] = {
                'time': prediction,
                'timestamp': time.time()
            }
        
        # Track prediction time
        prediction_time = (time.perf_counter() - start_time) * 1000
        self._prediction_times.append(prediction_time)
        
        return prediction
    
    def _get_feature_vector(self, features: UnifiedShaderFeatures):
        """Get or compute feature vector with caching"""
        # Check feature cache
        cache_key = features.shader_hash
        cached_vector = self.feature_cache.get(cache_key)
        if cached_vector is not None:
            return cached_vector
        
        # Get vector from pool and fill it
        vector = self._feature_vector_pool.acquire()
        try:
            # Efficient feature extraction
            vector[0] = features.instruction_count
            vector[1] = features.register_usage
            vector[2] = features.texture_samples
            vector[3] = features.memory_operations
            vector[4] = features.control_flow_complexity
            vector[5] = features.wave_size
            vector[6] = float(features.uses_derivatives)
            vector[7] = float(features.uses_tessellation)
            vector[8] = float(features.uses_geometry_shader)
            vector[9] = features.shader_type.value.__hash__() % 10
            vector[10] = features.optimization_level
            vector[11] = features.cache_priority
            
            # Cache the vector
            self.feature_cache.put(cache_key, vector)
            return vector
        finally:
            # Return vector to pool
            self._feature_vector_pool.release(vector)
    
    def _predict_with_model(self, feature_vector,
                           features: UnifiedShaderFeatures) -> float:
        """Make prediction with ML model or fallback"""
        if self.compilation_time_model is None:
            return self.heuristic_predictor.predict_compilation_time(features)
        
        try:
            # Check if model is fitted
            if HAS_LIGHTGBM and hasattr(self.compilation_time_model, 'booster_'):
                fitted = self.compilation_time_model.booster_ is not None
            elif HAS_SKLEARN:
                fitted = hasattr(self.compilation_time_model, 'n_estimators')
            else:
                fitted = False
            
            if not fitted:
                return self.heuristic_predictor.predict_compilation_time(features)
            
            # Prepare features
            if HAS_NUMPY:
                X = feature_vector.reshape(1, -1)
                if self.scaler:
                    X = self.scaler.transform(X)
            else:
                X = [feature_vector] if isinstance(feature_vector, list) else [[feature_vector]]
                if self.scaler:
                    # Simple scaling without sklearn
                    X = [[(x - 0.5) * 2 for x in row] for row in X]
            
            # Make prediction
            prediction = self.compilation_time_model.predict(X)[0]
            return max(1.0, float(prediction))  # Ensure positive
            
        except Exception as e:
            self.logger.debug(f"ML prediction failed: {e}")
            return self.heuristic_predictor.predict_compilation_time(features)
    
    async def predict_compilation_time_async(self, 
                                            features: UnifiedShaderFeatures) -> float:
        """Async version of compilation time prediction"""
        if not self.enable_async:
            return self.predict_compilation_time(features)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.predict_compilation_time,
            features
        )
    
    def add_training_sample(self, features: UnifiedShaderFeatures,
                           actual_time: float, success: bool = True):
        """Add training sample with memory management"""
        # Check memory pressure
        if self._memory_pressure and len(self.training_data) >= self.training_buffer_size:
            # Remove oldest to make room
            self.training_data.popleft()
        
        self.training_data.append({
            'features': features,
            'time': actual_time,
            'success': success,
            'timestamp': time.time()
        })
        
        # Trigger retraining if buffer is full
        if len(self.training_data) >= self.training_buffer_size:
            if self.enable_async:
                # Retrain in background
                self.thread_pool.submit(self._retrain_models)
            else:
                self._retrain_models()
    
    def _retrain_models(self):
        """Retrain models with current training data"""
        if not self.training_data or not self._models_loaded:
            return
        
        try:
            # Prepare training data
            X = []
            y_time = []
            y_success = []
            
            for sample in self.training_data:
                features = sample['features']
                X.append(self._get_feature_vector(features))
                y_time.append(sample['time'])
                y_success.append(int(sample['success']))
            
            if HAS_NUMPY:
                X = np.array(X, dtype=np.float32)
                y_time = np.array(y_time, dtype=np.float32)
                y_success = np.array(y_success, dtype=np.int32)
            else:
                # X is already a list of lists
                y_time = list(y_time)
                y_success = [int(y) for y in y_success]
            
            # Update scaler
            if self.scaler:
                X = self.scaler.fit_transform(X)
            
            # Retrain models
            if self.compilation_time_model:
                self.compilation_time_model.fit(X, y_time)
            
            if self.success_model:
                self.success_model.fit(X, y_success)
            
            # Save updated models
            self._save_models()
            
            self.logger.info("Models retrained successfully")
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    def _save_models(self):
        """Save models efficiently"""
        try:
            if self._compilation_time_model:
                with open(self.model_path / 'optimized_time_model.pkl', 'wb') as f:
                    pickle.dump(self._compilation_time_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self._success_model:
                with open(self.model_path / 'optimized_success_model.pkl', 'wb') as f:
                    pickle.dump(self._success_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self._scaler:
                with open(self.model_path / 'optimized_scaler.pkl', 'wb') as f:
                    pickle.dump(self._scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info("Optimized models saved")
        except Exception as e:
            self.logger.error(f"Could not save models: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_prediction_time = (
            sum(self._prediction_times) / len(self._prediction_times)
            if self._prediction_times else 0
        )
        
        return {
            'ml_backend': ML_BACKEND,
            'avg_prediction_time_ms': avg_prediction_time,
            'cache_hit_rate': self.feature_cache.hit_rate,
            'prediction_cache_size': len(self.prediction_cache),
            'training_buffer_size': len(self.training_data),
            'memory_usage_mb': self._current_memory_mb,
            'memory_pressure': self._memory_pressure
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.enable_async:
            self.thread_pool.shutdown(wait=False)
            self.io_pool.shutdown(wait=False)
        
        # Clear caches
        self.prediction_cache.clear()
        self.training_data.clear()
        
        self.logger.info("Cleanup completed")


# Singleton instance for global access
_global_predictor = None


def get_optimized_predictor() -> OptimizedMLPredictor:
    """Get or create global optimized predictor instance"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = OptimizedMLPredictor()
    return _global_predictor


if __name__ == "__main__":
    # Test the optimized predictor
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    predictor = get_optimized_predictor()
    
    # Create test features
    test_features = UnifiedShaderFeatures(
        shader_hash="test_hash_123",
        shader_type=ShaderType.FRAGMENT,
        instruction_count=500,
        register_usage=32,
        texture_samples=4,
        memory_operations=10,
        control_flow_complexity=5,
        wave_size=64,
        uses_derivatives=True,
        uses_tessellation=False,
        uses_geometry_shader=False,
        optimization_level=3,
        cache_priority=0.8
    )
    
    # Test prediction
    print("\n=== Optimized ML Predictor Test ===")
    
    # Warm up
    for _ in range(3):
        predictor.predict_compilation_time(test_features)
    
    # Measure performance
    import timeit
    
    time_taken = timeit.timeit(
        lambda: predictor.predict_compilation_time(test_features),
        number=100
    ) / 100 * 1000
    
    print(f"Average prediction time: {time_taken:.2f}ms")
    
    # Get stats
    stats = predictor.get_performance_stats()
    print(f"\nPerformance Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    predictor.cleanup()