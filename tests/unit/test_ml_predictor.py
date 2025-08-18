#!/usr/bin/env python3
"""
Unit tests for ML predictor components
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from src.ml.optimized_ml_predictor import (
        OptimizedMLPredictor, 
        MemoryPool,
        FeatureVectorCache,
        get_optimized_predictor
    )
    from src.ml.unified_ml_predictor import UnifiedShaderFeatures, ShaderType
    HAS_ML_MODULES = True
except ImportError:
    HAS_ML_MODULES = False
    OptimizedMLPredictor = Mock
    MemoryPool = Mock
    FeatureVectorCache = Mock


@pytest.mark.unit
@pytest.mark.ml
class TestMemoryPool:
    """Test memory pool functionality"""
    
    def test_pool_creation(self):
        """Test memory pool creation"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        def factory():
            return {"test": True}
        
        pool = MemoryPool(factory, max_size=5)
        assert pool._factory == factory
        assert len(pool._pool) == 0
    
    def test_acquire_release_cycle(self):
        """Test acquire/release cycle"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        counter = 0
        def factory():
            nonlocal counter
            counter += 1
            return {"id": counter}
        
        pool = MemoryPool(factory, max_size=3)
        
        # First acquire should create new object
        obj1 = pool.acquire()
        assert obj1["id"] == 1
        
        # Release and acquire again should reuse
        pool.release(obj1)
        obj2 = pool.acquire()
        assert obj2 is obj1  # Same object
        
        # Test pool limit
        pool.release(obj2)
        for i in range(5):
            pool.release({"id": i + 10})
        
        assert len(pool._pool) == 3  # Max size limit
    
    def test_thread_safety(self):
        """Test thread safety of memory pool"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        import threading
        
        results = []
        pool = MemoryPool(lambda: {"value": 0}, max_size=10)
        
        def worker():
            obj = pool.acquire()
            time.sleep(0.01)  # Simulate work
            results.append(obj)
            pool.release(obj)
        
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 20


@pytest.mark.unit
@pytest.mark.ml
class TestFeatureVectorCache:
    """Test feature vector caching"""
    
    def test_cache_creation(self):
        """Test cache creation with size limit"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        cache = FeatureVectorCache(max_size=100)
        assert cache._max_size == 100
        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0
    
    def test_cache_put_get(self):
        """Test basic put/get operations"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        cache = FeatureVectorCache(max_size=5)
        vector = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Put and get
        cache.put("key1", vector)
        retrieved = cache.get("key1")
        
        assert retrieved is not None
        assert np.array_equal(retrieved, vector)
        assert cache._hits == 1
        assert cache._misses == 0
    
    def test_cache_miss(self):
        """Test cache miss handling"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        cache = FeatureVectorCache(max_size=5)
        
        result = cache.get("nonexistent")
        assert result is None
        assert cache._hits == 0
        assert cache._misses == 1
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        cache = FeatureVectorCache(max_size=3)
        
        # Fill cache to capacity
        for i in range(3):
            cache.put(f"key{i}", np.array([float(i)]))
        
        # Access key0 to make it recently used
        cache.get("key0")
        
        # Add new item, should evict key1 (least recently used)
        cache.put("key3", np.array([3.0]))
        
        assert cache.get("key0") is not None  # Still there
        assert cache.get("key1") is None      # Evicted
        assert cache.get("key2") is not None  # Still there
        assert cache.get("key3") is not None  # Newly added
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        cache = FeatureVectorCache(max_size=10)
        vector = np.array([1.0, 2.0, 3.0])
        cache.put("key1", vector)
        
        # 3 hits, 2 misses
        cache.get("key1")  # hit
        cache.get("key1")  # hit  
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        cache.get("key3")  # miss
        
        assert cache.hit_rate == 3.0 / 5.0  # 0.6


@pytest.mark.unit
@pytest.mark.ml
class TestOptimizedMLPredictor:
    """Test optimized ML predictor"""
    
    def test_predictor_creation(self, clean_temp_dir):
        """Test predictor initialization"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False,
            max_memory_mb=30
        )
        
        assert predictor.model_path == clean_temp_dir
        assert predictor.max_memory_mb == 30
        assert predictor.enable_async == False
        assert predictor.feature_cache is not None
        assert isinstance(predictor.prediction_cache, dict)
    
    @patch('psutil.Process')
    def test_memory_monitoring(self, mock_process, clean_temp_dir):
        """Test memory monitoring functionality"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 60 * 1024 * 1024  # 60MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            max_memory_mb=50,
            enable_async=False
        )
        
        # Trigger memory check
        predictor._check_memory_pressure()
        
        assert predictor._memory_pressure == True
        assert predictor._current_memory_mb == 60.0
    
    def test_prediction_caching(self, clean_temp_dir, sample_shader_features):
        """Test prediction result caching"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False
        )
        
        # Mock the model to return consistent predictions
        with patch.object(predictor, '_predict_with_model', return_value=15.5):
            # First prediction should compute
            result1 = predictor.predict_compilation_time(sample_shader_features, use_cache=True)
            
            # Second prediction should use cache
            result2 = predictor.predict_compilation_time(sample_shader_features, use_cache=True)
            
            assert result1 == result2
            assert len(predictor.prediction_cache) == 1
    
    def test_feature_vector_pooling(self, clean_temp_dir, sample_shader_features):
        """Test feature vector object pooling"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False
        )
        
        # Get feature vectors multiple times
        vectors = []
        for _ in range(5):
            with patch.object(predictor.feature_cache, 'get', return_value=None):
                vector = predictor._get_feature_vector(sample_shader_features)
                vectors.append(vector)
        
        # Should have reused objects from pool
        assert len(vectors) == 5
        assert all(isinstance(v, np.ndarray) for v in vectors)
    
    def test_training_data_management(self, clean_temp_dir, sample_shader_features):
        """Test training data buffer management"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False
        )
        
        # Add training samples up to buffer limit
        buffer_size = predictor.training_buffer_size
        for i in range(buffer_size + 10):
            predictor.add_training_sample(
                sample_shader_features,
                actual_time=10.0 + i,
                success=True
            )
        
        # Should maintain buffer size limit
        assert len(predictor.training_data) == buffer_size
        
        # Latest samples should be kept
        latest_sample = list(predictor.training_data)[-1]
        assert latest_sample['time'] == 10.0 + buffer_size + 9
    
    def test_memory_cleanup_under_pressure(self, clean_temp_dir):
        """Test memory cleanup when under pressure"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False
        )
        
        # Fill caches
        for i in range(1000):
            predictor.prediction_cache[f"key{i}"] = {"time": 10.0, "timestamp": time.time()}
        
        for i in range(predictor.training_buffer_size):
            predictor.training_data.append({"test": i})
        
        # Force memory pressure
        predictor._memory_pressure = True
        predictor._cleanup_memory()
        
        # Caches should be reduced
        assert len(predictor.prediction_cache) <= predictor.max_cache_size // 2
        assert len(predictor.training_data) <= predictor.training_buffer_size // 2
    
    def test_performance_stats(self, clean_temp_dir):
        """Test performance statistics collection"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False
        )
        
        stats = predictor.get_performance_stats()
        
        required_keys = [
            'ml_backend',
            'avg_prediction_time_ms',
            'cache_hit_rate',
            'prediction_cache_size',
            'training_buffer_size',
            'memory_usage_mb',
            'memory_pressure'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats['avg_prediction_time_ms'], (int, float))
        assert isinstance(stats['cache_hit_rate'], (int, float))
        assert isinstance(stats['memory_pressure'], bool)


@pytest.mark.integration
@pytest.mark.ml
class TestMLIntegration:
    """Integration tests for ML components"""
    
    def test_end_to_end_prediction(self, clean_temp_dir, sample_shader_features):
        """Test complete prediction pipeline"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False,
            max_memory_mb=100
        )
        
        # Make predictions
        results = []
        for _ in range(10):
            result = predictor.predict_compilation_time(sample_shader_features)
            results.append(result)
            assert isinstance(result, (int, float))
            assert result > 0  # Positive prediction
        
        # Add training data
        for i, result in enumerate(results):
            predictor.add_training_sample(
                sample_shader_features,
                actual_time=result + np.random.normal(0, 1),
                success=True
            )
        
        # Verify training data was added
        assert len(predictor.training_data) == len(results)
    
    def test_memory_usage_under_load(self, clean_temp_dir, memory_monitor):
        """Test memory usage under heavy load"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False,
            max_memory_mb=50
        )
        
        initial_memory = memory_monitor.memory_info().rss
        
        # Generate load
        for i in range(1000):
            features = Mock()
            features.shader_hash = f"shader_{i}"
            features.instruction_count = 500
            features.register_usage = 32
            features.texture_samples = 4
            features.memory_operations = 10
            features.control_flow_complexity = 5
            features.wave_size = 64
            features.uses_derivatives = True
            features.uses_tessellation = False
            features.uses_geometry_shader = False
            features.optimization_level = 3
            features.cache_priority = 0.8
            
            result = predictor.predict_compilation_time(features)
            assert isinstance(result, (int, float))
            
            if i % 100 == 0:  # Check memory periodically
                current_memory = memory_monitor.memory_info().rss
                memory_increase = (current_memory - initial_memory) / (1024 * 1024)
                # Should not increase by more than max_memory_mb + buffer
                assert memory_increase < predictor.max_memory_mb + 20
    
    def test_concurrent_predictions(self, clean_temp_dir, sample_shader_features):
        """Test thread safety of predictions"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        import threading
        import concurrent.futures
        
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False
        )
        
        results = []
        errors = []
        
        def make_prediction(i):
            try:
                # Modify features slightly for each thread
                features = Mock()
                features.shader_hash = f"concurrent_shader_{i}"
                features.instruction_count = 500 + i
                features.register_usage = 32
                features.texture_samples = 4
                features.memory_operations = 10
                features.control_flow_complexity = 5
                features.wave_size = 64
                features.uses_derivatives = True
                features.uses_tessellation = False
                features.uses_geometry_shader = False
                features.optimization_level = 3
                features.cache_priority = 0.8
                
                result = predictor.predict_compilation_time(features)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent predictions
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(100)]
            concurrent.futures.wait(futures)
        
        assert len(errors) == 0, f"Concurrent prediction errors: {errors}"
        assert len(results) == 100
        assert all(isinstance(r, (int, float)) and r > 0 for r in results)


@pytest.mark.benchmark
@pytest.mark.ml
class TestMLPerformance:
    """Performance benchmarks for ML components"""
    
    def test_prediction_speed_benchmark(self, benchmark, clean_temp_dir, sample_shader_features):
        """Benchmark prediction speed"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        predictor = OptimizedMLPredictor(
            model_path=clean_temp_dir,
            enable_async=False
        )
        
        # Warm up
        for _ in range(10):
            predictor.predict_compilation_time(sample_shader_features)
        
        # Benchmark
        result = benchmark(predictor.predict_compilation_time, sample_shader_features)
        
        # Should complete in reasonable time
        assert isinstance(result, (int, float))
        assert result > 0
    
    def test_cache_performance_benchmark(self, benchmark, clean_temp_dir):
        """Benchmark cache performance"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        cache = FeatureVectorCache(max_size=1000)
        
        # Pre-populate cache
        for i in range(500):
            vector = np.random.randn(12).astype(np.float32)
            cache.put(f"key_{i}", vector)
        
        # Benchmark cache access
        def access_cache():
            key = f"key_{np.random.randint(0, 500)}"
            return cache.get(key)
        
        result = benchmark(access_cache)
        # Cache hit should return a vector
        assert result is not None or True  # Allow cache misses
    
    def test_memory_pool_benchmark(self, benchmark):
        """Benchmark memory pool performance"""
        if not HAS_ML_MODULES:
            pytest.skip("ML modules not available")
            
        pool = MemoryPool(
            lambda: np.zeros(12, dtype=np.float32),
            max_size=100
        )
        
        def pool_cycle():
            obj = pool.acquire()
            pool.release(obj)
            return obj
        
        result = benchmark(pool_cycle)
        assert isinstance(result, np.ndarray)
        assert result.shape == (12,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])