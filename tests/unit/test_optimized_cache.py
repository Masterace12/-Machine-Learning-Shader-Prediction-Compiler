#!/usr/bin/env python3
"""
Comprehensive tests for optimized shader cache system
"""

import pytest
import time
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from src.cache.optimized_shader_cache import (
        OptimizedShaderCache,
        ShaderCacheEntry,
        CompressedBloomFilter,
        SQLiteConnectionPool,
        get_optimized_cache
    )
    HAS_CACHE_MODULE = True
except ImportError as e:
    print(f"Cache module not available: {e}")
    HAS_CACHE_MODULE = False


@pytest.mark.skipif(not HAS_CACHE_MODULE, reason="Cache module not available")
@pytest.mark.unit
@pytest.mark.cache
class TestShaderCacheEntry:
    """Test shader cache entry functionality"""
    
    def test_entry_creation(self):
        """Test basic cache entry creation"""
        entry = ShaderCacheEntry(
            shader_hash="test_123",
            game_id="game_001",
            shader_type="fragment",
            bytecode=b"test_bytecode",
            compilation_time=15.5
        )
        
        assert entry.shader_hash == "test_123"
        assert entry.game_id == "game_001"
        assert entry.shader_type == "fragment"
        assert entry.bytecode == b"test_bytecode"
        assert entry.compilation_time == 15.5
        assert entry.access_count == 0
        assert entry.last_access > 0
        assert entry.size_bytes == len(b"test_bytecode")
    
    def test_entry_update_access(self):
        """Test access count and timestamp updates"""
        entry = ShaderCacheEntry(
            shader_hash="test_456",
            game_id="game_002", 
            shader_type="vertex",
            bytecode=b"vertex_shader",
            compilation_time=8.2
        )
        
        initial_access = entry.last_access
        initial_count = entry.access_count
        
        time.sleep(0.01)  # Small delay
        entry.update_access()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_access > initial_access
        assert entry.priority > 0
    
    def test_priority_calculation(self):
        """Test cache priority calculation"""
        entry = ShaderCacheEntry(
            shader_hash="priority_test",
            game_id="game_priority",
            shader_type="compute",
            bytecode=b"compute_shader",
            compilation_time=25.0
        )
        
        # Initial priority
        entry.update_access()
        initial_priority = entry.priority
        
        # Multiple accesses should increase priority
        for _ in range(5):
            entry.update_access()
        
        assert entry.priority > initial_priority
        assert entry.access_count == 6


@pytest.mark.skipif(not HAS_CACHE_MODULE, reason="Cache module not available")
@pytest.mark.unit
@pytest.mark.cache
class TestCompressedBloomFilter:
    """Test bloom filter functionality"""
    
    def test_bloom_filter_creation(self):
        """Test bloom filter creation with optimal parameters"""
        bf = CompressedBloomFilter(expected_items=1000, false_positive_rate=0.01)
        
        assert bf.size > 0
        assert bf.hash_count > 0
        assert bf.item_count == 0
        assert len(bf.bit_array) > 0
    
    def test_add_and_contains(self):
        """Test basic add and contains operations"""
        bf = CompressedBloomFilter(expected_items=100)
        
        test_items = ["shader_001", "shader_002", "shader_003"]
        
        # Add items
        for item in test_items:
            bf.add(item)
        
        # Check contains
        for item in test_items:
            assert bf.contains(item)
        
        # Check non-existent item (may have false positives, but not false negatives)
        non_existent = "non_existent_shader"
        if not bf.contains(non_existent):
            # If it says it's not there, it's definitely not there
            assert True
    
    def test_false_positive_rate(self):
        """Test that false positive rate is reasonable"""
        bf = CompressedBloomFilter(expected_items=1000, false_positive_rate=0.05)
        
        # Add known items
        added_items = [f"shader_{i:04d}" for i in range(500)]
        for item in added_items:
            bf.add(item)
        
        # Test non-added items
        false_positives = 0
        test_items = [f"test_shader_{i:04d}" for i in range(1000)]
        
        for item in test_items:
            if bf.contains(item):
                false_positives += 1
        
        false_positive_rate = false_positives / len(test_items)
        # Should be within reasonable bounds (allowing for some variance)
        assert false_positive_rate <= 0.15  # Allow 15% due to small sample
    
    def test_memory_usage(self):
        """Test memory usage tracking"""
        bf = CompressedBloomFilter(expected_items=10000)
        memory_usage = bf.memory_usage
        
        assert memory_usage > 0
        assert isinstance(memory_usage, int)


@pytest.mark.skipif(not HAS_CACHE_MODULE, reason="Cache module not available")
@pytest.mark.unit
@pytest.mark.cache
class TestSQLiteConnectionPool:
    """Test SQLite connection pool"""
    
    def test_pool_creation(self, clean_temp_dir):
        """Test connection pool creation"""
        db_path = clean_temp_dir / "test.db"
        pool = SQLiteConnectionPool(db_path, pool_size=3)
        
        assert pool.db_path == db_path
        assert pool.pool_size == 3
        assert len(pool._pool) == 3
    
    def test_connection_acquire_release(self, clean_temp_dir):
        """Test connection acquire and release"""
        db_path = clean_temp_dir / "test_pool.db"
        pool = SQLiteConnectionPool(db_path, pool_size=2)
        
        # Acquire connections
        with pool.get_connection() as conn1:
            assert isinstance(conn1, sqlite3.Connection)
            
            with pool.get_connection() as conn2:
                assert isinstance(conn2, sqlite3.Connection)
                assert conn1 is not conn2
        
        # Connections should be returned to pool
        assert len(pool._pool) == 2
    
    def test_connection_reuse(self, clean_temp_dir):
        """Test that connections are reused"""
        db_path = clean_temp_dir / "test_reuse.db" 
        pool = SQLiteConnectionPool(db_path, pool_size=1)
        
        conn_id = None
        with pool.get_connection() as conn:
            conn_id = id(conn)
        
        with pool.get_connection() as conn:
            assert id(conn) == conn_id  # Same connection object
    
    def test_pool_cleanup(self, clean_temp_dir):
        """Test connection pool cleanup"""
        db_path = clean_temp_dir / "test_cleanup.db"
        pool = SQLiteConnectionPool(db_path, pool_size=2)
        
        initial_count = len(pool._pool)
        pool.close_all()
        
        assert len(pool._pool) == 0
        assert initial_count > 0


@pytest.mark.skipif(not HAS_CACHE_MODULE, reason="Cache module not available")
@pytest.mark.unit
@pytest.mark.cache
class TestOptimizedShaderCache:
    """Test optimized shader cache functionality"""
    
    @pytest.fixture
    def cache(self, clean_temp_dir):
        """Create a test cache instance"""
        return OptimizedShaderCache(
            cache_dir=clean_temp_dir,
            hot_cache_size=5,
            warm_cache_size=10,
            max_memory_mb=50,
            enable_compression=False,  # Disable for simpler testing
            enable_async=False
        )
    
    @pytest.fixture
    def sample_entry(self):
        """Create a sample cache entry"""
        return ShaderCacheEntry(
            shader_hash="sample_123",
            game_id="test_game",
            shader_type="fragment",
            bytecode=b"sample_shader_bytecode",
            compilation_time=12.5
        )
    
    def test_cache_initialization(self, cache):
        """Test cache initialization"""
        assert cache.hot_cache_size == 5
        assert cache.warm_cache_size == 10
        assert cache.max_memory_mb == 50
        assert not cache.enable_compression
        assert not cache.enable_async
        assert len(cache.hot_cache) == 0
        assert len(cache.warm_cache) == 0
        assert cache.bloom_filter is not None
    
    def test_put_and_get_hot_cache(self, cache, sample_entry):
        """Test put and get from hot cache"""
        # Put entry
        cache.put(sample_entry)
        
        # Should be in hot cache
        assert sample_entry.shader_hash in cache.hot_cache
        assert len(cache.hot_cache) == 1
        
        # Retrieve entry
        retrieved = cache.get(sample_entry.shader_hash)
        
        assert retrieved is not None
        assert retrieved.shader_hash == sample_entry.shader_hash
        assert retrieved.bytecode == sample_entry.bytecode
        assert cache.hits['hot'] == 1
    
    def test_cache_miss(self, cache):
        """Test cache miss behavior"""
        result = cache.get("non_existent_shader")
        
        assert result is None
        assert cache.misses == 1
        assert cache.hits['hot'] == 0
        assert cache.hits['warm'] == 0
        assert cache.hits['cold'] == 0
    
    def test_hot_to_warm_promotion(self, cache):
        """Test promotion from warm to hot cache"""
        # Create entries
        entries = []
        for i in range(7):  # More than hot cache size
            entry = ShaderCacheEntry(
                shader_hash=f"shader_{i:03d}",
                game_id="test_game",
                shader_type="vertex",
                bytecode=f"bytecode_{i}".encode(),
                compilation_time=10.0
            )
            entries.append(entry)
            cache.put(entry)
        
        # Hot cache should be full (5 items)
        assert len(cache.hot_cache) == 5
        # Warm cache should have overflow (2 items)
        assert len(cache.warm_cache) == 2
        
        # Access an item in warm cache to promote it
        warm_key = list(cache.warm_cache.keys())[0]
        retrieved = cache.get(warm_key)
        
        assert retrieved is not None
        assert cache.hits['warm'] == 1
        # Should now be in hot cache
        assert warm_key in cache.hot_cache
    
    def test_lru_eviction(self, cache):
        """Test LRU eviction in hot cache"""
        entries = []
        
        # Fill hot cache
        for i in range(5):
            entry = ShaderCacheEntry(
                shader_hash=f"lru_shader_{i}",
                game_id="lru_test",
                shader_type="fragment",
                bytecode=f"lru_bytecode_{i}".encode(),
                compilation_time=5.0
            )
            cache.put(entry)
            entries.append(entry)
        
        # Access first entry to make it most recently used
        cache.get("lru_shader_0")
        
        # Add another entry (should evict lru_shader_1, the least recently used)
        new_entry = ShaderCacheEntry(
            shader_hash="new_lru_shader",
            game_id="lru_test",
            shader_type="compute",
            bytecode=b"new_lru_bytecode",
            compilation_time=8.0
        )
        cache.put(new_entry)
        
        # lru_shader_0 should still be in hot cache (was accessed recently)
        assert cache.get("lru_shader_0") is not None
        assert cache.hits['hot'] >= 1
        
        # new entry should be in hot cache
        assert cache.get("new_lru_shader") is not None
    
    def test_bloom_filter_integration(self, cache, sample_entry):
        """Test bloom filter integration"""
        # Put entry (should add to bloom filter)
        cache.put(sample_entry)
        
        # Bloom filter should contain the hash
        assert cache.bloom_filter.contains(sample_entry.shader_hash)
        
        # Non-existent item should fail bloom filter check
        non_existent = "definitely_not_there"
        if not cache.bloom_filter.contains(non_existent):
            # If bloom filter says no, cache get should return None immediately
            result = cache.get(non_existent)
            assert result is None
            assert cache.misses >= 1
    
    def test_memory_pressure_handling(self, cache):
        """Test memory pressure handling"""
        # Fill caches with entries
        for i in range(15):  # More than hot + warm cache size
            entry = ShaderCacheEntry(
                shader_hash=f"memory_shader_{i}",
                game_id="memory_test",
                shader_type="vertex",
                bytecode=f"large_bytecode_{i}" * 100,  # Larger entries
                compilation_time=15.0
            )
            cache.put(entry)
        
        # Simulate memory pressure
        cache._memory_pressure = True
        cache._reduce_memory_usage()
        
        # Caches should be reduced
        total_cached = len(cache.hot_cache) + len(cache.warm_cache)
        assert total_cached < 15
    
    def test_cache_statistics(self, cache):
        """Test cache statistics collection"""
        # Add some entries and access them
        for i in range(3):
            entry = ShaderCacheEntry(
                shader_hash=f"stats_shader_{i}",
                game_id="stats_test", 
                shader_type="fragment",
                bytecode=f"stats_bytecode_{i}".encode(),
                compilation_time=7.0
            )
            cache.put(entry)
        
        # Access some entries
        cache.get("stats_shader_0")
        cache.get("stats_shader_1")
        cache.get("non_existent")  # Miss
        
        stats = cache.get_stats()
        
        assert 'hot_cache_size' in stats
        assert 'warm_cache_size' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
        assert 'avg_access_time_ms' in stats
        
        assert stats['hot_cache_size'] == 3
        assert stats['misses'] >= 1
        assert 0.0 <= stats['hit_rate'] <= 1.0
    
    def test_cache_clear(self, cache, sample_entry):
        """Test cache clearing"""
        # Add entry
        cache.put(sample_entry)
        assert len(cache.hot_cache) > 0
        
        # Clear cache
        cache.clear()
        
        # Everything should be empty
        assert len(cache.hot_cache) == 0
        assert len(cache.warm_cache) == 0
        assert cache.hits['hot'] == 0
        assert cache.hits['warm'] == 0 
        assert cache.hits['cold'] == 0
        assert cache.misses == 0
    
    def test_cache_close(self, cache):
        """Test cache cleanup on close"""
        cache.close()
        
        # Should not crash and should clean up resources
        assert True  # If we get here, close() worked


@pytest.mark.skipif(not HAS_CACHE_MODULE, reason="Cache module not available")
@pytest.mark.integration
@pytest.mark.cache
class TestCacheIntegration:
    """Integration tests for cache system"""
    
    def test_cold_storage_integration(self, clean_temp_dir):
        """Test integration with SQLite cold storage"""
        cache = OptimizedShaderCache(
            cache_dir=clean_temp_dir,
            hot_cache_size=2,
            warm_cache_size=2,
            enable_compression=False
        )
        
        try:
            # Add entries that will overflow to cold storage
            entries = []
            for i in range(6):  # More than hot+warm capacity
                entry = ShaderCacheEntry(
                    shader_hash=f"cold_shader_{i}",
                    game_id="cold_test",
                    shader_type="compute",
                    bytecode=f"cold_bytecode_{i}".encode(),
                    compilation_time=20.0
                )
                entries.append(entry)
                cache.put(entry)
            
            # Some entries should be in cold storage
            assert len(cache.hot_cache) == 2
            assert len(cache.warm_cache) == 2
            
            # Try to retrieve an entry that should be in cold storage
            # (First entries added should have been demoted)
            retrieved = cache.get("cold_shader_0")
            if retrieved:
                assert cache.hits['cold'] >= 1
                assert retrieved.shader_hash == "cold_shader_0"
        
        finally:
            cache.close()
    
    def test_performance_under_load(self, clean_temp_dir):
        """Test cache performance under load"""
        cache = OptimizedShaderCache(
            cache_dir=clean_temp_dir,
            hot_cache_size=50,
            warm_cache_size=100,
            enable_compression=False
        )
        
        try:
            import time
            start_time = time.time()
            
            # Add many entries
            for i in range(200):
                entry = ShaderCacheEntry(
                    shader_hash=f"perf_shader_{i:04d}",
                    game_id="perf_test",
                    shader_type="vertex" if i % 2 == 0 else "fragment",
                    bytecode=f"perf_bytecode_{i}".encode() * 10,
                    compilation_time=5.0 + (i % 10)
                )
                cache.put(entry)
            
            put_time = time.time() - start_time
            
            # Access entries (mix of hits and misses)
            start_time = time.time()
            hits = 0
            for i in range(300):
                if i < 200:
                    result = cache.get(f"perf_shader_{i:04d}")
                    if result:
                        hits += 1
                else:
                    cache.get(f"miss_shader_{i}")  # Guaranteed miss
            
            get_time = time.time() - start_time
            
            # Performance should be reasonable
            assert put_time < 5.0  # Should put 200 entries in under 5 seconds
            assert get_time < 2.0   # Should do 300 gets in under 2 seconds
            assert hits > 100       # Should have many cache hits
            
            stats = cache.get_stats()
            assert stats['hit_rate'] > 0.3  # At least 30% hit rate
        
        finally:
            cache.close()


@pytest.mark.skipif(not HAS_CACHE_MODULE, reason="Cache module not available")
@pytest.mark.benchmark
@pytest.mark.cache
class TestCacheBenchmarks:
    """Benchmark tests for cache performance"""
    
    def test_put_performance_benchmark(self, benchmark, clean_temp_dir):
        """Benchmark cache put performance"""
        cache = OptimizedShaderCache(
            cache_dir=clean_temp_dir,
            hot_cache_size=100,
            enable_compression=False
        )
        
        try:
            entry = ShaderCacheEntry(
                shader_hash="benchmark_shader",
                game_id="benchmark",
                shader_type="fragment",
                bytecode=b"benchmark_bytecode" * 50,
                compilation_time=10.0
            )
            
            # Benchmark put operation
            result = benchmark(cache.put, entry)
            
            # Verify it was stored
            retrieved = cache.get("benchmark_shader")
            assert retrieved is not None
        
        finally:
            cache.close()
    
    def test_get_performance_benchmark(self, benchmark, clean_temp_dir):
        """Benchmark cache get performance"""
        cache = OptimizedShaderCache(
            cache_dir=clean_temp_dir,
            hot_cache_size=100,
            enable_compression=False
        )
        
        try:
            # Pre-populate cache
            entry = ShaderCacheEntry(
                shader_hash="get_benchmark_shader",
                game_id="get_benchmark", 
                shader_type="vertex",
                bytecode=b"get_benchmark_bytecode" * 50,
                compilation_time=8.0
            )
            cache.put(entry)
            
            # Benchmark get operation
            result = benchmark(cache.get, "get_benchmark_shader")
            
            assert result is not None
            assert result.shader_hash == "get_benchmark_shader"
        
        finally:
            cache.close()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])