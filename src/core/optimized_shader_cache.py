#!/usr/bin/env python3
"""
Optimized Multi-tier Shader Cache System for Steam Deck
High-performance caching with async I/O, memory management, and compression
"""

import os
import sys
import time
import json
import sqlite3
import hashlib
import asyncio
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
import threading
import mmap
import zlib
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    lz4 = None
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import OrderedDict, deque
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor
import logging
import struct
import heapq

import numpy as np
import psutil


@dataclass
class ShaderCacheEntry:
    """Optimized shader cache entry with compression"""
    shader_hash: str
    game_id: str
    shader_type: str
    bytecode: bytes  # Compressed bytecode
    compilation_time: float
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0
    compression_ratio: float = 1.0
    priority: float = 0.0
    
    def __post_init__(self):
        """Calculate size and update timestamp"""
        if self.bytecode:
            self.size_bytes = len(self.bytecode)
        if self.last_access == 0:
            self.last_access = time.time()
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()
        # Update priority based on access pattern
        self.priority = self._calculate_priority()
    
    def _calculate_priority(self) -> float:
        """Calculate cache priority score"""
        age = time.time() - self.last_access
        # Higher score = higher priority to keep in cache
        return (self.access_count * 10) / (age + 1)


class CompressedBloomFilter:
    """Memory-efficient bloom filter with dynamic sizing"""
    
    def __init__(self, expected_items: int = 10000, false_positive_rate: float = 0.01):
        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_items)
        self.bit_array = bytearray((self.size + 7) // 8)
        self.item_count = 0
        self._lock = threading.RLock()
    
    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bloom filter size"""
        if n == 0:
            return 1
        m = -n * np.log(p) / (np.log(2) ** 2)
        return int(m)
    
    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions"""
        if n == 0:
            return 1
        k = (m / n) * np.log(2)
        return max(1, int(k))
    
    def _hash(self, item: str, seed: int) -> int:
        """Generate hash with seed"""
        h = hashlib.sha256(f"{item}{seed}".encode()).digest()
        return int.from_bytes(h[:4], 'big') % self.size
    
    def add(self, item: str):
        """Add item to bloom filter"""
        with self._lock:
            for i in range(self.hash_count):
                pos = self._hash(item, i)
                byte_pos = pos // 8
                bit_pos = pos % 8
                self.bit_array[byte_pos] |= (1 << bit_pos)
            self.item_count += 1
    
    def contains(self, item: str) -> bool:
        """Check if item might be in the set"""
        with self._lock:
            for i in range(self.hash_count):
                pos = self._hash(item, i)
                byte_pos = pos // 8
                bit_pos = pos % 8
                if not (self.bit_array[byte_pos] & (1 << bit_pos)):
                    return False
            return True
    
    @property
    def memory_usage(self) -> int:
        """Get memory usage in bytes"""
        return len(self.bit_array)


class SQLiteConnectionPool:
    """Connection pool for SQLite with WAL mode"""
    
    def __init__(self, db_path: Path, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = deque(maxlen=pool_size)
        self._lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool with optimized settings"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
            self._pool.append(conn)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        with self._lock:
            while not self._pool:
                time.sleep(0.001)  # Wait for available connection
            conn = self._pool.popleft()
        
        try:
            yield conn
        finally:
            with self._lock:
                self._pool.append(conn)
    
    def close_all(self):
        """Close all connections"""
        with self._lock:
            while self._pool:
                conn = self._pool.popleft()
                conn.close()


class OptimizedShaderCache:
    """High-performance multi-tier shader cache with async support"""
    
    def __init__(self, cache_dir: Optional[Path] = None,
                 hot_cache_size: int = 100,
                 warm_cache_size: int = 500,
                 max_memory_mb: int = 100,
                 enable_compression: bool = True,
                 enable_async: bool = True):
        """
        Initialize optimized shader cache
        
        Args:
            cache_dir: Directory for cache storage
            hot_cache_size: Maximum hot cache entries
            warm_cache_size: Maximum warm cache entries
            max_memory_mb: Maximum memory usage in MB
            enable_compression: Enable LZ4 compression
            enable_async: Enable async operations
        """
        # Cache directory setup
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'shader-predict-compile'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache tiers
        self.hot_cache = OrderedDict()  # In-memory, most recently used
        self.warm_cache = OrderedDict()  # In-memory, less recently used
        self.hot_cache_size = hot_cache_size
        self.warm_cache_size = warm_cache_size
        
        # Memory management
        self.max_memory_mb = max_memory_mb
        self._current_memory_mb = 0
        self._memory_pressure = False
        
        # Compression
        self.enable_compression = enable_compression
        self.compression_level = 3  # LZ4 compression level
        
        # Bloom filter for quick existence checks
        self.bloom_filter = CompressedBloomFilter(expected_items=50000)
        
        # SQLite for cold storage with connection pool
        self.db_path = self.cache_dir / 'shader_cache_optimized.db'
        self.db_pool = SQLiteConnectionPool(self.db_path, pool_size=3)
        self._init_database()
        
        # Async support
        self.enable_async = enable_async
        if enable_async:
            self.io_pool = ThreadPoolExecutor(max_workers=2)
        
        # Performance metrics
        self.hits = {'hot': 0, 'warm': 0, 'cold': 0}
        self.misses = 0
        self._access_times = deque(maxlen=100)
        
        # Eviction heap for efficient LRU
        self._eviction_heap = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Start background tasks
        self._start_background_tasks()
    
    def _init_database(self):
        """Initialize SQLite database with optimized schema"""
        with self.db_pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS shader_cache (
                    shader_hash TEXT PRIMARY KEY,
                    game_id TEXT NOT NULL,
                    shader_type TEXT NOT NULL,
                    bytecode BLOB NOT NULL,
                    compilation_time REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_access REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    compression_ratio REAL DEFAULT 1.0,
                    priority REAL DEFAULT 0.0,
                    created_at REAL NOT NULL
                )
            ''')
            
            # Create indexes for fast lookups
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_game_id 
                ON shader_cache(game_id)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_priority 
                ON shader_cache(priority DESC)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_access 
                ON shader_cache(last_access DESC)
            ''')
            
            conn.commit()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def maintenance_loop():
            while True:
                time.sleep(60)  # Run every minute
                self._perform_maintenance()
        
        maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_thread.start()
    
    def _perform_maintenance(self):
        """Perform cache maintenance tasks"""
        try:
            # Check memory pressure
            self._check_memory_pressure()
            
            # Promote/demote cache entries based on access patterns
            self._rebalance_caches()
            
            # Clean up old entries from cold storage
            self._cleanup_cold_storage()
            
        except Exception as e:
            self.logger.error(f"Maintenance error: {e}")
    
    def _check_memory_pressure(self):
        """Monitor and respond to memory pressure"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self._current_memory_mb = memory_mb
        
        if memory_mb > self.max_memory_mb:
            self._memory_pressure = True
            self._reduce_memory_usage()
        else:
            self._memory_pressure = False
    
    def _reduce_memory_usage(self):
        """Reduce memory usage when under pressure"""
        # Evict from warm cache first
        evict_count = len(self.warm_cache) // 3
        for _ in range(evict_count):
            if self.warm_cache:
                self.warm_cache.popitem(last=False)
        
        # Then reduce hot cache if needed
        if self._current_memory_mb > self.max_memory_mb * 0.9:
            evict_count = len(self.hot_cache) // 4
            for _ in range(evict_count):
                if self.hot_cache:
                    key, entry = self.hot_cache.popitem(last=False)
                    # Move to warm cache instead of discarding
                    if len(self.warm_cache) < self.warm_cache_size:
                        self.warm_cache[key] = entry
        
        self.logger.info(f"Memory reduced to {self._current_memory_mb:.1f}MB")
    
    def _compress_bytecode(self, bytecode: bytes) -> Tuple[bytes, float]:
        """Compress shader bytecode using LZ4"""
        if not self.enable_compression or len(bytecode) < 1024 or not HAS_LZ4:
            return bytecode, 1.0
        
        try:
            compressed = lz4.frame.compress(
                bytecode,
                compression_level=self.compression_level
            )
            ratio = len(bytecode) / len(compressed)
            return compressed, ratio
        except Exception:
            return bytecode, 1.0
    
    def _decompress_bytecode(self, compressed: bytes, ratio: float) -> bytes:
        """Decompress shader bytecode"""
        if not self.enable_compression or ratio == 1.0 or not HAS_LZ4:
            return compressed
        
        try:
            return lz4.frame.decompress(compressed)
        except Exception:
            return compressed
    
    def get(self, shader_hash: str) -> Optional[ShaderCacheEntry]:
        """Get shader from cache with multi-tier lookup"""
        start_time = time.perf_counter()
        
        # Quick bloom filter check
        if not self.bloom_filter.contains(shader_hash):
            self.misses += 1
            return None
        
        # Check hot cache (fastest)
        if shader_hash in self.hot_cache:
            entry = self.hot_cache[shader_hash]
            # Move to end (most recently used)
            self.hot_cache.move_to_end(shader_hash)
            entry.update_access()
            self.hits['hot'] += 1
            self._record_access_time(start_time)
            return entry
        
        # Check warm cache
        if shader_hash in self.warm_cache:
            entry = self.warm_cache[shader_hash]
            entry.update_access()
            # Promote to hot cache
            self._promote_to_hot(shader_hash, entry)
            self.hits['warm'] += 1
            self._record_access_time(start_time)
            return entry
        
        # Check cold storage (database)
        entry = self._get_from_cold_storage(shader_hash)
        if entry:
            # Promote to warm cache
            if len(self.warm_cache) >= self.warm_cache_size:
                self.warm_cache.popitem(last=False)
            self.warm_cache[shader_hash] = entry
            self.hits['cold'] += 1
            self._record_access_time(start_time)
            return entry
        
        self.misses += 1
        return None
    
    async def get_async(self, shader_hash: str) -> Optional[ShaderCacheEntry]:
        """Async version of get"""
        if not self.enable_async:
            return self.get(shader_hash)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, self.get, shader_hash)
    
    def put(self, entry: ShaderCacheEntry):
        """Add shader to cache"""
        # Compress bytecode if enabled
        if self.enable_compression and entry.bytecode:
            compressed, ratio = self._compress_bytecode(entry.bytecode)
            entry.bytecode = compressed
            entry.compression_ratio = ratio
        
        # Add to bloom filter
        self.bloom_filter.add(entry.shader_hash)
        
        # Add to hot cache
        if len(self.hot_cache) >= self.hot_cache_size:
            # Evict least recently used
            evicted_key, evicted_entry = self.hot_cache.popitem(last=False)
            # Demote to warm cache
            if len(self.warm_cache) >= self.warm_cache_size:
                # Demote warm to cold
                cold_key, cold_entry = self.warm_cache.popitem(last=False)
                self._put_to_cold_storage(cold_entry)
            self.warm_cache[evicted_key] = evicted_entry
        
        self.hot_cache[entry.shader_hash] = entry
        entry.update_access()
    
    async def put_async(self, entry: ShaderCacheEntry):
        """Async version of put"""
        if not self.enable_async:
            return self.put(entry)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.io_pool, self.put, entry)
    
    def _promote_to_hot(self, key: str, entry: ShaderCacheEntry):
        """Promote entry from warm to hot cache"""
        # Remove from warm
        del self.warm_cache[key]
        
        # Make room in hot cache if needed
        if len(self.hot_cache) >= self.hot_cache_size:
            # Demote least recently used
            demoted_key, demoted_entry = self.hot_cache.popitem(last=False)
            self.warm_cache[demoted_key] = demoted_entry
        
        # Add to hot cache
        self.hot_cache[key] = entry
    
    def _get_from_cold_storage(self, shader_hash: str) -> Optional[ShaderCacheEntry]:
        """Get entry from SQLite database"""
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT game_id, shader_type, bytecode, compilation_time,
                           access_count, last_access, size_bytes, compression_ratio,
                           priority
                    FROM shader_cache
                    WHERE shader_hash = ?
                ''', (shader_hash,))
                
                row = cursor.fetchone()
                if row:
                    entry = ShaderCacheEntry(
                        shader_hash=shader_hash,
                        game_id=row[0],
                        shader_type=row[1],
                        bytecode=row[2],
                        compilation_time=row[3],
                        access_count=row[4],
                        last_access=row[5],
                        size_bytes=row[6],
                        compression_ratio=row[7],
                        priority=row[8]
                    )
                    
                    # Update access count in database
                    conn.execute('''
                        UPDATE shader_cache
                        SET access_count = access_count + 1,
                            last_access = ?
                        WHERE shader_hash = ?
                    ''', (time.time(), shader_hash))
                    conn.commit()
                    
                    return entry
        except Exception as e:
            self.logger.error(f"Cold storage read error: {e}")
        
        return None
    
    def _put_to_cold_storage(self, entry: ShaderCacheEntry):
        """Store entry in SQLite database"""
        try:
            with self.db_pool.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO shader_cache
                    (shader_hash, game_id, shader_type, bytecode, compilation_time,
                     access_count, last_access, size_bytes, compression_ratio,
                     priority, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.shader_hash, entry.game_id, entry.shader_type,
                    entry.bytecode, entry.compilation_time, entry.access_count,
                    entry.last_access, entry.size_bytes, entry.compression_ratio,
                    entry.priority, time.time()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Cold storage write error: {e}")
    
    def _rebalance_caches(self):
        """Rebalance cache tiers based on access patterns"""
        # Promote frequently accessed warm entries to hot
        for key, entry in list(self.warm_cache.items())[:10]:
            if entry.priority > 50:  # High priority threshold
                self._promote_to_hot(key, entry)
        
        # Demote rarely accessed hot entries to warm
        for key, entry in list(self.hot_cache.items())[:10]:
            if entry.priority < 10:  # Low priority threshold
                if len(self.warm_cache) < self.warm_cache_size:
                    del self.hot_cache[key]
                    self.warm_cache[key] = entry
    
    def _cleanup_cold_storage(self):
        """Clean up old entries from cold storage"""
        try:
            with self.db_pool.get_connection() as conn:
                # Delete entries not accessed in 30 days
                cutoff_time = time.time() - (30 * 24 * 3600)
                conn.execute('''
                    DELETE FROM shader_cache
                    WHERE last_access < ? AND priority < 5
                ''', (cutoff_time,))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Cold storage cleanup error: {e}")
    
    def _record_access_time(self, start_time: float):
        """Record access time for performance metrics"""
        access_time = (time.perf_counter() - start_time) * 1000
        self._access_times.append(access_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(self.hits.values())
        total_requests = total_hits + self.misses
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        avg_access_time = (
            sum(self._access_times) / len(self._access_times)
            if self._access_times else 0
        )
        
        return {
            'hot_cache_size': len(self.hot_cache),
            'warm_cache_size': len(self.warm_cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'avg_access_time_ms': avg_access_time,
            'memory_usage_mb': self._current_memory_mb,
            'memory_pressure': self._memory_pressure,
            'bloom_filter_size_bytes': self.bloom_filter.memory_usage
        }
    
    def clear(self):
        """Clear all cache tiers"""
        self.hot_cache.clear()
        self.warm_cache.clear()
        
        with self.db_pool.get_connection() as conn:
            conn.execute("DELETE FROM shader_cache")
            conn.commit()
        
        self.bloom_filter = CompressedBloomFilter(expected_items=50000)
        self.hits = {'hot': 0, 'warm': 0, 'cold': 0}
        self.misses = 0
        
        self.logger.info("Cache cleared")
    
    def close(self):
        """Clean up resources"""
        if self.enable_async:
            self.io_pool.shutdown(wait=False)
        
        self.db_pool.close_all()
        self.logger.info("Cache closed")


# Singleton instance
_global_cache = None


def get_optimized_cache() -> OptimizedShaderCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = OptimizedShaderCache()
    return _global_cache


if __name__ == "__main__":
    # Test the optimized cache
    logging.basicConfig(level=logging.INFO)
    
    cache = get_optimized_cache()
    
    # Create test entry
    test_entry = ShaderCacheEntry(
        shader_hash="test_shader_123",
        game_id="game_001",
        shader_type="fragment",
        bytecode=b"x" * 10000,  # Test data
        compilation_time=15.5
    )
    
    print("\n=== Optimized Shader Cache Test ===")
    
    # Test put and get
    cache.put(test_entry)
    retrieved = cache.get("test_shader_123")
    
    if retrieved:
        print(f"✓ Cache put/get successful")
        print(f"  Compression ratio: {retrieved.compression_ratio:.2f}x")
    
    # Test performance
    import timeit
    
    # Warm up cache
    for i in range(100):
        entry = ShaderCacheEntry(
            shader_hash=f"shader_{i}",
            game_id="game_001",
            shader_type="vertex",
            bytecode=b"x" * 5000,
            compilation_time=10.0
        )
        cache.put(entry)
    
    # Measure get performance
    def test_get():
        cache.get(f"shader_{np.random.randint(0, 100)}")
    
    time_taken = timeit.timeit(test_get, number=1000) / 1000 * 1000
    
    print(f"\nAverage cache lookup time: {time_taken:.2f}ms")
    
    # Get statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    cache.close()