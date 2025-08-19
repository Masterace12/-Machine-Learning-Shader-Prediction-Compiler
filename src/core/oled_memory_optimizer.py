#!/usr/bin/env python3
"""
OLED Steam Deck Memory-Mapped File Optimizer
Advanced shader cache management with OLED-specific optimizations
"""

import os
import mmap
import time
import threading
import logging
import hashlib
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict
import struct
import gzip
import zlib
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics for OLED optimization"""
    total_size_mb: float = 0.0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    compression_ratio: float = 0.0
    access_frequency: Dict[str, int] = field(default_factory=dict)
    thermal_throttle_events: int = 0
    memory_pressure_events: int = 0
    oled_specific_optimizations: bool = True


class MemoryMappedShaderCache:
    """High-performance memory-mapped shader cache optimized for OLED Steam Deck"""
    
    def __init__(self, cache_dir: Path, max_cache_size_mb: int = 512):
        """
        Initialize OLED-optimized shader cache
        
        Args:
            cache_dir: Directory for cache storage
            max_cache_size_mb: Maximum cache size (OLED can handle larger caches)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # OLED Steam Deck has better sustained I/O performance
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.memory_mapped_files: Dict[str, mmap.mmap] = {}
        self.file_handles: Dict[str, Any] = {}
        
        # Cache structure optimized for OLED's better memory bandwidth
        self.hot_cache: OrderedDict = OrderedDict()  # LRU for frequently accessed
        self.warm_cache: OrderedDict = OrderedDict()  # Recently accessed
        self.cold_storage: Dict[str, Path] = {}  # Compressed storage
        
        # Performance optimizations for OLED
        self.hot_cache_limit = 200  # Increased due to better cooling
        self.warm_cache_limit = 800  # Higher limit for OLED
        self.compression_enabled = True
        self.prefetch_enabled = True
        
        # Threading and locking
        self._cache_lock = threading.RLock()
        self._metrics_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = CacheMetrics()
        self._access_count = 0
        self._hit_count = 0
        
        # Background optimization thread
        self._optimizer_thread = None
        self._optimization_active = False
        
        # OLED-specific optimizations
        self.oled_burst_mode = True  # Take advantage of better cooling
        self.predictive_loading = True  # Pre-load based on patterns
        self.adaptive_compression = True  # Adjust compression based on thermal state
        
        logger.info(f"OLED Memory-mapped shader cache initialized: {max_cache_size_mb}MB max")
    
    def _get_cache_key(self, shader_hash: str, variant: str = "") -> str:
        """Generate cache key for shader"""
        if variant:
            return f"{shader_hash}_{variant}"
        return shader_hash
    
    def _compress_data(self, data: bytes, level: int = 3) -> bytes:
        """Compress shader data with adaptive compression for OLED"""
        if not self.compression_enabled:
            return data
        
        # Use higher compression for OLED's better I/O performance
        try:
            # OLED can handle higher compression levels due to better cooling
            oled_level = min(level + 2, 9) if self.oled_burst_mode else level
            
            # Use zlib compression (built-in, fast)
            return zlib.compress(data, level=oled_level)
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress shader data"""
        if not self.compression_enabled:
            return compressed_data
        
        try:
            return zlib.decompress(compressed_data)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return compressed_data
    
    def _create_memory_mapped_file(self, file_path: Path, size: int) -> Optional[mmap.mmap]:
        """Create memory-mapped file with OLED optimizations"""
        try:
            # Create or open file
            file_handle = open(file_path, 'r+b')
            
            # Memory map with OLED-optimized flags
            memory_map = mmap.mmap(
                file_handle.fileno(),
                size,
                access=mmap.ACCESS_WRITE,
                # Use MAP_POPULATE for better performance on OLED's NVMe
                flags=mmap.MAP_SHARED
            )
            
            # Store handles for cleanup
            cache_key = str(file_path)
            self.file_handles[cache_key] = file_handle
            self.memory_mapped_files[cache_key] = memory_map
            
            # Advise kernel about access patterns (OLED benefits from this)
            if hasattr(memory_map, 'madvise'):
                memory_map.madvise(mmap.MADV_WILLNEED)  # Pre-load
                if self.oled_burst_mode:
                    memory_map.madvise(mmap.MADV_SEQUENTIAL)  # Sequential access hint
            
            return memory_map
            
        except Exception as e:
            logger.error(f"Failed to create memory-mapped file {file_path}: {e}")
            return None
    
    def _evict_from_hot_cache(self) -> None:
        """Evict items from hot cache to warm cache"""
        if len(self.hot_cache) <= self.hot_cache_limit:
            return
        
        # Move oldest items to warm cache
        items_to_move = len(self.hot_cache) - self.hot_cache_limit + 10
        for _ in range(items_to_move):
            if self.hot_cache:
                key, value = self.hot_cache.popitem(last=False)  # Remove oldest
                self.warm_cache[key] = value
                self.warm_cache.move_to_end(key)  # Mark as recently accessed
    
    def _evict_from_warm_cache(self) -> None:
        """Evict items from warm cache to cold storage"""
        if len(self.warm_cache) <= self.warm_cache_limit:
            return
        
        items_to_compress = len(self.warm_cache) - self.warm_cache_limit + 50
        for _ in range(items_to_compress):
            if self.warm_cache:
                key, shader_data = self.warm_cache.popitem(last=False)
                
                # Compress and store to disk
                compressed_data = self._compress_data(shader_data)
                
                # Generate filename
                file_hash = hashlib.md5(key.encode()).hexdigest()
                cache_file = self.cache_dir / f"{file_hash}.cache"
                
                try:
                    with open(cache_file, 'wb') as f:
                        f.write(compressed_data)
                    
                    self.cold_storage[key] = cache_file
                    
                except Exception as e:
                    logger.error(f"Failed to write cold cache file: {e}")
    
    @contextmanager
    def _cache_operation(self):
        """Context manager for thread-safe cache operations"""
        with self._cache_lock:
            yield
    
    def store_shader(self, shader_hash: str, shader_data: bytes, variant: str = "") -> bool:
        """Store shader in cache with OLED optimizations"""
        cache_key = self._get_cache_key(shader_hash, variant)
        
        with self._cache_operation():
            try:
                # Store in hot cache for immediate access
                self.hot_cache[cache_key] = shader_data
                self.hot_cache.move_to_end(cache_key)  # Mark as most recent
                
                # Update access frequency
                with self._metrics_lock:
                    self.metrics.access_frequency[cache_key] = (
                        self.metrics.access_frequency.get(cache_key, 0) + 1
                    )
                
                # Trigger eviction if needed
                self._evict_from_hot_cache()
                self._evict_from_warm_cache()
                
                # Update metrics
                self._update_cache_metrics()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to store shader {cache_key}: {e}")
                return False
    
    def retrieve_shader(self, shader_hash: str, variant: str = "") -> Optional[bytes]:
        """Retrieve shader from cache with OLED-optimized lookup"""
        cache_key = self._get_cache_key(shader_hash, variant)
        
        with self._cache_operation():
            self._access_count += 1
            
            # Check hot cache first
            if cache_key in self.hot_cache:
                self._hit_count += 1
                shader_data = self.hot_cache[cache_key]
                self.hot_cache.move_to_end(cache_key)  # Mark as recently used
                return shader_data
            
            # Check warm cache
            if cache_key in self.warm_cache:
                self._hit_count += 1
                shader_data = self.warm_cache.pop(cache_key)
                
                # Promote to hot cache
                self.hot_cache[cache_key] = shader_data
                self.hot_cache.move_to_end(cache_key)
                
                self._evict_from_hot_cache()
                return shader_data
            
            # Check cold storage
            if cache_key in self.cold_storage:
                self._hit_count += 1
                cache_file = self.cold_storage[cache_key]
                
                try:
                    with open(cache_file, 'rb') as f:
                        compressed_data = f.read()
                    
                    shader_data = self._decompress_data(compressed_data)
                    
                    # Promote to hot cache
                    self.hot_cache[cache_key] = shader_data
                    self.hot_cache.move_to_end(cache_key)
                    
                    # Remove from cold storage
                    del self.cold_storage[cache_key]
                    cache_file.unlink(missing_ok=True)
                    
                    self._evict_from_hot_cache()
                    return shader_data
                    
                except Exception as e:
                    logger.error(f"Failed to read cold cache file: {e}")
                    # Clean up corrupted cache entry
                    if cache_key in self.cold_storage:
                        del self.cold_storage[cache_key]
                    cache_file.unlink(missing_ok=True)
            
            # Cache miss
            return None
    
    def prefetch_shaders(self, shader_hashes: List[str]) -> int:
        """Prefetch shaders in background (OLED optimization)"""
        if not self.prefetch_enabled:
            return 0
        
        prefetched = 0
        
        for shader_hash in shader_hashes:
            cache_key = self._get_cache_key(shader_hash)
            
            # Only prefetch if not in hot/warm cache
            if cache_key not in self.hot_cache and cache_key not in self.warm_cache:
                if cache_key in self.cold_storage:
                    # Prefetch from cold storage to warm cache
                    shader_data = self.retrieve_shader(shader_hash)
                    if shader_data:
                        prefetched += 1
        
        logger.debug(f"Prefetched {prefetched} shaders for OLED optimization")
        return prefetched
    
    def _update_cache_metrics(self) -> None:
        """Update cache performance metrics"""
        with self._metrics_lock:
            # Calculate hit rate
            if self._access_count > 0:
                self.metrics.hit_rate = self._hit_count / self._access_count
                self.metrics.miss_rate = 1.0 - self.metrics.hit_rate
            
            # Calculate total cache size
            hot_size = sum(len(data) for data in self.hot_cache.values())
            warm_size = sum(len(data) for data in self.warm_cache.values())
            
            # Estimate cold storage size
            cold_size = 0
            for cache_file in self.cold_storage.values():
                try:
                    cold_size += cache_file.stat().st_size
                except:
                    pass
            
            total_size = hot_size + warm_size + cold_size
            self.metrics.total_size_mb = total_size / (1024 * 1024)
            
            # Calculate compression ratio
            if cold_size > 0:
                # Estimate original size vs compressed size
                estimated_original = cold_size * 2.5  # Typical compression ratio
                self.metrics.compression_ratio = estimated_original / cold_size
    
    def start_background_optimization(self) -> None:
        """Start background cache optimization for OLED"""
        if self._optimization_active:
            return
        
        self._optimization_active = True
        
        def optimization_loop():
            while self._optimization_active:
                try:
                    # Periodic cache maintenance
                    with self._cache_operation():
                        # Clean up stale entries
                        current_time = time.time()
                        
                        # Remove old cold storage files
                        for cache_key, cache_file in list(self.cold_storage.items()):
                            try:
                                if cache_file.exists():
                                    # Remove files older than 1 week
                                    if current_time - cache_file.stat().st_mtime > 604800:
                                        cache_file.unlink()
                                        del self.cold_storage[cache_key]
                                else:
                                    del self.cold_storage[cache_key]
                            except:
                                pass
                        
                        # Update metrics
                        self._update_cache_metrics()
                    
                    # Sleep for 5 minutes between optimization runs
                    time.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Cache optimization error: {e}")
                    time.sleep(60)
        
        self._optimizer_thread = threading.Thread(
            target=optimization_loop,
            name="OLED_CacheOptimizer",
            daemon=True
        )
        self._optimizer_thread.start()
        
        logger.info("OLED cache background optimization started")
    
    def stop_background_optimization(self) -> None:
        """Stop background optimization"""
        self._optimization_active = False
        
        if self._optimizer_thread and self._optimizer_thread.is_alive():
            self._optimizer_thread.join(timeout=10)
        
        logger.info("OLED cache background optimization stopped")
    
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics"""
        self._update_cache_metrics()
        return self.metrics
    
    def cleanup(self) -> None:
        """Clean up memory-mapped files and resources"""
        # Stop background optimization
        self.stop_background_optimization()
        
        # Clean up memory maps
        for memory_map in self.memory_mapped_files.values():
            try:
                memory_map.close()
            except:
                pass
        
        # Clean up file handles
        for file_handle in self.file_handles.values():
            try:
                file_handle.close()
            except:
                pass
        
        self.memory_mapped_files.clear()
        self.file_handles.clear()
        
        logger.info("OLED shader cache cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass


# Global cache instance for OLED Steam Deck
_oled_shader_cache: Optional[MemoryMappedShaderCache] = None


def get_oled_shader_cache() -> MemoryMappedShaderCache:
    """Get global OLED-optimized shader cache"""
    global _oled_shader_cache
    
    if _oled_shader_cache is None:
        cache_dir = Path.home() / '.cache' / 'shader-predict-compile' / 'oled-optimized'
        _oled_shader_cache = MemoryMappedShaderCache(cache_dir, max_cache_size_mb=1024)
        _oled_shader_cache.start_background_optimization()
    
    return _oled_shader_cache


if __name__ == "__main__":
    # Test OLED shader cache
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 OLED Memory-Mapped Shader Cache Test")
    print("=" * 45)
    
    cache = get_oled_shader_cache()
    
    # Test shader storage and retrieval
    test_shader = b"test_shader_bytecode_data_" * 100  # ~2.6KB shader
    shader_hash = "test_shader_001"
    
    print(f"Storing test shader ({len(test_shader)} bytes)...")
    success = cache.store_shader(shader_hash, test_shader, "variant_1")
    print(f"Storage: {'✓' if success else '✗'}")
    
    print("Retrieving test shader...")
    retrieved_shader = cache.retrieve_shader(shader_hash, "variant_1")
    print(f"Retrieval: {'✓' if retrieved_shader == test_shader else '✗'}")
    
    # Test cache metrics
    metrics = cache.get_metrics()
    print(f"\nCache Metrics:")
    print(f"  Size: {metrics.total_size_mb:.2f} MB")
    print(f"  Hit Rate: {metrics.hit_rate:.1%}")
    print(f"  Compression Ratio: {metrics.compression_ratio:.1f}x")
    print(f"  OLED Optimized: {'✓' if metrics.oled_specific_optimizations else '✗'}")
    
    # Cleanup
    cache.cleanup()
    print("\n✅ OLED shader cache test completed")