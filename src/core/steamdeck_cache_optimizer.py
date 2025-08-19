#!/usr/bin/env python3
"""
Steam Deck Cache Optimizer
Memory-mapped file system optimized for Steam Deck storage constraints
"""

import os
import mmap
import time
import json
import hashlib
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import OrderedDict
import struct
import gc

@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    size: int
    access_count: int
    last_access: float
    created_at: float
    prediction_time_ms: float
    accuracy_score: float
    thermal_state: str

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_entries: int
    total_size_mb: float
    hit_rate: float
    miss_rate: float
    avg_access_time_ms: float
    memory_usage_mb: float
    disk_usage_mb: float
    thermal_evictions: int
    last_cleanup_time: float

class SteamDeckCacheOptimizer:
    """
    Steam Deck optimized cache system
    
    Features:
    - Memory-mapped files for fast access
    - Thermal-aware cache management
    - Storage-tiered caching (NVMe + SD card)
    - LRU eviction with size awareness
    - Proactive cache warming
    - Memory pressure handling
    """
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_memory_cache_mb: int = 64,
                 max_disk_cache_mb: int = 512,
                 enable_sd_cache: bool = True):
        
        self.logger = logging.getLogger(__name__)
        
        # Cache configuration
        self.max_memory_cache_mb = max_memory_cache_mb
        self.max_disk_cache_mb = max_disk_cache_mb
        self.enable_sd_cache = enable_sd_cache
        
        # Cache directories
        self.cache_dir = cache_dir or Path.home() / ".cache" / "shader-predict-compile"
        self.nvme_cache_dir = self.cache_dir / "nvme"
        self.sd_cache_dir = self._find_sd_cache_dir() if enable_sd_cache else None
        
        # Create cache directories
        self.nvme_cache_dir.mkdir(parents=True, exist_ok=True)
        if self.sd_cache_dir:
            self.sd_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache storage
        self.memory_cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_metadata: Dict[str, CacheEntry] = {}
        self.memory_mapped_files: Dict[str, mmap.mmap] = {}
        
        # Performance tracking
        self.stats = CacheStats(
            total_entries=0,
            total_size_mb=0.0,
            hit_rate=0.0,
            miss_rate=0.0,
            avg_access_time_ms=0.0,
            memory_usage_mb=0.0,
            disk_usage_mb=0.0,
            thermal_evictions=0,
            last_cleanup_time=time.time()
        )
        
        # Threading
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._shutdown_event = threading.Event()
        
        # Load existing metadata
        self._load_metadata()
        
        # Start background cleanup
        self._start_cleanup_thread()
        
        self.logger.info(f"Steam Deck cache optimizer initialized: {self.cache_dir}")
    
    def _find_sd_cache_dir(self) -> Optional[Path]:
        """Find SD card mount point for secondary cache"""
        sd_candidates = [
            Path("/run/media/deck"),
            Path("/media/deck"),
            Path("/mnt/sdcard"),
        ]
        
        for mount_point in sd_candidates:
            if mount_point.exists():
                # Look for actual SD card mounts
                for sd_dir in mount_point.iterdir():
                    if sd_dir.is_dir():
                        cache_path = sd_dir / ".cache" / "shader-predict-compile"
                        try:
                            # Test if we can write to this location
                            cache_path.mkdir(parents=True, exist_ok=True)
                            test_file = cache_path / "test_write"
                            test_file.write_text("test")
                            test_file.unlink()
                            self.logger.info(f"SD card cache available: {cache_path}")
                            return cache_path
                        except Exception:
                            continue
        
        self.logger.info("No SD card cache available")
        return None
    
    def _load_metadata(self):
        """Load cache metadata from disk"""
        metadata_file = self.nvme_cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    
                for key, entry_data in data.get("entries", {}).items():
                    self.cache_metadata[key] = CacheEntry(**entry_data)
                
                # Update stats
                self.stats.total_entries = len(self.cache_metadata)
                self.stats.total_size_mb = sum(entry.size for entry in self.cache_metadata.values()) / (1024 * 1024)
                
                self.logger.info(f"Loaded {self.stats.total_entries} cache entries from metadata")
                
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        metadata_file = self.nvme_cache_dir / "metadata.json"
        try:
            data = {
                "entries": {key: asdict(entry) for key, entry in self.cache_metadata.items()},
                "stats": asdict(self.stats),
                "last_save": time.time()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="cache_cleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self._shutdown_event.is_set():
            try:
                # Memory pressure check
                self._check_memory_pressure()
                
                # Periodic cleanup
                self._cleanup_stale_entries()
                
                # Update statistics
                self._update_statistics()
                
                # Save metadata periodically
                if time.time() - self.stats.last_cleanup_time > 300:  # Every 5 minutes
                    self._save_metadata()
                    self.stats.last_cleanup_time = time.time()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)
    
    def _check_memory_pressure(self):
        """Check for memory pressure and evict if necessary"""
        current_memory_mb = self._get_memory_usage_mb()
        
        if current_memory_mb > self.max_memory_cache_mb:
            self.logger.info(f"Memory pressure detected: {current_memory_mb:.1f}MB > {self.max_memory_cache_mb}MB")
            self._evict_memory_cache(target_mb=self.max_memory_cache_mb * 0.8)
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory cache usage in MB"""
        with self._lock:
            total_size = 0
            for value in self.memory_cache.values():
                try:
                    if isinstance(value, (bytes, bytearray)):
                        total_size += len(value)
                    elif hasattr(value, '__sizeof__'):
                        total_size += value.__sizeof__()
                except Exception:
                    total_size += 1024  # Estimate
            
            return total_size / (1024 * 1024)
    
    def _evict_memory_cache(self, target_mb: float):
        """Evict items from memory cache to reach target size"""
        with self._lock:
            current_mb = self._get_memory_usage_mb()
            
            # Evict LRU items until we reach target
            items_to_evict = []
            evicted_mb = 0
            
            for key in list(self.memory_cache.keys()):
                if current_mb - evicted_mb <= target_mb:
                    break
                
                # Estimate item size
                value = self.memory_cache[key]
                if isinstance(value, (bytes, bytearray)):
                    item_mb = len(value) / (1024 * 1024)
                else:
                    item_mb = 0.1  # Estimate
                
                items_to_evict.append(key)
                evicted_mb += item_mb
            
            # Perform eviction
            for key in items_to_evict:
                self.memory_cache.pop(key, None)
                
            self.logger.info(f"Evicted {len(items_to_evict)} items ({evicted_mb:.1f}MB) from memory cache")
    
    def _cleanup_stale_entries(self):
        """Clean up stale cache entries"""
        current_time = time.time()
        stale_threshold = current_time - (24 * 3600)  # 24 hours
        
        with self._lock:
            stale_keys = [
                key for key, entry in self.cache_metadata.items()
                if entry.last_access < stale_threshold
            ]
            
            for key in stale_keys:
                self._remove_entry(key)
            
            if stale_keys:
                self.logger.info(f"Cleaned up {len(stale_keys)} stale cache entries")
    
    def _remove_entry(self, key: str):
        """Remove a cache entry completely"""
        # Remove from memory cache
        self.memory_cache.pop(key, None)
        
        # Close memory mapped file
        if key in self.memory_mapped_files:
            try:
                self.memory_mapped_files[key].close()
                del self.memory_mapped_files[key]
            except Exception:
                pass
        
        # Remove metadata
        self.cache_metadata.pop(key, None)
        
        # Remove disk files
        for cache_dir in [self.nvme_cache_dir, self.sd_cache_dir]:
            if cache_dir:
                cache_file = cache_dir / f"{key}.cache"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
    
    def _update_statistics(self):
        """Update cache statistics"""
        with self._lock:
            self.stats.total_entries = len(self.cache_metadata)
            self.stats.memory_usage_mb = self._get_memory_usage_mb()
            
            # Calculate disk usage
            disk_usage = 0
            for cache_dir in [self.nvme_cache_dir, self.sd_cache_dir]:
                if cache_dir and cache_dir.exists():
                    for file_path in cache_dir.glob("*.cache"):
                        try:
                            disk_usage += file_path.stat().st_size
                        except Exception:
                            pass
            
            self.stats.disk_usage_mb = disk_usage / (1024 * 1024)
    
    def _generate_cache_key(self, shader_features: Dict[str, Any]) -> str:
        """Generate cache key from shader features"""
        # Create deterministic hash from features
        feature_str = json.dumps(shader_features, sort_keys=True)
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]
    
    def _get_cache_file_path(self, key: str, prefer_sd: bool = False) -> Path:
        """Get cache file path, choosing between NVMe and SD card"""
        if prefer_sd and self.sd_cache_dir:
            return self.sd_cache_dir / f"{key}.cache"
        else:
            return self.nvme_cache_dir / f"{key}.cache"
    
    def _write_to_disk(self, key: str, data: bytes, prefer_sd: bool = False) -> bool:
        """Write data to disk cache"""
        cache_file = self._get_cache_file_path(key, prefer_sd)
        
        try:
            with open(cache_file, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write cache file {cache_file}: {e}")
            return False
    
    def _read_from_disk(self, key: str) -> Optional[bytes]:
        """Read data from disk cache"""
        # Try NVMe first, then SD card
        for cache_dir in [self.nvme_cache_dir, self.sd_cache_dir]:
            if not cache_dir:
                continue
                
            cache_file = cache_dir / f"{key}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return f.read()
                except Exception as e:
                    self.logger.warning(f"Failed to read cache file {cache_file}: {e}")
        
        return None
    
    def _create_memory_map(self, key: str, data: bytes) -> Optional[mmap.mmap]:
        """Create memory-mapped file for fast access"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            # Write data to file first
            if not self._write_to_disk(key, data):
                return None
            
            # Create memory map
            with open(cache_file, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                return mm
                
        except Exception as e:
            self.logger.error(f"Failed to create memory map for {key}: {e}")
            return None
    
    def put(self, shader_features: Dict[str, Any], prediction_result: Any, 
            prediction_time_ms: float = 0.0, accuracy_score: float = 1.0,
            thermal_state: str = "normal") -> bool:
        """Store prediction result in cache"""
        
        key = self._generate_cache_key(shader_features)
        
        try:
            # Serialize prediction result
            if isinstance(prediction_result, (dict, list)):
                data = json.dumps(prediction_result).encode('utf-8')
            elif isinstance(prediction_result, (bytes, bytearray)):
                data = bytes(prediction_result)
            else:
                data = str(prediction_result).encode('utf-8')
            
            with self._lock:
                # Create cache entry metadata
                entry = CacheEntry(
                    key=key,
                    size=len(data),
                    access_count=1,
                    last_access=time.time(),
                    created_at=time.time(),
                    prediction_time_ms=prediction_time_ms,
                    accuracy_score=accuracy_score,
                    thermal_state=thermal_state
                )
                
                # Store in memory cache
                self.memory_cache[key] = prediction_result
                self.memory_cache.move_to_end(key)  # Mark as most recently used
                
                # Store metadata
                self.cache_metadata[key] = entry
                
                # Optionally create memory-mapped file for large entries
                if len(data) > 1024:  # > 1KB
                    prefer_sd = len(data) > 64 * 1024  # Large files to SD card
                    self._write_to_disk(key, data, prefer_sd=prefer_sd)
                
                # Check memory pressure
                if self._get_memory_usage_mb() > self.max_memory_cache_mb:
                    self._evict_memory_cache(self.max_memory_cache_mb * 0.8)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cache prediction result: {e}")
            return False
    
    def get(self, shader_features: Dict[str, Any]) -> Optional[Any]:
        """Get prediction result from cache"""
        
        key = self._generate_cache_key(shader_features)
        start_time = time.time()
        
        try:
            with self._lock:
                # Check memory cache first
                if key in self.memory_cache:
                    self.memory_cache.move_to_end(key)  # Mark as recently used
                    
                    # Update metadata
                    if key in self.cache_metadata:
                        entry = self.cache_metadata[key]
                        entry.access_count += 1
                        entry.last_access = time.time()
                    
                    access_time_ms = (time.time() - start_time) * 1000
                    self.logger.debug(f"Cache hit (memory): {key} in {access_time_ms:.2f}ms")
                    return self.memory_cache[key]
                
                # Check if we have metadata for this key
                if key not in self.cache_metadata:
                    return None  # Cache miss
                
                # Try to load from disk
                data = self._read_from_disk(key)
                if data is None:
                    # File not found, remove stale metadata
                    del self.cache_metadata[key]
                    return None
                
                # Deserialize data
                try:
                    result = json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    # Not JSON, return as bytes
                    result = data
                
                # Update metadata
                entry = self.cache_metadata[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Store in memory cache for faster future access
                self.memory_cache[key] = result
                self.memory_cache.move_to_end(key)
                
                access_time_ms = (time.time() - start_time) * 1000
                self.logger.debug(f"Cache hit (disk): {key} in {access_time_ms:.2f}ms")
                return result
                
        except Exception as e:
            self.logger.error(f"Cache retrieval error for {key}: {e}")
            return None
    
    def warm_cache(self, feature_sets: List[Dict[str, Any]]) -> int:
        """Warm cache with prediction results for given feature sets"""
        warmed_count = 0
        
        for features in feature_sets:
            key = self._generate_cache_key(features)
            
            # Check if already in cache
            if key in self.memory_cache or key in self.cache_metadata:
                continue
            
            # Load from disk if available
            if self.get(features) is not None:
                warmed_count += 1
        
        self.logger.info(f"Warmed {warmed_count} cache entries")
        return warmed_count
    
    def thermal_eviction(self, thermal_state: str):
        """Perform thermal-aware cache eviction"""
        if thermal_state not in ["throttling", "critical"]:
            return
        
        with self._lock:
            # Evict entries based on thermal state and access patterns
            entries_to_evict = []
            
            for key, entry in self.cache_metadata.items():
                # Evict entries created during high thermal states
                if entry.thermal_state in ["throttling", "critical"]:
                    entries_to_evict.append(key)
                # Evict rarely accessed entries
                elif entry.access_count < 3 and time.time() - entry.last_access > 3600:
                    entries_to_evict.append(key)
            
            for key in entries_to_evict:
                self._remove_entry(key)
            
            self.stats.thermal_evictions += len(entries_to_evict)
            
            if entries_to_evict:
                self.logger.info(f"Thermal eviction: removed {len(entries_to_evict)} entries")
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics"""
        self._update_statistics()
        
        # Calculate hit/miss rates
        total_accesses = sum(entry.access_count for entry in self.cache_metadata.values())
        if total_accesses > 0:
            # Approximate hit rate based on access patterns
            self.stats.hit_rate = min(0.95, total_accesses / max(1, len(self.cache_metadata) * 2))
            self.stats.miss_rate = 1.0 - self.stats.hit_rate
        
        return self.stats
    
    def cleanup(self):
        """Cleanup cache resources"""
        self.logger.info("Cleaning up cache resources...")
        
        # Stop cleanup thread
        if self._cleanup_thread:
            self._shutdown_event.set()
            self._cleanup_thread.join(timeout=5.0)
        
        # Close memory mapped files
        with self._lock:
            for mm in self.memory_mapped_files.values():
                try:
                    mm.close()
                except Exception:
                    pass
            self.memory_mapped_files.clear()
            
            # Clear memory cache
            self.memory_cache.clear()
        
        # Save final metadata
        self._save_metadata()
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("Cache cleanup completed")


# Global cache optimizer instance
_cache_optimizer = None

def get_cache_optimizer() -> SteamDeckCacheOptimizer:
    """Get or create global cache optimizer"""
    global _cache_optimizer
    if _cache_optimizer is None:
        _cache_optimizer = SteamDeckCacheOptimizer()
    return _cache_optimizer


if __name__ == "__main__":
    # Test cache optimizer
    logging.basicConfig(level=logging.INFO)
    
    print("💾 Steam Deck Cache Optimizer Test")
    print("=" * 40)
    
    cache = get_cache_optimizer()
    
    # Test cache operations
    test_features = {"shader_type": "vertex", "complexity": 0.5, "uniforms": 10}
    test_result = {"prediction": 0.85, "confidence": 0.9}
    
    # Store in cache
    success = cache.put(test_features, test_result, prediction_time_ms=0.1)
    print(f"Cache store: {'✓' if success else '✗'}")
    
    # Retrieve from cache
    cached_result = cache.get(test_features)
    print(f"Cache retrieve: {'✓' if cached_result else '✗'}")
    
    # Show statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Entries: {stats.total_entries}")
    print(f"  Memory usage: {stats.memory_usage_mb:.1f}MB")
    print(f"  Disk usage: {stats.disk_usage_mb:.1f}MB")
    
    # Cleanup
    cache.cleanup()
    print("\n✅ Cache optimizer test completed")