#!/usr/bin/env python3
"""
High-performance shader cache implementation using modern Python packages
Replaces Rust components with optimized Python alternatives
"""

import os
import sys
import time
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

# High-performance caching
try:
    import diskcache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False

# Fast numerical operations
try:
    import numexpr
    HAS_NUMEXPR = True
except ImportError:
    HAS_NUMEXPR = False

try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False

# Compression
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

# Serialization
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

import pickle
import json

logger = logging.getLogger(__name__)


class HighPerformanceCache:
    """
    High-performance shader cache using diskcache and optimized Python packages
    Provides equivalent functionality to Rust components with pure Python
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size_gb: float = 2.0):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "shader-predict-compile" / "hp-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize high-performance disk cache
        if HAS_DISKCACHE:
            self.cache = diskcache.Cache(
                str(self.cache_dir), 
                size_limit=int(max_size_gb * 1024**3),  # Convert GB to bytes
                eviction_policy='least-recently-used',
                statistics=True,
            )
        else:
            self.cache = {}  # Fallback to memory dict
            
        # Memory cache for hot shaders
        self.hot_cache: Dict[str, Any] = {}
        self.hot_cache_max_size = 100
        
        # Performance tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "hot_hits": 0,
            "compression_ratio": 0.0,
            "avg_access_time_ms": 0.0,
        }
        
        # Thread pool for background operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hpcache")
        
        logger.info(f"High-performance cache initialized at {self.cache_dir}")
        logger.info(f"Available optimizations: diskcache={HAS_DISKCACHE}, numexpr={HAS_NUMEXPR}, "
                   f"bottleneck={HAS_BOTTLENECK}, zstd={HAS_ZSTD}, msgpack={HAS_MSGPACK}")
    
    def _compute_hash(self, data: Union[str, bytes]) -> str:
        """Compute fast hash for cache keys"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.blake2b(data, digest_size=16).hexdigest()
    
    def _serialize(self, data: Any) -> bytes:
        """High-performance serialization with fallbacks"""
        if HAS_MSGPACK:
            try:
                return msgpack.packb(data, use_bin_type=True)
            except Exception:
                pass
        
        # Fallback to pickle
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize(self, data: bytes) -> Any:
        """High-performance deserialization with fallbacks"""
        if HAS_MSGPACK:
            try:
                return msgpack.unpackb(data, raw=False)
            except Exception:
                pass
        
        # Fallback to pickle
        return pickle.loads(data)
    
    def _compress(self, data: bytes) -> bytes:
        """High-performance compression with fallbacks"""
        if HAS_ZSTD:
            try:
                compressor = zstd.ZstdCompressor(level=3)  # Fast compression
                return compressor.compress(data)
            except Exception:
                pass
        
        if HAS_LZ4:
            try:
                return lz4.compress(data, compression_level=1)  # Fastest compression
            except Exception:
                pass
        
        # No compression fallback
        return data
    
    def _decompress(self, data: bytes) -> bytes:
        """High-performance decompression with fallbacks"""
        if HAS_ZSTD:
            try:
                decompressor = zstd.ZstdDecompressor()
                return decompressor.decompress(data)
            except Exception:
                pass
        
        if HAS_LZ4:
            try:
                return lz4.decompress(data)
            except Exception:
                pass
        
        # No compression fallback
        return data
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with performance optimizations"""
        start_time = time.perf_counter()
        
        # Check hot cache first
        if key in self.hot_cache:
            self.stats["hot_hits"] += 1
            self.stats["hits"] += 1
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._update_avg_time(elapsed_ms)
            return self.hot_cache[key]
        
        # Check disk cache
        try:
            if HAS_DISKCACHE:
                compressed_data = self.cache.get(key)
            else:
                compressed_data = self.cache.get(key)
            
            if compressed_data is not None:
                # Decompress and deserialize
                data = self._decompress(compressed_data)
                result = self._deserialize(data)
                
                # Add to hot cache if it fits
                if len(self.hot_cache) < self.hot_cache_max_size:
                    self.hot_cache[key] = result
                
                self.stats["hits"] += 1
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._update_avg_time(elapsed_ms)
                return result
            else:
                self.stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, hot: bool = False) -> bool:
        """Set item in cache with performance optimizations"""
        try:
            # Serialize and compress
            serialized = self._serialize(value)
            compressed = self._compress(serialized)
            
            # Calculate compression ratio
            if len(serialized) > 0:
                ratio = len(compressed) / len(serialized)
                self.stats["compression_ratio"] = (self.stats["compression_ratio"] + ratio) / 2
            
            # Store in disk cache
            if HAS_DISKCACHE:
                self.cache[key] = compressed
            else:
                self.cache[key] = compressed
            
            # Add to hot cache if requested or if hot cache has space
            if hot or len(self.hot_cache) < self.hot_cache_max_size:
                self.hot_cache[key] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        try:
            # Remove from hot cache
            self.hot_cache.pop(key, None)
            
            # Remove from disk cache
            if HAS_DISKCACHE:
                del self.cache[key]
            else:
                self.cache.pop(key, None)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    def _update_avg_time(self, elapsed_ms: float):
        """Update average access time"""
        current_avg = self.stats["avg_access_time_ms"]
        if current_avg == 0:
            self.stats["avg_access_time_ms"] = elapsed_ms
        else:
            # Exponential moving average
            self.stats["avg_access_time_ms"] = current_avg * 0.9 + elapsed_ms * 0.1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.stats.copy()
        
        # Add diskcache statistics if available
        if HAS_DISKCACHE and hasattr(self.cache, 'stats'):
            try:
                cache_stats = self.cache.stats()
                if isinstance(cache_stats, dict):
                    stats.update({
                        "disk_hits": cache_stats.get('hits', 0),
                        "disk_misses": cache_stats.get('misses', 0),
                    })
                elif hasattr(cache_stats, '_asdict'):  # namedtuple
                    cache_dict = cache_stats._asdict()
                    stats.update({
                        "disk_hits": cache_dict.get('hits', 0),
                        "disk_misses": cache_dict.get('misses', 0),
                    })
                
                stats.update({
                    "disk_size_bytes": self.cache.volume(),
                    "disk_count": len(self.cache),
                })
            except Exception as e:
                logger.warning(f"Failed to get diskcache stats: {e}")
                stats.update({
                    "disk_hits": 0,
                    "disk_misses": 0,
                    "disk_size_bytes": 0,
                    "disk_count": 0,
                })
        
        stats.update({
            "hot_cache_size": len(self.hot_cache),
            "hit_rate": self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"]),
            "memory_usage_mb": sys.getsizeof(self.hot_cache) / (1024 * 1024),
        })
        
        return stats
    
    def clear_hot_cache(self):
        """Clear the hot cache"""
        self.hot_cache.clear()
        logger.info("Hot cache cleared")
    
    def close(self):
        """Close the cache and cleanup resources"""
        if HAS_DISKCACHE and hasattr(self.cache, 'close'):
            self.cache.close()
        
        self.executor.shutdown(wait=True)
        logger.info("High-performance cache closed")


class FastMLPredictor:
    """
    High-performance ML predictor using numexpr and bottleneck optimizations
    Replaces Rust ML components with optimized Python alternatives
    """
    
    def __init__(self):
        self.model = None
        self.prediction_cache = {}
        self.stats = {
            "predictions_made": 0,
            "cache_hits": 0,
            "avg_prediction_time_ms": 0.0,
        }
        
        logger.info(f"Fast ML predictor initialized with optimizations: "
                   f"numexpr={HAS_NUMEXPR}, bottleneck={HAS_BOTTLENECK}")
    
    def predict_shader_compile_time(self, shader_features: Dict[str, float]) -> float:
        """Predict shader compilation time using optimized operations"""
        start_time = time.perf_counter()
        
        # Create cache key
        cache_key = self._hash_features(shader_features)
        
        # Check cache first
        if cache_key in self.prediction_cache:
            self.stats["cache_hits"] += 1
            return self.prediction_cache[cache_key]
        
        # Fast feature processing using optimizations
        result = self._fast_predict(shader_features)
        
        # Cache result
        self.prediction_cache[cache_key] = result
        
        # Update stats
        self.stats["predictions_made"] += 1
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_avg_time(elapsed_ms)
        
        return result
    
    def _hash_features(self, features: Dict[str, float]) -> str:
        """Create cache key from features"""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()
    
    def _fast_predict(self, features: Dict[str, float]) -> float:
        """Fast prediction using optimized numerical operations"""
        try:
            # Convert features to values
            values = list(features.values())
            
            if HAS_BOTTLENECK:
                # Use bottleneck for fast operations
                mean_val = bn.nanmean(values)
                std_val = bn.nanstd(values)
                max_val = bn.nanmax(values)
            else:
                # Fallback to standard operations
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
                max_val = max(values)
            
            # Simple heuristic-based prediction (replace with real model)
            if HAS_NUMEXPR:
                # Use numexpr for fast mathematical operations
                complexity = numexpr.evaluate("mean_val * 0.7 + std_val * 0.2 + max_val * 0.1", 
                                            local_dict={'mean_val': mean_val, 'std_val': std_val, 'max_val': max_val})
            else:
                complexity = mean_val * 0.7 + std_val * 0.2 + max_val * 0.1
            
            # Convert complexity to time estimate (in milliseconds)
            return max(10.0, complexity * 100.0)
            
        except Exception as e:
            logger.warning(f"Fast prediction error: {e}")
            return 100.0  # Default estimate
    
    def _update_avg_time(self, elapsed_ms: float):
        """Update average prediction time"""
        current_avg = self.stats["avg_prediction_time_ms"]
        if current_avg == 0:
            self.stats["avg_prediction_time_ms"] = elapsed_ms
        else:
            # Exponential moving average
            self.stats["avg_prediction_time_ms"] = current_avg * 0.9 + elapsed_ms * 0.1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get ML predictor performance statistics"""
        cache_hit_rate = 0.0
        if self.stats["predictions_made"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / self.stats["predictions_made"]
        
        return {
            "ml_backend": "optimized_python",
            "predictions_made": self.stats["predictions_made"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": cache_hit_rate,
            "avg_prediction_time_ms": self.stats["avg_prediction_time_ms"],
            "prediction_cache_size": len(self.prediction_cache),
            "memory_usage_mb": sys.getsizeof(self.prediction_cache) / (1024 * 1024),
            "optimizations_available": {
                "numexpr": HAS_NUMEXPR,
                "bottleneck": HAS_BOTTLENECK,
            }
        }
    
    def cleanup(self):
        """Cleanup ML predictor resources"""
        self.prediction_cache.clear()
        logger.info("Fast ML predictor cleaned up")


# Factory functions for integration
def get_high_performance_cache(cache_dir: Optional[Path] = None) -> HighPerformanceCache:
    """Get high-performance cache instance"""
    return HighPerformanceCache(cache_dir)

def get_fast_ml_predictor() -> FastMLPredictor:
    """Get fast ML predictor instance"""
    return FastMLPredictor()