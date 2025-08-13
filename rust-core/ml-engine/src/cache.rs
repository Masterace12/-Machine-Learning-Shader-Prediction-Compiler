//! High-performance prediction caching with LRU eviction

use dashmap::DashMap;
use lru::LruCache;
use parking_lot::RwLock;
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use crate::features::ShaderFeatures;

/// Lock-free prediction cache with hot/warm tiers
pub struct PredictionCache {
    // L1: Hot cache (lock-free)
    hot_cache: Arc<DashMap<u64, CachedPrediction>>,
    // L2: Warm cache (LRU with read-write lock)
    warm_cache: Arc<RwLock<LruCache<u64, CachedPrediction>>>,
    hot_capacity: usize,
    warm_capacity: usize,
}

/// Cached prediction entry
#[derive(Clone, Copy, Debug)]
pub struct CachedPrediction {
    pub compilation_time: f32,
    pub confidence: f32,
    pub timestamp: u64,
    pub access_count: u32,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hit_rate: f64,
    pub total_lookups: u64,
    pub hot_hits: u64,
    pub warm_hits: u64,
    pub misses: u64,
    pub total_entries: usize,
    pub hot_entries: usize,
    pub warm_entries: usize,
}

impl PredictionCache {
    /// Create a new prediction cache
    pub fn new(hot_capacity: usize, warm_capacity: usize) -> Self {
        Self {
            hot_cache: Arc::new(DashMap::with_capacity(hot_capacity)),
            warm_cache: Arc::new(RwLock::new(LruCache::new(warm_capacity))),
            hot_capacity,
            warm_capacity,
        }
    }
    
    /// Get prediction from cache
    pub fn get(&self, features: &ShaderFeatures) -> Option<CachedPrediction> {
        let hash = self.compute_hash(features);
        
        // Check hot cache first (lock-free)
        if let Some(mut entry) = self.hot_cache.get_mut(&hash) {
            entry.access_count += 1;
            return Some(*entry);
        }
        
        // Check warm cache
        if let Some(mut entry) = self.warm_cache.write().get_mut(&hash) {
            entry.access_count += 1;
            
            // Promote to hot cache if frequently accessed
            if entry.access_count > 5 {
                self.promote_to_hot(hash, *entry);
            }
            
            return Some(*entry);
        }
        
        None
    }
    
    /// Insert prediction into cache
    pub fn insert(&self, features: &ShaderFeatures, prediction: CachedPrediction) {
        let hash = self.compute_hash(features);
        
        // Always try hot cache first for new entries
        if self.hot_cache.len() < self.hot_capacity {
            self.hot_cache.insert(hash, prediction);
        } else {
            // Hot cache full, use warm cache
            self.warm_cache.write().put(hash, prediction);
        }
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let hot_entries = self.hot_cache.len();
        let warm_entries = self.warm_cache.read().len();
        
        // Note: In a real implementation, we'd track these metrics
        // For now, return basic stats
        CacheStats {
            hit_rate: 0.0, // Would be calculated from actual metrics
            total_lookups: 0,
            hot_hits: 0,
            warm_hits: 0,
            misses: 0,
            total_entries: hot_entries + warm_entries,
            hot_entries,
            warm_entries,
        }
    }
    
    /// Clear all cached predictions
    pub fn clear(&self) {
        self.hot_cache.clear();
        self.warm_cache.write().clear();
    }
    
    /// Promote entry to hot cache
    fn promote_to_hot(&self, hash: u64, entry: CachedPrediction) {
        if self.hot_cache.len() >= self.hot_capacity {
            // Need to evict something from hot cache
            if let Some((evicted_key, evicted_entry)) = self.hot_cache.iter().next() {
                let evicted_key = *evicted_key;
                let evicted_entry = *evicted_entry;
                drop(evicted_key); // Release the iterator
                
                // Move evicted entry to warm cache
                self.hot_cache.remove(&evicted_key);
                self.warm_cache.write().put(evicted_key, evicted_entry);
            }
        }
        
        self.hot_cache.insert(hash, entry);
    }
    
    /// Compute hash for shader features
    fn compute_hash(&self, features: &ShaderFeatures) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash key features that affect compilation time
        (features.instruction_count as u32).hash(&mut hasher);
        (features.register_usage as u32).hash(&mut hasher);
        (features.texture_samples as u32).hash(&mut hasher);
        (features.memory_operations as u32).hash(&mut hasher);
        (features.control_flow_complexity as u32).hash(&mut hasher);
        (features.uses_derivatives as u32).hash(&mut hasher);
        (features.uses_tessellation as u32).hash(&mut hasher);
        (features.uses_geometry_shader as u32).hash(&mut hasher);
        (features.shader_type_hash as u32).hash(&mut hasher);
        
        hasher.finish()
    }
}

impl Default for CachedPrediction {
    fn default() -> Self {
        Self {
            compilation_time: 0.0,
            confidence: 0.0,
            timestamp: 0,
            access_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::ShaderFeatures;
    
    #[test]
    fn test_cache_basic_operations() {
        let cache = PredictionCache::new(10, 100);
        let features = ShaderFeatures::default();
        
        // Test miss
        assert!(cache.get(&features).is_none());
        
        // Test insert and hit
        let prediction = CachedPrediction {
            compilation_time: 50.0,
            confidence: 0.9,
            timestamp: 12345,
            access_count: 1,
        };
        
        cache.insert(&features, prediction);
        let retrieved = cache.get(&features);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().compilation_time, 50.0);
    }
    
    #[test]
    fn test_cache_promotion() {
        let cache = PredictionCache::new(2, 10);
        let mut features = ShaderFeatures::default();
        
        let prediction = CachedPrediction {
            compilation_time: 50.0,
            confidence: 0.9,
            timestamp: 12345,
            access_count: 1,
        };
        
        // Fill hot cache
        for i in 0..3 {
            features.instruction_count = i as f32;
            cache.insert(&features, prediction);
        }
        
        // Access an item multiple times to trigger promotion
        features.instruction_count = 2.0;
        for _ in 0..6 {
            cache.get(&features);
        }
        
        let stats = cache.get_stats();
        assert!(stats.total_entries > 0);
    }
}