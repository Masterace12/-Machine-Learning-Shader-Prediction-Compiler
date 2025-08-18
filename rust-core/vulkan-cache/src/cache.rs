//! Core shader cache implementation with multi-tier storage

use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use blake3::Hasher;
use crate::mmap_store::{MmapCacheStore, CacheTier};
use zstd;

/// High-performance shader cache with intelligent tier management
pub struct ShaderCache {
    // Memory-mapped storage backend
    store: Arc<MmapCacheStore>,
    
    // In-memory index for O(1) lookups
    index: Arc<DashMap<u64, CacheEntry>>,
    
    // Bloom filter for fast negative lookups
    bloom_filter: Arc<BloomFilter>,
    
    // Performance statistics
    stats: Arc<ShaderCacheStats>,
}

/// Cache entry metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub hash: u64,
    pub size: usize,
    pub tier: CacheTier,
    pub timestamp: u64,
    pub access_count: u32,
    pub compilation_time_ms: f32,
    pub game_id: Option<String>,
}

/// Cache performance statistics
pub struct ShaderCacheStats {
    pub total_entries: AtomicUsize,
    pub hot_tier_entries: AtomicUsize,
    pub warm_tier_entries: AtomicUsize,
    pub cold_tier_entries: AtomicUsize,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub total_size_bytes: AtomicUsize,
    pub avg_lookup_time_ns: AtomicU64,
}

/// Bloom filter for fast negative cache lookups
struct BloomFilter {
    bits: Vec<AtomicU64>,
    size: usize,
    hash_functions: usize,
}

impl ShaderCache {
    /// Create a new shader cache
    pub fn new() -> Result<Self> {
        let cache_dir = std::env::var("SHADER_CACHE_DIR")
            .unwrap_or_else(|_| "/home/deck/.cache/shader-predict-compile".to_string());
        
        let store = Arc::new(MmapCacheStore::new(&cache_dir)?);
        let bloom_filter = Arc::new(BloomFilter::new(1_000_000, 0.01)?);
        
        let stats = Arc::new(ShaderCacheStats {
            total_entries: AtomicUsize::new(0),
            hot_tier_entries: AtomicUsize::new(0),
            warm_tier_entries: AtomicUsize::new(0),
            cold_tier_entries: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_size_bytes: AtomicUsize::new(0),
            avg_lookup_time_ns: AtomicU64::new(0),
        });
        
        Ok(Self {
            store,
            index: Arc::new(DashMap::new()),
            bloom_filter,
            stats,
        })
    }
    
    /// Store shader bytecode in the cache
    pub fn store_shader(&self, shader_code: &[u8], metadata: ShaderMetadata) -> Result<u64> {
        let hash = self.compute_shader_hash(shader_code);
        
        // Check if already exists
        if self.bloom_filter.contains(hash) && self.index.contains_key(&hash) {
            return Ok(hash);
        }
        
        // Compress shader data
        let compressed_data = self.compress_shader_data(shader_code)?;
        
        // Store in memory-mapped backend
        self.store.store(hash, &compressed_data)?;
        
        // Create cache entry
        let entry = CacheEntry {
            hash,
            size: compressed_data.len(),
            tier: self.select_initial_tier(compressed_data.len()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            access_count: 1,
            compilation_time_ms: metadata.compilation_time_ms,
            game_id: metadata.game_id,
        };
        
        // Update index and bloom filter
        self.index.insert(hash, entry.clone());
        self.bloom_filter.insert(hash);
        
        // Update statistics
        self.stats.total_entries.fetch_add(1, Ordering::Relaxed);
        self.stats.total_size_bytes.fetch_add(compressed_data.len(), Ordering::Relaxed);
        
        match entry.tier {
            CacheTier::Hot => self.stats.hot_tier_entries.fetch_add(1, Ordering::Relaxed),
            CacheTier::Warm => self.stats.warm_tier_entries.fetch_add(1, Ordering::Relaxed),
            CacheTier::Cold => self.stats.cold_tier_entries.fetch_add(1, Ordering::Relaxed),
        };
        
        Ok(hash)
    }
    
    /// Retrieve shader bytecode from the cache
    pub fn get_shader(&self, hash: u64) -> Result<Option<Vec<u8>>> {
        let start_time = std::time::Instant::now();
        
        // Quick bloom filter check
        if !self.bloom_filter.contains(hash) {
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        
        // Check in-memory index
        if let Some(mut entry) = self.index.get_mut(&hash) {
            // Update access statistics
            entry.access_count += 1;
            entry.timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            // Retrieve from storage
            if let Some(compressed_data) = self.store.get(hash)? {
                let shader_data = self.decompress_shader_data(&compressed_data)?;
                
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                
                // Update average lookup time
                let lookup_time_ns = start_time.elapsed().as_nanos() as u64;
                self.update_avg_lookup_time(lookup_time_ns);
                
                return Ok(Some(shader_data));
            }
        }
        
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }
    
    /// Check if a shader exists in the cache
    pub fn contains(&self, hash: u64) -> bool {
        self.bloom_filter.contains(hash) && self.index.contains_key(&hash)
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> crate::CacheStats {
        let total_lookups = self.stats.cache_hits.load(Ordering::Relaxed) + 
                           self.stats.cache_misses.load(Ordering::Relaxed);
        
        let hit_rate = if total_lookups > 0 {
            self.stats.cache_hits.load(Ordering::Relaxed) as f64 / total_lookups as f64
        } else {
            0.0
        };
        
        crate::CacheStats {
            hit_rate,
            total_lookups,
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
            memory_usage_mb: self.stats.total_size_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
            hot_tier_size: self.stats.hot_tier_entries.load(Ordering::Relaxed),
            warm_tier_size: self.stats.warm_tier_entries.load(Ordering::Relaxed),
            cold_tier_size: self.stats.cold_tier_entries.load(Ordering::Relaxed),
        }
    }
    
    /// Compute hash for shader bytecode
    fn compute_shader_hash(&self, shader_code: &[u8]) -> u64 {
        let mut hasher = Hasher::new();
        hasher.update(shader_code);
        let hash = hasher.finalize();
        
        // Use first 8 bytes of Blake3 hash as u64
        u64::from_le_bytes([
            hash.as_bytes()[0], hash.as_bytes()[1], hash.as_bytes()[2], hash.as_bytes()[3],
            hash.as_bytes()[4], hash.as_bytes()[5], hash.as_bytes()[6], hash.as_bytes()[7],
        ])
    }
    
    /// Compress shader data using zstd
    fn compress_shader_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(zstd::encode_all(data, 3)?)
    }
    
    /// Decompress shader data
    fn decompress_shader_data(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        Ok(zstd::decode_all(compressed_data)?)
    }
    
    /// Select initial storage tier based on shader size
    fn select_initial_tier(&self, size: usize) -> CacheTier {
        if size < 32 * 1024 {      // < 32KB - small shaders
            CacheTier::Hot
        } else if size < 256 * 1024 {  // < 256KB - medium shaders  
            CacheTier::Warm
        } else {                   // >= 256KB - large shaders
            CacheTier::Cold
        }
    }
    
    /// Update running average of lookup times
    fn update_avg_lookup_time(&self, new_time_ns: u64) {
        // Simple exponential moving average
        let current_avg = self.stats.avg_lookup_time_ns.load(Ordering::Relaxed);
        let new_avg = if current_avg == 0 {
            new_time_ns
        } else {
            (current_avg * 9 + new_time_ns) / 10  // 90% old, 10% new
        };
        self.stats.avg_lookup_time_ns.store(new_avg, Ordering::Relaxed);
    }
}

/// Shader metadata for cache entries
pub struct ShaderMetadata {
    pub compilation_time_ms: f32,
    pub game_id: Option<String>,
}

impl BloomFilter {
    /// Create a new bloom filter
    fn new(expected_items: usize, false_positive_rate: f64) -> Result<Self> {
        let size = Self::optimal_size(expected_items, false_positive_rate);
        let hash_functions = Self::optimal_hash_functions(size, expected_items);
        
        let num_words = (size + 63) / 64;  // Round up to nearest u64
        let bits = vec![AtomicU64::new(0); num_words];
        
        Ok(Self {
            bits,
            size,
            hash_functions,
        })
    }
    
    /// Insert a hash into the bloom filter
    fn insert(&self, hash: u64) {
        for i in 0..self.hash_functions {
            let bit_index = self.hash_to_index(hash, i);
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            
            if word_index < self.bits.len() {
                self.bits[word_index].fetch_or(1u64 << bit_offset, Ordering::Relaxed);
            }
        }
    }
    
    /// Check if a hash might be in the bloom filter
    fn contains(&self, hash: u64) -> bool {
        for i in 0..self.hash_functions {
            let bit_index = self.hash_to_index(hash, i);
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            
            if word_index >= self.bits.len() {
                return false;
            }
            
            let word = self.bits[word_index].load(Ordering::Relaxed);
            if (word & (1u64 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }
    
    /// Convert hash to bit index using double hashing
    fn hash_to_index(&self, hash: u64, i: usize) -> usize {
        let hash1 = hash as usize;
        let hash2 = (hash >> 32) as usize;
        (hash1.wrapping_add(i.wrapping_mul(hash2))) % self.size
    }
    
    /// Calculate optimal bloom filter size
    fn optimal_size(n: usize, p: f64) -> usize {
        let ln2 = std::f64::consts::LN_2;
        (-(n as f64) * p.ln() / (ln2 * ln2)).ceil() as usize
    }
    
    /// Calculate optimal number of hash functions
    fn optimal_hash_functions(m: usize, n: usize) -> usize {
        let ln2 = std::f64::consts::LN_2;
        ((m as f64 / n as f64) * ln2).round() as usize
    }
}