//! High-performance Vulkan shader cache with memory-mapped storage
//! 
//! This module provides a production-grade shader caching system optimized for Steam Deck,
//! featuring memory-mapped files, SIMD operations, and multi-tier caching strategies.

use anyhow::Result;
use std::sync::Arc;

pub mod cache;
pub mod mmap_store;

// Re-export main types
pub use cache::{ShaderCache, CacheEntry, CacheTier};
pub use mmap_store::{MmapCacheStore, MmapRegion};

/// Main shader cache manager that coordinates all caching strategies
pub struct VulkanShaderManager {
    cache: Arc<ShaderCache>,
    precompiler: Arc<PrecompilationEngine>,
    interceptor: Option<VulkanInterceptor>,
}

impl VulkanShaderManager {
    /// Create a new shader manager with default configuration
    pub fn new() -> Result<Self> {
        let cache = Arc::new(ShaderCache::new()?);
        let precompiler = Arc::new(PrecompilationEngine::new(cache.clone())?);
        
        Ok(Self {
            cache,
            precompiler,
            interceptor: None,
        })
    }
    
    /// Initialize Vulkan layer interception
    pub fn enable_interception(&mut self) -> Result<()> {
        self.interceptor = Some(VulkanInterceptor::new(
            self.cache.clone(),
            self.precompiler.clone()
        )?);
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.cache.get_stats()
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hit_rate: f64,
    pub total_lookups: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_usage_mb: f64,
    pub hot_tier_size: usize,
    pub warm_tier_size: usize,
    pub cold_tier_size: usize,
}

/// Error types for the vulkan cache system
#[derive(thiserror::Error, Debug)]
pub enum VulkanCacheError {
    #[error("Vulkan API error: {0}")]
    VulkanError(#[from] ash::vk::Result),
    
    #[error("Memory mapping error: {0}")]
    MmapError(String),
    
    #[error("SPIR-V parsing error: {0}")]
    SpirvError(String),
    
    #[error("Cache corruption detected: {0}")]
    CorruptionError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}