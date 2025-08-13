//! Memory-mapped file storage for high-performance shader caching

use anyhow::{Result, Context};
use memmap2::{MmapMut, MmapOptions};
use parking_lot::{RwLock, Mutex};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Memory-mapped cache store with three-tier architecture
pub struct MmapCacheStore {
    hot_tier: Arc<MmapRegion>,
    warm_tier: Arc<MmapRegion>,  
    cold_tier: Arc<MmapRegion>,
    index: Arc<RwLock<HashMap<u64, CacheLocation>>>,
    stats: Arc<CacheStoreStats>,
    base_path: PathBuf,
}

/// Individual memory-mapped region
pub struct MmapRegion {
    mmap: Mutex<MmapMut>,
    allocator: Mutex<RegionAllocator>,
    size: usize,
    name: String,
}

/// Allocation manager for a memory-mapped region
struct RegionAllocator {
    free_blocks: Vec<(usize, usize)>, // (offset, size) pairs
    allocated_blocks: HashMap<usize, usize>, // offset -> size
    next_offset: usize,
    total_size: usize,
}

/// Location of cached data within the storage system
#[derive(Debug, Clone, Copy)]
pub struct CacheLocation {
    pub tier: CacheTier,
    pub offset: usize,
    pub size: usize,
    pub timestamp: u64,
    pub access_count: u32,
}

/// Cache tier enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    Hot,   // 256MB - Fastest access, most frequently used
    Warm,  // 1GB - Medium access speed, recently used  
    Cold,  // 4GB - Slower access, archival storage
}

/// Cache store performance statistics
pub struct CacheStoreStats {
    pub lookups: AtomicU64,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub hot_tier_usage: AtomicUsize,
    pub warm_tier_usage: AtomicUsize,
    pub cold_tier_usage: AtomicUsize,
    pub promotions: AtomicU64,
    pub evictions: AtomicU64,
}

impl MmapCacheStore {
    /// Create a new memory-mapped cache store
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;
        
        // Create three tiers with different sizes
        let hot_tier = Arc::new(MmapRegion::new(
            &base_path.join("hot_cache.mmap"),
            256 * 1024 * 1024, // 256MB
            "hot".to_string(),
        )?);
        
        let warm_tier = Arc::new(MmapRegion::new(
            &base_path.join("warm_cache.mmap"),
            1024 * 1024 * 1024, // 1GB  
            "warm".to_string(),
        )?);
        
        let cold_tier = Arc::new(MmapRegion::new(
            &base_path.join("cold_cache.mmap"),
            4 * 1024 * 1024 * 1024, // 4GB
            "cold".to_string(),
        )?);
        
        let stats = Arc::new(CacheStoreStats {
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            hot_tier_usage: AtomicUsize::new(0),
            warm_tier_usage: AtomicUsize::new(0),
            cold_tier_usage: AtomicUsize::new(0),
            promotions: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        });
        
        Ok(Self {
            hot_tier,
            warm_tier, 
            cold_tier,
            index: Arc::new(RwLock::new(HashMap::new())),
            stats,
            base_path,
        })
    }
    
    /// Store data in the cache with intelligent tier placement
    pub fn store(&self, key: u64, data: &[u8]) -> Result<()> {
        let tier = self.select_tier(data.len());
        let region = self.get_region(tier);
        
        // Allocate space in the selected tier
        let offset = region.allocate(data.len())?;
        
        // Write data to memory-mapped region
        region.write_at(offset, data)?;
        
        // Update index
        let location = CacheLocation {
            tier,
            offset,
            size: data.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            access_count: 1,
        };
        
        self.index.write().insert(key, location);
        
        // Update statistics
        match tier {
            CacheTier::Hot => self.stats.hot_tier_usage.fetch_add(data.len(), Ordering::Relaxed),
            CacheTier::Warm => self.stats.warm_tier_usage.fetch_add(data.len(), Ordering::Relaxed),
            CacheTier::Cold => self.stats.cold_tier_usage.fetch_add(data.len(), Ordering::Relaxed),
        };
        
        Ok(())
    }
    
    /// Retrieve data from the cache
    pub fn get(&self, key: u64) -> Result<Option<Vec<u8>>> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        
        let mut index = self.index.write();
        if let Some(mut location) = index.get_mut(&key) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            
            // Update access statistics
            location.access_count += 1;
            location.timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            let region = self.get_region(location.tier);
            let data = region.read_at(location.offset, location.size)?;
            
            // Consider promoting frequently accessed items
            if location.access_count > 10 && location.tier != CacheTier::Hot {
                if let Ok(()) = self.promote_entry(key, &data) {
                    self.stats.promotions.fetch_add(1, Ordering::Relaxed);
                }
            }
            
            Ok(Some(data))
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStoreStats {
        CacheStoreStats {
            lookups: AtomicU64::new(self.stats.lookups.load(Ordering::Relaxed)),
            hits: AtomicU64::new(self.stats.hits.load(Ordering::Relaxed)),
            misses: AtomicU64::new(self.stats.misses.load(Ordering::Relaxed)),
            hot_tier_usage: AtomicUsize::new(self.stats.hot_tier_usage.load(Ordering::Relaxed)),
            warm_tier_usage: AtomicUsize::new(self.stats.warm_tier_usage.load(Ordering::Relaxed)),
            cold_tier_usage: AtomicUsize::new(self.stats.cold_tier_usage.load(Ordering::Relaxed)),
            promotions: AtomicU64::new(self.stats.promotions.load(Ordering::Relaxed)),
            evictions: AtomicU64::new(self.stats.evictions.load(Ordering::Relaxed)),
        }
    }
    
    /// Select appropriate tier based on data size and access patterns
    fn select_tier(&self, size: usize) -> CacheTier {
        // Small, frequently accessed shaders go to hot tier
        if size < 64 * 1024 { // < 64KB
            CacheTier::Hot
        } else if size < 512 * 1024 { // < 512KB
            CacheTier::Warm
        } else {
            CacheTier::Cold
        }
    }
    
    /// Get the memory-mapped region for a given tier
    fn get_region(&self, tier: CacheTier) -> &Arc<MmapRegion> {
        match tier {
            CacheTier::Hot => &self.hot_tier,
            CacheTier::Warm => &self.warm_tier,
            CacheTier::Cold => &self.cold_tier,
        }
    }
    
    /// Promote frequently accessed entry to a higher tier
    fn promote_entry(&self, key: u64, data: &[u8]) -> Result<()> {
        let current_location = self.index.read().get(&key).copied();
        
        if let Some(location) = current_location {
            let new_tier = match location.tier {
                CacheTier::Cold => CacheTier::Warm,
                CacheTier::Warm => CacheTier::Hot,
                CacheTier::Hot => return Ok(()), // Already at highest tier
            };
            
            // Try to allocate in higher tier
            let new_region = self.get_region(new_tier);
            if let Ok(new_offset) = new_region.allocate(data.len()) {
                // Write data to new location
                new_region.write_at(new_offset, data)?;
                
                // Update index
                let new_location = CacheLocation {
                    tier: new_tier,
                    offset: new_offset,
                    size: location.size,
                    timestamp: location.timestamp,
                    access_count: location.access_count,
                };
                
                self.index.write().insert(key, new_location);
                
                // Free old location
                let old_region = self.get_region(location.tier);
                old_region.deallocate(location.offset, location.size)?;
            }
        }
        
        Ok(())
    }
}

impl MmapRegion {
    /// Create a new memory-mapped region
    fn new<P: AsRef<Path>>(path: P, size: usize, name: String) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
            .context("Failed to open cache file")?;
        
        file.set_len(size as u64)
            .context("Failed to set file size")?;
        
        let mmap = unsafe {
            MmapOptions::new()
                .len(size)
                .map_mut(&file)
                .context("Failed to create memory mapping")?
        };
        
        let allocator = RegionAllocator::new(size);
        
        Ok(Self {
            mmap: Mutex::new(mmap),
            allocator: Mutex::new(allocator),
            size,
            name,
        })
    }
    
    /// Allocate space within the region
    fn allocate(&self, size: usize) -> Result<usize> {
        self.allocator.lock().allocate(size)
    }
    
    /// Deallocate space within the region
    fn deallocate(&self, offset: usize, size: usize) -> Result<()> {
        self.allocator.lock().deallocate(offset, size);
        Ok(())
    }
    
    /// Write data at the specified offset
    fn write_at(&self, offset: usize, data: &[u8]) -> Result<()> {
        let mut mmap = self.mmap.lock();
        if offset + data.len() > self.size {
            anyhow::bail!("Write would exceed region bounds");
        }
        
        mmap[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }
    
    /// Read data from the specified offset
    fn read_at(&self, offset: usize, size: usize) -> Result<Vec<u8>> {
        let mmap = self.mmap.lock();
        if offset + size > self.size {
            anyhow::bail!("Read would exceed region bounds");
        }
        
        Ok(mmap[offset..offset + size].to_vec())
    }
}

impl RegionAllocator {
    /// Create a new region allocator
    fn new(total_size: usize) -> Self {
        Self {
            free_blocks: vec![(0, total_size)],
            allocated_blocks: HashMap::new(),
            next_offset: 0,
            total_size,
        }
    }
    
    /// Allocate a block of the specified size
    fn allocate(&mut self, size: usize) -> Result<usize> {
        // Align size to 64-byte boundaries for better performance
        let aligned_size = (size + 63) & !63;
        
        // Find first fit in free blocks
        for i in 0..self.free_blocks.len() {
            let (offset, block_size) = self.free_blocks[i];
            if block_size >= aligned_size {
                // Use this block
                self.allocated_blocks.insert(offset, aligned_size);
                
                // Update free blocks
                if block_size == aligned_size {
                    // Exact fit - remove the block
                    self.free_blocks.remove(i);
                } else {
                    // Partial fit - update the block
                    self.free_blocks[i] = (offset + aligned_size, block_size - aligned_size);
                }
                
                return Ok(offset);
            }
        }
        
        anyhow::bail!("Out of memory in region allocator");
    }
    
    /// Deallocate a block
    fn deallocate(&mut self, offset: usize, size: usize) {
        let aligned_size = (size + 63) & !63;
        
        if self.allocated_blocks.remove(&offset).is_some() {
            // Add to free blocks and try to coalesce
            self.free_blocks.push((offset, aligned_size));
            self.coalesce_free_blocks();
        }
    }
    
    /// Coalesce adjacent free blocks
    fn coalesce_free_blocks(&mut self) {
        self.free_blocks.sort_by_key(|(offset, _)| *offset);
        
        let mut i = 0;
        while i < self.free_blocks.len() - 1 {
            let (offset1, size1) = self.free_blocks[i];
            let (offset2, size2) = self.free_blocks[i + 1];
            
            if offset1 + size1 == offset2 {
                // Adjacent blocks - coalesce them
                self.free_blocks[i] = (offset1, size1 + size2);
                self.free_blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
}