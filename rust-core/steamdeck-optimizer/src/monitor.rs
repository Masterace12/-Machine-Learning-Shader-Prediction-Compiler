//! System monitoring for Steam Deck resource tracking

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// System monitor for Steam Deck metrics
pub struct SystemMonitor {
    metrics: Arc<SystemMetricsInternal>,
    last_update: Arc<parking_lot::Mutex<Instant>>,
}

/// Current system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub memory_available_mb: u64,
    pub gpu_usage: f32,
    pub active_game_processes: u32,
    pub compilation_load: f32,
    pub timestamp: u64,
}

/// Internal metrics with atomic counters
struct SystemMetricsInternal {
    cpu_usage: AtomicU64, // f32 as u64 bits
    memory_usage: AtomicU64,
    memory_available_mb: AtomicU64,
    gpu_usage: AtomicU64,
    active_game_processes: AtomicU64,
    compilation_load: AtomicU64,
}

impl SystemMonitor {
    /// Create a new system monitor
    pub fn new() -> Result<Self> {
        Ok(Self {
            metrics: Arc::new(SystemMetricsInternal::new()),
            last_update: Arc::new(parking_lot::Mutex::new(Instant::now())),
        })
    }
    
    /// Get current system metrics
    pub fn get_current_metrics(&self) -> SystemMetrics {
        // Update metrics if they're stale
        {
            let mut last_update = self.last_update.lock();
            if last_update.elapsed() > Duration::from_secs(1) {
                if let Err(e) = self.update_metrics() {
                    tracing::warn!("Failed to update system metrics: {}", e);
                }
                *last_update = Instant::now();
            }
        }
        
        // Convert atomic values back to metrics
        SystemMetrics {
            cpu_usage: f32::from_bits(self.metrics.cpu_usage.load(Ordering::Relaxed) as u32),
            memory_usage: f32::from_bits(self.metrics.memory_usage.load(Ordering::Relaxed) as u32),
            memory_available_mb: self.metrics.memory_available_mb.load(Ordering::Relaxed),
            gpu_usage: f32::from_bits(self.metrics.gpu_usage.load(Ordering::Relaxed) as u32),
            active_game_processes: self.metrics.active_game_processes.load(Ordering::Relaxed) as u32,
            compilation_load: f32::from_bits(self.metrics.compilation_load.load(Ordering::Relaxed) as u32),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
    
    /// Update system metrics from hardware
    fn update_metrics(&self) -> Result<()> {
        // Use sysinfo to get system metrics
        let mut system = sysinfo::System::new();
        system.refresh_all();
        
        // CPU usage
        let cpu_usage = system.global_cpu_info().cpu_usage();
        self.metrics.cpu_usage.store(cpu_usage.to_bits() as u64, Ordering::Relaxed);
        
        // Memory usage
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let memory_usage = if total_memory > 0 {
            (used_memory as f32 / total_memory as f32) * 100.0
        } else {
            0.0
        };
        self.metrics.memory_usage.store(memory_usage.to_bits() as u64, Ordering::Relaxed);
        self.metrics.memory_available_mb.store((system.available_memory() / 1024 / 1024), Ordering::Relaxed);
        
        // GPU usage (try to read from Steam Deck specific paths)
        let gpu_usage = self.read_gpu_usage().unwrap_or(0.0);
        self.metrics.gpu_usage.store(gpu_usage.to_bits() as u64, Ordering::Relaxed);
        
        // Count game processes
        let game_processes = self.count_game_processes(&system);
        self.metrics.active_game_processes.store(game_processes as u64, Ordering::Relaxed);
        
        // Estimate compilation load
        let compilation_load = self.estimate_compilation_load(&system);
        self.metrics.compilation_load.store(compilation_load.to_bits() as u64, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Read GPU usage from Steam Deck sensors
    fn read_gpu_usage(&self) -> Option<f32> {
        // Try Steam Deck specific GPU usage paths
        let gpu_paths = [
            "/sys/class/drm/card0/device/gpu_busy_percent",
            "/sys/class/drm/card1/device/gpu_busy_percent",
        ];
        
        for path in &gpu_paths {
            if let Ok(usage_str) = std::fs::read_to_string(path) {
                if let Ok(usage) = usage_str.trim().parse::<f32>() {
                    return Some(usage);
                }
            }
        }
        
        None
    }
    
    /// Count active game processes
    fn count_game_processes(&self, system: &sysinfo::System) -> u32 {
        let mut count = 0;
        
        for (_, process) in system.processes() {
            let name = process.name().to_lowercase();
            
            // Common game process patterns
            if name.contains("game") || 
               name.contains("steam") ||
               name.contains("proton") ||
               name.contains("wine") ||
               name.ends_with(".exe") {
                count += 1;
            }
        }
        
        count
    }
    
    /// Estimate current compilation load
    fn estimate_compilation_load(&self, system: &sysinfo::System) -> f32 {
        let mut compilation_processes = 0;
        
        for (_, process) in system.processes() {
            let name = process.name().to_lowercase();
            
            // Look for compilation-related processes
            if name.contains("rustc") ||
               name.contains("gcc") ||
               name.contains("clang") ||
               name.contains("shader") ||
               name.contains("compile") {
                compilation_processes += 1;
            }
        }
        
        // Normalize to 0.0-1.0 range
        (compilation_processes as f32 / 10.0).min(1.0)
    }
    
    /// Check if system is suitable for background compilation
    pub fn is_suitable_for_background_work(&self) -> bool {
        let metrics = self.get_current_metrics();
        
        metrics.cpu_usage < 60.0 &&
        metrics.memory_usage < 80.0 &&
        metrics.gpu_usage < 50.0 &&
        metrics.active_game_processes == 0
    }
    
    /// Get performance recommendation
    pub fn get_performance_recommendation(&self) -> PerformanceRecommendation {
        let metrics = self.get_current_metrics();
        
        if metrics.cpu_usage > 80.0 || metrics.memory_usage > 90.0 {
            PerformanceRecommendation::Throttle
        } else if metrics.active_game_processes > 0 {
            PerformanceRecommendation::Conservative
        } else if metrics.cpu_usage < 30.0 && metrics.memory_usage < 50.0 {
            PerformanceRecommendation::Aggressive
        } else {
            PerformanceRecommendation::Normal
        }
    }
}

impl SystemMetricsInternal {
    fn new() -> Self {
        Self {
            cpu_usage: AtomicU64::new(0),
            memory_usage: AtomicU64::new(0),
            memory_available_mb: AtomicU64::new(0),
            gpu_usage: AtomicU64::new(0),
            active_game_processes: AtomicU64::new(0),
            compilation_load: AtomicU64::new(0),
        }
    }
}

/// Performance recommendation based on system state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceRecommendation {
    Throttle,     // System overloaded, reduce compilation
    Conservative, // Game running, be careful with resources
    Normal,       // Normal compilation intensity
    Aggressive,   // System idle, can do heavy compilation
}