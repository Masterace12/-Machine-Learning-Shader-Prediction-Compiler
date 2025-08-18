//! CPU monitoring for Steam Deck Van Gogh APU

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// CPU monitor for Steam Deck
pub struct CpuMonitor {
    core_count: usize,
}

/// CPU metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub usage_percent: f32,
    pub frequency_mhz: u32,
    pub per_core_usage: Vec<f32>,
    pub load_average_1m: f32,
    pub load_average_5m: f32,
    pub load_average_15m: f32,
    pub context_switches: u64,
    pub interrupts: u64,
}

impl CpuMonitor {
    /// Create new CPU monitor
    pub fn new() -> Result<Self> {
        let core_count = num_cpus::get();
        Ok(Self { core_count })
    }
    
    /// Get current CPU metrics
    pub async fn get_metrics(&self) -> Result<CpuMetrics> {
        tokio::task::spawn_blocking(|| {
            let mut system = sysinfo::System::new();
            system.refresh_cpu();
            
            // Get overall CPU usage
            let usage_percent = system.global_cpu_info().cpu_usage();
            
            // Get per-core usage
            let per_core_usage: Vec<f32> = system.cpus()
                .iter()
                .map(|cpu| cpu.cpu_usage())
                .collect();
            
            // Get frequency (approximate)
            let frequency_mhz = system.cpus()
                .first()
                .map(|cpu| cpu.frequency() as u32)
                .unwrap_or(2800); // Default for Steam Deck
            
            // Get load averages
            let load_avg = system.load_average();
            
            Ok(CpuMetrics {
                usage_percent,
                frequency_mhz,
                per_core_usage,
                load_average_1m: load_avg.one as f32,
                load_average_5m: load_avg.five as f32,
                load_average_15m: load_avg.fifteen as f32,
                context_switches: 0, // Would need /proc/stat parsing
                interrupts: 0,       // Would need /proc/interrupts parsing
            })
        }).await?
    }
}