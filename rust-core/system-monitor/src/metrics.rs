//! System metrics collection and aggregation

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use crate::{CpuMetrics, GpuMetrics, ThermalMetrics, PowerMetrics};

/// Aggregated system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: Instant,
    pub cpu: CpuMetrics,
    pub gpu: GpuMetrics,
    pub thermal: ThermalMetrics,
    pub power: PowerMetrics,
}

/// Metrics collector for historical data
pub struct MetricsCollector {
    history: parking_lot::Mutex<Vec<SystemMetrics>>,
    max_history_size: usize,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            history: parking_lot::Mutex::new(Vec::new()),
            max_history_size: 3600, // Keep 1 hour of 1Hz data
        }
    }
    
    /// Update metrics
    pub async fn update_metrics(&self, metrics: SystemMetrics) {
        let mut history = self.history.lock();
        history.push(metrics);
        
        // Trim history to max size
        if history.len() > self.max_history_size {
            history.remove(0);
        }
    }
    
    /// Get metrics history for duration
    pub fn get_history(&self, duration: Duration) -> Vec<SystemMetrics> {
        let history = self.history.lock();
        let cutoff = Instant::now() - duration;
        
        history.iter()
            .filter(|metrics| metrics.timestamp > cutoff)
            .cloned()
            .collect()
    }
    
    /// Get latest metrics
    pub fn get_latest(&self) -> Option<SystemMetrics> {
        self.history.lock().last().cloned()
    }
}