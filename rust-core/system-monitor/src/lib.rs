//! System monitoring for Steam Deck hardware and performance metrics
//! 
//! This module provides comprehensive system monitoring including CPU, GPU,
//! thermal, and power metrics specifically optimized for Steam Deck.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub mod cpu_monitor;
pub mod gpu_monitor; 
pub mod thermal_monitor;
pub mod power_monitor;
pub mod metrics;

// Re-export main types
pub use cpu_monitor::{CpuMonitor, CpuMetrics};
pub use gpu_monitor::{GpuMonitor, GpuMetrics};
pub use thermal_monitor::{ThermalMonitor, ThermalMetrics};
pub use power_monitor::{PowerMonitor, PowerMetrics};
pub use metrics::{SystemMetrics, MetricsCollector};

/// Main system monitor coordinating all monitoring subsystems
pub struct SystemMonitor {
    cpu_monitor: Arc<CpuMonitor>,
    gpu_monitor: Arc<GpuMonitor>,
    thermal_monitor: Arc<ThermalMonitor>,
    power_monitor: Arc<PowerMonitor>,
    metrics_collector: Arc<MetricsCollector>,
    monitoring_active: Arc<parking_lot::AtomicBool>,
}

impl SystemMonitor {
    /// Create a new system monitor
    pub fn new() -> Result<Self> {
        let cpu_monitor = Arc::new(CpuMonitor::new()?);
        let gpu_monitor = Arc::new(GpuMonitor::new()?);
        let thermal_monitor = Arc::new(ThermalMonitor::new()?);
        let power_monitor = Arc::new(PowerMonitor::new()?);
        let metrics_collector = Arc::new(MetricsCollector::new());
        
        Ok(Self {
            cpu_monitor,
            gpu_monitor,
            thermal_monitor,
            power_monitor,
            metrics_collector,
            monitoring_active: Arc::new(parking_lot::AtomicBool::new(false)),
        })
    }
    
    /// Start monitoring all subsystems
    pub async fn start_monitoring(&self) -> Result<()> {
        self.monitoring_active.store(true, std::sync::atomic::Ordering::Relaxed);
        
        let mut interval = tokio::time::interval(Duration::from_millis(1000)); // 1Hz
        
        while self.monitoring_active.load(std::sync::atomic::Ordering::Relaxed) {
            interval.tick().await;
            
            // Collect metrics from all subsystems
            let cpu_metrics = self.cpu_monitor.get_metrics().await?;
            let gpu_metrics = self.gpu_monitor.get_metrics().await?;
            let thermal_metrics = self.thermal_monitor.get_metrics().await?;
            let power_metrics = self.power_monitor.get_metrics().await?;
            
            // Aggregate into system metrics
            let system_metrics = SystemMetrics {
                timestamp: Instant::now(),
                cpu: cpu_metrics,
                gpu: gpu_metrics,
                thermal: thermal_metrics,
                power: power_metrics,
            };
            
            // Update metrics collector
            self.metrics_collector.update_metrics(system_metrics).await;
        }
        
        Ok(())
    }
    
    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Get current system metrics
    pub async fn get_current_metrics(&self) -> Result<SystemMetrics> {
        let cpu_metrics = self.cpu_monitor.get_metrics().await?;
        let gpu_metrics = self.gpu_monitor.get_metrics().await?;
        let thermal_metrics = self.thermal_monitor.get_metrics().await?;
        let power_metrics = self.power_monitor.get_metrics().await?;
        
        Ok(SystemMetrics {
            timestamp: Instant::now(),
            cpu: cpu_metrics,
            gpu: gpu_metrics,
            thermal: thermal_metrics,
            power: power_metrics,
        })
    }
    
    /// Get metrics history
    pub fn get_metrics_history(&self, duration: Duration) -> Vec<SystemMetrics> {
        self.metrics_collector.get_history(duration)
    }
    
    /// Check if system is under load
    pub async fn is_system_under_load(&self) -> Result<bool> {
        let metrics = self.get_current_metrics().await?;
        
        // Consider system under load if:
        // - CPU usage > 80%
        // - GPU usage > 70%
        // - Temperature > 80°C
        
        Ok(metrics.cpu.usage_percent > 80.0 ||
           metrics.gpu.usage_percent > 70.0 ||
           metrics.thermal.cpu_temp > 80.0)
    }
    
    /// Get optimization recommendations
    pub async fn get_optimization_recommendations(&self) -> Result<OptimizationRecommendations> {
        let metrics = self.get_current_metrics().await?;
        let mut recommendations = OptimizationRecommendations::default();
        
        // CPU recommendations
        if metrics.cpu.usage_percent > 90.0 {
            recommendations.reduce_cpu_threads = true;
            recommendations.cpu_intensity_multiplier = 0.5;
        } else if metrics.cpu.usage_percent > 70.0 {
            recommendations.cpu_intensity_multiplier = 0.7;
        }
        
        // GPU recommendations
        if metrics.gpu.usage_percent > 80.0 {
            recommendations.defer_gpu_operations = true;
        }
        
        // Thermal recommendations
        if metrics.thermal.cpu_temp > 85.0 {
            recommendations.enable_thermal_throttling = true;
            recommendations.thermal_intensity_multiplier = 0.3;
        } else if metrics.thermal.cpu_temp > 75.0 {
            recommendations.thermal_intensity_multiplier = 0.7;
        }
        
        // Power recommendations
        if metrics.power.on_battery && metrics.power.battery_percent < 20.0 {
            recommendations.enable_power_saving = true;
            recommendations.power_intensity_multiplier = 0.4;
        }
        
        Ok(recommendations)
    }
}

/// System optimization recommendations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationRecommendations {
    pub reduce_cpu_threads: bool,
    pub defer_gpu_operations: bool,
    pub enable_thermal_throttling: bool,
    pub enable_power_saving: bool,
    pub cpu_intensity_multiplier: f32,
    pub thermal_intensity_multiplier: f32,
    pub power_intensity_multiplier: f32,
}

/// Error types for system monitoring
#[derive(thiserror::Error, Debug)]
pub enum MonitoringError {
    #[error("CPU monitoring error: {0}")]
    CpuError(String),
    
    #[error("GPU monitoring error: {0}")]
    GpuError(String),
    
    #[error("Thermal monitoring error: {0}")]
    ThermalError(String),
    
    #[error("Power monitoring error: {0}")]
    PowerError(String),
    
    #[error("Metrics collection error: {0}")]
    MetricsError(String),
}

impl Default for OptimizationRecommendations {
    fn default() -> Self {
        Self {
            reduce_cpu_threads: false,
            defer_gpu_operations: false,
            enable_thermal_throttling: false,
            enable_power_saving: false,
            cpu_intensity_multiplier: 1.0,
            thermal_intensity_multiplier: 1.0,
            power_intensity_multiplier: 1.0,
        }
    }
}