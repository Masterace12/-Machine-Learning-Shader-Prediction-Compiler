//! Steam Deck specific optimizations for shader compilation
//! 
//! This module provides Van Gogh APU-specific optimizations, thermal management,
//! and power-aware scheduling for optimal shader compilation on Steam Deck.

use anyhow::Result;
use std::sync::Arc;

pub mod thermal;
pub mod power;
pub mod scheduler;
pub mod monitor;
pub mod hardware;

// Re-export main types
pub use thermal::{ThermalManager, ThermalState, ThermalProfile};
pub use power::{PowerManager, PowerProfile, PowerState};
pub use scheduler::{SteamDeckScheduler, CompilationTask, TaskPriority};
pub use monitor::{SystemMonitor, SystemMetrics};
pub use hardware::{HardwareDetector, SteamDeckModel, ApuType};

/// Main Steam Deck optimizer coordinating all optimization strategies
pub struct SteamDeckOptimizer {
    thermal_manager: Arc<ThermalManager>,
    power_manager: Arc<PowerManager>,
    scheduler: Arc<SteamDeckScheduler>,
    monitor: Arc<SystemMonitor>,
    hardware: Arc<HardwareDetector>,
}

impl SteamDeckOptimizer {
    /// Create a new Steam Deck optimizer
    pub fn new() -> Result<Self> {
        let hardware = Arc::new(HardwareDetector::new()?);
        let thermal_manager = Arc::new(ThermalManager::new(hardware.clone())?);
        let power_manager = Arc::new(PowerManager::new(hardware.clone())?);
        let monitor = Arc::new(SystemMonitor::new()?);
        let scheduler = Arc::new(SteamDeckScheduler::new(
            thermal_manager.clone(),
            power_manager.clone(),
            monitor.clone(),
        )?);
        
        Ok(Self {
            thermal_manager,
            power_manager,
            scheduler,
            monitor,
            hardware,
        })
    }
    
    /// Get current optimization recommendations
    pub fn get_optimization_profile(&self) -> OptimizationProfile {
        let thermal_state = self.thermal_manager.get_thermal_state();
        let power_state = self.power_manager.get_power_state();
        let system_metrics = self.monitor.get_current_metrics();
        
        OptimizationProfile {
            thermal_state,
            power_state,
            system_metrics,
            recommended_threads: self.scheduler.get_recommended_thread_count(),
            compilation_intensity: self.scheduler.get_compilation_intensity(),
            should_throttle: thermal_state == ThermalState::Hot || power_state == PowerState::BatterySave,
        }
    }
    
    /// Schedule shader compilation task
    pub async fn schedule_compilation(&self, task: CompilationTask) -> Result<()> {
        self.scheduler.schedule_task(task).await
    }
    
    /// Check if system is suitable for intensive compilation
    pub fn can_do_intensive_compilation(&self) -> bool {
        let profile = self.get_optimization_profile();
        !profile.should_throttle && profile.compilation_intensity > 0.5
    }
}

/// Current optimization profile for the system
#[derive(Debug, Clone)]
pub struct OptimizationProfile {
    pub thermal_state: ThermalState,
    pub power_state: PowerState,
    pub system_metrics: SystemMetrics,
    pub recommended_threads: usize,
    pub compilation_intensity: f32,
    pub should_throttle: bool,
}

/// Error types for Steam Deck optimization
#[derive(thiserror::Error, Debug)]
pub enum OptimizerError {
    #[error("Hardware detection error: {0}")]
    HardwareError(String),
    
    #[error("Thermal management error: {0}")]
    ThermalError(String),
    
    #[error("Power management error: {0}")]
    PowerError(String),
    
    #[error("Scheduling error: {0}")]
    SchedulingError(String),
    
    #[error("System monitoring error: {0}")]
    MonitoringError(String),
}