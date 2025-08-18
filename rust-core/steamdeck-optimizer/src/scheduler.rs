//! Smart scheduler for Steam Deck shader compilation

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use crate::thermal::{ThermalManager, ThermalState};
use crate::power::{PowerManager, PowerState};
use crate::monitor::{SystemMonitor, SystemMetrics};

/// Steam Deck optimized compilation scheduler
pub struct SteamDeckScheduler {
    thermal_manager: Arc<ThermalManager>,
    power_manager: Arc<PowerManager>,
    monitor: Arc<SystemMonitor>,
    compilation_semaphore: Arc<Semaphore>,
    task_queue: mpsc::UnboundedSender<CompilationTask>,
}

/// Compilation task with priority and metadata
#[derive(Debug, Clone)]
pub struct CompilationTask {
    pub id: String,
    pub priority: TaskPriority,
    pub estimated_duration_ms: u32,
    pub power_requirement: f32,
    pub thermal_impact: f32,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Critical = 0,   // Game startup shaders
    High = 1,       // Immediate gameplay needs
    Normal = 2,     // Background precompilation
    Low = 3,        // Cache warming
}

impl SteamDeckScheduler {
    /// Create a new scheduler
    pub fn new(
        thermal_manager: Arc<ThermalManager>,
        power_manager: Arc<PowerManager>,
        monitor: Arc<SystemMonitor>,
    ) -> Result<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let compilation_semaphore = Arc::new(Semaphore::new(2)); // Start with 2 concurrent tasks
        
        // Spawn task processing loop
        let thermal_manager_clone = thermal_manager.clone();
        let power_manager_clone = power_manager.clone();
        let monitor_clone = monitor.clone();
        let semaphore_clone = compilation_semaphore.clone();
        
        tokio::spawn(async move {
            while let Some(task) = rx.recv().await {
                let _permit = semaphore_clone.acquire().await.unwrap();
                
                // Check if we should process this task now
                if Self::should_process_task(&task, &thermal_manager_clone, &power_manager_clone, &monitor_clone).await {
                    // Process the task
                    Self::process_task(task).await;
                } else {
                    // Requeue the task for later
                    if let Err(_) = tx.send(task) {
                        break; // Channel closed
                    }
                    
                    // Wait before retrying
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            }
        });
        
        Ok(Self {
            thermal_manager,
            power_manager,
            monitor,
            compilation_semaphore,
            task_queue: tx,
        })
    }
    
    /// Schedule a compilation task
    pub async fn schedule_task(&self, task: CompilationTask) -> Result<()> {
        self.task_queue.send(task)?;
        Ok(())
    }
    
    /// Get recommended thread count for current conditions
    pub fn get_recommended_thread_count(&self) -> usize {
        let thermal_state = self.thermal_manager.get_thermal_state();
        let power_state = self.power_manager.get_power_state();
        
        let base_threads = match power_state {
            PowerState::BatterySave => 1,
            PowerState::Balanced => 2,
            PowerState::Performance => 4,
        };
        
        // Reduce based on thermal state
        let thermal_reduction = match thermal_state {
            ThermalState::Cool => 1.0,
            ThermalState::Normal => 1.0,
            ThermalState::Warm => 0.75,
            ThermalState::Hot => 0.5,
            ThermalState::Critical => 0.25,
        };
        
        ((base_threads as f32 * thermal_reduction).round() as usize).max(1)
    }
    
    /// Get compilation intensity factor (0.0 to 1.0)
    pub fn get_compilation_intensity(&self) -> f32 {
        let thermal_state = self.thermal_manager.get_thermal_state();
        let power_state = self.power_manager.get_power_state();
        
        let power_factor = match power_state {
            PowerState::BatterySave => 0.3,
            PowerState::Balanced => 0.7,
            PowerState::Performance => 1.0,
        };
        
        let thermal_factor = match thermal_state {
            ThermalState::Cool => 1.0,
            ThermalState::Normal => 0.9,
            ThermalState::Warm => 0.6,
            ThermalState::Hot => 0.3,
            ThermalState::Critical => 0.1,
        };
        
        power_factor * thermal_factor
    }
    
    /// Check if a task should be processed now
    async fn should_process_task(
        task: &CompilationTask,
        thermal_manager: &ThermalManager,
        power_manager: &PowerManager,
        monitor: &SystemMonitor,
    ) -> bool {
        let thermal_state = thermal_manager.get_thermal_state();
        let power_state = power_manager.get_power_state();
        let metrics = monitor.get_current_metrics();
        
        // Always process critical tasks
        if task.priority == TaskPriority::Critical {
            return true;
        }
        
        // Don't process if system is overloaded
        if metrics.cpu_usage > 80.0 || metrics.memory_usage > 90.0 {
            return false;
        }
        
        // Check thermal constraints
        if thermal_state == ThermalState::Critical {
            return false;
        }
        
        if thermal_state == ThermalState::Hot && task.priority != TaskPriority::High {
            return false;
        }
        
        // Check power constraints
        if power_state == PowerState::BatterySave && task.priority == TaskPriority::Low {
            return false;
        }
        
        true
    }
    
    /// Process a compilation task
    async fn process_task(task: CompilationTask) {
        tracing::debug!("Processing compilation task: {} (priority: {:?})", task.id, task.priority);
        
        // Simulate task processing
        let duration = std::time::Duration::from_millis(task.estimated_duration_ms as u64);
        tokio::time::sleep(duration).await;
        
        tracing::debug!("Completed compilation task: {}", task.id);
    }
}