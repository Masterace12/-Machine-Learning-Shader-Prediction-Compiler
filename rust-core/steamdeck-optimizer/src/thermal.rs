//! Advanced thermal management for Steam Deck Van Gogh APU

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use crate::hardware::{HardwareDetector, ApuType};

/// Thermal manager with predictive capabilities
pub struct ThermalManager {
    current_state: Arc<AtomicU8>,
    hardware: Arc<HardwareDetector>,
    thermal_profile: ThermalProfile,
    temperature_history: parking_lot::Mutex<Vec<(Instant, f32)>>,
    prediction_window: Duration,
}

/// Thermal state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    Cool = 0,     // < 65°C - Full performance
    Normal = 1,   // 65-75°C - Normal operation
    Warm = 2,     // 75-85°C - Slight throttling
    Hot = 3,      // 85-95°C - Significant throttling
    Critical = 4, // > 95°C - Emergency shutdown
}

/// Thermal profile with APU-specific settings
#[derive(Debug, Clone)]
pub struct ThermalProfile {
    pub cool_threshold: f32,
    pub normal_threshold: f32,
    pub warm_threshold: f32,
    pub hot_threshold: f32,
    pub critical_threshold: f32,
    pub hysteresis: f32,
    pub max_safe_temp: f32,
}

impl ThermalManager {
    /// Create a new thermal manager
    pub fn new(hardware: Arc<HardwareDetector>) -> Result<Self> {
        let thermal_profile = ThermalProfile::for_apu(hardware.get_apu_type());
        
        Ok(Self {
            current_state: Arc::new(AtomicU8::new(ThermalState::Normal as u8)),
            hardware,
            thermal_profile,
            temperature_history: parking_lot::Mutex::new(Vec::new()),
            prediction_window: Duration::from_secs(30),
        })
    }
    
    /// Start thermal monitoring loop
    pub async fn start_monitoring(&self) -> Result<()> {
        let mut interval = tokio::time::interval(Duration::from_millis(1000)); // 1Hz monitoring
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.update_thermal_state().await {
                tracing::warn!("Thermal update failed: {}", e);
            }
        }
    }
    
    /// Update thermal state based on current temperature
    async fn update_thermal_state(&self) -> Result<()> {
        let current_temp = self.read_cpu_temperature().await?;
        let now = Instant::now();
        
        // Update temperature history
        {
            let mut history = self.temperature_history.lock();
            history.push((now, current_temp));
            
            // Keep only last 5 minutes of data
            let cutoff = now - Duration::from_secs(300);
            history.retain(|(timestamp, _)| *timestamp > cutoff);
        }
        
        // Determine thermal state with hysteresis
        let new_state = self.calculate_thermal_state(current_temp);
        let current_state_val = self.current_state.load(Ordering::Relaxed);
        let current_state = unsafe { std::mem::transmute::<u8, ThermalState>(current_state_val) };
        
        // Apply hysteresis to prevent oscillation
        let should_update = match (current_state, new_state) {
            (ThermalState::Cool, ThermalState::Normal) => current_temp > self.thermal_profile.normal_threshold + self.thermal_profile.hysteresis,
            (ThermalState::Normal, ThermalState::Cool) => current_temp < self.thermal_profile.cool_threshold - self.thermal_profile.hysteresis,
            (ThermalState::Normal, ThermalState::Warm) => current_temp > self.thermal_profile.warm_threshold + self.thermal_profile.hysteresis,
            (ThermalState::Warm, ThermalState::Normal) => current_temp < self.thermal_profile.normal_threshold - self.thermal_profile.hysteresis,
            (ThermalState::Warm, ThermalState::Hot) => current_temp > self.thermal_profile.hot_threshold + self.thermal_profile.hysteresis,
            (ThermalState::Hot, ThermalState::Warm) => current_temp < self.thermal_profile.warm_threshold - self.thermal_profile.hysteresis,
            (_, ThermalState::Critical) => true, // Always update to critical
            _ => new_state != current_state,
        };
        
        if should_update {
            self.current_state.store(new_state as u8, Ordering::Relaxed);
            tracing::info!("Thermal state changed: {:?} -> {:?} ({}°C)", current_state, new_state, current_temp);
            
            // Handle critical temperature
            if new_state == ThermalState::Critical {
                self.handle_critical_temperature(current_temp).await?;
            }
        }
        
        Ok(())
    }
    
    /// Calculate thermal state from temperature
    fn calculate_thermal_state(&self, temperature: f32) -> ThermalState {
        if temperature >= self.thermal_profile.critical_threshold {
            ThermalState::Critical
        } else if temperature >= self.thermal_profile.hot_threshold {
            ThermalState::Hot
        } else if temperature >= self.thermal_profile.warm_threshold {
            ThermalState::Warm
        } else if temperature >= self.thermal_profile.normal_threshold {
            ThermalState::Normal
        } else {
            ThermalState::Cool
        }
    }
    
    /// Read CPU temperature from hardware sensors
    async fn read_cpu_temperature(&self) -> Result<f32> {
        // Try to read from Steam Deck specific sensors
        if let Ok(temp) = self.read_apu_temperature().await {
            return Ok(temp);
        }
        
        // Fallback to generic CPU temperature
        if let Ok(temp) = self.read_generic_cpu_temperature().await {
            return Ok(temp);
        }
        
        // If all else fails, estimate from system load
        Ok(self.estimate_temperature_from_load().await)
    }
    
    /// Read APU temperature from Steam Deck sensors
    async fn read_apu_temperature(&self) -> Result<f32> {
        // Steam Deck APU temperature path
        let temp_paths = [
            "/sys/class/hwmon/hwmon0/temp1_input",
            "/sys/class/hwmon/hwmon1/temp1_input",
            "/sys/class/hwmon/hwmon2/temp1_input",
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
        ];
        
        for path in &temp_paths {
            if let Ok(temp_str) = tokio::fs::read_to_string(path).await {
                if let Ok(temp_millidegrees) = temp_str.trim().parse::<i32>() {
                    let temp_celsius = temp_millidegrees as f32 / 1000.0;
                    
                    // Sanity check temperature reading
                    if temp_celsius > 0.0 && temp_celsius < 150.0 {
                        return Ok(temp_celsius);
                    }
                }
            }
        }
        
        anyhow::bail!("No valid APU temperature sensor found")
    }
    
    /// Read generic CPU temperature
    async fn read_generic_cpu_temperature(&self) -> Result<f32> {
        // Use sysinfo crate for cross-platform temperature reading
        tokio::task::spawn_blocking(|| {
            let mut system = sysinfo::System::new();
            system.refresh_components_list();
            system.refresh_components();
            
            for component in system.components() {
                if component.label().to_lowercase().contains("cpu") 
                   || component.label().to_lowercase().contains("core") 
                   || component.label().to_lowercase().contains("package") {
                    return Ok(component.temperature());
                }
            }
            
            anyhow::bail!("No CPU temperature sensor found")
        }).await.context("Failed to read CPU temperature")?
    }
    
    /// Estimate temperature from system load (fallback)
    async fn estimate_temperature_from_load(&self) -> f32 {
        tokio::task::spawn_blocking(|| {
            let mut system = sysinfo::System::new();
            system.refresh_cpu();
            
            let avg_usage = system.cpus()
                .iter()
                .map(|cpu| cpu.cpu_usage())
                .sum::<f32>() / system.cpus().len() as f32;
            
            // Rough estimation: 40°C base + load percentage
            40.0 + avg_usage * 0.4
        }).await.unwrap_or(50.0) // Default to 50°C if estimation fails
    }
    
    /// Handle critical temperature situation
    async fn handle_critical_temperature(&self, temperature: f32) -> Result<()> {
        tracing::error!("Critical temperature reached: {}°C", temperature);
        
        // Emergency throttling - stop all shader compilation
        // In a real implementation, this would signal the compilation system
        // to halt all operations immediately
        
        // Wait for temperature to drop
        for _ in 0..30 { // Wait up to 30 seconds
            sleep(Duration::from_secs(1)).await;
            let current_temp = self.read_cpu_temperature().await?;
            
            if current_temp < self.thermal_profile.hot_threshold {
                tracing::info!("Temperature recovered to {}°C", current_temp);
                return Ok(());
            }
        }
        
        tracing::error!("Temperature did not recover, system may need intervention");
        Ok(())
    }
    
    /// Get current thermal state
    pub fn get_thermal_state(&self) -> ThermalState {
        let state_val = self.current_state.load(Ordering::Relaxed);
        unsafe { std::mem::transmute::<u8, ThermalState>(state_val) }
    }
    
    /// Get current temperature
    pub async fn get_current_temperature(&self) -> Result<f32> {
        self.read_cpu_temperature().await
    }
    
    /// Predict future thermal state
    pub fn predict_thermal_state(&self, seconds_ahead: u32) -> ThermalState {
        let history = self.temperature_history.lock();
        
        if history.len() < 2 {
            return self.get_thermal_state();
        }
        
        // Simple linear trend prediction
        let recent_temps: Vec<f32> = history.iter()
            .rev()
            .take(10) // Use last 10 readings
            .map(|(_, temp)| *temp)
            .collect();
        
        if recent_temps.len() < 2 {
            return self.get_thermal_state();
        }
        
        let trend = (recent_temps[0] - recent_temps[recent_temps.len() - 1]) / recent_temps.len() as f32;
        let predicted_temp = recent_temps[0] + trend * seconds_ahead as f32;
        
        self.calculate_thermal_state(predicted_temp)
    }
    
    /// Get thermal management recommendations
    pub fn get_thermal_recommendations(&self) -> ThermalRecommendations {
        let current_state = self.get_thermal_state();
        let predicted_state = self.predict_thermal_state(15); // 15 seconds ahead
        
        ThermalRecommendations {
            current_state,
            predicted_state,
            recommended_thread_reduction: match current_state {
                ThermalState::Cool => 0.0,
                ThermalState::Normal => 0.0,
                ThermalState::Warm => 0.25,
                ThermalState::Hot => 0.5,
                ThermalState::Critical => 0.8,
            },
            compilation_intensity_multiplier: match current_state {
                ThermalState::Cool => 1.2,
                ThermalState::Normal => 1.0,
                ThermalState::Warm => 0.7,
                ThermalState::Hot => 0.4,
                ThermalState::Critical => 0.1,
            },
            should_pause_compilation: current_state == ThermalState::Critical,
        }
    }
}

impl ThermalProfile {
    /// Create thermal profile for specific APU type
    pub fn for_apu(apu_type: ApuType) -> Self {
        match apu_type {
            ApuType::VanGogh => Self {
                cool_threshold: 60.0,
                normal_threshold: 70.0,
                warm_threshold: 80.0,
                hot_threshold: 90.0,
                critical_threshold: 95.0,
                hysteresis: 2.0,
                max_safe_temp: 92.0,
            },
            ApuType::Phoenix => Self {
                cool_threshold: 62.0,
                normal_threshold: 72.0,
                warm_threshold: 82.0,
                hot_threshold: 92.0,
                critical_threshold: 97.0,
                hysteresis: 2.0,
                max_safe_temp: 94.0,
            },
            ApuType::Unknown => Self {
                cool_threshold: 60.0,
                normal_threshold: 70.0,
                warm_threshold: 80.0,
                hot_threshold: 85.0,
                critical_threshold: 90.0,
                hysteresis: 2.0,
                max_safe_temp: 87.0,
            },
        }
    }
}

/// Thermal management recommendations
#[derive(Debug, Clone)]
pub struct ThermalRecommendations {
    pub current_state: ThermalState,
    pub predicted_state: ThermalState,
    pub recommended_thread_reduction: f32,
    pub compilation_intensity_multiplier: f32,
    pub should_pause_compilation: bool,
}