//! Thermal monitoring for Steam Deck

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Thermal monitor
pub struct ThermalMonitor {
    sensor_paths: Vec<String>,
}

/// Thermal metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalMetrics {
    pub cpu_temp: f32,
    pub gpu_temp: f32,
    pub ambient_temp: f32,
    pub fan_speed_rpm: u32,
    pub thermal_throttling_active: bool,
    pub thermal_zone_temps: Vec<f32>,
}

impl ThermalMonitor {
    /// Create new thermal monitor
    pub fn new() -> Result<Self> {
        let sensor_paths = vec![
            "/sys/class/thermal/thermal_zone0/temp".to_string(),
            "/sys/class/thermal/thermal_zone1/temp".to_string(),
            "/sys/class/hwmon/hwmon0/temp1_input".to_string(),
            "/sys/class/hwmon/hwmon1/temp1_input".to_string(),
        ];
        
        Ok(Self { sensor_paths })
    }
    
    /// Get current thermal metrics
    pub async fn get_metrics(&self) -> Result<ThermalMetrics> {
        tokio::task::spawn_blocking(move || {
            let mut thermal_zone_temps = Vec::new();
            let mut cpu_temp = 50.0; // Default fallback
            let mut gpu_temp = 45.0;
            
            // Read thermal zone temperatures
            for i in 0..5 {
                let path = format!("/sys/class/thermal/thermal_zone{}/temp", i);
                if let Ok(temp_str) = std::fs::read_to_string(&path) {
                    if let Ok(temp_millidegrees) = temp_str.trim().parse::<i32>() {
                        let temp_celsius = temp_millidegrees as f32 / 1000.0;
                        if temp_celsius > 0.0 && temp_celsius < 150.0 {
                            thermal_zone_temps.push(temp_celsius);
                            
                            // First thermal zone is usually CPU
                            if i == 0 {
                                cpu_temp = temp_celsius;
                            }
                            // Second might be GPU
                            else if i == 1 {
                                gpu_temp = temp_celsius;
                            }
                        }
                    }
                }
            }
            
            // Read fan speed
            let fan_speed_rpm = Self::read_fan_speed().unwrap_or(2000);
            
            // Check for thermal throttling
            let thermal_throttling_active = cpu_temp > 85.0 || gpu_temp > 80.0;
            
            // Estimate ambient temperature
            let ambient_temp = thermal_zone_temps.iter().fold(0.0, |a, &b| a.min(b));
            
            Ok(ThermalMetrics {
                cpu_temp,
                gpu_temp,
                ambient_temp,
                fan_speed_rpm,
                thermal_throttling_active,
                thermal_zone_temps,
            })
        }).await?
    }
    
    /// Read fan speed from hwmon
    fn read_fan_speed() -> Option<u32> {
        let paths = [
            "/sys/class/hwmon/hwmon0/fan1_input",
            "/sys/class/hwmon/hwmon1/fan1_input",
            "/sys/class/hwmon/hwmon2/fan1_input",
        ];
        
        for path in &paths {
            if let Ok(speed_str) = std::fs::read_to_string(path) {
                if let Ok(speed) = speed_str.trim().parse::<u32>() {
                    return Some(speed);
                }
            }
        }
        None
    }
}