//! Power monitoring for Steam Deck battery and power management

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Power monitor
pub struct PowerMonitor {
    battery_path: String,
    power_supply_path: String,
}

/// Power metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    pub on_battery: bool,
    pub battery_percent: f32,
    pub battery_voltage: f32,
    pub battery_current_ma: i32,
    pub power_draw_watts: f32,
    pub time_remaining_minutes: Option<u32>,
    pub charging: bool,
    pub power_profile: String,
}

impl PowerMonitor {
    /// Create new power monitor
    pub fn new() -> Result<Self> {
        Ok(Self {
            battery_path: "/sys/class/power_supply/BAT1".to_string(),
            power_supply_path: "/sys/class/power_supply".to_string(),
        })
    }
    
    /// Get current power metrics
    pub async fn get_metrics(&self) -> Result<PowerMetrics> {
        tokio::task::spawn_blocking(move || {
            let battery_percent = Self::read_battery_percentage().unwrap_or(50.0);
            let on_battery = Self::is_on_battery();
            let charging = Self::is_charging();
            let battery_voltage = Self::read_battery_voltage().unwrap_or(3.7);
            let battery_current_ma = Self::read_battery_current().unwrap_or(0);
            let power_draw_watts = (battery_voltage * battery_current_ma.abs() as f32) / 1000.0;
            
            // Estimate time remaining based on current draw
            let time_remaining_minutes = if battery_current_ma < -100 && !charging {
                // Rough estimate: (battery_percent / 100) * typical_battery_capacity / current_draw
                Some(((battery_percent / 100.0) * 40.0 * 60.0 / (power_draw_watts + 1.0)) as u32)
            } else {
                None
            };
            
            let power_profile = Self::read_power_profile().unwrap_or_else(|| "balanced".to_string());
            
            Ok(PowerMetrics {
                on_battery,
                battery_percent,
                battery_voltage,
                battery_current_ma,
                power_draw_watts,
                time_remaining_minutes,
                charging,
                power_profile,
            })
        }).await?
    }
    
    /// Read battery percentage
    fn read_battery_percentage() -> Option<f32> {
        let paths = [
            "/sys/class/power_supply/BAT1/capacity",
            "/sys/class/power_supply/BAT0/capacity",
        ];
        
        for path in &paths {
            if let Ok(capacity_str) = std::fs::read_to_string(path) {
                if let Ok(capacity) = capacity_str.trim().parse::<f32>() {
                    return Some(capacity);
                }
            }
        }
        None
    }
    
    /// Check if running on battery
    fn is_on_battery() -> bool {
        let paths = [
            "/sys/class/power_supply/ADP1/online",
            "/sys/class/power_supply/AC/online",
        ];
        
        for path in &paths {
            if let Ok(online_str) = std::fs::read_to_string(path) {
                if let Ok(online) = online_str.trim().parse::<i32>() {
                    return online == 0; // 0 means AC adapter not connected
                }
            }
        }
        
        // Default to battery if we can't determine
        true
    }
    
    /// Check if battery is charging
    fn is_charging() -> bool {
        let paths = [
            "/sys/class/power_supply/BAT1/status",
            "/sys/class/power_supply/BAT0/status",
        ];
        
        for path in &paths {
            if let Ok(status_str) = std::fs::read_to_string(path) {
                let status = status_str.trim().to_lowercase();
                return status == "charging";
            }
        }
        false
    }
    
    /// Read battery voltage
    fn read_battery_voltage() -> Option<f32> {
        let paths = [
            "/sys/class/power_supply/BAT1/voltage_now",
            "/sys/class/power_supply/BAT0/voltage_now",
        ];
        
        for path in &paths {
            if let Ok(voltage_str) = std::fs::read_to_string(path) {
                if let Ok(voltage_microvolts) = voltage_str.trim().parse::<u64>() {
                    return Some(voltage_microvolts as f32 / 1_000_000.0); // Convert to volts
                }
            }
        }
        None
    }
    
    /// Read battery current
    fn read_battery_current() -> Option<i32> {
        let paths = [
            "/sys/class/power_supply/BAT1/current_now",
            "/sys/class/power_supply/BAT0/current_now",
        ];
        
        for path in &paths {
            if let Ok(current_str) = std::fs::read_to_string(path) {
                if let Ok(current_microamps) = current_str.trim().parse::<i64>() {
                    return Some((current_microamps / 1000) as i32); // Convert to milliamps
                }
            }
        }
        None
    }
    
    /// Read power profile
    fn read_power_profile() -> Option<String> {
        // Check for Steam Deck power profile
        if let Ok(profile) = std::fs::read_to_string("/sys/firmware/acpi/platform_profile") {
            return Some(profile.trim().to_string());
        }
        
        // Check for cpufreq governor as fallback
        if let Ok(governor) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") {
            return Some(governor.trim().to_string());
        }
        
        None
    }
}