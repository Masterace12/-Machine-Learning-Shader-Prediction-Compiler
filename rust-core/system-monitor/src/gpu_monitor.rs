//! GPU monitoring for Steam Deck RDNA2 GPU

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// GPU monitor for Steam Deck
pub struct GpuMonitor {
    device_path: String,
}

/// GPU metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub usage_percent: f32,
    pub frequency_mhz: u32,
    pub memory_used_mb: u32,
    pub memory_total_mb: u32,
    pub temperature_c: f32,
    pub power_draw_watts: f32,
    pub shader_clock_mhz: u32,
    pub memory_clock_mhz: u32,
}

impl GpuMonitor {
    /// Create new GPU monitor
    pub fn new() -> Result<Self> {
        Ok(Self {
            device_path: "/sys/class/drm/card0".to_string(),
        })
    }
    
    /// Get current GPU metrics
    pub async fn get_metrics(&self) -> Result<GpuMetrics> {
        tokio::task::spawn_blocking(move || {
            // Try to read GPU usage from sysfs
            let usage_percent = Self::read_gpu_usage().unwrap_or(0.0);
            
            // Try to read GPU frequency
            let frequency_mhz = Self::read_gpu_frequency().unwrap_or(1600); // Default max for Steam Deck
            
            // Estimate memory usage (Steam Deck uses shared memory)
            let memory_total_mb = 16384; // 16GB total system memory
            let memory_used_mb = (memory_total_mb as f32 * usage_percent / 100.0) as u32;
            
            // Try to read temperature
            let temperature_c = Self::read_gpu_temperature().unwrap_or(50.0);
            
            // Estimate power draw based on usage
            let power_draw_watts = 15.0 * (usage_percent / 100.0); // Max 15W for Van Gogh GPU
            
            Ok(GpuMetrics {
                usage_percent,
                frequency_mhz,
                memory_used_mb,
                memory_total_mb,
                temperature_c,
                power_draw_watts,
                shader_clock_mhz: frequency_mhz,
                memory_clock_mhz: 5500, // LPDDR5 effective speed
            })
        }).await?
    }
    
    /// Read GPU usage from sysfs
    fn read_gpu_usage() -> Option<f32> {
        let paths = [
            "/sys/class/drm/card0/device/gpu_busy_percent",
            "/sys/class/drm/card1/device/gpu_busy_percent",
        ];
        
        for path in &paths {
            if let Ok(usage_str) = std::fs::read_to_string(path) {
                if let Ok(usage) = usage_str.trim().parse::<f32>() {
                    return Some(usage);
                }
            }
        }
        None
    }
    
    /// Read GPU frequency from sysfs
    fn read_gpu_frequency() -> Option<u32> {
        let paths = [
            "/sys/class/drm/card0/device/pp_dpm_sclk",
            "/sys/class/drm/card1/device/pp_dpm_sclk",
        ];
        
        for path in &paths {
            if let Ok(freq_str) = std::fs::read_to_string(path) {
                // Parse current frequency from DPM output
                for line in freq_str.lines() {
                    if line.contains('*') {
                        if let Some(freq_part) = line.split_whitespace().nth(1) {
                            if let Some(freq_str) = freq_part.strip_suffix("Mhz") {
                                if let Ok(freq) = freq_str.parse::<u32>() {
                                    return Some(freq);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
    
    /// Read GPU temperature from hwmon
    fn read_gpu_temperature() -> Option<f32> {
        let paths = [
            "/sys/class/hwmon/hwmon0/temp2_input",
            "/sys/class/hwmon/hwmon1/temp2_input",
            "/sys/class/hwmon/hwmon2/temp2_input",
        ];
        
        for path in &paths {
            if let Ok(temp_str) = std::fs::read_to_string(path) {
                if let Ok(temp_millidegrees) = temp_str.trim().parse::<i32>() {
                    let temp_celsius = temp_millidegrees as f32 / 1000.0;
                    if temp_celsius > 0.0 && temp_celsius < 150.0 {
                        return Some(temp_celsius);
                    }
                }
            }
        }
        None
    }
}