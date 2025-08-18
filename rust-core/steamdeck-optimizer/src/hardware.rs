//! Hardware detection for Steam Deck models and APU types

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Hardware detector for Steam Deck identification
pub struct HardwareDetector {
    model: SteamDeckModel,
    apu_type: ApuType,
}

/// Steam Deck model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SteamDeckModel {
    LCD,     // Original 7nm Van Gogh
    OLED,    // 6nm Van Gogh refresh
    Unknown, // Fallback
}

/// APU type identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApuType {
    VanGogh,  // Steam Deck APU
    Phoenix,  // Newer AMD APU
    Unknown,  // Fallback
}

impl HardwareDetector {
    /// Create new hardware detector and identify system
    pub fn new() -> Result<Self> {
        let model = Self::detect_steam_deck_model()?;
        let apu_type = Self::detect_apu_type()?;
        
        Ok(Self { model, apu_type })
    }
    
    /// Get Steam Deck model
    pub fn get_model(&self) -> SteamDeckModel {
        self.model
    }
    
    /// Get APU type
    pub fn get_apu_type(&self) -> ApuType {
        self.apu_type
    }
    
    /// Check if running on Steam Deck
    pub fn is_steam_deck(&self) -> bool {
        matches!(self.model, SteamDeckModel::LCD | SteamDeckModel::OLED)
    }
    
    /// Detect Steam Deck model variant
    fn detect_steam_deck_model() -> Result<SteamDeckModel> {
        // Check DMI product name
        if let Ok(product_name) = std::fs::read_to_string("/sys/devices/virtual/dmi/id/product_name") {
            let product_name = product_name.trim();
            match product_name {
                "Jupiter" => return Ok(SteamDeckModel::LCD),
                "Galileo" => return Ok(SteamDeckModel::OLED),
                _ => {}
            }
        }
        
        // Check for Steam Deck environment variables
        if std::env::var("SteamDeck").is_ok() {
            // Try to determine model from other indicators
            if Self::is_oled_model() {
                return Ok(SteamDeckModel::OLED);
            } else {
                return Ok(SteamDeckModel::LCD);
            }
        }
        
        // Check if /home/deck exists (common Steam Deck indicator)
        if std::path::Path::new("/home/deck").exists() {
            // Assume LCD model if we can't determine otherwise
            return Ok(SteamDeckModel::LCD);
        }
        
        Ok(SteamDeckModel::Unknown)
    }
    
    /// Detect APU type
    fn detect_apu_type() -> Result<ApuType> {
        // Check CPU model name
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            if cpuinfo.contains("AMD Custom APU 0405") || cpuinfo.contains("Van Gogh") {
                return Ok(ApuType::VanGogh);
            }
            
            if cpuinfo.contains("Phoenix") {
                return Ok(ApuType::Phoenix);
            }
        }
        
        // Check device tree (if available)
        if let Ok(compatible) = std::fs::read_to_string("/proc/device-tree/compatible") {
            if compatible.contains("valve,jupiter") || compatible.contains("valve,galileo") {
                return Ok(ApuType::VanGogh);
            }
        }
        
        Ok(ApuType::Unknown)
    }
    
    /// Determine if this is the OLED model
    fn is_oled_model() -> bool {
        // Check for OLED-specific indicators
        
        // Check display panel information
        if let Ok(panel_info) = std::fs::read_to_string("/sys/class/drm/card0-eDP-1/status") {
            // OLED models may have different panel characteristics
            // This is a simplified check
        }
        
        // Check manufacturing date (OLED models are newer)
        if let Ok(dmi_date) = std::fs::read_to_string("/sys/devices/virtual/dmi/id/bios_date") {
            // OLED models typically have BIOS dates after 2023
            if let Ok(date) = chrono::NaiveDate::parse_from_str(&dmi_date.trim(), "%m/%d/%Y") {
                if date.year() >= 2023 && date.month() >= 11 {
                    return true;
                }
            }
        }
        
        // Default to LCD if uncertain
        false
    }
    
    /// Get hardware capabilities
    pub fn get_capabilities(&self) -> HardwareCapabilities {
        match self.model {
            SteamDeckModel::LCD => HardwareCapabilities {
                max_tdp_watts: 15.0,
                cpu_cores: 4,
                cpu_threads: 8,
                gpu_compute_units: 8,
                memory_gb: 16,
                memory_bandwidth_gbps: 88.0,
                supports_rdna2: true,
                supports_fsr: true,
                has_hardware_decode: true,
                power_efficiency_factor: 1.0,
            },
            SteamDeckModel::OLED => HardwareCapabilities {
                max_tdp_watts: 15.0,
                cpu_cores: 4,
                cpu_threads: 8,
                gpu_compute_units: 8,
                memory_gb: 16,
                memory_bandwidth_gbps: 88.0,
                supports_rdna2: true,
                supports_fsr: true,
                has_hardware_decode: true,
                power_efficiency_factor: 1.15, // ~15% more efficient
            },
            SteamDeckModel::Unknown => HardwareCapabilities {
                max_tdp_watts: 12.0,
                cpu_cores: 4,
                cpu_threads: 8,
                gpu_compute_units: 4,
                memory_gb: 8,
                memory_bandwidth_gbps: 50.0,
                supports_rdna2: false,
                supports_fsr: false,
                has_hardware_decode: false,
                power_efficiency_factor: 0.8,
            },
        }
    }
    
    /// Get compilation optimization recommendations
    pub fn get_compilation_recommendations(&self) -> CompilationRecommendations {
        let capabilities = self.get_capabilities();
        
        CompilationRecommendations {
            max_parallel_tasks: capabilities.cpu_threads.min(4),
            preferred_optimization_level: if self.is_steam_deck() { 2 } else { 1 },
            use_fast_math: true,
            enable_vectorization: capabilities.supports_rdna2,
            target_architecture: match self.apu_type {
                ApuType::VanGogh => "gfx1030".to_string(),
                ApuType::Phoenix => "gfx1100".to_string(),
                ApuType::Unknown => "gfx900".to_string(),
            },
            memory_budget_mb: (capabilities.memory_gb * 1024 / 4) as u32, // Use 25% of system memory
        }
    }
}

/// Hardware capabilities structure
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub max_tdp_watts: f32,
    pub cpu_cores: u32,
    pub cpu_threads: u32,
    pub gpu_compute_units: u32,
    pub memory_gb: u32,
    pub memory_bandwidth_gbps: f32,
    pub supports_rdna2: bool,
    pub supports_fsr: bool,
    pub has_hardware_decode: bool,
    pub power_efficiency_factor: f32,
}

/// Compilation optimization recommendations
#[derive(Debug, Clone)]
pub struct CompilationRecommendations {
    pub max_parallel_tasks: u32,
    pub preferred_optimization_level: u32,
    pub use_fast_math: bool,
    pub enable_vectorization: bool,
    pub target_architecture: String,
    pub memory_budget_mb: u32,
}