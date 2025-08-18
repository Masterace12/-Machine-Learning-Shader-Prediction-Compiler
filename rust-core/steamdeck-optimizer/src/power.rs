//! Power management for Steam Deck battery and AC optimization

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use crate::hardware::{HardwareDetector, SteamDeckModel};

/// Power manager for Steam Deck optimization
pub struct PowerManager {
    current_state: Arc<AtomicU8>,
    hardware: Arc<HardwareDetector>,
    power_profile: PowerProfile,
}

/// Power state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerState {
    BatterySave = 0,  // Minimal power usage
    Balanced = 1,     // Balance performance and power
    Performance = 2,  // Maximum performance
}

/// Power profile configuration
#[derive(Debug, Clone)]
pub struct PowerProfile {
    pub max_tdp_watts: f32,
    pub cpu_boost_enabled: bool,
    pub gpu_boost_enabled: bool,
    pub compilation_power_budget: f32,
}

impl PowerManager {
    /// Create a new power manager
    pub fn new(hardware: Arc<HardwareDetector>) -> Result<Self> {
        let power_profile = PowerProfile::for_model(hardware.get_model());
        
        Ok(Self {
            current_state: Arc<AtomicU8>::new(PowerState::Balanced as u8),
            hardware,
            power_profile,
        })
    }
    
    /// Get current power state
    pub fn get_power_state(&self) -> PowerState {
        let state_val = self.current_state.load(Ordering::Relaxed);
        unsafe { std::mem::transmute::<u8, PowerState>(state_val) }
    }
    
    /// Set power state
    pub fn set_power_state(&self, state: PowerState) {
        self.current_state.store(state as u8, Ordering::Relaxed);
    }
    
    /// Get recommended compilation thread count based on power state
    pub fn get_compilation_thread_limit(&self) -> usize {
        match self.get_power_state() {
            PowerState::BatterySave => 1,
            PowerState::Balanced => 2,
            PowerState::Performance => 4,
        }
    }
}

impl PowerProfile {
    /// Create power profile for specific Steam Deck model
    pub fn for_model(model: SteamDeckModel) -> Self {
        match model {
            SteamDeckModel::LCD => Self {
                max_tdp_watts: 15.0,
                cpu_boost_enabled: true,
                gpu_boost_enabled: true,
                compilation_power_budget: 8.0,
            },
            SteamDeckModel::OLED => Self {
                max_tdp_watts: 15.0,
                cpu_boost_enabled: true,
                gpu_boost_enabled: true,
                compilation_power_budget: 10.0, // Better efficiency
            },
            SteamDeckModel::Unknown => Self {
                max_tdp_watts: 12.0,
                cpu_boost_enabled: false,
                gpu_boost_enabled: false,
                compilation_power_budget: 6.0,
            },
        }
    }
}