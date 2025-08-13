//! Security analysis and validation for shader bytecode and caching
//! 
//! This module provides comprehensive security validation including SPIR-V analysis,
//! sandboxed execution, and hardware fingerprinting for Steam Deck.

use anyhow::Result;
use std::sync::Arc;

pub mod spirv_analyzer;
pub mod sandbox;
pub mod fingerprint;
pub mod validator;

// Re-export main types
pub use spirv_analyzer::{SpirvAnalyzer, SecurityThreat, ThreatLevel};
pub use sandbox::{SecureSandbox, SandboxConfig};
pub use fingerprint::{HardwareFingerprint, SteamDeckFingerprint};
pub use validator::{SecurityValidator, ValidationResult};

/// Main security analysis engine
pub struct SecurityAnalyzer {
    spirv_analyzer: Arc<SpirvAnalyzer>,
    sandbox: Arc<SecureSandbox>,
    fingerprint: Arc<HardwareFingerprint>,
    validator: Arc<SecurityValidator>,
}

impl SecurityAnalyzer {
    /// Create a new security analyzer
    pub fn new() -> Result<Self> {
        let spirv_analyzer = Arc::new(SpirvAnalyzer::new()?);
        let sandbox = Arc::new(SecureSandbox::new_steam_deck()?);
        let fingerprint = Arc::new(HardwareFingerprint::steam_deck()?);
        let validator = Arc::new(SecurityValidator::new(
            spirv_analyzer.clone(),
            sandbox.clone(),
            fingerprint.clone(),
        )?);
        
        Ok(Self {
            spirv_analyzer,
            sandbox,
            fingerprint,
            validator,
        })
    }
    
    /// Validate shader bytecode for security threats
    pub async fn validate_shader(&self, bytecode: &[u8]) -> Result<ValidationResult> {
        self.validator.validate_shader(bytecode).await
    }
    
    /// Get hardware fingerprint for anti-cheat compatibility
    pub fn get_hardware_fingerprint(&self) -> Result<Vec<u8>> {
        self.fingerprint.generate_fingerprint()
    }
}

/// Error types for security analysis
#[derive(thiserror::Error, Debug)]
pub enum SecurityError {
    #[error("SPIR-V analysis error: {0}")]
    SpirvError(String),
    
    #[error("Sandbox execution error: {0}")]
    SandboxError(String),
    
    #[error("Hardware fingerprinting error: {0}")]
    FingerprintError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
}