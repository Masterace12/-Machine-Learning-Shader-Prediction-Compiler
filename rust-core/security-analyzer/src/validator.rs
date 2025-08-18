//! Security validator combining all analysis methods

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::spirv_analyzer::{SpirvAnalyzer, SecurityThreat, ThreatLevel};
use crate::sandbox::{SecureSandbox, SandboxResult};
use crate::fingerprint::HardwareFingerprint;

/// Security validator
pub struct SecurityValidator {
    spirv_analyzer: Arc<SpirvAnalyzer>,
    sandbox: Arc<SecureSandbox>,
    fingerprint: Arc<HardwareFingerprint>,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_safe: bool,
    pub threats: Vec<SecurityThreat>,
    pub sandbox_result: Option<SandboxExecutionSummary>,
    pub overall_risk_score: f32,
    pub recommendation: SecurityRecommendation,
}

/// Sandbox execution summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxExecutionSummary {
    pub success: bool,
    pub execution_time_ms: u32,
    pub memory_used_mb: u32,
    pub violations_count: usize,
}

/// Security recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityRecommendation {
    Allow,
    AllowWithMonitoring,
    Quarantine,
    Block,
}

impl SecurityValidator {
    /// Create new security validator
    pub fn new(
        spirv_analyzer: Arc<SpirvAnalyzer>,
        sandbox: Arc<SecureSandbox>,
        fingerprint: Arc<HardwareFingerprint>,
    ) -> Result<Self> {
        Ok(Self {
            spirv_analyzer,
            sandbox,
            fingerprint,
        })
    }
    
    /// Validate shader bytecode
    pub async fn validate_shader(&self, bytecode: &[u8]) -> Result<ValidationResult> {
        // Analyze SPIR-V for threats
        let threats = self.spirv_analyzer.analyze_bytecode(bytecode)?;
        
        // Execute in sandbox if no critical threats
        let sandbox_result = if !threats.iter().any(|t| t.level == ThreatLevel::Critical) {
            Some(self.sandbox.execute_shader(bytecode).await?)
        } else {
            None
        };
        
        // Calculate overall risk score
        let risk_score = self.calculate_risk_score(&threats, &sandbox_result);
        
        // Determine recommendation
        let recommendation = self.determine_recommendation(risk_score, &threats);
        
        let is_safe = matches!(recommendation, SecurityRecommendation::Allow | SecurityRecommendation::AllowWithMonitoring);
        
        Ok(ValidationResult {
            is_safe,
            threats,
            sandbox_result: sandbox_result.map(|r| SandboxExecutionSummary {
                success: r.success,
                execution_time_ms: r.execution_time_ms,
                memory_used_mb: r.memory_used_mb,
                violations_count: r.violations.len(),
            }),
            overall_risk_score: risk_score,
            recommendation,
        })
    }
    
    /// Calculate overall risk score (0.0 = safe, 1.0 = dangerous)
    fn calculate_risk_score(&self, threats: &[SecurityThreat], sandbox_result: &Option<SandboxResult>) -> f32 {
        let mut score = 0.0;
        
        // Score based on threats
        for threat in threats {
            let threat_score = match threat.level {
                ThreatLevel::Low => 0.1,
                ThreatLevel::Medium => 0.3,
                ThreatLevel::High => 0.6,
                ThreatLevel::Critical => 1.0,
            };
            score = score.max(threat_score * threat.confidence);
        }
        
        // Adjust based on sandbox execution
        if let Some(sandbox) = sandbox_result {
            if !sandbox.success || !sandbox.violations.is_empty() {
                score = score.max(0.7);
            }
        }
        
        score.min(1.0)
    }
    
    /// Determine security recommendation
    fn determine_recommendation(&self, risk_score: f32, threats: &[SecurityThreat]) -> SecurityRecommendation {
        // Block critical threats immediately
        if threats.iter().any(|t| t.level == ThreatLevel::Critical) {
            return SecurityRecommendation::Block;
        }
        
        match risk_score {
            x if x < 0.2 => SecurityRecommendation::Allow,
            x if x < 0.5 => SecurityRecommendation::AllowWithMonitoring,
            x if x < 0.8 => SecurityRecommendation::Quarantine,
            _ => SecurityRecommendation::Block,
        }
    }
}