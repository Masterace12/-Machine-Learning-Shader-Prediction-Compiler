//! SPIR-V bytecode security analysis and threat detection

use anyhow::Result;
use serde::{Deserialize, Serialize};
use blake3::hash;

/// SPIR-V security analyzer
pub struct SpirvAnalyzer {
    threat_signatures: Vec<ThreatSignature>,
}

/// Security threat detected in SPIR-V bytecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityThreat {
    pub threat_type: ThreatType,
    pub level: ThreatLevel,
    pub description: String,
    pub location: Option<usize>,
    pub confidence: f32,
}

/// Type of security threat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatType {
    MaliciousCode,
    ResourceExhaustion,
    InformationLeakage,
    UnauthorizedAccess,
    BufferOverflow,
    InfiniteLoop,
}

/// Severity level of threat
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Threat signature for pattern matching
struct ThreatSignature {
    pattern: Vec<u8>,
    threat_type: ThreatType,
    level: ThreatLevel,
    description: String,
}

impl SpirvAnalyzer {
    /// Create a new SPIR-V analyzer
    pub fn new() -> Result<Self> {
        let threat_signatures = Self::load_threat_signatures();
        
        Ok(Self {
            threat_signatures,
        })
    }
    
    /// Analyze SPIR-V bytecode for security threats
    pub fn analyze_bytecode(&self, bytecode: &[u8]) -> Result<Vec<SecurityThreat>> {
        let mut threats = Vec::new();
        
        // Basic SPIR-V validation
        if !self.is_valid_spirv(bytecode) {
            threats.push(SecurityThreat {
                threat_type: ThreatType::MaliciousCode,
                level: ThreatLevel::High,
                description: "Invalid SPIR-V bytecode format".to_string(),
                location: None,
                confidence: 0.95,
            });
            return Ok(threats);
        }
        
        // Check for known threat signatures
        threats.extend(self.scan_threat_signatures(bytecode)?);
        
        // Analyze SPIR-V structure
        threats.extend(self.analyze_spirv_structure(bytecode)?);
        
        // Check for resource exhaustion patterns
        threats.extend(self.check_resource_exhaustion(bytecode)?);
        
        Ok(threats)
    }
    
    /// Check if bytecode is valid SPIR-V
    fn is_valid_spirv(&self, bytecode: &[u8]) -> bool {
        if bytecode.len() < 20 {
            return false;
        }
        
        // Check SPIR-V magic number
        let magic = u32::from_le_bytes([
            bytecode[0], bytecode[1], bytecode[2], bytecode[3]
        ]);
        
        magic == 0x07230203 // SPIR-V magic number
    }
    
    /// Scan for known threat signatures
    fn scan_threat_signatures(&self, bytecode: &[u8]) -> Result<Vec<SecurityThreat>> {
        let mut threats = Vec::new();
        
        for signature in &self.threat_signatures {
            if let Some(location) = self.find_pattern(bytecode, &signature.pattern) {
                threats.push(SecurityThreat {
                    threat_type: signature.threat_type.clone(),
                    level: signature.level,
                    description: signature.description.clone(),
                    location: Some(location),
                    confidence: 0.8,
                });
            }
        }
        
        Ok(threats)
    }
    
    /// Analyze SPIR-V instruction structure
    fn analyze_spirv_structure(&self, bytecode: &[u8]) -> Result<Vec<SecurityThreat>> {
        let mut threats = Vec::new();
        let mut offset = 20; // Skip SPIR-V header
        let mut instruction_count = 0;
        let mut loop_depth = 0;
        let max_safe_instructions = 100_000;
        let max_safe_loop_depth = 10;
        
        while offset + 4 <= bytecode.len() {
            let instruction = u32::from_le_bytes([
                bytecode[offset],
                bytecode[offset + 1], 
                bytecode[offset + 2],
                bytecode[offset + 3],
            ]);
            
            let opcode = instruction & 0xFFFF;
            let length = (instruction >> 16) as usize;
            
            if length == 0 {
                break;
            }
            
            // Check for suspicious opcodes
            match opcode {
                // Loop instructions
                245..=255 => {
                    loop_depth += 1;
                    if loop_depth > max_safe_loop_depth {
                        threats.push(SecurityThreat {
                            threat_type: ThreatType::InfiniteLoop,
                            level: ThreatLevel::High,
                            description: format!("Excessive loop nesting depth: {}", loop_depth),
                            location: Some(offset),
                            confidence: 0.9,
                        });
                    }
                }
                
                // Atomic operations (potential race conditions)
                230..=244 => {
                    threats.push(SecurityThreat {
                        threat_type: ThreatType::UnauthorizedAccess,
                        level: ThreatLevel::Medium,
                        description: "Atomic memory operation detected".to_string(),
                        location: Some(offset),
                        confidence: 0.5,
                    });
                }
                
                _ => {}
            }
            
            instruction_count += 1;
            offset += length * 4;
            
            // Check for excessive instruction count
            if instruction_count > max_safe_instructions {
                threats.push(SecurityThreat {
                    threat_type: ThreatType::ResourceExhaustion,
                    level: ThreatLevel::High,
                    description: format!("Excessive instruction count: {}", instruction_count),
                    location: Some(offset),
                    confidence: 0.95,
                });
                break;
            }
        }
        
        Ok(threats)
    }
    
    /// Check for resource exhaustion patterns
    fn check_resource_exhaustion(&self, bytecode: &[u8]) -> Result<Vec<SecurityThreat>> {
        let mut threats = Vec::new();
        
        // Check bytecode size
        if bytecode.len() > 10 * 1024 * 1024 { // 10MB limit
            threats.push(SecurityThreat {
                threat_type: ThreatType::ResourceExhaustion,
                level: ThreatLevel::High,
                description: format!("Excessive bytecode size: {} bytes", bytecode.len()),
                location: None,
                confidence: 1.0,
            });
        }
        
        // Check for excessive memory allocations
        let allocation_count = self.count_memory_allocations(bytecode);
        if allocation_count > 1000 {
            threats.push(SecurityThreat {
                threat_type: ThreatType::ResourceExhaustion,
                level: ThreatLevel::Medium,
                description: format!("Excessive memory allocations: {}", allocation_count),
                location: None,
                confidence: 0.7,
            });
        }
        
        Ok(threats)
    }
    
    /// Find pattern in bytecode
    fn find_pattern(&self, bytecode: &[u8], pattern: &[u8]) -> Option<usize> {
        bytecode.windows(pattern.len())
            .position(|window| window == pattern)
    }
    
    /// Count memory allocation instructions
    fn count_memory_allocations(&self, bytecode: &[u8]) -> usize {
        let mut count = 0;
        let mut offset = 20; // Skip header
        
        while offset + 4 <= bytecode.len() {
            let instruction = u32::from_le_bytes([
                bytecode[offset],
                bytecode[offset + 1],
                bytecode[offset + 2], 
                bytecode[offset + 3],
            ]);
            
            let opcode = instruction & 0xFFFF;
            let length = (instruction >> 16) as usize;
            
            if length == 0 {
                break;
            }
            
            // Check for memory allocation opcodes
            match opcode {
                26 | 27 | 59 => count += 1, // OpVariable, OpImageTexelPointer, OpAccessChain
                _ => {}
            }
            
            offset += length * 4;
        }
        
        count
    }
    
    /// Load threat signatures database
    fn load_threat_signatures() -> Vec<ThreatSignature> {
        vec![
            // Example signatures for common threats
            ThreatSignature {
                pattern: vec![0xFF, 0xFF, 0xFF, 0xFF], // Suspicious pattern
                threat_type: ThreatType::MaliciousCode,
                level: ThreatLevel::High,
                description: "Suspicious byte pattern detected".to_string(),
            },
            
            ThreatSignature {
                pattern: vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // Null padding
                threat_type: ThreatType::InformationLeakage,
                level: ThreatLevel::Low,
                description: "Potential information leakage via padding".to_string(),
            },
        ]
    }
    
    /// Generate security hash for shader
    pub fn generate_security_hash(&self, bytecode: &[u8]) -> String {
        let hash_result = hash(bytecode);
        hex::encode(hash_result.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spirv_validation() {
        let analyzer = SpirvAnalyzer::new().unwrap();
        
        // Invalid SPIR-V (too short)
        let invalid_bytecode = vec![0x01, 0x02, 0x03];
        assert!(!analyzer.is_valid_spirv(&invalid_bytecode));
        
        // Valid SPIR-V header
        let valid_header = vec![
            0x03, 0x02, 0x23, 0x07, // Magic number
            0x00, 0x00, 0x01, 0x00, // Version
            0x00, 0x00, 0x00, 0x00, // Generator
            0x01, 0x00, 0x00, 0x00, // Bound
            0x00, 0x00, 0x00, 0x00, // Schema
        ];
        assert!(analyzer.is_valid_spirv(&valid_header));
    }
    
    #[test]
    fn test_threat_detection() {
        let analyzer = SpirvAnalyzer::new().unwrap();
        let test_bytecode = vec![
            0x03, 0x02, 0x23, 0x07, // Magic number
            0x00, 0x00, 0x01, 0x00, // Version
            0x00, 0x00, 0x00, 0x00, // Generator
            0x01, 0x00, 0x00, 0x00, // Bound
            0x00, 0x00, 0x00, 0x00, // Schema
            // Add some instruction data
            0xFF, 0xFF, 0xFF, 0xFF, // Suspicious pattern
        ];
        
        let threats = analyzer.analyze_bytecode(&test_bytecode).unwrap();
        assert!(!threats.is_empty());
    }
}