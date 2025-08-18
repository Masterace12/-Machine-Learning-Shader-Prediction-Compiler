//! SIMD-optimized feature extraction for shader analysis

use anyhow::Result;
use serde::{Deserialize, Serialize};
// SIMD support is experimental, use fallback for now
#[cfg(feature = "simd")]
use std::simd::{f32x8, f32x4, SimdFloat};

/// Shader features optimized for SIMD operations
#[repr(C, align(32))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderFeatures {
    // Core complexity metrics (first 8 features - SIMD aligned)
    pub instruction_count: f32,
    pub register_usage: f32,
    pub texture_samples: f32,
    pub memory_operations: f32,
    pub control_flow_complexity: f32,
    pub wave_size: f32,
    pub uses_derivatives: f32,
    pub uses_tessellation: f32,
    
    // Additional features (second 4 features)
    pub uses_geometry_shader: f32,
    pub shader_type_hash: f32,
    pub optimization_level: f32,
    pub cache_priority: f32,
    
    // Steam Deck specific features
    pub van_gogh_optimized: f32,
    pub rdna2_features: f32,
    pub thermal_state: f32,
    pub power_mode: f32,
}

/// Feature extractor for shader bytecode analysis
pub struct FeatureExtractor {
    // Normalization parameters
    normalization_params: NormalizationParams,
}

/// Normalization parameters for feature scaling
#[derive(Debug, Clone)]
struct NormalizationParams {
    means: [f32; 16],
    stds: [f32; 16],
}

impl Default for ShaderFeatures {
    fn default() -> Self {
        Self {
            instruction_count: 0.0,
            register_usage: 0.0,
            texture_samples: 0.0,
            memory_operations: 0.0,
            control_flow_complexity: 0.0,
            wave_size: 64.0, // Default wave size for RDNA2
            uses_derivatives: 0.0,
            uses_tessellation: 0.0,
            uses_geometry_shader: 0.0,
            shader_type_hash: 0.0,
            optimization_level: 1.0,
            cache_priority: 0.5,
            van_gogh_optimized: 1.0, // Assume Steam Deck optimization
            rdna2_features: 1.0,
            thermal_state: 0.5, // Normal thermal state
            power_mode: 1.0, // Performance mode
        }
    }
}

impl ShaderFeatures {
    /// Create new shader features with basic metrics
    pub fn new(
        instruction_count: u32,
        register_usage: u32,
        texture_samples: u32,
        memory_operations: u32,
    ) -> Self {
        let mut features = Self::default();
        features.instruction_count = instruction_count as f32;
        features.register_usage = register_usage as f32;
        features.texture_samples = texture_samples as f32;
        features.memory_operations = memory_operations as f32;
        features
    }
    
    /// Extract first 8 features as SIMD vector for fast processing
    #[cfg(feature = "simd")]
    #[inline(always)]
    pub fn extract_simd_primary(&self) -> f32x8 {
        unsafe {
            let ptr = self as *const _ as *const f32;
            let slice = std::slice::from_raw_parts(ptr, 8);
            f32x8::from_slice(slice)
        }
    }
    
    /// Extract second 4 features as SIMD vector
    #[cfg(feature = "simd")]
    #[inline(always)]
    pub fn extract_simd_secondary(&self) -> f32x4 {
        unsafe {
            let ptr = (self as *const _ as *const f32).offset(8);
            let slice = std::slice::from_raw_parts(ptr, 4);
            f32x4::from_slice(slice)
        }
    }
    
    /// Normalize features using SIMD operations
    #[cfg(feature = "simd")]
    pub fn normalize_simd(&mut self, params: &NormalizationParams) {
        // Normalize first 8 features
        let primary = self.extract_simd_primary();
        let means_primary = f32x8::from_slice(&params.means[0..8]);
        let stds_primary = f32x8::from_slice(&params.stds[0..8]);
        let normalized_primary = (primary - means_primary) / stds_primary;
        
        unsafe {
            let ptr = self as *mut _ as *mut f32;
            let slice = std::slice::from_raw_parts_mut(ptr, 8);
            normalized_primary.copy_to_slice(slice);
        }
        
        // Normalize second 4 features
        let secondary = self.extract_simd_secondary();
        let means_secondary = f32x4::from_slice(&params.means[8..12]);
        let stds_secondary = f32x4::from_slice(&params.stds[8..12]);
        let normalized_secondary = (secondary - means_secondary) / stds_secondary;
        
        unsafe {
            let ptr = (self as *mut _ as *mut f32).offset(8);
            let slice = std::slice::from_raw_parts_mut(ptr, 4);
            normalized_secondary.copy_to_slice(slice);
        }
        
        // Normalize remaining features manually
        for i in 12..16 {
            let value = unsafe { *((self as *const _ as *const f32).offset(i as isize)) };
            let normalized = (value - params.means[i]) / params.stds[i];
            unsafe {
                *((self as *mut _ as *mut f32).offset(i as isize)) = normalized;
            }
        }
    }
    
    /// Fallback normalization without SIMD
    #[cfg(not(feature = "simd"))]
    pub fn normalize(&mut self, params: &NormalizationParams) {
        let values = unsafe {
            std::slice::from_raw_parts_mut(self as *mut _ as *mut f32, 16)
        };
        
        for (i, value) in values.iter_mut().enumerate() {
            if i < 16 {
                *value = (*value - params.means[i]) / params.stds[i];
            }
        }
    }
    
    /// Convert to ndarray for ONNX inference
    pub fn to_ndarray(&self) -> ndarray::Array2<f32> {
        let values = unsafe {
            std::slice::from_raw_parts(self as *const _ as *const f32, 16)
        };
        
        ndarray::Array2::from_shape_vec((1, 16), values.to_vec())
            .expect("Failed to create ndarray")
    }
    
    /// Set thermal state (0.0 = cool, 0.5 = normal, 1.0 = hot)
    pub fn set_thermal_state(&mut self, thermal_state: f32) {
        self.thermal_state = thermal_state.clamp(0.0, 1.0);
    }
    
    /// Set power mode (0.0 = battery save, 0.5 = balanced, 1.0 = performance)
    pub fn set_power_mode(&mut self, power_mode: f32) {
        self.power_mode = power_mode.clamp(0.0, 1.0);
    }
    
    /// Calculate feature complexity score for tier selection
    pub fn complexity_score(&self) -> f32 {
        let base_complexity = self.instruction_count * 0.001 +
                            self.register_usage * 0.01 +
                            self.texture_samples * 0.1 +
                            self.memory_operations * 0.05;
        
        let feature_multiplier = 1.0 +
                               self.uses_derivatives * 0.2 +
                               self.uses_tessellation * 0.3 +
                               self.uses_geometry_shader * 0.25;
        
        base_complexity * feature_multiplier
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor with default normalization
    pub fn new() -> Self {
        Self {
            normalization_params: NormalizationParams::default(),
        }
    }
    
    /// Create feature extractor with custom normalization parameters
    pub fn with_normalization(means: [f32; 16], stds: [f32; 16]) -> Self {
        Self {
            normalization_params: NormalizationParams { means, stds },
        }
    }
    
    /// Extract features from SPIR-V bytecode
    pub fn extract_from_spirv(&self, spirv_data: &[u8]) -> Result<ShaderFeatures> {
        // Basic SPIR-V parsing for feature extraction
        let mut features = ShaderFeatures::default();
        
        // Simple instruction counting (this would be more sophisticated in practice)
        features.instruction_count = (spirv_data.len() / 4) as f32; // Rough instruction count
        
        // Analyze SPIR-V opcodes for more detailed features
        self.analyze_spirv_opcodes(spirv_data, &mut features)?;
        
        Ok(features)
    }
    
    /// Extract features from shader source code
    pub fn extract_from_source(&self, source: &str, shader_type: ShaderType) -> Result<ShaderFeatures> {
        let mut features = ShaderFeatures::default();
        
        // Basic text analysis
        features.instruction_count = source.lines().count() as f32;
        features.shader_type_hash = self.compute_shader_type_hash(shader_type);
        
        // Look for specific patterns
        if source.contains("ddx") || source.contains("ddy") || source.contains("fwidth") {
            features.uses_derivatives = 1.0;
        }
        
        if source.contains("tessellation") {
            features.uses_tessellation = 1.0;
        }
        
        if shader_type == ShaderType::Geometry {
            features.uses_geometry_shader = 1.0;
        }
        
        // Count texture operations
        features.texture_samples = source.matches("texture").count() as f32 +
                                  source.matches("sample").count() as f32;
        
        Ok(features)
    }
    
    /// Normalize features for inference
    pub fn normalize(&self, features: &mut ShaderFeatures) {
        #[cfg(feature = "simd")]
        features.normalize_simd(&self.normalization_params);
        
        #[cfg(not(feature = "simd"))]
        features.normalize(&self.normalization_params);
    }
    
    /// Analyze SPIR-V opcodes for detailed feature extraction
    fn analyze_spirv_opcodes(&self, spirv_data: &[u8], features: &mut ShaderFeatures) -> Result<()> {
        if spirv_data.len() < 20 {
            return Ok(()); // Invalid SPIR-V
        }
        
        let mut offset = 20; // Skip SPIR-V header
        let mut instruction_count = 0u32;
        let mut texture_ops = 0u32;
        let mut memory_ops = 0u32;
        let mut control_flow_ops = 0u32;
        
        while offset + 4 <= spirv_data.len() {
            let instruction = u32::from_le_bytes([
                spirv_data[offset],
                spirv_data[offset + 1],
                spirv_data[offset + 2],
                spirv_data[offset + 3],
            ]);
            
            let opcode = instruction & 0xFFFF;
            let length = (instruction >> 16) as usize;
            
            if length == 0 {
                break;
            }
            
            // Analyze specific opcodes
            match opcode {
                // Texture operations
                87..=94 => texture_ops += 1,  // OpImageSample* operations
                
                // Memory operations
                61..=66 => memory_ops += 1,   // OpLoad, OpStore, etc.
                
                // Control flow
                245..=255 => control_flow_ops += 1, // OpBranch, OpSwitch, etc.
                
                // Derivatives
                76 | 77 | 78 => features.uses_derivatives = 1.0, // OpDPdx, OpDPdy, OpFwidth
                
                _ => {}
            }
            
            instruction_count += 1;
            offset += length * 4;
        }
        
        features.instruction_count = instruction_count as f32;
        features.texture_samples = texture_ops as f32;
        features.memory_operations = memory_ops as f32;
        features.control_flow_complexity = control_flow_ops as f32;
        
        Ok(())
    }
    
    /// Compute shader type hash
    fn compute_shader_type_hash(&self, shader_type: ShaderType) -> f32 {
        match shader_type {
            ShaderType::Vertex => 1.0,
            ShaderType::Fragment => 2.0,
            ShaderType::Geometry => 3.0,
            ShaderType::Compute => 4.0,
            ShaderType::TessellationControl => 5.0,
            ShaderType::TessellationEvaluation => 6.0,
        }
    }
}

impl Default for NormalizationParams {
    fn default() -> Self {
        // Default normalization parameters based on typical shader characteristics
        Self {
            means: [
                500.0,   // instruction_count
                32.0,    // register_usage
                4.0,     // texture_samples
                10.0,    // memory_operations
                5.0,     // control_flow_complexity
                64.0,    // wave_size
                0.1,     // uses_derivatives
                0.05,    // uses_tessellation
                0.02,    // uses_geometry_shader
                2.0,     // shader_type_hash
                1.0,     // optimization_level
                0.5,     // cache_priority
                1.0,     // van_gogh_optimized
                1.0,     // rdna2_features
                0.5,     // thermal_state
                1.0,     // power_mode
            ],
            stds: [
                300.0,   // instruction_count
                16.0,    // register_usage
                3.0,     // texture_samples
                8.0,     // memory_operations
                4.0,     // control_flow_complexity
                16.0,    // wave_size
                0.3,     // uses_derivatives
                0.2,     // uses_tessellation
                0.15,    // uses_geometry_shader
                1.5,     // shader_type_hash
                0.5,     // optimization_level
                0.3,     // cache_priority
                0.1,     // van_gogh_optimized
                0.1,     // rdna2_features
                0.3,     // thermal_state
                0.3,     // power_mode
            ],
        }
    }
}

/// Shader type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Geometry,
    Compute,
    TessellationControl,
    TessellationEvaluation,
}