//! Fast heuristic predictor as fallback for ML inference

use crate::features::{ShaderFeatures, ShaderType};
use std::collections::HashMap;

/// Heuristic predictor using rule-based estimation
pub struct HeuristicPredictor {
    base_times: HashMap<u32, f32>, // shader_type_hash -> base_time_ms
    complexity_weights: ComplexityWeights,
}

/// Weights for different complexity factors
#[derive(Debug, Clone)]
struct ComplexityWeights {
    instruction_weight: f32,
    register_weight: f32,
    texture_weight: f32,
    memory_weight: f32,
    control_flow_weight: f32,
    derivatives_multiplier: f32,
    tessellation_multiplier: f32,
    geometry_multiplier: f32,
}

impl Default for ComplexityWeights {
    fn default() -> Self {
        Self {
            instruction_weight: 0.005,      // 5ms per 1000 instructions
            register_weight: 0.1,           // 0.1ms per register
            texture_weight: 2.0,            // 2ms per texture sample
            memory_weight: 0.5,             // 0.5ms per memory operation
            control_flow_weight: 3.0,       // 3ms per branch/loop
            derivatives_multiplier: 1.4,    // 40% increase for derivatives
            tessellation_multiplier: 2.5,   // 150% increase for tessellation
            geometry_multiplier: 1.8,       // 80% increase for geometry shaders
        }
    }
}

impl HeuristicPredictor {
    /// Create a new heuristic predictor
    pub fn new() -> Self {
        let mut base_times = HashMap::new();
        
        // Base compilation times for different shader types (in milliseconds)
        base_times.insert(1, 8.0);   // Vertex shader
        base_times.insert(2, 12.0);  // Fragment shader
        base_times.insert(3, 25.0);  // Geometry shader
        base_times.insert(4, 15.0);  // Compute shader
        base_times.insert(5, 20.0);  // Tessellation control
        base_times.insert(6, 18.0);  // Tessellation evaluation
        
        Self {
            base_times,
            complexity_weights: ComplexityWeights::default(),
        }
    }
    
    /// Create heuristic predictor with Steam Deck optimized parameters
    pub fn new_steam_deck() -> Self {
        let mut predictor = Self::new();
        
        // Adjust weights for Steam Deck Van Gogh APU characteristics
        predictor.complexity_weights.instruction_weight = 0.008; // Slightly slower compilation
        predictor.complexity_weights.texture_weight = 1.5;      // Better texture units
        predictor.complexity_weights.control_flow_weight = 4.0; // Branching more expensive
        
        // Adjust base times for RDNA2 architecture
        for base_time in predictor.base_times.values_mut() {
            *base_time *= 1.1; // 10% slower base compilation than desktop
        }
        
        predictor
    }
    
    /// Predict compilation time using heuristics
    #[inline(always)]
    pub fn predict(&self, features: &ShaderFeatures) -> f32 {
        // Get base time for shader type
        let shader_type_key = features.shader_type_hash as u32;
        let base_time = self.base_times.get(&shader_type_key).copied().unwrap_or(10.0);
        
        // Calculate complexity factors
        let complexity = self.calculate_complexity(features);
        
        // Apply feature-specific multipliers
        let feature_multiplier = self.calculate_feature_multiplier(features);
        
        // Apply thermal and power adjustments
        let thermal_multiplier = self.calculate_thermal_multiplier(features);
        
        // Combine all factors
        let predicted_time = (base_time + complexity) * feature_multiplier * thermal_multiplier;
        
        // Ensure reasonable bounds
        predicted_time.max(0.5).min(5000.0)
    }
    
    /// Calculate complexity score from shader metrics
    fn calculate_complexity(&self, features: &ShaderFeatures) -> f32 {
        let weights = &self.complexity_weights;
        
        weights.instruction_weight * features.instruction_count +
        weights.register_weight * features.register_usage +
        weights.texture_weight * features.texture_samples +
        weights.memory_weight * features.memory_operations +
        weights.control_flow_weight * features.control_flow_complexity
    }
    
    /// Calculate multiplier based on shader features
    fn calculate_feature_multiplier(&self, features: &ShaderFeatures) -> f32 {
        let weights = &self.complexity_weights;
        
        let mut multiplier = 1.0;
        
        if features.uses_derivatives > 0.5 {
            multiplier *= weights.derivatives_multiplier;
        }
        
        if features.uses_tessellation > 0.5 {
            multiplier *= weights.tessellation_multiplier;
        }
        
        if features.uses_geometry_shader > 0.5 {
            multiplier *= weights.geometry_multiplier;
        }
        
        // Optimization level adjustment
        multiplier *= 2.0 - features.optimization_level; // Higher optimization = faster compile
        
        multiplier
    }
    
    /// Calculate thermal-based multiplier
    fn calculate_thermal_multiplier(&self, features: &ShaderFeatures) -> f32 {
        // Thermal state affects compilation speed
        let thermal_multiplier = 1.0 + features.thermal_state * 0.5; // Up to 50% slower when hot
        
        // Power mode affects compilation resources
        let power_multiplier = if features.power_mode < 0.5 {
            1.3 // 30% slower in power save mode
        } else {
            1.0
        };
        
        thermal_multiplier * power_multiplier
    }
    
    /// Estimate compilation probability (always returns high for heuristics)
    pub fn predict_success_probability(&self, features: &ShaderFeatures) -> f32 {
        // Heuristic compilation should almost always succeed
        let base_probability = 0.98;
        
        // Reduce slightly for very complex shaders
        let complexity_penalty = (features.complexity_score() / 1000.0).min(0.1);
        
        (base_probability - complexity_penalty).max(0.8)
    }
    
    /// Update predictor with observed compilation data
    pub fn update_with_feedback(&mut self, features: &ShaderFeatures, actual_time: f32) {
        let predicted_time = self.predict(features);
        let error_ratio = actual_time / predicted_time;
        
        // Adjust weights based on prediction error (simple adaptive mechanism)
        if error_ratio > 1.2 || error_ratio < 0.8 {
            let adjustment_factor = 0.95 + 0.1 * error_ratio;
            
            // Adjust the base time for this shader type
            let shader_type_key = features.shader_type_hash as u32;
            if let Some(base_time) = self.base_times.get_mut(&shader_type_key) {
                *base_time *= adjustment_factor.clamp(0.5, 2.0);
            }
        }
    }
}

/// Fast lookup table for common shader patterns
pub struct HeuristicLookupTable {
    patterns: HashMap<u64, f32>,
}

impl HeuristicLookupTable {
    /// Create a new lookup table with common patterns
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Pre-computed times for common shader patterns
        // Pattern hash -> compilation time (ms)
        patterns.insert(0x1234567890ABCDEF, 15.5); // Example pattern
        patterns.insert(0x2345678901BCDEF0, 22.3);
        patterns.insert(0x3456789012CDEF01, 8.7);
        
        Self { patterns }
    }
    
    /// Try to get compilation time from lookup table
    pub fn lookup(&self, pattern_hash: u64) -> Option<f32> {
        self.patterns.get(&pattern_hash).copied()
    }
    
    /// Add new pattern to lookup table
    pub fn add_pattern(&mut self, pattern_hash: u64, compilation_time: f32) {
        self.patterns.insert(pattern_hash, compilation_time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::ShaderFeatures;
    
    #[test]
    fn test_basic_prediction() {
        let predictor = HeuristicPredictor::new();
        let features = ShaderFeatures::default();
        
        let prediction = predictor.predict(&features);
        assert!(prediction > 0.0 && prediction < 1000.0);
    }
    
    #[test]
    fn test_steam_deck_predictor() {
        let predictor = HeuristicPredictor::new_steam_deck();
        let features = ShaderFeatures::default();
        
        let prediction = predictor.predict(&features);
        assert!(prediction > 0.0 && prediction < 1000.0);
    }
    
    #[test]
    fn test_complexity_scaling() {
        let predictor = HeuristicPredictor::new();
        
        let simple_features = ShaderFeatures::new(100, 8, 1, 2);
        let complex_features = ShaderFeatures::new(1000, 32, 10, 20);
        
        let simple_prediction = predictor.predict(&simple_features);
        let complex_prediction = predictor.predict(&complex_features);
        
        assert!(complex_prediction > simple_prediction);
    }
    
    #[test]
    fn test_thermal_impact() {
        let predictor = HeuristicPredictor::new();
        
        let mut features = ShaderFeatures::default();
        let normal_prediction = predictor.predict(&features);
        
        features.thermal_state = 1.0; // Hot
        let hot_prediction = predictor.predict(&features);
        
        assert!(hot_prediction > normal_prediction);
    }
    
    #[test]
    fn test_lookup_table() {
        let mut table = HeuristicLookupTable::new();
        
        let pattern_hash = 0x1234567890ABCDEF;
        let time = table.lookup(pattern_hash);
        assert!(time.is_some());
        
        table.add_pattern(0x9999999999999999, 42.0);
        let new_time = table.lookup(0x9999999999999999);
        assert_eq!(new_time, Some(42.0));
    }
}