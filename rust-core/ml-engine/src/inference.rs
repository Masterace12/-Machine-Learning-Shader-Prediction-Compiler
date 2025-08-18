//! ONNX Runtime integration for high-performance ML inference

use anyhow::{Result, Context};
use ort::{Environment, Session, SessionBuilder, Value};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::path::Path;
use crate::features::ShaderFeatures;

/// ONNX-based predictor for shader compilation times
pub struct ONNXPredictor {
    session: Session,
    input_shape: Vec<usize>,
    quantized: bool,
}

/// Model configuration options
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub optimization_level: ort::GraphOptimizationLevel,
    pub intra_threads: Option<i16>,
    pub inter_threads: Option<i16>,
    pub enable_parallel_execution: bool,
    pub enable_quantization: bool,
    pub use_gpu: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            optimization_level: ort::GraphOptimizationLevel::Level3,
            intra_threads: Some(2), // Limited for Steam Deck
            inter_threads: Some(1),
            enable_parallel_execution: true,
            enable_quantization: false,
            use_gpu: false, // CPU-only by default for stability
        }
    }
}

impl ModelConfig {
    /// Steam Deck optimized configuration
    pub fn steam_deck_optimized() -> Self {
        Self {
            optimization_level: ort::GraphOptimizationLevel::Level3,
            intra_threads: Some(2), // Van Gogh has 4 cores, leave 2 for game
            inter_threads: Some(1),
            enable_parallel_execution: true,
            enable_quantization: true, // Enable for memory efficiency
            use_gpu: false, // Keep GPU free for games
        }
    }
    
    /// High performance configuration (for desktop)
    pub fn high_performance() -> Self {
        Self {
            optimization_level: ort::GraphOptimizationLevel::Level3,
            intra_threads: None, // Use all available cores
            inter_threads: None,
            enable_parallel_execution: true,
            enable_quantization: false,
            use_gpu: true,
        }
    }
}

impl ONNXPredictor {
    /// Create a new ONNX predictor with default configuration
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::new_with_config(model_path, ModelConfig::default())
    }
    
    /// Create ONNX predictor with Steam Deck optimized configuration
    pub fn new_steam_deck<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::new_with_config(model_path, ModelConfig::steam_deck_optimized())
    }
    
    /// Create ONNX predictor with custom configuration
    pub fn new_with_config<P: AsRef<Path>>(model_path: P, config: ModelConfig) -> Result<Self> {
        // Initialize ONNX Runtime environment
        let environment = Environment::builder()
            .with_name("shader_predictor")
            .with_log_level(ort::LoggingLevel::Warning)
            .build()
            .context("Failed to create ONNX environment")?;
        
        // Configure session builder
        let mut builder = SessionBuilder::new(&environment)
            .context("Failed to create session builder")?;
        
        // Apply optimization settings
        builder = builder
            .with_optimization_level(config.optimization_level)
            .context("Failed to set optimization level")?;
        
        if let Some(intra_threads) = config.intra_threads {
            builder = builder
                .with_intra_threads(intra_threads)
                .context("Failed to set intra threads")?;
        }
        
        if let Some(inter_threads) = config.inter_threads {
            builder = builder
                .with_inter_threads(inter_threads)
                .context("Failed to set inter threads")?;
        }
        
        if config.enable_parallel_execution {
            builder = builder
                .with_parallel_execution(true)
                .context("Failed to enable parallel execution")?;
        }
        
        // GPU acceleration (if available and requested)
        if config.use_gpu {
            // Try to enable GPU acceleration
            if let Ok(builder_with_gpu) = builder.with_provider(ort::CUDAExecutionProvider::default()) {
                builder = builder_with_gpu;
            } else if let Ok(builder_with_rocm) = builder.with_provider(ort::ROCmExecutionProvider::default()) {
                builder = builder_with_rocm; // For AMD GPUs like Steam Deck
            }
        }
        
        // Load the model
        let session = builder
            .with_model_from_file(model_path)
            .context("Failed to load ONNX model")?;
        
        Ok(Self {
            session,
            input_shape: vec![1, 16], // Batch size 1, 16 features
            quantized: config.enable_quantization,
        })
    }
    
    /// Predict compilation time for a single shader
    #[inline(always)]
    pub fn predict(&self, features: &ShaderFeatures) -> Result<f32> {
        // Convert features to ndarray
        let input_array = features.to_ndarray();
        
        // Create ONNX tensor
        let input_tensor = Value::from_array(self.session.allocator(), &input_array)
            .context("Failed to create input tensor")?;
        
        // Run inference
        let outputs = self.session
            .run(vec![input_tensor])
            .context("Failed to run inference")?;
        
        // Extract prediction
        let output = outputs[0]
            .try_extract::<f32>()
            .context("Failed to extract output")?;
        
        let prediction = output.view().first().copied().unwrap_or(0.0);
        
        // Post-process prediction (ensure reasonable bounds)
        Ok(prediction.max(0.1).min(10000.0)) // 0.1ms to 10s range
    }
    
    /// Batch prediction for multiple shaders
    pub fn predict_batch(&self, features: &[ShaderFeatures]) -> Result<Vec<f32>> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        
        let batch_size = features.len();
        
        // Convert features to batch array
        let mut input_data = Vec::with_capacity(batch_size * 16);
        for feature in features {
            let feature_array = feature.to_ndarray();
            input_data.extend(feature_array.iter().copied());
        }
        
        let input_array = Array2::from_shape_vec((batch_size, 16), input_data)
            .context("Failed to create batch input array")?;
        
        // Create ONNX tensor
        let input_tensor = Value::from_array(self.session.allocator(), &input_array)
            .context("Failed to create batch input tensor")?;
        
        // Run batch inference
        let outputs = self.session
            .run(vec![input_tensor])
            .context("Failed to run batch inference")?;
        
        // Extract predictions
        let output = outputs[0]
            .try_extract::<f32>()
            .context("Failed to extract batch output")?;
        
        let predictions: Vec<f32> = output.view()
            .iter()
            .map(|&pred| pred.max(0.1).min(10000.0)) // Clamp to reasonable range
            .collect();
        
        Ok(predictions)
    }
    
    /// Get model information
    pub fn get_model_info(&self) -> ModelInfo {
        let inputs = self.session.inputs();
        let outputs = self.session.outputs();
        
        ModelInfo {
            input_count: inputs.len(),
            output_count: outputs.len(),
            input_shape: self.input_shape.clone(),
            quantized: self.quantized,
        }
    }
    
    /// Warm up the model with dummy data
    pub fn warmup(&self) -> Result<()> {
        let dummy_features = ShaderFeatures::default();
        self.predict(&dummy_features)?;
        Ok(())
    }
}

/// Model information structure
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub input_count: usize,
    pub output_count: usize,
    pub input_shape: Vec<usize>,
    pub quantized: bool,
}

/// Optimized ONNX predictor for Steam Deck with thermal awareness
pub struct SteamDeckPredictor {
    predictor: ONNXPredictor,
    thermal_state: Arc<std::sync::atomic::AtomicU8>, // 0=cool, 1=warm, 2=hot
}

impl SteamDeckPredictor {
    /// Create a Steam Deck optimized predictor
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let predictor = ONNXPredictor::new_steam_deck(model_path)?;
        
        Ok(Self {
            predictor,
            thermal_state: Arc::new(std::sync::atomic::AtomicU8::new(0)),
        })
    }
    
    /// Predict with thermal awareness
    pub fn predict_thermal_aware(&self, features: &mut ShaderFeatures) -> Result<f32> {
        // Adjust features based on thermal state
        let thermal_state = self.thermal_state.load(std::sync::atomic::Ordering::Relaxed);
        features.thermal_state = thermal_state as f32 / 2.0; // Normalize to 0.0-1.0
        
        // Modify prediction based on thermal constraints
        let base_prediction = self.predictor.predict(features)?;
        
        // Increase prediction if system is hot (compilation will be slower)
        let thermal_multiplier = match thermal_state {
            0 => 1.0,      // Cool - normal performance
            1 => 1.2,      // Warm - slightly slower
            2 => 1.5,      // Hot - significantly slower
            _ => 1.0,
        };
        
        Ok(base_prediction * thermal_multiplier)
    }
    
    /// Update thermal state (called by thermal monitor)
    pub fn update_thermal_state(&self, state: u8) {
        self.thermal_state.store(state.min(2), std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::ShaderFeatures;
    
    #[test]
    fn test_model_config() {
        let config = ModelConfig::steam_deck_optimized();
        assert_eq!(config.intra_threads, Some(2));
        assert!(config.enable_quantization);
        assert!(!config.use_gpu);
    }
    
    #[test]
    fn test_feature_prediction() {
        // This would require an actual model file to test
        // In practice, this would be integration tested
    }
}