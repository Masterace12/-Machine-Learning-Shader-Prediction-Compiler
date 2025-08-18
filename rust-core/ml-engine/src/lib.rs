//! High-performance ML inference engine for shader compilation prediction
//! 
//! This module provides sub-millisecond inference using ONNX Runtime with
//! SIMD optimizations and intelligent caching specifically optimized for Steam Deck.

use anyhow::Result;
use std::sync::Arc;

pub mod features;
pub mod inference;
pub mod cache;
pub mod quantization;
pub mod telemetry;
pub mod predictor;
pub mod heuristic;

// Re-export main types
pub use features::{ShaderFeatures, FeatureExtractor};
pub use inference::{ONNXPredictor, ModelConfig};
pub use cache::{PredictionCache, CachedPrediction};
pub use predictor::{MLPredictor, PredictionResult};
pub use telemetry::{TelemetryCollector, PredictionMetrics};

/// Main ML prediction engine
pub struct MLEngine {
    predictor: Arc<MLPredictor>,
    telemetry: Arc<TelemetryCollector>,
}

impl MLEngine {
    /// Create a new ML engine with default configuration
    pub fn new() -> Result<Self> {
        let predictor = Arc::new(MLPredictor::new_with_defaults()?);
        let telemetry = Arc::new(TelemetryCollector::new());
        
        Ok(Self {
            predictor,
            telemetry,
        })
    }
    
    /// Create ML engine with custom model path
    pub fn with_model<P: AsRef<std::path::Path>>(model_path: P) -> Result<Self> {
        let predictor = Arc::new(MLPredictor::new(model_path)?);
        let telemetry = Arc::new(TelemetryCollector::new());
        
        Ok(Self {
            predictor,
            telemetry,
        })
    }
    
    /// Predict shader compilation time
    pub async fn predict_compilation_time(&self, features: &ShaderFeatures) -> Result<f32> {
        let start = std::time::Instant::now();
        
        let result = self.predictor.predict(features)?;
        
        // Record telemetry
        let latency_ns = start.elapsed().as_nanos() as u64;
        self.telemetry.record_prediction(latency_ns, result.cache_hit);
        
        Ok(result.compilation_time)
    }
    
    /// Batch prediction for multiple shaders
    pub async fn predict_batch(&self, features: &[ShaderFeatures]) -> Result<Vec<f32>> {
        self.predictor.predict_batch(features)
    }
    
    /// Get prediction statistics
    pub fn get_metrics(&self) -> PredictionMetrics {
        self.telemetry.get_metrics()
    }
    
    /// Add feedback for model improvement
    pub fn add_feedback(&self, features: ShaderFeatures, predicted: f32, actual: f32) {
        self.telemetry.add_feedback(features, predicted, actual);
    }
}

/// Error types for the ML engine
#[derive(thiserror::Error, Debug)]
pub enum MLEngineError {
    #[error("ONNX Runtime error: {0}")]
    OnnxError(String),
    
    #[error("Feature extraction error: {0}")]
    FeatureError(String),
    
    #[error("Model loading error: {0}")]
    ModelError(String),
    
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
}