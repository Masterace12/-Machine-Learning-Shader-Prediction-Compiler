//! Main ML predictor combining ONNX inference with caching and fallbacks

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use std::path::Path;
use crate::features::ShaderFeatures;
use crate::inference::{ONNXPredictor, ModelConfig};
use crate::cache::{PredictionCache, CachedPrediction};
use crate::heuristic::HeuristicPredictor;
use crate::telemetry::TelemetryCollector;

/// Main ML predictor with intelligent fallbacks
pub struct MLPredictor {
    onnx_predictor: ONNXPredictor,
    cache: PredictionCache,
    heuristic_fallback: HeuristicPredictor,
    telemetry: Arc<TelemetryCollector>,
}

/// Prediction result with metadata
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub compilation_time: f32,
    pub confidence: f32,
    pub cache_hit: bool,
    pub fallback_used: bool,
    pub inference_time_ns: u64,
}

impl MLPredictor {
    /// Create predictor with default model path
    pub fn new_with_defaults() -> Result<Self> {
        let model_path = std::env::var("SHADER_MODEL_PATH")
            .unwrap_or_else(|_| "/home/deck/.local/share/shader-predict-compile/models/shader_predictor.onnx".to_string());
        Self::new(model_path)
    }
    
    /// Create predictor with specific model path
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let config = if is_steam_deck() {
            ModelConfig::steam_deck_optimized()
        } else {
            ModelConfig::default()
        };
        
        let onnx_predictor = ONNXPredictor::new_with_config(model_path, config)?;
        
        // Warm up the model
        onnx_predictor.warmup()?;
        
        Ok(Self {
            onnx_predictor,
            cache: PredictionCache::new(1000, 5000), // 1k hot, 5k warm
            heuristic_fallback: HeuristicPredictor::new(),
            telemetry: Arc::new(TelemetryCollector::new()),
        })
    }
    
    /// Predict compilation time with full pipeline
    #[inline]
    pub fn predict(&self, features: &ShaderFeatures) -> Result<PredictionResult> {
        let start = Instant::now();
        
        // Check cache first
        if let Some(cached) = self.cache.get(features) {
            return Ok(PredictionResult {
                compilation_time: cached.compilation_time,
                confidence: cached.confidence,
                cache_hit: true,
                fallback_used: false,
                inference_time_ns: start.elapsed().as_nanos() as u64,
            });
        }
        
        // Try ONNX inference
        let (prediction, fallback_used) = match self.onnx_predictor.predict(features) {
            Ok(pred) => (pred, false),
            Err(_) => {
                // Fallback to heuristic
                let heuristic_pred = self.heuristic_fallback.predict(features);
                (heuristic_pred, true)
            }
        };
        
        let inference_time_ns = start.elapsed().as_nanos() as u64;
        
        // Calculate confidence based on prediction method and feature complexity
        let confidence = if fallback_used {
            0.6 // Lower confidence for heuristic fallback
        } else {
            self.calculate_confidence(features, prediction)
        };
        
        // Cache the result
        let cached_prediction = CachedPrediction {
            compilation_time: prediction,
            confidence,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            access_count: 1,
        };
        
        self.cache.insert(features, cached_prediction);
        
        Ok(PredictionResult {
            compilation_time: prediction,
            confidence,
            cache_hit: false,
            fallback_used,
            inference_time_ns,
        })
    }
    
    /// Batch prediction for multiple shaders
    pub fn predict_batch(&self, features: &[ShaderFeatures]) -> Result<Vec<f32>> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut results = Vec::with_capacity(features.len());
        let mut uncached_features = Vec::new();
        let mut uncached_indices = Vec::new();
        
        // Check cache for each feature set
        for (i, feature) in features.iter().enumerate() {
            if let Some(cached) = self.cache.get(feature) {
                results.push(Some(cached.compilation_time));
            } else {
                results.push(None);
                uncached_features.push(feature.clone());
                uncached_indices.push(i);
            }
        }
        
        // Batch predict uncached items
        if !uncached_features.is_empty() {
            let batch_predictions = self.onnx_predictor.predict_batch(&uncached_features)?;
            
            // Fill in results and cache predictions
            for (pred_idx, &result_idx) in uncached_indices.iter().enumerate() {
                let prediction = batch_predictions[pred_idx];
                results[result_idx] = Some(prediction);
                
                // Cache the result
                let cached_prediction = CachedPrediction {
                    compilation_time: prediction,
                    confidence: self.calculate_confidence(&uncached_features[pred_idx], prediction),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    access_count: 1,
                };
                
                self.cache.insert(&uncached_features[pred_idx], cached_prediction);
            }
        }
        
        // Convert Option<f32> to f32, this should never panic due to logic above
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
    
    /// Get prediction statistics
    pub fn get_stats(&self) -> PredictorStats {
        let cache_stats = self.cache.get_stats();
        
        PredictorStats {
            cache_hit_rate: cache_stats.hit_rate,
            total_predictions: cache_stats.total_lookups,
            cache_size: cache_stats.total_entries,
            avg_inference_time_ns: self.telemetry.get_avg_inference_time(),
            fallback_usage_rate: self.telemetry.get_fallback_rate(),
        }
    }
    
    /// Add feedback for model improvement
    pub fn add_feedback(&self, features: ShaderFeatures, predicted: f32, actual: f32) {
        self.telemetry.add_feedback(features, predicted, actual);
    }
    
    /// Calculate prediction confidence based on features and result
    fn calculate_confidence(&self, features: &ShaderFeatures, prediction: f32) -> f32 {
        // Base confidence starts high for ONNX predictions
        let mut confidence = 0.9;
        
        // Reduce confidence for very complex shaders (harder to predict)
        let complexity = features.complexity_score();
        if complexity > 100.0 {
            confidence *= 0.8;
        } else if complexity > 50.0 {
            confidence *= 0.9;
        }
        
        // Reduce confidence for extreme predictions
        if prediction < 1.0 || prediction > 5000.0 {
            confidence *= 0.7;
        }
        
        // Factor in thermal state uncertainty
        if features.thermal_state > 0.8 {
            confidence *= 0.85; // Less confident when system is hot
        }
        
        confidence.clamp(0.1, 1.0)
    }
}

/// Predictor performance statistics
#[derive(Debug, Clone)]
pub struct PredictorStats {
    pub cache_hit_rate: f64,
    pub total_predictions: u64,
    pub cache_size: usize,
    pub avg_inference_time_ns: u64,
    pub fallback_usage_rate: f64,
}

/// Check if running on Steam Deck
fn is_steam_deck() -> bool {
    // Check for Steam Deck specific indicators
    std::fs::read_to_string("/sys/devices/virtual/dmi/id/product_name")
        .map(|s| s.trim() == "Jupiter" || s.trim() == "Galileo")
        .unwrap_or(false)
    ||
    std::env::var("SteamDeck").is_ok()
    ||
    std::path::Path::new("/home/deck").exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_steam_deck_detection() {
        // This will depend on the test environment
        let _is_deck = is_steam_deck();
    }
    
    #[test]
    fn test_confidence_calculation() {
        let predictor = MLPredictor::new_with_defaults().unwrap();
        let features = ShaderFeatures::default();
        let confidence = predictor.calculate_confidence(&features, 50.0);
        assert!(confidence > 0.0 && confidence <= 1.0);
    }
}