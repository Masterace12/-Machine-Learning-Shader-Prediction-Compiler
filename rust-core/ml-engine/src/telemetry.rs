//! Telemetry collection for ML model improvement and monitoring

use crate::features::ShaderFeatures;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

/// Telemetry collector for prediction metrics and feedback
pub struct TelemetryCollector {
    metrics: Arc<PredictionMetrics>,
    feedback_buffer: Arc<Mutex<VecDeque<FeedbackData>>>,
    max_feedback_entries: usize,
}

/// Real-time prediction metrics
pub struct PredictionMetrics {
    pub total_predictions: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub fallback_usage: AtomicU64,
    pub total_inference_time_ns: AtomicU64,
    pub min_inference_time_ns: AtomicU64,
    pub max_inference_time_ns: AtomicU64,
    pub error_count: AtomicU64,
}

/// Feedback data for model improvement
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FeedbackData {
    pub features: Vec<f32>,
    pub predicted_time: f32,
    pub actual_time: f32,
    pub prediction_error: f32,
    pub timestamp: u64,
    pub confidence: f32,
    pub cache_hit: bool,
    pub fallback_used: bool,
}

/// Aggregated statistics for monitoring
#[derive(Debug, Clone)]
pub struct AggregatedStats {
    pub total_predictions: u64,
    pub cache_hit_rate: f64,
    pub fallback_rate: f64,
    pub avg_inference_time_ns: u64,
    pub min_inference_time_ns: u64,
    pub max_inference_time_ns: u64,
    pub prediction_accuracy: f64,
    pub avg_prediction_error: f64,
    pub error_rate: f64,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(PredictionMetrics::new()),
            feedback_buffer: Arc::new(Mutex::new(VecDeque::new())),
            max_feedback_entries: 10000, // Keep last 10k entries
        }
    }
    
    /// Record a prediction event
    pub fn record_prediction(&self, inference_time_ns: u64, cache_hit: bool) {
        self.metrics.total_predictions.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_inference_time_ns.fetch_add(inference_time_ns, Ordering::Relaxed);
        
        if cache_hit {
            self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update min/max inference times
        self.update_min_time(inference_time_ns);
        self.update_max_time(inference_time_ns);
    }
    
    /// Record fallback usage
    pub fn record_fallback(&self) {
        self.metrics.fallback_usage.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record an error
    pub fn record_error(&self) {
        self.metrics.error_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Add feedback data for model improvement
    pub fn add_feedback(&self, features: ShaderFeatures, predicted: f32, actual: f32) {
        let prediction_error = (predicted - actual).abs();
        let relative_error = if actual > 0.0 {
            prediction_error / actual
        } else {
            prediction_error
        };
        
        let feedback = FeedbackData {
            features: self.features_to_vec(&features),
            predicted_time: predicted,
            actual_time: actual,
            prediction_error: relative_error,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            confidence: 0.0, // Would be filled by caller
            cache_hit: false, // Would be filled by caller
            fallback_used: false, // Would be filled by caller
        };
        
        let mut buffer = self.feedback_buffer.lock().unwrap();
        buffer.push_back(feedback);
        
        // Maintain buffer size limit
        while buffer.len() > self.max_feedback_entries {
            buffer.pop_front();
        }
        
        // Trigger model retraining if enough new feedback
        if buffer.len() % 1000 == 0 {
            self.trigger_model_update(&buffer);
        }
    }
    
    /// Get current prediction metrics
    pub fn get_metrics(&self) -> AggregatedStats {
        let total_predictions = self.metrics.total_predictions.load(Ordering::Relaxed);
        let cache_hits = self.metrics.cache_hits.load(Ordering::Relaxed);
        let fallback_usage = self.metrics.fallback_usage.load(Ordering::Relaxed);
        let total_inference_time = self.metrics.total_inference_time_ns.load(Ordering::Relaxed);
        let error_count = self.metrics.error_count.load(Ordering::Relaxed);
        
        let cache_hit_rate = if total_predictions > 0 {
            cache_hits as f64 / total_predictions as f64
        } else {
            0.0
        };
        
        let fallback_rate = if total_predictions > 0 {
            fallback_usage as f64 / total_predictions as f64
        } else {
            0.0
        };
        
        let avg_inference_time = if total_predictions > 0 {
            total_inference_time / total_predictions
        } else {
            0
        };
        
        let error_rate = if total_predictions > 0 {
            error_count as f64 / total_predictions as f64
        } else {
            0.0
        };
        
        // Calculate prediction accuracy from feedback
        let (prediction_accuracy, avg_error) = self.calculate_prediction_accuracy();
        
        AggregatedStats {
            total_predictions,
            cache_hit_rate,
            fallback_rate,
            avg_inference_time_ns: avg_inference_time,
            min_inference_time_ns: self.metrics.min_inference_time_ns.load(Ordering::Relaxed),
            max_inference_time_ns: self.metrics.max_inference_time_ns.load(Ordering::Relaxed),
            prediction_accuracy,
            avg_prediction_error: avg_error,
            error_rate,
        }
    }
    
    /// Get average inference time
    pub fn get_avg_inference_time(&self) -> u64 {
        let total_predictions = self.metrics.total_predictions.load(Ordering::Relaxed);
        let total_time = self.metrics.total_inference_time_ns.load(Ordering::Relaxed);
        
        if total_predictions > 0 {
            total_time / total_predictions
        } else {
            0
        }
    }
    
    /// Get fallback usage rate
    pub fn get_fallback_rate(&self) -> f64 {
        let total_predictions = self.metrics.total_predictions.load(Ordering::Relaxed);
        let fallback_usage = self.metrics.fallback_usage.load(Ordering::Relaxed);
        
        if total_predictions > 0 {
            fallback_usage as f64 / total_predictions as f64
        } else {
            0.0
        }
    }
    
    /// Export feedback data for model retraining
    pub fn export_feedback_data(&self) -> Vec<FeedbackData> {
        self.feedback_buffer.lock().unwrap().iter().cloned().collect()
    }
    
    /// Clear feedback buffer
    pub fn clear_feedback(&self) {
        self.feedback_buffer.lock().unwrap().clear();
    }
    
    /// Get real-time performance summary
    pub fn get_performance_summary(&self) -> String {
        let stats = self.get_metrics();
        
        format!(
            "Predictions: {}, Cache Hit Rate: {:.1}%, Fallback Rate: {:.1}%, \
             Avg Inference: {:.1}μs, Accuracy: {:.1}%, Error Rate: {:.2}%",
            stats.total_predictions,
            stats.cache_hit_rate * 100.0,
            stats.fallback_rate * 100.0,
            stats.avg_inference_time_ns as f64 / 1000.0,
            stats.prediction_accuracy * 100.0,
            stats.error_rate * 100.0
        )
    }
    
    /// Update minimum inference time
    fn update_min_time(&self, time_ns: u64) {
        let mut current_min = self.metrics.min_inference_time_ns.load(Ordering::Relaxed);
        while current_min == 0 || time_ns < current_min {
            match self.metrics.min_inference_time_ns.compare_exchange_weak(
                current_min,
                time_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_current) => current_min = new_current,
            }
        }
    }
    
    /// Update maximum inference time
    fn update_max_time(&self, time_ns: u64) {
        let mut current_max = self.metrics.max_inference_time_ns.load(Ordering::Relaxed);
        while time_ns > current_max {
            match self.metrics.max_inference_time_ns.compare_exchange_weak(
                current_max,
                time_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_current) => current_max = new_current,
            }
        }
    }
    
    /// Convert shader features to vector for serialization
    fn features_to_vec(&self, features: &ShaderFeatures) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(features as *const _ as *const f32, 16).to_vec()
        }
    }
    
    /// Calculate prediction accuracy from feedback data
    fn calculate_prediction_accuracy(&self) -> (f64, f64) {
        let buffer = self.feedback_buffer.lock().unwrap();
        
        if buffer.is_empty() {
            return (0.0, 0.0);
        }
        
        let mut total_error = 0.0;
        let mut accurate_predictions = 0;
        
        for feedback in buffer.iter() {
            total_error += feedback.prediction_error as f64;
            
            // Consider prediction accurate if within 20% of actual time
            if feedback.prediction_error < 0.2 {
                accurate_predictions += 1;
            }
        }
        
        let accuracy = accurate_predictions as f64 / buffer.len() as f64;
        let avg_error = total_error / buffer.len() as f64;
        
        (accuracy, avg_error)
    }
    
    /// Trigger model update with accumulated feedback
    fn trigger_model_update(&self, feedback_data: &VecDeque<FeedbackData>) {
        // In a real implementation, this would:
        // 1. Export feedback data to training format
        // 2. Trigger incremental model retraining
        // 3. Deploy updated model
        
        tracing::info!(
            "Model update triggered with {} feedback entries",
            feedback_data.len()
        );
        
        // For now, just log the trigger
        // TODO: Implement actual model retraining pipeline
    }
}

impl PredictionMetrics {
    /// Create new prediction metrics
    fn new() -> Self {
        Self {
            total_predictions: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            fallback_usage: AtomicU64::new(0),
            total_inference_time_ns: AtomicU64::new(0),
            min_inference_time_ns: AtomicU64::new(0),
            max_inference_time_ns: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::ShaderFeatures;
    
    #[test]
    fn test_telemetry_basic_operations() {
        let collector = TelemetryCollector::new();
        
        // Record some predictions
        collector.record_prediction(1000, true);  // 1μs, cache hit
        collector.record_prediction(5000, false); // 5μs, cache miss
        collector.record_fallback();
        
        let stats = collector.get_metrics();
        assert_eq!(stats.total_predictions, 2);
        assert_eq!(stats.cache_hit_rate, 0.5);
        assert_eq!(stats.fallback_rate, 0.5);
    }
    
    #[test]
    fn test_feedback_collection() {
        let collector = TelemetryCollector::new();
        let features = ShaderFeatures::default();
        
        collector.add_feedback(features, 50.0, 45.0);
        
        let feedback_data = collector.export_feedback_data();
        assert_eq!(feedback_data.len(), 1);
        assert_eq!(feedback_data[0].predicted_time, 50.0);
        assert_eq!(feedback_data[0].actual_time, 45.0);
    }
    
    #[test]
    fn test_prediction_accuracy() {
        let collector = TelemetryCollector::new();
        let features = ShaderFeatures::default();
        
        // Add accurate predictions
        collector.add_feedback(features.clone(), 100.0, 95.0);  // 5% error
        collector.add_feedback(features.clone(), 200.0, 210.0); // 5% error
        
        // Add inaccurate prediction
        collector.add_feedback(features, 100.0, 150.0); // 50% error
        
        let stats = collector.get_metrics();
        assert!(stats.prediction_accuracy > 0.6); // 2/3 accurate
        assert!(stats.avg_prediction_error > 0.0);
    }
}