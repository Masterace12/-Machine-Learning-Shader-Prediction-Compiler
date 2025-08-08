#!/usr/bin/env python3
"""
ML Metrics Collection Module
Collects data for improving machine learning prediction models
"""

import os
import asyncio
import logging
import json
import hashlib
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import pickle

@dataclass
class ShaderPredictionMetric:
    """Individual shader prediction metric"""
    shader_hash: str
    predicted_compilation: bool
    actual_compilation: bool
    prediction_confidence: float
    compilation_time: float
    game_context: str
    timestamp: float

@dataclass
class GameSessionMetrics:
    """Aggregated metrics for a game session"""
    app_id: str
    session_id: str
    total_shaders: int
    predicted_shaders: int
    actual_compilations: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    session_duration: float

class MLMetricsCollector:
    """Collects metrics for ML model improvement"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}")
        self.telemetry_enabled = config.get("telemetry", {}).get("collect_ml_data", True)
        self.local_storage_days = config.get("telemetry", {}).get("local_storage_days", 30)
        self.db_path = "data/telemetry/ml_metrics.db"
        self.session_metrics = []
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS shader_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        shader_hash TEXT NOT NULL,
                        app_id TEXT NOT NULL,
                        predicted_compilation INTEGER NOT NULL,
                        actual_compilation INTEGER NOT NULL,
                        prediction_confidence REAL,
                        compilation_time REAL,
                        game_context TEXT,
                        timestamp REAL,
                        session_id TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS session_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        app_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        total_shaders INTEGER,
                        predicted_shaders INTEGER,
                        actual_compilations INTEGER,
                        true_positives INTEGER,
                        false_positives INTEGER,
                        true_negatives INTEGER,
                        false_negatives INTEGER,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        session_duration REAL,
                        timestamp REAL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feature_importance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feature_name TEXT NOT NULL,
                        importance_score REAL,
                        model_version TEXT,
                        timestamp REAL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version TEXT NOT NULL,
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        training_samples INTEGER,
                        validation_samples INTEGER,
                        timestamp REAL
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_shader_hash ON shader_predictions(shader_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_app_id ON shader_predictions(app_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON shader_predictions(session_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON shader_predictions(timestamp)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics database: {e}")
    
    async def collect_prediction_metrics(self, app_id: str) -> Dict[str, Any]:
        """Collect ML prediction metrics for a specific game"""
        if not self.telemetry_enabled:
            return {"telemetry_disabled": True}
        
        self.logger.info(f"Collecting ML prediction metrics for app {app_id}")
        
        collection_result = {
            "app_id": app_id,
            "session_id": self._generate_session_id(),
            "collection_timestamp": time.time(),
            "shader_predictions": [],
            "aggregated_metrics": {},
            "model_performance": {},
            "feature_analysis": {},
            "data_quality": {}
        }
        
        try:
            # 1. Collect shader prediction data
            shader_metrics = await self._collect_shader_prediction_data(app_id, collection_result["session_id"])
            collection_result["shader_predictions"] = shader_metrics
            
            # 2. Calculate aggregated metrics
            aggregated = await self._calculate_aggregated_metrics(shader_metrics, app_id, collection_result["session_id"])
            collection_result["aggregated_metrics"] = aggregated
            
            # 3. Analyze model performance
            model_perf = await self._analyze_model_performance(shader_metrics)
            collection_result["model_performance"] = model_perf
            
            # 4. Feature importance analysis
            feature_analysis = await self._analyze_feature_importance(app_id, shader_metrics)
            collection_result["feature_analysis"] = feature_analysis
            
            # 5. Data quality assessment
            data_quality = await self._assess_data_quality(shader_metrics)
            collection_result["data_quality"] = data_quality
            
            # 6. Store metrics in database
            await self._store_metrics_in_database(collection_result)
            
            # 7. Generate improvement recommendations
            recommendations = await self._generate_ml_recommendations(collection_result)
            collection_result["recommendations"] = recommendations
            
        except Exception as e:
            self.logger.error(f"ML metrics collection failed: {e}")
            collection_result["error"] = str(e)
        
        return collection_result
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        timestamp = str(time.time())
        hash_obj = hashlib.md5(timestamp.encode())
        return hash_obj.hexdigest()[:16]
    
    async def _collect_shader_prediction_data(self, app_id: str, session_id: str) -> List[ShaderPredictionMetric]:
        """Collect shader prediction vs actual compilation data"""
        shader_metrics = []
        
        try:
            # Get cache directory for the game
            cache_dir = f"{self.config['steam_deck']['cache_directory']}/{app_id}"
            
            if not os.path.exists(cache_dir):
                self.logger.warning(f"Cache directory not found for app {app_id}")
                return shader_metrics
            
            # Read prediction logs if available
            prediction_log = f"{cache_dir}/prediction_log.json"
            if os.path.exists(prediction_log):
                with open(prediction_log, 'r') as f:
                    prediction_data = json.load(f)
                
                for entry in prediction_data.get("predictions", []):
                    metric = ShaderPredictionMetric(
                        shader_hash=entry.get("shader_hash", "unknown"),
                        predicted_compilation=entry.get("predicted_compilation", False),
                        actual_compilation=entry.get("actual_compilation", False),
                        prediction_confidence=entry.get("confidence", 0.0),
                        compilation_time=entry.get("compilation_time", 0.0),
                        game_context=entry.get("context", "unknown"),
                        timestamp=entry.get("timestamp", time.time())
                    )
                    shader_metrics.append(metric)
            
            # If no prediction log, generate synthetic data based on cache files
            else:
                synthetic_metrics = await self._generate_synthetic_prediction_data(cache_dir, app_id)
                shader_metrics.extend(synthetic_metrics)
        
        except Exception as e:
            self.logger.error(f"Error collecting shader prediction data: {e}")
        
        return shader_metrics
    
    async def _generate_synthetic_prediction_data(self, cache_dir: str, app_id: str) -> List[ShaderPredictionMetric]:
        """Generate synthetic prediction data from cache analysis"""
        synthetic_metrics = []
        
        try:
            # Analyze cache files to estimate prediction accuracy
            cache_files = []
            for root, dirs, files in os.walk(cache_dir):
                cache_files.extend([os.path.join(root, f) for f in files])
            
            # For each cache file, create synthetic prediction metrics
            for i, cache_file in enumerate(cache_files[:100]):  # Limit to 100 files
                file_stat = os.stat(cache_file)
                file_size = file_stat.st_size
                
                # Generate hash from file path
                shader_hash = hashlib.md5(cache_file.encode()).hexdigest()
                
                # Simulate prediction based on file characteristics
                predicted_compilation = file_size > 1024  # Predict compilation for larger files
                actual_compilation = os.path.exists(cache_file)  # File exists = was compiled
                
                # Simulate confidence based on file size and age
                confidence = min(0.95, max(0.1, file_size / 10000))
                
                metric = ShaderPredictionMetric(
                    shader_hash=shader_hash,
                    predicted_compilation=predicted_compilation,
                    actual_compilation=actual_compilation,
                    prediction_confidence=confidence,
                    compilation_time=file_size / 1000,  # Estimate compilation time
                    game_context="synthetic",
                    timestamp=file_stat.st_mtime
                )
                synthetic_metrics.append(metric)
        
        except Exception as e:
            self.logger.error(f"Error generating synthetic prediction data: {e}")
        
        return synthetic_metrics
    
    async def _calculate_aggregated_metrics(self, shader_metrics: List[ShaderPredictionMetric], app_id: str, session_id: str) -> GameSessionMetrics:
        """Calculate aggregated metrics for the session"""
        if not shader_metrics:
            return GameSessionMetrics(
                app_id=app_id,
                session_id=session_id,
                total_shaders=0,
                predicted_shaders=0,
                actual_compilations=0,
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                session_duration=0.0
            )
        
        try:
            total_shaders = len(shader_metrics)
            predicted_shaders = sum(1 for m in shader_metrics if m.predicted_compilation)
            actual_compilations = sum(1 for m in shader_metrics if m.actual_compilation)
            
            # Calculate confusion matrix
            true_positives = sum(1 for m in shader_metrics 
                               if m.predicted_compilation and m.actual_compilation)
            false_positives = sum(1 for m in shader_metrics 
                                if m.predicted_compilation and not m.actual_compilation)
            true_negatives = sum(1 for m in shader_metrics 
                               if not m.predicted_compilation and not m.actual_compilation)
            false_negatives = sum(1 for m in shader_metrics 
                                if not m.predicted_compilation and m.actual_compilation)
            
            # Calculate performance metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate session duration
            timestamps = [m.timestamp for m in shader_metrics if m.timestamp > 0]
            session_duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
            
            return GameSessionMetrics(
                app_id=app_id,
                session_id=session_id,
                total_shaders=total_shaders,
                predicted_shaders=predicted_shaders,
                actual_compilations=actual_compilations,
                true_positives=true_positives,
                false_positives=false_positives,
                true_negatives=true_negatives,
                false_negatives=false_negatives,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                session_duration=session_duration
            )
        
        except Exception as e:
            self.logger.error(f"Error calculating aggregated metrics: {e}")
            return GameSessionMetrics(
                app_id=app_id,
                session_id=session_id,
                total_shaders=0,
                predicted_shaders=0,
                actual_compilations=0,
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                session_duration=0.0
            )
    
    async def _analyze_model_performance(self, shader_metrics: List[ShaderPredictionMetric]) -> Dict[str, Any]:
        """Analyze ML model performance characteristics"""
        performance_analysis = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "confidence_calibration": {},
            "prediction_distribution": {},
            "error_analysis": {},
            "temporal_analysis": {}
        }
        
        try:
            if not shader_metrics:
                return performance_analysis
            
            # Calculate accuracy
            correct_predictions = sum(1 for m in shader_metrics 
                                    if m.predicted_compilation == m.actual_compilation)
            total_predictions = len(shader_metrics)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            performance_analysis["accuracy"] = accuracy
            
            # Calculate balanced accuracy
            tp = sum(1 for m in shader_metrics if m.predicted_compilation and m.actual_compilation)
            tn = sum(1 for m in shader_metrics if not m.predicted_compilation and not m.actual_compilation)
            fp = sum(1 for m in shader_metrics if m.predicted_compilation and not m.actual_compilation)
            fn = sum(1 for m in shader_metrics if not m.predicted_compilation and m.actual_compilation)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            balanced_accuracy = (sensitivity + specificity) / 2
            performance_analysis["balanced_accuracy"] = balanced_accuracy
            
            # Confidence calibration analysis
            confidence_analysis = await self._analyze_confidence_calibration(shader_metrics)
            performance_analysis["confidence_calibration"] = confidence_analysis
            
            # Prediction distribution
            predicted_true = sum(1 for m in shader_metrics if m.predicted_compilation)
            predicted_false = total_predictions - predicted_true
            performance_analysis["prediction_distribution"] = {
                "predicted_true": predicted_true,
                "predicted_false": predicted_false,
                "prediction_bias": (predicted_true / total_predictions) if total_predictions > 0 else 0.0
            }
            
            # Error analysis
            error_analysis = await self._analyze_prediction_errors(shader_metrics)
            performance_analysis["error_analysis"] = error_analysis
            
            # Temporal analysis
            temporal_analysis = await self._analyze_temporal_patterns(shader_metrics)
            performance_analysis["temporal_analysis"] = temporal_analysis
        
        except Exception as e:
            self.logger.error(f"Error analyzing model performance: {e}")
        
        return performance_analysis
    
    async def _analyze_confidence_calibration(self, shader_metrics: List[ShaderPredictionMetric]) -> Dict[str, Any]:
        """Analyze how well prediction confidence correlates with actual accuracy"""
        calibration_analysis = {
            "confidence_bins": {},
            "calibration_error": 0.0,
            "reliability_diagram": {},
            "overconfidence_score": 0.0
        }
        
        try:
            # Create confidence bins
            bins = [(i/10, (i+1)/10) for i in range(10)]  # 10 bins: [0-0.1), [0.1-0.2), ..., [0.9-1.0]
            
            for bin_min, bin_max in bins:
                bin_key = f"{bin_min:.1f}-{bin_max:.1f}"
                bin_metrics = [m for m in shader_metrics 
                             if bin_min <= m.prediction_confidence < bin_max]
                
                if bin_metrics:
                    accuracy_in_bin = sum(1 for m in bin_metrics 
                                        if m.predicted_compilation == m.actual_compilation) / len(bin_metrics)
                    avg_confidence = sum(m.prediction_confidence for m in bin_metrics) / len(bin_metrics)
                    
                    calibration_analysis["confidence_bins"][bin_key] = {
                        "count": len(bin_metrics),
                        "accuracy": accuracy_in_bin,
                        "avg_confidence": avg_confidence,
                        "calibration_gap": abs(avg_confidence - accuracy_in_bin)
                    }
            
            # Calculate expected calibration error (ECE)
            total_samples = len(shader_metrics)
            ece = 0.0
            for bin_data in calibration_analysis["confidence_bins"].values():
                bin_weight = bin_data["count"] / total_samples
                ece += bin_weight * bin_data["calibration_gap"]
            
            calibration_analysis["calibration_error"] = ece
            
            # Calculate overconfidence score
            overconfident_predictions = sum(1 for m in shader_metrics 
                                          if m.prediction_confidence > 0.8 and 
                                          m.predicted_compilation != m.actual_compilation)
            overconfidence_score = overconfident_predictions / total_samples if total_samples > 0 else 0.0
            calibration_analysis["overconfidence_score"] = overconfidence_score
        
        except Exception as e:
            self.logger.error(f"Error analyzing confidence calibration: {e}")
        
        return calibration_analysis
    
    async def _analyze_prediction_errors(self, shader_metrics: List[ShaderPredictionMetric]) -> Dict[str, Any]:
        """Analyze patterns in prediction errors"""
        error_analysis = {
            "false_positive_patterns": {},
            "false_negative_patterns": {},
            "high_confidence_errors": [],
            "context_based_errors": {}
        }
        
        try:
            false_positives = [m for m in shader_metrics 
                             if m.predicted_compilation and not m.actual_compilation]
            false_negatives = [m for m in shader_metrics 
                             if not m.predicted_compilation and m.actual_compilation]
            
            # Analyze false positives
            if false_positives:
                fp_confidence_avg = sum(m.prediction_confidence for m in false_positives) / len(false_positives)
                error_analysis["false_positive_patterns"] = {
                    "count": len(false_positives),
                    "avg_confidence": fp_confidence_avg,
                    "contexts": [m.game_context for m in false_positives]
                }
            
            # Analyze false negatives
            if false_negatives:
                fn_confidence_avg = sum(m.prediction_confidence for m in false_negatives) / len(false_negatives)
                error_analysis["false_negative_patterns"] = {
                    "count": len(false_negatives),
                    "avg_confidence": fn_confidence_avg,
                    "contexts": [m.game_context for m in false_negatives]
                }
            
            # High confidence errors
            high_conf_errors = [m for m in shader_metrics 
                              if m.prediction_confidence > 0.8 and 
                              m.predicted_compilation != m.actual_compilation]
            error_analysis["high_confidence_errors"] = len(high_conf_errors)
            
            # Context-based error analysis
            context_errors = {}
            for metric in shader_metrics:
                context = metric.game_context
                if context not in context_errors:
                    context_errors[context] = {"total": 0, "errors": 0}
                
                context_errors[context]["total"] += 1
                if metric.predicted_compilation != metric.actual_compilation:
                    context_errors[context]["errors"] += 1
            
            # Calculate error rates by context
            for context, data in context_errors.items():
                if data["total"] > 0:
                    data["error_rate"] = data["errors"] / data["total"]
            
            error_analysis["context_based_errors"] = context_errors
        
        except Exception as e:
            self.logger.error(f"Error analyzing prediction errors: {e}")
        
        return error_analysis
    
    async def _analyze_temporal_patterns(self, shader_metrics: List[ShaderPredictionMetric]) -> Dict[str, Any]:
        """Analyze temporal patterns in predictions"""
        temporal_analysis = {
            "accuracy_over_time": {},
            "compilation_time_patterns": {},
            "prediction_drift": {}
        }
        
        try:
            # Sort metrics by timestamp
            sorted_metrics = sorted([m for m in shader_metrics if m.timestamp > 0], 
                                  key=lambda x: x.timestamp)
            
            if not sorted_metrics:
                return temporal_analysis
            
            # Divide into time windows
            start_time = sorted_metrics[0].timestamp
            end_time = sorted_metrics[-1].timestamp
            window_size = (end_time - start_time) / 10  # 10 windows
            
            for i in range(10):
                window_start = start_time + i * window_size
                window_end = start_time + (i + 1) * window_size
                
                window_metrics = [m for m in sorted_metrics 
                                if window_start <= m.timestamp < window_end]
                
                if window_metrics:
                    correct = sum(1 for m in window_metrics 
                                if m.predicted_compilation == m.actual_compilation)
                    accuracy = correct / len(window_metrics)
                    
                    temporal_analysis["accuracy_over_time"][f"window_{i}"] = {
                        "start_time": window_start,
                        "end_time": window_end,
                        "sample_count": len(window_metrics),
                        "accuracy": accuracy
                    }
            
            # Analyze compilation time patterns
            comp_times = [m.compilation_time for m in sorted_metrics if m.compilation_time > 0]
            if comp_times:
                temporal_analysis["compilation_time_patterns"] = {
                    "mean_time": np.mean(comp_times),
                    "std_time": np.std(comp_times),
                    "trend": self._calculate_trend(comp_times)
                }
        
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {e}")
        
        return temporal_analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction in a list of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        slope = coefficients[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    async def _analyze_feature_importance(self, app_id: str, shader_metrics: List[ShaderPredictionMetric]) -> Dict[str, Any]:
        """Analyze feature importance for prediction model"""
        feature_analysis = {
            "feature_correlations": {},
            "most_predictive_features": [],
            "feature_stability": {},
            "recommendations": []
        }
        
        try:
            # Simulate feature importance analysis
            # In a real implementation, this would analyze actual model features
            simulated_features = {
                "shader_size": 0.25,
                "compilation_history": 0.20,
                "game_context": 0.18,
                "driver_version": 0.15,
                "hardware_capabilities": 0.12,
                "runtime_patterns": 0.10
            }
            
            feature_analysis["feature_correlations"] = simulated_features
            
            # Most predictive features
            sorted_features = sorted(simulated_features.items(), key=lambda x: x[1], reverse=True)
            feature_analysis["most_predictive_features"] = sorted_features[:3]
            
            # Feature stability analysis (how consistent features are across sessions)
            for feature, importance in simulated_features.items():
                # Simulate stability score
                stability = max(0.5, importance + np.random.normal(0, 0.1))
                feature_analysis["feature_stability"][feature] = min(1.0, stability)
            
            # Generate recommendations
            recommendations = []
            if simulated_features.get("shader_size", 0) > 0.3:
                recommendations.append("Consider optimizing shader size-based predictions")
            if simulated_features.get("compilation_history", 0) < 0.1:
                recommendations.append("Improve compilation history tracking for better predictions")
            
            feature_analysis["recommendations"] = recommendations
        
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {e}")
        
        return feature_analysis
    
    async def _assess_data_quality(self, shader_metrics: List[ShaderPredictionMetric]) -> Dict[str, Any]:
        """Assess quality of collected data"""
        quality_assessment = {
            "completeness": {},
            "consistency": {},
            "validity": {},
            "overall_score": 0.0,
            "issues": []
        }
        
        try:
            total_metrics = len(shader_metrics)
            if total_metrics == 0:
                quality_assessment["overall_score"] = 0.0
                quality_assessment["issues"].append("No metrics collected")
                return quality_assessment
            
            # Completeness assessment
            complete_metrics = sum(1 for m in shader_metrics 
                                 if all([m.shader_hash, m.prediction_confidence > 0, m.timestamp > 0]))
            completeness_score = complete_metrics / total_metrics
            quality_assessment["completeness"] = {
                "score": completeness_score,
                "complete_records": complete_metrics,
                "total_records": total_metrics
            }
            
            # Consistency assessment
            consistent_predictions = sum(1 for m in shader_metrics 
                                       if 0 <= m.prediction_confidence <= 1)
            consistency_score = consistent_predictions / total_metrics
            quality_assessment["consistency"] = {
                "score": consistency_score,
                "valid_confidences": consistent_predictions
            }
            
            # Validity assessment
            valid_timestamps = sum(1 for m in shader_metrics 
                                 if m.timestamp > 1000000000)  # Valid unix timestamp
            valid_hashes = sum(1 for m in shader_metrics 
                             if len(m.shader_hash) >= 8)  # Reasonable hash length
            validity_score = (valid_timestamps + valid_hashes) / (2 * total_metrics)
            quality_assessment["validity"] = {
                "score": validity_score,
                "valid_timestamps": valid_timestamps,
                "valid_hashes": valid_hashes
            }
            
            # Overall quality score
            overall_score = (completeness_score + consistency_score + validity_score) / 3
            quality_assessment["overall_score"] = overall_score
            
            # Identify issues
            if completeness_score < 0.8:
                quality_assessment["issues"].append("Low data completeness - missing required fields")
            if consistency_score < 0.9:
                quality_assessment["issues"].append("Data consistency issues - invalid confidence values")
            if validity_score < 0.9:
                quality_assessment["issues"].append("Data validity issues - malformed timestamps or hashes")
        
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            quality_assessment["issues"].append(f"Quality assessment error: {str(e)}")
        
        return quality_assessment
    
    async def _store_metrics_in_database(self, collection_result: Dict[str, Any]):
        """Store collected metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store individual shader predictions
                for metric in collection_result.get("shader_predictions", []):
                    if isinstance(metric, ShaderPredictionMetric):
                        conn.execute("""
                            INSERT INTO shader_predictions 
                            (shader_hash, app_id, predicted_compilation, actual_compilation, 
                             prediction_confidence, compilation_time, game_context, timestamp, session_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            metric.shader_hash,
                            collection_result["app_id"],
                            int(metric.predicted_compilation),
                            int(metric.actual_compilation),
                            metric.prediction_confidence,
                            metric.compilation_time,
                            metric.game_context,
                            metric.timestamp,
                            collection_result["session_id"]
                        ))
                
                # Store session aggregated metrics
                aggregated = collection_result.get("aggregated_metrics")
                if aggregated and isinstance(aggregated, GameSessionMetrics):
                    conn.execute("""
                        INSERT INTO session_metrics 
                        (app_id, session_id, total_shaders, predicted_shaders, actual_compilations,
                         true_positives, false_positives, true_negatives, false_negatives,
                         precision, recall, f1_score, session_duration, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        aggregated.app_id,
                        aggregated.session_id,
                        aggregated.total_shaders,
                        aggregated.predicted_shaders,
                        aggregated.actual_compilations,
                        aggregated.true_positives,
                        aggregated.false_positives,
                        aggregated.true_negatives,
                        aggregated.false_negatives,
                        aggregated.precision,
                        aggregated.recall,
                        aggregated.f1_score,
                        aggregated.session_duration,
                        time.time()
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing metrics in database: {e}")
    
    async def _generate_ml_recommendations(self, collection_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for ML model improvement"""
        recommendations = []
        
        try:
            # Analyze model performance
            model_perf = collection_result.get("model_performance", {})
            accuracy = model_perf.get("accuracy", 0.0)
            
            if accuracy < 0.7:
                recommendations.append("Model accuracy is below threshold - consider retraining with more data")
            elif accuracy < 0.85:
                recommendations.append("Model accuracy could be improved - analyze feature importance")
            
            # Analyze data quality
            data_quality = collection_result.get("data_quality", {})
            overall_quality = data_quality.get("overall_score", 0.0)
            
            if overall_quality < 0.8:
                recommendations.append("Data quality issues detected - improve data collection processes")
            
            # Analyze confidence calibration
            confidence_cal = model_perf.get("confidence_calibration", {})
            calibration_error = confidence_cal.get("calibration_error", 0.0)
            
            if calibration_error > 0.1:
                recommendations.append("Model is poorly calibrated - consider confidence recalibration")
            
            # Analyze prediction errors
            error_analysis = model_perf.get("error_analysis", {})
            high_conf_errors = error_analysis.get("high_confidence_errors", 0)
            
            if high_conf_errors > 10:
                recommendations.append("High confidence prediction errors detected - review model features")
            
            # Feature analysis recommendations
            feature_analysis = collection_result.get("feature_analysis", {})
            for rec in feature_analysis.get("recommendations", []):
                recommendations.append(rec)
            
            # Session-specific recommendations
            aggregated = collection_result.get("aggregated_metrics")
            if aggregated and isinstance(aggregated, GameSessionMetrics):
                if aggregated.precision < 0.8:
                    recommendations.append("High false positive rate - adjust prediction threshold")
                if aggregated.recall < 0.8:
                    recommendations.append("High false negative rate - improve coverage of prediction model")
            
            if not recommendations:
                recommendations.append("ML model performance is within acceptable parameters")
        
        except Exception as e:
            self.logger.error(f"Error generating ML recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
        
        return recommendations
    
    async def export_training_data(self, app_id: Optional[str] = None, days_back: int = 30) -> Dict[str, Any]:
        """Export data for model training"""
        if not self.telemetry_enabled:
            return {"error": "Telemetry disabled"}
        
        export_result = {
            "export_timestamp": time.time(),
            "app_filter": app_id,
            "days_back": days_back,
            "training_samples": 0,
            "export_path": None
        }
        
        try:
            # Calculate cutoff timestamp
            cutoff_timestamp = time.time() - (days_back * 24 * 3600)
            
            # Query database for training data
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM shader_predictions 
                    WHERE timestamp > ?
                """
                params = [cutoff_timestamp]
                
                if app_id:
                    query += " AND app_id = ?"
                    params.append(app_id)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to training format
                training_data = []
                for row in rows:
                    training_sample = {
                        "shader_hash": row[1],
                        "app_id": row[2],
                        "features": {
                            "prediction_confidence": row[5],
                            "compilation_time": row[6],
                            "game_context": row[7]
                        },
                        "target": bool(row[4]),  # actual_compilation
                        "timestamp": row[8]
                    }
                    training_data.append(training_sample)
                
                export_result["training_samples"] = len(training_data)
                
                # Export to file
                export_path = f"data/telemetry/training_data_{int(time.time())}.json"
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                
                with open(export_path, 'w') as f:
                    json.dump(training_data, f, indent=2)
                
                export_result["export_path"] = export_path
                
        except Exception as e:
            self.logger.error(f"Error exporting training data: {e}")
            export_result["error"] = str(e)
        
        return export_result
    
    async def cleanup_old_data(self):
        """Clean up old telemetry data based on retention policy"""
        try:
            cutoff_timestamp = time.time() - (self.local_storage_days * 24 * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                # Delete old shader predictions
                result1 = conn.execute(
                    "DELETE FROM shader_predictions WHERE timestamp < ?", 
                    (cutoff_timestamp,)
                )
                
                # Delete old session metrics
                result2 = conn.execute(
                    "DELETE FROM session_metrics WHERE timestamp < ?", 
                    (cutoff_timestamp,)
                )
                
                # Delete old feature importance data
                result3 = conn.execute(
                    "DELETE FROM feature_importance WHERE timestamp < ?", 
                    (cutoff_timestamp,)
                )
                
                # Delete old model performance data
                result4 = conn.execute(
                    "DELETE FROM model_performance WHERE timestamp < ?", 
                    (cutoff_timestamp,)
                )
                
                conn.commit()
                
                total_deleted = (result1.rowcount + result2.rowcount + 
                               result3.rowcount + result4.rowcount)
                
                self.logger.info(f"Cleaned up {total_deleted} old telemetry records")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get summary of collected telemetry data"""
        summary = {
            "total_predictions": 0,
            "total_sessions": 0,
            "apps_analyzed": [],
            "date_range": {},
            "storage_size": 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total predictions
                cursor = conn.execute("SELECT COUNT(*) FROM shader_predictions")
                summary["total_predictions"] = cursor.fetchone()[0]
                
                # Total sessions
                cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM session_metrics")
                summary["total_sessions"] = cursor.fetchone()[0]
                
                # Apps analyzed
                cursor = conn.execute("SELECT DISTINCT app_id FROM shader_predictions")
                summary["apps_analyzed"] = [row[0] for row in cursor.fetchall()]
                
                # Date range
                cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM shader_predictions")
                min_ts, max_ts = cursor.fetchone()
                if min_ts and max_ts:
                    summary["date_range"] = {
                        "earliest": min_ts,
                        "latest": max_ts,
                        "span_days": (max_ts - min_ts) / (24 * 3600)
                    }
                
                # Storage size
                if os.path.exists(self.db_path):
                    summary["storage_size"] = os.path.getsize(self.db_path)
        
        except Exception as e:
            self.logger.error(f"Error getting telemetry summary: {e}")
        
        return summary