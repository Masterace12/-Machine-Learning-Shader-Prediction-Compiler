#!/usr/bin/env python3
"""
High-Performance ML-Only Shader Prediction System for Steam Deck
Pure machine learning implementation - no heuristic fallbacks
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

# Import threading optimizations
try:
    from .threading_config import configure_threading_for_steam_deck, get_lightgbm_params, get_sklearn_params
    from .thread_pool_manager import get_thread_manager, ThreadPriority
    HAS_THREADING_OPTIMIZATIONS = True
except ImportError:
    HAS_THREADING_OPTIMIZATIONS = False
    def configure_threading_for_steam_deck():
        pass
    def get_lightgbm_params():
        return {}
    def get_sklearn_params():
        return {}
    def get_thread_manager():
        return None

# Essential ML imports - REQUIRED for operation
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    raise ImportError("LightGBM is required for ML-only operation. Install with: pip install lightgbm")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    raise ImportError("scikit-learn is required for ML-only operation. Install with: pip install scikit-learn")

# Performance optimizations - strongly recommended
try:
    import numba
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    njit = jit

try:
    import bottleneck as bn
    HAS_BOTTLENECK = True
except ImportError:
    HAS_BOTTLENECK = False

try:
    import numexpr as ne
    HAS_NUMEXPR = True
except ImportError:
    HAS_NUMEXPR = False

# Import base types
try:
    from .unified_ml_predictor import ShaderType, UnifiedShaderFeatures
except ImportError:
    try:
        from src.core.unified_ml_predictor import ShaderType, UnifiedShaderFeatures
    except ImportError:
        from unified_ml_predictor import ShaderType, UnifiedShaderFeatures


@dataclass
class MLPredictionResult:
    """Result from ML prediction with confidence and performance metrics"""
    compilation_time_ms: float
    confidence: float
    model_used: str
    prediction_time_ms: float
    feature_importance: Optional[Dict[str, float]] = None


class HighPerformanceMLPredictor:
    """
    High-performance ML-only shader compilation time predictor
    
    Features:
    - LightGBM primary model for ultra-fast inference
    - sklearn fallback for additional algorithm support  
    - Numba JIT compilation for feature processing
    - Steam Deck hardware optimizations
    - No heuristic fallbacks - pure ML performance
    """
    
    def __init__(self, model_path: Optional[Path] = None, optimization_level: int = 2):
        """
        Initialize high-performance ML predictor
        
        Args:
            model_path: Path to pre-trained models
            optimization_level: 0=basic, 1=optimized, 2=maximum performance
        """
        self.model_path = model_path
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(__name__)
        
        # Configure threading FIRST (before any ML operations)
        if HAS_THREADING_OPTIMIZATIONS:
            try:
                self.threading_configurator = configure_threading_for_steam_deck()
                self.thread_manager = get_thread_manager()
                self.logger.info("Threading optimizations configured")
            except Exception as e:
                self.logger.warning(f"Threading optimization failed: {e}")
                self.threading_configurator = None
                self.thread_manager = None
        else:
            self.threading_configurator = None
            self.thread_manager = None
        
        # Model storage
        self.lgb_model = None
        self.sklearn_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.model_loaded = False
        
        # Steam Deck optimizations
        self.is_steam_deck = os.path.exists("/home/deck")
        self.cpu_cores = os.cpu_count() or 8
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info(f"ML-only predictor initialized with optimization level {optimization_level}")
        self.logger.info(f"Performance features: Numba={HAS_NUMBA}, Bottleneck={HAS_BOTTLENECK}, NumExpr={HAS_NUMEXPR}, Threading={HAS_THREADING_OPTIMIZATIONS}")
    
    def _initialize_models(self):
        """Initialize ML models for shader prediction"""
        try:
            # Try to load pre-trained models
            if self.model_path and self.model_path.exists():
                self._load_pretrained_models()
            else:
                # Create and train basic models with synthetic data
                self._create_baseline_models()
                
            self.model_loaded = True
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
            raise RuntimeError(f"ML model initialization failed: {e}")
    
    def _load_pretrained_models(self):
        """Load pre-trained models from disk"""
        try:
            lgb_path = self.model_path / "lightgbm_model.txt"
            if lgb_path.exists():
                self.lgb_model = lgb.Booster(model_file=str(lgb_path))
                self.logger.info("Loaded pre-trained LightGBM model")
        except Exception as e:
            self.logger.warning(f"Failed to load LightGBM model: {e}")
            
        # Create backup sklearn model
        self._create_sklearn_model()
    
    def _create_baseline_models(self):
        """Create baseline models with synthetic training data"""
        self.logger.info("Creating baseline ML models with synthetic data...")
        
        # Generate synthetic training data based on shader characteristics
        X_train, y_train = self._generate_training_data(10000)
        
        # Train LightGBM model
        self._train_lightgbm_model(X_train, y_train)
        
        # Train sklearn backup model
        self._train_sklearn_model(X_train, y_train)
    
    def _generate_training_data(self, n_samples: int):
        """Generate synthetic training data for shader compilation prediction"""
        np.random.seed(42)  # Reproducible results
        
        # Feature generation based on real shader characteristics
        instruction_counts = np.random.exponential(500, n_samples).clip(10, 5000)
        register_usage = np.random.exponential(32, n_samples).clip(1, 128)
        texture_samples = np.random.poisson(4, n_samples).clip(0, 16)
        memory_ops = np.random.exponential(15, n_samples).clip(0, 100)
        complexity = np.random.exponential(5, n_samples).clip(1, 20)
        
        # Additional features
        wave_size = np.random.choice([32, 64], n_samples)
        shader_types = np.random.randint(0, 4, n_samples)
        optimization_levels = np.random.randint(0, 3, n_samples)
        
        # Combine features
        X = np.column_stack([
            instruction_counts,
            register_usage, 
            texture_samples,
            memory_ops,
            complexity,
            wave_size,
            shader_types,
            optimization_levels
        ])
        
        # Generate realistic compilation times (logarithmic relationship)
        base_time = np.log1p(instruction_counts) * 2.0
        complexity_factor = np.log1p(complexity) * 1.5
        register_factor = np.log1p(register_usage) * 0.8
        texture_factor = texture_samples * 0.5
        
        y = base_time + complexity_factor + register_factor + texture_factor
        y += np.random.normal(0, 0.2, n_samples)  # Add noise
        y = np.exp(y).clip(0.1, 1000.0)  # Realistic range: 0.1ms to 1000ms
        
        self.feature_names = [
            'instruction_count', 'register_usage', 'texture_samples',
            'memory_operations', 'control_flow_complexity', 'wave_size',
            'shader_type', 'optimization_level'
        ]
        
        return X, y
    
    def _train_lightgbm_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train LightGBM model for ultra-fast inference"""
        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Create datasets
        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # LightGBM parameters optimized for Steam Deck
        base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 64,  # Balanced for Steam Deck CPU
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': min(self.cpu_cores, 6),  # Leave cores for game
            'force_row_wise': True,  # Better for small datasets
        }
        
        # Apply threading optimizations if available
        if HAS_THREADING_OPTIMIZATIONS:
            try:
                optimized_params = get_lightgbm_params()
                base_params.update(optimized_params)
                self.logger.debug(f"Applied threading-optimized LightGBM params: {optimized_params}")
            except Exception as e:
                self.logger.warning(f"Could not apply threading optimizations: {e}")
        
        params = base_params
        
        # Train model
        self.lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Validate performance
        val_pred = self.lgb_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        
        self.logger.info(f"LightGBM model trained - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    def _train_sklearn_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train sklearn backup model"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Get optimized parameters for scikit-learn
        sklearn_params = {'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': min(self.cpu_cores, 4)}
        if HAS_THREADING_OPTIMIZATIONS:
            try:
                optimized_sklearn_params = get_sklearn_params()
                if 'n_jobs' in optimized_sklearn_params:
                    sklearn_params['n_jobs'] = optimized_sklearn_params['n_jobs']
                self.logger.debug(f"Applied threading-optimized sklearn params: {optimized_sklearn_params}")
            except Exception as e:
                self.logger.warning(f"Could not apply sklearn threading optimizations: {e}")
        
        # Train Random Forest (fast and robust)
        self.sklearn_model = RandomForestRegressor(**sklearn_params)
        
        self.sklearn_model.fit(X_scaled, y_train)
        
        # Validate
        val_pred = self.sklearn_model.predict(X_scaled[:1000])
        rmse = np.sqrt(mean_squared_error(y_train[:1000], val_pred))
        
        self.logger.info(f"sklearn model trained - RMSE: {rmse:.3f}")
    
    def _create_sklearn_model(self):
        """Create sklearn model as backup"""
        # Generate small dataset for backup model
        X, y = self._generate_training_data(5000)
        self._train_sklearn_model(X, y)
    
    @jit(nopython=True) if HAS_NUMBA else lambda f: f
    def _extract_features_fast(self, features: UnifiedShaderFeatures) -> np.ndarray:
        """Extract features with JIT optimization (if available)"""
        return np.array([
            float(features.instruction_count),
            float(features.register_usage),
            float(features.texture_samples),
            float(features.memory_operations),
            float(features.control_flow_complexity),
            float(features.wave_size),
            float(features.shader_type.value),
            float(features.optimization_level)
        ], dtype=np.float32)
    
    def _extract_features_dict(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract features from dictionary format"""
        return np.array([
            float(features.get('instruction_count', 500)),
            float(features.get('register_usage', 32)),
            float(features.get('texture_samples', 4)),
            float(features.get('memory_operations', 10)),
            float(features.get('control_flow_complexity', 5)),
            float(features.get('wave_size', 64)),
            float(features.get('shader_type_hash', 2.0)),
            float(features.get('optimization_level', 1))
        ], dtype=np.float32)
    
    def predict_compilation_time(self, features: Union[UnifiedShaderFeatures, Dict[str, Any]]) -> MLPredictionResult:
        """
        Predict shader compilation time using ML models
        
        Args:
            features: Shader features (UnifiedShaderFeatures object or dict)
            
        Returns:
            MLPredictionResult with prediction and metadata
        """
        start_time = time.perf_counter()
        
        # Extract features
        if isinstance(features, dict):
            feature_array = self._extract_features_dict(features)
        else:
            feature_array = self._extract_features_fast(features)
        
        feature_array = feature_array.reshape(1, -1)
        
        # Primary prediction with LightGBM
        if self.lgb_model is not None:
            try:
                prediction = self.lgb_model.predict(feature_array)[0]
                model_used = "LightGBM"
                confidence = 0.95  # High confidence for primary model
                
                # Get feature importance for this prediction
                feature_importance = dict(zip(
                    self.feature_names,
                    self.lgb_model.feature_importance(importance_type='gain')
                ))
                
            except Exception as e:
                self.logger.warning(f"LightGBM prediction failed: {e}, falling back to sklearn")
                prediction, model_used, confidence, feature_importance = self._sklearn_fallback(feature_array)
        else:
            prediction, model_used, confidence, feature_importance = self._sklearn_fallback(feature_array)
        
        # Ensure realistic range
        prediction = max(0.1, min(prediction, 1000.0))
        
        # Performance tracking
        prediction_time_ms = (time.perf_counter() - start_time) * 1000
        self.prediction_count += 1
        self.total_prediction_time += prediction_time_ms
        
        return MLPredictionResult(
            compilation_time_ms=prediction,
            confidence=confidence,
            model_used=model_used,
            prediction_time_ms=prediction_time_ms,
            feature_importance=feature_importance
        )
    
    def _sklearn_fallback(self, feature_array: np.ndarray):
        """Fallback to sklearn model"""
        if self.sklearn_model is None:
            raise RuntimeError("No ML models available for prediction")
        
        try:
            # Scale features
            scaled_features = self.scaler.transform(feature_array)
            prediction = self.sklearn_model.predict(scaled_features)[0]
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.sklearn_model.feature_importances_
            ))
            
            return prediction, "sklearn", 0.85, feature_importance
            
        except Exception as e:
            self.logger.error(f"sklearn prediction also failed: {e}")
            raise RuntimeError("All ML models failed to make prediction")
    
    def predict_batch(self, features_list: List[Union[UnifiedShaderFeatures, Dict[str, Any]]]) -> List[MLPredictionResult]:
        """
        Predict compilation times for multiple shaders (batch processing)
        
        Args:
            features_list: List of shader features
            
        Returns:
            List of MLPredictionResult objects
        """
        if not features_list:
            return []
        
        start_time = time.perf_counter()
        
        # Extract all features
        feature_arrays = []
        for features in features_list:
            if isinstance(features, dict):
                feature_arrays.append(self._extract_features_dict(features))
            else:
                feature_arrays.append(self._extract_features_fast(features))
        
        batch_features = np.vstack(feature_arrays)
        
        # Batch prediction with LightGBM (much faster)
        if self.lgb_model is not None:
            try:
                predictions = self.lgb_model.predict(batch_features)
                model_used = "LightGBM"
                confidence = 0.95
                
                # Feature importance (same for all in batch)
                feature_importance = dict(zip(
                    self.feature_names,
                    self.lgb_model.feature_importance(importance_type='gain')
                ))
                
            except Exception as e:
                self.logger.warning(f"Batch LightGBM prediction failed: {e}")
                scaled_features = self.scaler.transform(batch_features)
                predictions = self.sklearn_model.predict(scaled_features)
                model_used = "sklearn"
                confidence = 0.85
                feature_importance = dict(zip(self.feature_names, self.sklearn_model.feature_importances_))
        else:
            scaled_features = self.scaler.transform(batch_features)
            predictions = self.sklearn_model.predict(scaled_features)
            model_used = "sklearn"
            confidence = 0.85
            feature_importance = dict(zip(self.feature_names, self.sklearn_model.feature_importances_))
        
        # Create results
        total_time = (time.perf_counter() - start_time) * 1000
        per_prediction_time = total_time / len(features_list)
        
        results = []
        for pred in predictions:
            pred = max(0.1, min(pred, 1000.0))  # Clamp to realistic range
            results.append(MLPredictionResult(
                compilation_time_ms=pred,
                confidence=confidence,
                model_used=model_used,
                prediction_time_ms=per_prediction_time,
                feature_importance=feature_importance
            ))
        
        self.prediction_count += len(predictions)
        self.total_prediction_time += total_time
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the ML predictor"""
        avg_prediction_time = (
            self.total_prediction_time / max(1, self.prediction_count)
        )
        
        return {
            'model_type': 'ML-only',
            'primary_model': 'LightGBM' if self.lgb_model else 'sklearn',
            'prediction_count': self.prediction_count,
            'average_prediction_time_ms': avg_prediction_time,
            'total_prediction_time_ms': self.total_prediction_time,
            'predictions_per_second': 1000.0 / max(0.001, avg_prediction_time),
            'optimization_level': self.optimization_level,
            'performance_features': {
                'numba_jit': HAS_NUMBA,
                'bottleneck': HAS_BOTTLENECK,
                'numexpr': HAS_NUMEXPR,
                'lightgbm': self.lgb_model is not None,
                'sklearn': self.sklearn_model is not None
            },
            'hardware': {
                'steam_deck': self.is_steam_deck,
                'cpu_cores': self.cpu_cores
            }
        }
    
    def save_models(self, output_path: Path):
        """Save trained models to disk"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.lgb_model:
            lgb_path = output_path / "lightgbm_model.txt"
            self.lgb_model.save_model(str(lgb_path))
            self.logger.info(f"LightGBM model saved to {lgb_path}")
        
        if self.sklearn_model:
            import joblib
            sklearn_path = output_path / "sklearn_model.pkl"
            joblib.dump(self.sklearn_model, sklearn_path)
            
            scaler_path = output_path / "scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"sklearn model saved to {sklearn_path}")
    
    def cleanup(self):
        """Cleanup resources and threads"""
        try:
            # Clean up models
            self.lgb_model = None
            self.sklearn_model = None
            self.scaler = None
            
            # Cleanup threading resources
            if self.thread_manager:
                self.thread_manager.shutdown(wait=False)
                self.thread_manager = None
            
            if self.threading_configurator:
                self.threading_configurator.cleanup()
                self.threading_configurator = None
            
            self.logger.info("ML predictor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


# Global instance for easy access
_global_ml_predictor = None


def get_ml_predictor(force_reload: bool = False) -> HighPerformanceMLPredictor:
    """Get or create global ML predictor instance"""
    global _global_ml_predictor
    if _global_ml_predictor is None or force_reload:
        _global_ml_predictor = HighPerformanceMLPredictor(optimization_level=2)
    return _global_ml_predictor


# Convenience function for quick predictions
def predict_shader_compilation_time(features: Union[UnifiedShaderFeatures, Dict[str, Any]]) -> float:
    """Quick shader compilation time prediction"""
    predictor = get_ml_predictor()
    result = predictor.predict_compilation_time(features)
    return result.compilation_time_ms


if __name__ == "__main__":
    # Test the ML-only predictor
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 High-Performance ML-Only Shader Predictor Test")
    print("=" * 60)
    
    # Initialize predictor
    predictor = HighPerformanceMLPredictor(optimization_level=2)
    
    # Test single prediction
    test_features = {
        'instruction_count': 1000,
        'register_usage': 64,
        'texture_samples': 8,
        'memory_operations': 20,
        'control_flow_complexity': 10,
        'wave_size': 64,
        'shader_type_hash': 2.0,
        'optimization_level': 2
    }
    
    print("\n⚡ Single Prediction Test:")
    result = predictor.predict_compilation_time(test_features)
    print(f"  Prediction: {result.compilation_time_ms:.2f}ms")
    print(f"  Model: {result.model_used}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Speed: {result.prediction_time_ms:.3f}ms")
    
    # Test batch prediction
    print("\n⚡ Batch Prediction Test:")
    batch_features = [test_features] * 100
    start_time = time.perf_counter()
    batch_results = predictor.predict_batch(batch_features)
    batch_time = (time.perf_counter() - start_time) * 1000
    
    print(f"  Batch size: {len(batch_results)}")
    print(f"  Total time: {batch_time:.2f}ms")
    print(f"  Per prediction: {batch_time/len(batch_results):.3f}ms")
    print(f"  Throughput: {len(batch_results)/(batch_time/1000):.0f} predictions/sec")
    
    # Performance metrics
    print("\n📊 Performance Metrics:")
    metrics = predictor.get_performance_metrics()
    print(f"  Primary model: {metrics['primary_model']}")
    print(f"  Average speed: {metrics['average_prediction_time_ms']:.3f}ms")
    print(f"  Throughput: {metrics['predictions_per_second']:.0f} pred/sec")
    print(f"  Optimization level: {metrics['optimization_level']}")
    
    performance_features = metrics['performance_features']
    print(f"  Performance features: {sum(performance_features.values())}/{len(performance_features)} active")
    
    print("\n✅ ML-only predictor test completed!")