#!/usr/bin/env python3
"""
Unified ML Shader Prediction System for Steam Deck
Combines the best features from multiple ML implementations into a single, 
optimized system for shader compilation prediction and thermal management.
"""

import os
import json
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

# Core ML imports with comprehensive fallback handling
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("ERROR: NumPy not installed. Please run: pip install numpy>=1.19.0")
    print("Or use the installation script: ./install.sh")
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("WARNING: pandas not available. Some features may be limited.")
    HAS_PANDAS = False
    pd = None

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
    import joblib
    HAS_SKLEARN = True
except ImportError:
    print("WARNING: scikit-learn not available. ML features will be limited.")
    HAS_SKLEARN = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Optional PyTorch support for advanced models
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    if torch.cuda.is_available():
        print("GPU acceleration available via CUDA/ROCm")
    else:
        print("Using CPU inference (GPU acceleration not available)")
except ImportError:
    HAS_TORCH = False


class ShaderType(Enum):
    """Shader types commonly found in games"""
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    COMPUTE = "compute"
    GEOMETRY = "geometry"
    TESSELLATION = "tessellation"
    RAYTRACING = "raytracing"
    UNKNOWN = "unknown"


class ThermalState(Enum):
    """Steam Deck thermal states based on research"""
    COOL = "cool"        # < 65°C - Full performance
    NORMAL = "normal"    # 65-80°C - Standard operation  
    WARM = "warm"        # 80-85°C - Reduced background work
    HOT = "hot"          # 85-90°C - Essential shaders only
    THROTTLING = "throttling"  # 90-95°C - Compilation paused
    CRITICAL = "critical"      # > 95°C - Emergency stop


class SteamDeckModel(Enum):
    """Steam Deck hardware models"""
    LCD = "lcd"          # Original LCD model (Van Gogh APU)
    OLED = "oled"        # OLED model (Phoenix APU)
    UNKNOWN = "unknown"


@dataclass
class UnifiedShaderFeatures:
    """Comprehensive feature vector combining both ML implementations"""
    # Basic shader properties
    shader_hash: str
    shader_type: ShaderType
    instruction_count: int
    bytecode_size: int
    complexity_score: float
    
    # Shader characteristics
    register_pressure: float
    texture_samples: int
    branch_count: int
    loop_count: int
    loop_depth: int
    memory_access_pattern: str
    
    # Engine and game context
    engine_type: str
    game_id: str
    
    # Steam Deck optimizations
    rdna2_optimal: bool
    memory_bandwidth_sensitive: bool
    thermal_sensitive: bool
    
    # Hardware context
    steam_deck_model: SteamDeckModel = SteamDeckModel.UNKNOWN
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numeric feature vector for ML models"""
        if not HAS_NUMPY:
            return []
        
        return np.array([
            self.instruction_count,
            self.bytecode_size,
            self.complexity_score,
            self.register_pressure,
            self.texture_samples,
            self.branch_count,
            self.loop_count,
            self.loop_depth,
            1.0 if self.rdna2_optimal else 0.0,
            1.0 if self.memory_bandwidth_sensitive else 0.0,
            1.0 if self.thermal_sensitive else 0.0,
            1.0 if self.steam_deck_model == SteamDeckModel.OLED else 0.0
        ])


@dataclass
class CompilationResult:
    """Result of shader compilation with comprehensive telemetry"""
    shader_hash: str
    compilation_time_ms: float
    success: bool
    error_code: Optional[str] = None
    
    # Resource usage
    memory_used_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Thermal data
    cpu_temp_celsius: float = 0.0
    gpu_temp_celsius: float = 0.0
    apu_temp_celsius: float = 0.0
    thermal_throttled: bool = False
    
    # Power data
    power_draw_watts: float = 0.0
    battery_level_percent: float = 100.0
    
    # Context
    timestamp: float = field(default_factory=time.time)
    steam_deck_model: SteamDeckModel = SteamDeckModel.UNKNOWN
    variant_count: int = 1


class HeuristicPredictor:
    """Fallback prediction system when ML is unavailable"""
    
    def __init__(self):
        self.base_times = {
            ShaderType.VERTEX: 50,     # ms
            ShaderType.FRAGMENT: 100,  # ms
            ShaderType.COMPUTE: 200,   # ms
            ShaderType.GEOMETRY: 150,  # ms
            ShaderType.TESSELLATION: 300,  # ms
            ShaderType.RAYTRACING: 500,    # ms
            ShaderType.UNKNOWN: 100
        }
        
        self.complexity_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0,
            'very_high': 4.0
        }
    
    def predict_compilation_time(self, features: UnifiedShaderFeatures) -> float:
        """Heuristic-based compilation time prediction"""
        base_time = self.base_times.get(features.shader_type, 100)
        
        # Complexity adjustments
        complexity_factor = min(features.complexity_score / 100.0, 4.0)
        
        # Instruction count scaling
        instruction_factor = max(1.0, features.instruction_count / 1000.0)
        
        # Steam Deck model adjustments
        model_factor = 0.8 if features.steam_deck_model == SteamDeckModel.OLED else 1.0
        
        # Thermal sensitivity
        thermal_factor = 1.2 if features.thermal_sensitive else 1.0
        
        return base_time * complexity_factor * instruction_factor * model_factor * thermal_factor
    
    def predict_success_probability(self, features: UnifiedShaderFeatures) -> float:
        """Heuristic-based success probability prediction"""
        base_success = 0.95
        
        # Reduce success rate for complex shaders
        complexity_penalty = min(0.2, features.complexity_score / 500.0)
        
        # Reduce for high register pressure
        register_penalty = min(0.1, features.register_pressure / 100.0)
        
        # RDNA2 optimization bonus
        rdna2_bonus = 0.05 if features.rdna2_optimal else 0.0
        
        return max(0.1, base_success - complexity_penalty - register_penalty + rdna2_bonus)


class ThermalAwareScheduler:
    """Thermal-aware compilation scheduling for Steam Deck"""
    
    def __init__(self):
        self.thermal_limits = {
            SteamDeckModel.LCD: {
                ThermalState.COOL: {'threads': 4, 'max_queue': 20},
                ThermalState.NORMAL: {'threads': 3, 'max_queue': 15},
                ThermalState.WARM: {'threads': 2, 'max_queue': 10},
                ThermalState.HOT: {'threads': 1, 'max_queue': 5},
                ThermalState.THROTTLING: {'threads': 0, 'max_queue': 0},
                ThermalState.CRITICAL: {'threads': 0, 'max_queue': 0}
            },
            SteamDeckModel.OLED: {
                ThermalState.COOL: {'threads': 6, 'max_queue': 25},
                ThermalState.NORMAL: {'threads': 4, 'max_queue': 20},
                ThermalState.WARM: {'threads': 3, 'max_queue': 15},
                ThermalState.HOT: {'threads': 2, 'max_queue': 8},
                ThermalState.THROTTLING: {'threads': 0, 'max_queue': 0},
                ThermalState.CRITICAL: {'threads': 0, 'max_queue': 0}
            }
        }
    
    def get_thermal_state(self, apu_temp: float) -> ThermalState:
        """Determine thermal state based on APU temperature"""
        if apu_temp < 65:
            return ThermalState.COOL
        elif apu_temp < 80:
            return ThermalState.NORMAL
        elif apu_temp < 85:
            return ThermalState.WARM
        elif apu_temp < 90:
            return ThermalState.HOT
        elif apu_temp < 95:
            return ThermalState.THROTTLING
        else:
            return ThermalState.CRITICAL
    
    def get_compilation_config(self, model: SteamDeckModel, thermal_state: ThermalState) -> Dict[str, int]:
        """Get compilation configuration for current thermal state"""
        return self.thermal_limits.get(model, self.thermal_limits[SteamDeckModel.LCD]).get(
            thermal_state, {'threads': 0, 'max_queue': 0}
        )


class UnifiedMLPredictor:
    """Unified ML prediction system combining the best of both implementations"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path.home() / '.cache' / 'shader-predict-compile' / 'models'
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.compilation_time_model = None
        self.success_model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        
        # Fallback predictor
        self.heuristic_predictor = HeuristicPredictor()
        
        # Thermal management
        self.thermal_scheduler = ThermalAwareScheduler()
        
        # Training data
        self.training_data = deque(maxlen=10000)  # Memory-efficient circular buffer
        self.feature_cache = {}
        
        # Performance tracking
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize models
        self._initialize_models()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_models(self):
        """Initialize ML models with Steam Deck optimizations"""
        if not HAS_SKLEARN:
            self.logger.warning("scikit-learn not available, using heuristic predictor only")
            return
        
        # Use ExtraTreesRegressor for memory efficiency on Steam Deck
        self.compilation_time_model = ExtraTreesRegressor(
            n_estimators=50,        # Reduced from 100 for memory constraints
            max_depth=15,           # Limited depth for Steam Deck memory
            random_state=42,
            n_jobs=2,              # Conservative for Steam Deck CPU
            max_features='sqrt'     # Memory efficient feature selection
        )
        
        self.success_model = RandomForestClassifier(
            n_estimators=30,        # Reduced for memory efficiency
            max_depth=10,
            random_state=42,
            n_jobs=2,
            max_features='sqrt'
        )
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        if not HAS_SKLEARN:
            return
        
        try:
            time_model_path = self.model_path / 'compilation_time_model.pkl'
            success_model_path = self.model_path / 'success_model.pkl'
            scaler_path = self.model_path / 'scaler.pkl'
            
            if time_model_path.exists():
                self.compilation_time_model = joblib.load(time_model_path)
                self.logger.info("Loaded compilation time model")
            
            if success_model_path.exists():
                self.success_model = joblib.load(success_model_path)
                self.logger.info("Loaded success model")
            
            if scaler_path.exists() and self.scaler:
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Loaded feature scaler")
                
        except Exception as e:
            self.logger.warning(f"Could not load models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        if not HAS_SKLEARN:
            return
        
        try:
            if self.compilation_time_model:
                joblib.dump(self.compilation_time_model, self.model_path / 'compilation_time_model.pkl')
            
            if self.success_model:
                joblib.dump(self.success_model, self.model_path / 'success_model.pkl')
            
            if self.scaler:
                joblib.dump(self.scaler, self.model_path / 'scaler.pkl')
                
            self.logger.info("Models saved successfully")
        except Exception as e:
            self.logger.error(f"Could not save models: {e}")
    
    def predict_compilation_time(self, features: UnifiedShaderFeatures, 
                                use_cache: bool = True) -> float:
        """Predict shader compilation time"""
        # Check cache first
        if use_cache:
            cache_key = features.shader_hash
            if cache_key in self.prediction_cache:
                self.cache_hits += 1
                return self.prediction_cache[cache_key]['time']
            self.cache_misses += 1
        
        # Use ML model if available and trained
        if HAS_SKLEARN and self.compilation_time_model and hasattr(self.compilation_time_model, 'feature_importances_'):
            try:
                feature_vector = features.to_feature_vector().reshape(1, -1)
                if self.scaler:
                    feature_vector = self.scaler.transform(feature_vector)
                
                prediction = self.compilation_time_model.predict(feature_vector)[0]
                
                # Cache prediction
                if use_cache:
                    self.prediction_cache[cache_key] = {
                        'time': prediction,
                        'timestamp': time.time()
                    }
                
                return max(1.0, prediction)  # Ensure positive prediction
            except Exception as e:
                self.logger.warning(f"ML prediction failed, using heuristic: {e}")
        
        # Fallback to heuristic predictor
        return self.heuristic_predictor.predict_compilation_time(features)
    
    def predict_success_probability(self, features: UnifiedShaderFeatures) -> float:
        """Predict shader compilation success probability"""
        if HAS_SKLEARN and self.success_model and hasattr(self.success_model, 'feature_importances_'):
            try:
                feature_vector = features.to_feature_vector().reshape(1, -1)
                if self.scaler:
                    feature_vector = self.scaler.transform(feature_vector)
                
                probability = self.success_model.predict_proba(feature_vector)[0][1]
                return probability
            except Exception as e:
                self.logger.warning(f"ML success prediction failed, using heuristic: {e}")
        
        # Fallback to heuristic predictor
        return self.heuristic_predictor.predict_success_probability(features)
    
    def should_compile_now(self, features: UnifiedShaderFeatures, 
                          current_thermal_state: ThermalState,
                          steam_deck_model: SteamDeckModel) -> bool:
        """Determine if shader should be compiled now based on thermal constraints"""
        config = self.thermal_scheduler.get_compilation_config(steam_deck_model, current_thermal_state)
        
        if config['threads'] == 0:
            return False  # No compilation allowed in this thermal state
        
        # Predict compilation time and check if it's worth it
        predicted_time = self.predict_compilation_time(features)
        success_probability = self.predict_success_probability(features)
        
        # Don't compile if success probability is too low
        if success_probability < 0.7:
            return False
        
        # Don't compile very long shaders in warm states
        if current_thermal_state in [ThermalState.WARM, ThermalState.HOT]:
            if predicted_time > 500:  # ms
                return False
        
        return True
    
    def add_training_data(self, features: UnifiedShaderFeatures, result: CompilationResult):
        """Add new training data"""
        try:
            feature_vector = features.to_feature_vector()
            if len(feature_vector) > 0:  # Ensure we have valid features
                self.training_data.append({
                    'features': feature_vector,
                    'compilation_time': result.compilation_time_ms,
                    'success': result.success
                })
                
                # Trigger retraining if we have enough new data
                if len(self.training_data) >= 100 and len(self.training_data) % 50 == 0:
                    self._retrain_models()
                    
        except Exception as e:
            self.logger.error(f"Could not add training data: {e}")
    
    def _retrain_models(self):
        """Retrain models with accumulated data"""
        if not HAS_SKLEARN or len(self.training_data) < 50:
            return
        
        try:
            # Prepare training data
            X = np.array([item['features'] for item in self.training_data])
            y_time = np.array([item['compilation_time'] for item in self.training_data])
            y_success = np.array([item['success'] for item in self.training_data])
            
            # Scale features
            if self.scaler:
                X = self.scaler.fit_transform(X)
            
            # Train compilation time model
            if self.compilation_time_model:
                self.compilation_time_model.fit(X, y_time)
                self.logger.info(f"Retrained compilation time model with {len(X)} samples")
            
            # Train success model
            if self.success_model and len(np.unique(y_success)) > 1:  # Need both classes
                self.success_model.fit(X, y_success)
                self.logger.info(f"Retrained success model with {len(X)} samples")
            
            # Save updated models
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_predictions = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(1, total_predictions)
        
        return {
            'total_predictions': total_predictions,
            'cache_hit_rate': cache_hit_rate,
            'training_data_count': len(self.training_data),
            'models_available': {
                'compilation_time': self.compilation_time_model is not None,
                'success': self.success_model is not None,
                'sklearn_available': HAS_SKLEARN
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = UnifiedMLPredictor()
    
    # Example shader features
    features = UnifiedShaderFeatures(
        shader_hash="example_hash_123",
        shader_type=ShaderType.FRAGMENT,
        instruction_count=1000,
        bytecode_size=4096,
        complexity_score=75.0,
        register_pressure=60.0,
        texture_samples=4,
        branch_count=8,
        loop_count=2,
        loop_depth=3,
        memory_access_pattern="sequential",
        engine_type="unreal",
        game_id="cyberpunk2077",
        rdna2_optimal=True,
        memory_bandwidth_sensitive=False,
        thermal_sensitive=True,
        steam_deck_model=SteamDeckModel.OLED
    )
    
    # Make predictions
    predicted_time = predictor.predict_compilation_time(features)
    success_prob = predictor.predict_success_probability(features)
    
    print(f"Predicted compilation time: {predicted_time:.1f}ms")
    print(f"Success probability: {success_prob:.2f}")
    
    # Check if should compile now
    should_compile = predictor.should_compile_now(
        features, 
        ThermalState.NORMAL, 
        SteamDeckModel.OLED
    )
    print(f"Should compile now: {should_compile}")
    
    # Performance stats
    stats = predictor.get_performance_stats()
    print(f"Performance stats: {stats}")