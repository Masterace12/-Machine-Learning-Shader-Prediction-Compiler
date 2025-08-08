#!/usr/bin/env python3
"""
Advanced Machine Learning Shader Prediction System for Steam Deck
Implements ensemble models for compilation time prediction, success rate estimation,
and adaptive optimization based on hardware telemetry and gameplay patterns.
"""

import os
import json
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Fallback implementations
    np = None

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

@dataclass
class ShaderFeatures:
    """Feature vector for shader analysis"""
    shader_hash: str
    instruction_count: int
    complexity_score: float
    stage_type: str  # vertex, fragment, compute, etc.
    uses_textures: bool
    uses_uniforms: bool
    branch_count: int
    loop_count: int
    register_pressure: float
    memory_access_pattern: str
    engine_type: str
    game_id: str
    
    # Steam Deck specific features
    rdna2_optimal: bool
    memory_bandwidth_sensitive: bool
    thermal_sensitive: bool
    
    def to_vector(self) -> List[float]:
        """Convert to numeric feature vector for ML models"""
        return [
            self.instruction_count,
            self.complexity_score,
            1.0 if self.uses_textures else 0.0,
            1.0 if self.uses_uniforms else 0.0,
            self.branch_count,
            self.loop_count,
            self.register_pressure,
            1.0 if self.rdna2_optimal else 0.0,
            1.0 if self.memory_bandwidth_sensitive else 0.0,
            1.0 if self.thermal_sensitive else 0.0
        ]

@dataclass
class CompilationResult:
    """Result of shader compilation with telemetry data"""
    shader_hash: str
    compilation_time_ms: float
    success: bool
    error_code: Optional[str]
    memory_used_mb: float
    cpu_temp_c: float
    gpu_temp_c: float
    battery_level_percent: float
    thermal_throttled: bool
    timestamp: float
    steam_deck_model: str  # LCD or OLED
    
@dataclass
class GameplayPattern:
    """Gameplay pattern for predictive shader loading"""
    game_id: str
    level_or_scene: str
    shader_usage_sequence: List[str]  # Sequence of shader hashes
    time_intervals_ms: List[float]    # Time between shader usage
    frequency_of_occurrence: float
    user_action_trigger: str  # menu, combat, exploration, etc.

class SteamDeckMLPredictor:
    """Advanced ML-based shader prediction system for Steam Deck"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path.home() / ".steam-deck-shader-ml"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # ML Models
        self.compilation_time_model = None
        self.success_prediction_model = None
        self.priority_clustering_model = None
        self.gameplay_pattern_model = None
        
        # Feature processors
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.label_encoders = {}
        
        # Training data storage
        self.training_data = []
        self.compilation_history = deque(maxlen=10000)
        self.gameplay_patterns = defaultdict(list)
        
        # Steam Deck specific configurations
        self.steam_deck_config = {
            'LCD': {
                'memory_bandwidth_gb_s': 88.0,
                'thermal_limit_c': 80.0,
                'max_parallel_compiles': 4,
                'power_budget_w': 15.0
            },
            'OLED': {
                'memory_bandwidth_gb_s': 102.4,
                'thermal_limit_c': 85.0,
                'max_parallel_compiles': 6,
                'power_budget_w': 18.0
            }
        }
        
        # Online learning parameters
        self.online_learning_enabled = True
        self.model_update_interval = 100  # Update after N new samples
        self.samples_since_update = 0
        
        # Load existing models
        self._load_models()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for ML predictor"""
        logger = logging.getLogger('SteamDeckMLPredictor')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.model_dir / 'ml_predictor.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        if not HAS_SKLEARN or not HAS_JOBLIB:
            self.logger.warning("sklearn or joblib not available, using fallback predictors")
            return
            
        try:
            model_files = {
                'compilation_time': self.model_dir / 'compilation_time_model.pkl',
                'success_prediction': self.model_dir / 'success_prediction_model.pkl',
                'priority_clustering': self.model_dir / 'priority_clustering_model.pkl',
                'scaler': self.model_dir / 'feature_scaler.pkl'
            }
            
            for model_name, file_path in model_files.items():
                if file_path.exists():
                    if model_name == 'scaler':
                        self.scaler = joblib.load(file_path)
                    else:
                        setattr(self, f'{model_name}_model', joblib.load(file_path))
                    self.logger.info(f"Loaded {model_name} model from {file_path}")
                    
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        if not HAS_JOBLIB:
            return
            
        try:
            models_to_save = {
                'compilation_time_model': self.compilation_time_model,
                'success_prediction_model': self.success_prediction_model,
                'priority_clustering_model': self.priority_clustering_model,
                'scaler': self.scaler
            }
            
            for model_name, model in models_to_save.items():
                if model is not None:
                    file_path = self.model_dir / f'{model_name.replace("_model", "")}_model.pkl'
                    if model_name == 'scaler':
                        file_path = self.model_dir / 'feature_scaler.pkl'
                    joblib.dump(model, file_path)
                    self.logger.info(f"Saved {model_name} to {file_path}")
                    
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def extract_shader_features(self, shader_path: Path, game_id: str, 
                               steam_deck_model: str = 'LCD') -> ShaderFeatures:
        """Extract comprehensive features from shader for ML prediction"""
        try:
            with open(shader_path, 'rb') as f:
                shader_data = f.read()
                
            shader_hash = hashlib.sha256(shader_data).hexdigest()[:16]
            
            # Basic analysis
            instruction_count = self._estimate_instruction_count(shader_data)
            complexity_score = self._calculate_complexity_score(shader_data)
            stage_type = self._detect_stage_type(shader_path, shader_data)
            
            # Pattern detection
            uses_textures = b'texture' in shader_data.lower()
            uses_uniforms = b'uniform' in shader_data.lower()
            branch_count = shader_data.count(b'if') + shader_data.count(b'switch')
            loop_count = shader_data.count(b'for') + shader_data.count(b'while')
            
            # Advanced analysis
            register_pressure = self._estimate_register_pressure(shader_data)
            memory_access_pattern = self._analyze_memory_access(shader_data)
            engine_type = self._detect_engine_type(shader_path)
            
            # Steam Deck specific analysis
            rdna2_optimal = self._check_rdna2_optimization(shader_data)
            memory_bandwidth_sensitive = self._check_memory_bandwidth_sensitivity(shader_data)
            thermal_sensitive = complexity_score > 0.7 or register_pressure > 0.8
            
            return ShaderFeatures(
                shader_hash=shader_hash,
                instruction_count=instruction_count,
                complexity_score=complexity_score,
                stage_type=stage_type,
                uses_textures=uses_textures,
                uses_uniforms=uses_uniforms,
                branch_count=branch_count,
                loop_count=loop_count,
                register_pressure=register_pressure,
                memory_access_pattern=memory_access_pattern,
                engine_type=engine_type,
                game_id=game_id,
                rdna2_optimal=rdna2_optimal,
                memory_bandwidth_sensitive=memory_bandwidth_sensitive,
                thermal_sensitive=thermal_sensitive
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting features from {shader_path}: {e}")
            # Return default features
            return ShaderFeatures(
                shader_hash="unknown",
                instruction_count=50,
                complexity_score=0.5,
                stage_type="unknown",
                uses_textures=False,
                uses_uniforms=False,
                branch_count=0,
                loop_count=0,
                register_pressure=0.5,
                memory_access_pattern="linear",
                engine_type="unknown",
                game_id=game_id,
                rdna2_optimal=False,
                memory_bandwidth_sensitive=False,
                thermal_sensitive=False
            )
    
    def _estimate_instruction_count(self, shader_data: bytes) -> int:
        """Estimate shader instruction count"""
        if shader_data.startswith(b'DXBC'):
            # DirectX bytecode
            return len(shader_data) // 4  # Rough estimate
        elif shader_data.startswith(b'\x03\x02\x23\x07'):
            # SPIR-V
            return len(shader_data) // 4
        else:
            # Text-based shader
            lines = shader_data.decode('utf-8', errors='ignore').split('\n')
            return len([line for line in lines if line.strip() and not line.strip().startswith('//')])
    
    def _calculate_complexity_score(self, shader_data: bytes) -> float:
        """Calculate shader complexity score (0.0 to 1.0)"""
        size_factor = min(len(shader_data) / 10000.0, 1.0)  # Normalize by 10KB
        
        # Count complex operations
        complex_ops = [b'sqrt', b'sin', b'cos', b'pow', b'exp', b'log']
        complexity_ops = sum(shader_data.lower().count(op) for op in complex_ops)
        
        # Branch complexity
        branches = shader_data.count(b'if') + shader_data.count(b'switch') * 2
        loops = (shader_data.count(b'for') + shader_data.count(b'while')) * 1.5
        
        complexity_score = (size_factor + complexity_ops * 0.1 + branches * 0.05 + loops * 0.1)
        return min(complexity_score, 1.0)
    
    def _detect_stage_type(self, shader_path: Path, shader_data: bytes) -> str:
        """Detect shader stage type"""
        filename = shader_path.name.lower()
        
        stage_patterns = {
            'vertex': ['vert', 'vs', 'vertex'],
            'fragment': ['frag', 'ps', 'pixel', 'fragment'],
            'compute': ['comp', 'cs', 'compute'],
            'geometry': ['geom', 'gs', 'geometry'],
            'tessellation': ['tess', 'hs', 'ds', 'hull', 'domain']
        }
        
        for stage, patterns in stage_patterns.items():
            if any(pattern in filename for pattern in patterns):
                return stage
                
        # Check content
        content = shader_data.decode('utf-8', errors='ignore').lower()
        if 'gl_position' in content or 'sv_position' in content:
            return 'vertex'
        elif 'gl_fragcolor' in content or 'sv_target' in content:
            return 'fragment'
        elif 'numthreads' in content:
            return 'compute'
            
        return 'unknown'
    
    def _estimate_register_pressure(self, shader_data: bytes) -> float:
        """Estimate register pressure (0.0 to 1.0)"""
        content = shader_data.decode('utf-8', errors='ignore').lower()
        
        # Count variable declarations and temporary usage
        temp_vars = content.count('temp') + content.count('tmp')
        declarations = content.count('float') + content.count('vec') + content.count('mat')
        
        # Normalize to pressure score
        pressure = min((temp_vars + declarations) / 50.0, 1.0)
        return pressure
    
    def _analyze_memory_access(self, shader_data: bytes) -> str:
        """Analyze memory access patterns"""
        content = shader_data.decode('utf-8', errors='ignore').lower()
        
        if 'texture' in content:
            if 'random' in content or 'noise' in content:
                return 'random'
            elif 'atlas' in content or 'array' in content:
                return 'scattered'
            else:
                return 'textured'
        elif 'uniform' in content:
            return 'uniform'
        else:
            return 'linear'
    
    def _detect_engine_type(self, shader_path: Path) -> str:
        """Detect game engine type from shader path"""
        path_str = str(shader_path).lower()
        
        if 'unreal' in path_str or 'ue4' in path_str or 'ue5' in path_str:
            return 'unreal'
        elif 'unity' in path_str:
            return 'unity'
        elif 'source2' in path_str or 'valve' in path_str:
            return 'source2'
        elif 'cryengine' in path_str or 'crytek' in path_str:
            return 'cryengine'
        elif 'godot' in path_str:
            return 'godot'
        else:
            return 'unknown'
    
    def _check_rdna2_optimization(self, shader_data: bytes) -> bool:
        """Check if shader is optimized for RDNA2 architecture"""
        content = shader_data.decode('utf-8', errors='ignore').lower()
        
        # Look for RDNA2-specific optimizations
        rdna2_hints = ['wave32', 'subgroup', 'warp', 'simd']
        return any(hint in content for hint in rdna2_hints)
    
    def _check_memory_bandwidth_sensitivity(self, shader_data: bytes) -> bool:
        """Check if shader is sensitive to memory bandwidth"""
        content = shader_data.decode('utf-8', errors='ignore').lower()
        
        # High bandwidth usage indicators
        bandwidth_sensitive = [
            'texture2darray', 'texturearray', 'cubemap',
            'sample', 'load', 'gather',
            'atomic', 'imageload', 'imagestore'
        ]
        
        sensitive_count = sum(content.count(indicator) for indicator in bandwidth_sensitive)
        return sensitive_count > 3
    
    def train_compilation_time_model(self, training_data: List[Tuple[ShaderFeatures, CompilationResult]]):
        """Train ensemble model for compilation time prediction"""
        if not HAS_SKLEARN or len(training_data) < 10:
            self.logger.warning("Insufficient data or sklearn unavailable for training")
            return
        
        try:
            # Prepare feature matrix and target vector
            X = np.array([features.to_vector() for features, _ in training_data])
            y = np.array([result.compilation_time_ms for _, result in training_data])
            
            # Feature scaling
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Create ensemble model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            lr_model = LinearRegression()
            
            ensemble = VotingRegressor([
                ('rf', rf_model),
                ('gb', gb_model),
                ('lr', lr_model)
            ])
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Evaluate
            train_score = ensemble.score(X_train, y_train)
            test_score = ensemble.score(X_test, y_test)
            predictions = ensemble.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            
            self.logger.info(f"Compilation time model - Train R²: {train_score:.3f}, "
                           f"Test R²: {test_score:.3f}, MAE: {mae:.2f}ms")
            
            self.compilation_time_model = ensemble
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Error training compilation time model: {e}")
    
    def train_success_prediction_model(self, training_data: List[Tuple[ShaderFeatures, CompilationResult]]):
        """Train model for compilation success prediction"""
        if not HAS_SKLEARN or len(training_data) < 10:
            return
        
        try:
            # Prepare data
            X = np.array([features.to_vector() for features, _ in training_data])
            y = np.array([1 if result.success else 0 for _, result in training_data])
            
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            self.logger.info(f"Success prediction model - Train Acc: {train_acc:.3f}, "
                           f"Test Acc: {test_acc:.3f}")
            
            self.success_prediction_model = model
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Error training success prediction model: {e}")
    
    def predict_compilation_time(self, features: ShaderFeatures, 
                               steam_deck_model: str = 'LCD',
                               thermal_state: Dict = None) -> float:
        """Predict shader compilation time with Steam Deck adjustments"""
        if self.compilation_time_model is None or self.scaler is None:
            # Fallback heuristic
            base_time = features.instruction_count * 2.0  # 2ms per instruction
            complexity_multiplier = 1.0 + features.complexity_score
            return base_time * complexity_multiplier
        
        try:
            # Get base prediction
            X = np.array([features.to_vector()]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            base_prediction = self.compilation_time_model.predict(X_scaled)[0]
            
            # Apply Steam Deck specific adjustments
            config = self.steam_deck_config[steam_deck_model]
            
            # Thermal throttling adjustment
            thermal_factor = 1.0
            if thermal_state:
                cpu_temp = thermal_state.get('cpu_temp_c', 70.0)
                if cpu_temp > config['thermal_limit_c']:
                    thermal_factor = 1.0 + (cpu_temp - config['thermal_limit_c']) * 0.02
            
            # Memory bandwidth adjustment
            bandwidth_factor = 1.0
            if features.memory_bandwidth_sensitive:
                if steam_deck_model == 'LCD':
                    bandwidth_factor = 1.2  # Higher penalty for LCD
                else:
                    bandwidth_factor = 1.05  # Lower penalty for OLED
            
            adjusted_prediction = base_prediction * thermal_factor * bandwidth_factor
            return max(adjusted_prediction, 1.0)  # Minimum 1ms
            
        except Exception as e:
            self.logger.error(f"Error predicting compilation time: {e}")
            return features.instruction_count * 2.0
    
    def predict_compilation_success(self, features: ShaderFeatures) -> float:
        """Predict probability of successful compilation"""
        if self.success_prediction_model is None or self.scaler is None:
            # Fallback heuristic
            success_prob = 1.0 - features.complexity_score * 0.1
            return max(success_prob, 0.5)
        
        try:
            X = np.array([features.to_vector()]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            success_prob = self.success_prediction_model.predict_proba(X_scaled)[0][1]
            return success_prob
            
        except Exception as e:
            self.logger.error(f"Error predicting compilation success: {e}")
            return 0.8
    
    def add_training_sample(self, features: ShaderFeatures, result: CompilationResult):
        """Add new training sample for online learning"""
        self.training_data.append((features, result))
        self.compilation_history.append(result)
        self.samples_since_update += 1
        
        # Trigger model update if threshold reached
        if (self.online_learning_enabled and 
            self.samples_since_update >= self.model_update_interval):
            self._update_models_online()
    
    def _update_models_online(self):
        """Update models with new data (online learning)"""
        if len(self.training_data) < 50:  # Need minimum samples
            return
        
        try:
            # Use recent data for updates
            recent_data = self.training_data[-self.model_update_interval*2:]
            
            # Retrain models
            self.train_compilation_time_model(recent_data)
            self.train_success_prediction_model(recent_data)
            
            self.samples_since_update = 0
            self.logger.info(f"Updated models with {len(recent_data)} new samples")
            
        except Exception as e:
            self.logger.error(f"Error in online learning update: {e}")
    
    def analyze_gameplay_patterns(self, game_id: str, shader_usage_log: List[Dict]) -> List[GameplayPattern]:
        """Analyze gameplay patterns to predict shader preloading opportunities"""
        patterns = []
        
        try:
            # Group shader usage by time windows
            time_windows = self._group_by_time_windows(shader_usage_log, window_size=30.0)
            
            for window_start, shader_sequence in time_windows.items():
                if len(shader_sequence) < 3:  # Need minimum sequence length
                    continue
                
                # Extract pattern
                shader_hashes = [entry['shader_hash'] for entry in shader_sequence]
                time_intervals = [
                    shader_sequence[i+1]['timestamp'] - shader_sequence[i]['timestamp']
                    for i in range(len(shader_sequence)-1)
                ]
                
                # Determine user action context
                user_action = self._detect_user_action(shader_sequence)
                
                # Calculate frequency (simplified)
                frequency = len(shader_sequence) / 100.0  # Normalize
                
                pattern = GameplayPattern(
                    game_id=game_id,
                    level_or_scene=f"sequence_{window_start}",
                    shader_usage_sequence=shader_hashes,
                    time_intervals_ms=time_intervals,
                    frequency_of_occurrence=frequency,
                    user_action_trigger=user_action
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            self.logger.error(f"Error analyzing gameplay patterns: {e}")
        
        return patterns
    
    def _group_by_time_windows(self, usage_log: List[Dict], window_size: float) -> Dict:
        """Group shader usage by time windows"""
        windows = defaultdict(list)
        
        for entry in usage_log:
            window_start = int(entry['timestamp'] // window_size) * window_size
            windows[window_start].append(entry)
        
        return windows
    
    def _detect_user_action(self, shader_sequence: List[Dict]) -> str:
        """Detect user action type from shader usage patterns"""
        shader_types = [entry.get('shader_type', 'unknown') for entry in shader_sequence]
        
        # Simple heuristics for action detection
        if shader_types.count('ui') > len(shader_types) * 0.5:
            return 'menu_navigation'
        elif shader_types.count('particle') > 3:
            return 'combat'
        elif shader_types.count('post_process') > 2:
            return 'scene_transition'
        else:
            return 'exploration'
    
    def get_prioritized_compilation_queue(self, shader_features_list: List[ShaderFeatures],
                                        steam_deck_model: str = 'LCD',
                                        thermal_state: Dict = None) -> List[Tuple[str, float, float]]:
        """Get prioritized shader compilation queue based on ML predictions"""
        queue = []
        
        for features in shader_features_list:
            # Predict compilation time and success probability
            compile_time = self.predict_compilation_time(features, steam_deck_model, thermal_state)
            success_prob = self.predict_compilation_success(features)
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(features, compile_time, success_prob)
            
            queue.append((features.shader_hash, priority_score, compile_time))
        
        # Sort by priority (higher is better)
        queue.sort(key=lambda x: x[1], reverse=True)
        
        return queue
    
    def _calculate_priority_score(self, features: ShaderFeatures, 
                                compile_time: float, success_prob: float) -> float:
        """Calculate priority score for shader compilation"""
        # Base score from success probability
        base_score = success_prob * 100
        
        # Adjust for compilation time (prefer faster compiles when under pressure)
        time_factor = max(0.1, 1.0 - (compile_time / 1000.0))  # Normalize by 1 second
        
        # Boost important shader types
        type_multipliers = {
            'vertex': 1.2,
            'fragment': 1.1,
            'compute': 0.9,
            'unknown': 0.8
        }
        type_factor = type_multipliers.get(features.stage_type, 1.0)
        
        # Steam Deck specific adjustments
        deck_factor = 1.0
        if features.rdna2_optimal:
            deck_factor += 0.2
        if features.thermal_sensitive:
            deck_factor -= 0.1
        
        priority_score = base_score * time_factor * type_factor * deck_factor
        return max(priority_score, 1.0)
    
    def export_training_data(self, output_path: Path):
        """Export training data for analysis or backup"""
        try:
            export_data = {
                'version': '1.0',
                'samples': [],
                'metadata': {
                    'total_samples': len(self.training_data),
                    'model_performance': self._get_model_performance_stats(),
                    'export_timestamp': time.time()
                }
            }
            
            for features, result in self.training_data:
                sample = {
                    'features': asdict(features),
                    'result': asdict(result)
                }
                export_data['samples'].append(sample)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(self.training_data)} training samples to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting training data: {e}")
    
    def _get_model_performance_stats(self) -> Dict:
        """Get current model performance statistics"""
        stats = {
            'compilation_time_model_loaded': self.compilation_time_model is not None,
            'success_prediction_model_loaded': self.success_prediction_model is not None,
            'scaler_fitted': self.scaler is not None,
            'total_training_samples': len(self.training_data),
            'recent_accuracy': 'unknown'
        }
        
        # Calculate recent prediction accuracy if we have enough recent data
        if len(self.compilation_history) > 10:
            recent_results = list(self.compilation_history)[-10:]
            success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
            stats['recent_success_rate'] = success_rate
        
        return stats