"""
Steam Deck Shader Prediction System
Optimized for AMD RDNA2 GPU with thermal and power constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import pickle
import json
import time
import hashlib
from enum import Enum
from pathlib import Path

# ML imports - using lightweight models for Steam Deck constraints
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# For neural network components (optional, can fallback to sklearn)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    # Check if CUDA/ROCm is available for Steam Deck GPU acceleration
    if torch.cuda.is_available():
        print("GPU acceleration available via CUDA/ROCm")
    else:
        print("Using CPU inference (GPU acceleration not available)")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using sklearn models only")


class ShaderType(Enum):
    """Shader types commonly found in games"""
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    COMPUTE = "compute"
    GEOMETRY = "geometry"
    TESSELLATION = "tessellation"
    RAYTRACING = "raytracing"


class ThermalState(Enum):
    """Steam Deck thermal states - Updated based on research"""
    COOL = "cool"  # < 65°C
    NORMAL = "normal"  # 65-80°C  
    WARM = "warm"  # 80-85°C
    HOT = "hot"  # 85-90°C
    THROTTLING = "throttling"  # > 90°C
    CRITICAL = "critical"  # > 95°C (APU junction limit)


@dataclass
class ShaderMetrics:
    """Metrics for a single shader compilation"""
    shader_hash: str
    shader_type: ShaderType
    bytecode_size: int
    instruction_count: int
    register_pressure: int
    texture_samples: int
    branch_complexity: int
    loop_depth: int
    compilation_time_ms: float
    gpu_temp_celsius: float
    power_draw_watts: float
    memory_used_mb: float
    timestamp: float
    game_id: str
    success: bool = True
    variant_count: int = 1
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to ML feature vector"""
        return np.array([
            self.bytecode_size,
            self.instruction_count,
            self.register_pressure,
            self.texture_samples,
            self.branch_complexity,
            self.loop_depth,
            self.shader_type.value == "compute",  # Binary features for shader types
            self.shader_type.value == "fragment",
            self.shader_type.value == "vertex",
            self.variant_count,
            self.gpu_temp_celsius,
            self.power_draw_watts,
            self.memory_used_mb
        ])


@dataclass
class GameplayPattern:
    """Analyzed gameplay pattern for shader prediction"""
    game_id: str
    scene_id: str
    shader_sequence: List[str]  # Ordered list of shader hashes
    transition_matrix: np.ndarray  # Probability of shader A -> shader B
    common_shaders: List[str]  # Most frequently used shaders
    peak_shader_load: int  # Maximum shaders compiled in a frame
    average_frame_time: float
    

class ShaderCompilationPredictor:
    """
    Main ML model for predicting shader compilation times
    Optimized for Steam Deck's limited resources
    """
    
    def __init__(self, model_type: str = "ensemble", cache_size: int = 1000):
        self.model_type = model_type
        self.cache_size = cache_size
        self.feature_scaler = StandardScaler()
        self.models = {}
        self.compilation_cache = {}  # LRU cache for predictions
        self.cache_order = deque(maxlen=cache_size)
        
        # Initialize models based on type
        self._initialize_models()
        
        # Performance tracking
        self.prediction_history = deque(maxlen=100)
        self.model_version = "1.0.0"
        
    def _initialize_models(self):
        """Initialize ML models optimized for Steam Deck constraints"""
        if self.model_type == "ensemble":
            # Optimized ensemble for Steam Deck (4-core AMD Zen 2)
            self.models['rf'] = RandomForestRegressor(
                n_estimators=30,  # Reduced for faster inference
                max_depth=8,      # Reduced depth for efficiency
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',  # Feature subsampling for speed
                n_jobs=2,  # Conservative thread usage
                random_state=42
            )
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=40,  # Reduced from 50
                max_depth=4,      # Reduced from 5
                learning_rate=0.15,  # Slightly higher for faster convergence
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == "lightweight":
            # Ultra-lightweight single model for minimal resource usage
            from sklearn.ensemble import ExtraTreesRegressor
            self.models['main'] = ExtraTreesRegressor(
                n_estimators=20,  # Very fast ensemble
                max_depth=6,
                min_samples_split=10,
                n_jobs=1,  # Single thread for minimal impact
                random_state=42
            )
        elif self.model_type == "neural" and TORCH_AVAILABLE:
            # Neural network with Steam Deck GPU optimization
            self.models['nn'] = ShaderNeuralNetwork()
            # Set device based on availability
            if torch.cuda.is_available():
                self.models['nn'] = self.models['nn'].cuda()
                print("Neural network using GPU acceleration")
            
    def train(self, shader_metrics: List[ShaderMetrics], 
              validation_split: float = 0.2) -> Dict[str, float]:
        """Train the prediction model on collected shader metrics"""
        
        # Prepare training data
        X = np.array([m.to_feature_vector() for m in shader_metrics])
        y = np.array([m.compilation_time_ms for m in shader_metrics])
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Train models
        results = {}
        for name, model in self.models.items():
            if name == 'nn' and TORCH_AVAILABLE:
                # Special handling for neural network
                results[name] = self._train_neural_network(
                    model, X_train_scaled, y_train, X_val_scaled, y_val
                )
            else:
                # Sklearn models
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_val_scaled)
                mae = np.mean(np.abs(predictions - y_val))
                results[name] = mae
                
        # Clear cache after retraining
        self.compilation_cache.clear()
        self.cache_order.clear()
        
        return results
    
    def predict(self, shader_metrics: ShaderMetrics, 
                use_cache: bool = True) -> Tuple[float, float]:
        """
        Predict compilation time for a shader
        Returns: (predicted_time_ms, confidence)
        """
        
        # Check cache first
        if use_cache and shader_metrics.shader_hash in self.compilation_cache:
            cached = self.compilation_cache[shader_metrics.shader_hash]
            self._update_cache_order(shader_metrics.shader_hash)
            return cached['prediction'], cached['confidence']
        
        # Prepare features
        features = shader_metrics.to_feature_vector().reshape(1, -1)
        features_scaled = self.feature_scaler.transform(features)
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            if name == 'nn' and TORCH_AVAILABLE:
                pred = self._predict_neural_network(model, features_scaled)
            else:
                pred = model.predict(features_scaled)[0]
            predictions.append(pred)
        
        # Combine predictions (ensemble)
        if len(predictions) > 1:
            final_prediction = np.mean(predictions)
            confidence = 1.0 - (np.std(predictions) / np.mean(predictions))
        else:
            final_prediction = predictions[0]
            confidence = 0.8  # Default confidence for single model
            
        # Cache result
        if use_cache:
            self._cache_prediction(
                shader_metrics.shader_hash, 
                final_prediction, 
                confidence
            )
        
        # Track prediction for online learning
        self.prediction_history.append({
            'timestamp': time.time(),
            'prediction': final_prediction,
            'shader_hash': shader_metrics.shader_hash
        })
        
        return final_prediction, confidence
    
    def _cache_prediction(self, shader_hash: str, prediction: float, confidence: float):
        """Manage LRU cache for predictions"""
        if len(self.compilation_cache) >= self.cache_size:
            # Remove oldest item
            oldest = self.cache_order.popleft()
            del self.compilation_cache[oldest]
            
        self.compilation_cache[shader_hash] = {
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self.cache_order.append(shader_hash)
    
    def _update_cache_order(self, shader_hash: str):
        """Update LRU order for cache hit"""
        self.cache_order.remove(shader_hash)
        self.cache_order.append(shader_hash)
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        model_data = {
            'models': self.models,
            'scaler': self.feature_scaler,
            'version': self.model_version,
            'model_type': self.model_type
        }
        joblib.dump(model_data, path)
        
    def load_model(self, path: str):
        """Load trained model from disk"""
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.feature_scaler = model_data['scaler']
        self.model_version = model_data['version']
        self.model_type = model_data['model_type']
        
    def _train_neural_network(self, model, X_train, y_train, X_val, y_val):
        """Train neural network model if PyTorch is available"""
        if not TORCH_AVAILABLE:
            return float('inf')
            
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train.reshape(-1, 1))
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val.reshape(-1, 1))
        
        # Training loop (simplified)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(100):  # Limited epochs for Steam Deck
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
        # Validation
        with torch.no_grad():
            val_pred = model(X_val_t)
            mae = torch.mean(torch.abs(val_pred - y_val_t)).item()
            
        return mae
    
    def _predict_neural_network(self, model, features):
        """Get prediction from neural network"""
        if not TORCH_AVAILABLE:
            return 0.0
            
        with torch.no_grad():
            features_t = torch.FloatTensor(features)
            prediction = model(features_t).item()
            
        return prediction


class ThermalAwareScheduler:
    """
    Thermal-aware shader compilation scheduler for Steam Deck
    Manages compilation timing based on thermal state and power budget
    """
    
    def __init__(self, max_temp: float = 85.0, power_budget: float = 15.0):
        self.max_temp = max_temp
        self.power_budget = power_budget  # Watts allocated for GPU
        self.current_thermal_state = ThermalState.NORMAL
        self.compilation_queue = deque()
        self.priority_queue = []  # High priority compilations
        self.thermal_history = deque(maxlen=60)  # 60 second window
        
        # Updated thermal state thresholds based on Steam Deck research
        self.thermal_thresholds = {
            ThermalState.COOL: (0, 65),
            ThermalState.NORMAL: (65, 80),
            ThermalState.WARM: (80, 85),
            ThermalState.HOT: (85, 90),
            ThermalState.THROTTLING: (90, 95),
            ThermalState.CRITICAL: (95, float('inf'))
        }
        
        # Compilation slots per thermal state (optimized for Steam Deck)
        self.compilation_slots = {
            ThermalState.COOL: 4,        # Full compilation capacity
            ThermalState.NORMAL: 3,      # Slight reduction
            ThermalState.WARM: 2,        # Half capacity
            ThermalState.HOT: 1,         # Minimal compilation
            ThermalState.THROTTLING: 0,  # Stop all compilation
            ThermalState.CRITICAL: 0     # Emergency stop
        }
        
    def update_thermal_state(self, temp: float, power: float):
        """Update current thermal state based on temperature and power"""
        self.thermal_history.append({
            'temp': temp,
            'power': power,
            'timestamp': time.time()
        })
        
        # Determine thermal state
        for state, (min_temp, max_temp) in self.thermal_thresholds.items():
            if min_temp <= temp < max_temp:
                self.current_thermal_state = state
                break
                
    def can_compile_now(self, shader_metrics: ShaderMetrics, 
                        predictor: ShaderCompilationPredictor) -> bool:
        """
        Determine if shader compilation should proceed based on thermal state
        """
        # Get predicted compilation time and power impact
        predicted_time, confidence = predictor.predict(shader_metrics)
        
        # Estimate power impact (simplified model)
        estimated_power = self._estimate_power_impact(shader_metrics, predicted_time)
        
        # Check thermal headroom
        if self.current_thermal_state == ThermalState.THROTTLING:
            return False
            
        # Check power budget
        current_power = self._get_current_power_usage()
        if current_power + estimated_power > self.power_budget:
            return False
            
        # Check compilation slots
        available_slots = self.compilation_slots[self.current_thermal_state]
        if available_slots <= 0:
            return False
            
        return True
    
    def schedule_compilation(self, shader_metrics: ShaderMetrics, 
                           priority: int = 0) -> str:
        """
        Schedule shader compilation with thermal awareness
        Returns: scheduled time or 'immediate' if can compile now
        """
        if priority > 5:  # High priority, add to priority queue
            self.priority_queue.append((shader_metrics, priority))
            return "priority"
        else:
            self.compilation_queue.append(shader_metrics)
            
        # Estimate best compilation window
        if self.current_thermal_state in [ThermalState.COOL, ThermalState.NORMAL]:
            return "immediate"
        elif self.current_thermal_state == ThermalState.WARM:
            return "delayed_5s"
        else:
            return "delayed_30s"
            
    def get_compilation_batch(self, max_batch_size: int = 10) -> List[ShaderMetrics]:
        """
        Get next batch of shaders to compile based on thermal state
        """
        batch = []
        available_slots = self.compilation_slots[self.current_thermal_state]
        
        # Priority compilations first
        while self.priority_queue and len(batch) < min(available_slots, max_batch_size):
            shader, _ = self.priority_queue.pop(0)
            batch.append(shader)
            
        # Regular queue
        while self.compilation_queue and len(batch) < min(available_slots, max_batch_size):
            batch.append(self.compilation_queue.popleft())
            
        return batch
    
    def _estimate_power_impact(self, shader_metrics: ShaderMetrics, 
                              compilation_time_ms: float) -> float:
        """Estimate power draw for shader compilation"""
        # Simplified power model based on shader complexity
        base_power = 5.0  # Base GPU power for compilation
        
        complexity_factor = (
            shader_metrics.instruction_count / 1000.0 +
            shader_metrics.register_pressure / 100.0 +
            shader_metrics.branch_complexity / 10.0
        )
        
        time_factor = compilation_time_ms / 1000.0  # Convert to seconds
        
        return base_power * (1 + complexity_factor * 0.1) * time_factor
    
    def _get_current_power_usage(self) -> float:
        """Get current GPU power usage from thermal history"""
        if not self.thermal_history:
            return 0.0
            
        recent_power = [h['power'] for h in list(self.thermal_history)[-5:]]
        return np.mean(recent_power) if recent_power else 0.0


class GameplayPatternAnalyzer:
    """
    Analyzes gameplay patterns to predict upcoming shader needs
    """
    
    def __init__(self, sequence_length: int = 50, min_confidence: float = 0.7):
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.shader_sequences = {}  # Per-game shader sequences
        self.transition_matrices = {}  # Per-game transition probabilities
        self.scene_patterns = {}  # Scene-specific patterns
        
    def record_shader_usage(self, game_id: str, scene_id: str, 
                           shader_hash: str, timestamp: float):
        """Record shader usage during gameplay"""
        if game_id not in self.shader_sequences:
            self.shader_sequences[game_id] = deque(maxlen=self.sequence_length)
            
        self.shader_sequences[game_id].append({
            'shader_hash': shader_hash,
            'scene_id': scene_id,
            'timestamp': timestamp
        })
        
        # Update scene patterns
        scene_key = f"{game_id}_{scene_id}"
        if scene_key not in self.scene_patterns:
            self.scene_patterns[scene_key] = {
                'shaders': set(),
                'transitions': {},
                'frequency': {}
            }
            
        self.scene_patterns[scene_key]['shaders'].add(shader_hash)
        
        # Update frequency
        freq = self.scene_patterns[scene_key]['frequency']
        freq[shader_hash] = freq.get(shader_hash, 0) + 1
        
    def build_transition_matrix(self, game_id: str) -> np.ndarray:
        """Build Markov chain transition matrix for shader predictions"""
        if game_id not in self.shader_sequences:
            return np.array([])
            
        sequence = list(self.shader_sequences[game_id])
        if len(sequence) < 2:
            return np.array([])
            
        # Get unique shaders
        unique_shaders = list(set(s['shader_hash'] for s in sequence))
        n_shaders = len(unique_shaders)
        shader_to_idx = {s: i for i, s in enumerate(unique_shaders)}
        
        # Build transition counts
        transitions = np.zeros((n_shaders, n_shaders))
        for i in range(len(sequence) - 1):
            curr_idx = shader_to_idx[sequence[i]['shader_hash']]
            next_idx = shader_to_idx[sequence[i + 1]['shader_hash']]
            transitions[curr_idx, next_idx] += 1
            
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transitions / row_sums
        
        self.transition_matrices[game_id] = {
            'matrix': transition_matrix,
            'shader_map': unique_shaders,
            'shader_to_idx': shader_to_idx
        }
        
        return transition_matrix
    
    def predict_next_shaders(self, game_id: str, current_shader: str, 
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next likely shaders based on current shader and patterns
        Returns: List of (shader_hash, probability) tuples
        """
        if game_id not in self.transition_matrices:
            self.build_transition_matrix(game_id)
            
        if game_id not in self.transition_matrices:
            return []
            
        tm = self.transition_matrices[game_id]
        
        if current_shader not in tm['shader_to_idx']:
            return []
            
        curr_idx = tm['shader_to_idx'][current_shader]
        probabilities = tm['matrix'][curr_idx]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        predictions = []
        
        for idx in top_indices:
            if probabilities[idx] >= self.min_confidence:
                shader_hash = tm['shader_map'][idx]
                predictions.append((shader_hash, probabilities[idx]))
                
        return predictions
    
    def get_scene_shaders(self, game_id: str, scene_id: str) -> List[str]:
        """Get commonly used shaders for a specific scene"""
        scene_key = f"{game_id}_{scene_id}"
        if scene_key not in self.scene_patterns:
            return []
            
        # Return shaders sorted by frequency
        freq = self.scene_patterns[scene_key]['frequency']
        sorted_shaders = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [shader for shader, _ in sorted_shaders]
    
    def analyze_patterns(self, game_id: str) -> GameplayPattern:
        """Analyze and return gameplay patterns for a game"""
        if game_id not in self.shader_sequences:
            return None
            
        sequence = list(self.shader_sequences[game_id])
        
        # Build transition matrix if not exists
        if game_id not in self.transition_matrices:
            self.build_transition_matrix(game_id)
            
        # Get most common shaders
        shader_counts = {}
        for item in sequence:
            shader = item['shader_hash']
            shader_counts[shader] = shader_counts.get(shader, 0) + 1
            
        common_shaders = sorted(shader_counts.keys(), 
                               key=lambda x: shader_counts[x], 
                               reverse=True)[:10]
        
        # Calculate peak shader load (shaders per second)
        if len(sequence) > 1:
            time_window = 1.0  # 1 second window
            peak_load = 0
            for i in range(len(sequence)):
                count = 1
                for j in range(i + 1, len(sequence)):
                    if sequence[j]['timestamp'] - sequence[i]['timestamp'] > time_window:
                        break
                    count += 1
                peak_load = max(peak_load, count)
        else:
            peak_load = 1
            
        pattern = GameplayPattern(
            game_id=game_id,
            scene_id="",  # Will be set per-scene
            shader_sequence=[s['shader_hash'] for s in sequence],
            transition_matrix=self.transition_matrices.get(game_id, {}).get('matrix', np.array([])),
            common_shaders=common_shaders,
            peak_shader_load=peak_load,
            average_frame_time=16.67  # Default 60 FPS
        )
        
        return pattern


class PerformanceMetricsCollector:
    """
    Collects and manages performance metrics for ML training
    """
    
    def __init__(self, buffer_size: int = 10000, save_interval: int = 100):
        self.buffer_size = buffer_size
        self.save_interval = save_interval
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.collection_count = 0
        self.start_time = time.time()
        
        # Statistics tracking
        self.stats = {
            'total_compilations': 0,
            'failed_compilations': 0,
            'total_compilation_time': 0.0,
            'avg_compilation_time': 0.0,
            'max_compilation_time': 0.0,
            'min_compilation_time': float('inf')
        }
        
    def collect_shader_metrics(self, 
                              shader_data: Dict[str, Any],
                              compilation_result: Dict[str, Any]) -> ShaderMetrics:
        """Collect metrics from shader compilation"""
        
        # Generate shader hash if not provided
        shader_hash = shader_data.get('hash', self._generate_shader_hash(shader_data))
        
        metrics = ShaderMetrics(
            shader_hash=shader_hash,
            shader_type=ShaderType(shader_data.get('type', 'fragment')),
            bytecode_size=shader_data.get('bytecode_size', 0),
            instruction_count=shader_data.get('instruction_count', 0),
            register_pressure=shader_data.get('register_pressure', 0),
            texture_samples=shader_data.get('texture_samples', 0),
            branch_complexity=shader_data.get('branch_complexity', 0),
            loop_depth=shader_data.get('loop_depth', 0),
            compilation_time_ms=compilation_result.get('time_ms', 0.0),
            gpu_temp_celsius=compilation_result.get('gpu_temp', 70.0),
            power_draw_watts=compilation_result.get('power_draw', 10.0),
            memory_used_mb=compilation_result.get('memory_mb', 0.0),
            timestamp=time.time(),
            game_id=shader_data.get('game_id', 'unknown'),
            success=compilation_result.get('success', True),
            variant_count=shader_data.get('variant_count', 1)
        )
        
        self.metrics_buffer.append(metrics)
        self.collection_count += 1
        
        # Update statistics
        self._update_statistics(metrics)
        
        # Auto-save if interval reached
        if self.collection_count % self.save_interval == 0:
            self.save_metrics()
            
        return metrics
    
    def _generate_shader_hash(self, shader_data: Dict) -> str:
        """Generate unique hash for shader"""
        hash_input = f"{shader_data.get('type', '')}"
        hash_input += f"{shader_data.get('bytecode_size', 0)}"
        hash_input += f"{shader_data.get('instruction_count', 0)}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _update_statistics(self, metrics: ShaderMetrics):
        """Update running statistics"""
        self.stats['total_compilations'] += 1
        if not metrics.success:
            self.stats['failed_compilations'] += 1
            
        self.stats['total_compilation_time'] += metrics.compilation_time_ms
        self.stats['avg_compilation_time'] = (
            self.stats['total_compilation_time'] / self.stats['total_compilations']
        )
        self.stats['max_compilation_time'] = max(
            self.stats['max_compilation_time'], 
            metrics.compilation_time_ms
        )
        self.stats['min_compilation_time'] = min(
            self.stats['min_compilation_time'], 
            metrics.compilation_time_ms
        )
        
    def get_training_data(self, min_samples: int = 100) -> List[ShaderMetrics]:
        """Get collected metrics for training"""
        if len(self.metrics_buffer) < min_samples:
            print(f"Warning: Only {len(self.metrics_buffer)} samples available")
            
        return list(self.metrics_buffer)
    
    def save_metrics(self, filepath: str = "shader_metrics.pkl"):
        """Save collected metrics to disk"""
        data = {
            'metrics': list(self.metrics_buffer),
            'stats': self.stats,
            'collection_time': time.time() - self.start_time
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    def load_metrics(self, filepath: str = "shader_metrics.pkl"):
        """Load metrics from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.metrics_buffer = deque(data['metrics'], maxlen=self.buffer_size)
        self.stats = data['stats']
        
    def export_to_csv(self, filepath: str = "shader_metrics.csv"):
        """Export metrics to CSV for analysis"""
        if not self.metrics_buffer:
            print("No metrics to export")
            return
            
        # Convert to dataframe format
        rows = []
        for m in self.metrics_buffer:
            rows.append({
                'shader_hash': m.shader_hash,
                'shader_type': m.shader_type.value,
                'bytecode_size': m.bytecode_size,
                'instruction_count': m.instruction_count,
                'register_pressure': m.register_pressure,
                'texture_samples': m.texture_samples,
                'branch_complexity': m.branch_complexity,
                'loop_depth': m.loop_depth,
                'compilation_time_ms': m.compilation_time_ms,
                'gpu_temp_celsius': m.gpu_temp_celsius,
                'power_draw_watts': m.power_draw_watts,
                'memory_used_mb': m.memory_used_mb,
                'timestamp': m.timestamp,
                'game_id': m.game_id,
                'success': m.success,
                'variant_count': m.variant_count
            })
            
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Exported {len(rows)} metrics to {filepath}")


# Optional Neural Network Model (if PyTorch available)
if TORCH_AVAILABLE:
    class ShaderNeuralNetwork(nn.Module):
        """
        Lightweight neural network for shader compilation prediction
        Optimized for Steam Deck's limited resources
        """
        
        def __init__(self, input_size: int = 13, hidden_size: int = 64):
            super(ShaderNeuralNetwork, self).__init__()
            
            # Small network to minimize overhead
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, 1)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x


# Main orchestrator class
class SteamDeckShaderPredictor:
    """
    Main orchestrator for the entire shader prediction system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.predictor = ShaderCompilationPredictor(
            model_type=self.config['model_type'],
            cache_size=self.config['cache_size']
        )
        
        self.scheduler = ThermalAwareScheduler(
            max_temp=self.config['max_temp'],
            power_budget=self.config['power_budget']
        )
        
        self.pattern_analyzer = GameplayPatternAnalyzer(
            sequence_length=self.config['sequence_length']
        )
        
        self.metrics_collector = PerformanceMetricsCollector(
            buffer_size=self.config['buffer_size']
        )
        
        self.is_running = False
        
    def _default_config(self) -> Dict:
        """Default configuration for Steam Deck"""
        return {
            'model_type': 'ensemble',
            'cache_size': 1000,
            'max_temp': 85.0,
            'power_budget': 15.0,
            'sequence_length': 50,
            'buffer_size': 10000,
            'auto_train_interval': 1000,
            'min_training_samples': 500
        }
        
    def process_shader_compilation(self, 
                                  shader_data: Dict,
                                  gpu_state: Dict) -> Dict:
        """
        Main entry point for processing shader compilation requests
        """
        
        # Update thermal state
        self.scheduler.update_thermal_state(
            gpu_state.get('temperature', 70.0),
            gpu_state.get('power', 10.0)
        )
        
        # Create metrics from shader data
        temp_metrics = ShaderMetrics(
            shader_hash=shader_data.get('hash', ''),
            shader_type=ShaderType(shader_data.get('type', 'fragment')),
            bytecode_size=shader_data.get('bytecode_size', 0),
            instruction_count=shader_data.get('instruction_count', 0),
            register_pressure=shader_data.get('register_pressure', 0),
            texture_samples=shader_data.get('texture_samples', 0),
            branch_complexity=shader_data.get('branch_complexity', 0),
            loop_depth=shader_data.get('loop_depth', 0),
            compilation_time_ms=0,  # Will be predicted
            gpu_temp_celsius=gpu_state.get('temperature', 70.0),
            power_draw_watts=gpu_state.get('power', 10.0),
            memory_used_mb=gpu_state.get('memory_used', 0),
            timestamp=time.time(),
            game_id=shader_data.get('game_id', 'unknown'),
            variant_count=shader_data.get('variant_count', 1)
        )
        
        # Get prediction
        predicted_time, confidence = self.predictor.predict(temp_metrics)
        
        # Check if we can compile now
        can_compile = self.scheduler.can_compile_now(temp_metrics, self.predictor)
        
        # Schedule if needed
        if not can_compile:
            schedule = self.scheduler.schedule_compilation(
                temp_metrics, 
                priority=shader_data.get('priority', 0)
            )
        else:
            schedule = "immediate"
            
        # Record pattern for future predictions
        self.pattern_analyzer.record_shader_usage(
            shader_data.get('game_id', 'unknown'),
            shader_data.get('scene_id', 'default'),
            shader_data.get('hash', ''),
            time.time()
        )
        
        # Predict next shaders
        next_predictions = self.pattern_analyzer.predict_next_shaders(
            shader_data.get('game_id', 'unknown'),
            shader_data.get('hash', ''),
            top_k=3
        )
        
        return {
            'predicted_compilation_time_ms': predicted_time,
            'confidence': confidence,
            'can_compile_now': can_compile,
            'schedule': schedule,
            'thermal_state': self.scheduler.current_thermal_state.value,
            'next_shader_predictions': next_predictions,
            'timestamp': time.time()
        }
        
    def record_compilation_result(self, 
                                 shader_data: Dict,
                                 compilation_result: Dict):
        """Record actual compilation results for training"""
        metrics = self.metrics_collector.collect_shader_metrics(
            shader_data, 
            compilation_result
        )
        
        # Auto-train if enough samples
        if (self.metrics_collector.collection_count % 
            self.config['auto_train_interval'] == 0):
            
            training_data = self.metrics_collector.get_training_data(
                min_samples=self.config['min_training_samples']
            )
            
            if len(training_data) >= self.config['min_training_samples']:
                print(f"Auto-training with {len(training_data)} samples...")
                results = self.predictor.train(training_data)
                print(f"Training results: {results}")
                
    def get_statistics(self) -> Dict:
        """Get current system statistics"""
        return {
            'metrics_stats': self.metrics_collector.stats,
            'thermal_state': self.scheduler.current_thermal_state.value,
            'cache_size': len(self.predictor.compilation_cache),
            'queue_size': len(self.scheduler.compilation_queue),
            'prediction_history_size': len(self.predictor.prediction_history)
        }
        
    def save_state(self, directory: str = "."):
        """Save entire system state"""
        # Save predictor model
        self.predictor.save_model(f"{directory}/predictor_model.pkl")
        
        # Save metrics
        self.metrics_collector.save_metrics(f"{directory}/metrics.pkl")
        
        # Save configuration
        with open(f"{directory}/config.json", 'w') as f:
            json.dump(self.config, f)
            
        print(f"System state saved to {directory}")
        
    def load_state(self, directory: str = "."):
        """Load system state"""
        # Load predictor model
        self.predictor.load_model(f"{directory}/predictor_model.pkl")
        
        # Load metrics
        self.metrics_collector.load_metrics(f"{directory}/metrics.pkl")
        
        # Load configuration
        with open(f"{directory}/config.json", 'r') as f:
            self.config = json.load(f)
            
        print(f"System state loaded from {directory}")


if __name__ == "__main__":
    # Example usage
    print("Steam Deck Shader Prediction System initialized")
    
    # Create system
    system = SteamDeckShaderPredictor()
    
    # Example shader data
    example_shader = {
        'hash': 'abc123def456',
        'type': 'fragment',
        'bytecode_size': 2048,
        'instruction_count': 150,
        'register_pressure': 32,
        'texture_samples': 4,
        'branch_complexity': 3,
        'loop_depth': 2,
        'game_id': 'game_001',
        'scene_id': 'main_menu',
        'priority': 1,
        'variant_count': 3
    }
    
    # Example GPU state
    gpu_state = {
        'temperature': 72.5,
        'power': 12.3,
        'memory_used': 512
    }
    
    # Process shader compilation request
    result = system.process_shader_compilation(example_shader, gpu_state)
    print(f"Prediction result: {json.dumps(result, indent=2)}")
    
    # Simulate compilation result
    compilation_result = {
        'time_ms': 45.2,
        'gpu_temp': 73.1,
        'power_draw': 13.5,
        'memory_mb': 48,
        'success': True
    }
    
    # Record actual result for training
    system.record_compilation_result(example_shader, compilation_result)
    
    # Get statistics
    stats = system.get_statistics()
    print(f"System statistics: {json.dumps(stats, indent=2)}")