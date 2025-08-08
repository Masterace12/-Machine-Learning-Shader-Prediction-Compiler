#!/usr/bin/env python3
"""
Gameplay Pattern Analysis and Predictive Shader Loading System
Uses machine learning to analyze gameplay patterns and predict which shaders
will be needed next, enabling proactive compilation and caching.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    np = None

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

@dataclass
class GameplayState:
    """Current gameplay state for pattern matching"""
    game_id: str
    level_or_scene: str
    player_position: Tuple[float, float, float]  # x, y, z coordinates
    camera_direction: Tuple[float, float, float]  # pitch, yaw, roll
    player_action: str  # idle, walking, running, combat, menu, etc.
    ui_state: str      # in_game, menu, inventory, map, etc.
    timestamp: float
    performance_profile: str  # high, medium, low based on current settings
    
@dataclass
class ShaderUsageEvent:
    """Shader usage event during gameplay"""
    timestamp: float
    shader_hash: str
    shader_type: str
    usage_context: str  # rendering_context: ui, world, effects, post_process
    gameplay_state: GameplayState
    compilation_required: bool
    cache_hit: bool
    
@dataclass
class PredictionResult:
    """Result of shader usage prediction"""
    shader_hash: str
    probability: float
    confidence: float
    predicted_usage_time: float  # Seconds from now
    context: str
    priority_score: float

class GameplayShaderPredictor:
    """Predicts shader usage patterns based on gameplay analysis"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path.home() / ".steam-deck-shader-ml"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Pattern analysis models
        self.sequence_predictor = None
        self.context_classifier = None
        self.priority_ranker = None
        self.pattern_clusterer = None
        
        # Data structures for pattern analysis
        self.shader_usage_history = deque(maxlen=5000)
        self.gameplay_patterns = defaultdict(list)
        self.shader_sequences = defaultdict(list)
        self.context_transitions = defaultdict(lambda: defaultdict(int))
        
        # Game-specific pattern libraries
        self.game_pattern_libraries = {}
        self.common_sequences = defaultdict(list)
        
        # Prediction caching
        self.prediction_cache = {}
        self.cache_timeout = 30.0  # 30 seconds
        
        # Performance tracking
        self.prediction_accuracy = deque(maxlen=100)
        self.preload_success_rate = deque(maxlen=100)
        
        # Configuration
        self.prediction_horizon = 10.0  # Predict 10 seconds ahead
        self.min_confidence_threshold = 0.6
        self.max_preload_shaders = 20
        
        self._load_models()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for gameplay predictor"""
        logger = logging.getLogger('GameplayShaderPredictor')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.model_dir / 'gameplay_predictor.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_models(self):
        """Load pre-trained models"""
        if not HAS_SKLEARN or not HAS_JOBLIB:
            self.logger.warning("sklearn/joblib not available, using heuristic prediction")
            return
        
        model_files = {
            'sequence_predictor': 'sequence_predictor.pkl',
            'context_classifier': 'context_classifier.pkl',
            'priority_ranker': 'priority_ranker.pkl',
            'pattern_clusterer': 'pattern_clusterer.pkl'
        }
        
        for attr_name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    setattr(self, attr_name, joblib.load(model_path))
                    self.logger.info(f"Loaded {attr_name} from {model_path}")
                except Exception as e:
                    self.logger.error(f"Error loading {attr_name}: {e}")
    
    def _save_models(self):
        """Save trained models"""
        if not HAS_JOBLIB:
            return
        
        models = {
            'sequence_predictor': self.sequence_predictor,
            'context_classifier': self.context_classifier,
            'priority_ranker': self.priority_ranker,
            'pattern_clusterer': self.pattern_clusterer
        }
        
        for attr_name, model in models.items():
            if model is not None:
                model_path = self.model_dir / f"{attr_name}.pkl"
                try:
                    joblib.dump(model, model_path)
                    self.logger.info(f"Saved {attr_name} to {model_path}")
                except Exception as e:
                    self.logger.error(f"Error saving {attr_name}: {e}")
    
    def record_shader_usage(self, shader_hash: str, shader_type: str, 
                          usage_context: str, gameplay_state: GameplayState,
                          compilation_required: bool = False, cache_hit: bool = False):
        """Record shader usage event for pattern analysis"""
        event = ShaderUsageEvent(
            timestamp=time.time(),
            shader_hash=shader_hash,
            shader_type=shader_type,
            usage_context=usage_context,
            gameplay_state=gameplay_state,
            compilation_required=compilation_required,
            cache_hit=cache_hit
        )
        
        self.shader_usage_history.append(event)
        
        # Update pattern libraries
        self._update_pattern_libraries(event)
        
        # Check prediction accuracy if we had a prediction for this shader
        self._validate_prediction(event)
    
    def _update_pattern_libraries(self, event: ShaderUsageEvent):
        """Update pattern libraries with new usage event"""
        game_id = event.gameplay_state.game_id
        
        # Add to game-specific patterns
        if game_id not in self.game_pattern_libraries:
            self.game_pattern_libraries[game_id] = {
                'shader_sequences': defaultdict(list),
                'context_patterns': defaultdict(list),
                'state_transitions': defaultdict(int),
                'common_shaders': defaultdict(int)
            }
        
        game_lib = self.game_pattern_libraries[game_id]
        
        # Track shader usage frequency
        game_lib['common_shaders'][event.shader_hash] += 1
        
        # Track context patterns
        context_key = f"{event.gameplay_state.ui_state}_{event.gameplay_state.player_action}"
        game_lib['context_patterns'][context_key].append(event.shader_hash)
        
        # Track sequences (last 5 shaders)
        recent_shaders = [e.shader_hash for e in list(self.shader_usage_history)[-6:-1]]
        if len(recent_shaders) >= 1:
            sequence_key = '_'.join(recent_shaders[-3:])  # Last 3 shaders as sequence
            game_lib['shader_sequences'][sequence_key].append(event.shader_hash)
    
    def _validate_prediction(self, event: ShaderUsageEvent):
        """Validate previous predictions against actual usage"""
        current_time = time.time()
        cache_key = f"{event.gameplay_state.game_id}_{event.shader_hash}"
        
        if cache_key in self.prediction_cache:
            prediction_data = self.prediction_cache[cache_key]
            prediction_time = prediction_data['timestamp']
            predicted_prob = prediction_data['probability']
            
            # Check if prediction was within time window
            if current_time - prediction_time <= self.prediction_horizon:
                # Successful prediction
                accuracy = min(predicted_prob, 1.0)
                self.prediction_accuracy.append(accuracy)
                
                # Remove from cache
                del self.prediction_cache[cache_key]
    
    def predict_upcoming_shaders(self, current_state: GameplayState, 
                                max_predictions: int = None) -> List[PredictionResult]:
        """Predict shaders likely to be used in the near future"""
        max_predictions = max_predictions or self.max_preload_shaders
        predictions = []
        
        # Use multiple prediction strategies and combine results
        sequence_predictions = self._predict_from_sequences(current_state)
        context_predictions = self._predict_from_context(current_state)
        pattern_predictions = self._predict_from_patterns(current_state)
        
        # Combine and rank predictions
        all_predictions = {}
        
        # Weight predictions from different sources
        weights = {
            'sequence': 0.4,
            'context': 0.35,
            'pattern': 0.25
        }
        
        for source, preds in [
            ('sequence', sequence_predictions),
            ('context', context_predictions),
            ('pattern', pattern_predictions)
        ]:
            weight = weights[source]
            for pred in preds:
                shader_hash = pred.shader_hash
                if shader_hash not in all_predictions:
                    all_predictions[shader_hash] = pred
                    all_predictions[shader_hash].probability *= weight
                else:
                    # Combine probabilities
                    existing_prob = all_predictions[shader_hash].probability
                    new_prob = pred.probability * weight
                    all_predictions[shader_hash].probability = min(existing_prob + new_prob, 1.0)
                    all_predictions[shader_hash].confidence = max(
                        all_predictions[shader_hash].confidence, pred.confidence
                    )
        
        # Filter by confidence threshold and sort by priority
        filtered_predictions = [
            pred for pred in all_predictions.values()
            if pred.confidence >= self.min_confidence_threshold
        ]
        
        # Calculate final priority scores
        for pred in filtered_predictions:
            pred.priority_score = self._calculate_priority_score(pred, current_state)
        
        # Sort by priority score
        filtered_predictions.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Limit to max predictions
        final_predictions = filtered_predictions[:max_predictions]
        
        # Cache predictions for validation
        self._cache_predictions(final_predictions, current_state)
        
        return final_predictions
    
    def _predict_from_sequences(self, current_state: GameplayState) -> List[PredictionResult]:
        """Predict based on recent shader usage sequences"""
        predictions = []
        game_id = current_state.game_id
        
        if game_id not in self.game_pattern_libraries:
            return predictions
        
        game_lib = self.game_pattern_libraries[game_id]
        
        # Get recent shader sequence
        recent_events = [e for e in list(self.shader_usage_history)[-10:] 
                        if e.gameplay_state.game_id == game_id]
        
        if len(recent_events) < 2:
            return predictions
        
        recent_sequence = [e.shader_hash for e in recent_events[-3:]]
        sequence_key = '_'.join(recent_sequence)
        
        # Look for matching sequences
        if sequence_key in game_lib['shader_sequences']:
            next_shaders = game_lib['shader_sequences'][sequence_key]
            shader_counts = defaultdict(int)
            
            for shader in next_shaders:
                shader_counts[shader] += 1
            
            total_occurrences = sum(shader_counts.values())
            
            for shader_hash, count in shader_counts.items():
                probability = count / total_occurrences
                confidence = min(count / 10.0, 1.0)  # Higher confidence with more data
                
                prediction = PredictionResult(
                    shader_hash=shader_hash,
                    probability=probability,
                    confidence=confidence,
                    predicted_usage_time=time.time() + (probability * self.prediction_horizon),
                    context='sequence_prediction',
                    priority_score=0.0  # Will be calculated later
                )
                
                predictions.append(prediction)
        
        return predictions
    
    def _predict_from_context(self, current_state: GameplayState) -> List[PredictionResult]:
        """Predict based on current gameplay context"""
        predictions = []
        game_id = current_state.game_id
        
        if game_id not in self.game_pattern_libraries:
            return predictions
        
        game_lib = self.game_pattern_libraries[game_id]
        context_key = f"{current_state.ui_state}_{current_state.player_action}"
        
        if context_key in game_lib['context_patterns']:
            context_shaders = game_lib['context_patterns'][context_key]
            shader_counts = defaultdict(int)
            
            for shader in context_shaders:
                shader_counts[shader] += 1
            
            total_uses = sum(shader_counts.values())
            
            for shader_hash, count in shader_counts.items():
                probability = count / total_uses
                confidence = min(count / 5.0, 1.0)
                
                prediction = PredictionResult(
                    shader_hash=shader_hash,
                    probability=probability,
                    confidence=confidence,
                    predicted_usage_time=time.time() + 2.0,  # Context changes happen quickly
                    context=f'context_{context_key}',
                    priority_score=0.0
                )
                
                predictions.append(prediction)
        
        return predictions
    
    def _predict_from_patterns(self, current_state: GameplayState) -> List[PredictionResult]:
        """Predict using learned gameplay patterns"""
        predictions = []
        game_id = current_state.game_id
        
        if game_id not in self.game_pattern_libraries:
            return predictions
        
        game_lib = self.game_pattern_libraries[game_id]
        
        # Predict based on most common shaders for this game
        common_shaders = game_lib['common_shaders']
        if not common_shaders:
            return predictions
        
        total_usage = sum(common_shaders.values())
        
        # Get top shaders by frequency
        sorted_shaders = sorted(common_shaders.items(), key=lambda x: x[1], reverse=True)
        
        for shader_hash, usage_count in sorted_shaders[:10]:  # Top 10
            base_probability = usage_count / total_usage
            
            # Adjust probability based on recency
            recent_usage = self._get_recent_usage_factor(shader_hash, game_id)
            adjusted_probability = base_probability * (1.0 + recent_usage)
            
            confidence = min(usage_count / 20.0, 1.0)
            
            prediction = PredictionResult(
                shader_hash=shader_hash,
                probability=min(adjusted_probability, 1.0),
                confidence=confidence,
                predicted_usage_time=time.time() + 5.0,  # General pattern prediction
                context='frequency_pattern',
                priority_score=0.0
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _get_recent_usage_factor(self, shader_hash: str, game_id: str) -> float:
        """Get recency factor for shader usage"""
        current_time = time.time()
        recent_threshold = 300.0  # 5 minutes
        
        recent_uses = [
            e for e in self.shader_usage_history
            if (e.shader_hash == shader_hash and 
                e.gameplay_state.game_id == game_id and
                current_time - e.timestamp < recent_threshold)
        ]
        
        if not recent_uses:
            return 0.0
        
        # Higher factor for more recent usage
        recency_factor = len(recent_uses) / 10.0  # Normalize
        return min(recency_factor, 1.0)
    
    def _calculate_priority_score(self, prediction: PredictionResult, 
                                current_state: GameplayState) -> float:
        """Calculate final priority score for shader preloading"""
        # Base score from probability and confidence
        base_score = prediction.probability * prediction.confidence
        
        # Context-based adjustments
        context_multipliers = {
            'sequence_prediction': 1.5,  # Sequence predictions are more reliable
            'context_ui_state': 1.3,     # UI context is predictable
            'context_combat': 1.2,       # Combat shaders are important
            'frequency_pattern': 1.0      # Base frequency patterns
        }
        
        context_multiplier = 1.0
        for context, multiplier in context_multipliers.items():
            if context in prediction.context:
                context_multiplier = multiplier
                break
        
        # Time-based priority (sooner predictions are more valuable)
        time_until_use = prediction.predicted_usage_time - time.time()
        time_factor = max(0.1, 1.0 - (time_until_use / self.prediction_horizon))
        
        # Shader type priority
        type_priorities = {
            'vertex': 1.2,
            'fragment': 1.1,
            'compute': 0.9,
            'ui': 1.3,      # UI shaders are critical for responsiveness
            'post_process': 0.8
        }
        
        # Extract shader type from context or use default
        shader_type_priority = 1.0
        for shader_type, priority in type_priorities.items():
            if shader_type in prediction.context.lower():
                shader_type_priority = priority
                break
        
        priority_score = (base_score * context_multiplier * 
                         time_factor * shader_type_priority)
        
        return min(priority_score, 10.0)  # Cap at 10.0
    
    def _cache_predictions(self, predictions: List[PredictionResult], 
                          current_state: GameplayState):
        """Cache predictions for accuracy validation"""
        current_time = time.time()
        
        for prediction in predictions:
            cache_key = f"{current_state.game_id}_{prediction.shader_hash}"
            self.prediction_cache[cache_key] = {
                'timestamp': current_time,
                'probability': prediction.probability,
                'confidence': prediction.confidence,
                'context': prediction.context
            }
        
        # Clean old cache entries
        cutoff_time = current_time - self.cache_timeout
        expired_keys = [
            key for key, data in self.prediction_cache.items()
            if data['timestamp'] < cutoff_time
        ]
        
        for key in expired_keys:
            del self.prediction_cache[key]
    
    def train_sequence_model(self, min_samples: int = 100):
        """Train sequence prediction model using collected data"""
        if not HAS_SKLEARN or len(self.shader_usage_history) < min_samples:
            self.logger.warning("Insufficient data for training sequence model")
            return
        
        try:
            # Prepare training data
            sequences = []
            targets = []
            
            # Create sequences of shader usage
            events_by_game = defaultdict(list)
            for event in self.shader_usage_history:
                events_by_game[event.gameplay_state.game_id].append(event)
            
            sequence_length = 5
            for game_id, events in events_by_game.items():
                if len(events) < sequence_length + 1:
                    continue
                
                for i in range(len(events) - sequence_length):
                    # Create sequence features
                    sequence_events = events[i:i + sequence_length]
                    target_event = events[i + sequence_length]
                    
                    # Extract features from sequence
                    sequence_features = self._extract_sequence_features(sequence_events)
                    target_shader = target_event.shader_hash
                    
                    sequences.append(sequence_features)
                    targets.append(target_shader)
            
            if len(sequences) < min_samples:
                self.logger.warning(f"Only {len(sequences)} sequences available, need {min_samples}")
                return
            
            # Train classifier
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            encoded_targets = label_encoder.fit_transform(targets)
            
            self.sequence_predictor = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1
            )
            
            self.sequence_predictor.fit(sequences, encoded_targets)
            
            # Store label encoder
            joblib.dump(label_encoder, self.model_dir / 'sequence_label_encoder.pkl')
            
            self.logger.info(f"Trained sequence model with {len(sequences)} samples")
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Error training sequence model: {e}")
    
    def _extract_sequence_features(self, events: List[ShaderUsageEvent]) -> List[float]:
        """Extract features from shader usage sequence"""
        features = []
        
        # Basic sequence features
        shader_types = [e.shader_type for e in events]
        usage_contexts = [e.usage_context for e in events]
        
        # Shader type distribution
        type_counts = defaultdict(int)
        for shader_type in shader_types:
            type_counts[shader_type] += 1
        
        # Normalize by sequence length
        for shader_type in ['vertex', 'fragment', 'compute', 'geometry', 'ui']:
            features.append(type_counts[shader_type] / len(events))
        
        # Context distribution
        context_counts = defaultdict(int)
        for context in usage_contexts:
            context_counts[context] += 1
        
        for context in ['world', 'ui', 'effects', 'post_process']:
            features.append(context_counts[context] / len(events))
        
        # Temporal features
        time_diffs = []
        for i in range(1, len(events)):
            time_diff = events[i].timestamp - events[i-1].timestamp
            time_diffs.append(time_diff)
        
        if time_diffs:
            features.extend([
                np.mean(time_diffs),
                np.std(time_diffs),
                min(time_diffs),
                max(time_diffs)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Gameplay state features
        ui_states = [e.gameplay_state.ui_state for e in events]
        player_actions = [e.gameplay_state.player_action for e in events]
        
        # Most common UI state and action in sequence
        ui_state_counts = defaultdict(int)
        action_counts = defaultdict(int)
        
        for ui_state in ui_states:
            ui_state_counts[ui_state] += 1
        for action in player_actions:
            action_counts[action] += 1
        
        # Add dominant state/action as features
        if ui_state_counts:
            dominant_ui = max(ui_state_counts, key=ui_state_counts.get)
            features.append(1.0 if dominant_ui == 'in_game' else 0.0)
            features.append(1.0 if dominant_ui == 'menu' else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        if action_counts:
            dominant_action = max(action_counts, key=action_counts.get)
            features.append(1.0 if dominant_action == 'combat' else 0.0)
            features.append(1.0 if dominant_action == 'walking' else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def get_prediction_accuracy_stats(self) -> Dict:
        """Get statistics about prediction accuracy"""
        if not self.prediction_accuracy:
            return {'status': 'no_data'}
        
        accuracy_values = list(self.prediction_accuracy)
        
        return {
            'average_accuracy': sum(accuracy_values) / len(accuracy_values),
            'min_accuracy': min(accuracy_values),
            'max_accuracy': max(accuracy_values),
            'total_predictions': len(accuracy_values),
            'recent_trend': accuracy_values[-10:] if len(accuracy_values) >= 10 else accuracy_values
        }
    
    def analyze_shader_patterns(self, game_id: str) -> Dict:
        """Analyze shader usage patterns for a specific game"""
        if game_id not in self.game_pattern_libraries:
            return {'error': 'No data for game'}
        
        game_lib = self.game_pattern_libraries[game_id]
        
        # Most common shaders
        common_shaders = dict(sorted(
            game_lib['common_shaders'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])
        
        # Most predictable sequences
        predictable_sequences = {}
        for sequence, next_shaders in game_lib['shader_sequences'].items():
            if len(next_shaders) >= 3:  # At least 3 occurrences
                shader_counts = defaultdict(int)
                for shader in next_shaders:
                    shader_counts[shader] += 1
                
                # Calculate predictability (how often the most common next shader occurs)
                total = sum(shader_counts.values())
                most_common_count = max(shader_counts.values())
                predictability = most_common_count / total
                
                if predictability > 0.5:  # More than 50% predictable
                    predictable_sequences[sequence] = {
                        'predictability': predictability,
                        'most_likely_next': max(shader_counts, key=shader_counts.get),
                        'total_occurrences': total
                    }
        
        return {
            'game_id': game_id,
            'total_shader_uses': sum(game_lib['common_shaders'].values()),
            'unique_shaders': len(game_lib['common_shaders']),
            'common_shaders': common_shaders,
            'predictable_sequences': dict(sorted(
                predictable_sequences.items(),
                key=lambda x: x[1]['predictability'],
                reverse=True
            )[:10]),
            'context_patterns': len(game_lib['context_patterns'])
        }
    
    def export_pattern_library(self, game_id: str, output_path: Path):
        """Export pattern library for a specific game"""
        if game_id not in self.game_pattern_libraries:
            self.logger.error(f"No pattern library for game {game_id}")
            return
        
        try:
            library_data = {
                'game_id': game_id,
                'version': '1.0',
                'export_timestamp': time.time(),
                'patterns': self.game_pattern_libraries[game_id],
                'statistics': self.analyze_shader_patterns(game_id)
            }
            
            with open(output_path, 'w') as f:
                json.dump(library_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported pattern library for {game_id} to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting pattern library: {e}")