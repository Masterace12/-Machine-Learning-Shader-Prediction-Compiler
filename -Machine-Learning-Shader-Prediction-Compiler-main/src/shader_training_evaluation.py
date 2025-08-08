"""
Training and Evaluation Script for Steam Deck Shader Prediction
Provides tools for training models, evaluating performance, and generating synthetic data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import time
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Import our systems
from shader_prediction_system import (
    SteamDeckShaderPredictor, 
    ShaderMetrics, 
    ShaderType,
    PerformanceMetricsCollector,
    GameplayPatternAnalyzer
)
from steam_deck_integration import (
    SteamDeckOptimizedSystem,
    SteamDeckHardwareState,
    create_example_game_profiles
)


class SyntheticDataGenerator:
    """
    Generate realistic synthetic shader data for training and testing
    Based on real Steam Deck game patterns
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Define realistic ranges based on Steam Deck games
        self.shader_profiles = {
            ShaderType.VERTEX: {
                'bytecode_size': (512, 2048),
                'instruction_count': (20, 150),
                'register_pressure': (8, 32),
                'texture_samples': (0, 4),
                'branch_complexity': (0, 3),
                'loop_depth': (0, 2),
                'base_compile_time': 15.0,
                'complexity_factor': 0.8
            },
            ShaderType.FRAGMENT: {
                'bytecode_size': (1024, 8192),
                'instruction_count': (50, 500),
                'register_pressure': (16, 64),
                'texture_samples': (2, 16),
                'branch_complexity': (1, 8),
                'loop_depth': (0, 4),
                'base_compile_time': 35.0,
                'complexity_factor': 1.2
            },
            ShaderType.COMPUTE: {
                'bytecode_size': (2048, 16384),
                'instruction_count': (100, 1000),
                'register_pressure': (24, 96),
                'texture_samples': (0, 8),
                'branch_complexity': (2, 12),
                'loop_depth': (1, 6),
                'base_compile_time': 85.0,
                'complexity_factor': 1.8
            }
        }
        
        # Game-specific patterns
        self.game_patterns = {
            'cyberpunk': {
                'shader_distribution': {
                    ShaderType.VERTEX: 0.2,
                    ShaderType.FRAGMENT: 0.7,
                    ShaderType.COMPUTE: 0.1
                },
                'complexity_multiplier': 1.4,
                'variant_range': (1, 8)
            },
            'elden_ring': {
                'shader_distribution': {
                    ShaderType.VERTEX: 0.3,
                    ShaderType.FRAGMENT: 0.6,
                    ShaderType.COMPUTE: 0.1
                },
                'complexity_multiplier': 1.1,
                'variant_range': (1, 5)
            },
            'portal2': {
                'shader_distribution': {
                    ShaderType.VERTEX: 0.4,
                    ShaderType.FRAGMENT: 0.6,
                    ShaderType.COMPUTE: 0.0
                },
                'complexity_multiplier': 0.7,
                'variant_range': (1, 3)
            }
        }
        
    def generate_shader_metrics(self, n_samples: int = 1000, 
                              game_type: str = 'mixed') -> List[ShaderMetrics]:
        """Generate realistic shader metrics for training"""
        
        metrics = []
        
        for i in range(n_samples):
            # Choose game pattern
            if game_type == 'mixed':
                game_id = random.choice(list(self.game_patterns.keys()))
            else:
                game_id = game_type
                
            pattern = self.game_patterns.get(game_id, self.game_patterns['elden_ring'])
            
            # Choose shader type based on game pattern
            shader_type = np.random.choice(
                list(pattern['shader_distribution'].keys()),
                p=list(pattern['shader_distribution'].values())
            )
            
            profile = self.shader_profiles[shader_type]
            
            # Generate shader characteristics
            bytecode_size = np.random.randint(*profile['bytecode_size'])
            instruction_count = np.random.randint(*profile['instruction_count'])
            register_pressure = np.random.randint(*profile['register_pressure'])
            texture_samples = np.random.randint(*profile['texture_samples'])
            branch_complexity = np.random.randint(*profile['branch_complexity'])
            loop_depth = np.random.randint(*profile['loop_depth'])
            variant_count = np.random.randint(*pattern['variant_range'])
            
            # Generate thermal conditions
            gpu_temp = np.random.normal(72, 8)  # Normal distribution around 72°C
            gpu_temp = np.clip(gpu_temp, 55, 95)  # Realistic range
            
            # Power based on temperature (higher temp = higher power)
            power_base = 8 + (gpu_temp - 60) * 0.2
            power_draw = np.random.normal(power_base, 1.5)
            power_draw = np.clip(power_draw, 5, 18)
            
            # Memory usage
            memory_used = np.random.uniform(32, 256)
            
            # Calculate realistic compilation time
            compilation_time = self._calculate_compilation_time(
                profile, pattern, bytecode_size, instruction_count,
                register_pressure, texture_samples, branch_complexity,
                loop_depth, variant_count, gpu_temp, power_draw
            )
            
            # Add noise and realistic variations
            compilation_time += np.random.normal(0, compilation_time * 0.1)
            compilation_time = max(1.0, compilation_time)  # Minimum 1ms
            
            # Create metrics object
            shader_hash = f"{game_id}_{shader_type.value}_{i:06d}"
            
            metrics.append(ShaderMetrics(
                shader_hash=shader_hash,
                shader_type=shader_type,
                bytecode_size=bytecode_size,
                instruction_count=instruction_count,
                register_pressure=register_pressure,
                texture_samples=texture_samples,
                branch_complexity=branch_complexity,
                loop_depth=loop_depth,
                compilation_time_ms=compilation_time,
                gpu_temp_celsius=gpu_temp,
                power_draw_watts=power_draw,
                memory_used_mb=memory_used,
                timestamp=time.time() + i,
                game_id=game_id,
                success=np.random.random() > 0.02,  # 2% failure rate
                variant_count=variant_count
            ))
            
        return metrics
        
    def _calculate_compilation_time(self, profile: Dict, pattern: Dict,
                                   bytecode_size: int, instruction_count: int,
                                   register_pressure: int, texture_samples: int,
                                   branch_complexity: int, loop_depth: int,
                                   variant_count: int, gpu_temp: float,
                                   power_draw: float) -> float:
        """Calculate realistic compilation time based on shader characteristics"""
        
        # Base time from profile
        base_time = profile['base_compile_time']
        
        # Size factor
        size_factor = (bytecode_size / 2048) ** 0.7
        
        # Instruction complexity
        instruction_factor = (instruction_count / 100) ** 0.8
        
        # Register pressure impact
        register_factor = 1.0 + (register_pressure - 20) * 0.02
        
        # Texture sampling overhead
        texture_factor = 1.0 + texture_samples * 0.05
        
        # Branch complexity (exponential impact)
        branch_factor = 1.0 + branch_complexity * 0.15
        
        # Loop depth (significant impact)
        loop_factor = 1.0 + loop_depth * 0.25
        
        # Variant compilation overhead
        variant_factor = 1.0 + (variant_count - 1) * 0.1
        
        # Thermal throttling
        if gpu_temp > 80:
            thermal_factor = 1.0 + (gpu_temp - 80) * 0.1
        else:
            thermal_factor = 1.0
            
        # Power throttling
        if power_draw < 8:
            power_factor = 1.2  # Low power = slower clocks
        elif power_draw > 15:
            power_factor = 1.1  # High power = some throttling
        else:
            power_factor = 1.0
            
        # Game-specific complexity
        game_factor = pattern['complexity_multiplier']
        
        # Combine all factors
        total_time = (base_time * size_factor * instruction_factor * 
                     register_factor * texture_factor * branch_factor * 
                     loop_factor * variant_factor * thermal_factor * 
                     power_factor * game_factor)
                     
        return total_time
        
    def generate_gameplay_sequence(self, game_id: str, n_frames: int = 1000) -> List[Dict]:
        """Generate realistic shader usage sequence during gameplay"""
        
        pattern = self.game_patterns.get(game_id, self.game_patterns['elden_ring'])
        sequence = []
        
        # Common shaders that repeat often
        common_shaders = []
        for i in range(10):  # 10 common shaders
            shader_type = np.random.choice(
                list(pattern['shader_distribution'].keys()),
                p=list(pattern['shader_distribution'].values())
            )
            common_shaders.append(f"{game_id}_common_{shader_type.value}_{i}")
            
        # Rare shaders for special effects
        rare_shaders = []
        for i in range(50):  # 50 rare shaders
            shader_type = np.random.choice(
                list(pattern['shader_distribution'].keys()),
                p=list(pattern['shader_distribution'].values())
            )
            rare_shaders.append(f"{game_id}_rare_{shader_type.value}_{i}")
            
        current_time = time.time()
        
        for frame in range(n_frames):
            frame_time = current_time + frame * (1/60)  # 60 FPS
            
            # Number of shaders per frame (varies by game intensity)
            n_shaders = np.random.poisson(3) + 1
            
            for _ in range(n_shaders):
                # Choose shader based on frequency
                if np.random.random() < 0.7:  # 70% common shaders
                    shader_hash = np.random.choice(common_shaders)
                else:  # 30% rare shaders
                    shader_hash = np.random.choice(rare_shaders)
                    
                sequence.append({
                    'timestamp': frame_time + np.random.uniform(0, 1/60),
                    'shader_hash': shader_hash,
                    'scene_id': f"scene_{frame // 100}",  # Scene changes every 100 frames
                    'frame': frame
                })
                
        return sequence


class ModelEvaluator:
    """
    Comprehensive evaluation of shader prediction models
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_predictor(self, predictor: SteamDeckShaderPredictor,
                          test_data: List[ShaderMetrics],
                          verbose: bool = True) -> Dict:
        """Comprehensive evaluation of a predictor model"""
        
        if len(test_data) < 10:
            print("Warning: Test data too small for meaningful evaluation")
            return {}
            
        # Prepare test data
        y_true = [m.compilation_time_ms for m in test_data]
        y_pred = []
        confidences = []
        prediction_times = []
        
        # Make predictions
        for metrics in test_data:
            start_time = time.time()
            pred, conf = predictor.predict(metrics)
            prediction_time = (time.time() - start_time) * 1000  # ms
            
            y_pred.append(pred)
            confidences.append(conf)
            prediction_times.append(prediction_time)
            
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage errors
        percentage_errors = [abs(t - p) / max(t, 1) * 100 for t, p in zip(y_true, y_pred)]
        mape = np.mean(percentage_errors)
        
        # Confidence analysis
        avg_confidence = np.mean(confidences)
        confidence_vs_error = np.corrcoef(confidences, percentage_errors)[0, 1]
        
        # Prediction speed analysis
        avg_prediction_time = np.mean(prediction_times)
        
        results = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'avg_confidence': avg_confidence,
            'confidence_correlation': confidence_vs_error,
            'avg_prediction_time_ms': avg_prediction_time,
            'n_samples': len(test_data),
            'predictions': list(zip(y_true, y_pred, confidences))
        }
        
        if verbose:
            print(f"Model Evaluation Results:")
            print(f"  Mean Absolute Error: {mae:.2f} ms")
            print(f"  Root Mean Square Error: {rmse:.2f} ms")
            print(f"  R² Score: {r2:.4f}")
            print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
            print(f"  Average Confidence: {avg_confidence:.4f}")
            print(f"  Average Prediction Time: {avg_prediction_time:.3f} ms")
            
        return results
        
    def evaluate_by_shader_type(self, predictor: SteamDeckShaderPredictor,
                               test_data: List[ShaderMetrics]) -> Dict:
        """Evaluate performance by shader type"""
        
        results_by_type = {}
        
        for shader_type in ShaderType:
            type_data = [m for m in test_data if m.shader_type == shader_type]
            
            if len(type_data) < 5:
                continue
                
            results = self.evaluate_predictor(predictor, type_data, verbose=False)
            results_by_type[shader_type.value] = results
            
        return results_by_type
        
    def evaluate_thermal_impact(self, predictor: SteamDeckShaderPredictor,
                               test_data: List[ShaderMetrics]) -> Dict:
        """Evaluate how thermal conditions affect predictions"""
        
        # Group by temperature ranges
        temp_ranges = [
            (0, 65, "cool"),
            (65, 75, "normal"),
            (75, 85, "warm"),
            (85, 100, "hot")
        ]
        
        results_by_temp = {}
        
        for min_temp, max_temp, label in temp_ranges:
            temp_data = [m for m in test_data 
                        if min_temp <= m.gpu_temp_celsius < max_temp]
            
            if len(temp_data) < 5:
                continue
                
            results = self.evaluate_predictor(predictor, temp_data, verbose=False)
            results_by_temp[label] = results
            
        return results_by_temp
        
    def create_evaluation_plots(self, results: Dict, save_path: str = None):
        """Create comprehensive evaluation plots"""
        
        if 'predictions' not in results:
            print("No prediction data available for plotting")
            return
            
        y_true, y_pred, confidences = zip(*results['predictions'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Shader Compilation Prediction Evaluation', fontsize=16)
        
        # Prediction vs Actual scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Compilation Time (ms)')
        axes[0, 0].set_ylabel('Predicted Compilation Time (ms)')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = np.array(y_pred) - np.array(y_true)
        axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Prediction Error (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confidence vs Error
        percentage_errors = [abs(t - p) / max(t, 1) * 100 for t, p in zip(y_true, y_pred)]
        axes[1, 0].scatter(confidences, percentage_errors, alpha=0.6, s=20)
        axes[1, 0].set_xlabel('Prediction Confidence')
        axes[1, 0].set_ylabel('Absolute Percentage Error (%)')
        axes[1, 0].set_title('Confidence vs Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics bar chart
        metrics = ['MAE', 'RMSE', 'MAPE', 'R²']
        values = [results['mae'], results['rmse'], results['mape'], results['r2']]
        
        bars = axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        else:
            plt.show()
            
    def benchmark_models(self, models: Dict[str, SteamDeckShaderPredictor],
                        test_data: List[ShaderMetrics]) -> Dict:
        """Benchmark multiple models"""
        
        benchmark_results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            results = self.evaluate_predictor(model, test_data, verbose=False)
            benchmark_results[name] = results
            
        # Create comparison
        comparison = pd.DataFrame({
            name: {
                'MAE': results['mae'],
                'RMSE': results['rmse'],
                'R²': results['r2'],
                'MAPE': results['mape'],
                'Confidence': results['avg_confidence'],
                'Speed (ms)': results['avg_prediction_time_ms']
            }
            for name, results in benchmark_results.items()
        }).T
        
        print("\nModel Comparison:")
        print(comparison.round(4))
        
        return benchmark_results


class TrainingManager:
    """
    Manages training process with validation and hyperparameter tuning
    """
    
    def __init__(self):
        self.training_history = []
        
    def train_with_validation(self, predictor: SteamDeckShaderPredictor,
                             training_data: List[ShaderMetrics],
                             validation_split: float = 0.2,
                             n_iterations: int = 5) -> Dict:
        """Train model with cross-validation"""
        
        print(f"Training with {len(training_data)} samples...")
        
        iteration_results = []
        
        for i in range(n_iterations):
            print(f"Training iteration {i+1}/{n_iterations}")
            
            # Shuffle data
            shuffled_data = training_data.copy()
            np.random.shuffle(shuffled_data)
            
            # Train model
            train_results = predictor.train(shuffled_data, validation_split)
            
            # Evaluate on held-out test set
            split_idx = int(len(shuffled_data) * (1 - validation_split))
            test_data = shuffled_data[split_idx:]
            
            evaluator = ModelEvaluator()
            eval_results = evaluator.evaluate_predictor(predictor, test_data, verbose=False)
            
            iteration_result = {
                'iteration': i + 1,
                'train_results': train_results,
                'eval_results': eval_results
            }
            
            iteration_results.append(iteration_result)
            
        # Calculate average performance
        avg_mae = np.mean([r['eval_results']['mae'] for r in iteration_results])
        avg_r2 = np.mean([r['eval_results']['r2'] for r in iteration_results])
        
        final_results = {
            'iterations': iteration_results,
            'avg_mae': avg_mae,
            'avg_r2': avg_r2,
            'n_samples': len(training_data),
            'n_iterations': n_iterations
        }
        
        self.training_history.append(final_results)
        
        print(f"Training completed:")
        print(f"  Average MAE: {avg_mae:.2f} ms")
        print(f"  Average R²: {avg_r2:.4f}")
        
        return final_results
        
    def hyperparameter_search(self, training_data: List[ShaderMetrics],
                             param_grid: Dict) -> Dict:
        """Simple hyperparameter search"""
        
        best_score = float('inf')
        best_params = None
        results = []
        
        # Generate parameter combinations
        param_combinations = []
        keys = list(param_grid.keys())
        
        def generate_combinations(idx, current_params):
            if idx == len(keys):
                param_combinations.append(current_params.copy())
                return
                
            key = keys[idx]
            for value in param_grid[key]:
                current_params[key] = value
                generate_combinations(idx + 1, current_params)
                
        generate_combinations(0, {})
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for params in param_combinations:
            print(f"Testing params: {params}")
            
            # Create predictor with these parameters
            config = {
                'model_type': params.get('model_type', 'ensemble'),
                'cache_size': params.get('cache_size', 1000),
                'max_temp': 85.0,
                'power_budget': 15.0
            }
            
            predictor = SteamDeckShaderPredictor(config)
            
            # Train and evaluate
            train_results = self.train_with_validation(
                predictor, training_data, n_iterations=3
            )
            
            score = train_results['avg_mae']
            
            result = {
                'params': params,
                'score': score,
                'r2': train_results['avg_r2']
            }
            
            results.append(result)
            
            if score < best_score:
                best_score = score
                best_params = params
                
        print(f"Best parameters: {best_params}")
        print(f"Best score (MAE): {best_score:.2f} ms")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }


def run_comprehensive_evaluation():
    """Run a comprehensive evaluation of the shader prediction system"""
    
    print("=== Steam Deck Shader Prediction Comprehensive Evaluation ===\n")
    
    # Generate synthetic training data
    print("1. Generating synthetic training data...")
    generator = SyntheticDataGenerator()
    
    # Generate diverse training data
    training_data = []
    training_data.extend(generator.generate_shader_metrics(500, 'cyberpunk'))
    training_data.extend(generator.generate_shader_metrics(500, 'elden_ring'))
    training_data.extend(generator.generate_shader_metrics(300, 'portal2'))
    
    print(f"Generated {len(training_data)} training samples")
    
    # Generate test data
    test_data = generator.generate_shader_metrics(200, 'mixed')
    print(f"Generated {len(test_data)} test samples")
    
    # Create and train predictor
    print("\n2. Training predictor models...")
    predictor = SteamDeckShaderPredictor({
        'model_type': 'ensemble',
        'cache_size': 2000,
        'max_temp': 83.0,
        'power_budget': 12.0
    })
    
    training_manager = TrainingManager()
    training_results = training_manager.train_with_validation(
        predictor, training_data, n_iterations=3
    )
    
    # Evaluate predictor
    print("\n3. Evaluating predictor performance...")
    evaluator = ModelEvaluator()
    
    # Overall evaluation
    overall_results = evaluator.evaluate_predictor(predictor, test_data)
    
    # Evaluation by shader type
    print("\n4. Evaluation by shader type...")
    type_results = evaluator.evaluate_by_shader_type(predictor, test_data)
    for shader_type, results in type_results.items():
        print(f"  {shader_type}: MAE={results['mae']:.2f}ms, R²={results['r2']:.4f}")
    
    # Thermal impact evaluation
    print("\n5. Thermal impact evaluation...")
    thermal_results = evaluator.evaluate_thermal_impact(predictor, test_data)
    for temp_range, results in thermal_results.items():
        print(f"  {temp_range}: MAE={results['mae']:.2f}ms, R²={results['r2']:.4f}")
    
    # Test gameplay pattern analysis
    print("\n6. Testing gameplay pattern analysis...")
    pattern_analyzer = GameplayPatternAnalyzer()
    
    # Generate gameplay sequence
    sequence = generator.generate_gameplay_sequence('cyberpunk', 500)
    for item in sequence[:10]:  # Process first 10 items
        pattern_analyzer.record_shader_usage(
            'cyberpunk', item['scene_id'], item['shader_hash'], item['timestamp']
        )
    
    # Analyze patterns
    patterns = pattern_analyzer.analyze_patterns('cyberpunk')
    if patterns:
        print(f"  Identified {len(patterns.common_shaders)} common shaders")
        print(f"  Peak shader load: {patterns.peak_shader_load} shaders/second")
    
    # Test integrated system
    print("\n7. Testing integrated system...")
    system = SteamDeckOptimizedSystem()
    system.start()
    
    # Set active game
    system.set_active_game('1091500', 'Cyberpunk 2077')
    
    # Test shader processing
    test_shader = {
        'hash': 'test_shader_001',
        'type': 'fragment',
        'bytecode_size': 4096,
        'instruction_count': 280,
        'register_pressure': 48,
        'texture_samples': 8,
        'branch_complexity': 5,
        'loop_depth': 3,
        'scene_id': 'night_city',
        'priority': 2
    }
    
    result = system.process_shader(test_shader)
    print(f"  Shader processing result: {result['predicted_compilation_time_ms']:.2f}ms")
    print(f"  Can compile now: {result['can_compile_now']}")
    print(f"  Thermal state: {result['thermal_state']}")
    
    # Get system status
    status = system.get_system_status()
    print(f"  System running: {status['running']}")
    print(f"  Hardware detected: {status['hardware'].get('is_steam_deck', False)}")
    
    system.stop()
    
    # Create evaluation plots
    print("\n8. Creating evaluation plots...")
    try:
        evaluator.create_evaluation_plots(overall_results, 'shader_evaluation.png')
        print("  Evaluation plots saved to 'shader_evaluation.png'")
    except Exception as e:
        print(f"  Plot creation failed: {e}")
    
    # Export results
    print("\n9. Exporting results...")
    export_data = {
        'training_results': training_results,
        'overall_results': overall_results,
        'type_results': type_results,
        'thermal_results': thermal_results,
        'system_status': status,
        'timestamp': time.time()
    }
    
    with open('evaluation_results.json', 'w') as f:
        # Convert numpy types to JSON serializable
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
            
        json.dump(export_data, f, indent=2, default=convert_numpy)
    
    print("  Results exported to 'evaluation_results.json'")
    
    print("\n=== Evaluation Complete ===")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Training samples: {len(training_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Overall MAE: {overall_results['mae']:.2f} ms")
    print(f"  Overall R²: {overall_results['r2']:.4f}")
    print(f"  Average prediction time: {overall_results['avg_prediction_time_ms']:.3f} ms")
    print(f"  System successfully integrated: {status['running']}")


def run_hyperparameter_search():
    """Run hyperparameter search for optimal model configuration"""
    
    print("=== Hyperparameter Search ===")
    
    # Generate training data
    generator = SyntheticDataGenerator()
    training_data = generator.generate_shader_metrics(1000, 'mixed')
    
    # Define parameter grid
    param_grid = {
        'model_type': ['ensemble', 'lightweight'],
        'cache_size': [1000, 2000, 3000]
    }
    
    # Run search
    training_manager = TrainingManager()
    search_results = training_manager.hyperparameter_search(training_data, param_grid)
    
    print(f"Search completed. Best parameters: {search_results['best_params']}")
    
    return search_results


if __name__ == "__main__":
    # Set style for plots
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    # Run comprehensive evaluation
    run_comprehensive_evaluation()
    
    print("\n" + "="*50 + "\n")
    
    # Optional: Run hyperparameter search
    response = input("Run hyperparameter search? (y/n): ").lower().strip()
    if response == 'y':
        run_hyperparameter_search()
    
    print("\nAll evaluations complete!")
    print("Check the generated files:")
    print("  - shader_evaluation.png: Performance plots")
    print("  - evaluation_results.json: Detailed results")
    print("  - steamdeck_config.json: Example configuration")