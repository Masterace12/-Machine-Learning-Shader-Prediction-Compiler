"""
Practical Usage Example for Steam Deck Shader Prediction System
Demonstrates real-world integration patterns and API usage
"""

import time
import json
import threading
from typing import Dict, List, Optional
from dataclasses import asdict

# Import our systems
from shader_prediction_system import SteamDeckShaderPredictor, ShaderType
from steam_deck_integration import SteamDeckOptimizedSystem, create_example_game_profiles
from shader_training_evaluation import SyntheticDataGenerator, TrainingManager


class GameShaderManager:
    """
    Example integration showing how a game engine would use the shader prediction system
    This demonstrates the API from a game developer's perspective
    """
    
    def __init__(self, app_id: str, game_name: str):
        self.app_id = app_id
        self.game_name = game_name
        self.system = None
        self.shader_queue = []
        self.compilation_stats = {
            'total_requested': 0,
            'immediate_compilations': 0,
            'delayed_compilations': 0,
            'cache_hits': 0,
            'total_time_saved': 0.0
        }
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the shader prediction system"""
        
        # Create optimized configuration for this game
        config = {
            'predictor': {
                'model_type': 'ensemble',
                'cache_size': 2000,
                'max_temp': 83.0,
                'power_budget': 12.0,
                'sequence_length': 100
            },
            'thermal_protection': True,
            'battery_optimization': True,
            'performance_mode': 'balanced',
            'game_profiles': create_example_game_profiles()
        }
        
        # Save configuration
        with open('game_shader_config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        # Create and start system
        self.system = SteamDeckOptimizedSystem('game_shader_config.json')
        self.system.start()
        
        # Register this game
        self.system.set_active_game(self.app_id, self.game_name)
        
        print(f"Shader manager initialized for {self.game_name}")
        
    def request_shader_compilation(self, shader_data: Dict) -> Dict:
        """
        Request shader compilation - main API entry point
        This is what game engines would call for each shader
        """
        
        self.compilation_stats['total_requested'] += 1
        
        # Process shader request
        result = self.system.process_shader(shader_data)
        
        if 'error' in result:
            print(f"Shader request error: {result['error']}")
            return result
            
        # Handle result based on scheduling decision
        if result['can_compile_now']:
            self.compilation_stats['immediate_compilations'] += 1
            
            # Simulate immediate compilation
            compile_result = self._simulate_compilation(shader_data, result)
            
            # Record result for ML training
            self.system.record_compilation_result(shader_data, compile_result)
            
            return {
                'status': 'compiled',
                'actual_time_ms': compile_result['time_ms'],
                'predicted_time_ms': result['predicted_compilation_time_ms'],
                'thermal_state': result['thermal_state']
            }
            
        else:
            self.compilation_stats['delayed_compilations'] += 1
            
            # Add to queue for later compilation
            self.shader_queue.append({
                'shader_data': shader_data,
                'prediction': result,
                'queued_at': time.time()
            })
            
            return {
                'status': 'queued',
                'schedule': result['schedule'],
                'predicted_time_ms': result['predicted_compilation_time_ms'],
                'thermal_state': result['thermal_state'],
                'queue_position': len(self.shader_queue)
            }
            
    def _simulate_compilation(self, shader_data: Dict, prediction: Dict) -> Dict:
        """Simulate actual shader compilation for demonstration"""
        
        # Use prediction as base, add some realistic variation
        predicted_time = prediction['predicted_compilation_time_ms']
        
        # Simulate compilation time with some noise
        actual_time = predicted_time + (predicted_time * 0.1 * (2 * time.time() % 1 - 1))
        actual_time = max(1.0, actual_time)  # Minimum 1ms
        
        # Simulate compilation
        time.sleep(actual_time / 1000.0)  # Convert to seconds for sleep
        
        # Get current hardware state for recording
        hardware_state = self.system.hardware_monitor.hardware_state
        
        return {
            'time_ms': actual_time,
            'gpu_temp': hardware_state.gpu_temp if hardware_state else 70.0,
            'power_draw': hardware_state.gpu_power if hardware_state else 10.0,
            'memory_mb': 64,  # Simulated memory usage
            'success': True
        }
        
    def process_queued_shaders(self) -> List[Dict]:
        """Process shaders from queue when thermal conditions allow"""
        
        if not self.shader_queue:
            return []
            
        processed = []
        remaining_queue = []
        
        for queued_item in self.shader_queue:
            # Check if we can compile this shader now
            current_result = self.system.process_shader(queued_item['shader_data'])
            
            if current_result.get('can_compile_now', False):
                # Compile now
                compile_result = self._simulate_compilation(
                    queued_item['shader_data'], 
                    current_result
                )
                
                # Record result
                self.system.record_compilation_result(
                    queued_item['shader_data'], 
                    compile_result
                )
                
                processed.append({
                    'shader_hash': queued_item['shader_data'].get('hash', 'unknown'),
                    'queue_time_ms': (time.time() - queued_item['queued_at']) * 1000,
                    'compile_time_ms': compile_result['time_ms']
                })
                
            else:
                # Keep in queue
                remaining_queue.append(queued_item)
                
        self.shader_queue = remaining_queue
        return processed
        
    def preload_shaders_for_scene(self, scene_id: str, shader_hashes: List[str]):
        """Preload shaders for an upcoming scene"""
        
        print(f"Preloading {len(shader_hashes)} shaders for scene: {scene_id}")
        
        preload_results = []
        
        for shader_hash in shader_hashes:
            # Create shader data for preloading
            shader_data = {
                'hash': shader_hash,
                'type': 'fragment',  # Default type
                'scene_id': scene_id,
                'priority': 1,  # Medium priority for preloading
                'bytecode_size': 2048,  # Default values
                'instruction_count': 100,
                'register_pressure': 32,
                'texture_samples': 4,
                'branch_complexity': 2,
                'loop_depth': 1,
                'variant_count': 2
            }
            
            result = self.request_shader_compilation(shader_data)
            preload_results.append({
                'shader_hash': shader_hash,
                'result': result
            })
            
        return preload_results
        
    def get_next_shader_predictions(self, current_shader: str, n_predictions: int = 5) -> List[Dict]:
        """Get predictions for next likely shaders"""
        
        # Use the pattern analyzer to get predictions
        predictions = self.system.predictor.pattern_analyzer.predict_next_shaders(
            self.app_id, current_shader, top_k=n_predictions
        )
        
        return [
            {'shader_hash': shader, 'probability': prob}
            for shader, prob in predictions
        ]
        
    def optimize_for_battery_mode(self):
        """Switch to battery-optimized compilation strategy"""
        
        print("Switching to battery optimization mode")
        
        # Adjust system configuration
        self.system.config['predictor']['power_budget'] = 8.0  # Lower power budget
        self.system.predictor.scheduler.power_budget = 8.0
        
        # Reduce cache size to save memory
        self.system.config['predictor']['cache_size'] = 1000
        
        # More aggressive thermal limits
        self.system.config['predictor']['max_temp'] = 80.0
        self.system.predictor.scheduler.max_temp = 80.0
        
    def optimize_for_performance_mode(self):
        """Switch to performance-optimized compilation strategy"""
        
        print("Switching to performance optimization mode")
        
        # Increase power budget
        self.system.config['predictor']['power_budget'] = 15.0
        self.system.predictor.scheduler.power_budget = 15.0
        
        # Larger cache for better hit rates
        self.system.config['predictor']['cache_size'] = 3000
        
        # Higher thermal tolerance
        self.system.config['predictor']['max_temp'] = 85.0
        self.system.predictor.scheduler.max_temp = 85.0
        
    def get_compilation_statistics(self) -> Dict:
        """Get comprehensive compilation statistics"""
        
        stats = self.compilation_stats.copy()
        
        # Add system statistics
        system_stats = self.system.get_system_status()
        stats['system_status'] = system_stats
        
        # Calculate efficiency metrics
        if stats['total_requested'] > 0:
            stats['immediate_compile_rate'] = stats['immediate_compilations'] / stats['total_requested']
            stats['queue_rate'] = stats['delayed_compilations'] / stats['total_requested']
            
        stats['current_queue_size'] = len(self.shader_queue)
        
        return stats
        
    def cleanup(self):
        """Clean up resources"""
        
        if self.system:
            # Export performance data
            self.system.export_performance_data(f"{self.game_name}_performance.json")
            
            # Stop system
            self.system.stop()
            
        print(f"Shader manager cleaned up for {self.game_name}")


class GameEngineIntegrationExample:
    """
    Example showing how a game engine would integrate the shader prediction system
    """
    
    def __init__(self):
        self.shader_manager = None
        self.current_scene = None
        self.frame_count = 0
        
    def initialize_game(self, app_id: str, game_name: str):
        """Initialize game with shader prediction"""
        
        print(f"Initializing {game_name}...")
        
        # Create shader manager
        self.shader_manager = GameShaderManager(app_id, game_name)
        
        # Train the system with some initial data if needed
        self._initial_training()
        
        print("Game initialization complete")
        
    def _initial_training(self):
        """Perform initial training with synthetic data"""
        
        print("Performing initial ML model training...")
        
        # Generate training data
        generator = SyntheticDataGenerator()
        training_data = generator.generate_shader_metrics(500, 'mixed')
        
        # Train the predictor
        training_manager = TrainingManager()
        training_manager.train_with_validation(
            self.shader_manager.system.predictor, 
            training_data, 
            n_iterations=2
        )
        
        print("Initial training complete")
        
    def load_scene(self, scene_id: str, scene_shaders: List[str]):
        """Load a new scene with shader preloading"""
        
        self.current_scene = scene_id
        print(f"Loading scene: {scene_id}")
        
        # Preload common shaders for this scene
        preload_results = self.shader_manager.preload_shaders_for_scene(
            scene_id, scene_shaders
        )
        
        print(f"Preloaded {len(preload_results)} shaders")
        
        # Show preload results
        immediate_count = sum(1 for r in preload_results if r['result']['status'] == 'compiled')
        queued_count = sum(1 for r in preload_results if r['result']['status'] == 'queued')
        
        print(f"  Immediate: {immediate_count}, Queued: {queued_count}")
        
    def render_frame(self):
        """Simulate rendering a frame with shader requests"""
        
        self.frame_count += 1
        
        # Simulate shader requests during rendering
        frame_shaders = [
            {
                'hash': f"frame_shader_{self.frame_count % 10}",
                'type': 'fragment',
                'scene_id': self.current_scene or 'default',
                'priority': 3,  # High priority for frame rendering
                'bytecode_size': 1024 + (self.frame_count % 3) * 512,
                'instruction_count': 80 + (self.frame_count % 5) * 20,
                'register_pressure': 24 + (self.frame_count % 4) * 8,
                'texture_samples': 2 + (self.frame_count % 3),
                'branch_complexity': 1 + (self.frame_count % 4),
                'loop_depth': (self.frame_count % 3),
                'variant_count': 1 + (self.frame_count % 3)
            }
        ]
        
        frame_results = []
        for shader in frame_shaders:
            result = self.shader_manager.request_shader_compilation(shader)
            frame_results.append(result)
            
        # Process any queued shaders that are now ready
        processed = self.shader_manager.process_queued_shaders()
        if processed:
            print(f"Frame {self.frame_count}: Processed {len(processed)} queued shaders")
            
        return frame_results
        
    def handle_thermal_event(self, thermal_state: str):
        """Handle thermal state changes"""
        
        print(f"Thermal event: {thermal_state}")
        
        if thermal_state in ['hot', 'throttling']:
            # Switch to battery mode to reduce heat
            self.shader_manager.optimize_for_battery_mode()
        elif thermal_state == 'cool':
            # Switch back to performance mode
            self.shader_manager.optimize_for_performance_mode()
            
    def handle_battery_event(self, battery_level: float, charging: bool):
        """Handle battery state changes"""
        
        if battery_level < 20 and not charging:
            print("Low battery: Switching to battery optimization")
            self.shader_manager.optimize_for_battery_mode()
        elif charging and battery_level > 50:
            print("Charging with good battery: Switching to performance mode")
            self.shader_manager.optimize_for_performance_mode()
            
    def get_performance_report(self) -> str:
        """Generate performance report"""
        
        stats = self.shader_manager.get_compilation_statistics()
        
        report = f"""
=== Shader Compilation Performance Report ===
Game: {self.shader_manager.game_name}
Frames Rendered: {self.frame_count}

Compilation Statistics:
  Total Requests: {stats['total_requested']}
  Immediate Compilations: {stats['immediate_compilations']}
  Delayed Compilations: {stats['delayed_compilations']}
  Current Queue Size: {stats['current_queue_size']}
  
Efficiency:
  Immediate Rate: {stats.get('immediate_compile_rate', 0):.2%}
  Queue Rate: {stats.get('queue_rate', 0):.2%}

System Status:
  Running: {stats['system_status']['running']}
  Thermal State: {stats['system_status'].get('hardware', {}).get('gpu_temp', 'N/A')}°C
  Cache Size: {stats['system_status']['predictor']['cache_size']}
"""
        
        return report
        
    def shutdown_game(self):
        """Shutdown game and clean up"""
        
        print("Shutting down game...")
        
        # Print final performance report
        report = self.get_performance_report()
        print(report)
        
        # Clean up shader manager
        if self.shader_manager:
            self.shader_manager.cleanup()
            
        print("Game shutdown complete")


def demonstrate_shader_prediction_system():
    """
    Complete demonstration of the shader prediction system
    """
    
    print("=== Steam Deck Shader Prediction System Demonstration ===\n")
    
    # Create game engine integration
    game_engine = GameEngineIntegrationExample()
    
    # Initialize for Cyberpunk 2077
    game_engine.initialize_game("1091500", "Cyberpunk 2077")
    
    # Define some scenes and their common shaders
    scenes = {
        'main_menu': [
            'menu_background_shader',
            'menu_ui_shader',
            'menu_effects_shader'
        ],
        'night_city': [
            'building_shader',
            'neon_lights_shader',
            'car_paint_shader',
            'street_reflection_shader',
            'atmospheric_shader'
        ],
        'combat': [
            'muzzle_flash_shader',
            'impact_effect_shader',
            'blood_shader',
            'explosion_shader'
        ]
    }
    
    # Demonstrate scene loading and rendering
    for scene_name, scene_shaders in scenes.items():
        print(f"\n--- Loading Scene: {scene_name} ---")
        
        # Load scene
        game_engine.load_scene(scene_name, scene_shaders)
        
        # Simulate some frames
        print("Rendering frames...")
        for frame in range(5):
            results = game_engine.render_frame()
            
            # Show frame results
            compiled = sum(1 for r in results if r['status'] == 'compiled')
            queued = sum(1 for r in results if r['status'] == 'queued')
            print(f"  Frame {frame + 1}: {compiled} compiled, {queued} queued")
            
            # Small delay to simulate frame time
            time.sleep(0.1)
            
    # Demonstrate thermal event handling
    print(f"\n--- Simulating Thermal Events ---")
    game_engine.handle_thermal_event('hot')
    
    # Render some more frames under thermal stress
    print("Rendering under thermal stress...")
    for frame in range(3):
        results = game_engine.render_frame()
        time.sleep(0.1)
        
    # Demonstrate battery event handling
    print(f"\n--- Simulating Battery Events ---")
    game_engine.handle_battery_event(15.0, False)  # Low battery, not charging
    
    # Render frames in battery save mode
    print("Rendering in battery save mode...")
    for frame in range(3):
        results = game_engine.render_frame()
        time.sleep(0.1)
        
    # Final performance report and shutdown
    print(f"\n--- Final Performance Report ---")
    game_engine.shutdown_game()
    
    print(f"\n=== Demonstration Complete ===")
    print("Check generated files:")
    print("  - game_shader_config.json: Game-specific configuration")
    print("  - Cyberpunk 2077_performance.json: Performance data export")
    print("  - steamdeck_save/: Saved ML models and state")


def demonstrate_api_usage():
    """
    Demonstrate basic API usage patterns
    """
    
    print("\n=== Basic API Usage Demonstration ===\n")
    
    # 1. Direct predictor usage
    print("1. Direct Predictor Usage:")
    predictor = SteamDeckShaderPredictor()
    
    # Generate some training data
    generator = SyntheticDataGenerator()
    training_data = generator.generate_shader_metrics(100, 'elden_ring')
    
    # Train predictor
    print("   Training predictor...")
    results = predictor.train(training_data)
    print(f"   Training MAE: {list(results.values())[0]:.2f} ms")
    
    # Make prediction
    test_shader = training_data[0]
    predicted_time, confidence = predictor.predict(test_shader)
    print(f"   Prediction: {predicted_time:.2f} ms (confidence: {confidence:.3f})")
    
    # 2. System integration usage
    print("\n2. System Integration Usage:")
    system = SteamDeckOptimizedSystem()
    system.start()
    system.set_active_game("test_game", "Test Game")
    
    # Process shader
    shader_data = {
        'hash': 'test_shader',
        'type': 'fragment',
        'bytecode_size': 2048,
        'instruction_count': 150,
        'register_pressure': 32,
        'texture_samples': 4,
        'branch_complexity': 3,
        'loop_depth': 1,
        'scene_id': 'test_scene',
        'priority': 1,
        'variant_count': 2
    }
    
    result = system.process_shader(shader_data)
    print(f"   System result: {result}")
    
    system.stop()
    
    print("\n=== API Usage Demonstration Complete ===")


if __name__ == "__main__":
    # Run comprehensive demonstration
    demonstrate_shader_prediction_system()
    
    # Run basic API usage demonstration
    demonstrate_api_usage()
    
    print("\n" + "="*60)
    print("All demonstrations completed successfully!")
    print("The system is now ready for integration into game engines.")
    print("="*60)