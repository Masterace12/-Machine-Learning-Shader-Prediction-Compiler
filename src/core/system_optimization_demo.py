#!/usr/bin/env python3
"""
Steam Deck System Optimization Demo for Enhanced ML Shader Predictor
Demonstrates system-level optimizations for gaming performance
"""

import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from enhanced_ml_predictor import (
        EnhancedMLPredictor, UnifiedShaderFeatures, ShaderType,
        SteamDeckSystemMonitor, SteamDeckResourceManager
    )
except ImportError:
    print("Cannot import enhanced_ml_predictor. Make sure you're in the correct directory.")
    exit(1)


def demo_system_optimization():
    """Demonstrate comprehensive system optimization features"""
    
    print("=" * 80)
    print("STEAM DECK SYSTEM OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Initialize the enhanced ML predictor with system optimizations
    print("\n1. Initializing Enhanced ML Predictor with System Optimizations...")
    predictor = EnhancedMLPredictor(
        model_path=Path.home() / '.cache' / 'shader-predict-demo',
        enable_async=True,
        max_memory_mb=40  # Steam Deck optimized limit
    )
    
    # Display system information
    print("\n2. System Detection and Configuration:")
    system_info = predictor.get_system_performance_report()
    
    print(f"   Steam Deck Detection: {system_info['is_steam_deck']}")
    print(f"   APU Model: {system_info['apu_model']}")
    print(f"   Current Thermal State: {system_info['current_thermal_state']}")
    print(f"   System Health: {system_info['system_health']}")
    print(f"   Background CPU Cores: {system_info['background_cores']}")
    print(f"   Gaming CPU Cores: {system_info['gaming_cores']}")
    print(f"   Optimal Thread Count: {system_info['optimal_thread_count']}")
    print(f"   Memory Pressure: {system_info['average_memory_pressure']:.2%}")
    print(f"   Gaming Activity: {system_info['gaming_activity_ratio']:.2%}")
    
    power_state = system_info['power_state']
    print(f"   Power State: {'Battery' if power_state['on_battery'] else 'Plugged'}")
    if power_state['on_battery']:
        print(f"   Battery Level: {power_state['battery_percent']}%")
    
    # Demonstrate gaming optimization
    print("\n3. Gaming Session Optimization:")
    print("   Optimizing for hypothetical gaming session...")
    predictor.optimize_for_gaming_session("cyberpunk2077")
    
    # Create sample shader features for testing
    print("\n4. System-Aware Shader Prediction Performance:")
    sample_features = UnifiedShaderFeatures(
        shader_hash="demo_vertex_shader_123",
        shader_type=ShaderType.VERTEX,
        instruction_count=450,
        register_usage=32,
        texture_samples=2,
        memory_operations=15,
        control_flow_complexity=3,
        wave_size=64,
        uses_derivatives=False,
        uses_tessellation=False,
        uses_geometry_shader=False,
        optimization_level=2,
        target_profile="steam_deck_optimized"
    )
    
    # Test predictions under different system conditions
    print("   Testing prediction performance under various system states...")
    
    # Baseline prediction
    start_time = time.perf_counter()
    prediction1 = predictor.predict_compilation_time(
        sample_features, 
        use_cache=True, 
        game_context="cyberpunk2077"
    )
    baseline_time = (time.perf_counter() - start_time) * 1000
    
    print(f"   Baseline Prediction: {prediction1:.2f}ms (took {baseline_time:.4f}ms)")
    
    # Cached prediction (should be much faster)
    start_time = time.perf_counter()
    prediction2 = predictor.predict_compilation_time(
        sample_features, 
        use_cache=True, 
        game_context="cyberpunk2077"
    )
    cached_time = (time.perf_counter() - start_time) * 1000
    
    print(f"   Cached Prediction: {prediction2:.2f}ms (took {cached_time:.4f}ms)")
    print(f"   Cache Speedup: {baseline_time/cached_time:.1f}x faster")
    
    # Test multiple predictions to show system adaptation
    print("\n5. System Adaptation Testing:")
    prediction_times = []
    
    for i in range(10):
        # Vary the features slightly for each test
        test_features = UnifiedShaderFeatures(
            shader_hash=f"test_shader_{i}",
            shader_type=ShaderType.FRAGMENT,
            instruction_count=400 + i * 50,
            register_usage=28 + i * 2,
            texture_samples=1 + i % 3,
            memory_operations=12 + i * 2,
            control_flow_complexity=2 + i % 4,
            wave_size=64,
            uses_derivatives=i % 2 == 0,
            uses_tessellation=False,
            uses_geometry_shader=i % 5 == 0,
            optimization_level=1 + i % 3,
            target_profile="steam_deck_optimized"
        )
        
        start_time = time.perf_counter()
        prediction = predictor.predict_compilation_time(
            test_features, 
            use_cache=True, 
            game_context="cyberpunk2077"
        )
        prediction_time = (time.perf_counter() - start_time) * 1000
        prediction_times.append(prediction_time)
        
        if i % 3 == 0:  # Show progress
            print(f"   Test {i+1}: {prediction:.2f}ms compile time (predicted in {prediction_time:.4f}ms)")
    
    avg_prediction_time = sum(prediction_times) / len(prediction_times)
    print(f"   Average Prediction Time: {avg_prediction_time:.4f}ms")
    
    # Display final system performance report
    print("\n6. Final System Performance Analysis:")
    final_report = predictor.get_system_performance_report()
    
    print(f"   System Health: {final_report['system_health']}")
    print(f"   Memory Pressure: {final_report['average_memory_pressure']:.2%}")
    print(f"   Thermal Throttle Events: {final_report['thermal_throttle_events']}")
    print(f"   CPU Affinity Changes: {final_report['cpu_affinity_changes']}")
    
    # Test resource manager capabilities
    print("\n7. Resource Management Capabilities:")
    resource_manager = predictor.resource_manager
    
    print(f"   Should Throttle Operations: {resource_manager.should_throttle_operations()}")
    print(f"   Current Memory Pressure: {resource_manager.get_memory_pressure():.2%}")
    print(f"   Gaming Activity Detected: {resource_manager.detect_gaming_activity()}")
    print(f"   Current Thermal State: {predictor.system_monitor.get_current_thermal_state().value}")
    
    # Demonstrate thermal awareness
    print("\n8. Thermal Awareness Demonstration:")
    thermal_state = predictor.system_monitor.get_current_thermal_state()
    thermal_factor = predictor._get_thermal_factor(thermal_state)
    print(f"   Current Thermal State: {thermal_state.value}")
    print(f"   Thermal Adjustment Factor: {thermal_factor}")
    print(f"   Adjusted Prediction: {prediction1 * thermal_factor:.2f}ms")
    
    # Memory optimization demonstration
    print("\n9. Memory Optimization Status:")
    memory_optimizer = predictor.memory_optimizer
    memory_usage = memory_optimizer.get_current_memory_usage()
    memory_pressure = memory_optimizer.get_memory_pressure()
    
    print(f"   Current Memory Usage: {memory_usage / (1024*1024):.1f}MB")
    print(f"   Memory Pressure: {memory_pressure:.2%}")
    print(f"   Should Perform GC: {memory_optimizer.should_perform_gc()}")
    
    # Cleanup
    print("\n10. System Cleanup:")
    print("    Performing comprehensive cleanup...")
    predictor.cleanup()
    
    print("\n" + "=" * 80)
    print("SYSTEM OPTIMIZATION DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Benefits Demonstrated:")
    print("• Steam Deck hardware detection and optimization")
    print("• Gaming-aware CPU core allocation and process priority")
    print("• Thermal-aware prediction adjustments")
    print("• Memory-optimized allocation and garbage collection")
    print("• Gaming activity detection and system adaptation")
    print("• Background task scheduling with minimal gaming impact")
    print("• Comprehensive system performance monitoring")
    print("• Power-aware optimizations for handheld usage")


if __name__ == "__main__":
    demo_system_optimization()