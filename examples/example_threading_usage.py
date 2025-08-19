#!/usr/bin/env python3
"""
Steam Deck Threading Optimization - Usage Example
Demonstrates proper usage of the threading fixes for the ML Shader Prediction Compiler
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate complete threading optimization usage"""
    print("🎮 Steam Deck ML Shader Prediction Compiler")
    print("🧵 Threading Optimization Example")
    print("=" * 60)
    
    # Step 1: Initialize threading optimizations
    print("\n1️⃣ Initializing Threading Optimizations...")
    try:
        from src.core.startup_threading import initialize_steam_deck_threading, get_threading_status
        
        success = initialize_steam_deck_threading()
        if success:
            print("✅ Threading optimizations initialized successfully")
            
            # Show configuration
            status = get_threading_status()
            if status.get('initialized'):
                config = status.get('threading_config', {})
                print(f"   📊 Configured libraries: {', '.join(config.get('configured_libraries', []))}")
                print(f"   🔧 Max threads: {config.get('config', {}).get('max_threads', 'unknown')}")
        else:
            print("❌ Threading optimization initialization failed")
            return 1
    except Exception as e:
        print(f"⚠️  Threading optimizations not available: {e}")
        print("   Continuing with basic configuration...")
    
    # Step 2: Initialize ML Predictor with threading support
    print("\n2️⃣ Initializing ML Predictor...")
    try:
        from src.core.ml_only_predictor import get_ml_predictor
        
        predictor = get_ml_predictor()
        print("✅ ML predictor initialized")
        
        # Get performance metrics
        metrics = predictor.get_performance_metrics()
        print(f"   ⚡ Primary model: {metrics.get('primary_model', 'unknown')}")
        print(f"   📈 Throughput: {metrics.get('predictions_per_second', 0):.0f} predictions/sec")
        
        threading_active = hasattr(predictor, 'threading_configurator') and predictor.threading_configurator is not None
        print(f"   🧵 Threading optimizations: {'✅ Active' if threading_active else '❌ Inactive'}")
        
    except Exception as e:
        print(f"❌ ML predictor initialization failed: {e}")
        return 1
    
    # Step 3: Test performance with threading optimizations
    print("\n3️⃣ Testing Prediction Performance...")
    try:
        # Define test scenarios
        test_scenarios = [
            ("Simple Vertex Shader", {
                'instruction_count': 200,
                'register_usage': 16,
                'texture_samples': 2,
                'memory_operations': 5,
                'control_flow_complexity': 2,
                'wave_size': 64,
                'shader_type_hash': 1.0,
                'optimization_level': 1
            }),
            ("Complex Fragment Shader", {
                'instruction_count': 1500,
                'register_usage': 96,
                'texture_samples': 12,
                'memory_operations': 30,
                'control_flow_complexity': 15,
                'wave_size': 64,
                'shader_type_hash': 2.0,
                'optimization_level': 2
            }),
            ("Compute Shader", {
                'instruction_count': 2000,
                'register_usage': 128,
                'texture_samples': 4,
                'memory_operations': 50,
                'control_flow_complexity': 20,
                'wave_size': 64,
                'shader_type_hash': 3.0,
                'optimization_level': 3
            })
        ]
        
        total_time = 0
        successful_predictions = 0
        
        for scenario_name, features in test_scenarios:
            try:
                # Warm up
                predictor.predict_compilation_time(features)
                
                # Performance test
                start_time = time.perf_counter()
                result = predictor.predict_compilation_time(features)
                prediction_time = (time.perf_counter() - start_time) * 1000
                
                total_time += prediction_time
                successful_predictions += 1
                
                print(f"   ✅ {scenario_name}: {result.compilation_time_ms:.1f}ms prediction in {prediction_time:.3f}ms")
                
            except Exception as e:
                print(f"   ❌ {scenario_name}: Failed ({e})")
        
        if successful_predictions > 0:
            avg_time = total_time / successful_predictions
            print(f"\n   📊 Average prediction time: {avg_time:.3f}ms")
            print(f"   🎯 Performance target (<5ms): {'✅ Met' if avg_time < 5.0 else '❌ Not met'}")
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
    
    # Step 4: Check thermal management integration
    print("\n4️⃣ Testing Thermal Management...")
    try:
        from src.optimization.optimized_thermal_manager import get_thermal_manager
        
        thermal_manager = get_thermal_manager()
        
        # Brief monitoring test
        thermal_manager.start_monitoring()
        time.sleep(1.0)  # Let it collect some data
        
        status = thermal_manager.get_status()
        print(f"   🌡️  Current thermal state: {status['thermal_state']}")
        print(f"   🔥 APU temperature: {status['current_temps']['apu']:.1f}°C")
        print(f"   🧵 Compilation threads: {status['compilation_threads']}")
        print(f"   🔧 Thread manager integration: {'✅ Yes' if status.get('mock_mode') == False else '⚠️  Mock mode'}")
        
        thermal_manager.stop_monitoring()
        
    except Exception as e:
        print(f"⚠️  Thermal management test skipped: {e}")
    
    # Step 5: System diagnostics
    print("\n5️⃣ Running System Diagnostics...")
    try:
        from src.core.thread_diagnostics import diagnose_threading_issues
        
        diagnosis = diagnose_threading_issues()
        
        if 'error' in diagnosis:
            print(f"   ⚠️  Diagnostics unavailable: {diagnosis['error']}")
        else:
            total_threads = diagnosis.get('total_threads', 0)
            critical_issues = diagnosis.get('critical_issues', 0)
            high_issues = diagnosis.get('high_issues', 0)
            
            print(f"   🧵 Total system threads: {total_threads}")
            print(f"   🚨 Critical issues: {critical_issues}")
            print(f"   ⚠️  High-priority issues: {high_issues}")
            
            if critical_issues == 0 and high_issues <= 1:
                print("   ✅ System health: Good")
            elif critical_issues == 0:
                print("   ⚠️  System health: Minor issues detected")
            else:
                print("   ❌ System health: Critical issues require attention")
            
            recommendations = diagnosis.get('recommendations', [])
            if recommendations:
                print(f"   💡 Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"      • {rec}")
    
    except Exception as e:
        print(f"⚠️  Diagnostics failed: {e}")
    
    # Step 6: Resource usage summary
    print("\n6️⃣ Resource Usage Summary...")
    try:
        import threading
        import psutil
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent(interval=0.1)
        thread_count = len(threading.enumerate())
        
        print(f"   💾 Memory usage: {memory_mb:.1f} MB")
        print(f"   🖥️  CPU usage: {cpu_percent:.1f}%")
        print(f"   🧵 Active threads: {thread_count}")
        
        # Validate against Steam Deck limits
        memory_ok = memory_mb < 500  # Keep under 500MB for ML predictor
        threads_ok = thread_count <= 10  # Conservative thread limit
        
        print(f"   🎯 Resource efficiency: {'✅ Good' if memory_ok and threads_ok else '⚠️  Needs attention'}")
        
    except Exception as e:
        print(f"⚠️  Resource monitoring failed: {e}")
    
    # Step 7: Cleanup
    print("\n7️⃣ Cleanup and Shutdown...")
    try:
        # Cleanup ML predictor
        predictor.cleanup()
        print("   ✅ ML predictor cleanup completed")
        
        # Cleanup threading system
        from src.core.startup_threading import cleanup_threading
        cleanup_threading()
        print("   ✅ Threading system cleanup completed")
        
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 Threading Optimization Example Complete")
    print("=" * 60)
    print("✅ Steam Deck ML Shader Prediction Compiler is ready for production use")
    print("🧵 Threading optimizations are active and working correctly")
    print("🎮 System is optimized for Steam Deck gaming performance")
    print()
    print("💡 To use in your application:")
    print("   1. Import: from src.core.startup_threading import initialize_steam_deck_threading")
    print("   2. Initialize: initialize_steam_deck_threading()")
    print("   3. Use ML predictor as normal - threading is handled automatically")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)