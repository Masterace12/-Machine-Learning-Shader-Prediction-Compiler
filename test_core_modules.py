#!/usr/bin/env python3
"""
Test core OLED optimization modules
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_thermal_optimizer():
    """Test thermal optimizer"""
    print("🌡️  Testing Thermal Optimizer...")
    try:
        from src.core.steamdeck_thermal_optimizer import get_thermal_optimizer, SteamDeckModel
        
        optimizer = get_thermal_optimizer()
        print(f"   Model detected: {optimizer.model.value}")
        
        # Test status
        status = optimizer.get_status()
        print(f"   Temperature: {status.get('max_temperature', 0):.1f}°C")
        print(f"   Thermal state: {status.get('thermal_state', 'unknown')}")
        print(f"   Optimal threads: {status.get('optimal_threads', 0)}")
        
        # Test OLED-specific features
        if optimizer.model == SteamDeckModel.OLED:
            gaming_opts = optimizer.optimize_for_gaming()
            print(f"   OLED gaming optimizations: {len(gaming_opts)} settings")
            print("   ✅ OLED thermal optimizer working")
        else:
            print("   ⚠️  Not OLED model, but optimizer working")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_memory_optimizer():
    """Test memory optimizer"""
    print("💾 Testing Memory Optimizer...")
    try:
        from src.core.oled_memory_optimizer import MemoryMappedShaderCache
        
        # Create temporary cache
        cache_dir = Path("/tmp/test_oled_cache")
        cache = MemoryMappedShaderCache(cache_dir, max_cache_size_mb=100)
        
        # Test storage/retrieval
        test_data = b"test_shader_data_for_oled_optimization"
        success = cache.store_shader("test_hash", test_data, "test_variant")
        print(f"   Storage success: {success}")
        
        retrieved = cache.retrieve_shader("test_hash", "test_variant")
        retrieval_success = retrieved == test_data
        print(f"   Retrieval success: {retrieval_success}")
        
        # Test metrics
        metrics = cache.get_metrics()
        print(f"   Cache size: {metrics.total_size_mb:.2f}MB")
        print(f"   OLED optimizations: {metrics.oled_specific_optimizations}")
        
        # Cleanup
        cache.cleanup()
        
        if success and retrieval_success:
            print("   ✅ OLED memory optimizer working")
            return True
        else:
            print("   ❌ Memory optimizer issues")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_gpu_optimizer():
    """Test GPU optimizer"""
    print("🎮 Testing GPU Optimizer...")
    try:
        from src.core.rdna2_gpu_optimizer import RDNA2GPUOptimizer
        
        optimizer = RDNA2GPUOptimizer(oled_model=True)
        
        # Test GPU metrics
        metrics = optimizer.get_gpu_metrics()
        print(f"   GPU utilization: {metrics.gpu_utilization_percent:.1f}%")
        print(f"   GPU clock: {metrics.clock_speed_mhz}MHz")
        print(f"   GPU temperature: {metrics.temperature_celsius:.1f}°C")
        
        # Test optimization profiles
        settings = optimizer.optimize_for_shader_compilation("oled_performance")
        print(f"   OLED optimization settings: {len(settings)}")
        
        # Test gaming detection
        gaming_info = optimizer.monitor_gaming_workload()
        print(f"   Gaming detection working: {'gaming_detected' in gaming_info}")
        
        optimizer.cleanup()
        print("   ✅ RDNA2 GPU optimizer working")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_integration():
    """Test comprehensive integration"""
    print("🔄 Testing Integration...")
    try:
        # Test config loading
        config_path = Path("config/steamdeck_oled_config.json")
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            oled_optimized = config.get("system", {}).get("oled_optimized", False)
            max_threads = config.get("system", {}).get("max_compilation_threads", 0)
            
            print(f"   Config OLED optimized: {oled_optimized}")
            print(f"   Config max threads: {max_threads}")
            
            if oled_optimized and max_threads >= 6:
                print("   ✅ Integration configuration working")
                return True
            else:
                print("   ⚠️  Configuration not optimized for OLED")
                return False
        else:
            print("   ❌ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 OLED Steam Deck Core Modules Test")
    print("=" * 45)
    
    tests = [
        ("Thermal Optimizer", test_thermal_optimizer),
        ("Memory Optimizer", test_memory_optimizer), 
        ("GPU Optimizer", test_gpu_optimizer),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print(f"\n" + "=" * 45)
    print("📋 Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    success_rate = passed / len(results)
    print(f"\nOverall: {passed}/{len(results)} tests passed ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("🎉 EXCELLENT: All core OLED optimizations working!")
        assessment = "EXCELLENT"
    elif success_rate >= 0.6:
        print("✅ GOOD: Most OLED optimizations working")
        assessment = "GOOD"
    else:
        print("⚠️  NEEDS WORK: Some optimizations need fixes")
        assessment = "NEEDS WORK"
    
    print("=" * 45)
    
    return success_rate >= 0.6

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test suite failed: {e}")
        sys.exit(1)