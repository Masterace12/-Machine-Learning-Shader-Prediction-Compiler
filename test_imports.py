#!/usr/bin/env python3
"""
Import test script for ML Shader Prediction Compiler
Tests all components with graceful degradation
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import(module_name, description):
    """Test an import and report results"""
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {description}: FAILED - {e}")
        return False

def test_from_import(module_name, items, description):
    """Test a from import and report results"""
    try:
        items_str = ", ".join(items)
        exec(f"from {module_name} import {items_str}")
        print(f"✅ {description}: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {description}: FAILED - {e}")
        return False

def test_helper_functions():
    """Test helper functions are available"""
    print("\n📦 Testing Helper Functions:")
    
    try:
        from src.core import get_optimized_predictor_safe, get_optimized_cache_safe
        predictor = get_optimized_predictor_safe()
        cache = get_optimized_cache_safe()
        print(f"✅ Core helpers: predictor={predictor is not None}, cache={cache is not None}")
    except Exception as e:
        print(f"❌ Core helpers: {e}")
    
    try:
        from src.optimization import get_thermal_manager
        thermal = get_thermal_manager()
        print(f"✅ Thermal helper: thermal={thermal is not None}")
    except Exception as e:
        print(f"❌ Thermal helper: {e}")
    
    try:
        from src.monitoring import get_performance_monitor
        perf = get_performance_monitor()
        print(f"✅ Performance helper: perf={perf is not None}")
    except Exception as e:
        print(f"❌ Performance helper: {e}")

def main():
    print("🔍 ML Shader Prediction Compiler - Import Test Suite")
    print("=" * 60)
    
    # Basic imports
    print("\n📝 Testing Basic Imports:")
    test_import("src", "Main src module")
    test_import("src.core", "Core module")
    test_import("src.optimization", "Optimization module")
    test_import("src.monitoring", "Monitoring module")
    test_import("src.rust_integration", "Rust integration module")
    
    # Component imports
    print("\n🧩 Testing Component Imports:")
    test_from_import("src.core", ["OptimizedMLPredictor"], "ML Predictor")
    test_from_import("src.core", ["OptimizedShaderCache"], "Shader Cache")
    test_from_import("src.core", ["HybridMLPredictor", "HybridVulkanCache"], "Hybrid Components")
    test_from_import("src.optimization", ["ThermalManager", "OptimizedThermalManager"], "Thermal Managers")
    test_from_import("src.monitoring", ["PerformanceMonitor"], "Performance Monitor")
    
    # Test helper functions
    test_helper_functions()
    
    # Test main module functionality
    print("\n🚀 Testing Main Module:")
    try:
        import main
        system = main.OptimizedShaderSystem()
        print(f"✅ Main system creation: SUCCESS")
        
        # Test component initialization
        if hasattr(system, 'ml_predictor'):
            ml = system.ml_predictor
            print(f"✅ ML predictor access: {ml is not None}")
        
        if hasattr(system, 'cache_manager'):
            cache = system.cache_manager
            print(f"✅ Cache manager access: {cache is not None}")
            
        if hasattr(system, 'thermal_manager'):
            thermal = system.thermal_manager
            print(f"✅ Thermal manager access: {thermal is not None}")
            
        if hasattr(system, 'performance_monitor'):
            perf = system.performance_monitor
            print(f"✅ Performance monitor access: {perf is not None}")
        
    except Exception as e:
        print(f"❌ Main system creation: {e}")
    
    print("\n🎉 Import test completed!")

if __name__ == "__main__":
    main()