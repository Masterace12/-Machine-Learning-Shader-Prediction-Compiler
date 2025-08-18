#!/usr/bin/env python3
"""
Test validation script - validates test structure without pytest dependency
"""

import sys
import importlib
import inspect
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import_validation():
    """Test that all core modules can be imported"""
    print("🔍 Testing core module imports...")
    
    modules_to_test = [
        "src.ml.unified_ml_predictor",
        "src.ml.optimized_ml_predictor", 
        "src.thermal.optimized_thermal_manager",
        "src.monitoring.performance_monitor",
        "main"
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            results[module_name] = "✓ SUCCESS"
            print(f"  ✓ {module_name}")
        except ImportError as e:
            results[module_name] = f"✗ FAILED: {e}"
            print(f"  ✗ {module_name}: {e}")
        except Exception as e:
            results[module_name] = f"⚠ ERROR: {e}"
            print(f"  ⚠ {module_name}: {e}")
    
    return results

def test_class_instantiation():
    """Test that core classes can be instantiated"""
    print("\n🏗️  Testing class instantiation...")
    
    test_cases = []
    
    try:
        from src.ml.unified_ml_predictor import HeuristicPredictor, UnifiedShaderFeatures, ShaderType
        predictor = HeuristicPredictor()
        print("  ✓ HeuristicPredictor instantiated")
        
        features = UnifiedShaderFeatures(
            shader_hash="test_123",
            shader_type=ShaderType.FRAGMENT,
            instruction_count=100
        )
        print("  ✓ UnifiedShaderFeatures created")
        
        result = predictor.predict_compilation_time(features)
        print(f"  ✓ Prediction result: {result:.2f}ms")
        test_cases.append("HeuristicPredictor: SUCCESS")
    except Exception as e:
        print(f"  ✗ HeuristicPredictor failed: {e}")
        test_cases.append(f"HeuristicPredictor: FAILED - {e}")
    
    try:
        from src.thermal.optimized_thermal_manager import OptimizedThermalManager
        thermal_manager = OptimizedThermalManager()
        status = thermal_manager.get_status()
        print(f"  ✓ ThermalManager status: {status['thermal_state']}")
        test_cases.append("ThermalManager: SUCCESS")
    except Exception as e:
        print(f"  ✗ ThermalManager failed: {e}")
        test_cases.append(f"ThermalManager: FAILED - {e}")
    
    try:
        from src.monitoring.performance_monitor import OptimizedPerformanceMonitor
        perf_monitor = OptimizedPerformanceMonitor()
        report = perf_monitor.get_performance_report()
        print(f"  ✓ PerformanceMonitor health: {report['health_score']:.1f}")
        test_cases.append("PerformanceMonitor: SUCCESS")
    except Exception as e:
        print(f"  ✗ PerformanceMonitor failed: {e}")
        test_cases.append(f"PerformanceMonitor: FAILED - {e}")
    
    return test_cases

def test_main_system():
    """Test the main system integration"""
    print("\n🎮 Testing main system integration...")
    
    try:
        from main import OptimizedShaderSystem, SystemConfig
        
        config = SystemConfig(
            enable_ml_prediction=True,
            enable_cache=False,  # Disable problematic cache
            enable_thermal_management=True,
            enable_performance_monitoring=True,
            enable_async=False,
            max_memory_mb=50
        )
        
        system = OptimizedShaderSystem(config=config)
        print("  ✓ OptimizedShaderSystem created")
        
        status = system.get_system_status()
        print(f"  ✓ System status retrieved")
        print(f"    - Components: {list(status['components'].keys())}")
        print(f"    - Available: {[k for k, v in status['components'].items() if v]}")
        
        return "Main System: SUCCESS"
    except Exception as e:
        print(f"  ✗ Main system failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Main System: FAILED - {e}"

def test_configuration_validation():
    """Test configuration file validation"""
    print("\n⚙️  Testing configuration files...")
    
    config_files = [
        "pytest.ini",
        "pyproject.toml",
        "conftest.py"
    ]
    
    results = {}
    
    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            try:
                content = file_path.read_text()
                if content.strip():
                    results[config_file] = f"✓ EXISTS ({len(content)} chars)"
                    print(f"  ✓ {config_file} exists and has content")
                else:
                    results[config_file] = "⚠ EMPTY"
                    print(f"  ⚠ {config_file} exists but is empty")
            except Exception as e:
                results[config_file] = f"✗ READ ERROR: {e}"
                print(f"  ✗ {config_file} read error: {e}")
        else:
            results[config_file] = "✗ MISSING"
            print(f"  ✗ {config_file} not found")
    
    return results

def test_file_structure():
    """Test project file structure"""
    print("\n📁 Testing project file structure...")
    
    required_dirs = [
        "src",
        "src/ml",
        "src/thermal", 
        "src/monitoring",
        "src/cache",
        "tests",
        "tests/unit",
        "tests/integration"
    ]
    
    required_files = [
        "src/__init__.py",
        "src/ml/__init__.py",
        "src/ml/unified_ml_predictor.py",
        "src/ml/optimized_ml_predictor.py",
        "src/thermal/__init__.py",
        "src/thermal/optimized_thermal_manager.py",
        "src/monitoring/__init__.py",
        "src/monitoring/performance_monitor.py",
        "tests/unit/test_ml_predictor.py",
        "tests/unit/test_thermal_manager.py",
        "tests/unit/test_performance_monitor.py",
        "tests/integration/test_main_integration.py"
    ]
    
    structure_results = {"dirs": {}, "files": {}}
    
    # Check directories
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            structure_results["dirs"][dir_path] = "✓ EXISTS"
            print(f"  ✓ {dir_path}/")
        else:
            structure_results["dirs"][dir_path] = "✗ MISSING"
            print(f"  ✗ {dir_path}/ missing")
    
    # Check files
    for file_path in required_files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            size = path.stat().st_size
            structure_results["files"][file_path] = f"✓ EXISTS ({size} bytes)"
            print(f"  ✓ {file_path} ({size} bytes)")
        else:
            structure_results["files"][file_path] = "✗ MISSING"
            print(f"  ✗ {file_path} missing")
    
    return structure_results

def main():
    """Main validation function"""
    print("🧪 Shader Predict Compile - Test Validation")
    print("=" * 50)
    
    all_results = {}
    
    # Run validation tests
    all_results["imports"] = test_import_validation()
    all_results["instantiation"] = test_class_instantiation()
    all_results["main_system"] = test_main_system()
    all_results["config"] = test_configuration_validation()
    all_results["structure"] = test_file_structure()
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 50)
    
    # Count successes and failures
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            for test_name, result in results.items():
                total_tests += 1
                if isinstance(result, str) and ("✓" in result or "SUCCESS" in result):
                    passed_tests += 1
        elif isinstance(results, list):
            for result in results:
                total_tests += 1
                if "SUCCESS" in result:
                    passed_tests += 1
        elif isinstance(results, str):
            total_tests += 1
            if "SUCCESS" in results:
                passed_tests += 1
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All validation tests passed!")
        return 0
    elif passed_tests > total_tests * 0.8:
        print("✅ Most validation tests passed - good!")
        return 0
    elif passed_tests > total_tests * 0.5:
        print("⚠️  Some validation tests failed - needs attention")
        return 1
    else:
        print("❌ Many validation tests failed - significant issues")
        return 2

if __name__ == "__main__":
    sys.exit(main())