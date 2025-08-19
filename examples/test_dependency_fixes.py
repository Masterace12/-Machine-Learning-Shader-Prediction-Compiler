#!/usr/bin/env python3
"""
Test script to verify all dependency fixes are working correctly
"""

import sys
import os
import time
import traceback

# Add project to path
sys.path.insert(0, '/home/deck/Downloads/-Machine-Learning-Shader-Prediction-Compiler')

def test_dependency_imports():
    """Test that all dependencies can be imported"""
    print("Testing dependency imports...")
    print("-" * 60)
    
    dependencies = [
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('lightgbm', 'lightgbm'),
        ('numba', 'numba'),
        ('numexpr', 'numexpr'),
        ('bottleneck', 'bottleneck'),
        ('msgpack', 'msgpack'),
        ('zstandard', 'zstandard'),
        ('psutil', 'psutil'),
        ('dbus-next', 'dbus_next'),
        ('distro', 'distro'),
    ]
    
    success_count = 0
    for name, import_name in dependencies:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:20} v{version}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {name:20} MISSING - {e}")
        except Exception as e:
            print(f"⚠ {name:20} ERROR - {e}")
    
    print(f"\nResult: {success_count}/{len(dependencies)} dependencies available")
    return success_count == len(dependencies)

def test_ml_functionality():
    """Test that ML functionality works"""
    print("\nTesting ML functionality...")
    print("-" * 60)
    
    try:
        import numpy as np
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        
        # Create sample data
        print("Creating sample data...")
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train LightGBM model
        print("Training LightGBM model...")
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'num_threads': 1
        }
        model = lgb.train(params, train_data, num_boost_round=10)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate simple error
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        print(f"✓ LightGBM model trained successfully (RMSE: {rmse:.4f})")
        
        return True
        
    except Exception as e:
        print(f"✗ ML functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_optimizations():
    """Test performance optimization libraries"""
    print("\nTesting performance optimizations...")
    print("-" * 60)
    
    results = []
    
    # Test Numba
    try:
        from numba import njit
        import numpy as np
        
        @njit
        def numba_sum(arr):
            return np.sum(arr ** 2)
        
        arr = np.random.rand(1000)
        result = numba_sum(arr)
        print(f"✓ Numba JIT compilation works")
        results.append(True)
    except Exception as e:
        print(f"✗ Numba test failed: {e}")
        results.append(False)
    
    # Test NumExpr
    try:
        import numexpr as ne
        import numpy as np
        
        a = np.random.rand(10000)
        b = np.random.rand(10000)
        result = ne.evaluate('a * b + a - b')
        print(f"✓ NumExpr evaluation works")
        results.append(True)
    except Exception as e:
        print(f"✗ NumExpr test failed: {e}")
        results.append(False)
    
    # Test Bottleneck
    try:
        import bottleneck as bn
        import numpy as np
        
        arr = np.random.rand(1000, 100)
        result = bn.nanmean(arr, axis=1)
        print(f"✓ Bottleneck operations work")
        results.append(True)
    except Exception as e:
        print(f"✗ Bottleneck test failed: {e}")
        results.append(False)
    
    return all(results)

def test_dependency_coordinator():
    """Test the dependency coordinator"""
    print("\nTesting dependency coordinator...")
    print("-" * 60)
    
    try:
        from src.core.dependency_coordinator import DependencyCoordinator
        
        coordinator = DependencyCoordinator()
        status = coordinator.detect_all_dependencies()
        
        working = sum(1 for s in status.values() if s.available and s.test_passed)
        total = len(status)
        
        print(f"✓ Dependency coordinator initialized")
        print(f"  Dependencies detected: {len(status)}")
        print(f"  Working dependencies: {working}/{total}")
        
        # Validate installation
        health = coordinator.validate_installation()
        health_score = health.get('overall_health', 0)
        print(f"  Overall health: {health_score:.1%}")
        
        return health_score >= 0.8
        
    except Exception as e:
        print(f"✗ Dependency coordinator test failed: {e}")
        traceback.print_exc()
        return False

def test_health_checker():
    """Test the health checker"""
    print("\nTesting dependency health checker...")
    print("-" * 60)
    
    try:
        from src.core.dependency_health_checker import DependencyHealthChecker, quick_health_check
        
        # Quick check
        all_good, score, issues = quick_health_check()
        
        print(f"✓ Health checker initialized")
        print(f"  Health score: {score:.1%}")
        print(f"  Status: {'GOOD' if all_good else 'ISSUES FOUND'}")
        
        if issues:
            print("  Issues:")
            for issue in issues:
                print(f"    - {issue}")
        
        return score >= 0.8
        
    except Exception as e:
        print(f"✗ Health checker test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("DEPENDENCY SYSTEM VALIDATION TEST")
    print("=" * 80)
    
    tests = [
        ("Dependency Imports", test_dependency_imports),
        ("ML Functionality", test_ml_functionality),
        ("Performance Optimizations", test_performance_optimizations),
        ("Dependency Coordinator", test_dependency_coordinator),
        ("Health Checker", test_health_checker),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:30} {status}")
    
    print("-" * 80)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Dependency system is working correctly!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Please review the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())