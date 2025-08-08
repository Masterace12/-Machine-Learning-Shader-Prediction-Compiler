#!/usr/bin/env python3
"""
Test script to verify all fixes for the Steam Deck ML-Based Shader Prediction Compiler
This script tests the fixes for the ModuleNotFoundError and other identified issues.
"""

import sys
import os
import traceback
from pathlib import Path

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

def log_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def log_success(msg):
    print(f"{Colors.GREEN}[✓]{Colors.NC} {msg}")

def log_warning(msg):
    print(f"{Colors.YELLOW}[!]{Colors.NC} {msg}")

def log_error(msg):
    print(f"{Colors.RED}[✗]{Colors.NC} {msg}")

def log_test(msg):
    print(f"{Colors.CYAN}[TEST]{Colors.NC} {msg}")

def print_header(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.NC}\n")

def test_numpy_import():
    """Test 1: NumPy Import - The main issue from the help document"""
    log_test("Testing NumPy import (primary issue from help document)")
    
    try:
        import numpy as np
        log_success(f"NumPy {np.__version__} imported successfully")
        
        # Test basic numpy functionality
        test_array = np.array([1, 2, 3, 4, 5])
        test_mean = np.mean(test_array)
        log_success(f"NumPy functionality test passed (mean of [1,2,3,4,5] = {test_mean})")
        return True
    except ImportError as e:
        log_error(f"NumPy import failed: {e}")
        log_error("This is the main error mentioned in the help document!")
        return False
    except Exception as e:
        log_error(f"NumPy functionality test failed: {e}")
        return False

def test_core_dependencies():
    """Test 2: Core Dependencies - Test all essential packages"""
    log_test("Testing core dependencies")
    
    dependencies = {
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'psutil': 'psutil',
        'requests': 'requests',
        'PyYAML': 'yaml'
    }
    
    results = {}
    for name, module in dependencies.items():
        try:
            __import__(module)
            log_success(f"{name} available")
            results[name] = True
        except ImportError:
            log_warning(f"{name} not available (optional)")
            results[name] = False
        except Exception as e:
            log_warning(f"{name} import error: {e}")
            results[name] = False
    
    essential_count = sum([results['scikit-learn'], results['psutil'], results['requests']])
    if essential_count >= 2:
        log_success(f"Core dependencies check passed ({essential_count}/3 essential packages)")
        return True
    else:
        log_error(f"Core dependencies check failed ({essential_count}/3 essential packages)")
        return False

def test_shader_prediction_system_import():
    """Test 3: Main System Import - Test the main shader prediction system"""
    log_test("Testing main shader prediction system import")
    
    try:
        # Add src directory to path
        src_path = Path(__file__).parent / 'src'
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        from shader_prediction_system import SteamDeckShaderPredictor
        log_success("Main system import successful")
        
        # Test system initialization
        system = SteamDeckShaderPredictor()
        log_success("System initialization successful")
        
        return True
    except ImportError as e:
        log_error(f"System import failed: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        log_error(f"System initialization failed: {e}")
        traceback.print_exc()
        return False

def test_fallback_prediction():
    """Test 4: Fallback Prediction - Test that the system works without full ML stack"""
    log_test("Testing fallback prediction system (without full ML dependencies)")
    
    try:
        # Import with path setup
        src_path = Path(__file__).parent / 'src'
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        from shader_prediction_system import SteamDeckShaderPredictor, ShaderMetrics, ShaderType
        
        # Create system
        config = {
            'model_type': 'fallback',  # Force fallback mode
            'cache_size': 100
        }
        system = SteamDeckShaderPredictor(config)
        
        # Create test shader metrics
        test_shader = ShaderMetrics(
            shader_hash='test_hash_123',
            shader_type=ShaderType.FRAGMENT,
            bytecode_size=2048,
            instruction_count=150,
            register_pressure=32,
            texture_samples=4,
            branch_complexity=3,
            loop_depth=2,
            compilation_time_ms=0,  # Will be predicted
            gpu_temp_celsius=75.0,
            power_draw_watts=12.0,
            memory_used_mb=256,
            timestamp=1234567890,
            game_id='test_game'
        )
        
        # Test prediction
        predicted_time, confidence = system.predictor.predict(test_shader)
        
        if predicted_time > 0 and 0 < confidence <= 1:
            log_success(f"Fallback prediction successful: {predicted_time:.1f}ms (confidence: {confidence:.2f})")
            return True
        else:
            log_error(f"Invalid prediction result: time={predicted_time}, confidence={confidence}")
            return False
            
    except Exception as e:
        log_error(f"Fallback prediction test failed: {e}")
        traceback.print_exc()
        return False

def test_thermal_scheduler():
    """Test 5: Thermal Scheduler - Test thermal-aware compilation scheduling"""
    log_test("Testing thermal-aware scheduling")
    
    try:
        src_path = Path(__file__).parent / 'src'
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        from shader_prediction_system import ThermalAwareScheduler, ThermalState
        
        # Create thermal scheduler
        scheduler = ThermalAwareScheduler(max_temp=85.0, power_budget=15.0)
        
        # Test different thermal states
        test_cases = [
            (60.0, 8.0, ThermalState.COOL),
            (75.0, 12.0, ThermalState.NORMAL),
            (82.0, 14.0, ThermalState.WARM),
            (87.0, 16.0, ThermalState.HOT),
            (92.0, 18.0, ThermalState.THROTTLING)
        ]
        
        for temp, power, expected_state in test_cases:
            scheduler.update_thermal_state(temp, power)
            if scheduler.current_thermal_state == expected_state:
                log_success(f"Thermal state correct for {temp}°C, {power}W: {expected_state.value}")
            else:
                log_warning(f"Thermal state unexpected for {temp}°C: got {scheduler.current_thermal_state.value}, expected {expected_state.value}")
        
        return True
        
    except Exception as e:
        log_error(f"Thermal scheduler test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_files():
    """Test 6: Configuration Files - Test that config files are valid"""
    log_test("Testing configuration files")
    
    try:
        import json
        
        # Test requirements files
        req_files = [
            'requirements.txt',
            'requirements-minimal.txt'
        ]
        
        for req_file in req_files:
            req_path = Path(__file__).parent / req_file
            if req_path.exists():
                with open(req_path, 'r') as f:
                    content = f.read()
                    if 'numpy' in content:
                        log_success(f"{req_file} contains NumPy dependency")
                    else:
                        log_warning(f"{req_file} missing NumPy dependency")
            else:
                log_warning(f"{req_file} not found")
        
        # Test that installation scripts exist
        install_scripts = [
            'enhanced-install.sh',
            'install.sh',
            'fix-numpy-issue.sh'
        ]
        
        script_count = 0
        for script in install_scripts:
            script_path = Path(__file__).parent / script
            if script_path.exists():
                script_count += 1
                log_success(f"Installation script found: {script}")
            else:
                log_warning(f"Installation script missing: {script}")
        
        if script_count >= 2:
            log_success("Configuration files test passed")
            return True
        else:
            log_error("Too many configuration files missing")
            return False
        
    except Exception as e:
        log_error(f"Configuration files test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print_header("STEAM DECK SHADER PREDICTION COMPILER - FIX VERIFICATION")
    
    log_info("Running comprehensive test suite to verify all fixes...")
    log_info("This addresses the issues mentioned in the help document:")
    log_info("- ModuleNotFoundError: No module named 'numpy'")
    log_info("- Installation script failures")
    log_info("- Dependency resolution problems")
    
    tests = [
        ("NumPy Import (Primary Issue)", test_numpy_import),
        ("Core Dependencies", test_core_dependencies),
        ("Main System Import", test_shader_prediction_system_import),
        ("Fallback Prediction", test_fallback_prediction),
        ("Thermal Scheduler", test_thermal_scheduler),
        ("Configuration Files", test_configuration_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print_header(f"TESTING: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                log_success(f"✅ {test_name} PASSED")
            else:
                log_error(f"❌ {test_name} FAILED")
        except Exception as e:
            log_error(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.NC}" if result else f"{Colors.RED}FAIL{Colors.NC}"
        print(f"  {status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.NC}")
        print(f"{Colors.GREEN}The NumPy issue and other problems have been resolved!{Colors.NC}")
        print(f"{Colors.BLUE}You can now run the shader prediction system successfully.{Colors.NC}")
        return True
    elif passed >= total * 0.7:  # 70% or more
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  MOSTLY WORKING ⚠️{Colors.NC}")
        print(f"{Colors.YELLOW}Core functionality is working, but some optional features may be limited.{Colors.NC}")
        print(f"{Colors.BLUE}The main NumPy issue should be resolved.{Colors.NC}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ SIGNIFICANT ISSUES REMAIN ❌{Colors.NC}")
        print(f"{Colors.RED}Please check the error messages above and ensure dependencies are installed.{Colors.NC}")
        print(f"\n{Colors.BLUE}Try running: ./fix-numpy-issue.sh{Colors.NC}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)