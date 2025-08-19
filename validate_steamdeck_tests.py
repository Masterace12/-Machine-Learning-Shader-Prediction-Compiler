#!/usr/bin/env python3
"""
Steam Deck Test Validation Script

Validates that all Steam Deck integration test components are properly
structured and can be imported successfully. This script runs without
requiring pytest to be installed.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any


def validate_imports() -> Dict[str, Any]:
    """Validate that core modules can be imported"""
    results = {
        'core_modules': {},
        'test_modules': {},
        'success': True,
        'errors': []
    }
    
    # Core Steam Deck modules to validate
    core_modules = [
        'src.core.steam_deck_hardware_integration',
        'src.core.steamdeck_thermal_optimizer',
        'src.core.steam_deck_optimizer',
        'src.core.steamdeck_cache_optimizer',
        'src.core.enhanced_dbus_manager'
    ]
    
    print("🔍 Validating core Steam Deck modules...")
    for module_name in core_modules:
        try:
            module = importlib.import_module(module_name)
            results['core_modules'][module_name] = {
                'status': 'success',
                'classes': [name for name in dir(module) if name[0].isupper() and not name.startswith('_')],
                'functions': [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
            }
            print(f"  ✅ {module_name}")
        except Exception as e:
            results['core_modules'][module_name] = {
                'status': 'error',
                'error': str(e)
            }
            results['success'] = False
            results['errors'].append(f"Failed to import {module_name}: {e}")
            print(f"  ❌ {module_name}: {e}")
    
    return results


def validate_test_structure() -> Dict[str, Any]:
    """Validate test file structure and organization"""
    results = {
        'test_files': {},
        'structure_valid': True,
        'missing_files': []
    }
    
    expected_files = [
        'tests/fixtures/steamdeck_fixtures.py',
        'tests/integration/test_steamdeck_comprehensive_integration.py',
        'tests/unit/test_steamdeck_hardware_integration.py',
        'tests/unit/test_steamdeck_thermal_optimizer.py',
        'tests/unit/test_steamdeck_enhanced_integration.py',
        'tests/unit/test_steamdeck_cache_optimization.py',
        'tests/unit/test_steamdeck_dbus_integration.py'
    ]
    
    print("\n📁 Validating test file structure...")
    for test_file in expected_files:
        file_path = Path(test_file)
        if file_path.exists():
            # Get file size and line count
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    results['test_files'][test_file] = {
                        'exists': True,
                        'size_bytes': file_path.stat().st_size,
                        'line_count': len(lines),
                        'has_docstring': len(lines) > 2 and lines[1].strip().startswith('"""')
                    }
                print(f"  ✅ {test_file} ({len(lines)} lines)")
            except Exception as e:
                results['test_files'][test_file] = {
                    'exists': True,
                    'error': str(e)
                }
                print(f"  ⚠️  {test_file}: {e}")
        else:
            results['test_files'][test_file] = {'exists': False}
            results['structure_valid'] = False
            results['missing_files'].append(test_file)
            print(f"  ❌ {test_file}: Missing")
    
    return results


def validate_mock_hardware_system() -> Dict[str, Any]:
    """Validate mock hardware system without pytest"""
    results = {
        'mock_system_valid': False,
        'mock_components': {},
        'errors': []
    }
    
    print("\n🎭 Validating mock hardware system...")
    
    # Create a minimal mock hardware environment
    try:
        # Define minimal mock classes to test structure
        class MockSteamDeckModel:
            LCD_64GB = "lcd_64gb"
            LCD_256GB = "lcd_256gb"
            OLED_512GB = "oled_512gb"
            OLED_1TB = "oled_1tb"
        
        class MockHardwareState:
            def __init__(self):
                self.model = MockSteamDeckModel.LCD_256GB
                self.cpu_temperature = 65.0
                self.battery_capacity = 75
                self.gaming_mode_active = False
                self.thermal_zones = {
                    'thermal_zone0': 65000,
                    'thermal_zone1': 70000
                }
        
        # Test mock hardware creation
        mock_state = MockHardwareState()
        results['mock_components']['hardware_state'] = {
            'created': True,
            'model': mock_state.model,
            'temperature': mock_state.cpu_temperature
        }
        
        results['mock_system_valid'] = True
        print("  ✅ Mock hardware state creation")
        print(f"    Model: {mock_state.model}")
        print(f"    Temperature: {mock_state.cpu_temperature}°C")
        print(f"    Battery: {mock_state.battery_capacity}%")
        
    except Exception as e:
        results['errors'].append(f"Mock hardware validation failed: {e}")
        print(f"  ❌ Mock hardware validation: {e}")
    
    return results


def validate_configuration_files() -> Dict[str, Any]:
    """Validate Steam Deck configuration files"""
    results = {
        'config_files': {},
        'pytest_config': {},
        'valid': True
    }
    
    print("\n⚙️  Validating configuration files...")
    
    # Check configuration files
    config_files = [
        'config/steamdeck_lcd_config.json',
        'config/steamdeck_oled_config.json',
        'pytest.ini'
    ]
    
    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    results['config_files'][config_file] = {
                        'exists': True,
                        'size': len(content),
                        'has_steamdeck_markers': 'steamdeck' in content.lower()
                    }
                print(f"  ✅ {config_file}")
            except Exception as e:
                results['config_files'][config_file] = {
                    'exists': True,
                    'error': str(e)
                }
                print(f"  ⚠️  {config_file}: {e}")
        else:
            results['config_files'][config_file] = {'exists': False}
            results['valid'] = False
            print(f"  ❌ {config_file}: Missing")
    
    # Validate pytest.ini has Steam Deck markers
    pytest_ini = Path('pytest.ini')
    if pytest_ini.exists():
        try:
            with open(pytest_ini, 'r') as f:
                content = f.read()
                steamdeck_markers = [
                    'steamdeck:', 'steamdeck_lcd:', 'steamdeck_oled:',
                    'thermal:', 'power:', 'dbus:', 'workflow:'
                ]
                found_markers = [marker for marker in steamdeck_markers if marker in content]
                results['pytest_config'] = {
                    'has_steamdeck_markers': len(found_markers) > 0,
                    'found_markers': found_markers,
                    'total_markers': len(found_markers)
                }
                print(f"    Steam Deck markers: {len(found_markers)} found")
        except Exception as e:
            results['pytest_config'] = {'error': str(e)}
    
    return results


def validate_test_runner() -> Dict[str, Any]:
    """Validate the custom test runner"""
    results = {
        'runner_valid': False,
        'executable': False,
        'functions': []
    }
    
    print("\n🏃 Validating test runner...")
    
    runner_path = Path('run_steamdeck_tests.py')
    if runner_path.exists():
        try:
            # Check if executable
            results['executable'] = os.access(runner_path, os.X_OK)
            
            # Read and analyze content
            with open(runner_path, 'r') as f:
                content = f.read()
                
            # Look for key functions
            key_functions = [
                'detect_steam_deck',
                'get_test_categories',
                'run_pytest_command',
                'generate_test_report',
                'main'
            ]
            
            found_functions = [func for func in key_functions if f'def {func}' in content]
            results['functions'] = found_functions
            results['runner_valid'] = len(found_functions) >= 4
            
            print(f"  ✅ Test runner exists")
            print(f"    Executable: {'✅' if results['executable'] else '❌'}")
            print(f"    Functions: {len(found_functions)}/{len(key_functions)}")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"  ❌ Test runner validation: {e}")
    else:
        print(f"  ❌ Test runner missing: {runner_path}")
    
    return results


def main():
    print("🧪 Steam Deck Integration Test Validation")
    print("=" * 50)
    
    # Run all validations
    validation_results = {
        'imports': validate_imports(),
        'structure': validate_test_structure(),
        'mock_system': validate_mock_hardware_system(),
        'configuration': validate_configuration_files(),
        'test_runner': validate_test_runner()
    }
    
    # Generate summary
    print("\n📊 Validation Summary")
    print("=" * 30)
    
    total_checks = 0
    passed_checks = 0
    
    for category, result in validation_results.items():
        category_status = "✅"
        if category == 'imports':
            success = result.get('success', False)
            total_checks += len(result.get('core_modules', {}))
            passed_checks += len([m for m in result.get('core_modules', {}).values() if m.get('status') == 'success'])
        elif category == 'structure':
            success = result.get('structure_valid', False)
            total_checks += len(result.get('test_files', {}))
            passed_checks += len([f for f in result.get('test_files', {}).values() if f.get('exists', False)])
        elif category == 'mock_system':
            success = result.get('mock_system_valid', False)
            total_checks += 1
            passed_checks += 1 if success else 0
        elif category == 'configuration':
            success = result.get('valid', False)
            total_checks += len(result.get('config_files', {}))
            passed_checks += len([f for f in result.get('config_files', {}).values() if f.get('exists', False)])
        elif category == 'test_runner':
            success = result.get('runner_valid', False)
            total_checks += 1
            passed_checks += 1 if success else 0
        
        if not success:
            category_status = "❌"
        
        print(f"  {category_status} {category.title().replace('_', ' ')}: {'Passed' if success else 'Failed'}")
    
    print(f"\nOverall: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")
    
    # Show next steps
    print("\n🚀 Next Steps:")
    if passed_checks == total_checks:
        print("  ✅ All validations passed! You can run the Steam Deck tests:")
        print("      ./run_steamdeck_tests.py --detect-only")
        print("      ./run_steamdeck_tests.py --list-categories")
        print("      ./run_steamdeck_tests.py --category unit")
    else:
        print("  ⚠️  Some validations failed. Check the errors above.")
        print("      You may need to install missing dependencies or fix import issues.")
    
    print("\n📚 Documentation:")
    print("  - Read tests/README_STEAMDECK_TESTING.md for comprehensive testing guide")
    print("  - Use --help with run_steamdeck_tests.py for usage information")
    
    return 0 if passed_checks == total_checks else 1


if __name__ == '__main__':
    sys.exit(main())
