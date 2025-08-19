#!/usr/bin/env python3
"""
Steam Deck Test Runner

Custom test runner for Steam Deck integration tests with support for:
- Hardware detection and conditional test execution
- Performance monitoring during test runs
- Test result reporting with Steam Deck-specific metrics
- Mocked vs real hardware test execution
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time


def detect_steam_deck() -> Dict[str, Any]:
    """Detect if running on Steam Deck and get hardware info"""
    detection_info = {
        'is_steam_deck': False,
        'model': 'unknown',
        'detection_method': None,
        'confidence': 0.0
    }
    
    # Check DMI information
    dmi_files = [
        '/sys/devices/virtual/dmi/id/product_name',
        '/sys/devices/virtual/dmi/id/board_name'
    ]
    
    for dmi_file in dmi_files:
        try:
            if os.path.exists(dmi_file):
                with open(dmi_file, 'r') as f:
                    content = f.read().strip().lower()
                    if 'galileo' in content:
                        detection_info.update({
                            'is_steam_deck': True,
                            'model': 'oled',
                            'detection_method': 'dmi_galileo',
                            'confidence': 0.95
                        })
                        return detection_info
                    elif 'jupiter' in content:
                        detection_info.update({
                            'is_steam_deck': True,
                            'model': 'lcd',
                            'detection_method': 'dmi_jupiter',
                            'confidence': 0.95
                        })
                        return detection_info
        except Exception:
            continue
    
    # Check for Steam Deck user
    if os.path.exists('/home/deck'):
        detection_info.update({
            'is_steam_deck': True,
            'model': 'unknown',
            'detection_method': 'user_directory',
            'confidence': 0.8
        })
        return detection_info
    
    # Check environment variable
    if os.environ.get('SteamDeck'):
        detection_info.update({
            'is_steam_deck': True,
            'model': 'unknown',
            'detection_method': 'environment',
            'confidence': 0.7
        })
    
    return detection_info


def get_test_categories() -> Dict[str, List[str]]:
    """Get available test categories with descriptions"""
    return {
        'unit': [
            'tests/unit/test_steamdeck_hardware_integration.py',
            'tests/unit/test_steamdeck_thermal_optimizer.py',
            'tests/unit/test_steamdeck_enhanced_integration.py',
            'tests/unit/test_steamdeck_cache_optimization.py',
            'tests/unit/test_steamdeck_dbus_integration.py'
        ],
        'integration': [
            'tests/integration/test_steamdeck_comprehensive_integration.py'
        ],
        'hardware': [
            'tests/unit/test_steamdeck_hardware_integration.py::TestSteamDeckHardwareMonitor',
            'tests/integration/test_steamdeck_comprehensive_integration.py::TestHardwareDetectionIntegration'
        ],
        'thermal': [
            'tests/unit/test_steamdeck_thermal_optimizer.py',
            'tests/integration/test_steamdeck_comprehensive_integration.py::TestThermalManagementIntegration'
        ],
        'power': [
            'tests/integration/test_steamdeck_comprehensive_integration.py::TestPowerManagementIntegration'
        ],
        'steam': [
            'tests/unit/test_steamdeck_dbus_integration.py',
            'tests/integration/test_steamdeck_comprehensive_integration.py::TestSteamIntegrationWorkflow'
        ],
        'cache': [
            'tests/unit/test_steamdeck_cache_optimization.py',
            'tests/integration/test_steamdeck_comprehensive_integration.py::TestPerformanceOptimizationIntegration'
        ],
        'workflow': [
            'tests/integration/test_steamdeck_comprehensive_integration.py::TestCompleteWorkflowValidation'
        ],
        'benchmark': [
            'tests/integration/test_steamdeck_comprehensive_integration.py::TestSteamDeckPerformanceBenchmarks',
            'tests/unit/test_steamdeck_enhanced_integration.py::TestPerformanceRegressionDetection'
        ]
    }


def run_pytest_command(test_paths: List[str], markers: List[str], 
                      extra_args: List[str], mock_hardware: bool = True) -> subprocess.CompletedProcess:
    """Run pytest with specified parameters"""
    cmd = ['python', '-m', 'pytest']
    
    # Add test paths
    cmd.extend(test_paths)
    
    # Add markers
    if markers:
        marker_expr = ' and '.join(markers)
        cmd.extend(['-m', marker_expr])
    
    # Add Steam Deck specific options
    cmd.extend([
        '--tb=short',
        '--verbose',
        '--durations=10',
        '--strict-markers'
    ])
    
    # Add mock hardware environment variable if needed
    env = os.environ.copy()
    if mock_hardware:
        env['STEAMDECK_TEST_MOCK'] = '1'
    
    # Add extra arguments
    cmd.extend(extra_args)
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Mock hardware: {mock_hardware}")
    
    return subprocess.run(cmd, env=env, capture_output=False)


def generate_test_report(detection_info: Dict[str, Any], 
                        test_results: subprocess.CompletedProcess,
                        test_duration: float) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    report = {
        'timestamp': time.time(),
        'test_duration_seconds': test_duration,
        'hardware_detection': detection_info,
        'test_execution': {
            'return_code': test_results.returncode,
            'success': test_results.returncode == 0
        },
        'environment': {
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd()
        }
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Steam Deck Integration Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --category unit                    # Run unit tests
  %(prog)s --category integration             # Run integration tests  
  %(prog)s --category hardware --real-hw      # Run hardware tests on real Steam Deck
  %(prog)s --markers steamdeck,thermal        # Run tests with specific markers
  %(prog)s --test-file test_thermal.py        # Run specific test file
  %(prog)s --list-categories                  # Show available test categories
        """
    )
    
    parser.add_argument(
        '--category', 
        choices=['unit', 'integration', 'hardware', 'thermal', 'power', 'steam', 'cache', 'workflow', 'benchmark'],
        help='Run tests from specific category'
    )
    
    parser.add_argument(
        '--markers', 
        type=str,
        help='Comma-separated list of pytest markers (e.g., steamdeck,thermal)'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        help='Run specific test file'
    )
    
    parser.add_argument(
        '--real-hw',
        action='store_true',
        help='Run tests on real Steam Deck hardware (default: use mocks)'
    )
    
    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List available test categories and exit'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        default='steamdeck_test_report.json',
        help='Output file for test report (default: steamdeck_test_report.json)'
    )
    
    parser.add_argument(
        '--detect-only',
        action='store_true',
        help='Only run hardware detection and exit'
    )
    
    parser.add_argument(
        'pytest_args',
        nargs='*',
        help='Additional arguments passed to pytest'
    )
    
    args = parser.parse_args()
    
    # Detect Steam Deck hardware
    print("🔍 Detecting Steam Deck hardware...")
    detection_info = detect_steam_deck()
    
    print(f"Hardware Detection Results:")
    print(f"  Is Steam Deck: {detection_info['is_steam_deck']}")
    print(f"  Model: {detection_info['model']}")
    print(f"  Detection Method: {detection_info['detection_method']}")
    print(f"  Confidence: {detection_info['confidence']:.1%}")
    
    if args.detect_only:
        return 0
    
    # Show available categories
    if args.list_categories:
        print("\n📋 Available Test Categories:")
        categories = get_test_categories()
        for category, test_files in categories.items():
            print(f"\n  {category.upper()}:")
            for test_file in test_files:
                print(f"    - {test_file}")
        return 0
    
    # Determine test paths
    test_paths = []
    if args.category:
        categories = get_test_categories()
        if args.category in categories:
            test_paths = categories[args.category]
        else:
            print(f"❌ Unknown category: {args.category}")
            return 1
    elif args.test_file:
        test_paths = [args.test_file]
    else:
        # Default: run all Steam Deck tests
        test_paths = ['tests/unit/test_steamdeck*.py', 'tests/integration/test_steamdeck*.py']
    
    # Parse markers
    markers = []
    if args.markers:
        markers = [m.strip() for m in args.markers.split(',')]
    
    # Add steamdeck marker by default
    if 'steamdeck' not in markers:
        markers.append('steamdeck')
    
    # Warn about real hardware testing
    if args.real_hw:
        if not detection_info['is_steam_deck']:
            print("⚠️  WARNING: --real-hw specified but Steam Deck not detected!")
            print("   Tests may fail or behave unexpectedly.")
            response = input("   Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return 1
        else:
            print("🎮 Running tests on real Steam Deck hardware")
    else:
        print("🎭 Running tests with mocked hardware")
    
    # Run tests
    print(f"\n🧪 Running Steam Deck integration tests...")
    print(f"   Test paths: {test_paths}")
    print(f"   Markers: {markers}")
    
    start_time = time.time()
    
    try:
        test_results = run_pytest_command(
            test_paths=test_paths,
            markers=markers,
            extra_args=args.pytest_args,
            mock_hardware=not args.real_hw
        )
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return 130
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    # Generate report
    report = generate_test_report(detection_info, test_results, test_duration)
    
    # Save report
    with open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n📊 Test Execution Summary:")
    print(f"   Duration: {test_duration:.2f} seconds")
    print(f"   Exit Code: {test_results.returncode}")
    print(f"   Success: {'✅' if test_results.returncode == 0 else '❌'}")
    print(f"   Report: {args.output_report}")
    
    if detection_info['is_steam_deck']:
        print(f"   Hardware: Steam Deck {detection_info['model'].upper()}")
    
    return test_results.returncode


if __name__ == '__main__':
    sys.exit(main())
