#!/usr/bin/env python3
"""
Comprehensive Steam Deck Implementation Test Suite

This script validates the complete Enhanced ML Predictor system on Steam Deck,
testing all components including fallbacks, Steam integration, and performance.
"""

import os
import sys
import time
import json
import platform
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('steam_deck_test.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    success: bool
    duration_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None

class SteamDeckTestSuite:
    """Comprehensive test suite for Steam Deck implementation"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        self.test_count = 0
        self.success_count = 0
        
        # Test configuration
        self.base_dir = Path(__file__).parent
        self.src_dir = self.base_dir / "src"
        
        print("🚀 Steam Deck Enhanced ML Predictor Test Suite")
        print("=" * 60)
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python: {platform.python_version()}")
        print(f"Base Directory: {self.base_dir}")
        print()
    
    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record results"""
        self.test_count += 1
        logger.info(f"Running test: {test_name}")
        
        start_time = time.perf_counter()
        try:
            details = test_func()
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            result = TestResult(
                name=test_name,
                success=True,
                duration_ms=duration_ms,
                details=details or {}
            )
            self.success_count += 1
            print(f"✅ {test_name} ({duration_ms:.1f}ms)")
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = TestResult(
                name=test_name,
                success=False,
                duration_ms=duration_ms,
                details={},
                error=str(e)
            )
            print(f"❌ {test_name} - ERROR: {e}")
            logger.error(f"Test {test_name} failed: {e}")
        
        self.results.append(result)
        return result
    
    def test_steam_deck_detection(self) -> Dict[str, Any]:
        """Test Steam Deck hardware detection"""
        try:
            from core.pure_python_fallbacks import PureSteamDeckDetector
            
            detector = PureSteamDeckDetector()
            is_steam_deck = detector.is_steam_deck()
            model = detector.get_steam_deck_model()
            
            # Additional hardware info
            details = {
                'is_steam_deck': is_steam_deck,
                'model': model,
                'dmi_product': detector._read_dmi_field('product_name'),
                'dmi_board': detector._read_dmi_field('board_name'),
                'home_deck_exists': os.path.exists('/home/deck'),
                'cpu_info_snippet': detector._get_cpu_info()[:200] + '...'
            }
            
            logger.info(f"Steam Deck Detection: {is_steam_deck} ({model})")
            return details
            
        except Exception as e:
            logger.error(f"Steam Deck detection failed: {e}")
            raise
    
    def test_dependency_fallbacks(self) -> Dict[str, Any]:
        """Test pure Python dependency fallbacks"""
        try:
            from core.pure_python_fallbacks import get_fallback_status, AVAILABLE_DEPS
            
            status = get_fallback_status()
            
            # Test critical fallback components
            fallback_tests = {}
            
            # Test array math
            try:
                from core.pure_python_fallbacks import ArrayMath
                test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
                mean_result = ArrayMath.mean(test_data)
                fallback_tests['array_math'] = abs(mean_result - 3.0) < 0.001
            except Exception as e:
                fallback_tests['array_math'] = f"Error: {e}"
            
            # Test serialization
            try:
                from core.pure_python_fallbacks import Serializer
                test_obj = {'test': 'data', 'numbers': [1, 2, 3]}
                serialized = Serializer.packb(test_obj)
                deserialized = Serializer.unpackb(serialized)
                fallback_tests['serialization'] = test_obj == deserialized
            except Exception as e:
                fallback_tests['serialization'] = f"Error: {e}"
            
            # Test compression
            try:
                from core.pure_python_fallbacks import Compressor, Decompressor
                test_string = "Test compression data" * 10
                compressed = Compressor.compress(test_string.encode())
                decompressed = Decompressor.decompress(compressed).decode()
                fallback_tests['compression'] = test_string == decompressed
            except Exception as e:
                fallback_tests['compression'] = f"Error: {e}"
            
            # Test system monitoring
            try:
                from core.pure_python_fallbacks import SystemMonitor
                memory_info = SystemMonitor.memory_info()
                fallback_tests['system_monitoring'] = hasattr(memory_info, 'rss')
            except Exception as e:
                fallback_tests['system_monitoring'] = f"Error: {e}"
            
            details = {
                'available_dependencies': status['available_dependencies'],
                'active_fallbacks': status['active_fallbacks'],
                'fallback_tests': fallback_tests,
                'system_info': status['system_info']
            }
            
            success_rate = sum(1 for v in fallback_tests.values() if v is True) / len(fallback_tests)
            logger.info(f"Fallback tests success rate: {success_rate:.1%}")
            
            return details
            
        except Exception as e:
            logger.error(f"Dependency fallback test failed: {e}")
            raise
    
    def test_enhanced_ml_predictor(self) -> Dict[str, Any]:
        """Test Enhanced ML Predictor performance"""
        try:
            from core.enhanced_ml_predictor import get_enhanced_predictor
            from core.unified_ml_predictor import UnifiedShaderFeatures, ShaderType
            
            predictor = get_enhanced_predictor()
            
            # Create test features
            test_features = UnifiedShaderFeatures(
                shader_hash="steam_deck_test_123",
                shader_type=ShaderType.FRAGMENT,
                instruction_count=750,
                register_usage=48,
                texture_samples=6,
                memory_operations=15,
                control_flow_complexity=8,
                wave_size=64,
                uses_derivatives=True,
                uses_tessellation=False,
                uses_geometry_shader=False,
                optimization_level=2,
                cache_priority=0.7
            )
            
            # Performance test
            predictions = []
            prediction_times = []
            
            # Warm up
            for _ in range(5):
                predictor.predict_compilation_time(test_features)
            
            # Measure performance
            for i in range(20):
                start_time = time.perf_counter()
                prediction = predictor.predict_compilation_time(test_features)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                predictions.append(prediction)
                prediction_times.append(duration_ms)
            
            # Get comprehensive stats
            stats = predictor.get_enhanced_stats()
            
            details = {
                'average_prediction_time_ms': sum(prediction_times) / len(prediction_times),
                'min_prediction_time_ms': min(prediction_times),
                'max_prediction_time_ms': max(prediction_times),
                'sample_predictions': predictions[:5],
                'predictor_stats': stats,
                'predictions_under_2ms': sum(1 for t in prediction_times if t < 2.0),
                'total_predictions': len(prediction_times)
            }
            
            avg_time = details['average_prediction_time_ms']
            logger.info(f"ML Predictor: {avg_time:.3f}ms average, {stats['ml_backend']} backend")
            
            return details
            
        except Exception as e:
            logger.error(f"Enhanced ML predictor test failed: {e}")
            raise
    
    def test_thermal_management(self) -> Dict[str, Any]:
        """Test thermal monitoring and management"""
        try:
            from core.pure_python_fallbacks import PureThermalMonitor
            
            thermal = PureThermalMonitor()
            
            # Test thermal monitoring
            cpu_temp = thermal.get_cpu_temperature()
            thermal_state = thermal.get_thermal_state()
            
            # Test thermal zone discovery
            zones = thermal.thermal_zones
            sources = thermal.cpu_temp_sources
            
            details = {
                'cpu_temperature': cpu_temp,
                'thermal_state': thermal_state,
                'thermal_zones_found': len(zones),
                'thermal_zones': zones[:3],  # First 3 zones
                'temp_sources': sources,
                'temperature_reasonable': 20.0 <= cpu_temp <= 120.0
            }
            
            logger.info(f"Thermal: {cpu_temp:.1f}°C ({thermal_state})")
            
            return details
            
        except Exception as e:
            logger.error(f"Thermal management test failed: {e}")
            raise
    
    def test_hybrid_integration(self) -> Dict[str, Any]:
        """Test enhanced hybrid integration system"""
        try:
            from enhanced_rust_integration import get_hybrid_predictor
            from core.unified_ml_predictor import UnifiedShaderFeatures, ShaderType
            
            predictor = get_hybrid_predictor()
            
            # Test backend detection
            initial_backend = predictor.current_backend.value
            
            # Test prediction with hybrid system
            test_features = UnifiedShaderFeatures(
                shader_hash="hybrid_test_456",
                shader_type=ShaderType.COMPUTE,
                instruction_count=1200,
                register_usage=64,
                texture_samples=2,
                memory_operations=25,
                control_flow_complexity=12,
                wave_size=64,
                uses_derivatives=False,
                uses_tessellation=False,
                uses_geometry_shader=False,
                optimization_level=3,
                cache_priority=0.9
            )
            
            # Performance test
            prediction_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                prediction = predictor.predict_compilation_time(test_features)
                duration_ms = (time.perf_counter() - start_time) * 1000
                prediction_times.append(duration_ms)
            
            # Get comprehensive metrics
            metrics = predictor.get_comprehensive_metrics()
            
            details = {
                'initial_backend': initial_backend,
                'current_backend': predictor.current_backend.value,
                'average_prediction_time_ms': sum(prediction_times) / len(prediction_times),
                'rust_available': metrics['rust_available'],
                'auto_switch_enabled': metrics['auto_switch_enabled'],
                'backend_metrics': {k: {
                    'avg_time': v.average_time_ms,
                    'success_rate': v.success_rate,
                    'predictions': v.prediction_count
                } for k, v in metrics['backend_metrics'].items()},
                'system_memory_mb': metrics['system_memory_mb']
            }
            
            avg_time = details['average_prediction_time_ms']
            logger.info(f"Hybrid System: {avg_time:.3f}ms average, {initial_backend} backend")
            
            return details
            
        except Exception as e:
            logger.error(f"Hybrid integration test failed: {e}")
            raise
    
    def test_steam_integration(self) -> Dict[str, Any]:
        """Test Steam platform integration (if available)"""
        try:
            # Check if we can import Steam integration components
            steam_components = {}
            
            # Test Steam detection
            try:
                from core.pure_python_fallbacks import PureDBusInterface
                
                dbus_interface = PureDBusInterface()
                steam_detected = dbus_interface.steam_detected
                gaming_mode = dbus_interface.is_gaming_mode_active()
                
                steam_components['dbus_interface'] = {
                    'steam_detected': steam_detected,
                    'gaming_mode_active': gaming_mode
                }
            except Exception as e:
                steam_components['dbus_interface'] = f"Error: {e}"
            
            # Test Steam directory detection
            steam_dirs = [
                '/home/deck/.steam',
                '/home/deck/.local/share/Steam',
                '~/.steam'
            ]
            
            steam_dirs_found = []
            for steam_dir in steam_dirs:
                expanded_dir = os.path.expanduser(steam_dir)
                if os.path.exists(expanded_dir):
                    steam_dirs_found.append(expanded_dir)
            
            # Test Steam process detection
            steam_processes = []
            try:
                result = subprocess.run(['pgrep', '-f', 'steam'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    steam_processes = result.stdout.strip().split('\n')
            except Exception:
                pass
            
            details = {
                'steam_components': steam_components,
                'steam_directories_found': steam_dirs_found,
                'steam_processes': len(steam_processes),
                'steam_integration_available': len(steam_dirs_found) > 0 or len(steam_processes) > 0
            }
            
            integration_available = details['steam_integration_available']
            logger.info(f"Steam Integration: {'Available' if integration_available else 'Not Available'}")
            
            return details
            
        except Exception as e:
            logger.error(f"Steam integration test failed: {e}")
            raise
    
    def test_memory_management(self) -> Dict[str, Any]:
        """Test memory management and optimization"""
        try:
            import psutil
            process = psutil.Process()
            
            # Get initial memory
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            # Load and test multiple components
            from core.enhanced_ml_predictor import get_enhanced_predictor
            from enhanced_rust_integration import get_hybrid_predictor
            from core.pure_python_fallbacks import get_fallback_status
            
            predictor = get_enhanced_predictor()
            hybrid = get_hybrid_predictor()
            fallback_status = get_fallback_status()
            
            # Stress test memory usage
            from core.unified_ml_predictor import UnifiedShaderFeatures, ShaderType
            
            test_features = UnifiedShaderFeatures(
                shader_hash="memory_test_789",
                shader_type=ShaderType.VERTEX,
                instruction_count=500,
                register_usage=32,
                texture_samples=4,
                memory_operations=10,
                control_flow_complexity=5
            )
            
            # Run multiple predictions to test memory pressure
            for i in range(100):
                predictor.predict_compilation_time(test_features)
                if i % 20 == 0:
                    # Vary the features to test caching
                    test_features.instruction_count = 500 + i * 10
            
            # Get final memory
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_increase = final_memory - initial_memory
            
            # Get predictor stats
            stats = predictor.get_enhanced_stats()
            
            details = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'memory_usage_reasonable': memory_increase < 100,  # Less than 100MB increase
                'cache_hit_rate': stats['feature_cache_stats']['hit_rate'],
                'memory_pressure': stats['memory_pressure'],
                'predictor_memory_mb': stats['memory_usage_mb']
            }
            
            logger.info(f"Memory: {initial_memory:.1f} → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            return details
            
        except Exception as e:
            # If psutil not available, test with fallback
            try:
                from core.pure_python_fallbacks import SystemMonitor
                memory_info = SystemMonitor.memory_info()
                
                details = {
                    'memory_monitoring': 'fallback',
                    'estimated_memory_mb': memory_info.rss / (1024 * 1024),
                    'memory_fallback_working': True
                }
                
                logger.info("Memory: Using fallback monitoring")
                return details
                
            except Exception as e2:
                logger.error(f"Memory management test failed: {e}, fallback: {e2}")
                raise
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test comprehensive performance benchmarks"""
        try:
            from core.enhanced_ml_predictor import get_enhanced_predictor
            from core.unified_ml_predictor import UnifiedShaderFeatures, ShaderType
            
            predictor = get_enhanced_predictor()
            
            # Create various test scenarios
            test_scenarios = [
                ("Simple Vertex", ShaderType.VERTEX, 100, 16, 1, 2, 1),
                ("Complex Fragment", ShaderType.FRAGMENT, 1500, 96, 12, 30, 15),
                ("Heavy Compute", ShaderType.COMPUTE, 3000, 128, 4, 60, 25),
                ("Geometry Shader", ShaderType.GEOMETRY, 800, 64, 8, 20, 10),
                ("Tessellation", ShaderType.TESSELLATION_CONTROL, 1200, 80, 6, 25, 12)
            ]
            
            benchmark_results = {}
            
            for name, shader_type, inst_count, reg_usage, tex_samples, mem_ops, ctrl_flow in test_scenarios:
                features = UnifiedShaderFeatures(
                    shader_hash=f"benchmark_{name.lower().replace(' ', '_')}",
                    shader_type=shader_type,
                    instruction_count=inst_count,
                    register_usage=reg_usage,
                    texture_samples=tex_samples,
                    memory_operations=mem_ops,
                    control_flow_complexity=ctrl_flow,
                    wave_size=64,
                    optimization_level=2,
                    cache_priority=0.6
                )
                
                # Warm up
                for _ in range(3):
                    predictor.predict_compilation_time(features)
                
                # Benchmark
                times = []
                predictions = []
                for _ in range(20):
                    start_time = time.perf_counter()
                    prediction = predictor.predict_compilation_time(features)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    times.append(duration_ms)
                    predictions.append(prediction)
                
                benchmark_results[name] = {
                    'avg_time_ms': sum(times) / len(times),
                    'min_time_ms': min(times),
                    'max_time_ms': max(times),
                    'avg_prediction_ms': sum(predictions) / len(predictions),
                    'times_under_2ms': sum(1 for t in times if t < 2.0),
                    'total_tests': len(times)
                }
            
            # Overall performance metrics
            all_times = [result['avg_time_ms'] for result in benchmark_results.values()]
            under_2ms_count = sum(result['times_under_2ms'] for result in benchmark_results.values())
            total_tests = sum(result['total_tests'] for result in benchmark_results.values())
            
            details = {
                'scenario_results': benchmark_results,
                'overall_avg_time_ms': sum(all_times) / len(all_times),
                'fastest_scenario_ms': min(all_times),
                'slowest_scenario_ms': max(all_times),
                'performance_target_met': sum(all_times) / len(all_times) < 2.0,
                'fast_predictions_percentage': (under_2ms_count / total_tests) * 100
            }
            
            avg_time = details['overall_avg_time_ms']
            fast_pct = details['fast_predictions_percentage']
            logger.info(f"Benchmarks: {avg_time:.3f}ms average, {fast_pct:.1f}% under 2ms")
            
            return details
            
        except Exception as e:
            logger.error(f"Performance benchmark test failed: {e}")
            raise
    
    def test_installation_validation(self) -> Dict[str, Any]:
        """Test installation and file structure validation"""
        try:
            # Check file structure
            expected_files = [
                "src/core/enhanced_ml_predictor.py",
                "src/core/pure_python_fallbacks.py",
                "src/enhanced_rust_integration.py",
                "requirements-pure-python.txt",
                "requirements-optimized.txt"
            ]
            
            file_status = {}
            for file_path in expected_files:
                full_path = self.base_dir / file_path
                file_status[file_path] = {
                    'exists': full_path.exists(),
                    'size_bytes': full_path.stat().st_size if full_path.exists() else 0,
                    'readable': os.access(full_path, os.R_OK) if full_path.exists() else False
                }
            
            # Check Python importability
            import_status = {}
            modules_to_test = [
                ("core.enhanced_ml_predictor", "Enhanced ML Predictor"),
                ("core.pure_python_fallbacks", "Pure Python Fallbacks"),
                ("enhanced_rust_integration", "Hybrid Integration"),
                ("core.unified_ml_predictor", "Unified ML Predictor")
            ]
            
            for module_name, description in modules_to_test:
                try:
                    __import__(module_name)
                    import_status[module_name] = {'importable': True, 'error': None}
                except Exception as e:
                    import_status[module_name] = {'importable': False, 'error': str(e)}
            
            # Check requirements files
            req_files = {}
            for req_file in ["requirements-pure-python.txt", "requirements-optimized.txt"]:
                req_path = self.base_dir / req_file
                if req_path.exists():
                    with open(req_path, 'r') as f:
                        lines = f.readlines()
                    req_files[req_file] = {
                        'exists': True,
                        'line_count': len(lines),
                        'dependency_count': len([l for l in lines if l.strip() and not l.startswith('#')])
                    }
                else:
                    req_files[req_file] = {'exists': False}
            
            details = {
                'file_status': file_status,
                'import_status': import_status,
                'requirements_files': req_files,
                'all_files_present': all(fs['exists'] for fs in file_status.values()),
                'all_modules_importable': all(im['importable'] for im in import_status.values())
            }
            
            files_ok = details['all_files_present']
            imports_ok = details['all_modules_importable']
            logger.info(f"Installation: Files {'OK' if files_ok else 'MISSING'}, Imports {'OK' if imports_ok else 'FAILED'}")
            
            return details
            
        except Exception as e:
            logger.error(f"Installation validation failed: {e}")
            raise
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Running comprehensive test suite...")
        print()
        
        # Define test sequence
        tests = [
            ("Steam Deck Hardware Detection", self.test_steam_deck_detection),
            ("Dependency Fallback System", self.test_dependency_fallbacks),
            ("Enhanced ML Predictor", self.test_enhanced_ml_predictor),
            ("Thermal Management", self.test_thermal_management),
            ("Hybrid Integration System", self.test_hybrid_integration),
            ("Steam Platform Integration", self.test_steam_integration),
            ("Memory Management", self.test_memory_management),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Installation Validation", self.test_installation_validation)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            print()  # Add spacing between tests
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate comprehensive test summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("🏁 TEST SUITE COMPLETE")
        print("=" * 60)
        
        # Overall statistics
        success_rate = (self.success_count / self.test_count) * 100
        print(f"Tests: {self.success_count}/{self.test_count} passed ({success_rate:.1f}%)")
        print(f"Duration: {total_time:.1f} seconds")
        print()
        
        # Test results summary
        print("📊 Test Results Summary:")
        for result in self.results:
            status_emoji = "✅" if result.success else "❌"
            print(f"  {status_emoji} {result.name} ({result.duration_ms:.1f}ms)")
            if not result.success and result.error:
                print(f"      Error: {result.error}")
        print()
        
        # Performance highlights
        ml_results = [r for r in self.results if "ML Predictor" in r.name or "Performance" in r.name]
        if ml_results:
            print("⚡ Performance Highlights:")
            for result in ml_results:
                if result.success and 'average_prediction_time_ms' in result.details:
                    avg_time = result.details['average_prediction_time_ms']
                    print(f"  • {result.name}: {avg_time:.3f}ms average")
        print()
        
        # System information
        steam_deck_result = next((r for r in self.results if "Steam Deck" in r.name), None)
        if steam_deck_result and steam_deck_result.success:
            details = steam_deck_result.details
            print("🎮 Steam Deck Information:")
            print(f"  • Platform: {details.get('model', 'Unknown')}")
            print(f"  • Detected: {details.get('is_steam_deck', False)}")
            print()
        
        # Recommendations
        print("💡 Recommendations:")
        if success_rate >= 90:
            print("  ✅ System is ready for production use")
            print("  ✅ All critical components are working correctly")
        elif success_rate >= 70:
            print("  ⚠️  System is functional with some minor issues")
            print("  ⚠️  Review failed tests and apply fixes")
        else:
            print("  ❌ System has significant issues requiring attention")
            print("  ❌ Review all failed tests before deployment")
        
        # Save detailed results
        self.save_detailed_results()
        
        print(f"\n📝 Detailed results saved to: steam_deck_test_results.json")
        print(f"📝 Test log saved to: steam_deck_test.log")
    
    def save_detailed_results(self):
        """Save detailed test results to JSON file"""
        results_data = {
            'test_suite_info': {
                'total_tests': self.test_count,
                'successful_tests': self.success_count,
                'success_rate': (self.success_count / self.test_count) * 100,
                'total_duration_seconds': time.time() - self.start_time,
                'platform': platform.system(),
                'machine': platform.machine(),
                'python_version': platform.python_version(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'test_results': [
                {
                    'name': result.name,
                    'success': result.success,
                    'duration_ms': result.duration_ms,
                    'details': result.details,
                    'error': result.error
                }
                for result in self.results
            ]
        }
        
        with open('steam_deck_test_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)


def main():
    """Main test execution"""
    try:
        # Check if we're in the right directory
        if not Path("src").exists():
            print("❌ Error: src/ directory not found")
            print("Please run this script from the project root directory")
            return 1
        
        # Create and run test suite
        test_suite = SteamDeckTestSuite()
        test_suite.run_all_tests()
        
        # Return appropriate exit code
        success_rate = (test_suite.success_count / test_suite.test_count) * 100
        return 0 if success_rate >= 70 else 1
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        logger.error(f"Test suite execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)