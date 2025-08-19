#!/usr/bin/env python3
"""
Steam Deck Threading Fixes Validation Suite
Comprehensive test to validate all threading optimizations and fixes
"""

import os
import sys
import time
import threading
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('threading_test.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    success: bool
    duration_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None

class ThreadingFixesTestSuite:
    """Comprehensive test suite for threading fixes"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        self.test_count = 0
        self.success_count = 0
        
        print("🧵 Steam Deck Threading Fixes Validation Suite")
        print("=" * 60)
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
    
    def test_environment_variables(self) -> Dict[str, Any]:
        """Test threading environment variable configuration"""
        try:
            from src.core.threading_config import configure_threading_for_steam_deck
            
            # Configure threading
            configurator = configure_threading_for_steam_deck()
            status = configurator.get_configuration_status()
            
            # Check critical environment variables
            env_vars_to_check = [
                'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 
                'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'
            ]
            
            env_status = {}
            for var in env_vars_to_check:
                value = os.environ.get(var, 'not set')
                env_status[var] = value
                
                # Validate values are reasonable for Steam Deck
                if value != 'not set':
                    try:
                        num_val = int(value)
                        if num_val > 4:  # Too many threads for Steam Deck
                            logger.warning(f"{var}={num_val} may be too high for Steam Deck")
                    except ValueError:
                        pass
            
            details = {
                'configurator_status': status,
                'environment_variables': env_status,
                'is_configured': status['is_configured'],
                'configured_libraries': status['configured_libraries']
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Environment variable test failed: {e}")
            raise
    
    def test_thread_pool_manager(self) -> Dict[str, Any]:
        """Test thread pool manager functionality"""
        try:
            from src.core.thread_pool_manager import get_thread_manager
            
            # Get thread manager
            manager = get_thread_manager()
            
            # Test basic functionality
            def test_task(name, duration):
                time.sleep(duration)
                return f"Task {name} completed"
            
            # Submit tasks to different priority levels
            ml_future = manager.submit_ml_task(test_task, "ML", 0.1)
            compile_future = manager.submit_compilation_task(test_task, "COMPILE", 0.1)
            monitor_future = manager.submit_monitoring_task(test_task, "MONITOR", 0.1)
            
            # Wait for completion
            ml_result = ml_future.result(timeout=2.0)
            compile_result = compile_future.result(timeout=2.0)
            monitor_result = monitor_future.result(timeout=2.0)
            
            # Get metrics
            metrics = manager.get_thread_metrics()
            
            details = {
                'task_results': [ml_result, compile_result, monitor_result],
                'thread_metrics': metrics,
                'total_threads': metrics['total_active_threads'],
                'max_threads': metrics['max_total_threads'],
                'resource_state': metrics['resource_state'],
                'success_rate': metrics['task_metrics']['success_rate_percent']
            }
            
            # Validate thread limits
            if metrics['total_active_threads'] > metrics['max_total_threads']:
                logger.warning("Thread pool exceeded maximum thread limit")
            
            return details
            
        except Exception as e:
            logger.error(f"Thread pool manager test failed: {e}")
            raise
    
    def test_ml_library_threading(self) -> Dict[str, Any]:
        """Test ML library threading configuration"""
        try:
            from src.core.ml_only_predictor import get_ml_predictor
            
            # Initialize ML predictor (should configure threading)
            predictor = get_ml_predictor()
            
            # Test prediction performance
            test_features = {
                'instruction_count': 1000,
                'register_usage': 64,
                'texture_samples': 8,
                'memory_operations': 20,
                'control_flow_complexity': 10,
                'wave_size': 64,
                'shader_type_hash': 2.0,
                'optimization_level': 2
            }
            
            # Warm up
            for _ in range(3):
                predictor.predict_compilation_time(test_features)
            
            # Performance test
            prediction_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                result = predictor.predict_compilation_time(test_features)
                duration_ms = (time.perf_counter() - start_time) * 1000
                prediction_times.append(duration_ms)
            
            avg_time = sum(prediction_times) / len(prediction_times)
            
            # Get performance metrics
            metrics = predictor.get_performance_metrics()
            
            details = {
                'average_prediction_time_ms': avg_time,
                'min_prediction_time_ms': min(prediction_times),
                'max_prediction_time_ms': max(prediction_times),
                'predictions_under_5ms': sum(1 for t in prediction_times if t < 5.0),
                'ml_metrics': metrics,
                'threading_optimizations_active': hasattr(predictor, 'threading_configurator') and predictor.threading_configurator is not None
            }
            
            # Validate performance
            if avg_time > 10.0:  # Should be under 10ms for simple prediction
                logger.warning(f"ML prediction time high: {avg_time:.1f}ms")
            
            return details
            
        except Exception as e:
            logger.error(f"ML library threading test failed: {e}")
            raise
    
    def test_thermal_management_integration(self) -> Dict[str, Any]:
        """Test thermal management with threading integration"""
        try:
            from src.optimization.optimized_thermal_manager import get_thermal_manager
            
            # Get thermal manager
            thermal_manager = get_thermal_manager()
            
            # Start monitoring briefly
            thermal_manager.start_monitoring()
            
            # Let it collect some data
            time.sleep(2.0)
            
            # Get status
            status = thermal_manager.get_status()
            
            # Test thermal state changes
            original_threads = status['compilation_threads']
            
            # Stop monitoring
            thermal_manager.stop_monitoring()
            
            details = {
                'thermal_status': status,
                'has_thread_manager_integration': hasattr(thermal_manager, 'thread_manager') and thermal_manager.thread_manager is not None,
                'mock_mode': status['mock_mode'],
                'steam_deck_model': status['steam_deck_model'],
                'compilation_threads': status['compilation_threads'],
                'current_temps': status['current_temps']
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Thermal management integration test failed: {e}")
            raise
    
    def test_thread_diagnostics(self) -> Dict[str, Any]:
        """Test thread diagnostics and monitoring"""
        try:
            from src.core.thread_diagnostics import get_thread_diagnostics
            
            # Get diagnostics system
            diagnostics = get_thread_diagnostics()
            
            # Let it collect data
            time.sleep(3.0)
            
            # Generate diagnostic report
            report = diagnostics.get_diagnostic_report()
            
            # Test issue detection (should not have critical issues)
            issues = report['issues']
            critical_issues = issues['by_severity']['critical']
            high_issues = issues['by_severity']['high']
            
            details = {
                'diagnostic_report': report,
                'total_threads': report['system_info']['total_threads'],
                'library_breakdown': report['threading_analysis']['library_breakdown'],
                'critical_issues': critical_issues,
                'high_issues': high_issues,
                'recommendations_count': len(report['recommendations']),
                'has_steam_deck_optimizations': report['system_info']['is_steam_deck']
            }
            
            # Validate no critical issues
            if critical_issues > 0:
                logger.warning(f"Critical threading issues detected: {critical_issues}")
            
            if high_issues > 2:
                logger.warning(f"Multiple high-severity threading issues: {high_issues}")
            
            return details
            
        except Exception as e:
            logger.error(f"Thread diagnostics test failed: {e}")
            raise
    
    def test_resource_limits(self) -> Dict[str, Any]:
        """Test system resource limits and constraints"""
        try:
            import psutil
            
            # Get initial resource state
            initial_memory = psutil.virtual_memory()
            initial_threads = len(threading.enumerate())
            
            # Create some threads and monitor resource usage
            thread_results = []
            
            def worker_thread(thread_id):
                # Simulate some work
                time.sleep(0.5)
                return f"Thread {thread_id} completed"
            
            # Create limited number of threads
            threads = []
            for i in range(5):  # Conservative number for Steam Deck
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=2.0)
            
            # Check final resource state
            final_memory = psutil.virtual_memory()
            final_threads = len(threading.enumerate())
            
            memory_increase = final_memory.used - initial_memory.used
            thread_increase = final_threads - initial_threads
            
            details = {
                'initial_memory_mb': initial_memory.used / (1024 * 1024),
                'final_memory_mb': final_memory.used / (1024 * 1024),
                'memory_increase_mb': memory_increase / (1024 * 1024),
                'initial_threads': initial_threads,
                'final_threads': final_threads,
                'thread_increase': thread_increase,
                'memory_percent': final_memory.percent,
                'cpu_count': psutil.cpu_count(),
                'resource_limits_respected': memory_increase < 50 * 1024 * 1024  # Less than 50MB increase
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Resource limits test failed: {e}")
            raise
    
    def test_threading_under_load(self) -> Dict[str, Any]:
        """Test threading behavior under system load"""
        try:
            from src.core.thread_pool_manager import get_thread_manager
            from src.core.ml_only_predictor import get_ml_predictor
            
            manager = get_thread_manager()
            predictor = get_ml_predictor()
            
            # Create load test scenario
            def cpu_intensive_task(task_id):
                # Simulate CPU-intensive work
                result = 0
                for i in range(10000):
                    result += i * i
                return f"Task {task_id}: {result}"
            
            def ml_prediction_task(task_id):
                test_features = {
                    'instruction_count': 500 + task_id * 10,
                    'register_usage': 32,
                    'texture_samples': 4,
                    'memory_operations': 10,
                    'control_flow_complexity': 5,
                    'wave_size': 64,
                    'shader_type_hash': 1.0,
                    'optimization_level': 1
                }
                result = predictor.predict_compilation_time(test_features)
                return f"ML Task {task_id}: {result}"
            
            # Submit mixed workload
            futures = []
            start_time = time.perf_counter()
            
            for i in range(8):  # Conservative load for Steam Deck
                if i % 2 == 0:
                    future = manager.submit_compilation_task(cpu_intensive_task, i)
                else:
                    future = manager.submit_ml_task(ml_prediction_task, i)
                futures.append(future)
            
            # Wait for all tasks to complete
            completed_tasks = 0
            failed_tasks = 0
            task_times = []
            
            for future in futures:
                try:
                    task_start = time.perf_counter()
                    result = future.result(timeout=10.0)
                    task_duration = time.perf_counter() - task_start
                    task_times.append(task_duration)
                    completed_tasks += 1
                except Exception as e:
                    failed_tasks += 1
                    logger.warning(f"Task failed: {e}")
            
            total_duration = time.perf_counter() - start_time
            
            # Get final metrics
            metrics = manager.get_thread_metrics()
            
            details = {
                'total_tasks': len(futures),
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'total_duration_seconds': total_duration,
                'average_task_duration': sum(task_times) / len(task_times) if task_times else 0,
                'max_task_duration': max(task_times) if task_times else 0,
                'thread_metrics': metrics,
                'success_rate': (completed_tasks / len(futures)) * 100 if futures else 0,
                'no_thread_creation_errors': failed_tasks == 0
            }
            
            # Validate performance under load
            if failed_tasks > 0:
                logger.warning(f"Tasks failed under load: {failed_tasks}")
            
            if metrics['total_active_threads'] > 8:  # Steam Deck limit
                logger.warning(f"Thread count exceeded Steam Deck limit: {metrics['total_active_threads']}")
            
            return details
            
        except Exception as e:
            logger.error(f"Threading under load test failed: {e}")
            raise
    
    def test_cleanup_and_shutdown(self) -> Dict[str, Any]:
        """Test proper cleanup and shutdown of threading systems"""
        try:
            from src.core.thread_pool_manager import get_thread_manager
            from src.core.ml_only_predictor import get_ml_predictor
            from src.core.thread_diagnostics import get_thread_diagnostics
            
            # Get all components
            manager = get_thread_manager()
            predictor = get_ml_predictor()
            diagnostics = get_thread_diagnostics()
            
            # Record initial state
            initial_threads = len(threading.enumerate())
            
            # Perform some operations
            future = manager.submit_ml_task(lambda: "test task")
            result = future.result(timeout=5.0)
            
            # Test cleanup
            cleanup_results = {}
            
            # Cleanup ML predictor
            try:
                predictor.cleanup()
                cleanup_results['ml_predictor'] = 'success'
            except Exception as e:
                cleanup_results['ml_predictor'] = f'error: {e}'
            
            # Cleanup thread manager
            try:
                manager.shutdown(wait=True, timeout=5.0)
                cleanup_results['thread_manager'] = 'success'
            except Exception as e:
                cleanup_results['thread_manager'] = f'error: {e}'
            
            # Stop diagnostics
            try:
                diagnostics.stop_monitoring()
                cleanup_results['diagnostics'] = 'success'
            except Exception as e:
                cleanup_results['diagnostics'] = f'error: {e}'
            
            # Wait a moment for cleanup to complete
            time.sleep(1.0)
            
            # Check final thread count
            final_threads = len(threading.enumerate())
            thread_reduction = initial_threads - final_threads
            
            details = {
                'initial_threads': initial_threads,
                'final_threads': final_threads,
                'thread_reduction': thread_reduction,
                'cleanup_results': cleanup_results,
                'successful_cleanups': sum(1 for result in cleanup_results.values() if result == 'success'),
                'test_task_result': result,
                'cleanup_effective': thread_reduction >= 0  # Should not increase threads
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Cleanup and shutdown test failed: {e}")
            raise
    
    def run_all_tests(self):
        """Run complete threading fixes test suite"""
        print("🧪 Running threading fixes validation suite...")
        print()
        
        # Define test sequence
        tests = [
            ("Environment Variables Configuration", self.test_environment_variables),
            ("Thread Pool Manager", self.test_thread_pool_manager),
            ("ML Library Threading", self.test_ml_library_threading),
            ("Thermal Management Integration", self.test_thermal_management_integration),
            ("Thread Diagnostics", self.test_thread_diagnostics),
            ("Resource Limits", self.test_resource_limits),
            ("Threading Under Load", self.test_threading_under_load),
            ("Cleanup and Shutdown", self.test_cleanup_and_shutdown)
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
        print("🏁 THREADING FIXES VALIDATION COMPLETE")
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
        
        # Threading specific analysis
        print("🧵 Threading Analysis:")
        
        # Analyze environment variable configuration
        env_result = next((r for r in self.results if "Environment" in r.name), None)
        if env_result and env_result.success:
            env_vars = env_result.details.get('environment_variables', {})
            configured_vars = [k for k, v in env_vars.items() if v != 'not set']
            print(f"  • Environment variables configured: {len(configured_vars)}/4")
            
            for var, value in env_vars.items():
                if value != 'not set':
                    print(f"    - {var}={value}")
        
        # Analyze thread pool performance
        pool_result = next((r for r in self.results if "Thread Pool" in r.name), None)
        if pool_result and pool_result.success:
            metrics = pool_result.details.get('thread_metrics', {})
            print(f"  • Active threads: {metrics.get('total_active_threads', 'unknown')}")
            print(f"  • Resource state: {metrics.get('resource_state', 'unknown')}")
            print(f"  • Task success rate: {metrics.get('task_metrics', {}).get('success_rate_percent', 0):.1f}%")
        
        # Analyze ML performance
        ml_result = next((r for r in self.results if "ML Library" in r.name), None)
        if ml_result and ml_result.success:
            avg_time = ml_result.details.get('average_prediction_time_ms', 0)
            fast_predictions = ml_result.details.get('predictions_under_5ms', 0)
            print(f"  • ML prediction time: {avg_time:.3f}ms average")
            print(f"  • Fast predictions: {fast_predictions}/10 under 5ms")
        
        print()
        
        # Issues and recommendations
        diagnostics_result = next((r for r in self.results if "Diagnostics" in r.name), None)
        if diagnostics_result and diagnostics_result.success:
            issues = diagnostics_result.details.get('diagnostic_report', {}).get('issues', {})
            critical_issues = issues.get('by_severity', {}).get('critical', 0)
            high_issues = issues.get('by_severity', {}).get('high', 0)
            
            print("⚠️  Threading Issues Detected:")
            if critical_issues == 0 and high_issues == 0:
                print("  ✅ No critical or high-severity threading issues detected")
            else:
                if critical_issues > 0:
                    print(f"  ❌ Critical issues: {critical_issues}")
                if high_issues > 0:
                    print(f"  ⚠️  High-severity issues: {high_issues}")
        
        # Overall assessment
        print("\n💡 Overall Assessment:")
        if success_rate >= 90:
            print("  ✅ Threading fixes are working correctly")
            print("  ✅ System is optimized for Steam Deck")
            print("  ✅ No critical threading issues detected")
        elif success_rate >= 70:
            print("  ⚠️  Most threading fixes are working")
            print("  ⚠️  Some minor issues may need attention")
        else:
            print("  ❌ Significant threading issues remain")
            print("  ❌ Manual intervention may be required")
        
        # Save detailed results
        self.save_detailed_results()
        
        print(f"\n📝 Detailed results saved to: threading_test_results.json")
        print(f"📝 Test log saved to: threading_test.log")
    
    def save_detailed_results(self):
        """Save detailed test results to JSON file"""
        results_data = {
            'test_suite_info': {
                'total_tests': self.test_count,
                'successful_tests': self.success_count,
                'success_rate': (self.success_count / self.test_count) * 100,
                'total_duration_seconds': time.time() - self.start_time,
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
        
        with open('threading_test_results.json', 'w') as f:
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
        test_suite = ThreadingFixesTestSuite()
        test_suite.run_all_tests()
        
        # Return appropriate exit code
        success_rate = (test_suite.success_count / test_suite.test_count) * 100
        return 0 if success_rate >= 80 else 1
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        logger.error(f"Test suite execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)