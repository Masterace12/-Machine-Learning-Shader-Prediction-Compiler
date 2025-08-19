#!/usr/bin/env python3
"""
Steam Deck Performance Validation
Comprehensive test of ML Shader Prediction Compiler performance on Steam Deck hardware
"""

import os
import sys
import time
import json
import logging
import numpy as np
import statistics
from typing import Dict, List, Any, Tuple
from pathlib import Path
import gc
import psutil
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import Steam Deck optimizations
try:
    from src.core.steamdeck_integration import get_integration_system
    from src.core.steamdeck_thermal_optimizer import get_thermal_optimizer
    from src.core.steamdeck_gaming_detector import get_gaming_detector
    from src.core.steamdeck_cache_optimizer import get_cache_optimizer
    from src.core.ml_only_predictor import get_ml_predictor
    HAS_STEAMDECK_OPTIMIZATIONS = True
except ImportError as e:
    print(f"⚠️  Steam Deck optimizations not available: {e}")
    HAS_STEAMDECK_OPTIMIZATIONS = False

class SteamDeckPerformanceValidator:
    """Comprehensive performance validation for Steam Deck"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.results = {}
        self.test_data = self._generate_test_data()
        
        # Performance targets
        self.targets = {
            "prediction_time_ms": 0.1,         # Target < 0.1ms per prediction
            "throughput_predictions_sec": 280000,  # Target > 280k predictions/sec
            "memory_usage_mb": 200,             # Target < 200MB total
            "thread_count": 8,                  # Target < 8 threads
            "thermal_stable_temp": 85.0,        # Target < 85°C under load
            "cache_hit_rate": 0.8,              # Target > 80% cache hit rate
        }
        
        self.logger.info("Steam Deck performance validator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('steamdeck_performance_validation.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic shader data for testing"""
        np.random.seed(42)  # Reproducible results
        
        # Generate realistic shader feature data
        n_samples = 10000
        feature_data = {
            "vertex_shader_complexity": np.random.uniform(0.1, 1.0, n_samples),
            "fragment_shader_complexity": np.random.uniform(0.1, 1.0, n_samples),
            "texture_count": np.random.randint(0, 16, n_samples),
            "uniform_count": np.random.randint(0, 64, n_samples),
            "instruction_count": np.random.randint(50, 2000, n_samples),
            "register_pressure": np.random.uniform(0.0, 1.0, n_samples),
            "memory_bandwidth": np.random.uniform(0.1, 1.0, n_samples),
            "branching_factor": np.random.uniform(0.0, 0.5, n_samples),
        }
        
        # Create feature matrix
        features = np.column_stack([feature_data[key] for key in sorted(feature_data.keys())])
        
        return {
            "features": features,
            "feature_names": sorted(feature_data.keys()),
            "n_samples": n_samples
        }
    
    def test_hardware_detection(self) -> Dict[str, Any]:
        """Test Steam Deck hardware detection"""
        self.logger.info("Testing hardware detection...")
        
        results = {
            "test_name": "Hardware Detection",
            "success": True,
            "details": {},
            "errors": []
        }
        
        try:
            # Check DMI information
            dmi_path = Path("/sys/class/dmi/id/product_name")
            if dmi_path.exists():
                product_name = dmi_path.read_text().strip()
                results["details"]["product_name"] = product_name
                results["details"]["is_steam_deck"] = "galileo" in product_name.lower() or "jupiter" in product_name.lower()
            
            # Check CPU information
            cpu_info = {}
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info["model"] = line.split(":")[1].strip()
                        break
            
            cpu_info["core_count"] = os.cpu_count()
            results["details"]["cpu_info"] = cpu_info
            
            # Check memory
            memory = psutil.virtual_memory()
            results["details"]["memory_gb"] = round(memory.total / (1024**3), 1)
            
            # Check thermal sensors
            thermal_sensors = []
            for i in range(5):
                sensor_path = f"/sys/class/thermal/thermal_zone{i}/temp"
                if Path(sensor_path).exists():
                    thermal_sensors.append(sensor_path)
            
            results["details"]["thermal_sensors"] = len(thermal_sensors)
            
            self.logger.info(f"Hardware detection completed: {results['details']}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            self.logger.error(f"Hardware detection failed: {e}")
        
        return results
    
    def test_optimization_initialization(self) -> Dict[str, Any]:
        """Test Steam Deck optimization system initialization"""
        self.logger.info("Testing optimization system initialization...")
        
        results = {
            "test_name": "Optimization Initialization",
            "success": True,
            "details": {},
            "errors": []
        }
        
        if not HAS_STEAMDECK_OPTIMIZATIONS:
            results["success"] = False
            results["errors"].append("Steam Deck optimizations not available")
            return results
        
        try:
            # Test integration system
            integration_system = get_integration_system()
            init_success = integration_system.initialize()
            
            results["details"]["integration_system"] = init_success
            
            if init_success:
                # Test individual components
                thermal_optimizer = get_thermal_optimizer()
                results["details"]["thermal_optimizer"] = thermal_optimizer is not None
                
                cache_optimizer = get_cache_optimizer()
                results["details"]["cache_optimizer"] = cache_optimizer is not None
                
                gaming_detector = get_gaming_detector()
                results["details"]["gaming_detector"] = gaming_detector is not None
                
                # Get system status
                status = integration_system.get_system_status()
                results["details"]["system_status"] = {
                    "active": status.get("optimization_active", False),
                    "profile": status.get("current_profile", "unknown"),
                    "components": len([k for k, v in status.items() if isinstance(v, dict)])
                }
                
                self.logger.info("Optimization system initialized successfully")
            else:
                results["success"] = False
                results["errors"].append("Integration system initialization failed")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            self.logger.error(f"Optimization initialization failed: {e}")
        
        return results
    
    def test_ml_prediction_performance(self) -> Dict[str, Any]:
        """Test ML prediction performance"""
        self.logger.info("Testing ML prediction performance...")
        
        results = {
            "test_name": "ML Prediction Performance",
            "success": True,
            "details": {},
            "errors": []
        }
        
        try:
            # Initialize ML predictor
            ml_predictor = get_ml_predictor()
            
            if ml_predictor is None:
                results["success"] = False
                results["errors"].append("ML predictor not available")
                return results
            
            # Warm up
            warmup_features = self.test_data["features"][:10]
            for features in warmup_features:
                feature_dict = {name: float(val) for name, val in zip(self.test_data["feature_names"], features)}
                ml_predictor.predict_compilation_time(feature_dict)
            
            # Performance test
            test_features = self.test_data["features"][:1000]  # Test 1000 predictions
            prediction_times = []
            
            start_time = time.time()
            
            for features in test_features:
                feature_dict = {name: float(val) for name, val in zip(self.test_data["feature_names"], features)}
                
                pred_start = time.time()
                result = ml_predictor.predict_compilation_time(feature_dict)
                pred_end = time.time()
                
                prediction_times.append((pred_end - pred_start) * 1000)  # Convert to ms
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            avg_prediction_time = statistics.mean(prediction_times)
            min_prediction_time = min(prediction_times)
            max_prediction_time = max(prediction_times)
            p95_prediction_time = np.percentile(prediction_times, 95)
            
            predictions_per_second = len(test_features) / total_time
            
            results["details"] = {
                "total_predictions": len(test_features),
                "total_time_seconds": total_time,
                "average_prediction_time_ms": avg_prediction_time,
                "min_prediction_time_ms": min_prediction_time,
                "max_prediction_time_ms": max_prediction_time,
                "p95_prediction_time_ms": p95_prediction_time,
                "predictions_per_second": predictions_per_second,
                "target_met": predictions_per_second >= self.targets["throughput_predictions_sec"],
                "performance_ratio": predictions_per_second / self.targets["throughput_predictions_sec"]
            }
            
            # Check if targets are met
            if avg_prediction_time > self.targets["prediction_time_ms"]:
                results["errors"].append(f"Prediction time too slow: {avg_prediction_time:.3f}ms > {self.targets['prediction_time_ms']}ms")
            
            if predictions_per_second < self.targets["throughput_predictions_sec"]:
                results["errors"].append(f"Throughput too low: {predictions_per_second:.0f} < {self.targets['throughput_predictions_sec']}")
            
            results["success"] = len(results["errors"]) == 0
            
            self.logger.info(f"ML Performance: {predictions_per_second:.0f} pred/sec, {avg_prediction_time:.3f}ms avg")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            self.logger.error(f"ML prediction test failed: {e}")
        
        return results
    
    def test_thermal_management(self) -> Dict[str, Any]:
        """Test thermal management under load"""
        self.logger.info("Testing thermal management...")
        
        results = {
            "test_name": "Thermal Management",
            "success": True,
            "details": {},
            "errors": []
        }
        
        try:
            thermal_optimizer = get_thermal_optimizer()
            
            if thermal_optimizer is None:
                results["success"] = False
                results["errors"].append("Thermal optimizer not available")
                return results
            
            # Get initial thermal state
            initial_status = thermal_optimizer.get_status()
            initial_temp = initial_status.get("max_temperature", 70.0)
            
            # Run load test
            load_duration = 10  # seconds
            temperatures = []
            thermal_states = []
            
            def cpu_load_worker():
                """Worker function to create CPU load"""
                end_time = time.time() + load_duration
                while time.time() < end_time:
                    # Create some CPU load
                    np.random.random((100, 100)) @ np.random.random((100, 100))
            
            # Start load workers
            workers = []
            for _ in range(2):  # 2 workers for moderate load
                worker = threading.Thread(target=cpu_load_worker)
                worker.start()
                workers.append(worker)
            
            # Monitor thermal state during load
            monitor_start = time.time()
            while time.time() - monitor_start < load_duration:
                status = thermal_optimizer.get_status()
                temperatures.append(status.get("max_temperature", 70.0))
                thermal_states.append(status.get("thermal_state", "unknown"))
                time.sleep(0.5)
            
            # Wait for workers to complete
            for worker in workers:
                worker.join()
            
            # Get final thermal state
            final_status = thermal_optimizer.get_status()
            final_temp = final_status.get("max_temperature", 70.0)
            
            # Analyze results
            max_temp = max(temperatures) if temperatures else initial_temp
            avg_temp = statistics.mean(temperatures) if temperatures else initial_temp
            temp_increase = max_temp - initial_temp
            
            results["details"] = {
                "initial_temperature": initial_temp,
                "final_temperature": final_temp,
                "max_temperature": max_temp,
                "average_temperature": avg_temp,
                "temperature_increase": temp_increase,
                "thermal_states": list(set(thermal_states)),
                "monitoring_duration": load_duration,
                "stable_under_load": max_temp < self.targets["thermal_stable_temp"]
            }
            
            # Check thermal stability
            if max_temp >= self.targets["thermal_stable_temp"]:
                results["errors"].append(f"Temperature too high under load: {max_temp:.1f}°C >= {self.targets['thermal_stable_temp']}°C")
            
            results["success"] = len(results["errors"]) == 0
            
            self.logger.info(f"Thermal test: {initial_temp:.1f}°C -> {max_temp:.1f}°C (Δ{temp_increase:.1f}°C)")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            self.logger.error(f"Thermal management test failed: {e}")
        
        return results
    
    def test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory usage efficiency"""
        self.logger.info("Testing memory efficiency...")
        
        results = {
            "test_name": "Memory Efficiency",
            "success": True,
            "details": {},
            "errors": []
        }
        
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Run memory-intensive operations
            cache_optimizer = get_cache_optimizer()
            ml_predictor = get_ml_predictor()
            
            # Test cache operations
            for i in range(100):
                features = {name: float(val) for name, val in zip(
                    self.test_data["feature_names"], 
                    self.test_data["features"][i]
                )}
                
                # ML prediction
                result = ml_predictor.predict_compilation_time(features)
                
                # Cache storage
                cache_optimizer.put(features, result, prediction_time_ms=0.1)
                
                # Cache retrieval
                cached_result = cache_optimizer.get(features)
            
            # Force garbage collection
            gc.collect()
            
            # Get final memory usage
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            # Get cache statistics
            cache_stats = cache_optimizer.get_stats()
            
            results["details"] = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "cache_memory_mb": cache_stats.memory_usage_mb,
                "total_memory_mb": final_memory,
                "within_target": final_memory < self.targets["memory_usage_mb"],
                "cache_entries": cache_stats.total_entries,
                "cache_hit_rate": cache_stats.hit_rate
            }
            
            # Check memory usage
            if final_memory >= self.targets["memory_usage_mb"]:
                results["errors"].append(f"Memory usage too high: {final_memory:.1f}MB >= {self.targets['memory_usage_mb']}MB")
            
            results["success"] = len(results["errors"]) == 0
            
            self.logger.info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (Δ{memory_increase:.1f}MB)")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            self.logger.error(f"Memory efficiency test failed: {e}")
        
        return results
    
    def test_thread_stability(self) -> Dict[str, Any]:
        """Test thread stability and resource management"""
        self.logger.info("Testing thread stability...")
        
        results = {
            "test_name": "Thread Stability",
            "success": True,
            "details": {},
            "errors": []
        }
        
        try:
            # Get initial thread count
            initial_threads = len(threading.enumerate())
            
            # Run concurrent operations
            def worker_task(worker_id: int):
                """Worker task for concurrent testing"""
                ml_predictor = get_ml_predictor()
                cache_optimizer = get_cache_optimizer()
                
                for i in range(10):
                    features = {name: float(val) for name, val in zip(
                        self.test_data["feature_names"], 
                        self.test_data["features"][worker_id * 10 + i]
                    )}
                    
                    result = ml_predictor.predict_compilation_time(features)
                    cache_optimizer.put(features, result)
                    time.sleep(0.01)  # Small delay
            
            # Start multiple workers
            workers = []
            for i in range(4):  # 4 concurrent workers
                worker = threading.Thread(target=worker_task, args=(i,))
                workers.append(worker)
                worker.start()
            
            # Monitor thread count during execution
            max_threads = initial_threads
            for _ in range(20):  # Monitor for 2 seconds
                current_threads = len(threading.enumerate())
                max_threads = max(max_threads, current_threads)
                time.sleep(0.1)
            
            # Wait for workers to complete
            for worker in workers:
                worker.join(timeout=10)
            
            # Get final thread count
            final_threads = len(threading.enumerate())
            
            results["details"] = {
                "initial_threads": initial_threads,
                "max_threads": max_threads,
                "final_threads": final_threads,
                "thread_increase": max_threads - initial_threads,
                "thread_cleanup": max_threads - final_threads,
                "within_target": max_threads <= self.targets["thread_count"],
                "stable_threading": abs(final_threads - initial_threads) <= 2
            }
            
            # Check thread limits
            if max_threads > self.targets["thread_count"]:
                results["errors"].append(f"Too many threads: {max_threads} > {self.targets['thread_count']}")
            
            results["success"] = len(results["errors"]) == 0
            
            self.logger.info(f"Thread usage: {initial_threads} -> {max_threads} -> {final_threads}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            self.logger.error(f"Thread stability test failed: {e}")
        
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        self.logger.info("🚀 Starting comprehensive Steam Deck performance validation")
        self.logger.info("=" * 60)
        
        validation_results = {
            "timestamp": time.time(),
            "steam_deck_optimized": HAS_STEAMDECK_OPTIMIZATIONS,
            "test_results": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            },
            "performance_scores": {},
            "recommendations": []
        }
        
        # Run all tests
        test_methods = [
            self.test_hardware_detection,
            self.test_optimization_initialization,
            self.test_ml_prediction_performance,
            self.test_thermal_management,
            self.test_memory_efficiency,
            self.test_thread_stability
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                validation_results["test_results"].append(result)
                
                # Update summary
                validation_results["summary"]["total_tests"] += 1
                if result["success"]:
                    validation_results["summary"]["passed_tests"] += 1
                else:
                    validation_results["summary"]["failed_tests"] += 1
                
                # Log result
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                self.logger.info(f"{status} - {result['test_name']}")
                
                if not result["success"] and result.get("errors"):
                    for error in result["errors"]:
                        self.logger.error(f"  Error: {error}")
                
            except Exception as e:
                self.logger.error(f"Test execution failed: {test_method.__name__}: {e}")
        
        # Calculate success rate
        total_tests = validation_results["summary"]["total_tests"]
        passed_tests = validation_results["summary"]["passed_tests"]
        validation_results["summary"]["success_rate"] = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate performance scores
        self._calculate_performance_scores(validation_results)
        
        # Generate recommendations
        self._generate_recommendations(validation_results)
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info(f"🏁 VALIDATION COMPLETE")
        self.logger.info(f"Tests: {passed_tests}/{total_tests} passed ({validation_results['summary']['success_rate']:.1f}%)")
        
        for score_name, score_value in validation_results["performance_scores"].items():
            self.logger.info(f"{score_name}: {score_value:.1f}%")
        
        if validation_results["recommendations"]:
            self.logger.info("💡 Recommendations:")
            for rec in validation_results["recommendations"]:
                self.logger.info(f"  • {rec}")
        
        return validation_results
    
    def _calculate_performance_scores(self, validation_results: Dict[str, Any]):
        """Calculate performance scores based on test results"""
        scores = {}
        
        # Find ML performance test
        ml_test = next((t for t in validation_results["test_results"] if t["test_name"] == "ML Prediction Performance"), None)
        if ml_test and ml_test["success"]:
            details = ml_test["details"]
            target_throughput = self.targets["throughput_predictions_sec"]
            actual_throughput = details.get("predictions_per_second", 0)
            scores["ML Performance"] = min(100.0, (actual_throughput / target_throughput) * 100)
        
        # Find thermal test
        thermal_test = next((t for t in validation_results["test_results"] if t["test_name"] == "Thermal Management"), None)
        if thermal_test and thermal_test["success"]:
            details = thermal_test["details"]
            max_temp = details.get("max_temperature", 100)
            target_temp = self.targets["thermal_stable_temp"]
            scores["Thermal Efficiency"] = max(0.0, min(100.0, ((target_temp - max_temp + 20) / 20) * 100))
        
        # Find memory test
        memory_test = next((t for t in validation_results["test_results"] if t["test_name"] == "Memory Efficiency"), None)
        if memory_test and memory_test["success"]:
            details = memory_test["details"]
            memory_usage = details.get("final_memory_mb", 200)
            target_memory = self.targets["memory_usage_mb"]
            scores["Memory Efficiency"] = max(0.0, min(100.0, ((target_memory - memory_usage + 50) / 50) * 100))
        
        validation_results["performance_scores"] = scores
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check ML performance
        ml_test = next((t for t in validation_results["test_results"] if t["test_name"] == "ML Prediction Performance"), None)
        if ml_test:
            if not ml_test["success"]:
                recommendations.append("ML performance below target - consider reducing model complexity")
            elif ml_test["details"].get("predictions_per_second", 0) < self.targets["throughput_predictions_sec"] * 0.8:
                recommendations.append("ML performance marginal - monitor under gaming loads")
        
        # Check thermal management
        thermal_test = next((t for t in validation_results["test_results"] if t["test_name"] == "Thermal Management"), None)
        if thermal_test and thermal_test["details"].get("max_temperature", 0) > 80:
            recommendations.append("Consider more aggressive thermal throttling for sustained loads")
        
        # Check memory usage
        memory_test = next((t for t in validation_results["test_results"] if t["test_name"] == "Memory Efficiency"), None)
        if memory_test and memory_test["details"].get("final_memory_mb", 0) > self.targets["memory_usage_mb"] * 0.8:
            recommendations.append("Memory usage high - consider more aggressive cache eviction")
        
        # Overall recommendations
        success_rate = validation_results["summary"]["success_rate"]
        if success_rate == 100:
            recommendations.append("All systems optimal for Steam Deck operation")
        elif success_rate >= 80:
            recommendations.append("System performance good with minor optimizations needed")
        else:
            recommendations.append("Significant optimization needed for Steam Deck deployment")
        
        validation_results["recommendations"] = recommendations


def main():
    """Main validation function"""
    print("🎮 Steam Deck ML Shader Prediction Compiler - Performance Validation")
    print("=" * 70)
    
    # Check if running on Steam Deck
    try:
        with open("/sys/class/dmi/id/product_name", "r") as f:
            product_name = f.read().strip().lower()
            is_steam_deck = "galileo" in product_name or "jupiter" in product_name
            print(f"Hardware: {product_name} {'(Steam Deck detected)' if is_steam_deck else '(Not Steam Deck)'}")
    except Exception:
        print("Hardware: Unknown (could not detect)")
        is_steam_deck = False
    
    # Initialize validator
    validator = SteamDeckPerformanceValidator()
    
    # Run validation
    try:
        results = validator.run_comprehensive_validation()
        
        # Save results
        results_file = "steamdeck_performance_validation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Detailed results saved to: {results_file}")
        
        # Return appropriate exit code
        success_rate = results["summary"]["success_rate"]
        if success_rate >= 90:
            print("🎉 Excellent performance - ready for production")
            return 0
        elif success_rate >= 75:
            print("✅ Good performance - minor optimizations recommended")
            return 0
        else:
            print("⚠️  Performance issues detected - optimization required")
            return 1
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)