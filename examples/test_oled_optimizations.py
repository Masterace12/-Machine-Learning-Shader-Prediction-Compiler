#!/usr/bin/env python3
"""
OLED Steam Deck Optimizations Validation Script
Comprehensive testing and validation of all OLED-specific optimizations
"""

import os
import sys
import asyncio
import logging
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import OLED optimization components
try:
    from src.core.steamdeck_thermal_optimizer import get_thermal_optimizer, SteamDeckModel
    from src.core.oled_memory_optimizer import get_oled_shader_cache
    from src.core.rdna2_gpu_optimizer import get_rdna2_optimizer
    from src.core.oled_steamdeck_integration import get_oled_optimizer
    from src.optimization.thermal_manager import get_thermal_manager
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all OLED optimization modules are properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OLEDValidationSuite:
    """Comprehensive validation suite for OLED Steam Deck optimizations"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        self.start_time = time.time()
        
        # Test configuration
        self.test_duration = 30  # seconds per test
        self.performance_threshold = 0.8  # 80% success rate threshold
        
        print("🎮 OLED Steam Deck Optimization Validation Suite")
        print("=" * 60)
        print(f"System: {self._get_system_info()}")
        print(f"Steam Deck Model: {self._detect_steam_deck_model()}")
        print("=" * 60)
    
    def _get_system_info(self) -> str:
        """Get system information"""
        try:
            with open('/proc/version', 'r') as f:
                kernel = f.read().split()[2]
            
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                if 'AMD Custom APU' in cpu_info:
                    cpu = "AMD Van Gogh APU"
                else:
                    cpu = "Unknown CPU"
            
            return f"{cpu}, Kernel {kernel}"
        except:
            return "Unknown system"
    
    def _detect_steam_deck_model(self) -> str:
        """Detect Steam Deck model"""
        try:
            dmi_path = Path("/sys/class/dmi/id/product_name")
            if dmi_path.exists():
                product = dmi_path.read_text().strip().lower()
                if "galileo" in product:
                    return "OLED (Galileo)"
                elif "jupiter" in product:
                    return "LCD (Jupiter)"
            return "Unknown"
        except:
            return "Detection failed"
    
    async def run_validation_suite(self):
        """Run complete validation suite"""
        print("🚀 Starting OLED optimization validation...\n")
        
        # Test 1: Hardware Detection
        await self._test_hardware_detection()
        
        # Test 2: Thermal Management
        await self._test_thermal_management()
        
        # Test 3: Memory Optimization
        await self._test_memory_optimization()
        
        # Test 4: GPU Integration
        await self._test_gpu_integration()
        
        # Test 5: Comprehensive Integration
        await self._test_comprehensive_integration()
        
        # Test 6: Performance Benchmarks
        await self._test_performance_benchmarks()
        
        # Test 7: Configuration Loading
        await self._test_configuration_loading()
        
        # Generate final report
        self._generate_final_report()
    
    async def _test_hardware_detection(self):
        """Test hardware detection and model identification"""
        print("🔍 Testing Hardware Detection...")
        test_name = "hardware_detection"
        
        try:
            # Test thermal optimizer detection
            thermal_optimizer = get_thermal_optimizer()
            model_detected = thermal_optimizer.model
            
            # Test results
            results = {
                "model_detected": model_detected.value,
                "is_oled": model_detected == SteamDeckModel.OLED,
                "thermal_limits_set": thermal_optimizer.limits is not None,
                "sensors_discovered": len(thermal_optimizer.sensor_paths) > 0
            }
            
            # Validation
            success = all([
                model_detected != SteamDeckModel.UNKNOWN,
                results["thermal_limits_set"],
                results["sensors_discovered"]
            ])
            
            self.results[test_name] = {
                "success": success,
                "details": results,
                "score": sum(results.values()) / len(results) if results else 0.0
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status} - Model: {model_detected.value}, Sensors: {len(thermal_optimizer.sensor_paths)}")
            
        except Exception as e:
            self._handle_test_error(test_name, e)
    
    async def _test_thermal_management(self):
        """Test enhanced thermal management for OLED"""
        print("🌡️  Testing Thermal Management...")
        test_name = "thermal_management"
        
        try:
            thermal_optimizer = get_thermal_optimizer()
            thermal_manager = get_thermal_manager()
            
            # Start monitoring
            thermal_optimizer.start_monitoring(interval=1.0)
            thermal_manager.start_monitoring()
            
            await asyncio.sleep(5)  # Wait for initial readings
            
            # Test thermal readings
            status = thermal_optimizer.get_status()
            thermal_readings = thermal_optimizer.get_thermal_readings()
            
            # Test optimization decisions
            thread_count = thermal_optimizer.get_optimal_thread_count()
            power_budget = thermal_optimizer.get_power_budget_estimate()
            gaming_opts = thermal_optimizer.optimize_for_gaming()
            
            results = {
                "thermal_readings_available": len(thermal_readings) > 0,
                "status_complete": all(k in status for k in ["thermal_state", "max_temperature"]),
                "oled_thread_optimization": thread_count >= 6,  # OLED should allow more threads
                "power_budget_calculated": power_budget > 0,
                "gaming_optimizations": len(gaming_opts) > 5,
                "oled_specific_features": any("oled" in str(v).lower() for v in gaming_opts.values())
            }
            
            success = sum(results.values()) >= 4  # At least 4/6 tests pass
            
            self.results[test_name] = {
                "success": success,
                "details": results,
                "thermal_state": status.get("thermal_state", "unknown"),
                "max_temperature": status.get("max_temperature", 0),
                "optimal_threads": thread_count,
                "power_budget_watts": power_budget,
                "score": sum(results.values()) / len(results)
            }
            
            # Cleanup
            thermal_optimizer.stop_monitoring()
            thermal_manager.stop_monitoring()
            
            status_icon = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status_icon} - Thermal: {status.get('thermal_state', 'unknown')}, "
                  f"Temp: {status.get('max_temperature', 0):.1f}°C, Threads: {thread_count}")
            
        except Exception as e:
            self._handle_test_error(test_name, e)
    
    async def _test_memory_optimization(self):
        """Test memory-mapped file optimization"""
        print("💾 Testing Memory Optimization...")
        test_name = "memory_optimization"
        
        try:
            shader_cache = get_oled_shader_cache()
            
            # Test shader storage and retrieval
            test_shaders = [
                (f"test_shader_{i}", f"shader_bytecode_data_{i}" * 50)
                for i in range(20)
            ]
            
            store_success = 0
            retrieve_success = 0
            
            # Store test shaders
            for shader_hash, shader_data in test_shaders:
                if shader_cache.store_shader(shader_hash, shader_data.encode(), "test_variant"):
                    store_success += 1
            
            await asyncio.sleep(1)  # Allow background processing
            
            # Retrieve test shaders
            for shader_hash, expected_data in test_shaders:
                retrieved = shader_cache.retrieve_shader(shader_hash, "test_variant")
                if retrieved and retrieved.decode() == expected_data:
                    retrieve_success += 1
            
            # Test prefetching (OLED feature)
            shader_hashes = [shader[0] for shader in test_shaders[:10]]
            prefetch_count = shader_cache.prefetch_shaders(shader_hashes)
            
            # Get cache metrics
            metrics = shader_cache.get_metrics()
            
            results = {
                "storage_success_rate": store_success / len(test_shaders),
                "retrieval_success_rate": retrieve_success / len(test_shaders),
                "prefetch_functional": prefetch_count > 0,
                "oled_optimizations_enabled": metrics.oled_specific_optimizations,
                "compression_working": metrics.compression_ratio > 1.0,
                "cache_size_reasonable": 0 < metrics.total_size_mb < 100
            }
            
            success = all([
                results["storage_success_rate"] > 0.8,
                results["retrieval_success_rate"] > 0.8,
                results["oled_optimizations_enabled"]
            ])
            
            self.results[test_name] = {
                "success": success,
                "details": results,
                "cache_metrics": {
                    "hit_rate": metrics.hit_rate,
                    "total_size_mb": metrics.total_size_mb,
                    "compression_ratio": metrics.compression_ratio
                },
                "score": sum(results.values()) / len(results)
            }
            
            status_icon = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status_icon} - Store: {results['storage_success_rate']:.1%}, "
                  f"Retrieve: {results['retrieval_success_rate']:.1%}, "
                  f"Size: {metrics.total_size_mb:.1f}MB")
            
            # Cleanup test data
            shader_cache.cleanup()
            
        except Exception as e:
            self._handle_test_error(test_name, e)
    
    async def _test_gpu_integration(self):
        """Test RDNA 2 GPU integration"""
        print("🎮 Testing GPU Integration...")
        test_name = "gpu_integration"
        
        try:
            gpu_optimizer = get_rdna2_optimizer(oled_model=True)
            
            # Start monitoring
            gpu_optimizer.start_monitoring(interval=1.0)
            await asyncio.sleep(3)  # Allow metrics collection
            
            # Test GPU metrics
            metrics = gpu_optimizer.get_gpu_metrics()
            
            # Test optimization profiles
            oled_settings = gpu_optimizer.optimize_for_shader_compilation("oled_performance")
            balanced_settings = gpu_optimizer.optimize_for_shader_compilation("balanced")
            
            # Test memory budget setting
            memory_budget_set = gpu_optimizer.set_memory_budget(512)
            
            # Test gaming workload detection
            gaming_info = gpu_optimizer.monitor_gaming_workload()
            
            # Get optimization recommendations
            recommendations = gpu_optimizer.get_optimization_recommendations()
            
            results = {
                "metrics_available": metrics.gpu_utilization_percent >= 0,
                "oled_profile_applied": len(oled_settings) > 3,
                "balanced_profile_applied": len(balanced_settings) > 3,
                "memory_budget_functional": memory_budget_set,
                "gaming_detection_working": "gaming_detected" in gaming_info,
                "recommendations_generated": len(recommendations) > 0,
                "rdna2_features_enabled": any("wave" in str(v).lower() for v in oled_settings.values())
            }
            
            success = sum(results.values()) >= 5  # At least 5/7 tests pass
            
            self.results[test_name] = {
                "success": success,
                "details": results,
                "gpu_metrics": {
                    "utilization": metrics.gpu_utilization_percent,
                    "clock_mhz": metrics.clock_speed_mhz,
                    "temperature": metrics.temperature_celsius
                },
                "optimization_settings": oled_settings,
                "score": sum(results.values()) / len(results)
            }
            
            # Cleanup
            gpu_optimizer.stop_monitoring()
            gpu_optimizer.cleanup()
            
            status_icon = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status_icon} - GPU: {metrics.gpu_utilization_percent:.1f}%, "
                  f"Clock: {metrics.clock_speed_mhz}MHz, Temp: {metrics.temperature_celsius:.1f}°C")
            
        except Exception as e:
            self._handle_test_error(test_name, e)
    
    async def _test_comprehensive_integration(self):
        """Test comprehensive OLED integration"""
        print("🔄 Testing Comprehensive Integration...")
        test_name = "comprehensive_integration"
        
        try:
            oled_optimizer = get_oled_optimizer()
            
            # Test initialization
            oled_optimizer.start_comprehensive_optimization()
            await asyncio.sleep(5)  # Allow initialization
            
            # Test status collection
            status = oled_optimizer.get_comprehensive_status()
            
            # Test recommendations
            recommendations = oled_optimizer.get_optimization_recommendations()
            
            # Test performance callback
            callback_called = False
            def test_callback(metrics):
                nonlocal callback_called
                callback_called = True
            
            oled_optimizer.add_performance_callback(test_callback)
            await asyncio.sleep(2)  # Wait for callback
            
            # Test temporary performance modes
            async with oled_optimizer.temporary_performance_mode("maximum"):
                await asyncio.sleep(1)
                max_mode_status = oled_optimizer.get_comprehensive_status()
            
            results = {
                "initialization_successful": status.get("optimization_active", False),
                "oled_verification": status.get("oled_verified", False),
                "components_active": len(status.get("components", {})) == 4,
                "metrics_collection": "performance" in status,
                "recommendations_working": len(recommendations) > 0,
                "callbacks_functional": callback_called,
                "temporary_modes_working": max_mode_status.get("optimization_active", False),
                "oled_advantages_calculated": bool(status.get("oled_advantages", {}))
            }
            
            success = sum(results.values()) >= 6  # At least 6/8 tests pass
            
            # Export test report
            report_path = Path("/tmp/oled_integration_test_report.json")
            oled_optimizer.export_performance_report(report_path)
            
            self.results[test_name] = {
                "success": success,
                "details": results,
                "status_snapshot": status,
                "report_exported": report_path.exists(),
                "score": sum(results.values()) / len(results)
            }
            
            # Cleanup
            oled_optimizer.stop_comprehensive_optimization()
            
            status_icon = "✅ PASSED" if success else "❌ FAILED"
            oled_verified = "✓" if status.get("oled_verified") else "✗"
            active_components = len(status.get("components", {}))
            print(f"   {status_icon} - OLED: {oled_verified}, Active Components: {active_components}/4")
            
        except Exception as e:
            self._handle_test_error(test_name, e)
    
    async def _test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("📊 Testing Performance Benchmarks...")
        test_name = "performance_benchmarks"
        
        try:
            # Simulated workload tests
            results = {}
            
            # Test 1: Thermal optimizer performance
            thermal_optimizer = get_thermal_optimizer()
            thermal_optimizer.start_monitoring(interval=0.5)
            
            start_time = time.time()
            for _ in range(100):
                thermal_optimizer.get_thermal_readings()
                thermal_optimizer.get_optimal_thread_count()
            thermal_read_time = time.time() - start_time
            
            thermal_optimizer.stop_monitoring()
            results["thermal_read_performance"] = thermal_read_time < 5.0  # Should complete in <5s
            
            # Test 2: Cache performance
            shader_cache = get_oled_shader_cache()
            
            # Generate test data
            test_data = [(f"perf_test_{i}", f"data_{i}" * 100) for i in range(50)]
            
            # Store performance test
            start_time = time.time()
            for shader_hash, data in test_data:
                shader_cache.store_shader(shader_hash, data.encode())
            store_time = time.time() - start_time
            
            # Retrieve performance test  
            start_time = time.time()
            for shader_hash, _ in test_data:
                shader_cache.retrieve_shader(shader_hash)
            retrieve_time = time.time() - start_time
            
            results["cache_store_performance"] = store_time < 2.0  # <2s for 50 shaders
            results["cache_retrieve_performance"] = retrieve_time < 1.0  # <1s for 50 retrievals
            
            # Test 3: Integration overhead
            oled_optimizer = get_oled_optimizer()
            
            start_time = time.time()
            oled_optimizer.start_comprehensive_optimization()
            init_time = time.time() - start_time
            
            start_time = time.time()
            for _ in range(10):
                oled_optimizer.get_comprehensive_status()
            status_time = time.time() - start_time
            
            oled_optimizer.stop_comprehensive_optimization()
            
            results["integration_init_performance"] = init_time < 3.0  # <3s initialization
            results["status_query_performance"] = status_time < 2.0  # <2s for 10 queries
            
            success = sum(results.values()) >= 3  # At least 3/5 benchmarks pass
            
            self.results[test_name] = {
                "success": success,
                "details": results,
                "timings": {
                    "thermal_reads": thermal_read_time,
                    "cache_store": store_time,
                    "cache_retrieve": retrieve_time,
                    "integration_init": init_time,
                    "status_queries": status_time
                },
                "score": sum(results.values()) / len(results)
            }
            
            # Cleanup
            shader_cache.cleanup()
            
            status_icon = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status_icon} - Init: {init_time:.2f}s, Store: {store_time:.2f}s, "
                  f"Retrieve: {retrieve_time:.2f}s")
            
        except Exception as e:
            self._handle_test_error(test_name, e)
    
    async def _test_configuration_loading(self):
        """Test configuration loading and validation"""
        print("⚙️  Testing Configuration Loading...")
        test_name = "configuration_loading"
        
        try:
            config_path = Path(__file__).parent / "config" / "steamdeck_oled_config.json"
            
            # Test config file existence
            config_exists = config_path.exists()
            
            # Test config loading
            config_data = None
            config_valid = False
            
            if config_exists:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Validate config structure
                required_sections = ["system", "thermal", "ml", "cache", "gpu", "power"]
                config_valid = all(section in config_data for section in required_sections)
            
            # Test OLED-specific settings
            oled_settings = {}
            if config_data:
                oled_settings = {
                    "oled_optimized": config_data.get("system", {}).get("oled_optimized", False),
                    "enhanced_cooling": config_data.get("thermal", {}).get("oled_enhanced_cooling", False),
                    "rdna2_optimized": config_data.get("gpu", {}).get("rdna2_optimized", False),
                    "burst_mode": config_data.get("cache", {}).get("oled_burst_mode", False)
                }
            
            # Test integration with optimizer
            oled_optimizer = get_oled_optimizer()
            integration_working = hasattr(oled_optimizer, 'config') and bool(oled_optimizer.config)
            
            results = {
                "config_file_exists": config_exists,
                "config_loads_successfully": config_data is not None,
                "config_structure_valid": config_valid,
                "oled_settings_present": sum(oled_settings.values()) >= 2,
                "integration_working": integration_working
            }
            
            success = sum(results.values()) >= 4  # At least 4/5 tests pass
            
            self.results[test_name] = {
                "success": success,
                "details": results,
                "config_path": str(config_path),
                "oled_settings": oled_settings,
                "config_version": config_data.get("version", "unknown") if config_data else None,
                "score": sum(results.values()) / len(results)
            }
            
            status_icon = "✅ PASSED" if success else "❌ FAILED"
            version = config_data.get("version", "unknown") if config_data else "none"
            oled_count = sum(oled_settings.values()) if oled_settings else 0
            print(f"   {status_icon} - Config: {version}, OLED Settings: {oled_count}/4")
            
        except Exception as e:
            self._handle_test_error(test_name, e)
    
    def _handle_test_error(self, test_name: str, error: Exception):
        """Handle test errors"""
        error_msg = f"Test {test_name} failed: {str(error)}"
        self.errors.append(error_msg)
        logger.error(error_msg)
        
        self.results[test_name] = {
            "success": False,
            "error": str(error),
            "score": 0.0
        }
        
        print(f"   ❌ FAILED - {str(error)}")
    
    def _generate_final_report(self):
        """Generate final validation report"""
        print("\n" + "=" * 60)
        print("📋 OLED OPTIMIZATION VALIDATION REPORT")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result.get("success", False))
        overall_score = sum(result.get("score", 0) for result in self.results.values()) / total_tests
        
        print(f"Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
        print(f"Overall Score: {overall_score:.1%}")
        print(f"Execution Time: {time.time() - self.start_time:.1f} seconds")
        
        # Individual test results
        print(f"\nDetailed Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get("success") else "❌ FAIL"
            score = result.get("score", 0.0)
            print(f"  {status} {test_name:<25} ({score:.1%})")
            
            if "error" in result:
                print(f"      Error: {result['error']}")
        
        # Performance summary
        if "performance_benchmarks" in self.results:
            print(f"\nPerformance Summary:")
            timings = self.results["performance_benchmarks"].get("timings", {})
            for metric, timing in timings.items():
                print(f"  {metric}: {timing:.2f}s")
        
        # OLED-specific benefits
        oled_benefits = []
        if self.results.get("hardware_detection", {}).get("details", {}).get("is_oled", False):
            oled_benefits.append("✓ OLED model detected")
        if self.results.get("thermal_management", {}).get("details", {}).get("oled_thread_optimization", False):
            oled_benefits.append("✓ Enhanced thread management")
        if self.results.get("memory_optimization", {}).get("details", {}).get("oled_optimizations_enabled", False):
            oled_benefits.append("✓ OLED memory optimizations active")
        if self.results.get("gpu_integration", {}).get("details", {}).get("rdna2_features_enabled", False):
            oled_benefits.append("✓ RDNA2 optimizations enabled")
        
        if oled_benefits:
            print(f"\nOLED-Specific Benefits Validated:")
            for benefit in oled_benefits:
                print(f"  {benefit}")
        
        # Warnings and recommendations
        if self.errors:
            print(f"\nErrors Encountered:")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  ⚠️  {error}")
        
        # Final assessment
        print(f"\n" + "=" * 60)
        if overall_score >= 0.8 and passed_tests >= total_tests * 0.8:
            print("🎉 EXCELLENT: OLED optimizations are working exceptionally well!")
            assessment = "EXCELLENT"
        elif overall_score >= 0.6 and passed_tests >= total_tests * 0.6:
            print("✅ GOOD: OLED optimizations are working well with minor issues")
            assessment = "GOOD"
        elif overall_score >= 0.4:
            print("⚠️  FAIR: OLED optimizations are partially working, improvements needed")
            assessment = "FAIR"
        else:
            print("❌ POOR: OLED optimizations need significant fixes")
            assessment = "POOR"
        
        print("=" * 60)
        
        # Export detailed report
        report_path = Path("/tmp/oled_validation_report.json")
        detailed_report = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "steam_deck_model": self._detect_steam_deck_model(),
            "overall_assessment": assessment,
            "overall_score": overall_score,
            "tests_passed": f"{passed_tests}/{total_tests}",
            "execution_time": time.time() - self.start_time,
            "test_results": self.results,
            "errors": self.errors,
            "oled_benefits": oled_benefits
        }
        
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"Detailed report exported to: {report_path}")
        
        return assessment == "EXCELLENT" or assessment == "GOOD"


async def main():
    """Main validation function"""
    validation_suite = OLEDValidationSuite()
    success = await validation_suite.run_validation_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        logging.exception("Validation error")
        sys.exit(1)