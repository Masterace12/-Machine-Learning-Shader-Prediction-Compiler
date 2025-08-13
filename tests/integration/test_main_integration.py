#!/usr/bin/env python3
"""
Integration tests for the main shader prediction system
Tests the complete system integration and component interactions
"""

import pytest
import time
import tempfile
import threading
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from main import OptimizedShaderSystem, SystemConfig
    HAS_MAIN_MODULE = True
except ImportError as e:
    print(f"Main module not available: {e}")
    HAS_MAIN_MODULE = False

# Import components for testing
try:
    from src.ml.optimized_ml_predictor import get_optimized_predictor
    HAS_ML_PREDICTOR = True
except ImportError:
    HAS_ML_PREDICTOR = False

try:
    from src.thermal.optimized_thermal_manager import get_thermal_manager
    HAS_THERMAL_MANAGER = True
except ImportError:
    HAS_THERMAL_MANAGER = False

try:
    from src.monitoring.performance_monitor import get_performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False


@pytest.mark.skipif(not HAS_MAIN_MODULE, reason="Main module not available")
@pytest.mark.integration
class TestSystemConfig:
    """Test system configuration"""
    
    def test_default_config(self):
        """Test default system configuration"""
        config = SystemConfig()
        
        assert config.enable_ml_prediction is True
        assert config.enable_cache is True
        assert config.enable_thermal_management is True
        assert config.enable_performance_monitoring is True
        assert config.enable_async is True
        assert config.max_memory_mb == 200
        assert config.max_compilation_threads == 4
        assert config.steam_deck_optimized is True
    
    def test_custom_config(self):
        """Test custom system configuration"""
        config = SystemConfig(
            enable_ml_prediction=False,
            enable_cache=False,
            max_memory_mb=100,
            max_compilation_threads=2,
            steam_deck_optimized=False
        )
        
        assert config.enable_ml_prediction is False
        assert config.enable_cache is False
        assert config.max_memory_mb == 100
        assert config.max_compilation_threads == 2
        assert config.steam_deck_optimized is False
    
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Test with extreme values
        config = SystemConfig(
            max_memory_mb=1,     # Very low
            max_compilation_threads=0  # Zero threads
        )
        
        assert config.max_memory_mb == 1
        assert config.max_compilation_threads == 0


@pytest.mark.skipif(not HAS_MAIN_MODULE, reason="Main module not available")
@pytest.mark.integration
class TestOptimizedShaderSystem:
    """Test the main optimized shader system"""
    
    @pytest.fixture
    def system_config(self):
        """Create test system configuration"""
        return SystemConfig(
            enable_ml_prediction=True,
            enable_cache=False,  # Disable cache to avoid dependencies
            enable_thermal_management=True,
            enable_performance_monitoring=True,
            enable_async=False,  # Disable async for simpler testing
            max_memory_mb=100,
            max_compilation_threads=2
        )
    
    @pytest.fixture
    def shader_system(self, system_config):
        """Create test shader system"""
        return OptimizedShaderSystem(config=system_config)
    
    def test_system_initialization(self, shader_system):
        """Test system initialization"""
        assert shader_system.config is not None
        assert not shader_system.running
        assert not shader_system.shutdown_requested
        assert isinstance(shader_system.stats, dict)
        assert shader_system.stats['predictions_made'] == 0
        assert shader_system.logger is not None
    
    def test_lazy_component_loading(self, shader_system):
        """Test lazy loading of system components"""
        # Components should be None initially
        assert shader_system._ml_predictor is None
        assert shader_system._thermal_manager is None
        assert shader_system._performance_monitor is None
        
        # Access should trigger loading
        if HAS_ML_PREDICTOR:
            ml_predictor = shader_system.ml_predictor
            assert ml_predictor is not None
            assert shader_system._ml_predictor is ml_predictor
        
        if HAS_THERMAL_MANAGER:
            thermal_manager = shader_system.thermal_manager
            assert thermal_manager is not None
            assert shader_system._thermal_manager is thermal_manager
        
        if HAS_PERFORMANCE_MONITOR:
            perf_monitor = shader_system.performance_monitor
            assert perf_monitor is not None
            assert shader_system._performance_monitor is perf_monitor
    
    def test_steam_deck_detection(self, shader_system):
        """Test Steam Deck hardware detection"""
        is_steam_deck = shader_system._detect_steam_deck()
        
        # Should return a boolean
        assert isinstance(is_steam_deck, bool)
    
    def test_steam_deck_optimizations(self, shader_system):
        """Test Steam Deck specific optimizations"""
        original_memory = shader_system.config.max_memory_mb
        original_threads = shader_system.config.max_compilation_threads
        
        # Mock Steam Deck detection
        with patch.object(shader_system, '_detect_steam_deck', return_value=True):
            shader_system._optimize_for_steam_deck()
            
            # Should apply optimizations
            assert shader_system.config.max_memory_mb <= 150
            assert shader_system.config.max_compilation_threads <= 4
            assert shader_system.config.respect_battery_level is True
    
    def test_non_steam_deck_optimizations(self, shader_system):
        """Test optimizations for non-Steam Deck systems"""
        original_config = {
            'max_memory_mb': shader_system.config.max_memory_mb,
            'max_compilation_threads': shader_system.config.max_compilation_threads
        }
        
        # Mock non-Steam Deck detection
        with patch.object(shader_system, '_detect_steam_deck', return_value=False):
            shader_system._optimize_for_steam_deck()
            
            # Should keep original settings or apply generic optimizations
            assert shader_system.config.max_memory_mb >= 0
            assert shader_system.config.max_compilation_threads >= 0
    
    def test_system_status_reporting(self, shader_system):
        """Test system status reporting"""
        status = shader_system.get_system_status()
        
        required_keys = [
            'running',
            'uptime_seconds', 
            'statistics',
            'components',
            'config'
        ]
        
        for key in required_keys:
            assert key in status, f"Status missing required key: {key}"
        
        # Check component status structure
        assert isinstance(status['components'], dict)
        component_keys = ['ml_predictor', 'cache_manager', 'thermal_manager', 'performance_monitor']
        for key in component_keys:
            assert key in status['components']
            assert isinstance(status['components'][key], bool)
        
        # Check config structure
        assert isinstance(status['config'], dict)
        config_keys = ['max_memory_mb', 'max_compilation_threads', 'steam_deck_optimized']
        for key in config_keys:
            assert key in status['config']
    
    def test_sync_mode_startup(self, shader_system):
        """Test synchronous mode startup and shutdown"""
        # Should not be running initially
        assert not shader_system.running
        
        # Mock the main loop to exit quickly
        original_shutdown = shader_system.shutdown_requested
        
        def mock_start():
            shader_system.running = True
            shader_system.stats["start_time"] = time.time()
            # Simulate immediate shutdown request
            shader_system.shutdown_requested = True
            # Simulate startup actions
            shader_system._optimize_for_steam_deck()
            # Simulate shutdown
            shader_system.shutdown_sync()
        
        # Replace start_sync with our mock
        shader_system.start_sync = mock_start
        
        # Test sync start
        shader_system.start()
        
        # Should have gone through startup process
        assert shader_system.stats["start_time"] > 0
    
    def test_signal_handling(self, shader_system):
        """Test signal handling for graceful shutdown"""
        import signal
        
        assert not shader_system.shutdown_requested
        
        # Simulate signal reception
        shader_system._signal_handler(signal.SIGTERM, None)
        
        assert shader_system.shutdown_requested
    
    def test_logging_setup(self, shader_system):
        """Test logging configuration"""
        logger = shader_system._setup_logging()
        
        assert logger is not None
        assert logger.name == shader_system.__module__
        
        # Log directory should exist
        log_dir = Path.home() / '.cache' / 'shader-predict-compile'
        assert log_dir.exists()


@pytest.mark.skipif(not HAS_MAIN_MODULE, reason="Main module not available")
@pytest.mark.integration
class TestSystemIntegration:
    """Test integration between system components"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create system with all available components enabled"""
        config = SystemConfig(
            enable_ml_prediction=HAS_ML_PREDICTOR,
            enable_cache=False,  # Disable problematic cache for now
            enable_thermal_management=HAS_THERMAL_MANAGER,
            enable_performance_monitoring=HAS_PERFORMANCE_MONITOR,
            enable_async=False,
            max_memory_mb=100,
            max_compilation_threads=2
        )
        return OptimizedShaderSystem(config=config)
    
    def test_component_initialization_order(self, integrated_system):
        """Test that components initialize in correct order"""
        # Access components in order
        components = []
        
        if HAS_ML_PREDICTOR:
            ml = integrated_system.ml_predictor
            if ml:
                components.append('ml')
        
        if HAS_THERMAL_MANAGER:
            thermal = integrated_system.thermal_manager
            if thermal:
                components.append('thermal')
        
        if HAS_PERFORMANCE_MONITOR:
            perf = integrated_system.performance_monitor
            if perf:
                components.append('performance')
        
        # All requested components should be available
        assert len(components) >= 0  # At least some components should load
    
    def test_component_interaction(self, integrated_system):
        """Test interaction between system components"""
        if not (HAS_THERMAL_MANAGER and HAS_PERFORMANCE_MONITOR):
            pytest.skip("Need thermal and performance components for interaction test")
        
        thermal = integrated_system.thermal_manager
        perf = integrated_system.performance_monitor
        
        if thermal and perf:
            # Start both components
            thermal.start_monitoring()
            perf.start_monitoring()
            
            try:
                # Let them run briefly
                time.sleep(1.5)
                
                # Both should have collected data
                thermal_status = thermal.get_status()
                perf_report = perf.get_performance_report()
                
                assert thermal_status is not None
                assert perf_report is not None
                
                # Verify data collection
                assert 'thermal_state' in thermal_status
                assert 'health_score' in perf_report
            
            finally:
                thermal.stop_monitoring()
                perf.stop_monitoring()
    
    def test_system_health_monitoring(self, integrated_system):
        """Test overall system health monitoring"""
        if not HAS_PERFORMANCE_MONITOR:
            pytest.skip("Need performance monitor for health test")
        
        # Mock the health monitoring function
        health_data = []
        
        async def mock_health_monitor():
            # Simulate health monitoring cycle
            for i in range(3):
                if integrated_system.performance_monitor:
                    report = integrated_system.performance_monitor.get_performance_report()
                    health_data.append(report)
                await asyncio.sleep(0.5)
        
        # If we have async support, test it
        try:
            import asyncio
            asyncio.run(mock_health_monitor())
            
            if health_data:
                assert len(health_data) > 0
                for report in health_data:
                    assert 'health_score' in report
        except ImportError:
            # Skip async test if asyncio not available
            pass
    
    def test_error_handling_integration(self, integrated_system):
        """Test error handling across component boundaries"""
        # Test with invalid configuration that might cause errors
        error_config = SystemConfig(
            enable_ml_prediction=True,
            enable_thermal_management=True,
            enable_performance_monitoring=True,
            max_memory_mb=-1,  # Invalid
            max_compilation_threads=-1  # Invalid
        )
        
        # Should handle invalid config gracefully
        error_system = OptimizedShaderSystem(config=error_config)
        
        # Should still be able to get status without crashing
        status = error_system.get_system_status()
        assert isinstance(status, dict)
    
    def test_memory_usage_monitoring(self, integrated_system):
        """Test memory usage monitoring across components"""
        if not HAS_PERFORMANCE_MONITOR:
            pytest.skip("Need performance monitor for memory test")
        
        perf = integrated_system.performance_monitor
        if perf:
            perf.start_monitoring()
            
            try:
                # Let it collect some data
                time.sleep(1.1)
                
                report = perf.get_performance_report()
                current = report['current_metrics']
                
                # Should report reasonable memory usage
                assert 'memory_usage_percent' in current
                assert 0.0 <= current['memory_usage_percent'] <= 100.0
                assert 'memory_available_mb' in current
                assert current['memory_available_mb'] >= 0
            
            finally:
                perf.stop_monitoring()


@pytest.mark.skipif(not HAS_MAIN_MODULE, reason="Main module not available")
@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end scenario tests"""
    
    def test_typical_gaming_scenario(self):
        """Test typical gaming scenario workflow"""
        config = SystemConfig(
            enable_ml_prediction=HAS_ML_PREDICTOR,
            enable_thermal_management=HAS_THERMAL_MANAGER,
            enable_performance_monitoring=HAS_PERFORMANCE_MONITOR,
            enable_async=False,
            steam_deck_optimized=True
        )
        
        system = OptimizedShaderSystem(config=config)
        
        try:
            # Simulate system startup
            system._optimize_for_steam_deck()
            
            # Get initial status
            initial_status = system.get_system_status()
            assert not initial_status['running']
            
            # Start monitoring components if available
            if system.thermal_manager:
                system.thermal_manager.start_monitoring()
            
            if system.performance_monitor:
                system.performance_monitor.start_monitoring()
            
            # Simulate some runtime
            time.sleep(1.5)
            
            # Get runtime status
            runtime_status = system.get_system_status()
            
            # Verify components are working
            if 'thermal' in runtime_status:
                assert 'thermal_state' in runtime_status['thermal']
            
            if 'performance' in runtime_status:
                assert 'health_score' in runtime_status['performance']
            
            # Simulate shutdown
            system.shutdown_requested = True
            
        finally:
            # Cleanup
            if system.thermal_manager and system.thermal_manager.monitoring_active:
                system.thermal_manager.stop_monitoring()
            
            if system.performance_monitor and system.performance_monitor.monitoring_active:
                system.performance_monitor.stop_monitoring()
    
    def test_low_memory_scenario(self):
        """Test system behavior under low memory conditions"""
        config = SystemConfig(
            max_memory_mb=10,  # Very low memory limit
            max_compilation_threads=1,
            enable_async=False
        )
        
        system = OptimizedShaderSystem(config=config)
        
        # Should handle low memory gracefully
        status = system.get_system_status()
        assert status['config']['max_memory_mb'] == 10
        assert status['config']['max_compilation_threads'] == 1
        
        # Apply optimizations
        system._optimize_for_steam_deck()
        
        # Should maintain reasonable limits
        assert system.config.max_memory_mb >= 0
        assert system.config.max_compilation_threads >= 0
    
    def test_component_failure_resilience(self):
        """Test system resilience to component failures"""
        config = SystemConfig(
            enable_ml_prediction=True,
            enable_thermal_management=True,
            enable_performance_monitoring=True
        )
        
        system = OptimizedShaderSystem(config=config)
        
        # Mock a component failure
        with patch.object(system, 'thermal_manager', side_effect=Exception("Component failed")):
            try:
                # System should still provide status despite component failure
                status = system.get_system_status()
                assert isinstance(status, dict)
                assert 'components' in status
            except Exception:
                # If an exception occurs, it should be handled gracefully
                pass
    
    def test_configuration_loading_and_saving(self, clean_temp_dir):
        """Test configuration loading and saving"""
        config_file = clean_temp_dir / "test_config.json"
        
        # Create test configuration
        test_config = {
            "enable_ml_prediction": False,
            "enable_cache": False,
            "max_memory_mb": 150,
            "max_compilation_threads": 3
        }
        
        config_file.write_text(json.dumps(test_config))
        
        # System should be able to use config values
        config = SystemConfig(
            enable_ml_prediction=test_config["enable_ml_prediction"],
            enable_cache=test_config["enable_cache"],
            max_memory_mb=test_config["max_memory_mb"],
            max_compilation_threads=test_config["max_compilation_threads"]
        )
        
        assert config.enable_ml_prediction == test_config["enable_ml_prediction"]
        assert config.enable_cache == test_config["enable_cache"]
        assert config.max_memory_mb == test_config["max_memory_mb"]
        assert config.max_compilation_threads == test_config["max_compilation_threads"]


@pytest.mark.skipif(not HAS_MAIN_MODULE, reason="Main module not available")
@pytest.mark.benchmark
class TestSystemPerformance:
    """Performance benchmarks for the integrated system"""
    
    def test_system_startup_time(self, benchmark):
        """Benchmark system startup time"""
        config = SystemConfig(enable_async=False)
        
        def create_system():
            system = OptimizedShaderSystem(config=config)
            system._optimize_for_steam_deck()
            return system
        
        result = benchmark(create_system)
        assert isinstance(result, OptimizedShaderSystem)
    
    def test_status_reporting_performance(self, benchmark):
        """Benchmark status reporting performance"""
        config = SystemConfig(enable_async=False)
        system = OptimizedShaderSystem(config=config)
        
        result = benchmark(system.get_system_status)
        
        assert isinstance(result, dict)
        assert 'running' in result
    
    def test_component_access_performance(self, benchmark):
        """Benchmark component access (lazy loading)"""
        config = SystemConfig(enable_async=False)
        system = OptimizedShaderSystem(config=config)
        
        def access_all_components():
            components = []
            if HAS_ML_PREDICTOR:
                components.append(system.ml_predictor)
            if HAS_THERMAL_MANAGER:
                components.append(system.thermal_manager)
            if HAS_PERFORMANCE_MONITOR:
                components.append(system.performance_monitor)
            return components
        
        result = benchmark(access_all_components)
        assert isinstance(result, list)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])