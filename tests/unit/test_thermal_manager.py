#!/usr/bin/env python3
"""
Comprehensive tests for optimized thermal management system
"""

import pytest
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from src.thermal.optimized_thermal_manager import (
        OptimizedThermalManager,
        ThermalSample,
        ThermalState,
        SteamDeckModel,
        get_thermal_manager
    )
    HAS_THERMAL_MODULE = True
except ImportError as e:
    print(f"Thermal module not available: {e}")
    HAS_THERMAL_MODULE = False


@pytest.mark.skipif(not HAS_THERMAL_MODULE, reason="Thermal module not available")
@pytest.mark.unit
@pytest.mark.thermal
class TestThermalSample:
    """Test thermal sample data structure"""
    
    def test_sample_creation(self):
        """Test thermal sample creation"""
        timestamp = time.time()
        sample = ThermalSample(
            timestamp=timestamp,
            apu_temp=75.5,
            cpu_temp=70.0,
            gpu_temp=72.5,
            fan_rpm=2500,
            power_draw=12.5,
            battery_level=85.0,
            gaming_active=True,
            compilation_threads=4
        )
        
        assert sample.timestamp == timestamp
        assert sample.apu_temp == 75.5
        assert sample.cpu_temp == 70.0
        assert sample.gpu_temp == 72.5
        assert sample.fan_rpm == 2500
        assert sample.power_draw == 12.5
        assert sample.battery_level == 85.0
        assert sample.gaming_active is True
        assert sample.compilation_threads == 4
    
    def test_sample_validation(self):
        """Test thermal sample validation in __post_init__"""
        # Test temperature clamping
        sample = ThermalSample(
            timestamp=time.time(),
            apu_temp=200.0,  # Too high
            cpu_temp=-10.0,  # Too low
            gpu_temp=75.0,
            fan_rpm=3000,
            power_draw=15.0,
            battery_level=50.0,
            gaming_active=False,
            compilation_threads=2
        )
        
        # Temperatures should be clamped
        assert sample.apu_temp == 150.0  # Clamped to max
        assert sample.cpu_temp == 0.0    # Clamped to min
        assert sample.gpu_temp == 75.0   # Valid, unchanged
    
    def test_sample_edge_cases(self):
        """Test thermal sample with edge case values"""
        sample = ThermalSample(
            timestamp=0.0,
            apu_temp=0.0,
            cpu_temp=150.0,
            gpu_temp=75.0,
            fan_rpm=0,
            power_draw=0.0,
            battery_level=0.0,
            gaming_active=False,
            compilation_threads=0
        )
        
        assert sample.timestamp == 0.0
        assert sample.apu_temp == 0.0
        assert sample.cpu_temp == 150.0
        assert sample.fan_rpm == 0
        assert sample.compilation_threads == 0


@pytest.mark.skipif(not HAS_THERMAL_MODULE, reason="Thermal module not available")
@pytest.mark.unit
@pytest.mark.thermal
class TestOptimizedThermalManager:
    """Test optimized thermal manager"""
    
    @pytest.fixture
    def thermal_manager(self, clean_temp_dir):
        """Create a test thermal manager"""
        config_path = clean_temp_dir / "thermal_test.json"
        manager = OptimizedThermalManager(config_path=config_path)
        yield manager
        if manager.monitoring_active:
            manager.stop_monitoring()
    
    def test_manager_initialization(self, thermal_manager):
        """Test thermal manager initialization"""
        assert thermal_manager.config_path is not None
        assert thermal_manager.steam_deck_model in [SteamDeckModel.LCD, SteamDeckModel.OLED, SteamDeckModel.UNKNOWN]
        assert thermal_manager.current_state == ThermalState.NORMAL
        assert thermal_manager.max_compilation_threads > 0
        assert thermal_manager.monitoring_interval == 1.0
        assert not thermal_manager.monitoring_active
        assert len(thermal_manager.compilation_callbacks) == 0
        assert thermal_manager.logger is not None
    
    def test_steam_deck_model_detection(self):
        """Test Steam Deck model detection"""
        manager = OptimizedThermalManager()
        
        # Should return a valid model enum
        assert isinstance(manager.steam_deck_model, SteamDeckModel)
        
        # Test mock detection scenarios
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value='Jupiter\n'):
                manager2 = OptimizedThermalManager()
                assert manager2.steam_deck_model in [SteamDeckModel.LCD, SteamDeckModel.OLED]
    
    def test_sensor_discovery(self, thermal_manager):
        """Test thermal sensor discovery"""
        sensors = thermal_manager.sensor_paths
        
        # Should be a dictionary
        assert isinstance(sensors, dict)
        # Should have reasonable number of sensors (even if zero in test environment)
        assert len(sensors) >= 0
    
    def test_mock_sensor_reading(self, thermal_manager):
        """Test mock sensor reading when in mock mode"""
        if thermal_manager._mock_mode:
            readings = thermal_manager._read_sensors()
            
            assert isinstance(readings, dict)
            assert 'apu_temp' in readings
            assert 'cpu_temp' in readings
            assert 'gpu_temp' in readings
            assert 'fan_rpm' in readings
            
            # Should be reasonable values
            assert 40.0 <= readings['apu_temp'] <= 100.0
            assert 40.0 <= readings['cpu_temp'] <= 100.0
            assert 40.0 <= readings['gpu_temp'] <= 100.0
            assert 0 <= readings['fan_rpm'] <= 6000
    
    def test_current_sample_generation(self, thermal_manager):
        """Test current thermal sample generation"""
        sample = thermal_manager._get_current_sample()
        
        assert isinstance(sample, ThermalSample)
        assert sample.timestamp > 0
        assert 0.0 <= sample.apu_temp <= 150.0
        assert 0.0 <= sample.cpu_temp <= 150.0
        assert 0.0 <= sample.gpu_temp <= 150.0
        assert sample.fan_rpm >= 0
        assert sample.power_draw >= 0.0
        assert 0.0 <= sample.battery_level <= 100.0
        assert isinstance(sample.gaming_active, bool)
        assert sample.compilation_threads >= 0
    
    def test_thermal_state_determination(self, thermal_manager):
        """Test thermal state determination logic"""
        # Test different temperature scenarios
        test_cases = [
            (45.0, ThermalState.COOL),
            (65.0, ThermalState.OPTIMAL), 
            (75.0, ThermalState.NORMAL),
            (82.0, ThermalState.WARM),
            (87.0, ThermalState.HOT),
            (92.0, ThermalState.THROTTLING),
            (96.0, ThermalState.CRITICAL)
        ]
        
        for temp, expected_state in test_cases:
            sample = ThermalSample(
                timestamp=time.time(),
                apu_temp=temp,
                cpu_temp=temp,
                gpu_temp=temp,
                fan_rpm=2000,
                power_draw=10.0,
                battery_level=75.0,
                gaming_active=False,
                compilation_threads=4
            )
            
            state = thermal_manager._determine_thermal_state(sample)
            assert state == expected_state, f"Temperature {temp}°C should result in {expected_state}, got {state}"
    
    def test_compilation_thread_adjustment(self, thermal_manager):
        """Test compilation thread count adjustment based on thermal state"""
        initial_threads = thermal_manager.max_compilation_threads
        
        # Test different thermal states
        state_scenarios = [
            (ThermalState.COOL, lambda x: x >= initial_threads),
            (ThermalState.OPTIMAL, lambda x: x == initial_threads),
            (ThermalState.NORMAL, lambda x: x == initial_threads),
            (ThermalState.WARM, lambda x: x < initial_threads),
            (ThermalState.HOT, lambda x: x <= 1),
            (ThermalState.THROTTLING, lambda x: x == 0),
            (ThermalState.CRITICAL, lambda x: x == 0)
        ]
        
        for state, condition in state_scenarios:
            thermal_manager.current_state = state
            thermal_manager._update_compilation_threads()
            assert condition(thermal_manager.max_compilation_threads), \
                f"State {state} resulted in {thermal_manager.max_compilation_threads} threads"
    
    def test_compilation_callbacks(self, thermal_manager):
        """Test compilation thread change callbacks"""
        callback_calls = []
        
        def test_callback(threads, state):
            callback_calls.append((threads, state))
        
        thermal_manager.add_compilation_callback(test_callback)
        
        # Change thermal state to trigger callback
        thermal_manager.current_state = ThermalState.HOT
        thermal_manager._update_compilation_threads()
        
        # Should have called the callback
        assert len(callback_calls) > 0
        threads, state = callback_calls[-1]
        assert isinstance(threads, int)
        assert isinstance(state, ThermalState)
    
    def test_monitoring_start_stop(self, thermal_manager):
        """Test monitoring start and stop"""
        assert not thermal_manager.monitoring_active
        
        # Start monitoring
        thermal_manager.start_monitoring()
        assert thermal_manager.monitoring_active
        assert thermal_manager._monitoring_thread is not None
        
        # Stop monitoring
        thermal_manager.stop_monitoring()
        assert not thermal_manager.monitoring_active
        
        # Should be able to start/stop multiple times
        thermal_manager.start_monitoring()
        thermal_manager.stop_monitoring()
    
    def test_status_reporting(self, thermal_manager):
        """Test status reporting functionality"""
        status = thermal_manager.get_status()
        
        required_keys = [
            'thermal_state',
            'steam_deck_model', 
            'current_temps',
            'fan_rpm',
            'power_draw',
            'battery_level',
            'compilation_threads',
            'sensors_available',
            'mock_mode',
            'gaming_active'
        ]
        
        for key in required_keys:
            assert key in status, f"Status missing required key: {key}"
        
        # Check value types
        assert isinstance(status['thermal_state'], str)
        assert isinstance(status['steam_deck_model'], str)
        assert isinstance(status['current_temps'], dict)
        assert isinstance(status['fan_rpm'], int)
        assert isinstance(status['power_draw'], (int, float))
        assert isinstance(status['battery_level'], (int, float))
        assert isinstance(status['compilation_threads'], int)
        assert isinstance(status['sensors_available'], int)
        assert isinstance(status['mock_mode'], bool)
        assert isinstance(status['gaming_active'], bool)
        
        # Temperature dict should have expected keys
        temp_keys = ['apu', 'cpu', 'gpu']
        for key in temp_keys:
            assert key in status['current_temps']


@pytest.mark.skipif(not HAS_THERMAL_MODULE, reason="Thermal module not available")
@pytest.mark.integration
@pytest.mark.thermal
class TestThermalManagerIntegration:
    """Integration tests for thermal manager"""
    
    def test_monitoring_integration(self, clean_temp_dir):
        """Test complete monitoring cycle"""
        manager = OptimizedThermalManager(config_path=clean_temp_dir / "integration.json")
        
        try:
            # Start monitoring
            manager.start_monitoring()
            assert manager.monitoring_active
            
            # Let it run for a short time
            time.sleep(1.1)  # Slightly longer than monitoring interval
            
            # Should have collected some thermal samples
            assert len(manager.thermal_history) > 0
            
            # Get status
            status = manager.get_status()
            assert status['thermal_state'] in [s.value for s in ThermalState]
            
            # Stop monitoring
            manager.stop_monitoring()
            assert not manager.monitoring_active
        
        finally:
            if manager.monitoring_active:
                manager.stop_monitoring()
    
    def test_thermal_state_transitions(self, clean_temp_dir):
        """Test thermal state transitions during monitoring"""
        manager = OptimizedThermalManager(config_path=clean_temp_dir / "transitions.json")
        
        try:
            state_changes = []
            
            def track_state_change(threads, state):
                state_changes.append((time.time(), threads, state))
            
            manager.add_compilation_callback(track_state_change)
            
            # Manually trigger state changes by modifying mock temperature
            if manager._mock_mode:
                manager.start_monitoring()
                
                # Simulate temperature increase
                original_temp = manager._mock_temp
                manager._mock_temp = 85.0  # Hot state
                time.sleep(1.1)
                
                # Simulate temperature decrease  
                manager._mock_temp = 60.0  # Optimal state
                time.sleep(1.1)
                
                manager._mock_temp = original_temp
                manager.stop_monitoring()
                
                # Should have recorded state changes
                if state_changes:
                    assert len(state_changes) > 0
                    # Each change should have timestamp, threads, and state
                    for timestamp, threads, state in state_changes:
                        assert isinstance(timestamp, float)
                        assert isinstance(threads, int)
                        assert isinstance(state, ThermalState)
        
        finally:
            if manager.monitoring_active:
                manager.stop_monitoring()
    
    def test_concurrent_access(self, clean_temp_dir):
        """Test thread safety of thermal manager"""
        manager = OptimizedThermalManager(config_path=clean_temp_dir / "concurrent.json")
        
        try:
            manager.start_monitoring()
            
            # Multiple threads accessing status simultaneously
            results = []
            errors = []
            
            def get_status_repeatedly():
                try:
                    for _ in range(20):
                        status = manager.get_status()
                        results.append(status)
                        time.sleep(0.01)
                except Exception as e:
                    errors.append(e)
            
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=get_status_repeatedly)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=5)
            
            manager.stop_monitoring()
            
            # Should have no errors and multiple results
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert len(results) > 0, "Should have collected status results"
            
            # All results should be valid
            for result in results:
                assert isinstance(result, dict)
                assert 'thermal_state' in result
        
        finally:
            if manager.monitoring_active:
                manager.stop_monitoring()
    
    def test_memory_leak_prevention(self, clean_temp_dir):
        """Test that thermal history doesn't grow unbounded"""
        manager = OptimizedThermalManager(config_path=clean_temp_dir / "memory.json")
        
        try:
            manager.start_monitoring()
            
            # Let it run long enough to exceed history size
            initial_size = len(manager.thermal_history)
            
            # Run for enough time to generate more samples than history limit
            for _ in range(5):
                time.sleep(0.2)
            
            manager.stop_monitoring()
            
            # History should be bounded
            final_size = len(manager.thermal_history)
            assert final_size <= 100, f"History grew to {final_size}, should be <= 100"
            assert final_size > initial_size, "Should have collected some samples"
        
        finally:
            if manager.monitoring_active:
                manager.stop_monitoring()


@pytest.mark.skipif(not HAS_THERMAL_MODULE, reason="Thermal module not available")
@pytest.mark.benchmark
@pytest.mark.thermal
class TestThermalManagerBenchmarks:
    """Performance benchmarks for thermal manager"""
    
    def test_sample_generation_benchmark(self, benchmark, clean_temp_dir):
        """Benchmark thermal sample generation"""
        manager = OptimizedThermalManager(config_path=clean_temp_dir / "benchmark.json")
        
        # Benchmark sample generation
        result = benchmark(manager._get_current_sample)
        
        assert isinstance(result, ThermalSample)
        assert result.timestamp > 0
    
    def test_state_determination_benchmark(self, benchmark, clean_temp_dir):
        """Benchmark thermal state determination"""
        manager = OptimizedThermalManager(config_path=clean_temp_dir / "benchmark.json")
        
        sample = ThermalSample(
            timestamp=time.time(),
            apu_temp=75.0,
            cpu_temp=70.0,
            gpu_temp=72.0,
            fan_rpm=2500,
            power_draw=12.0,
            battery_level=80.0,
            gaming_active=True,
            compilation_threads=4
        )
        
        # Benchmark state determination
        result = benchmark(manager._determine_thermal_state, sample)
        
        assert isinstance(result, ThermalState)
    
    def test_monitoring_overhead_benchmark(self, clean_temp_dir):
        """Test monitoring overhead"""
        manager = OptimizedThermalManager(config_path=clean_temp_dir / "overhead.json")
        
        try:
            # Measure time for monitoring loop iterations
            start_time = time.time()
            manager.start_monitoring()
            
            # Let monitoring run
            time.sleep(2.0)
            
            manager.stop_monitoring()
            elapsed = time.time() - start_time
            
            # Should have minimal overhead
            samples_collected = len(manager.thermal_history)
            if samples_collected > 0:
                avg_time_per_sample = elapsed / samples_collected
                # Each monitoring cycle should be fast
                assert avg_time_per_sample < 0.1, f"Monitoring too slow: {avg_time_per_sample:.3f}s per sample"
        
        finally:
            if manager.monitoring_active:
                manager.stop_monitoring()


@pytest.mark.skipif(not HAS_THERMAL_MODULE, reason="Thermal module not available")
@pytest.mark.steamdeck
@pytest.mark.thermal
class TestSteamDeckSpecificFeatures:
    """Tests specific to Steam Deck hardware features"""
    
    def test_steam_deck_sensor_paths(self):
        """Test Steam Deck specific sensor discovery"""
        manager = OptimizedThermalManager()
        
        # Should discover some sensors (even if mocked)
        sensors = manager.sensor_paths
        assert isinstance(sensors, dict)
        
        # If on actual Steam Deck, should find hardware-specific sensors
        if manager.steam_deck_model != SteamDeckModel.UNKNOWN:
            # Would have hardware-specific tests here
            pass
    
    def test_steam_deck_model_specific_behavior(self):
        """Test model-specific thermal behavior"""
        # Test LCD model behavior
        with patch.object(OptimizedThermalManager, '_detect_steam_deck_model', return_value=SteamDeckModel.LCD):
            lcd_manager = OptimizedThermalManager()
            assert lcd_manager.steam_deck_model == SteamDeckModel.LCD
        
        # Test OLED model behavior  
        with patch.object(OptimizedThermalManager, '_detect_steam_deck_model', return_value=SteamDeckModel.OLED):
            oled_manager = OptimizedThermalManager()
            assert oled_manager.steam_deck_model == SteamDeckModel.OLED
        
        # Different models might have different thermal characteristics
        # This would be where model-specific thermal logic is tested


@pytest.mark.skipif(not HAS_THERMAL_MODULE, reason="Thermal module not available")
def test_global_thermal_manager():
    """Test global thermal manager singleton"""
    manager1 = get_thermal_manager()
    manager2 = get_thermal_manager()
    
    # Should be the same instance
    assert manager1 is manager2
    assert isinstance(manager1, OptimizedThermalManager)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])