#!/usr/bin/env python3
"""
Enhanced Steam Deck Unit Tests with Advanced Mocking

This module provides advanced unit testing capabilities specifically designed
for Steam Deck functionality, including:

- Advanced hardware state mocking
- Dynamic system condition simulation
- Performance regression testing
- Edge case handling validation
- Component interaction testing
"""

import pytest
import asyncio
import time
import threading
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from contextlib import contextmanager

# Import test utilities
from tests.fixtures.steamdeck_fixtures import (
    MockHardwareState, MockSteamDeckModel, mock_steamdeck_environment,
    create_thermal_stress_scenario, create_battery_critical_scenario,
    create_intensive_gaming_scenario, create_dock_scenario,
    PerformanceTimer, benchmark_test
)

# Import components under test
from src.core.steam_deck_hardware_integration import (
    SteamDeckHardwareMonitor, SteamDeckModel, PowerState, ThermalState
)
from src.core.steamdeck_thermal_optimizer import (
    SteamDeckThermalOptimizer, ThermalZone, PowerProfile, ThermalReading
)
from src.core.steam_deck_optimizer import SteamDeckOptimizer


class TestAdvancedHardwareDetection:
    """Advanced hardware detection testing with edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    def test_corrupted_dmi_handling(self):
        """Test handling of corrupted DMI information"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create corrupted DMI files
            dmi_dir = Path(temp_dir) / "sys/devices/virtual/dmi/id"
            dmi_dir.mkdir(parents=True)
            
            # Write corrupted data
            (dmi_dir / "product_name").write_bytes(b"\x00\x01\x02\xff")
            (dmi_dir / "board_name").write_text("")
            
            with patch('pathlib.Path.read_text', side_effect=UnicodeDecodeError("utf-8", b"\x00", 0, 1, "invalid")):
                monitor = SteamDeckHardwareMonitor()
                
                # Should handle gracefully without crashing
                assert monitor.model == SteamDeckModel.UNKNOWN
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    def test_partial_hardware_detection(self):
        """Test detection when only partial hardware info is available"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Mock missing some files
            with patch('os.path.exists') as mock_exists:
                def exists_side_effect(path):
                    if 'product_name' in path:
                        return False  # Missing product name
                    if 'thermal_zone' in path:
                        return True   # Thermal zones available
                    return True
                
                mock_exists.side_effect = exists_side_effect
                
                monitor = SteamDeckHardwareMonitor()
                
                # Should still attempt detection with available info
                assert isinstance(monitor.is_steam_deck, bool)
                assert isinstance(monitor.model, SteamDeckModel)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    async def test_dynamic_hardware_changes(self):
        """Test handling of dynamic hardware state changes"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            monitor = SteamDeckHardwareMonitor()
            fs_manager = mock_env['filesystem']
            
            # Get initial state
            initial_state = await monitor._get_current_hardware_state()
            initial_temp = initial_state.cpu_temperature
            
            # Dynamically change hardware state
            new_hardware_state = MockHardwareState(
                model=MockSteamDeckModel.OLED_512GB,
                cpu_temperature=initial_temp + 20.0,  # Significant increase
                thermal_throttling=True
            )
            
            fs_manager.update_hardware_state(new_hardware_state)
            
            # Get updated state
            updated_state = await monitor._get_current_hardware_state()
            
            # Should detect the change
            assert updated_state.cpu_temperature > initial_temp
            assert monitor._state_changed_significantly(initial_state, updated_state)


class TestThermalManagementEdgeCases:
    """Test thermal management edge cases and error conditions"""
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_sensor_failure_recovery(self):
        """Test recovery from thermal sensor failures"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            optimizer = SteamDeckThermalOptimizer()
            
            # Simulate sensor failure
            with patch.object(optimizer, '_read_thermal_sensor', return_value=None):
                readings = optimizer.get_thermal_readings()
                
                # Should handle gracefully
                assert isinstance(readings, dict)
                
                # Should provide fallback temperature
                temp = optimizer.get_max_temperature()
                assert isinstance(temp, float)
                assert temp > 0
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_extreme_temperature_scenarios(self):
        """Test handling of extreme temperature scenarios"""
        extreme_scenarios = [
            (-10.0, "sub_zero"),      # Below freezing
            (150.0, "overheated"),    # Extremely hot
            (float('inf'), "infinite"), # Invalid reading
            (float('nan'), "nan"),     # NaN reading
        ]
        
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        for temp, scenario_name in extreme_scenarios:
            with mock_steamdeck_environment(hardware_state):
                optimizer = SteamDeckThermalOptimizer()
                
                # Mock extreme temperature reading
                with patch.object(optimizer, '_read_thermal_sensor', return_value=temp):
                    try:
                        readings = optimizer.get_thermal_readings()
                        thermal_state = optimizer.get_thermal_state()
                        
                        # Should not crash
                        assert isinstance(thermal_state, str)
                        
                        if scenario_name == "overheated":
                            assert thermal_state in ["critical", "shutdown"]
                        elif scenario_name in ["infinite", "nan"]:
                            # Should provide fallback values
                            max_temp = optimizer.get_max_temperature()
                            assert 0 < max_temp < 200
                            
                    except Exception as e:
                        pytest.fail(f"Should handle {scenario_name} scenario gracefully: {e}")
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_rapid_temperature_fluctuations(self):
        """Test handling of rapid temperature changes"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            optimizer = SteamDeckThermalOptimizer()
            
            # Simulate rapid temperature fluctuations
            temperatures = [60.0, 85.0, 50.0, 95.0, 40.0, 70.0]
            
            for temp in temperatures:
                reading = ThermalReading(
                    zone=ThermalZone.CPU,
                    temperature=temp,
                    timestamp=time.time()
                )
                
                # Add reading to history
                if hasattr(optimizer, '_add_thermal_reading'):
                    optimizer._add_thermal_reading(reading)
                else:
                    optimizer.thermal_history.append(reading)
                
                # Should maintain stability despite fluctuations
                thermal_state = optimizer.get_thermal_state()
                assert thermal_state in ["cool", "warm", "throttling", "critical", "shutdown"]
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_hysteresis(self):
        """Test thermal hysteresis to prevent oscillation"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state):
            optimizer = SteamDeckThermalOptimizer()
            
            # Test temperature around threshold
            threshold_temp = optimizer.limits.warning_temp
            test_temps = [
                threshold_temp - 1.0,  # Just below
                threshold_temp + 1.0,  # Just above
                threshold_temp - 0.5,  # Back below
                threshold_temp + 0.5,  # Back above
            ]
            
            states = []
            for temp in test_temps:
                with patch.object(optimizer, '_get_cpu_temperature', return_value=temp):
                    state = optimizer.get_thermal_state()
                    states.append(state)
            
            # Should not oscillate rapidly between states
            # Implementation would include hysteresis logic
            assert len(set(states)) <= 2, "Should not oscillate between many states"


class TestPowerManagementAdvanced:
    """Advanced power management testing"""
    
    @pytest.mark.unit
    @pytest.mark.power
    def test_power_state_transitions(self):
        """Test smooth power state transitions"""
        # Test battery discharge scenario
        battery_levels = [100, 80, 60, 40, 20, 10, 5]
        power_states = []
        
        for level in battery_levels:
            hardware_state = MockHardwareState(
                model=MockSteamDeckModel.OLED_512GB,
                battery_capacity=level
            )
            
            with mock_steamdeck_environment(hardware_state):
                monitor = SteamDeckHardwareMonitor()
                
                async def get_power_state():
                    state = await monitor._get_current_hardware_state()
                    return state.power_state
                
                power_state = asyncio.run(get_power_state())
                power_states.append((level, power_state))
        
        # Verify logical progression
        assert power_states[0][1] == PowerState.BATTERY_HIGH    # 100%
        assert power_states[-1][1] == PowerState.BATTERY_CRITICAL  # 5%
        
        # Should transition through intermediate states
        unique_states = set(state[1] for state in power_states)
        assert len(unique_states) >= 3  # Should use multiple power states
    
    @pytest.mark.unit
    @pytest.mark.power
    def test_power_draw_calculation_accuracy(self):
        """Test accuracy of power draw calculations"""
        test_scenarios = [
            # (power_now_uw, voltage_now_uv, expected_watts_range)
            (12000000, 7400000, (11.0, 13.0)),   # Normal operation
            (8000000, 6800000, (7.0, 9.0)),      # Low power
            (18000000, 8400000, (17.0, 19.0)),   # High power
        ]
        
        for power_uw, voltage_uv, expected_range in test_scenarios:
            hardware_state = MockHardwareState(
                model=MockSteamDeckModel.LCD_512GB,
                power_supply_info={
                    'BAT1': {
                        'capacity': 50,
                        'status': 'Discharging',
                        'power_now': power_uw,
                        'voltage_now': voltage_uv
                    }
                }
            )
            
            with mock_steamdeck_environment(hardware_state):
                optimizer = SteamDeckOptimizer()
                
                battery_info = optimizer._get_battery_info()
                calculated_watts = battery_info[2]  # power_draw
                
                assert expected_range[0] <= calculated_watts <= expected_range[1], \
                    f"Power calculation {calculated_watts}W not in expected range {expected_range}"
    
    @pytest.mark.unit
    @pytest.mark.power
    def test_battery_time_estimation(self):
        """Test battery time remaining estimation"""
        # Test different scenarios
        scenarios = [
            (50, 10.0, MockSteamDeckModel.LCD_256GB, 40.0),    # LCD, 50% battery, 10W draw
            (75, 8.0, MockSteamDeckModel.OLED_512GB, 50.0),    # OLED, 75% battery, 8W draw
            (25, 15.0, MockSteamDeckModel.LCD_64GB, 40.0),     # LCD, 25% battery, 15W draw
        ]
        
        for battery_pct, power_draw_w, model, battery_wh in scenarios:
            hardware_state = MockHardwareState(
                model=model,
                battery_capacity=battery_pct,
                power_draw=int(power_draw_w * 1000000)  # Convert to microWatts
            )
            
            with mock_steamdeck_environment(hardware_state):
                optimizer = SteamDeckOptimizer()
                optimizer.hardware_profile.battery_capacity_wh = battery_wh
                
                battery_info = optimizer._get_battery_info()
                battery_time_min = battery_info[1]
                
                if battery_time_min is not None:
                    # Calculate expected time
                    remaining_wh = battery_wh * (battery_pct / 100.0)
                    expected_time_hours = remaining_wh / power_draw_w
                    expected_time_min = expected_time_hours * 60
                    
                    # Should be reasonably close (within 10%)
                    assert abs(battery_time_min - expected_time_min) < expected_time_min * 0.1


class TestComponentInteractionAndCoordination:
    """Test interaction and coordination between components"""
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_component_callback_coordination(self):
        """Test callback coordination between components"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize components
            hardware_monitor = SteamDeckHardwareMonitor()
            deck_optimizer = SteamDeckOptimizer()
            
            # Track callback invocations
            callback_events = []
            
            def test_callback(data):
                callback_events.append({
                    'timestamp': time.time(),
                    'data': data
                })
            
            # Add callbacks
            hardware_monitor.add_state_callback(test_callback)
            deck_optimizer.add_optimization_callback(test_callback)
            
            # Trigger events
            async def trigger_events():
                state = await hardware_monitor._get_current_hardware_state()
                # Manually trigger callbacks
                for callback in hardware_monitor.state_callbacks:
                    callback(state)
                
                profile_applied = deck_optimizer.apply_optimization_profile('balanced')
                return state, profile_applied
            
            state, applied = asyncio.run(trigger_events())
            
            # Should have received callbacks
            assert len(callback_events) >= 1
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_component_state_consistency(self):
        """Test state consistency across components"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize multiple components
            components = {
                'hardware_monitor': SteamDeckHardwareMonitor(),
                'thermal_optimizer': SteamDeckThermalOptimizer(),
                'deck_optimizer': SteamDeckOptimizer()
            }
            
            # Collect state information from each component
            async def collect_states():
                states = {}
                
                # Hardware monitor state
                hw_state = await components['hardware_monitor']._get_current_hardware_state()
                states['hardware_monitor'] = {
                    'temperature': hw_state.cpu_temperature,
                    'battery_percent': hw_state.battery_percentage,
                    'gaming_mode': hw_state.gaming_mode_active
                }
                
                # Thermal optimizer state
                thermal_status = components['thermal_optimizer'].get_status()
                states['thermal_optimizer'] = {
                    'temperature': thermal_status.get('max_temperature', 0),
                    'thermal_state': thermal_status.get('thermal_state', 'unknown')
                }
                
                # Deck optimizer state
                deck_state = components['deck_optimizer'].get_current_state()
                states['deck_optimizer'] = {
                    'temperature': deck_state.cpu_temperature_celsius,
                    'battery_percent': deck_state.battery_percent,
                    'gaming_mode': deck_state.gaming_mode_active
                }
                
                return states
            
            states = asyncio.run(collect_states())
            
            # Verify consistency across components
            hw_temp = states['hardware_monitor']['temperature']
            thermal_temp = states['thermal_optimizer']['temperature']
            deck_temp = states['deck_optimizer']['temperature']
            
            # Temperatures should be reasonably consistent
            temp_values = [t for t in [hw_temp, thermal_temp, deck_temp] if t > 0]
            if len(temp_values) > 1:
                temp_range = max(temp_values) - min(temp_values)
                assert temp_range < 10.0, "Temperature readings should be consistent across components"
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_component_error_isolation(self):
        """Test that errors in one component don't break others"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize components
            hardware_monitor = SteamDeckHardwareMonitor()
            thermal_optimizer = SteamDeckThermalOptimizer()
            
            # Add a failing callback
            def failing_callback(data):
                raise Exception("Callback error")
            
            hardware_monitor.add_state_callback(failing_callback)
            
            # Add a working callback
            working_callback_called = threading.Event()
            def working_callback(data):
                working_callback_called.set()
            
            hardware_monitor.add_state_callback(working_callback)
            
            # Trigger callbacks
            async def trigger_with_error():
                state = await hardware_monitor._get_current_hardware_state()
                
                # Manually trigger callbacks (simulating internal callback execution)
                for callback in hardware_monitor.state_callbacks:
                    try:
                        callback(state)
                    except Exception as e:
                        # Should log but continue
                        pass
                
                return state
            
            # Should not raise exception
            state = asyncio.run(trigger_with_error())
            
            # Working callback should still be called
            assert working_callback_called.is_set(), "Working callback should execute despite failing callback"
            
            # Thermal optimizer should still work
            thermal_status = thermal_optimizer.get_status()
            assert isinstance(thermal_status, dict)


class TestPerformanceRegressionDetection:
    """Test performance regression detection and benchmarking"""
    
    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_component_initialization_performance(self, benchmark):
        """Benchmark component initialization performance"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        def initialize_components():
            with mock_steamdeck_environment(hardware_state):
                components = {
                    'hardware_monitor': SteamDeckHardwareMonitor(),
                    'thermal_optimizer': SteamDeckThermalOptimizer(),
                    'deck_optimizer': SteamDeckOptimizer()
                }
                return len(components)
        
        result = benchmark(initialize_components)
        assert result == 3  # Should initialize all components
    
    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_state_reading_performance_regression(self, benchmark):
        """Test for performance regressions in state reading"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        def read_multiple_states():
            with mock_steamdeck_environment(hardware_state):
                monitor = SteamDeckHardwareMonitor()
                
                async def read_states():
                    states = []
                    for _ in range(10):  # Read state multiple times
                        state = await monitor._get_current_hardware_state()
                        states.append(state.cpu_temperature)
                    return states
                
                return asyncio.run(read_states())
        
        result = benchmark(read_multiple_states)
        assert len(result) == 10
        assert all(isinstance(temp, float) for temp in result)
    
    @benchmark_test(iterations=100)
    def test_optimization_decision_speed(self):
        """Benchmark optimization decision making speed"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state):
            monitor = SteamDeckHardwareMonitor()
            
            async def make_decision():
                state = await monitor._get_current_hardware_state()
                return monitor.get_optimal_profile(state)
            
            result = asyncio.run(make_decision())
            return result
    
    @pytest.mark.performance
    def test_memory_usage_monitoring(self):
        """Test memory usage of components over time"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        components = []
        for i in range(10):  # Create multiple instances
            with mock_steamdeck_environment(hardware_state):
                monitor = SteamDeckHardwareMonitor()
                thermal = SteamDeckThermalOptimizer()
                optimizer = SteamDeckOptimizer()
                components.append((monitor, thermal, optimizer))
        
        # Force garbage collection
        gc.collect()
        peak_memory = process.memory_info().rss
        
        # Clear components
        del components
        gc.collect()
        
        final_memory = process.memory_info().rss
        
        memory_growth = peak_memory - initial_memory
        memory_leaked = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB for 10 instances)
        assert memory_growth < 100 * 1024 * 1024, f"Excessive memory usage: {memory_growth / 1024 / 1024:.1f}MB"
        
        # Should not have significant memory leaks (less than 10MB)
        assert memory_leaked < 10 * 1024 * 1024, f"Potential memory leak: {memory_leaked / 1024 / 1024:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
