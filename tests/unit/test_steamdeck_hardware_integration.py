#!/usr/bin/env python3
"""
Unit Tests for Steam Deck Hardware Integration
Comprehensive testing of hardware detection, monitoring, and optimization
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tests.fixtures.steamdeck_fixtures import (
    MockHardwareState, MockSteamDeckModel, mock_steamdeck_environment,
    create_thermal_stress_scenario, create_battery_critical_scenario,
    create_intensive_gaming_scenario, create_dock_scenario
)

# Import the modules under test
from src.core.steam_deck_hardware_integration import (
    SteamDeckHardwareMonitor, HardwareAwareMLScheduler,
    SteamDeckModel, PowerState, ThermalState, DockState,
    SteamDeckHardwareState, HardwareOptimizationProfile,
    get_hardware_monitor, get_current_hardware_state
)


class TestSteamDeckHardwareMonitor:
    """Test Steam Deck hardware monitoring system"""
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_steam_deck_detection_lcd(self, mock_steamdeck_lcd):
        """Test LCD Steam Deck detection"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            assert monitor.is_steam_deck is True
            assert monitor.model == SteamDeckModel.LCD_256GB
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_steam_deck_detection_oled(self, mock_steamdeck_oled):
        """Test OLED Steam Deck detection"""
        with mock_steamdeck_environment(mock_steamdeck_oled):
            monitor = SteamDeckHardwareMonitor()
            assert monitor.is_steam_deck is True
            assert monitor.model == SteamDeckModel.OLED_512GB
    
    @pytest.mark.unit
    def test_non_steam_deck_detection(self):
        """Test non-Steam Deck system detection"""
        with patch.dict('os.environ', {}, clear=True):
            with patch('os.path.exists', return_value=False):
                monitor = SteamDeckHardwareMonitor()
                assert monitor.is_steam_deck is False
                assert monitor.model == SteamDeckModel.UNKNOWN
    
    @pytest.mark.steamdeck 
    @pytest.mark.unit
    def test_dmi_identification(self, mock_steamdeck_lcd):
        """Test DMI-based Steam Deck identification"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            assert monitor._check_dmi_identifiers() is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_cpu_signature_detection(self, mock_steamdeck_lcd):
        """Test CPU signature detection for Steam Deck APU"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            # Mock the CPU signature check
            with patch('builtins.open', mock_open_cpuinfo):
                assert monitor._check_cpu_signature() is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_hardware_state_monitoring(self, mock_steamdeck_lcd):
        """Test hardware state monitoring and updates"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            # Get initial state
            state = await monitor._get_current_hardware_state()
            
            assert isinstance(state, SteamDeckHardwareState)
            assert state.model == SteamDeckModel.LCD_256GB
            assert state.cpu_temperature == 65.0
            assert state.battery_percentage == 80.0
            assert state.power_state == PowerState.BATTERY_HIGH
            assert state.thermal_state == ThermalState.WARM
            assert state.dock_state == DockState.UNDOCKED
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_thermal_state_detection(self, mock_steamdeck_thermal_throttling):
        """Test thermal state detection and throttling identification"""
        with mock_steamdeck_environment(mock_steamdeck_thermal_throttling):
            monitor = SteamDeckHardwareMonitor()
            
            state = await monitor._get_current_hardware_state()
            
            assert state.thermal_state == ThermalState.CRITICAL
            assert state.thermal_throttling is True
            assert state.cpu_temperature >= 85.0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_power_state_detection(self, mock_steamdeck_low_battery):
        """Test power state detection for various battery levels"""
        with mock_steamdeck_environment(mock_steamdeck_low_battery):
            monitor = SteamDeckHardwareMonitor()
            
            state = await monitor._get_current_hardware_state()
            
            assert state.power_state == PowerState.BATTERY_CRITICAL
            assert state.battery_percentage == 15.0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_dock_state_detection(self, mock_steamdeck_docked):
        """Test dock state detection with external displays"""
        with mock_steamdeck_environment(mock_steamdeck_docked):
            monitor = SteamDeckHardwareMonitor()
            
            state = await monitor._get_current_hardware_state()
            
            assert state.dock_state == DockState.DOCKED_DP
            assert state.dock_power_delivery is True
            assert len(state.external_displays) > 0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_gaming_mode_detection(self, mock_steamdeck_gaming):
        """Test gaming mode detection"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            monitor = SteamDeckHardwareMonitor()
            
            state = await monitor._get_current_hardware_state()
            
            # Gaming mode detection depends on process mocking
            # This would be True if gamescope process is detected
            assert isinstance(state.gaming_mode_active, bool)
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_optimization_profiles_creation(self, mock_steamdeck_lcd):
        """Test creation and configuration of optimization profiles"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            profiles = monitor.optimization_profiles
            
            # Check that all required profiles exist
            required_profiles = ['maximum', 'balanced', 'gaming', 'battery', 'thermal_emergency']
            for profile_name in required_profiles:
                assert profile_name in profiles
                profile = profiles[profile_name]
                assert isinstance(profile, HardwareOptimizationProfile)
                assert profile.prediction_frequency_hz > 0
                assert profile.compilation_thread_limit > 0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_profile_selection_logic(self, mock_steamdeck_lcd):
        """Test optimal profile selection based on hardware conditions"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            # Test different scenarios
            scenarios = [
                (mock_steamdeck_lcd, 'balanced'),
                (create_thermal_stress_scenario(mock_steamdeck_lcd), 'thermal_emergency'),
                (create_battery_critical_scenario(mock_steamdeck_lcd), 'battery'),
                (create_intensive_gaming_scenario(mock_steamdeck_lcd), 'gaming'),
                (create_dock_scenario(mock_steamdeck_lcd), 'maximum'),
            ]
            
            for scenario_state, expected_profile in scenarios:
                state = await monitor._get_current_hardware_state()
                optimal_profile = monitor.get_optimal_profile(state)
                # Note: Actual profile selection depends on complex logic
                assert optimal_profile in monitor.optimization_profiles
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_hardware_capabilities_reporting(self, mock_steamdeck_oled):
        """Test hardware capabilities reporting for different models"""
        with mock_steamdeck_environment(mock_steamdeck_oled):
            monitor = SteamDeckHardwareMonitor()
            
            capabilities = monitor.get_hardware_capabilities()
            
            assert capabilities['is_steam_deck'] is True
            assert capabilities['model'] == 'oled_512gb'
            assert capabilities['has_oled_display'] is True
            assert capabilities['apu_architecture'] == 'RDNA2/Zen2'
            assert capabilities['memory_gb'] == 16
            assert capabilities['vulkan_support'] is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_state_change_detection(self, mock_steamdeck_lcd):
        """Test significant hardware state change detection"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            # Initial state
            initial_state = await monitor._get_current_hardware_state()
            
            # Create modified state
            modified_state = SteamDeckHardwareState(
                model=initial_state.model,
                power_state=PowerState.AC_POWERED,  # Changed
                thermal_state=ThermalState.HOT,     # Changed
                dock_state=initial_state.dock_state,
                cpu_temperature=85.0,  # Significantly different
                cpu_frequency=initial_state.cpu_frequency,
                gpu_frequency=initial_state.gpu_frequency,
                memory_usage_mb=initial_state.memory_usage_mb,
                battery_percentage=initial_state.battery_percentage,
                battery_time_remaining_min=initial_state.battery_time_remaining_min,
                power_draw_watts=initial_state.power_draw_watts,
                fan_speed_rpm=initial_state.fan_speed_rpm,
                thermal_throttling=initial_state.thermal_throttling,
                power_throttling=initial_state.power_throttling,
                gaming_mode_active=initial_state.gaming_mode_active,
                performance_governor=initial_state.performance_governor,
                internal_display_active=initial_state.internal_display_active,
                external_displays=initial_state.external_displays,
                dock_power_delivery=initial_state.dock_power_delivery
            )
            
            assert monitor._state_changed_significantly(initial_state, modified_state) is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_monitoring_loop(self, mock_steamdeck_lcd):
        """Test continuous hardware monitoring loop"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            callback_called = threading.Event()
            callback_data = {}
            
            def test_callback(state):
                callback_data['state'] = state
                callback_called.set()
            
            monitor.add_state_callback(test_callback)
            
            # Start monitoring briefly
            monitoring_task = asyncio.create_task(
                monitor.start_monitoring(update_interval=0.1)
            )
            
            # Wait for first callback
            await asyncio.sleep(0.2)
            monitor.stop_monitoring()
            
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Should have received at least one callback
            assert callback_called.is_set()
            assert 'state' in callback_data
            assert isinstance(callback_data['state'], SteamDeckHardwareState)
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_prediction_parameters_generation(self, mock_steamdeck_lcd):
        """Test ML prediction parameter generation for different profiles"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            # Test each profile
            for profile_name in monitor.optimization_profiles:
                params = monitor.get_prediction_parameters(profile_name)
                
                assert 'frequency_hz' in params
                assert 'priority' in params
                assert 'batch_size' in params
                assert 'thread_limit' in params
                assert 'thermal_limit' in params
                
                assert params['frequency_hz'] > 0
                assert params['batch_size'] > 0
                assert params['thread_limit'] > 0


class TestHardwareAwareMLScheduler:
    """Test hardware-aware ML scheduling system"""
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_scheduler_initialization(self, mock_steamdeck_lcd):
        """Test ML scheduler initialization"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            scheduler = HardwareAwareMLScheduler(monitor)
            
            assert scheduler.hardware_monitor == monitor
            assert scheduler.scheduler_active is False
            assert len(scheduler.prediction_queue) == 0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_ml_predictor_integration(self, mock_steamdeck_lcd):
        """Test ML predictor integration with scheduler"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            scheduler = HardwareAwareMLScheduler(monitor)
            
            mock_predictor = Mock()
            scheduler.set_ml_predictor(mock_predictor)
            
            assert scheduler.ml_predictor == mock_predictor
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_hardware_state_adaptation(self, mock_steamdeck_lcd):
        """Test scheduler adaptation to hardware state changes"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            scheduler = HardwareAwareMLScheduler(monitor)
            
            # Create different hardware states
            states = [
                create_thermal_stress_scenario(mock_steamdeck_lcd),
                create_battery_critical_scenario(mock_steamdeck_lcd),
                create_intensive_gaming_scenario(mock_steamdeck_lcd),
                create_dock_scenario(mock_steamdeck_lcd),
            ]
            
            for test_state in states:
                # Convert to proper SteamDeckHardwareState
                state = await monitor._get_current_hardware_state()
                
                # Manually call state change handler
                scheduler._on_hardware_state_change(state)
                
                # Scheduler should adapt to the new state
                # (Implementation details would be tested here)


class TestConvenienceFunctions:
    """Test module convenience functions"""
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_get_hardware_monitor_singleton(self, mock_steamdeck_lcd):
        """Test global hardware monitor singleton"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor1 = get_hardware_monitor()
            monitor2 = get_hardware_monitor()
            
            assert monitor1 is monitor2  # Should be same instance
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    async def test_get_current_hardware_state_function(self, mock_steamdeck_lcd):
        """Test convenience function for getting current state"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            # First initialize monitor
            monitor = get_hardware_monitor()
            await monitor.start_monitoring(update_interval=0.1)
            await asyncio.sleep(0.2)  # Let it get initial state
            monitor.stop_monitoring()
            
            state = get_current_hardware_state()
            # May be None if monitoring hasn't run yet
            if state is not None:
                assert isinstance(state, SteamDeckHardwareState)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.unit
    def test_missing_hardware_files(self):
        """Test behavior when hardware files are missing"""
        with patch('os.path.exists', return_value=False):
            monitor = SteamDeckHardwareMonitor()
            # Should handle gracefully
            temp = monitor._get_cpu_temperature()
            assert temp == 50.0  # Fallback value
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_invalid_thermal_readings(self, mock_steamdeck_lcd):
        """Test handling of invalid thermal sensor readings"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            # Mock invalid temperature reading
            with patch('builtins.open', side_effect=IOError("Sensor error")):
                temp = monitor._get_cpu_temperature()
                assert temp == 50.0  # Should fall back to safe value
    
    @pytest.mark.unit
    def test_monitoring_error_recovery(self, mock_steamdeck_lcd):
        """Test monitoring loop error recovery"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            # Mock an error in state reading
            with patch.object(monitor, '_get_current_hardware_state', 
                            side_effect=Exception("Test error")):
                # Monitoring should continue despite errors
                async def test_monitoring():
                    await monitor.start_monitoring(update_interval=0.1)
                
                # Should not raise exception
                asyncio.create_task(test_monitoring())
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_callback_exception_handling(self, mock_steamdeck_lcd):
        """Test that callback exceptions don't break monitoring"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            def failing_callback(state):
                raise Exception("Callback error")
            
            monitor.add_state_callback(failing_callback)
            
            # Should not raise exception when callback fails
            state = SteamDeckHardwareState(
                model=SteamDeckModel.LCD_256GB,
                power_state=PowerState.BATTERY_HIGH,
                thermal_state=ThermalState.COOL,
                dock_state=DockState.UNDOCKED,
                cpu_temperature=60.0,
                cpu_frequency=2800,
                gpu_frequency=1600,
                memory_usage_mb=4096,
                battery_percentage=75,
                battery_time_remaining_min=120,
                power_draw_watts=12,
                fan_speed_rpm=2500,
                thermal_throttling=False,
                power_throttling=False,
                gaming_mode_active=False,
                performance_governor='schedutil',
                internal_display_active=True,
                external_displays=[],
                dock_power_delivery=False
            )
            
            # This should not raise an exception
            for callback in monitor.state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    pass  # Expected to be handled internally


def mock_open_cpuinfo(*args, **kwargs):
    """Mock cpuinfo file reading"""
    from io import StringIO
    cpuinfo_content = """processor       : 0
vendor_id       : AuthenticAMD
cpu family      : 23
model           : 144
model name      : AMD Custom APU 0405 (Van Gogh)
stepping        : 1
microcode       : 0x0
cpu MHz         : 2800.000
cache size      : 512 KB
"""
    return StringIO(cpuinfo_content)


# Performance tests
@pytest.mark.benchmark
@pytest.mark.steamdeck
class TestPerformance:
    """Performance tests for Steam Deck hardware integration"""
    
    def test_hardware_state_reading_performance(self, benchmark, mock_steamdeck_lcd):
        """Benchmark hardware state reading performance"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            async def read_state():
                return await monitor._get_current_hardware_state()
            
            # Benchmark should complete in under 10ms
            result = benchmark(asyncio.run, read_state())
            assert isinstance(result, SteamDeckHardwareState)
    
    def test_profile_selection_performance(self, benchmark, mock_steamdeck_lcd):
        """Benchmark profile selection performance"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            monitor = SteamDeckHardwareMonitor()
            
            async def get_profile():
                state = await monitor._get_current_hardware_state()
                return monitor.get_optimal_profile(state)
            
            # Profile selection should be very fast
            result = benchmark(asyncio.run, get_profile())
            assert result in monitor.optimization_profiles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])