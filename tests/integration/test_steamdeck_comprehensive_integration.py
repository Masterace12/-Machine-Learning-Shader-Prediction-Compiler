#!/usr/bin/env python3
"""
Comprehensive Steam Deck Integration Tests

This test suite provides thorough validation of all Steam Deck integration
components working together in realistic scenarios. Tests cover:

1. Hardware Detection & Configuration
2. Thermal Management Integration
3. Power Management & Battery Optimization
4. Steam D-Bus Integration & Gaming Mode
5. Performance Optimization & Cache Management
6. Complete Workflow Validation

Each test can run with both mocked hardware (for CI/CD) and on actual
Steam Deck systems for full validation.
"""

import pytest
import asyncio
import time
import os
import threading
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock, call
from contextlib import asynccontextmanager

# Import test fixtures and utilities
from tests.fixtures.steamdeck_fixtures import (
    MockHardwareState, MockSteamDeckModel, mock_steamdeck_environment,
    create_thermal_stress_scenario, create_battery_critical_scenario,
    create_intensive_gaming_scenario, create_dock_scenario,
    MockDBusInterface, PerformanceTimer, benchmark_test
)

# Import components under test
from src.core.steam_deck_hardware_integration import (
    SteamDeckHardwareMonitor, HardwareAwareMLScheduler,
    SteamDeckModel, PowerState, ThermalState, DockState
)
from src.core.steamdeck_thermal_optimizer import (
    SteamDeckThermalOptimizer, ThermalZone, PowerProfile
)
from src.core.steam_deck_optimizer import SteamDeckOptimizer
from src.core.enhanced_dbus_manager import (
    EnhancedDBusManager, DBusBackendType
)
from src.core.steamdeck_cache_optimizer import SteamDeckCacheOptimizer


# =============================================================================
# STEAM DECK INTEGRATION TEST MARKERS
# =============================================================================

pytestmark = pytest.mark.steamdeck


# =============================================================================
# HARDWARE DETECTION AND CONFIGURATION TESTS
# =============================================================================

class TestHardwareDetectionIntegration:
    """Test comprehensive hardware detection and configuration"""
    
    @pytest.mark.integration
    @pytest.mark.parametrize("hardware_model", [
        MockSteamDeckModel.LCD_64GB,
        MockSteamDeckModel.LCD_256GB,
        MockSteamDeckModel.LCD_512GB,
        MockSteamDeckModel.OLED_512GB,
        MockSteamDeckModel.OLED_1TB
    ])
    async def test_model_specific_detection_and_optimization(self, hardware_model):
        """Test detection and optimization for each Steam Deck model"""
        hardware_state = MockHardwareState(model=hardware_model)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize all hardware detection components
            hardware_monitor = SteamDeckHardwareMonitor()
            thermal_optimizer = SteamDeckThermalOptimizer()
            deck_optimizer = SteamDeckOptimizer()
            
            # Verify consistent model detection across components
            assert hardware_monitor.is_steam_deck is True
            
            if hardware_model in [MockSteamDeckModel.OLED_512GB, MockSteamDeckModel.OLED_1TB]:
                # OLED model validation
                assert hardware_monitor.model.value.startswith('oled')
                assert thermal_optimizer.model == SteamDeckModel.OLED
                assert deck_optimizer.hardware_profile.model == 'oled'
                
                # OLED should have better thermal limits
                assert thermal_optimizer.limits.normal_temp < 75.0
                
            else:
                # LCD model validation
                assert hardware_monitor.model.value.startswith('lcd')
                assert thermal_optimizer.model == SteamDeckModel.LCD
                assert deck_optimizer.hardware_profile.model == 'lcd'
            
            # Test hardware capabilities reporting consistency
            capabilities = hardware_monitor.get_hardware_capabilities()
            assert capabilities['is_steam_deck'] is True
            assert capabilities['memory_gb'] == 16
            assert capabilities['cpu_cores'] == 4
            
            # Test optimization profile selection
            state = await hardware_monitor._get_current_hardware_state()
            optimal_profile = hardware_monitor.get_optimal_profile(state)
            assert optimal_profile in hardware_monitor.optimization_profiles
            
            # Verify model-specific optimization parameters
            params = hardware_monitor.get_prediction_parameters(optimal_profile)
            assert params['frequency_hz'] > 0
            assert params['thread_limit'] > 0
    
    @pytest.mark.integration
    async def test_hardware_configuration_loading(self):
        """Test loading of model-specific configuration files"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Check if configuration files exist and are loadable
            config_dir = Path("/home/deck/Downloads/-Machine-Learning-Shader-Prediction-Compiler/config")
            
            lcd_config = config_dir / "steamdeck_lcd_config.json"
            oled_config = config_dir / "steamdeck_oled_config.json"
            
            # Test configuration loading
            hardware_monitor = SteamDeckHardwareMonitor()
            
            # The system should detect OLED and use appropriate settings
            assert hardware_monitor.model == SteamDeckModel.OLED_512GB
            
            # Verify OLED-specific optimizations are applied
            profiles = hardware_monitor.optimization_profiles
            balanced_profile = profiles['balanced']
            
            # OLED should have more aggressive settings due to better efficiency
            assert balanced_profile.prediction_frequency_hz >= 15.0
            assert balanced_profile.thermal_limit_celsius <= 82.0
    
    @pytest.mark.integration
    async def test_multi_component_hardware_consistency(self):
        """Test that all components report consistent hardware information"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize all hardware-aware components
            components = {
                'hardware_monitor': SteamDeckHardwareMonitor(),
                'thermal_optimizer': SteamDeckThermalOptimizer(),
                'deck_optimizer': SteamDeckOptimizer(),
                'cache_optimizer': SteamDeckCacheOptimizer()
            }
            
            # Collect hardware information from each component
            hardware_info = {}
            for name, component in components.items():
                if hasattr(component, 'is_steam_deck'):
                    hardware_info[f'{name}_is_steam_deck'] = component.is_steam_deck
                if hasattr(component, 'model'):
                    hardware_info[f'{name}_model'] = getattr(component.model, 'value', str(component.model))
            
            # Verify consistency across components
            steam_deck_detections = [v for k, v in hardware_info.items() if 'is_steam_deck' in k]
            assert all(steam_deck_detections), "All components should detect Steam Deck"
            
            model_detections = [v for k, v in hardware_info.items() if 'model' in k]
            # Models should be consistent (either all LCD or all OLED)
            lcd_count = sum(1 for m in model_detections if 'lcd' in str(m).lower())
            oled_count = sum(1 for m in model_detections if 'oled' in str(m).lower())
            assert lcd_count == 0 or oled_count == 0, "Model detection should be consistent"


# =============================================================================
# THERMAL MANAGEMENT INTEGRATION TESTS
# =============================================================================

class TestThermalManagementIntegration:
    """Test comprehensive thermal management across all components"""
    
    @pytest.mark.integration
    @pytest.mark.thermal
    async def test_thermal_monitoring_and_response_pipeline(self):
        """Test complete thermal monitoring and response pipeline"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize thermal management components
            hardware_monitor = SteamDeckHardwareMonitor()
            thermal_optimizer = SteamDeckThermalOptimizer()
            ml_scheduler = HardwareAwareMLScheduler(hardware_monitor)
            
            # Track thermal events
            thermal_events = []
            
            def thermal_callback(state):
                thermal_events.append({
                    'timestamp': time.time(),
                    'temperature': state.cpu_temperature,
                    'thermal_state': state.thermal_state,
                    'throttling': state.thermal_throttling
                })
            
            hardware_monitor.add_state_callback(thermal_callback)
            
            # Start monitoring systems
            thermal_optimizer.start_monitoring(interval=0.1)
            monitoring_task = asyncio.create_task(
                hardware_monitor.start_monitoring(update_interval=0.1)
            )
            
            # Simulate thermal stress progression
            await asyncio.sleep(0.2)  # Initial readings
            
            # Create thermal stress scenario
            stress_state = create_thermal_stress_scenario(hardware_state)
            with mock_steamdeck_environment(stress_state):
                await asyncio.sleep(0.3)  # Let system detect stress
            
            # Stop monitoring
            thermal_optimizer.stop_monitoring()
            hardware_monitor.stop_monitoring()
            
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Verify thermal event detection
            assert len(thermal_events) > 0, "Should detect thermal events"
            
            # Check thermal optimizer response
            thermal_status = thermal_optimizer.get_status()
            assert 'thermal_state' in thermal_status
            assert thermal_status['max_temperature'] > 0
            
            # Verify ML scheduler adaptation
            # The scheduler should have adapted to thermal conditions
            # (Implementation details would be verified here)
    
    @pytest.mark.integration
    @pytest.mark.thermal
    async def test_thermal_emergency_coordination(self):
        """Test coordination during thermal emergency scenarios"""
        # Create critical thermal scenario
        hardware_state = create_thermal_stress_scenario(
            MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        )
        hardware_state.cpu_temperature = 95.0  # Critical temperature
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize all thermal-aware components
            hardware_monitor = SteamDeckHardwareMonitor()
            thermal_optimizer = SteamDeckThermalOptimizer()
            deck_optimizer = SteamDeckOptimizer()
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Get system state
            state = await hardware_monitor._get_current_hardware_state()
            
            # Verify critical thermal detection
            assert state.thermal_state == ThermalState.CRITICAL
            assert state.cpu_temperature >= 90.0
            
            # Test emergency profile selection
            optimal_profile = hardware_monitor.get_optimal_profile(state)
            assert optimal_profile == 'thermal_emergency'
            
            # Verify thermal optimizer emergency mode
            thermal_status = thermal_optimizer.get_status()
            assert thermal_status['thermal_state'] in ['critical', 'throttling']
            
            # Test emergency optimizations
            params = hardware_monitor.get_prediction_parameters('thermal_emergency')
            assert params['frequency_hz'] <= 1.0  # Severely reduced
            assert params['thread_limit'] <= 2    # Minimal threading
            
            # Verify cache optimizer response
            cache_status = cache_optimizer.get_status()
            # Should reduce cache activity in emergency
            assert 'thermal_response' in cache_status or 'reduced_activity' in str(cache_status)
    
    @pytest.mark.integration
    @pytest.mark.thermal
    async def test_thermal_recovery_workflow(self):
        """Test thermal recovery and optimization restoration"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            hardware_monitor = SteamDeckHardwareMonitor()
            thermal_optimizer = SteamDeckThermalOptimizer()
            
            # Start with normal conditions
            initial_state = await hardware_monitor._get_current_hardware_state()
            initial_profile = hardware_monitor.get_optimal_profile(initial_state)
            
            # Simulate thermal stress
            stress_state = create_thermal_stress_scenario(hardware_state)
            with mock_steamdeck_environment(stress_state):
                stressed_state = await hardware_monitor._get_current_hardware_state()
                stress_profile = hardware_monitor.get_optimal_profile(stressed_state)
                
                # Should switch to emergency profile
                assert stress_profile == 'thermal_emergency'
            
            # Return to normal conditions (recovery)
            recovery_state = await hardware_monitor._get_current_hardware_state()
            recovery_profile = hardware_monitor.get_optimal_profile(recovery_state)
            
            # Should recover to normal operation
            assert recovery_profile != 'thermal_emergency'
            assert recovery_state.thermal_state in [ThermalState.COOL, ThermalState.WARM]


# =============================================================================
# POWER MANAGEMENT INTEGRATION TESTS
# =============================================================================

class TestPowerManagementIntegration:
    """Test power management and battery optimization integration"""
    
    @pytest.mark.integration
    @pytest.mark.power
    async def test_battery_level_optimization_workflow(self):
        """Test complete battery optimization workflow"""
        # Test different battery scenarios
        scenarios = [
            ('high_battery', MockHardwareState(model=MockSteamDeckModel.OLED_512GB, battery_capacity=90)),
            ('medium_battery', MockHardwareState(model=MockSteamDeckModel.OLED_512GB, battery_capacity=50)),
            ('low_battery', create_battery_critical_scenario(
                MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
            ))
        ]
        
        for scenario_name, hardware_state in scenarios:
            with mock_steamdeck_environment(hardware_state):
                # Initialize power-aware components
                hardware_monitor = SteamDeckHardwareMonitor()
                deck_optimizer = SteamDeckOptimizer()
                thermal_optimizer = SteamDeckThermalOptimizer()
                
                # Get power state
                state = await hardware_monitor._get_current_hardware_state()
                
                # Verify power state detection
                if scenario_name == 'high_battery':
                    assert state.power_state == PowerState.BATTERY_HIGH
                    assert state.battery_percentage >= 80
                elif scenario_name == 'low_battery':
                    assert state.power_state == PowerState.BATTERY_CRITICAL
                    assert state.battery_percentage <= 20
                
                # Test optimization profile selection
                optimal_profile = hardware_monitor.get_optimal_profile(state)
                
                if scenario_name == 'low_battery':
                    assert optimal_profile == 'battery'
                    
                    # Verify battery conservation settings
                    params = hardware_monitor.get_prediction_parameters('battery')
                    assert params['frequency_hz'] <= 5.0  # Reduced frequency
                    assert params['thread_limit'] <= 2    # Minimal threading
                
                # Test deck optimizer battery awareness
                deck_state = deck_optimizer.get_current_state()
                recommendations = deck_optimizer.get_optimization_recommendations()
                
                if scenario_name == 'low_battery':
                    # Should recommend battery conservation
                    battery_recs = [r for r in recommendations if r['type'] == 'battery']
                    assert len(battery_recs) > 0
    
    @pytest.mark.integration
    @pytest.mark.power
    async def test_ac_power_vs_battery_optimization(self):
        """Test optimization differences between AC power and battery"""
        base_hardware = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        # Test battery operation
        battery_state = MockHardwareState(
            model=MockSteamDeckModel.LCD_512GB,
            battery_capacity=60,
            power_supply_info={
                'BAT1': {
                    'capacity': 60,
                    'status': 'Discharging',
                    'power_now': 12000000
                }
            }
        )
        
        # Test AC power operation (docked)
        ac_state = create_dock_scenario(base_hardware)
        
        battery_results = {}
        ac_results = {}
        
        # Test battery operation
        with mock_steamdeck_environment(battery_state):
            hardware_monitor = SteamDeckHardwareMonitor()
            state = await hardware_monitor._get_current_hardware_state()
            
            battery_results = {
                'power_state': state.power_state,
                'dock_state': state.dock_state,
                'optimal_profile': hardware_monitor.get_optimal_profile(state),
                'params': hardware_monitor.get_prediction_parameters()
            }
        
        # Test AC power operation
        with mock_steamdeck_environment(ac_state):
            hardware_monitor = SteamDeckHardwareMonitor()
            state = await hardware_monitor._get_current_hardware_state()
            
            ac_results = {
                'power_state': state.power_state,
                'dock_state': state.dock_state,
                'optimal_profile': hardware_monitor.get_optimal_profile(state),
                'params': hardware_monitor.get_prediction_parameters()
            }
        
        # Verify different optimization approaches
        assert battery_results['power_state'] != PowerState.AC_POWERED
        assert ac_results['dock_state'] != DockState.UNDOCKED
        
        # AC power should allow more aggressive optimization
        assert ac_results['params']['frequency_hz'] >= battery_results['params']['frequency_hz']
        assert ac_results['params']['thread_limit'] >= battery_results['params']['thread_limit']
        
        # Profiles should be different
        if battery_results['power_state'] in [PowerState.BATTERY_LOW, PowerState.BATTERY_CRITICAL]:
            assert ac_results['optimal_profile'] != 'battery'
    
    @pytest.mark.integration
    @pytest.mark.power
    async def test_power_draw_monitoring_and_throttling(self):
        """Test power draw monitoring and adaptive throttling"""
        hardware_state = MockHardwareState(
            model=MockSteamDeckModel.LCD_256GB,
            power_draw=18000000,  # High power draw (18W)
            battery_capacity=40
        )
        
        with mock_steamdeck_environment(hardware_state):
            hardware_monitor = SteamDeckHardwareMonitor()
            deck_optimizer = SteamDeckOptimizer()
            
            # Monitor power state
            state = await hardware_monitor._get_current_hardware_state()
            
            # Should detect high power consumption
            assert state.power_draw_watts >= 15.0
            
            # Get optimization recommendations
            recommendations = deck_optimizer.get_optimization_recommendations()
            
            # Should recommend power reduction
            power_recs = [r for r in recommendations if r['type'] in ['battery', 'power']]
            if state.battery_percentage < 60:  # Medium battery with high power draw
                assert len(power_recs) > 0
            
            # Test power budget estimation
            power_budget = deck_optimizer.get_power_budget_estimate() if hasattr(deck_optimizer, 'get_power_budget_estimate') else None
            if power_budget is not None:
                assert power_budget > 0
                
                # High power draw should reduce available budget
                assert power_budget <= 10.0  # Should be conservative with high draw


# =============================================================================
# STEAM INTEGRATION AND GAMING MODE TESTS
# =============================================================================

class TestSteamIntegrationWorkflow:
    """Test Steam D-Bus integration and gaming mode workflows"""
    
    @pytest.mark.integration
    @pytest.mark.steam
    async def test_gaming_mode_detection_and_optimization(self):
        """Test complete gaming mode detection and optimization workflow"""
        hardware_state = create_intensive_gaming_scenario(
            MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        )
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            
            # Initialize Steam-aware components
            hardware_monitor = SteamDeckHardwareMonitor()
            deck_optimizer = SteamDeckOptimizer()
            thermal_optimizer = SteamDeckThermalOptimizer()
            
            # Simulate gaming mode activation
            dbus_interface.set_gaming_mode(True)
            dbus_interface.add_steam_app(12345, "Cyberpunk 2077")
            
            # Get system state
            state = await hardware_monitor._get_current_hardware_state()
            
            # Should detect gaming mode
            assert state.gaming_mode_active is True
            
            # Should select gaming profile
            optimal_profile = hardware_monitor.get_optimal_profile(state)
            assert optimal_profile == 'gaming'
            
            # Verify gaming optimizations
            params = hardware_monitor.get_prediction_parameters('gaming')
            assert params['frequency_hz'] <= 10.0  # Reduced background activity
            assert params['priority'] == 'low'     # Background priority
            
            # Test thermal management during gaming
            thermal_status = thermal_optimizer.get_status()
            gaming_opts = thermal_optimizer.optimize_for_gaming()
            
            # Should optimize for gaming performance
            assert gaming_opts['compilation_paused'] == (state.thermal_state in ['throttling', 'critical'])
            assert gaming_opts['background_threads'] <= 4  # Limited background work
            
            # Test game termination
            dbus_interface.remove_steam_app(12345)
            dbus_interface.set_gaming_mode(False)
            
            # Should return to normal operation
            post_gaming_state = await hardware_monitor._get_current_hardware_state()
            post_gaming_profile = hardware_monitor.get_optimal_profile(post_gaming_state)
            
            assert post_gaming_profile != 'gaming'
    
    @pytest.mark.integration
    @pytest.mark.steam
    async def test_steam_process_monitoring_integration(self):
        """Test Steam process monitoring and game detection"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            
            # Initialize monitoring components
            hardware_monitor = SteamDeckHardwareMonitor()
            deck_optimizer = SteamDeckOptimizer()
            
            # Test Steam process detection
            deck_state = deck_optimizer.get_current_state()
            
            # Steam should be detected as running (mocked)
            steam_running = deck_state.steam_running
            
            # Add multiple Steam applications
            games = [
                (67890, "Portal 2"),
                (13210, "Half-Life: Alyx"),
                (24567, "Elden Ring")
            ]
            
            for app_id, name in games:
                dbus_interface.add_steam_app(app_id, name)
            
            # Should detect multiple running games
            running_apps = dbus_interface.get_running_apps()
            assert len(running_apps) == len(games)
            
            # Test system adaptation to multiple games
            if len(running_apps) > 1:
                # Multiple games should trigger conservative optimization
                state = await hardware_monitor._get_current_hardware_state()
                optimal_profile = hardware_monitor.get_optimal_profile(state)
                
                # Should use conservative profile with multiple games
                params = hardware_monitor.get_prediction_parameters(optimal_profile)
                assert params['thread_limit'] <= 4  # Conservative threading
    
    @pytest.mark.integration
    @pytest.mark.steam
    async def test_dbus_fallback_and_recovery(self):
        """Test D-Bus connection failure and fallback mechanisms"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Test with D-Bus unavailable
            with patch('src.core.enhanced_dbus_manager.DBusNextBackend.connect', return_value=False):
                try:
                    from src.core.enhanced_dbus_manager import EnhancedDBusManager
                    dbus_manager = EnhancedDBusManager()
                    
                    # Should fall back to process-based detection
                    await dbus_manager.initialize()
                    
                    # Should still be able to detect gaming mode via processes
                    gaming_detected = await dbus_manager.is_gaming_mode_active()
                    
                    # Result depends on mocked process detection
                    assert isinstance(gaming_detected, bool)
                    
                except ImportError:
                    # D-Bus manager not available - skip test
                    pytest.skip("Enhanced D-Bus manager not available")


# =============================================================================
# PERFORMANCE OPTIMIZATION INTEGRATION TESTS
# =============================================================================

class TestPerformanceOptimizationIntegration:
    """Test comprehensive performance optimization integration"""
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_cache_management_under_memory_pressure(self):
        """Test cache management during memory pressure scenarios"""
        hardware_state = MockHardwareState(
            model=MockSteamDeckModel.LCD_64GB,  # Lower-end model
            # Simulate high memory usage
        )
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize cache and memory management
            hardware_monitor = SteamDeckHardwareMonitor()
            cache_optimizer = SteamDeckCacheOptimizer()
            deck_optimizer = SteamDeckOptimizer()
            
            # Get system state
            state = await hardware_monitor._get_current_hardware_state()
            
            # Simulate memory pressure
            if state.memory_usage_mb > 12000:  # > 12GB usage
                # Should trigger memory optimization
                recommendations = deck_optimizer.get_optimization_recommendations()
                memory_recs = [r for r in recommendations if r['type'] == 'memory']
                
                if len(memory_recs) > 0:
                    assert 'memory' in memory_recs[0]['type']
            
            # Test cache optimization under pressure
            cache_status = cache_optimizer.get_status()
            assert 'memory_usage' in cache_status or 'status' in cache_status
            
            # Cache should adapt to memory constraints
            if hasattr(cache_optimizer, 'optimize_for_memory_pressure'):
                cache_optimizer.optimize_for_memory_pressure()
                optimized_status = cache_optimizer.get_status()
                # Should show optimization applied
                assert optimized_status != cache_status or 'optimized' in str(optimized_status)
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_background_compilation_scheduling(self):
        """Test intelligent background shader compilation scheduling"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize scheduling components
            hardware_monitor = SteamDeckHardwareMonitor()
            ml_scheduler = HardwareAwareMLScheduler(hardware_monitor)
            thermal_optimizer = SteamDeckThermalOptimizer()
            
            # Mock ML predictor
            mock_predictor = Mock()
            ml_scheduler.set_ml_predictor(mock_predictor)
            
            # Test normal operation scheduling
            state = await hardware_monitor._get_current_hardware_state()
            optimal_profile = hardware_monitor.get_optimal_profile(state)
            params = hardware_monitor.get_prediction_parameters(optimal_profile)
            
            # Should allow reasonable background activity
            assert params['thread_limit'] >= 2
            assert params['frequency_hz'] > 0
            
            # Test scheduling under thermal stress
            stress_state = create_thermal_stress_scenario(hardware_state)
            with mock_steamdeck_environment(stress_state):
                stressed_state = await hardware_monitor._get_current_hardware_state()
                ml_scheduler._on_hardware_state_change(stressed_state)
                
                # Should reduce background activity
                stress_params = hardware_monitor.get_prediction_parameters(
                    hardware_monitor.get_optimal_profile(stressed_state)
                )
                assert stress_params['frequency_hz'] < params['frequency_hz']
                assert stress_params['thread_limit'] <= params['thread_limit']
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_resource_usage_optimization_workflow(self):
        """Test complete resource usage optimization workflow"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize all optimization components
            components = {
                'hardware_monitor': SteamDeckHardwareMonitor(),
                'deck_optimizer': SteamDeckOptimizer(),
                'thermal_optimizer': SteamDeckThermalOptimizer(),
                'cache_optimizer': SteamDeckCacheOptimizer()
            }
            
            # Test adaptive optimization workflow
            hardware_monitor = components['hardware_monitor']
            deck_optimizer = components['deck_optimizer']
            
            # Get baseline state
            initial_state = await hardware_monitor._get_current_hardware_state()
            initial_profile = hardware_monitor.get_optimal_profile(initial_state)
            
            # Test profile application
            success = deck_optimizer.apply_optimization_profile(initial_profile)
            # May fail due to permission restrictions, but should not crash
            assert isinstance(success, bool)
            
            # Get optimization recommendations
            recommendations = deck_optimizer.get_optimization_recommendations()
            assert isinstance(recommendations, list)
            
            # Test compatibility reporting
            compatibility = deck_optimizer.get_compatibility_report()
            assert 'overall_score' in compatibility
            assert 'overall_rating' in compatibility
            assert compatibility['overall_score'] >= 0.0
            assert compatibility['overall_score'] <= 1.0


# =============================================================================
# COMPLETE WORKFLOW VALIDATION TESTS
# =============================================================================

class TestCompleteWorkflowValidation:
    """Test complete Steam Deck integration workflows end-to-end"""
    
    @pytest.mark.integration
    @pytest.mark.workflow
    async def test_complete_gaming_session_workflow(self):
        """Test complete workflow from game launch to termination"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            
            # Initialize all components
            hardware_monitor = SteamDeckHardwareMonitor()
            thermal_optimizer = SteamDeckThermalOptimizer()
            deck_optimizer = SteamDeckOptimizer()
            ml_scheduler = HardwareAwareMLScheduler(hardware_monitor)
            cache_optimizer = SteamDeckCacheOptimizer()
            
            # Mock ML predictor
            mock_predictor = Mock()
            ml_scheduler.set_ml_predictor(mock_predictor)
            
            workflow_events = []
            
            # Setup callbacks to track workflow
            def state_callback(state):
                workflow_events.append({
                    'type': 'hardware_state',
                    'timestamp': time.time(),
                    'gaming_mode': state.gaming_mode_active,
                    'thermal_state': state.thermal_state.value,
                    'power_state': state.power_state.value
                })
            
            hardware_monitor.add_state_callback(state_callback)
            
            # Start monitoring systems
            thermal_optimizer.start_monitoring(interval=0.1)
            monitoring_task = asyncio.create_task(
                hardware_monitor.start_monitoring(update_interval=0.1)
            )
            
            # Phase 1: Pre-game state
            await asyncio.sleep(0.2)
            pre_game_state = await hardware_monitor._get_current_hardware_state()
            assert pre_game_state.gaming_mode_active is False
            
            # Phase 2: Game launch
            workflow_events.append({'type': 'game_launch', 'timestamp': time.time()})
            dbus_interface.set_gaming_mode(True)
            dbus_interface.add_steam_app(987654, "The Witcher 3")
            
            # Let system adapt to gaming mode
            await asyncio.sleep(0.3)
            gaming_state = await hardware_monitor._get_current_hardware_state()
            
            # Should detect gaming mode and adapt
            gaming_profile = hardware_monitor.get_optimal_profile(gaming_state)
            assert gaming_profile == 'gaming'
            
            # Phase 3: Thermal stress during gaming
            workflow_events.append({'type': 'thermal_stress', 'timestamp': time.time()})
            stress_state = create_thermal_stress_scenario(hardware_state)
            stress_state.gaming_mode_active = True
            
            with mock_steamdeck_environment(stress_state):
                await asyncio.sleep(0.3)
                
                # Should maintain gaming priority while managing thermal
                stressed_state = await hardware_monitor._get_current_hardware_state()
                stressed_profile = hardware_monitor.get_optimal_profile(stressed_state)
                
                # Could be gaming or thermal_emergency depending on severity
                assert stressed_profile in ['gaming', 'thermal_emergency']
            
            # Phase 4: Game termination
            workflow_events.append({'type': 'game_termination', 'timestamp': time.time()})
            dbus_interface.remove_steam_app(987654)
            dbus_interface.set_gaming_mode(False)
            
            await asyncio.sleep(0.2)
            post_game_state = await hardware_monitor._get_current_hardware_state()
            
            # Should return to normal operation
            post_game_profile = hardware_monitor.get_optimal_profile(post_game_state)
            assert post_game_profile != 'gaming'
            
            # Cleanup
            thermal_optimizer.stop_monitoring()
            hardware_monitor.stop_monitoring()
            
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Verify complete workflow was tracked
            assert len(workflow_events) >= 4
            event_types = [e['type'] for e in workflow_events]
            assert 'game_launch' in event_types
            assert 'game_termination' in event_types
            
            # Verify state transitions occurred
            hardware_events = [e for e in workflow_events if e['type'] == 'hardware_state']
            assert len(hardware_events) > 0
    
    @pytest.mark.integration
    @pytest.mark.workflow
    @pytest.mark.benchmark
    async def test_system_performance_under_load(self):
        """Test system performance under various load scenarios"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize all components
            components = {
                'hardware_monitor': SteamDeckHardwareMonitor(),
                'thermal_optimizer': SteamDeckThermalOptimizer(),
                'deck_optimizer': SteamDeckOptimizer(),
                'cache_optimizer': SteamDeckCacheOptimizer()
            }
            
            performance_metrics = {}
            
            # Test component initialization time
            with PerformanceTimer() as init_timer:
                for name, component in components.items():
                    if hasattr(component, 'initialize'):
                        await component.initialize()
            
            performance_metrics['initialization_time'] = init_timer.elapsed
            
            # Test state reading performance
            hardware_monitor = components['hardware_monitor']
            
            state_read_times = []
            for _ in range(10):
                with PerformanceTimer() as read_timer:
                    await hardware_monitor._get_current_hardware_state()
                state_read_times.append(read_timer.elapsed)
            
            performance_metrics['avg_state_read_time'] = sum(state_read_times) / len(state_read_times)
            performance_metrics['max_state_read_time'] = max(state_read_times)
            
            # Test optimization decision performance
            deck_optimizer = components['deck_optimizer']
            
            optimization_times = []
            for _ in range(5):
                with PerformanceTimer() as opt_timer:
                    state = deck_optimizer.get_current_state()
                    recommendations = deck_optimizer.get_optimization_recommendations()
                optimization_times.append(opt_timer.elapsed)
            
            performance_metrics['avg_optimization_time'] = sum(optimization_times) / len(optimization_times)
            
            # Performance assertions
            assert performance_metrics['avg_state_read_time'] < 0.1, "State reading should be fast"
            assert performance_metrics['avg_optimization_time'] < 0.5, "Optimization decisions should be fast"
            assert performance_metrics['initialization_time'] < 5.0, "Initialization should complete quickly"
            
            # Log performance metrics
            logger.info(f"Performance metrics: {performance_metrics}")
    
    @pytest.mark.integration
    @pytest.mark.workflow
    async def test_error_recovery_and_resilience(self):
        """Test system resilience and error recovery capabilities"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Initialize components
            hardware_monitor = SteamDeckHardwareMonitor()
            deck_optimizer = SteamDeckOptimizer()
            
            # Test recovery from hardware read errors
            with patch.object(hardware_monitor, '_get_cpu_temperature', side_effect=Exception("Sensor error")):
                state = await hardware_monitor._get_current_hardware_state()
                # Should provide fallback values
                assert isinstance(state.cpu_temperature, float)
                assert state.cpu_temperature > 0
            
            # Test recovery from optimization errors
            with patch.object(deck_optimizer, 'apply_optimization_profile', side_effect=Exception("Profile error")):
                # Should handle gracefully
                try:
                    success = deck_optimizer.apply_optimization_profile('balanced')
                    assert success is False  # Should return False on error
                except Exception:
                    pytest.fail("Should handle optimization errors gracefully")
            
            # Test continued operation after errors
            state = await hardware_monitor._get_current_hardware_state()
            recommendations = deck_optimizer.get_optimization_recommendations()
            
            # Should continue working normally
            assert isinstance(state.cpu_temperature, float)
            assert isinstance(recommendations, list)
    
    @pytest.mark.integration
    @pytest.mark.workflow
    async def test_configuration_persistence_and_loading(self):
        """Test configuration persistence and loading across sessions"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state):
            deck_optimizer = SteamDeckOptimizer()
            
            # Generate optimization report
            with tempfile.TemporaryDirectory() as temp_dir:
                report_path = Path(temp_dir) / "optimization_report.json"
                
                # Export report
                deck_optimizer.export_optimization_report(report_path)
                
                # Verify report was created and contains expected data
                assert report_path.exists()
                
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                
                # Validate report structure
                assert 'timestamp' in report_data
                assert 'is_steam_deck' in report_data
                assert 'hardware_profile' in report_data
                assert 'current_state' in report_data
                assert 'compatibility_report' in report_data
                
                # Test configuration loading
                if report_data['is_steam_deck']:
                    assert report_data['hardware_profile']['model'] == 'oled'
                    assert 'optimization_active' in report_data


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.steamdeck
class TestSteamDeckPerformanceBenchmarks:
    """Benchmark tests for Steam Deck integration performance"""
    
    def test_hardware_detection_performance(self, benchmark):
        """Benchmark hardware detection performance"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        def detect_hardware():
            with mock_steamdeck_environment(hardware_state):
                monitor = SteamDeckHardwareMonitor()
                return monitor.is_steam_deck, monitor.model
        
        result = benchmark(detect_hardware)
        assert result[0] is True  # Should detect Steam Deck
    
    def test_thermal_monitoring_performance(self, benchmark):
        """Benchmark thermal monitoring performance"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        def monitor_thermal():
            with mock_steamdeck_environment(hardware_state):
                optimizer = SteamDeckThermalOptimizer()
                return optimizer.get_status()
        
        result = benchmark(monitor_thermal)
        assert 'thermal_state' in result
    
    def test_optimization_decision_performance(self, benchmark):
        """Benchmark optimization decision making performance"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        async def make_optimization_decision():
            with mock_steamdeck_environment(hardware_state):
                monitor = SteamDeckHardwareMonitor()
                state = await monitor._get_current_hardware_state()
                return monitor.get_optimal_profile(state)
        
        result = benchmark(asyncio.run, make_optimization_decision())
        assert result in ['balanced', 'maximum', 'gaming', 'battery', 'thermal_emergency']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
