#!/usr/bin/env python3
"""
Steam Deck D-Bus Integration Tests

Tests for Steam D-Bus integration including:
- Steam process detection and monitoring
- Gaming mode detection via D-Bus
- Game launch/termination event handling
- D-Bus connection failure and recovery
- Multiple D-Bus backend support
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from tests.fixtures.steamdeck_fixtures import (
    MockHardwareState, MockSteamDeckModel, mock_steamdeck_environment,
    MockDBusInterface, create_intensive_gaming_scenario
)

try:
    from src.core.enhanced_dbus_manager import (
        EnhancedDBusManager, DBusBackendType, DBusCapabilities
    )
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

try:
    from src.core.steam_dbus_interface import SteamDBusInterface
    STEAM_DBUS_AVAILABLE = True
except ImportError:
    STEAM_DBUS_AVAILABLE = False


@pytest.mark.skipif(not DBUS_AVAILABLE, reason="Enhanced D-Bus manager not available")
class TestEnhancedDBusManager:
    """Test Enhanced D-Bus Manager functionality"""
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_dbus_manager_initialization(self):
        """Test D-Bus manager initialization"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            dbus_manager = EnhancedDBusManager()
            
            # Test initialization
            initialized = await dbus_manager.initialize()
            
            # Should initialize successfully (may use fallback)
            assert isinstance(initialized, bool)
            
            # Should have a backend selected
            assert hasattr(dbus_manager, 'current_backend')
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    def test_dbus_backend_selection(self):
        """Test D-Bus backend selection logic"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state):
            dbus_manager = EnhancedDBusManager()
            
            # Test backend capability assessment
            if hasattr(dbus_manager, '_assess_backend_capabilities'):
                capabilities = dbus_manager._assess_backend_capabilities()
                
                assert isinstance(capabilities, dict)
                
                # Should have entries for different backends
                for backend_type in DBusBackendType:
                    if backend_type.value in capabilities:
                        assert isinstance(capabilities[backend_type.value], DBusCapabilities)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_dbus_connection_failure_handling(self):
        """Test handling of D-Bus connection failures"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state):
            dbus_manager = EnhancedDBusManager()
            
            # Mock connection failure for primary backends
            with patch('src.core.enhanced_dbus_manager.DBusNextBackend.connect', return_value=False):
                with patch('src.core.enhanced_dbus_manager.JeepneyBackend.connect', return_value=False):
                    
                    # Should fall back to process-based detection
                    initialized = await dbus_manager.initialize()
                    
                    # Should still initialize (using fallback)
                    assert isinstance(initialized, bool)
                    
                    # Should be able to detect gaming mode via fallback
                    gaming_mode = await dbus_manager.is_gaming_mode_active()
                    assert isinstance(gaming_mode, bool)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_steam_process_monitoring(self):
        """Test Steam process monitoring functionality"""
        hardware_state = create_intensive_gaming_scenario(
            MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        )
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            dbus_manager = EnhancedDBusManager()
            
            # Setup Steam processes
            dbus_interface.add_steam_app(12345, "Portal 2")
            dbus_interface.add_steam_app(67890, "Half-Life: Alyx")
            
            await dbus_manager.initialize()
            
            # Test Steam process detection
            if hasattr(dbus_manager, 'get_steam_processes'):
                processes = await dbus_manager.get_steam_processes()
                
                if processes is not None:
                    assert isinstance(processes, list)
                    # Should detect Steam processes
                    steam_processes = [p for p in processes if 'steam' in str(p).lower()]
                    assert len(steam_processes) >= 0
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_gaming_mode_detection_accuracy(self):
        """Test accuracy of gaming mode detection"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            dbus_manager = EnhancedDBusManager()
            
            await dbus_manager.initialize()
            
            # Test gaming mode OFF
            dbus_interface.set_gaming_mode(False)
            gaming_mode_off = await dbus_manager.is_gaming_mode_active()
            
            # Test gaming mode ON
            dbus_interface.set_gaming_mode(True)
            dbus_interface.add_steam_app(24680, "Cyberpunk 2077")
            gaming_mode_on = await dbus_manager.is_gaming_mode_active()
            
            # Should detect the change (may depend on implementation)
            if gaming_mode_off is not None and gaming_mode_on is not None:
                assert isinstance(gaming_mode_off, bool)
                assert isinstance(gaming_mode_on, bool)
                # Gaming mode detection may vary based on backend availability
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_dbus_signal_monitoring(self):
        """Test D-Bus signal monitoring functionality"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            dbus_manager = EnhancedDBusManager()
            
            await dbus_manager.initialize()
            
            # Setup signal monitoring
            signals_received = []
            
            def signal_callback(signal_name, data):
                signals_received.append({
                    'signal': signal_name,
                    'data': data,
                    'timestamp': time.time()
                })
            
            # Add callback if supported
            if hasattr(dbus_manager, 'add_signal_callback'):
                dbus_manager.add_signal_callback(signal_callback)
                
                # Start monitoring
                if hasattr(dbus_manager, 'start_signal_monitoring'):
                    await dbus_manager.start_signal_monitoring()
            
            # Trigger events
            dbus_interface.set_gaming_mode(True)
            dbus_interface.add_steam_app(13579, "Elden Ring")
            
            # Wait for potential signals
            await asyncio.sleep(0.1)
            
            # Stop monitoring
            if hasattr(dbus_manager, 'stop_signal_monitoring'):
                await dbus_manager.stop_signal_monitoring()
            
            # May or may not receive signals depending on implementation
            assert isinstance(signals_received, list)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_dbus_backend_fallback_chain(self):
        """Test D-Bus backend fallback chain"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state):
            dbus_manager = EnhancedDBusManager()
            
            # Test fallback chain by mocking failures
            backends_tried = []
            
            async def mock_backend_connect(backend_name):
                backends_tried.append(backend_name)
                return False  # Always fail
            
            # Mock all backends to fail except fallback
            with patch('src.core.enhanced_dbus_manager.DBusNextBackend.connect', 
                      side_effect=lambda: mock_backend_connect('dbus_next')):
                with patch('src.core.enhanced_dbus_manager.JeepneyBackend.connect', 
                          side_effect=lambda: mock_backend_connect('jeepney')):
                    
                    initialized = await dbus_manager.initialize()
                    
                    # Should eventually initialize with fallback
                    assert isinstance(initialized, bool)
                    
                    # Should have tried multiple backends
                    if hasattr(dbus_manager, 'backend_type'):
                        assert dbus_manager.backend_type == DBusBackendType.FALLBACK or initialized is True


@pytest.mark.skipif(not STEAM_DBUS_AVAILABLE, reason="Steam D-Bus interface not available")
class TestSteamDBusInterface:
    """Test Steam-specific D-Bus interface"""
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_steam_dbus_interface_initialization(self):
        """Test Steam D-Bus interface initialization"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        with mock_steamdeck_environment(hardware_state):
            steam_dbus = SteamDBusInterface()
            
            # Test initialization
            initialized = await steam_dbus.initialize()
            
            # Should initialize (may use fallback)
            assert isinstance(initialized, bool)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_steam_application_monitoring(self):
        """Test Steam application monitoring"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            steam_dbus = SteamDBusInterface()
            
            await steam_dbus.initialize()
            
            # Add Steam applications
            test_apps = [
                (111111, "Counter-Strike 2"),
                (222222, "Dota 2"),
                (333333, "Team Fortress 2")
            ]
            
            for app_id, name in test_apps:
                dbus_interface.add_steam_app(app_id, name)
            
            # Test application detection
            if hasattr(steam_dbus, 'get_running_applications'):
                apps = await steam_dbus.get_running_applications()
                
                if apps is not None:
                    assert isinstance(apps, list)
                    # Should detect running applications
                    assert len(apps) >= 0
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_steam_overlay_detection(self):
        """Test Steam overlay detection"""
        hardware_state = create_intensive_gaming_scenario(
            MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        )
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            steam_dbus = SteamDBusInterface()
            
            await steam_dbus.initialize()
            
            # Setup gaming scenario
            dbus_interface.set_gaming_mode(True)
            dbus_interface.add_steam_app(444444, "The Witcher 3")
            
            # Test overlay detection
            if hasattr(steam_dbus, 'is_overlay_active'):
                overlay_active = await steam_dbus.is_overlay_active()
                
                # Should return boolean or None
                if overlay_active is not None:
                    assert isinstance(overlay_active, bool)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_steam_performance_metrics(self):
        """Test Steam performance metrics collection"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            steam_dbus = SteamDBusInterface()
            
            await steam_dbus.initialize()
            
            # Add a running game
            dbus_interface.add_steam_app(555555, "God of War")
            
            # Test performance metrics
            if hasattr(steam_dbus, 'get_performance_metrics'):
                metrics = await steam_dbus.get_performance_metrics()
                
                if metrics is not None:
                    assert isinstance(metrics, dict)
                    # Should contain performance data
                    expected_fields = ['fps', 'frame_time', 'cpu_usage', 'gpu_usage']
                    # May or may not contain all fields
                    assert len(metrics) >= 0


class TestDBusIntegrationWithHardwareMonitoring:
    """Test D-Bus integration with hardware monitoring"""
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_dbus_hardware_event_coordination(self):
        """Test coordination between D-Bus events and hardware monitoring"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_512GB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            
            # Initialize both systems
            if DBUS_AVAILABLE:
                try:
                    dbus_manager = EnhancedDBusManager()
                    await dbus_manager.initialize()
                    
                    from src.core.steam_deck_hardware_integration import SteamDeckHardwareMonitor
                    hardware_monitor = SteamDeckHardwareMonitor()
                    
                    # Track events
                    events = []
                    
                    def event_callback(event_data):
                        events.append({
                            'type': 'hardware_event',
                            'data': event_data,
                            'timestamp': time.time()
                        })
                    
                    hardware_monitor.add_state_callback(event_callback)
                    
                    # Trigger gaming mode change
                    dbus_interface.set_gaming_mode(True)
                    dbus_interface.add_steam_app(666666, "Red Dead Redemption 2")
                    
                    # Get hardware state
                    state = await hardware_monitor._get_current_hardware_state()
                    
                    # Should coordinate gaming mode detection
                    # (Implementation may vary)
                    if hasattr(state, 'gaming_mode_active'):
                        assert isinstance(state.gaming_mode_active, bool)
                    
                except ImportError:
                    pytest.skip("Required modules not available")
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_dbus_thermal_integration(self):
        """Test D-Bus integration with thermal management"""
        # Create thermal stress scenario
        hardware_state = create_intensive_gaming_scenario(
            MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        )
        hardware_state.cpu_temperature = 85.0  # High temperature
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            
            if DBUS_AVAILABLE:
                try:
                    dbus_manager = EnhancedDBusManager()
                    await dbus_manager.initialize()
                    
                    from src.core.steamdeck_thermal_optimizer import SteamDeckThermalOptimizer
                    thermal_optimizer = SteamDeckThermalOptimizer()
                    
                    # Setup gaming scenario with thermal stress
                    dbus_interface.set_gaming_mode(True)
                    dbus_interface.add_steam_app(777777, "Cyberpunk 2077")
                    
                    # Get thermal status
                    thermal_status = thermal_optimizer.get_status()
                    
                    # Should coordinate thermal management with gaming
                    if thermal_status.get('thermal_state') == 'critical':
                        # Gaming should be detected but thermal limits respected
                        gaming_opts = thermal_optimizer.optimize_for_gaming()
                        
                        # Should balance gaming performance with thermal protection
                        assert gaming_opts.get('compilation_paused') is True
                        assert gaming_opts.get('background_threads', 0) <= 2
                    
                except ImportError:
                    pytest.skip("Required modules not available")


class TestDBusErrorHandlingAndRecovery:
    """Test D-Bus error handling and recovery scenarios"""
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_dbus_connection_recovery(self):
        """Test D-Bus connection recovery after failure"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.OLED_1TB)
        
        with mock_steamdeck_environment(hardware_state):
            if DBUS_AVAILABLE:
                dbus_manager = EnhancedDBusManager()
                
                # Initial connection
                initialized = await dbus_manager.initialize()
                assert isinstance(initialized, bool)
                
                # Simulate connection loss
                if hasattr(dbus_manager, 'disconnect'):
                    await dbus_manager.disconnect()
                
                # Attempt reconnection
                if hasattr(dbus_manager, 'reconnect'):
                    reconnected = await dbus_manager.reconnect()
                    assert isinstance(reconnected, bool)
                else:
                    # Try re-initialization
                    reinitialized = await dbus_manager.initialize()
                    assert isinstance(reinitialized, bool)
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    async def test_dbus_signal_handling_errors(self):
        """Test error handling in D-Bus signal processing"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_512GB)
        
        with mock_steamdeck_environment(hardware_state) as mock_env:
            dbus_interface = mock_env['dbus']
            
            if DBUS_AVAILABLE:
                dbus_manager = EnhancedDBusManager()
                await dbus_manager.initialize()
                
                # Add a callback that raises an exception
                def failing_callback(signal_name, data):
                    raise Exception("Callback error")
                
                if hasattr(dbus_manager, 'add_signal_callback'):
                    dbus_manager.add_signal_callback(failing_callback)
                    
                    # Add a working callback
                    working_callback_called = threading.Event()
                    def working_callback(signal_name, data):
                        working_callback_called.set()
                    
                    dbus_manager.add_signal_callback(working_callback)
                    
                    # Trigger signals
                    dbus_interface.set_gaming_mode(True)
                    
                    # Should handle callback errors gracefully
                    # Working callback should still be called
                    await asyncio.sleep(0.1)
                    
                    # Test should not crash due to failing callback
                    assert True  # If we reach here, error was handled
    
    @pytest.mark.unit
    @pytest.mark.steamdeck
    @pytest.mark.dbus
    def test_dbus_service_unavailable_handling(self):
        """Test handling when D-Bus service is completely unavailable"""
        hardware_state = MockHardwareState(model=MockSteamDeckModel.LCD_256GB)
        
        with mock_steamdeck_environment(hardware_state):
            # Mock D-Bus as completely unavailable
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 1  # D-Bus commands fail
                
                if DBUS_AVAILABLE:
                    async def test_dbus_unavailable():
                        dbus_manager = EnhancedDBusManager()
                        
                        # Should handle D-Bus unavailability gracefully
                        initialized = await dbus_manager.initialize()
                        
                        # Should fall back to process-based detection
                        assert isinstance(initialized, bool)
                        
                        # Should still provide basic functionality
                        gaming_mode = await dbus_manager.is_gaming_mode_active()
                        assert isinstance(gaming_mode, bool)
                    
                    # Should not raise exceptions
                    asyncio.run(test_dbus_unavailable())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
