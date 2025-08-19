#!/usr/bin/env python3
"""
Unit Tests for Steam Deck Gaming Detector
Testing gaming mode detection, D-Bus integration, and resource monitoring
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tests.fixtures.steamdeck_fixtures import (
    MockHardwareState, MockSteamDeckModel, MockDBusInterface,
    mock_steamdeck_environment, create_intensive_gaming_scenario,
    benchmark_test
)

# Import the module under test
from src.core.steamdeck_gaming_detector import (
    SteamDeckGamingDetector, GamingState, GameProcess, ResourceUsage,
    get_gaming_detector
)


class TestSteamDeckGamingDetector:
    """Test Steam Deck gaming detection system"""
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_detector_initialization(self, mock_steamdeck_lcd):
        """Test gaming detector initialization"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            assert detector.current_state == GamingState.IDLE
            assert len(detector.detected_games) == 0
            assert detector.gpu_threshold_2d < detector.gpu_threshold_3d
            assert detector.cpu_threshold_game > 0
            assert detector.memory_threshold_game > 0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_game_process_detection(self, mock_steamdeck_gaming):
        """Test game process detection from system processes"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            detector = SteamDeckGamingDetector()
            
            # Mock process detection
            mock_processes = [
                Mock(
                    pid=1234,
                    name="TestGame.exe",
                    exe="/path/to/TestGame.exe",
                    cpu_percent=lambda: 45.0,
                    memory_info=lambda: Mock(rss=512*1024*1024)
                ),
                Mock(
                    pid=5678,
                    name="gamescope",
                    exe="/usr/bin/gamescope",
                    cpu_percent=lambda: 15.0,
                    memory_info=lambda: Mock(rss=128*1024*1024)
                )
            ]
            
            with patch('psutil.process_iter', return_value=mock_processes):
                detected_games = detector._detect_game_processes()
                
                assert len(detected_games) >= 1
                assert any(game.name == "TestGame.exe" for game in detected_games.values())
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_gaming_state_classification(self, mock_steamdeck_lcd):
        """Test gaming state classification based on resource usage"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Test idle state
            idle_usage = ResourceUsage(
                cpu_percent=5.0,
                memory_percent=30.0,
                gpu_percent=2,
                thermal_temp=55.0,
                battery_percent=75,
                is_charging=False
            )
            state = detector._classify_gaming_state(idle_usage, {})
            assert state == GamingState.IDLE
            
            # Test 2D gaming state
            light_gaming_usage = ResourceUsage(
                cpu_percent=25.0,
                memory_percent=45.0,
                gpu_percent=15,
                thermal_temp=65.0,
                battery_percent=70,
                is_charging=False
            )
            light_game_processes = {
                1234: GameProcess(
                    pid=1234,
                    name="Simple2DGame",
                    exe_path="/path/to/game",
                    cpu_percent=20.0,
                    memory_mb=256,
                    gpu_usage=15,
                    is_steam_game=True,
                    launch_time=time.time()
                )
            }
            state = detector._classify_gaming_state(light_gaming_usage, light_game_processes)
            assert state == GamingState.ACTIVE_2D
            
            # Test 3D gaming state
            intense_gaming_usage = ResourceUsage(
                cpu_percent=65.0,
                memory_percent=70.0,
                gpu_percent=85,
                thermal_temp=80.0,
                battery_percent=60,
                is_charging=False
            )
            intense_game_processes = {
                5678: GameProcess(
                    pid=5678,
                    name="Intense3DGame.exe",
                    exe_path="/path/to/intense/game.exe",
                    cpu_percent=60.0,
                    memory_mb=1024,
                    gpu_usage=80,
                    is_steam_game=True,
                    launch_time=time.time()
                )
            }
            state = detector._classify_gaming_state(intense_gaming_usage, intense_game_processes)
            assert state == GamingState.ACTIVE_3D
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_steam_game_identification(self, mock_steamdeck_lcd):
        """Test identification of Steam games vs other processes"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Test Steam game path
            steam_path = "/home/deck/.steam/steamapps/common/TestGame/game.exe"
            assert detector._is_steam_game(steam_path) is True
            
            # Test Proton game path
            proton_path = "/home/deck/.steam/steamapps/compatdata/123456/pfx/drive_c/Program Files/Game/game.exe"
            assert detector._is_steam_game(proton_path) is True
            
            # Test non-Steam path
            other_path = "/usr/bin/firefox"
            assert detector._is_steam_game(other_path) is False
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_process_filtering(self, mock_steamdeck_lcd):
        """Test filtering of system processes from game detection"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Test system process filtering
            system_processes = ['systemd', 'kworker', 'python3', 'chrome']
            for process_name in system_processes:
                assert detector._is_likely_game_process(process_name) is False
            
            # Test game-like processes
            game_processes = ['TestGame.exe', 'game.x86_64', 'MyAwesomeGame']
            for process_name in game_processes:
                assert detector._is_likely_game_process(process_name) is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_gpu_usage_monitoring(self, mock_steamdeck_gaming):
        """Test GPU usage monitoring for gaming detection"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            detector = SteamDeckGamingDetector()
            
            # Mock GPU usage reading
            with patch.object(detector, '_get_gpu_usage', return_value=75):
                usage = detector._get_resource_usage()
                assert usage.gpu_percent == 75
                
                # High GPU usage should indicate 3D gaming
                if usage.gpu_percent > detector.gpu_threshold_3d:
                    # Should classify as 3D gaming with appropriate processes
                    pass
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_dbus_steam_integration(self, mock_steamdeck_lcd, mock_dbus_interface):
        """Test D-Bus integration with Steam"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock D-Bus Steam interface
            detector.steam_dbus = mock_dbus_interface
            
            # Add a Steam game
            mock_dbus_interface.add_steam_app(12345, "Test Steam Game")
            mock_dbus_interface.set_gaming_mode(True)
            
            # Should detect gaming mode
            steam_info = detector._get_steam_info()
            assert steam_info['gaming_mode_active'] is True
            assert len(steam_info['running_apps']) > 0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_gamescope_detection(self, mock_steamdeck_gaming):
        """Test gamescope process detection for Gaming Mode"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            detector = SteamDeckGamingDetector()
            
            # Mock gamescope process detection
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="1234\n")
                
                is_gamescope_running = detector._is_gamescope_running()
                assert is_gamescope_running is True
                
                mock_run.assert_called_once()
                assert 'gamescope' in str(mock_run.call_args)
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_gaming_state_transitions(self, mock_steamdeck_lcd):
        """Test gaming state transitions and callbacks"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            callback_called = threading.Event()
            state_changes = []
            
            def state_change_callback(old_state, new_state):
                state_changes.append((old_state, new_state))
                callback_called.set()
            
            detector.add_state_callback(state_change_callback)
            
            # Simulate state change
            detector._update_gaming_state(GamingState.ACTIVE_3D)
            
            # Should have called callback
            callback_called.wait(timeout=1.0)
            assert len(state_changes) > 0
            assert state_changes[-1][1] == GamingState.ACTIVE_3D
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_resource_recommendations(self, mock_steamdeck_gaming):
        """Test resource usage recommendations during gaming"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            detector = SteamDeckGamingDetector()
            detector.current_state = GamingState.ACTIVE_3D
            
            recommendations = detector.get_resource_recommendations()
            
            # Should recommend reducing background work during intensive gaming
            assert recommendations['reduce_background_work'] is True
            assert recommendations['max_background_threads'] <= 2
            assert recommendations['pause_shader_compilation'] is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_monitoring_start_stop(self, mock_steamdeck_lcd):
        """Test gaming monitoring start and stop"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Start monitoring
            detector.start_monitoring()
            assert detector._monitoring_thread is not None
            assert detector._monitoring_thread.is_alive()
            
            # Stop monitoring
            detector.stop_monitoring()
            time.sleep(0.1)  # Give thread time to stop
            assert not detector._monitoring_thread.is_alive()
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_game_launch_detection(self, mock_steamdeck_lcd):
        """Test detection of games being launched"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock recent process launch
            recent_process = GameProcess(
                pid=9999,
                name="NewGame.exe",
                exe_path="/path/to/new/game.exe",
                cpu_percent=30.0,
                memory_mb=512,
                gpu_usage=25,
                is_steam_game=True,
                launch_time=time.time() - 2  # Launched 2 seconds ago
            )
            
            detector.detected_games[9999] = recent_process
            
            # Should detect launching state
            is_launching = detector._is_game_launching(recent_process)
            assert is_launching is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_suspended_game_detection(self, mock_steamdeck_lcd):
        """Test detection of suspended/minimized games"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock suspended game process (low CPU usage)
            suspended_process = GameProcess(
                pid=7777,
                name="SuspendedGame.exe",
                exe_path="/path/to/suspended/game.exe",
                cpu_percent=1.0,  # Very low CPU
                memory_mb=1024,   # Still using memory
                gpu_usage=0,      # No GPU usage
                is_steam_game=True,
                launch_time=time.time() - 300  # Been running for 5 minutes
            )
            
            detector.detected_games[7777] = suspended_process
            
            # Should detect as suspended if resource usage is very low
            # but process still exists with significant memory usage
            current_usage = ResourceUsage(
                cpu_percent=5.0,   # Low overall CPU
                memory_percent=40.0,
                gpu_percent=1,     # Minimal GPU
                thermal_temp=55.0,
                battery_percent=75,
                is_charging=False
            )
            
            state = detector._classify_gaming_state(current_usage, detector.detected_games)
            # Could be SUSPENDED or IDLE depending on implementation
            assert state in [GamingState.SUSPENDED, GamingState.IDLE]
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_steam_overlay_detection(self, mock_steamdeck_lcd):
        """Test Steam overlay detection"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock Steam overlay process
            with patch('psutil.process_iter') as mock_iter:
                overlay_process = Mock(
                    pid=8888,
                    name="steamoverlay",
                    exe="/usr/bin/steamoverlay",
                    cpu_percent=lambda: 10.0,
                    memory_info=lambda: Mock(rss=64*1024*1024)
                )
                mock_iter.return_value = [overlay_process]
                
                is_overlay_active = detector._is_steam_overlay_active()
                # Implementation would detect overlay
                assert isinstance(is_overlay_active, bool)
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_performance_impact_assessment(self, mock_steamdeck_gaming):
        """Test assessment of gaming performance impact"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            detector = SteamDeckGamingDetector()
            
            # High performance impact scenario
            high_impact_usage = ResourceUsage(
                cpu_percent=85.0,
                memory_percent=80.0,
                gpu_percent=90,
                thermal_temp=85.0,
                battery_percent=30,
                is_charging=False
            )
            
            impact = detector._assess_performance_impact(high_impact_usage)
            
            assert impact['cpu_impact'] == 'high'
            assert impact['gpu_impact'] == 'high'
            assert impact['thermal_impact'] == 'high'
            assert impact['battery_impact'] == 'high'
            assert impact['overall_impact'] == 'high'
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_status_reporting(self, mock_steamdeck_gaming):
        """Test comprehensive status reporting"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            detector = SteamDeckGamingDetector()
            
            status = detector.get_status()
            
            # Should contain comprehensive status information
            expected_fields = [
                'gaming_state',
                'active_games_count',
                'total_cpu_usage',
                'total_gpu_usage',
                'thermal_state',
                'performance_impact',
                'recommendations_active'
            ]
            
            for field in expected_fields:
                assert field in status


class TestGameProcess:
    """Test GameProcess dataclass"""
    
    @pytest.mark.unit
    def test_game_process_creation(self):
        """Test GameProcess object creation"""
        process = GameProcess(
            pid=1234,
            name="TestGame.exe",
            exe_path="/path/to/game.exe",
            cpu_percent=45.0,
            memory_mb=512.0,
            gpu_usage=75,
            is_steam_game=True,
            launch_time=time.time()
        )
        
        assert process.pid == 1234
        assert process.name == "TestGame.exe"
        assert process.is_steam_game is True
        assert process.gpu_usage == 75


class TestResourceUsage:
    """Test ResourceUsage dataclass"""
    
    @pytest.mark.unit
    def test_resource_usage_creation(self):
        """Test ResourceUsage object creation"""
        usage = ResourceUsage(
            cpu_percent=65.0,
            memory_percent=70.0,
            gpu_percent=85,
            thermal_temp=80.0,
            battery_percent=60,
            is_charging=False
        )
        
        assert usage.cpu_percent == 65.0
        assert usage.gpu_percent == 85
        assert usage.is_charging is False


class TestConvenienceFunctions:
    """Test module convenience functions"""
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_get_gaming_detector_singleton(self, mock_steamdeck_lcd):
        """Test global gaming detector singleton"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector1 = get_gaming_detector()
            detector2 = get_gaming_detector()
            
            assert detector1 is detector2  # Should be same instance


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.unit
    def test_process_enumeration_error(self, mock_steamdeck_lcd):
        """Test handling of process enumeration errors"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock process enumeration error
            with patch('psutil.process_iter', side_effect=Exception("Process access denied")):
                games = detector._detect_game_processes()
                assert isinstance(games, dict)  # Should return empty dict on error
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_dbus_connection_error(self, mock_steamdeck_lcd):
        """Test handling of D-Bus connection errors"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock D-Bus error
            with patch.object(detector, '_connect_steam_dbus', 
                            side_effect=Exception("D-Bus connection failed")):
                steam_info = detector._get_steam_info()
                assert steam_info['gaming_mode_active'] is False  # Fallback
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_gpu_monitoring_error(self, mock_steamdeck_lcd):
        """Test handling of GPU monitoring errors"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock GPU monitoring error
            with patch.object(detector, '_get_gpu_usage', side_effect=Exception("GPU error")):
                usage = detector._get_resource_usage()
                assert usage.gpu_percent == 0  # Should fallback to 0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    def test_monitoring_thread_exception_handling(self, mock_steamdeck_lcd):
        """Test monitoring thread exception handling"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock an error in the monitoring loop
            with patch.object(detector, '_monitoring_loop', 
                            side_effect=Exception("Monitoring error")):
                detector.start_monitoring()
                time.sleep(0.1)
                
                # Thread should handle the exception gracefully
                detector.stop_monitoring()


@pytest.mark.benchmark
@pytest.mark.steamdeck
class TestGamingDetectorPerformance:
    """Performance tests for gaming detection"""
    
    def test_game_process_detection_performance(self, benchmark, mock_steamdeck_lcd):
        """Benchmark game process detection performance"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            # Mock multiple processes
            mock_processes = []
            for i in range(50):  # Simulate 50 processes
                mock_processes.append(Mock(
                    pid=1000+i,
                    name=f"process_{i}",
                    exe=f"/path/to/process_{i}",
                    cpu_percent=lambda: 5.0,
                    memory_info=lambda: Mock(rss=64*1024*1024)
                ))
            
            with patch('psutil.process_iter', return_value=mock_processes):
                # Should complete quickly even with many processes
                result = benchmark(detector._detect_game_processes)
                assert isinstance(result, dict)
    
    def test_gaming_state_classification_performance(self, benchmark, mock_steamdeck_lcd):
        """Benchmark gaming state classification performance"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            detector = SteamDeckGamingDetector()
            
            usage = ResourceUsage(
                cpu_percent=65.0,
                memory_percent=70.0,
                gpu_percent=85,
                thermal_temp=80.0,
                battery_percent=60,
                is_charging=False
            )
            
            games = {
                1234: GameProcess(
                    pid=1234,
                    name="TestGame.exe",
                    exe_path="/path/to/game.exe",
                    cpu_percent=45.0,
                    memory_mb=512.0,
                    gpu_usage=75,
                    is_steam_game=True,
                    launch_time=time.time()
                )
            }
            
            # Should be very fast
            result = benchmark(detector._classify_gaming_state, usage, games)
            assert isinstance(result, GamingState)
    
    @benchmark_test(iterations=1000)
    def test_resource_usage_creation_performance(self, mock_steamdeck_lcd):
        """Benchmark ResourceUsage object creation"""
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            gpu_percent=40,
            thermal_temp=70.0,
            battery_percent=80,
            is_charging=True
        )
        return usage


@pytest.mark.steamdeck
@pytest.mark.integration
class TestGamingDetectorIntegration:
    """Integration tests for gaming detector with other systems"""
    
    def test_gaming_thermal_integration(self, mock_steamdeck_gaming):
        """Test integration with thermal management during gaming"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            detector = SteamDeckGamingDetector()
            detector.current_state = GamingState.ACTIVE_3D
            
            recommendations = detector.get_resource_recommendations()
            
            # Should coordinate with thermal management
            assert recommendations['thermal_priority'] == 'game_performance'
            assert recommendations['background_thermal_monitoring'] is True
    
    def test_gaming_ml_integration(self, mock_steamdeck_gaming):
        """Test integration with ML prediction system during gaming"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            detector = SteamDeckGamingDetector()
            detector.current_state = GamingState.ACTIVE_3D
            
            recommendations = detector.get_resource_recommendations()
            
            # Should recommend ML prediction adjustments
            assert recommendations['reduce_ml_predictions'] is True
            assert recommendations['ml_prediction_priority'] == 'background'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])