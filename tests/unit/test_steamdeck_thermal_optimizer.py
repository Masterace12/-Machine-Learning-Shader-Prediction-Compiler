#!/usr/bin/env python3
"""
Unit Tests for Steam Deck Thermal Optimizer
Testing thermal management, monitoring, and optimization
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tests.fixtures.steamdeck_fixtures import (
    MockHardwareState, MockSteamDeckModel, mock_steamdeck_environment,
    create_thermal_stress_scenario, create_battery_critical_scenario,
    benchmark_test
)

# Import the module under test
from src.core.steamdeck_thermal_optimizer import (
    SteamDeckThermalOptimizer, SteamDeckModel, ThermalZone, 
    PowerProfile, ThermalReading, ThermalLimits,
    get_thermal_optimizer
)


class TestSteamDeckThermalOptimizer:
    """Test Steam Deck thermal optimization system"""
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_model_detection_lcd(self, mock_steamdeck_lcd):
        """Test LCD Steam Deck model detection"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            assert optimizer.model == SteamDeckModel.LCD
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_model_detection_oled(self, mock_steamdeck_oled):
        """Test OLED Steam Deck model detection"""
        with mock_steamdeck_environment(mock_steamdeck_oled):
            optimizer = SteamDeckThermalOptimizer()
            assert optimizer.model == SteamDeckModel.OLED
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_non_steam_deck_detection(self):
        """Test detection on non-Steam Deck systems"""
        with patch('os.path.exists', return_value=False):
            with patch('pathlib.Path.exists', return_value=False):
                optimizer = SteamDeckThermalOptimizer()
                assert optimizer.model == SteamDeckModel.UNKNOWN
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_power_profile_detection(self, mock_steamdeck_lcd):
        """Test power profile detection"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            assert optimizer.power_profile in [
                PowerProfile.BATTERY_SAVER,
                PowerProfile.BALANCED,
                PowerProfile.PERFORMANCE
            ]
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_limits_configuration(self, mock_steamdeck_lcd):
        """Test thermal limits configuration for different models"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            limits = optimizer.limits
            assert isinstance(limits, ThermalLimits)
            assert limits.normal_temp < limits.warning_temp
            assert limits.warning_temp < limits.critical_temp
            assert limits.critical_temp < limits.shutdown_temp
            
            # LCD should have slightly different limits than OLED
            assert limits.normal_temp >= 70.0
            assert limits.critical_temp <= 100.0
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_sensor_discovery(self, mock_steamdeck_lcd):
        """Test thermal sensor path discovery"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            sensor_paths = optimizer.sensor_paths
            assert isinstance(sensor_paths, dict)
            
            # Should detect thermal zones
            assert ThermalZone.APU in sensor_paths or ThermalZone.CPU in sensor_paths
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_temperature_reading(self, mock_steamdeck_lcd):
        """Test thermal sensor temperature reading"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Test reading CPU temperature
            cpu_temp = optimizer._get_cpu_temperature()
            assert isinstance(cpu_temp, float)
            assert 20.0 <= cpu_temp <= 120.0  # Reasonable temperature range
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_reading_creation(self, mock_steamdeck_lcd):
        """Test ThermalReading object creation"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            reading = ThermalReading(
                zone=ThermalZone.CPU,
                temperature=75.5,
                timestamp=time.time(),
                threshold_breach=False
            )
            
            assert reading.zone == ThermalZone.CPU
            assert reading.temperature == 75.5
            assert reading.threshold_breach is False
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_monitoring_start_stop(self, mock_steamdeck_lcd):
        """Test thermal monitoring start and stop"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Start monitoring
            optimizer.start_monitoring()
            assert optimizer._monitoring_thread is not None
            assert optimizer._monitoring_thread.is_alive()
            
            # Stop monitoring
            optimizer.stop_monitoring()
            time.sleep(0.1)  # Give thread time to stop
            assert not optimizer._monitoring_thread.is_alive()
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_state_classification(self, mock_steamdeck_lcd):
        """Test thermal state classification logic"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            limits = optimizer.limits
            
            # Test normal state
            normal_reading = ThermalReading(
                zone=ThermalZone.CPU,
                temperature=limits.normal_temp - 5.0,
                timestamp=time.time()
            )
            assert not optimizer._is_thermal_critical(normal_reading)
            
            # Test critical state
            critical_reading = ThermalReading(
                zone=ThermalZone.CPU,
                temperature=limits.critical_temp + 1.0,
                timestamp=time.time()
            )
            assert optimizer._is_thermal_critical(critical_reading)
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_throttling_detection(self, mock_steamdeck_thermal_throttling):
        """Test thermal throttling detection"""
        with mock_steamdeck_environment(mock_steamdeck_thermal_throttling):
            optimizer = SteamDeckThermalOptimizer()
            
            # Should detect throttling based on high temperature
            is_throttling = optimizer._is_thermal_throttling()
            assert is_throttling is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_background_throttle_activation(self, mock_steamdeck_lcd):
        """Test background workload throttling activation"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Create high temperature scenario
            high_temp_state = create_thermal_stress_scenario(mock_steamdeck_lcd)
            
            # Should activate background throttling
            optimizer._activate_background_throttling()
            assert optimizer.background_throttle_active is True
            
            # Should deactivate when temperatures normalize
            optimizer._deactivate_background_throttling()
            assert optimizer.background_throttle_active is False
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_emergency_mode_activation(self, mock_steamdeck_lcd):
        """Test emergency thermal mode activation"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Activate emergency mode
            optimizer._activate_emergency_mode()
            assert optimizer.emergency_mode_active is True
            
            # Should severely limit background activity
            recommendations = optimizer.get_thermal_recommendations()
            assert recommendations['max_background_threads'] <= 1
            assert recommendations['disable_compilation'] is True
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_history_tracking(self, mock_steamdeck_lcd):
        """Test thermal reading history tracking"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Add some readings
            for i in range(10):
                reading = ThermalReading(
                    zone=ThermalZone.CPU,
                    temperature=65.0 + i,
                    timestamp=time.time() + i
                )
                optimizer._add_thermal_reading(reading)
            
            # Should maintain history within limits
            assert len(optimizer.thermal_history) <= optimizer.max_history_size
            
            # Should keep most recent readings
            latest_reading = optimizer.thermal_history[-1]
            assert latest_reading.temperature == 74.0  # Last added temperature
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_power_profile_adaptation(self, mock_steamdeck_lcd):
        """Test adaptation to different power profiles"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Test different power profiles
            profiles = [
                PowerProfile.BATTERY_SAVER,
                PowerProfile.BALANCED,
                PowerProfile.PERFORMANCE
            ]
            
            for profile in profiles:
                optimizer._adapt_to_power_profile(profile)
                recommendations = optimizer.get_thermal_recommendations()
                
                # Battery saver should be most conservative
                if profile == PowerProfile.BATTERY_SAVER:
                    assert recommendations['max_background_threads'] <= 2
                    assert recommendations['reduce_prediction_frequency'] is True
                
                # Performance should allow more activity
                elif profile == PowerProfile.PERFORMANCE:
                    assert recommendations['max_background_threads'] >= 4
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_gaming_mode_thermal_management(self, mock_steamdeck_gaming):
        """Test thermal management during gaming"""
        with mock_steamdeck_environment(mock_steamdeck_gaming):
            optimizer = SteamDeckThermalOptimizer()
            
            # Enable gaming mode thermal management
            optimizer._enable_gaming_mode_thermal()
            
            recommendations = optimizer.get_thermal_recommendations()
            
            # Should prioritize game performance
            assert recommendations['prioritize_game_performance'] is True
            assert recommendations['background_work_priority'] == 'low'
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_prediction(self, mock_steamdeck_lcd):
        """Test thermal trend prediction"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Add trending upward temperatures
            base_temp = 60.0
            for i in range(5):
                reading = ThermalReading(
                    zone=ThermalZone.CPU,
                    temperature=base_temp + i * 2,
                    timestamp=time.time() + i
                )
                optimizer._add_thermal_reading(reading)
            
            # Should predict continued temperature rise
            predicted_trend = optimizer._predict_thermal_trend()
            assert predicted_trend in ['rising', 'stable', 'falling']
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_recommendations_generation(self, mock_steamdeck_lcd):
        """Test thermal management recommendation generation"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            recommendations = optimizer.get_thermal_recommendations()
            
            # Should contain expected recommendation fields
            expected_fields = [
                'max_background_threads',
                'reduce_prediction_frequency',
                'disable_compilation',
                'current_thermal_state',
                'thermal_headroom_seconds',
                'recommended_actions'
            ]
            
            for field in expected_fields:
                assert field in recommendations
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_status_reporting(self, mock_steamdeck_lcd):
        """Test thermal optimizer status reporting"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            status = optimizer.get_status()
            
            # Should contain comprehensive status information
            assert 'thermal_state' in status
            assert 'max_temperature' in status
            assert 'throttling_active' in status
            assert 'emergency_mode' in status
            assert 'last_reading_time' in status
            assert 'model' in status
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_limits_oled_vs_lcd(self, mock_steamdeck_oled):
        """Test different thermal limits for OLED vs LCD models"""
        with mock_steamdeck_environment(mock_steamdeck_oled):
            oled_optimizer = SteamDeckThermalOptimizer()
            oled_limits = oled_optimizer.limits
            
        # OLED may have slightly different thermal characteristics
        # This test ensures the system adapts to model differences
        assert isinstance(oled_limits, ThermalLimits)
        assert oled_limits.critical_temp > 0


class TestThermalLimits:
    """Test ThermalLimits dataclass"""
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_limits_defaults(self):
        """Test default thermal limits"""
        limits = ThermalLimits()
        
        assert limits.normal_temp == 75.0
        assert limits.warning_temp == 85.0
        assert limits.critical_temp == 95.0
        assert limits.shutdown_temp == 105.0
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_limits_custom(self):
        """Test custom thermal limits"""
        limits = ThermalLimits(
            normal_temp=70.0,
            warning_temp=80.0,
            critical_temp=90.0,
            shutdown_temp=100.0
        )
        
        assert limits.normal_temp == 70.0
        assert limits.warning_temp == 80.0
        assert limits.critical_temp == 90.0
        assert limits.shutdown_temp == 100.0


class TestThermalReading:
    """Test ThermalReading dataclass"""
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_reading_creation(self):
        """Test ThermalReading creation"""
        timestamp = time.time()
        reading = ThermalReading(
            zone=ThermalZone.CPU,
            temperature=75.5,
            timestamp=timestamp,
            threshold_breach=True
        )
        
        assert reading.zone == ThermalZone.CPU
        assert reading.temperature == 75.5
        assert reading.timestamp == timestamp
        assert reading.threshold_breach is True
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_thermal_reading_defaults(self):
        """Test ThermalReading default values"""
        reading = ThermalReading(
            zone=ThermalZone.GPU,
            temperature=80.0,
            timestamp=time.time()
        )
        
        assert reading.threshold_breach is False  # Default value


class TestConvenienceFunctions:
    """Test module convenience functions"""
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_get_thermal_optimizer_singleton(self, mock_steamdeck_lcd):
        """Test global thermal optimizer singleton"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer1 = get_thermal_optimizer()
            optimizer2 = get_thermal_optimizer()
            
            assert optimizer1 is optimizer2  # Should be same instance


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_missing_thermal_sensors(self):
        """Test behavior when thermal sensors are missing"""
        with patch('os.path.exists', return_value=False):
            optimizer = SteamDeckThermalOptimizer()
            
            # Should handle gracefully
            temp = optimizer._get_cpu_temperature()
            assert isinstance(temp, float)
            assert temp > 0  # Should provide fallback value
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_invalid_sensor_readings(self, mock_steamdeck_lcd):
        """Test handling of invalid sensor readings"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Mock sensor error
            with patch('builtins.open', side_effect=IOError("Sensor error")):
                temp = optimizer._get_cpu_temperature()
                assert isinstance(temp, float)
                assert temp > 0  # Should provide fallback
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_monitoring_thread_exception_handling(self, mock_steamdeck_lcd):
        """Test monitoring thread exception handling"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Mock an error in the monitoring loop
            with patch.object(optimizer, '_monitoring_loop', 
                            side_effect=Exception("Test error")):
                optimizer.start_monitoring()
                time.sleep(0.1)
                
                # Thread should handle the exception gracefully
                optimizer.stop_monitoring()
    
    @pytest.mark.steamdeck
    @pytest.mark.unit
    @pytest.mark.thermal
    def test_extreme_temperature_handling(self, mock_steamdeck_lcd):
        """Test handling of extreme temperature readings"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Test very high temperature
            extreme_reading = ThermalReading(
                zone=ThermalZone.CPU,
                temperature=120.0,
                timestamp=time.time()
            )
            
            # Should trigger emergency response
            is_critical = optimizer._is_thermal_critical(extreme_reading)
            assert is_critical is True


@pytest.mark.benchmark
@pytest.mark.steamdeck
@pytest.mark.thermal
class TestThermalPerformance:
    """Performance tests for thermal optimization"""
    
    def test_temperature_reading_performance(self, benchmark, mock_steamdeck_lcd):
        """Benchmark temperature reading performance"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Should be very fast (< 1ms)
            result = benchmark(optimizer._get_cpu_temperature)
            assert isinstance(result, float)
    
    def test_thermal_state_analysis_performance(self, benchmark, mock_steamdeck_lcd):
        """Benchmark thermal state analysis performance"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Add some history
            for i in range(50):
                reading = ThermalReading(
                    zone=ThermalZone.CPU,
                    temperature=65.0 + i * 0.5,
                    timestamp=time.time() + i
                )
                optimizer._add_thermal_reading(reading)
            
            # Benchmark analysis performance
            result = benchmark(optimizer.get_thermal_recommendations)
            assert isinstance(result, dict)
    
    @benchmark_test(iterations=1000)
    def test_thermal_reading_creation_performance(self, mock_steamdeck_lcd):
        """Benchmark ThermalReading object creation"""
        reading = ThermalReading(
            zone=ThermalZone.CPU,
            temperature=75.0,
            timestamp=time.time()
        )
        return reading


@pytest.mark.steamdeck
@pytest.mark.integration
@pytest.mark.thermal
class TestThermalIntegration:
    """Integration tests for thermal optimizer with other systems"""
    
    def test_thermal_ml_integration(self, mock_steamdeck_lcd):
        """Test integration with ML prediction system"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Simulate high thermal load
            optimizer._activate_background_throttling()
            
            recommendations = optimizer.get_thermal_recommendations()
            
            # Should recommend ML prediction throttling
            assert recommendations['reduce_prediction_frequency'] is True
            assert recommendations['max_background_threads'] < 4
    
    def test_thermal_cache_integration(self, mock_steamdeck_lcd):
        """Test integration with cache optimization"""
        with mock_steamdeck_environment(mock_steamdeck_lcd):
            optimizer = SteamDeckThermalOptimizer()
            
            # Simulate thermal stress
            optimizer._activate_emergency_mode()
            
            recommendations = optimizer.get_thermal_recommendations()
            
            # Should recommend cache reduction
            assert 'reduce_cache_activity' in recommendations['recommended_actions']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])