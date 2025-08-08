#!/usr/bin/env python3
"""
Unified Thermal Management System for Steam Deck
Consolidates thermal management capabilities from multiple implementations
into a single, optimized system for shader compilation thermal awareness.
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import math

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDeckModel(Enum):
    """Steam Deck hardware models with thermal characteristics"""
    LCD = "lcd"      # Original LCD model (Van Gogh APU)
    OLED = "oled"    # OLED model (Phoenix APU)
    UNKNOWN = "unknown"


class ThermalState(Enum):
    """Thermal states for compilation scheduling"""
    COOL = "cool"        # < 65°C - Full performance
    NORMAL = "normal"    # 65-80°C - Standard operation  
    WARM = "warm"        # 80-85°C - Reduced background work
    HOT = "hot"          # 85-90°C - Essential shaders only
    THROTTLING = "throttling"  # 90-95°C - Compilation paused
    CRITICAL = "critical"      # > 95°C - Emergency stop


class PowerProfile(Enum):
    """Power management profiles"""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVER = "power_saver"
    BATTERY_SAVER = "battery_saver"
    THERMAL_LIMIT = "thermal_limit"


@dataclass
class ThermalSensorReading:
    """Individual thermal sensor reading"""
    sensor_name: str
    temperature_celsius: float
    timestamp: float
    zone: str = "unknown"
    
    
@dataclass
class ThermalSnapshot:
    """Complete thermal state snapshot"""
    apu_temp: float
    cpu_temp: float
    gpu_temp: float
    skin_temp: float
    fan_rpm: int
    power_draw: float
    battery_temp: float
    thermal_state: ThermalState
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompilationThermalConfig:
    """Thermal-aware compilation configuration"""
    max_threads: int
    max_queue_size: int
    max_compilation_time: float  # seconds
    pause_between_shaders: float  # seconds
    memory_limit_mb: int


class SteamDeckThermalProfiles:
    """Hardware-specific thermal profiles"""
    
    LCD_PROFILE = {
        'thermal_limits': {
            'apu_max': 95.0,      # APU junction temperature
            'cpu_max': 85.0,      # CPU cores
            'gpu_max': 90.0,      # GPU cores
            'skin_max': 50.0,     # Surface temperature
            'battery_max': 60.0   # Battery temperature
        },
        'fan_curve': {  # temp -> rpm percentage
            60: 30, 65: 40, 70: 50, 75: 65, 80: 80, 85: 90, 90: 100
        },
        'power_limits': {
            'total_max': 15.0,     # Total system power (watts)
            'apu_max': 10.0,       # APU power
            'cpu_max': 6.0,        # CPU power
            'gpu_max': 8.0         # GPU power
        },
        'compilation_configs': {
            ThermalState.COOL: CompilationThermalConfig(4, 20, 5.0, 0.1, 512),
            ThermalState.NORMAL: CompilationThermalConfig(3, 15, 3.0, 0.2, 384),
            ThermalState.WARM: CompilationThermalConfig(2, 10, 2.0, 0.5, 256),
            ThermalState.HOT: CompilationThermalConfig(1, 5, 1.0, 1.0, 128),
            ThermalState.THROTTLING: CompilationThermalConfig(0, 0, 0.0, 0.0, 0),
            ThermalState.CRITICAL: CompilationThermalConfig(0, 0, 0.0, 0.0, 0)
        }
    }
    
    OLED_PROFILE = {
        'thermal_limits': {
            'apu_max': 97.0,      # Slightly higher for newer APU
            'cpu_max': 87.0,      # Improved cooling
            'gpu_max': 92.0,      # Better thermal design
            'skin_max': 52.0,     # Improved heat dissipation
            'battery_max': 60.0   # Same battery limits
        },
        'fan_curve': {  # More aggressive curve for better performance
            60: 25, 65: 35, 70: 45, 75: 60, 80: 75, 85: 85, 90: 100
        },
        'power_limits': {
            'total_max': 18.0,     # Higher power budget
            'apu_max': 12.0,       # More APU power
            'cpu_max': 8.0,        # Higher CPU power
            'gpu_max': 10.0        # Higher GPU power
        },
        'compilation_configs': {
            ThermalState.COOL: CompilationThermalConfig(6, 25, 6.0, 0.1, 768),
            ThermalState.NORMAL: CompilationThermalConfig(4, 20, 4.0, 0.15, 512),
            ThermalState.WARM: CompilationThermalConfig(3, 15, 3.0, 0.3, 384),
            ThermalState.HOT: CompilationThermalConfig(2, 8, 2.0, 0.8, 256),
            ThermalState.THROTTLING: CompilationThermalConfig(0, 0, 0.0, 0.0, 0),
            ThermalState.CRITICAL: CompilationThermalConfig(0, 0, 0.0, 0.0, 0)
        }
    }


class SteamDeckDetector:
    """Hardware detection for Steam Deck models"""
    
    @staticmethod
    def detect_model() -> SteamDeckModel:
        """Detect Steam Deck model"""
        try:
            # Check DMI information
            board_name_paths = [
                '/sys/devices/virtual/dmi/id/board_name',
                '/sys/class/dmi/id/board_name'
            ]
            
            for path in board_name_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        board_name = f.read().strip()
                    
                    if 'Jupiter' in board_name:
                        return SteamDeckModel.LCD
                    elif 'Galileo' in board_name:
                        return SteamDeckModel.OLED
                        
            # Check product name as fallback
            product_paths = [
                '/sys/devices/virtual/dmi/id/product_name',
                '/sys/class/dmi/id/product_name'
            ]
            
            for path in product_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        product_name = f.read().strip()
                    
                    if any(keyword in product_name.lower() for keyword in ['jupiter', 'steam', 'deck']):
                        # Default to LCD if we can't determine specific model
                        return SteamDeckModel.LCD
                        
        except Exception as e:
            logger.warning(f"Could not detect Steam Deck model: {e}")
            
        return SteamDeckModel.UNKNOWN
    
    @staticmethod
    def is_steam_deck() -> bool:
        """Check if running on Steam Deck"""
        return SteamDeckDetector.detect_model() != SteamDeckModel.UNKNOWN


class ThermalSensorManager:
    """Manages thermal sensor readings"""
    
    def __init__(self):
        self.sensor_paths = self._discover_sensors()
        self.reading_history = deque(maxlen=100)  # Keep last 100 readings
        
    def _discover_sensors(self) -> Dict[str, str]:
        """Discover available thermal sensors"""
        sensors = {}
        
        # Common thermal sensor paths on Steam Deck
        common_paths = [
            '/sys/class/thermal/thermal_zone0/temp',  # Usually APU
            '/sys/class/thermal/thermal_zone1/temp',  # Usually CPU
            '/sys/class/thermal/thermal_zone2/temp',  # Usually GPU
            '/sys/class/hwmon/hwmon0/temp1_input',    # Hardware monitor
            '/sys/class/hwmon/hwmon1/temp1_input',
            '/sys/class/hwmon/hwmon2/temp1_input',
        ]
        
        zone_names = ['apu', 'cpu', 'gpu', 'sensor3', 'sensor4', 'sensor5']
        
        for i, path in enumerate(common_paths):
            if os.path.exists(path):
                zone_name = zone_names[i] if i < len(zone_names) else f'sensor{i}'
                sensors[zone_name] = path
                
        return sensors
    
    def read_temperature(self, sensor_path: str) -> Optional[float]:
        """Read temperature from sensor path"""
        try:
            with open(sensor_path, 'r') as f:
                temp_millidegrees = int(f.read().strip())
            return temp_millidegrees / 1000.0
        except (FileNotFoundError, ValueError, PermissionError):
            return None
    
    def get_thermal_snapshot(self) -> ThermalSnapshot:
        """Get current thermal state snapshot"""
        readings = {}
        
        # Read all available sensors
        for zone, path in self.sensor_paths.items():
            temp = self.read_temperature(path)
            if temp is not None:
                readings[zone] = temp
        
        # Extract key temperatures (with fallbacks)
        apu_temp = readings.get('apu', readings.get('sensor0', 70.0))
        cpu_temp = readings.get('cpu', readings.get('sensor1', apu_temp))
        gpu_temp = readings.get('gpu', readings.get('sensor2', apu_temp))
        
        # Get fan RPM if available
        fan_rpm = self._get_fan_rpm()
        
        # Get power draw if available
        power_draw = self._get_power_draw()
        
        # Determine thermal state
        thermal_state = self._calculate_thermal_state(apu_temp)
        
        snapshot = ThermalSnapshot(
            apu_temp=apu_temp,
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            skin_temp=min(apu_temp - 20, 45.0),  # Estimate skin temp
            fan_rpm=fan_rpm,
            power_draw=power_draw,
            battery_temp=readings.get('battery', 40.0),
            thermal_state=thermal_state
        )
        
        self.reading_history.append(snapshot)
        return snapshot
    
    def _get_fan_rpm(self) -> int:
        """Get fan RPM if available"""
        fan_paths = [
            '/sys/class/hwmon/hwmon0/fan1_input',
            '/sys/class/hwmon/hwmon1/fan1_input',
            '/sys/class/hwmon/hwmon2/fan1_input',
        ]
        
        for path in fan_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        return int(f.read().strip())
            except (FileNotFoundError, ValueError, PermissionError):
                continue
        
        return 0
    
    def _get_power_draw(self) -> float:
        """Get power draw if available"""
        if HAS_PSUTIL:
            try:
                # Estimate power draw from CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                # Rough estimation: idle ~5W, full load ~15W
                estimated_power = 5.0 + (cpu_percent / 100.0) * 10.0
                return estimated_power
            except:
                pass
        
        return 10.0  # Default estimate
    
    def _calculate_thermal_state(self, apu_temp: float) -> ThermalState:
        """Calculate thermal state from APU temperature"""
        if apu_temp < 65:
            return ThermalState.COOL
        elif apu_temp < 80:
            return ThermalState.NORMAL
        elif apu_temp < 85:
            return ThermalState.WARM
        elif apu_temp < 90:
            return ThermalState.HOT
        elif apu_temp < 95:
            return ThermalState.THROTTLING
        else:
            return ThermalState.CRITICAL
    
    def get_thermal_trend(self, window_seconds: int = 30) -> float:
        """Get thermal trend (degrees/second) over time window"""
        if len(self.reading_history) < 2:
            return 0.0
        
        current_time = time.time()
        recent_readings = [
            r for r in self.reading_history 
            if current_time - r.timestamp <= window_seconds
        ]
        
        if len(recent_readings) < 2:
            return 0.0
        
        # Linear regression on temperature over time
        times = [r.timestamp for r in recent_readings]
        temps = [r.apu_temp for r in recent_readings]
        
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(temps)
        sum_xy = sum(t * temp for t, temp in zip(times, temps))
        sum_x2 = sum(t * t for t in times)
        
        # Calculate slope (trend)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


class UnifiedThermalManager:
    """Unified thermal management system"""
    
    def __init__(self):
        self.steam_deck_model = SteamDeckDetector.detect_model()
        self.sensor_manager = ThermalSensorManager()
        
        # Load appropriate profile
        if self.steam_deck_model == SteamDeckModel.OLED:
            self.profile = SteamDeckThermalProfiles.OLED_PROFILE
        else:
            self.profile = SteamDeckThermalProfiles.LCD_PROFILE
        
        self.current_power_profile = PowerProfile.BALANCED
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks = []
        
        logger.info(f"Initialized thermal manager for {self.steam_deck_model.value}")
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start thermal monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started thermal monitoring")
    
    def stop_monitoring(self):
        """Stop thermal monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Stopped thermal monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                snapshot = self.sensor_manager.get_thermal_snapshot()
                
                # Check for emergency conditions
                self._check_emergency_conditions(snapshot)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _check_emergency_conditions(self, snapshot: ThermalSnapshot):
        """Check for thermal emergency conditions"""
        limits = self.profile['thermal_limits']
        
        if snapshot.apu_temp > limits['apu_max']:
            logger.critical(f"APU temperature critical: {snapshot.apu_temp}°C")
            self._trigger_emergency_cooling()
        
        if snapshot.thermal_state == ThermalState.CRITICAL:
            logger.critical("Critical thermal state reached")
            self._trigger_emergency_cooling()
    
    def _trigger_emergency_cooling(self):
        """Trigger emergency cooling measures"""
        logger.warning("Triggering emergency cooling measures")
        
        # Set most restrictive power profile
        self.current_power_profile = PowerProfile.THERMAL_LIMIT
        
        # Could add additional measures like:
        # - Reduce CPU frequency
        # - Increase fan speed
        # - Pause all compilation
        # - Notify user
    
    def get_compilation_config(self, thermal_state: ThermalState = None) -> CompilationThermalConfig:
        """Get current compilation configuration based on thermal state"""
        if thermal_state is None:
            snapshot = self.sensor_manager.get_thermal_snapshot()
            thermal_state = snapshot.thermal_state
        
        return self.profile['compilation_configs'][thermal_state]
    
    def should_allow_compilation(self, estimated_duration_seconds: float = 1.0) -> bool:
        """Determine if shader compilation should be allowed"""
        snapshot = self.sensor_manager.get_thermal_snapshot()
        config = self.get_compilation_config(snapshot.thermal_state)
        
        # No compilation in critical states
        if snapshot.thermal_state in [ThermalState.THROTTLING, ThermalState.CRITICAL]:
            return False
        
        # Check if compilation duration is within limits
        if estimated_duration_seconds > config.max_compilation_time:
            return False
        
        # Check thermal trend - don't compile if heating up rapidly
        trend = self.sensor_manager.get_thermal_trend(30)  # 30 second window
        if trend > 0.5:  # Heating up more than 0.5°C per second
            return False
        
        return True
    
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count for current thermal state"""
        config = self.get_compilation_config()
        return config.max_threads
    
    def register_callback(self, callback: Callable[[ThermalSnapshot], None]):
        """Register callback for thermal updates"""
        self.callbacks.append(callback)
    
    def get_current_state(self) -> ThermalSnapshot:
        """Get current thermal state"""
        return self.sensor_manager.get_thermal_snapshot()
    
    def get_thermal_history(self, seconds: int = 300) -> List[ThermalSnapshot]:
        """Get thermal history for specified time period"""
        current_time = time.time()
        return [
            snapshot for snapshot in self.sensor_manager.reading_history
            if current_time - snapshot.timestamp <= seconds
        ]
    
    def estimate_cooling_time(self, target_temp: float) -> float:
        """Estimate time to cool to target temperature"""
        current_snapshot = self.get_current_state()
        if current_snapshot.apu_temp <= target_temp:
            return 0.0
        
        trend = self.sensor_manager.get_thermal_trend(60)  # 1 minute trend
        if trend >= 0:  # Not cooling
            return float('inf')
        
        temp_diff = current_snapshot.apu_temp - target_temp
        cooling_rate = abs(trend)
        
        return temp_diff / cooling_rate if cooling_rate > 0 else float('inf')


# Example usage and testing
if __name__ == "__main__":
    thermal_manager = UnifiedThermalManager()
    
    # Get current state
    state = thermal_manager.get_current_state()
    print(f"Current thermal state: {state.thermal_state.value}")
    print(f"APU temperature: {state.apu_temp:.1f}°C")
    print(f"Compilation allowed: {thermal_manager.should_allow_compilation()}")
    
    # Get compilation config
    config = thermal_manager.get_compilation_config()
    print(f"Max threads: {config.max_threads}")
    print(f"Memory limit: {config.memory_limit_mb}MB")
    
    # Start monitoring for testing
    def thermal_callback(snapshot):
        print(f"Thermal update: {snapshot.thermal_state.value} - {snapshot.apu_temp:.1f}°C")
    
    thermal_manager.register_callback(thermal_callback)
    thermal_manager.start_monitoring(interval_seconds=2.0)
    
    try:
        # Monitor for 30 seconds
        time.sleep(30)
    finally:
        thermal_manager.stop_monitoring()