#!/usr/bin/env python3
"""
Steam Deck Thermal Optimizer
Hardware-specific thermal management and performance optimization
"""

import os
import time
import logging
import threading
import subprocess
from typing import Dict, List, Optional, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import psutil

class SteamDeckModel(Enum):
    """Steam Deck hardware models"""
    LCD = "lcd"           # 7nm APU (Jupiter)
    OLED = "oled"         # 6nm APU (Galileo)
    UNKNOWN = "unknown"

class ThermalZone(Enum):
    """Steam Deck thermal zones"""
    APU = "apu"
    CPU = "cpu" 
    GPU = "gpu"
    BATTERY = "battery"
    AMBIENT = "ambient"

class PowerProfile(Enum):
    """Steam Deck power profiles"""
    BATTERY_SAVER = "battery_saver"    # 3-5W TDP
    BALANCED = "balanced"              # 7-10W TDP
    PERFORMANCE = "performance"        # 12-15W TDP

@dataclass
class ThermalReading:
    """Thermal sensor reading"""
    zone: ThermalZone
    temperature: float
    timestamp: float
    threshold_breach: bool = False

@dataclass 
class ThermalLimits:
    """Steam Deck thermal limits"""
    normal_temp: float = 75.0      # Normal operating temperature
    warning_temp: float = 85.0     # Start throttling
    critical_temp: float = 95.0    # Emergency throttling
    shutdown_temp: float = 105.0   # Hardware protection

class SteamDeckThermalOptimizer:
    """
    Steam Deck specific thermal management system
    
    Features:
    - Hardware model detection (LCD vs OLED)
    - APU thermal monitoring 
    - Power profile awareness
    - Gaming mode thermal management
    - Background workload throttling
    - Fan curve optimization hints
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Hardware detection
        self.model = self._detect_steam_deck_model()
        self.power_profile = self._detect_power_profile()
        
        # Thermal configuration
        self.limits = self._get_thermal_limits()
        self.sensor_paths = self._discover_thermal_sensors()
        
        # State tracking
        self.current_readings: Dict[ThermalZone, ThermalReading] = {}
        self.thermal_history: List[ThermalReading] = []
        self.max_history_size = 100
        
        # Optimization state
        self.background_throttle_active = False
        self.emergency_mode_active = False
        self.last_optimization_time = 0
        
        # Threading
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        
        self.logger.info(f"Steam Deck thermal optimizer initialized: {self.model.value} model")
    
    def _detect_steam_deck_model(self) -> SteamDeckModel:
        """Detect Steam Deck hardware model"""
        try:
            # Check DMI product name
            dmi_path = Path("/sys/class/dmi/id/product_name")
            if dmi_path.exists():
                product_name = dmi_path.read_text().strip().lower()
                
                if "galileo" in product_name:
                    return SteamDeckModel.OLED
                elif "jupiter" in product_name:
                    return SteamDeckModel.LCD
                
            # Check board name as fallback
            board_path = Path("/sys/class/dmi/id/board_name")
            if board_path.exists():
                board_name = board_path.read_text().strip().lower()
                
                if "galileo" in board_name:
                    return SteamDeckModel.OLED
                elif "jupiter" in board_name:
                    return SteamDeckModel.LCD
        
        except Exception as e:
            self.logger.warning(f"Could not detect Steam Deck model: {e}")
        
        return SteamDeckModel.UNKNOWN
    
    def _detect_power_profile(self) -> PowerProfile:
        """Detect current power profile"""
        try:
            # Check battery status
            battery_path = Path("/sys/class/power_supply/BAT1/status")
            if battery_path.exists():
                status = battery_path.read_text().strip()
                if status == "Discharging":
                    # Check battery level for profile hints
                    capacity_path = Path("/sys/class/power_supply/BAT1/capacity")
                    if capacity_path.exists():
                        capacity = int(capacity_path.read_text().strip())
                        if capacity < 20:
                            return PowerProfile.BATTERY_SAVER
                        elif capacity < 60:
                            return PowerProfile.BALANCED
                
            # Check for performance indicators
            # High CPU frequency usually indicates performance mode
            cpu_freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            if cpu_freq_path.exists():
                freq_khz = int(cpu_freq_path.read_text().strip())
                freq_ghz = freq_khz / 1000000
                
                if freq_ghz > 3.0:
                    return PowerProfile.PERFORMANCE
                elif freq_ghz < 2.0:
                    return PowerProfile.BATTERY_SAVER
        
        except Exception as e:
            self.logger.warning(f"Could not detect power profile: {e}")
        
        return PowerProfile.BALANCED
    
    def _get_thermal_limits(self) -> ThermalLimits:
        """Get thermal limits based on Steam Deck model with OLED optimizations"""
        if self.model == SteamDeckModel.OLED:
            # OLED model (6nm Phoenix APU) runs cooler and more efficiently
            # Enhanced thermal design allows for higher sustained performance
            return ThermalLimits(
                normal_temp=68.0,      # Lower baseline for better sustained performance
                warning_temp=84.0,     # Higher threshold due to better cooling
                critical_temp=94.0,    # Higher critical temp
                shutdown_temp=104.0    # Higher emergency threshold
            )
        elif self.model == SteamDeckModel.LCD:
            # LCD model (7nm Van Gogh APU) runs hotter
            return ThermalLimits(
                normal_temp=75.0,
                warning_temp=85.0,
                critical_temp=95.0,
                shutdown_temp=105.0
            )
        else:
            # Conservative defaults for unknown models
            return ThermalLimits(
                normal_temp=70.0,
                warning_temp=80.0,
                critical_temp=90.0,
                shutdown_temp=100.0
            )
    
    def _discover_thermal_sensors(self) -> Dict[ThermalZone, str]:
        """Discover available thermal sensors"""
        sensor_paths = {}
        
        # Common Steam Deck thermal sensor locations
        sensor_candidates = [
            ("/sys/class/thermal/thermal_zone0/temp", ThermalZone.APU),
            ("/sys/class/thermal/thermal_zone1/temp", ThermalZone.CPU),
            ("/sys/class/thermal/thermal_zone2/temp", ThermalZone.GPU),
            ("/sys/class/hwmon/hwmon0/temp1_input", ThermalZone.APU),
            ("/sys/class/hwmon/hwmon1/temp1_input", ThermalZone.CPU),
            ("/sys/devices/pci0000:00/0000:00:08.1/0000:04:00.0/hwmon/hwmon*/temp1_input", ThermalZone.GPU),
        ]
        
        for path_pattern, zone in sensor_candidates:
            if "*" in path_pattern:
                # Handle wildcard patterns
                from glob import glob
                paths = glob(path_pattern)
                if paths:
                    sensor_paths[zone] = paths[0]
            else:
                if Path(path_pattern).exists():
                    sensor_paths[zone] = path_pattern
        
        # Log discovered sensors
        for zone, path in sensor_paths.items():
            self.logger.info(f"Thermal sensor discovered: {zone.value} -> {path}")
        
        return sensor_paths
    
    def _read_thermal_sensor(self, path: str) -> Optional[float]:
        """Read temperature from thermal sensor"""
        try:
            with open(path, 'r') as f:
                temp_raw = int(f.read().strip())
                # Convert millidegrees to degrees if necessary
                return temp_raw / 1000.0 if temp_raw > 1000 else temp_raw
        except Exception as e:
            self.logger.warning(f"Failed to read thermal sensor {path}: {e}")
            return None
    
    def get_thermal_readings(self) -> Dict[ThermalZone, ThermalReading]:
        """Get current thermal readings from all sensors"""
        readings = {}
        current_time = time.time()
        
        for zone, path in self.sensor_paths.items():
            temp = self._read_thermal_sensor(path)
            if temp is not None:
                reading = ThermalReading(
                    zone=zone,
                    temperature=temp,
                    timestamp=current_time,
                    threshold_breach=temp > self.limits.warning_temp
                )
                readings[zone] = reading
                
                # Update history
                self.thermal_history.append(reading)
                if len(self.thermal_history) > self.max_history_size:
                    self.thermal_history.pop(0)
        
        self.current_readings = readings
        return readings
    
    def get_max_temperature(self) -> float:
        """Get maximum temperature across all sensors"""
        readings = self.get_thermal_readings()
        if not readings:
            return 70.0  # Safe default
        
        return max(reading.temperature for reading in readings.values())
    
    def get_thermal_state(self) -> str:
        """Get current thermal state description"""
        max_temp = self.get_max_temperature()
        
        if max_temp >= self.limits.shutdown_temp:
            return "shutdown"
        elif max_temp >= self.limits.critical_temp:
            return "critical"
        elif max_temp >= self.limits.warning_temp:
            return "throttling"
        elif max_temp >= self.limits.normal_temp:
            return "warm"
        else:
            return "cool"
    
    def should_throttle_background_work(self) -> bool:
        """Determine if background work should be throttled"""
        max_temp = self.get_max_temperature()
        
        # Throttle if temperature is high
        if max_temp >= self.limits.warning_temp:
            return True
        
        # Throttle if on battery with low charge
        if self.power_profile == PowerProfile.BATTERY_SAVER:
            return True
        
        # Check if gaming (high GPU utilization)
        try:
            gpu_busy_path = "/sys/class/drm/card0/device/gpu_busy_percent"
            if Path(gpu_busy_path).exists():
                with open(gpu_busy_path, 'r') as f:
                    gpu_usage = int(f.read().strip())
                    if gpu_usage > 50:  # Gaming likely active
                        return True
        except Exception:
            pass
        
        return False
    
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count based on thermal state with OLED model optimizations"""
        thermal_state = self.get_thermal_state()
        
        # OLED model can handle more threads due to better cooling
        if self.model == SteamDeckModel.OLED:
            if thermal_state == "critical":
                return 1  # Minimal threading
            elif thermal_state == "throttling":
                return 3  # Better thermal headroom allows more threads
            elif thermal_state == "warm":
                return 6  # OLED can sustain higher performance
            else:
                return 8  # Maximum threading for OLED (better cooling)
        else:
            # LCD or unknown model - conservative threading
            if thermal_state == "critical":
                return 1  # Minimal threading
            elif thermal_state == "throttling":
                return 2  # Reduced threading
            elif thermal_state == "warm":
                return 4  # Normal threading
            else:
                return 6  # Maximum threading for LCD
    
    def get_ml_performance_target(self) -> str:
        """Get ML performance target based on thermal/power state"""
        if self.should_throttle_background_work():
            return "power_efficient"  # Lower accuracy, faster inference
        elif self.get_thermal_state() in ["cool", "warm"]:
            return "balanced"  # Normal accuracy and performance
        else:
            return "minimal"  # Fastest inference, lowest power
    
    def optimize_for_gaming(self) -> Dict[str, Any]:
        """Apply gaming-specific thermal optimizations with OLED model enhancements"""
        thermal_state = self.get_thermal_state()
        
        # OLED model can sustain better performance during gaming
        if self.model == SteamDeckModel.OLED:
            optimizations = {
                "background_threads": 3,  # OLED can handle more background work
                "ml_threads": 2,          # Better cooling allows dual ML threads
                "compilation_paused": thermal_state in ["throttling", "critical"],
                "memory_aggressive_gc": False,  # Less aggressive GC due to better thermal headroom
                "thermal_monitoring_interval": 1.5,  # More frequent monitoring for OLED
                "shader_cache_priority": "high",  # Prioritize shader compilation
                "gpu_memory_optimization": True,  # Optimize unified memory usage
                "predictive_cooling": True  # Enable predictive thermal management
            }
        else:
            # LCD or unknown model - conservative settings
            optimizations = {
                "background_threads": 2,  # Minimal background work
                "ml_threads": 1,          # Single ML thread
                "compilation_paused": thermal_state in ["throttling", "critical"],
                "memory_aggressive_gc": True,
                "thermal_monitoring_interval": 2.0,  # Standard monitoring
                "shader_cache_priority": "normal",
                "gpu_memory_optimization": False,
                "predictive_cooling": False
            }
        
        self.logger.info(f"Gaming optimizations applied for {self.model.value}: {optimizations}")
        return optimizations
    
    def get_power_budget_estimate(self) -> float:
        """Estimate available power budget for background work with OLED optimizations"""
        thermal_state = self.get_thermal_state()
        
        # OLED model has better power efficiency
        if self.model == SteamDeckModel.OLED:
            if self.power_profile == PowerProfile.BATTERY_SAVER:
                base_budget = 1.2  # Better efficiency allows slightly higher budget
            elif self.power_profile == PowerProfile.BALANCED:
                base_budget = 3.0  # Higher sustained performance
            else:  # PERFORMANCE
                base_budget = 5.0  # OLED can sustain higher power draw
        else:
            # LCD or unknown model
            if self.power_profile == PowerProfile.BATTERY_SAVER:
                base_budget = 1.0  # 1W for background work
            elif self.power_profile == PowerProfile.BALANCED:
                base_budget = 2.5  # 2.5W for background work
            else:  # PERFORMANCE
                base_budget = 4.0  # 4W for background work
        
        # Reduce budget based on thermal state
        thermal_multiplier = {
            "cool": 1.0,
            "warm": 0.8,
            "throttling": 0.5,
            "critical": 0.2,
            "shutdown": 0.0
        }.get(thermal_state, 0.5)
        
        return base_budget * thermal_multiplier
    
    def start_monitoring(self, interval: float = 5.0):
        """Start thermal monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._shutdown_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            name="thermal_monitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Thermal monitoring started")
    
    def _monitoring_loop(self, interval: float):
        """Main thermal monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Get current readings
                readings = self.get_thermal_readings()
                
                # Check for thermal emergencies
                max_temp = max((r.temperature for r in readings.values()), default=70.0)
                
                if max_temp >= self.limits.critical_temp and not self.emergency_mode_active:
                    self.logger.warning(f"Thermal emergency: {max_temp:.1f}°C - activating emergency throttling")
                    self.emergency_mode_active = True
                elif max_temp < self.limits.warning_temp and self.emergency_mode_active:
                    self.logger.info("Thermal emergency cleared")
                    self.emergency_mode_active = False
                
                # Update background throttling
                should_throttle = self.should_throttle_background_work()
                if should_throttle != self.background_throttle_active:
                    self.background_throttle_active = should_throttle
                    if should_throttle:
                        self.logger.info("Background work throttling activated")
                    else:
                        self.logger.info("Background work throttling deactivated")
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Thermal monitoring error: {e}")
                time.sleep(interval * 2)
    
    def stop_monitoring(self):
        """Stop thermal monitoring"""
        if self._monitoring_thread:
            self._shutdown_event.set()
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("Thermal monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive thermal status"""
        readings = self.get_thermal_readings()
        
        return {
            "model": self.model.value,
            "power_profile": self.power_profile.value,
            "thermal_state": self.get_thermal_state(),
            "max_temperature": self.get_max_temperature(),
            "should_throttle": self.should_throttle_background_work(),
            "optimal_threads": self.get_optimal_thread_count(),
            "power_budget_watts": self.get_power_budget_estimate(),
            "emergency_mode": self.emergency_mode_active,
            "thermal_readings": {
                zone.value: reading.temperature 
                for zone, reading in readings.items()
            },
            "thermal_limits": {
                "normal": self.limits.normal_temp,
                "warning": self.limits.warning_temp,
                "critical": self.limits.critical_temp
            }
        }


# Global thermal optimizer instance
_thermal_optimizer = None

def get_thermal_optimizer() -> SteamDeckThermalOptimizer:
    """Get or create global thermal optimizer"""
    global _thermal_optimizer
    if _thermal_optimizer is None:
        _thermal_optimizer = SteamDeckThermalOptimizer()
    return _thermal_optimizer


if __name__ == "__main__":
    # Test thermal optimizer
    logging.basicConfig(level=logging.INFO)
    
    print("🌡️  Steam Deck Thermal Optimizer Test")
    print("=" * 40)
    
    optimizer = get_thermal_optimizer()
    
    # Get initial status
    status = optimizer.get_status()
    print(f"Model: {status['model']}")
    print(f"Power Profile: {status['power_profile']}")
    print(f"Thermal State: {status['thermal_state']}")
    print(f"Max Temperature: {status['max_temperature']:.1f}°C")
    print(f"Optimal Threads: {status['optimal_threads']}")
    print(f"Power Budget: {status['power_budget_watts']:.1f}W")
    
    # Show thermal readings
    if status['thermal_readings']:
        print("\nThermal Readings:")
        for zone, temp in status['thermal_readings'].items():
            print(f"  {zone}: {temp:.1f}°C")
    
    # Test gaming optimizations
    gaming_opts = optimizer.optimize_for_gaming()
    print(f"\nGaming Optimizations: {gaming_opts}")
    
    print("\n✅ Thermal optimizer test completed")