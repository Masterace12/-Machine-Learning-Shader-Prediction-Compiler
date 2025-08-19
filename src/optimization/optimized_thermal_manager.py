#!/usr/bin/env python3
"""
Optimized Thermal Management System for Steam Deck
Simplified version with mock capabilities for systems without full hardware access
"""

import os
import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from collections import deque
from enum import Enum

# Import base classes
try:
    # Try relative import first (when used as package)
    from ..core.unified_ml_predictor import ThermalState, SteamDeckModel
except ImportError:
    try:
        # Try absolute import (when used as script)
        from src.core.unified_ml_predictor import ThermalState, SteamDeckModel
    except ImportError:
        # Fallback definitions if base module not available
        class ThermalState(Enum):
            COOL = "cool"
            OPTIMAL = "optimal"
            NORMAL = "normal"
            WARM = "warm"
            HOT = "hot"
            THROTTLING = "throttling"
            CRITICAL = "critical"
            PREDICTIVE_WARM = "predictive_warm"
        
        class SteamDeckModel(Enum):
            LCD = "lcd"
            OLED = "oled"
            UNKNOWN = "unknown"


@dataclass
class ThermalSample:
    """Single thermal measurement"""
    timestamp: float
    apu_temp: float
    cpu_temp: float
    gpu_temp: float
    fan_rpm: int
    power_draw: float
    battery_level: float
    gaming_active: bool
    compilation_threads: int
    
    def __post_init__(self):
        """Validate thermal data"""
        self.apu_temp = max(0, min(150, self.apu_temp))
        self.cpu_temp = max(0, min(150, self.cpu_temp))
        self.gpu_temp = max(0, min(150, self.gpu_temp))


class OptimizedThermalManager:
    """Simplified thermal manager with mock capabilities"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize thermal manager"""
        self.config_path = config_path or Path.home() / '.config' / 'shader-predict-compile' / 'thermal.json'
        
        # Logger (initialize first)
        self.logger = logging.getLogger(__name__ if __name__ != '__main__' else 'optimized_thermal_manager')
        
        # Hardware detection
        self.steam_deck_model = self._detect_steam_deck_model()
        self.sensor_paths = self._discover_thermal_sensors()
        
        # Current state
        self.current_state = ThermalState.NORMAL
        self.thermal_history = deque(maxlen=100)
        
        # Compilation control
        self.max_compilation_threads = 4
        self.compilation_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_interval = 1.0
        self._monitoring_thread = None
        self._lock = threading.RLock()
        
        # Mock data for systems without hardware access
        self._mock_mode = len(self.sensor_paths) == 0
        self._mock_temp = 70.0  # Starting mock temperature
        
        if self._mock_mode:
            self.logger.info("Running in mock mode - no thermal sensors detected")
    
    def _detect_steam_deck_model(self) -> SteamDeckModel:
        """Detect Steam Deck model"""
        try:
            dmi_path = Path("/sys/class/dmi/id/product_name")
            if dmi_path.exists():
                product_name = dmi_path.read_text().strip().lower()
                if "jupiter" in product_name or "steamdeck" in product_name:
                    # Try to detect OLED vs LCD
                    try:
                        lspci_result = os.popen("lspci 2>/dev/null | grep VGA").read().lower()
                        if "1002:163f" in lspci_result:  # Van Gogh
                            return SteamDeckModel.LCD
                        elif "1002:15bf" in lspci_result:  # Phoenix
                            return SteamDeckModel.OLED
                    except:
                        pass
                    return SteamDeckModel.LCD  # Default to LCD
        except Exception:
            pass
        
        return SteamDeckModel.UNKNOWN
    
    def _discover_thermal_sensors(self) -> Dict[str, Path]:
        """Discover available thermal sensors"""
        sensors = {}
        
        # Common thermal sensor paths
        sensor_candidates = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
            "/sys/class/hwmon/hwmon0/temp1_input",
            "/sys/class/hwmon/hwmon1/temp1_input",
            "/sys/class/hwmon/hwmon2/temp1_input",
        ]
        
        for i, path_str in enumerate(sensor_candidates):
            path = Path(path_str)
            if path.exists():
                try:
                    # Test read to ensure it's accessible
                    test_value = int(path.read_text().strip())
                    if 0 < test_value < 200000:  # Reasonable temperature range (0-200°C in millidegrees)
                        sensors[f"thermal_sensor_{i}"] = path
                except Exception:
                    pass
        
        self.logger.info(f"Discovered {len(sensors)} thermal sensors")
        return sensors
    
    def _read_sensors(self) -> Dict[str, float]:
        """Read thermal sensors or generate mock data"""
        if self._mock_mode:
            # Generate realistic mock thermal data
            import random
            
            # Simulate temperature changes over time
            temp_change = random.uniform(-2.0, 2.0)
            self._mock_temp += temp_change * 0.1  # Gradual changes
            self._mock_temp = max(50.0, min(95.0, self._mock_temp))  # Clamp to realistic range
            
            return {
                "apu_temp": self._mock_temp + random.uniform(-2, 2),
                "cpu_temp": self._mock_temp + random.uniform(-3, 1),
                "gpu_temp": self._mock_temp + random.uniform(-1, 3),
                "fan_rpm": max(0, int(2000 + (self._mock_temp - 70) * 50 + random.uniform(-200, 200)))
            }
        
        readings = {}
        for sensor_name, sensor_path in self.sensor_paths.items():
            try:
                value = int(sensor_path.read_text().strip())
                # Convert millidegrees to degrees
                readings[sensor_name] = value / 1000.0 if value > 1000 else value
            except Exception as e:
                self.logger.debug(f"Could not read sensor {sensor_name}: {e}")
        
        return readings
    
    def _get_current_sample(self) -> ThermalSample:
        """Get current thermal sample"""
        sensor_data = self._read_sensors()
        
        # Extract temperatures
        apu_temp = sensor_data.get("apu_temp", 70.0)
        cpu_temp = sensor_data.get("cpu_temp", apu_temp)
        gpu_temp = sensor_data.get("gpu_temp", apu_temp)
        fan_rpm = int(sensor_data.get("fan_rpm", 2000))
        
        # Mock additional data
        power_draw = 10.0 + (apu_temp - 50) * 0.2  # Estimate power from temperature
        battery_level = 80.0  # Mock battery level
        gaming_active = False  # Simplified detection
        
        return ThermalSample(
            timestamp=time.time(),
            apu_temp=apu_temp,
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            fan_rpm=fan_rpm,
            power_draw=power_draw,
            battery_level=battery_level,
            gaming_active=gaming_active,
            compilation_threads=self.max_compilation_threads
        )
    
    def _determine_thermal_state(self, sample: ThermalSample) -> ThermalState:
        """Determine thermal state from sample"""
        temp = sample.apu_temp
        
        if temp >= 95:
            return ThermalState.CRITICAL
        elif temp >= 90:
            return ThermalState.THROTTLING
        elif temp >= 85:
            return ThermalState.HOT
        elif temp >= 80:
            return ThermalState.WARM
        elif temp >= 70:
            return ThermalState.NORMAL
        elif temp >= 60:
            return ThermalState.OPTIMAL
        else:
            return ThermalState.COOL
    
    def _update_compilation_threads(self):
        """Update compilation thread count based on thermal state"""
        previous_threads = self.max_compilation_threads
        
        state_thread_map = {
            ThermalState.COOL: 6,
            ThermalState.OPTIMAL: 4,
            ThermalState.NORMAL: 4,
            ThermalState.WARM: 2,
            ThermalState.HOT: 1,
            ThermalState.THROTTLING: 0,
            ThermalState.CRITICAL: 0
        }
        
        self.max_compilation_threads = state_thread_map.get(self.current_state, 2)
        
        # Notify callbacks if changed
        if self.max_compilation_threads != previous_threads:
            for callback in self.compilation_callbacks:
                try:
                    callback(self.max_compilation_threads, self.current_state)
                except Exception as e:
                    self.logger.error(f"Compilation callback error: {e}")
    
    def add_compilation_callback(self, callback: Callable[[int, ThermalState], None]):
        """Add callback for compilation thread changes"""
        self.compilation_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start thermal monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    sample = self._get_current_sample()
                    
                    with self._lock:
                        self.thermal_history.append(sample)
                        new_state = self._determine_thermal_state(sample)
                        
                        if new_state != self.current_state:
                            self.logger.info(f"Thermal state: {self.current_state.value} → {new_state.value}")
                            self.current_state = new_state
                            self._update_compilation_threads()
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(self.monitoring_interval * 2)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Thermal monitoring started")
    
    def stop_monitoring(self):
        """Stop thermal monitoring"""
        self.monitoring_active = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2)
        
        self.logger.info("Thermal monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current thermal status"""
        with self._lock:
            current_sample = self.thermal_history[-1] if self.thermal_history else None
            
            return {
                "thermal_state": self.current_state.value,
                "steam_deck_model": self.steam_deck_model.value,
                "current_temps": {
                    "apu": current_sample.apu_temp if current_sample else 0.0,
                    "cpu": current_sample.cpu_temp if current_sample else 0.0,
                    "gpu": current_sample.gpu_temp if current_sample else 0.0
                },
                "fan_rpm": current_sample.fan_rpm if current_sample else 0,
                "power_draw": current_sample.power_draw if current_sample else 0.0,
                "battery_level": current_sample.battery_level if current_sample else 100.0,
                "compilation_threads": self.max_compilation_threads,
                "sensors_available": len(self.sensor_paths),
                "mock_mode": self._mock_mode,
                "gaming_active": current_sample.gaming_active if current_sample else False
            }


# Global instance
_thermal_manager = None


def get_thermal_manager() -> OptimizedThermalManager:
    """Get global thermal manager instance"""
    global _thermal_manager
    if _thermal_manager is None:
        _thermal_manager = OptimizedThermalManager()
    return _thermal_manager


if __name__ == "__main__":
    # Test thermal manager
    logging.basicConfig(level=logging.INFO)
    
    manager = get_thermal_manager()
    
    print("🌡️  Thermal Manager Test")
    print("=" * 30)
    
    # Show initial status
    status = manager.get_status()
    print(f"Steam Deck Model: {status['steam_deck_model']}")
    print(f"Mock Mode: {status['mock_mode']}")
    print(f"Sensors Available: {status['sensors_available']}")
    print(f"Current Temperature: {status['current_temps']['apu']:.1f}°C")
    print(f"Thermal State: {status['thermal_state']}")
    print(f"Compilation Threads: {status['compilation_threads']}")
    
    # Start monitoring for a few seconds
    manager.start_monitoring()
    
    try:
        print("\nMonitoring for 10 seconds...")
        for i in range(10):
            time.sleep(1)
            status = manager.get_status()
            print(f"[{i+1:2d}s] APU: {status['current_temps']['apu']:.1f}°C "
                  f"State: {status['thermal_state']} "
                  f"Threads: {status['compilation_threads']}")
    
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_monitoring()
        print("✓ Thermal manager test completed")