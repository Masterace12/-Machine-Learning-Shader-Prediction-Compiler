#!/usr/bin/env python3
"""
Optimized Thermal Management System for Steam Deck
Advanced thermal control with predictive modeling and game-specific profiles
"""

import os
import time
import threading
import asyncio
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import psutil

# Steam Deck specific imports
try:
    import pyudev
    HAS_UDEV = True
except ImportError:
    HAS_UDEV = False


class ThermalState(Enum):
    """Enhanced thermal states with predictive capabilities"""
    COOL = "cool"              # < 60°C - Aggressive compilation
    OPTIMAL = "optimal"        # 60-70°C - Full compilation capacity
    NORMAL = "normal"          # 70-80°C - Standard operation
    WARM = "warm"              # 80-85°C - Reduced background work
    HOT = "hot"                # 85-90°C - Essential shaders only
    THROTTLING = "throttling"  # 90-95°C - Compilation paused
    CRITICAL = "critical"      # > 95°C - Emergency shutdown
    PREDICTIVE_WARM = "predictive_warm"  # Predicted to become warm


class SteamDeckModel(Enum):
    """Steam Deck models with thermal characteristics"""
    LCD = "lcd"     # Van Gogh APU - 7nm, higher thermals
    OLED = "oled"   # Phoenix APU - 6nm, better thermal efficiency
    UNKNOWN = "unknown"


class PowerProfile(Enum):
    """Power management profiles"""
    BATTERY_SAVER = "battery_saver"      # Minimal compilation
    BALANCED = "balanced"                # Standard compilation
    PERFORMANCE = "performance"          # Maximum compilation
    GAMING = "gaming"                    # Game-optimized
    DOCKED = "docked"                   # AC power, maximum performance


@dataclass
class ThermalSample:
    """Single thermal measurement with metadata"""
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
        if self.apu_temp < 0 or self.apu_temp > 150:
            self.apu_temp = max(0, min(150, self.apu_temp))


@dataclass
class ThermalProfile:
    """Thermal management profile for specific scenarios"""
    name: str
    description: str
    
    # Temperature limits
    temp_limits: Dict[str, float] = field(default_factory=dict)
    
    # Compilation settings
    max_compilation_threads: int = 4
    compilation_priority: str = "normal"  # low, normal, high
    background_compilation: bool = True
    
    # Predictive settings
    enable_prediction: bool = True
    prediction_window_seconds: int = 30
    thermal_trend_threshold: float = 2.0  # °C/min
    
    # Power management
    max_power_watts: float = 15.0
    battery_threshold_percent: float = 20.0
    
    def __post_init__(self):
        """Set default temperature limits if not provided"""
        if not self.temp_limits:
            self.temp_limits = {
                "apu_max": 95.0,
                "cpu_max": 85.0,
                "gpu_max": 90.0,
                "predictive_threshold": 80.0
            }


class GameSpecificProfiles:
    """Game-specific thermal profiles"""
    
    def __init__(self):
        self.profiles = {
            # High-performance games
            "1091500": ThermalProfile(  # Cyberpunk 2077
                name="cyberpunk_2077",
                description="High GPU load, aggressive thermal management",
                temp_limits={"apu_max": 92.0, "cpu_max": 80.0, "gpu_max": 88.0},
                max_compilation_threads=2,
                background_compilation=False,
                prediction_window_seconds=60
            ),
            "1245620": ThermalProfile(  # Elden Ring
                name="elden_ring", 
                description="CPU intensive, moderate thermal control",
                temp_limits={"apu_max": 94.0, "cpu_max": 82.0, "gpu_max": 89.0},
                max_compilation_threads=3,
                background_compilation=True,
                prediction_window_seconds=45
            ),
            # Less demanding games
            "1145360": ThermalProfile(  # Hades
                name="hades",
                description="Light load, aggressive compilation",
                temp_limits={"apu_max": 97.0, "cpu_max": 87.0, "gpu_max": 92.0},
                max_compilation_threads=6,
                background_compilation=True,
                prediction_window_seconds=20
            )
        }
    
    def get_profile(self, game_id: str) -> Optional[ThermalProfile]:
        """Get thermal profile for specific game"""
        return self.profiles.get(game_id)


class ThermalPredictor:
    """Predictive thermal modeling using moving averages and trend analysis"""
    
    def __init__(self, history_size: int = 100):
        self.history = deque(maxlen=history_size)
        self.trend_window = 30  # seconds
        self._prediction_cache = {}
        self._cache_expiry = 5.0  # seconds
    
    def add_sample(self, sample: ThermalSample):
        """Add thermal sample to history"""
        self.history.append(sample)
        # Clear cache when new data arrives
        self._prediction_cache.clear()
    
    def predict_temperature(self, prediction_horizon: int = 30) -> Tuple[float, float]:
        """
        Predict temperature after given time horizon
        
        Returns:
            (predicted_apu_temp, confidence)
        """
        cache_key = f"temp_{prediction_horizon}"
        
        # Check cache
        if cache_key in self._prediction_cache:
            cached_time, result = self._prediction_cache[cache_key]
            if time.time() - cached_time < self._cache_expiry:
                return result
        
        if len(self.history) < 10:
            # Not enough data for prediction
            current_temp = self.history[-1].apu_temp if self.history else 70.0
            return current_temp, 0.1
        
        # Extract recent temperature data
        recent_samples = list(self.history)[-self.trend_window:]
        temperatures = [s.apu_temp for s in recent_samples]
        timestamps = [s.timestamp for s in recent_samples]
        
        # Calculate trend using simple linear regression
        n = len(temperatures)
        if n < 3:
            return temperatures[-1], 0.2
        
        # Normalize timestamps
        t_start = timestamps[0]
        t_norm = [(t - t_start) for t in timestamps]
        
        # Linear regression coefficients
        sum_t = sum(t_norm)
        sum_t2 = sum(t * t for t in t_norm)
        sum_temp = sum(temperatures)
        sum_t_temp = sum(t * temp for t, temp in zip(t_norm, temperatures))
        
        # Calculate slope (temperature change rate)
        denominator = n * sum_t2 - sum_t * sum_t
        if abs(denominator) < 1e-10:
            # No trend
            return temperatures[-1], 0.3
        
        slope = (n * sum_t_temp - sum_t * sum_temp) / denominator
        intercept = (sum_temp - slope * sum_t) / n
        
        # Predict temperature
        future_time = t_norm[-1] + prediction_horizon
        predicted_temp = slope * future_time + intercept
        
        # Calculate confidence based on trend consistency
        predicted_temps = [slope * t + intercept for t in t_norm]
        residuals = [actual - predicted for actual, predicted in zip(temperatures, predicted_temps)]
        mse = sum(r * r for r in residuals) / len(residuals)
        confidence = max(0.1, min(0.9, 1.0 / (1.0 + mse)))
        
        # Clamp prediction to reasonable range
        current_temp = temperatures[-1]
        max_change = abs(slope) * prediction_horizon + 2.0  # Allow some uncertainty
        predicted_temp = max(
            current_temp - max_change,
            min(current_temp + max_change, predicted_temp)
        )
        
        result = (predicted_temp, confidence)
        self._prediction_cache[cache_key] = (time.time(), result)
        return result
    
    def calculate_thermal_trend(self, window_seconds: int = 30) -> float:
        """Calculate thermal trend in °C per minute"""
        if len(self.history) < 2:
            return 0.0
        
        # Get samples within time window
        current_time = time.time()
        window_samples = [
            s for s in self.history
            if current_time - s.timestamp <= window_seconds
        ]
        
        if len(window_samples) < 2:
            return 0.0
        
        # Calculate temperature change rate
        first_sample = window_samples[0]
        last_sample = window_samples[-1]
        
        temp_change = last_sample.apu_temp - first_sample.apu_temp
        time_change = max(1.0, last_sample.timestamp - first_sample.timestamp)
        
        # Convert to °C per minute
        trend = (temp_change / time_change) * 60.0
        return trend


class OptimizedThermalManager:
    """High-performance thermal management with predictive capabilities"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize thermal manager
        
        Args:
            config_path: Path to thermal configuration file
        """
        self.config_path = config_path or Path.home() / '.config' / 'shader-predict-compile' / 'thermal.json'
        
        # Hardware detection
        self.steam_deck_model = self._detect_steam_deck_model()
        self.power_profile = PowerProfile.BALANCED
        
        # Thermal monitoring
        self.sensor_paths = self._discover_thermal_sensors()
        self.current_state = ThermalState.NORMAL
        self.thermal_history = deque(maxlen=1000)  # 1000 samples ~16 minutes at 1Hz
        
        # Predictive modeling
        self.predictor = ThermalPredictor(history_size=200)
        self.game_profiles = GameSpecificProfiles()
        self.active_profile = self._get_default_profile()
        
        # Compilation control
        self.max_compilation_threads = 4
        self.compilation_paused = False
        self.compilation_callbacks: List[Callable] = []
        
        # Async support
        self.monitoring_active = False
        self.monitoring_interval = 1.0  # seconds
        self.prediction_interval = 10.0  # seconds
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring_thread = None
        self._prediction_thread = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_config()
    
    def _detect_steam_deck_model(self) -> SteamDeckModel:
        """Detect Steam Deck model using DMI and hardware info"""
        try:
            # Check DMI product name
            dmi_path = Path("/sys/class/dmi/id/product_name")
            if dmi_path.exists():
                product_name = dmi_path.read_text().strip().lower()
                if "jupiter" in product_name or "steamdeck" in product_name:
                    # Check APU type via PCI device
                    try:
                        lspci_result = os.popen("lspci | grep VGA").read().lower()
                        if "1002:163f" in lspci_result:  # Van Gogh
                            return SteamDeckModel.LCD
                        elif "1002:15bf" in lspci_result:  # Phoenix
                            return SteamDeckModel.OLED
                    except:
                        pass
            
            # Fallback detection
            if Path("/dev/hwmon0").exists():
                return SteamDeckModel.LCD  # Assume LCD if hwmon exists
                
        except Exception as e:
            self.logger.debug(f"Steam Deck detection failed: {e}")
        
        return SteamDeckModel.UNKNOWN
    
    def _discover_thermal_sensors(self) -> Dict[str, Path]:
        """Discover available thermal sensors"""
        sensors = {}
        
        # Standard hwmon sensors
        for hwmon_dir in Path("/sys/class/hwmon").glob("hwmon*"):
            try:
                name_file = hwmon_dir / "name"
                if not name_file.exists():
                    continue
                
                sensor_name = name_file.read_text().strip()
                
                # Steam Deck specific sensors
                if sensor_name in ["k10temp", "amdgpu", "jupiter"]:
                    # Find temperature inputs
                    for temp_input in hwmon_dir.glob("temp*_input"):
                        label_file = hwmon_dir / temp_input.name.replace("input", "label")
                        if label_file.exists():
                            label = label_file.read_text().strip()
                            sensors[f"{sensor_name}_{label}"] = temp_input
                        else:
                            sensors[f"{sensor_name}_{temp_input.name}"] = temp_input
                
                # Fan sensors
                for fan_input in hwmon_dir.glob("fan*_input"):
                    sensors[f"{sensor_name}_fan"] = fan_input
                    
            except Exception as e:
                self.logger.debug(f"Sensor discovery error for {hwmon_dir}: {e}")
        
        # Fallback sensors
        if not sensors:
            # Try common sensor paths
            common_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp",
                "/sys/class/hwmon/hwmon0/temp1_input",
                "/sys/class/hwmon/hwmon1/temp1_input"
            ]
            
            for i, path_str in enumerate(common_paths):
                path = Path(path_str)
                if path.exists():
                    sensors[f"thermal_zone_{i}"] = path
        
        self.logger.info(f"Discovered {len(sensors)} thermal sensors")
        return sensors
    
    def _get_default_profile(self) -> ThermalProfile:
        """Get default thermal profile based on hardware"""
        if self.steam_deck_model == SteamDeckModel.LCD:
            return ThermalProfile(
                name="steamdeck_lcd_default",
                description="Conservative profile for LCD Steam Deck",
                temp_limits={
                    "apu_max": 95.0,
                    "cpu_max": 85.0,
                    "gpu_max": 90.0,
                    "predictive_threshold": 80.0
                },
                max_compilation_threads=4,
                max_power_watts=15.0,
                prediction_window_seconds=30
            )
        elif self.steam_deck_model == SteamDeckModel.OLED:
            return ThermalProfile(
                name="steamdeck_oled_default",
                description="Optimized profile for OLED Steam Deck",
                temp_limits={
                    "apu_max": 97.0,
                    "cpu_max": 87.0,
                    "gpu_max": 92.0,
                    "predictive_threshold": 82.0
                },
                max_compilation_threads=6,
                max_power_watts=18.0,
                prediction_window_seconds=25
            )
        else:
            return ThermalProfile(
                name="generic_default",
                description="Safe profile for unknown hardware",
                temp_limits={
                    "apu_max": 90.0,
                    "cpu_max": 80.0,
                    "gpu_max": 85.0,
                    "predictive_threshold": 75.0
                },
                max_compilation_threads=2,
                max_power_watts=12.0,
                prediction_window_seconds=60
            )
    
    def _load_config(self):
        """Load thermal configuration"""
        if self.config_path.exists():
            try:
                config = json.loads(self.config_path.read_text())
                
                # Update monitoring intervals
                self.monitoring_interval = config.get("monitoring_interval", 1.0)
                self.prediction_interval = config.get("prediction_interval", 10.0)
                
                # Update profile settings
                profile_config = config.get("active_profile", {})
                if profile_config:
                    self.active_profile.temp_limits.update(
                        profile_config.get("temp_limits", {})
                    )
                    self.active_profile.max_compilation_threads = profile_config.get(
                        "max_compilation_threads",
                        self.active_profile.max_compilation_threads
                    )
                
                self.logger.info("Thermal configuration loaded")
            except Exception as e:
                self.logger.warning(f"Could not load thermal config: {e}")
    
    def _save_config(self):
        """Save current thermal configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                "steam_deck_model": self.steam_deck_model.value,
                "monitoring_interval": self.monitoring_interval,
                "prediction_interval": self.prediction_interval,
                "active_profile": {
                    "name": self.active_profile.name,
                    "temp_limits": self.active_profile.temp_limits,
                    "max_compilation_threads": self.active_profile.max_compilation_threads,
                    "background_compilation": self.active_profile.background_compilation
                }
            }
            
            self.config_path.write_text(json.dumps(config, indent=2))
            self.logger.debug("Thermal configuration saved")
        except Exception as e:
            self.logger.error(f"Could not save thermal config: {e}")
    
    def _read_sensors(self) -> Dict[str, float]:
        """Read all available thermal sensors"""
        readings = {}
        
        for sensor_name, sensor_path in self.sensor_paths.items():
            try:
                value = int(sensor_path.read_text().strip())
                
                # Convert millidegrees to degrees for temperature sensors
                if "temp" in sensor_name:
                    readings[sensor_name] = value / 1000.0
                else:
                    readings[sensor_name] = float(value)
                    
            except Exception as e:
                self.logger.debug(f"Could not read sensor {sensor_name}: {e}")
        
        return readings
    
    def _get_current_sample(self) -> ThermalSample:
        """Get current thermal sample"""
        sensor_data = self._read_sensors()
        
        # Extract key temperatures
        apu_temp = 0.0
        cpu_temp = 0.0
        gpu_temp = 0.0
        fan_rpm = 0
        
        for sensor_name, value in sensor_data.items():
            if "k10temp" in sensor_name and "temp" in sensor_name:
                apu_temp = max(apu_temp, value)
            elif "amdgpu" in sensor_name and "temp" in sensor_name:
                gpu_temp = max(gpu_temp, value)
            elif "fan" in sensor_name:
                fan_rpm = max(fan_rpm, int(value))
        
        # Use highest temperature as CPU temp if not found specifically
        if cpu_temp == 0.0:
            cpu_temp = apu_temp
        
        # Get power and battery info
        power_draw = self._get_power_draw()
        battery_level = self._get_battery_level()
        gaming_active = self._is_gaming_active()
        
        return ThermalSample(
            timestamp=time.time(),
            apu_temp=apu_temp or 70.0,  # Fallback temperature
            cpu_temp=cpu_temp or 70.0,
            gpu_temp=gpu_temp or 70.0,
            fan_rpm=fan_rpm,
            power_draw=power_draw,
            battery_level=battery_level,
            gaming_active=gaming_active,
            compilation_threads=self.max_compilation_threads
        )
    
    def _get_power_draw(self) -> float:
        """Get current power draw in watts"""
        try:
            # Try Steam Deck specific power sensors
            for power_file in [
                "/sys/class/power_supply/BAT0/power_now",
                "/sys/class/power_supply/BAT1/power_now"
            ]:
                path = Path(power_file)
                if path.exists():
                    power_uw = int(path.read_text().strip())
                    return power_uw / 1_000_000  # Convert to watts
            
            # Fallback: estimate from CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            estimated_power = 5.0 + (cpu_percent / 100.0) * 10.0  # 5-15W estimate
            return estimated_power
            
        except Exception:
            return 10.0  # Conservative fallback
    
    def _get_battery_level(self) -> float:
        """Get battery level percentage"""
        try:
            battery = psutil.sensors_battery()
            return battery.percent if battery else 100.0
        except Exception:
            return 100.0
    
    def _is_gaming_active(self) -> bool:
        """Check if gaming is currently active"""
        try:
            # Check for common gaming processes
            gaming_processes = ["gamescope", "steam", "wine", "proton"]
            
            for proc in psutil.process_iter(['name', 'cpu_percent']):
                try:
                    name = proc.info['name'].lower()
                    cpu_usage = proc.info['cpu_percent']
                    
                    if any(game_proc in name for game_proc in gaming_processes):
                        if cpu_usage > 10:  # Active gaming process
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            return False
        except Exception:
            return False
    
    def _determine_thermal_state(self, sample: ThermalSample) -> ThermalState:
        """Determine thermal state from sample"""
        apu_temp = sample.apu_temp
        limits = self.active_profile.temp_limits
        
        # Critical temperature check
        if apu_temp >= limits["apu_max"]:
            return ThermalState.CRITICAL
        
        # Throttling check
        elif apu_temp >= limits["apu_max"] - 5:
            return ThermalState.THROTTLING
        
        # Hot state
        elif apu_temp >= limits["predictive_threshold"] + 5:
            return ThermalState.HOT
        
        # Predictive warm state
        elif self.active_profile.enable_prediction:
            predicted_temp, confidence = self.predictor.predict_temperature(
                self.active_profile.prediction_window_seconds
            )
            
            if confidence > 0.5 and predicted_temp >= limits["predictive_threshold"]:
                return ThermalState.PREDICTIVE_WARM
        
        # Warm state
        if apu_temp >= limits["predictive_threshold"]:
            return ThermalState.WARM
        
        # Normal operating ranges
        elif apu_temp >= 70:
            return ThermalState.NORMAL
        elif apu_temp >= 60:
            return ThermalState.OPTIMAL
        else:
            return ThermalState.COOL
    
    def _update_compilation_threads(self):
        """Update compilation thread count based on thermal state"""
        previous_threads = self.max_compilation_threads
        
        state_thread_map = {
            ThermalState.COOL: self.active_profile.max_compilation_threads + 2,
            ThermalState.OPTIMAL: self.active_profile.max_compilation_threads,
            ThermalState.NORMAL: self.active_profile.max_compilation_threads,
            ThermalState.WARM: max(1, self.active_profile.max_compilation_threads - 1),
            ThermalState.PREDICTIVE_WARM: max(1, self.active_profile.max_compilation_threads - 1),
            ThermalState.HOT: 1,
            ThermalState.THROTTLING: 0,
            ThermalState.CRITICAL: 0
        }
        
        self.max_compilation_threads = state_thread_map.get(
            self.current_state,
            self.active_profile.max_compilation_threads
        )
        
        # Apply power profile adjustments
        if self.power_profile == PowerProfile.BATTERY_SAVER:
            self.max_compilation_threads = max(0, self.max_compilation_threads - 2)
        elif self.power_profile == PowerProfile.PERFORMANCE:
            self.max_compilation_threads += 1
        
        # Notify callbacks of thread count change
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
                    # Get thermal sample
                    sample = self._get_current_sample()
                    
                    with self._lock:
                        # Update history
                        self.thermal_history.append(sample)
                        self.predictor.add_sample(sample)
                        
                        # Determine thermal state
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
        
        # Save configuration
        self._save_config()
        
        self.logger.info("Thermal monitoring stopped")
    
    def set_game_profile(self, game_id: str):
        """Set thermal profile for specific game"""
        profile = self.game_profiles.get_profile(game_id)
        if profile:
            with self._lock:
                self.active_profile = profile
                self.logger.info(f"Using thermal profile: {profile.name}")
        else:
            self.logger.debug(f"No specific thermal profile for game: {game_id}")
    
    def set_power_profile(self, profile: PowerProfile):
        """Set power management profile"""
        with self._lock:
            self.power_profile = profile
            self._update_compilation_threads()
            self.logger.info(f"Power profile: {profile.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current thermal status"""
        with self._lock:
            current_sample = self.thermal_history[-1] if self.thermal_history else None
            
            if current_sample:
                trend = self.predictor.calculate_thermal_trend()
                predicted_temp, confidence = self.predictor.predict_temperature()
            else:
                trend = 0.0
                predicted_temp, confidence = 0.0, 0.0
            
            return {
                "thermal_state": self.current_state.value,
                "steam_deck_model": self.steam_deck_model.value,
                "power_profile": self.power_profile.value,
                "active_profile": self.active_profile.name,
                "current_temps": {
                    "apu": current_sample.apu_temp if current_sample else 0.0,
                    "cpu": current_sample.cpu_temp if current_sample else 0.0,
                    "gpu": current_sample.gpu_temp if current_sample else 0.0
                },
                "fan_rpm": current_sample.fan_rpm if current_sample else 0,
                "power_draw": current_sample.power_draw if current_sample else 0.0,
                "battery_level": current_sample.battery_level if current_sample else 100.0,
                "compilation_threads": self.max_compilation_threads,
                "thermal_trend_per_minute": trend,
                "predicted_temp": predicted_temp,
                "prediction_confidence": confidence,
                "gaming_active": current_sample.gaming_active if current_sample else False,
                "sensors_available": len(self.sensor_paths)
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
    
    print("🌡️  Optimized Thermal Manager Test")
    print("="*50)
    
    # Show initial status
    status = manager.get_status()
    print(f"Steam Deck Model: {status['steam_deck_model']}")
    print(f"Sensors Available: {status['sensors_available']}")
    print(f"Current Temperature: {status['current_temps']['apu']:.1f}°C")
    print(f"Thermal State: {status['thermal_state']}")
    print(f"Compilation Threads: {status['compilation_threads']}")
    
    # Start monitoring
    manager.start_monitoring()
    
    try:
        print("\nMonitoring thermal state (Ctrl+C to stop)...")
        while True:
            time.sleep(5)
            status = manager.get_status()
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"APU: {status['current_temps']['apu']:.1f}°C "
                  f"State: {status['thermal_state']} "
                  f"Threads: {status['compilation_threads']} "
                  f"Trend: {status['thermal_trend_per_minute']:+.1f}°C/min")
    
    except KeyboardInterrupt:
        print("\nStopping thermal manager...")
        manager.stop_monitoring()
        print("✓ Thermal manager stopped")