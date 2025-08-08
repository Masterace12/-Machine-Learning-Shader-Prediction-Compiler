#!/usr/bin/env python3
"""
Advanced Thermal Management System for Steam Deck

This module provides comprehensive thermal management specifically designed
for Steam Deck hardware, including both LCD and OLED models with their
different thermal characteristics.

Key Features:
- Hardware-specific thermal profiles
- Adaptive performance scaling
- Battery-aware power management
- Gaming vs desktop mode optimization
- Predictive thermal modeling
- Emergency protection systems
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
import psutil
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDeckModel(Enum):
    """Steam Deck model types"""
    LCD_64GB = "lcd_64gb"
    LCD_256GB = "lcd_256gb"
    LCD_512GB = "lcd_512gb"
    OLED_512GB = "oled_512gb"
    OLED_1TB = "oled_1tb"
    UNKNOWN = "unknown"


class PowerProfile(Enum):
    """Power management profiles"""
    MAX_PERFORMANCE = "max_performance"
    BALANCED = "balanced"
    POWER_SAVER = "power_saver"
    BATTERY_EMERGENCY = "battery_emergency"
    THERMAL_LIMIT = "thermal_limit"


class ThermalZone(Enum):
    """Thermal monitoring zones"""
    APU_CORE = "apu_core"          # Main APU temperature
    GPU_CORE = "gpu_core"          # GPU die temperature
    CPU_CORE = "cpu_core"          # CPU cores temperature
    SKIN_TEMP = "skin_temp"        # Surface temperature
    BATTERY = "battery"            # Battery temperature
    AMBIENT = "ambient"            # Ambient temperature


@dataclass
class ThermalSensor:
    """Individual thermal sensor configuration"""
    zone: ThermalZone
    path: str
    scale_factor: float = 1000.0  # Convert from millidegrees
    offset: float = 0.0
    critical_temp: float = 100.0
    warning_temp: float = 85.0
    target_temp: float = 75.0


@dataclass
class PowerLimits:
    """Power consumption limits"""
    cpu_watts_max: float
    gpu_watts_max: float
    total_watts_max: float
    battery_watts_max: float
    thermal_watts_max: float


@dataclass
class ThermalProfile:
    """Complete thermal management profile"""
    name: str
    sensors: List[ThermalSensor]
    power_limits: PowerLimits
    fan_curve: Dict[float, float]  # temp -> fan_speed_percent
    thermal_trip_points: Dict[str, float]
    performance_scaling: Dict[str, float]
    emergency_actions: List[str]


@dataclass
class SystemMetrics:
    """Current system thermal and power metrics"""
    temperatures: Dict[ThermalZone, float] = field(default_factory=dict)
    power_consumption: Dict[str, float] = field(default_factory=dict)
    fan_speeds: Dict[str, float] = field(default_factory=dict)
    cpu_freq: float = 0.0
    gpu_freq: float = 0.0
    battery_level: float = 0.0
    battery_charging: bool = False
    ac_connected: bool = False
    gaming_mode_active: bool = False
    thermal_throttling_active: bool = False


class ThermalManager:
    """Advanced thermal management system for Steam Deck"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model = self._detect_steam_deck_model()
        self.current_profile = None
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_history = []
        self.thermal_events = []
        self.callbacks = {}
        
        # Load thermal profiles
        self.thermal_profiles = self._initialize_thermal_profiles()
        
        # Initialize with appropriate profile
        self._select_initial_profile()
        
        # Emergency protection
        self.emergency_temperature = 95.0
        self.emergency_triggered = False
        
    def _detect_steam_deck_model(self) -> SteamDeckModel:
        """Detect specific Steam Deck model for thermal tuning"""
        try:
            # Check DMI information
            dmi_files = [
                "/sys/devices/virtual/dmi/id/board_name",
                "/sys/class/dmi/id/board_name"
            ]
            
            for dmi_file in dmi_files:
                if os.path.exists(dmi_file):
                    with open(dmi_file, 'r') as f:
                        board_name = f.read().strip()
                        if "Galileo" in board_name:
                            # OLED model - determine storage size
                            return self._determine_oled_variant()
                        elif "Jupiter" in board_name:
                            # LCD model - determine storage size
                            return self._determine_lcd_variant()
            
            # Check GPU device ID
            try:
                result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
                if result.returncode == 0:
                    if "1002:163f" in result.stdout:
                        return self._determine_lcd_variant()
                    elif "1002:15bf" in result.stdout:
                        return self._determine_oled_variant()
            except Exception:
                pass
            
            logger.warning("Could not detect specific Steam Deck model")
            return SteamDeckModel.UNKNOWN
            
        except Exception as e:
            logger.error(f"Model detection failed: {e}")
            return SteamDeckModel.UNKNOWN
    
    def _determine_lcd_variant(self) -> SteamDeckModel:
        """Determine LCD model variant based on storage"""
        try:
            # Check root filesystem size as approximation
            statvfs = os.statvfs('/')
            total_gb = (statvfs.f_frsize * statvfs.f_blocks) / (1024**3)
            
            if total_gb < 50:
                return SteamDeckModel.LCD_64GB
            elif total_gb < 300:
                return SteamDeckModel.LCD_256GB
            else:
                return SteamDeckModel.LCD_512GB
                
        except Exception:
            return SteamDeckModel.LCD_256GB  # Default LCD
    
    def _determine_oled_variant(self) -> SteamDeckModel:
        """Determine OLED model variant based on storage"""
        try:
            statvfs = os.statvfs('/')
            total_gb = (statvfs.f_frsize * statvfs.f_blocks) / (1024**3)
            
            if total_gb > 800:
                return SteamDeckModel.OLED_1TB
            else:
                return SteamDeckModel.OLED_512GB
                
        except Exception:
            return SteamDeckModel.OLED_512GB  # Default OLED
    
    def _initialize_thermal_profiles(self) -> Dict[str, ThermalProfile]:
        """Initialize hardware-specific thermal profiles"""
        profiles = {}
        
        # LCD Steam Deck thermal profile
        if self.model.value.startswith('lcd'):
            profiles['lcd_balanced'] = ThermalProfile(
                name="LCD Balanced",
                sensors=[
                    ThermalSensor(ThermalZone.APU_CORE, "/sys/class/hwmon/hwmon0/temp1_input", 
                                critical_temp=95.0, warning_temp=85.0, target_temp=75.0),
                    ThermalSensor(ThermalZone.GPU_CORE, "/sys/class/hwmon/hwmon1/temp1_input", 
                                critical_temp=90.0, warning_temp=80.0, target_temp=70.0),
                    ThermalSensor(ThermalZone.SKIN_TEMP, "/sys/class/hwmon/hwmon2/temp1_input", 
                                critical_temp=50.0, warning_temp=45.0, target_temp=40.0),
                    ThermalSensor(ThermalZone.BATTERY, "/sys/class/power_supply/BAT1/temp", 
                                scale_factor=10.0, critical_temp=60.0, warning_temp=50.0, target_temp=35.0)
                ],
                power_limits=PowerLimits(
                    cpu_watts_max=15.0,
                    gpu_watts_max=15.0,
                    total_watts_max=25.0,
                    battery_watts_max=20.0,
                    thermal_watts_max=18.0
                ),
                fan_curve={
                    30.0: 0.0,    # Below 30°C: Fan off
                    40.0: 20.0,   # 40°C: 20% fan speed
                    50.0: 35.0,   # 50°C: 35% fan speed
                    60.0: 50.0,   # 60°C: 50% fan speed
                    70.0: 70.0,   # 70°C: 70% fan speed
                    80.0: 85.0,   # 80°C: 85% fan speed
                    90.0: 100.0   # 90°C+: 100% fan speed
                },
                thermal_trip_points={
                    'warning': 85.0,
                    'critical': 95.0,
                    'emergency': 100.0
                },
                performance_scaling={
                    'cpu_freq_scale': 1.0,
                    'gpu_freq_scale': 1.0,
                    'memory_freq_scale': 1.0
                },
                emergency_actions=['reduce_clocks', 'increase_fan', 'limit_power']
            )
            
            profiles['lcd_performance'] = ThermalProfile(
                name="LCD Performance",
                sensors=profiles['lcd_balanced'].sensors,
                power_limits=PowerLimits(
                    cpu_watts_max=20.0,
                    gpu_watts_max=20.0,
                    total_watts_max=35.0,
                    battery_watts_max=25.0,
                    thermal_watts_max=30.0
                ),
                fan_curve={
                    30.0: 10.0,   # More aggressive cooling
                    40.0: 30.0,
                    50.0: 45.0,
                    60.0: 60.0,
                    70.0: 80.0,
                    80.0: 95.0,
                    90.0: 100.0
                },
                thermal_trip_points={'warning': 87.0, 'critical': 97.0, 'emergency': 102.0},
                performance_scaling={'cpu_freq_scale': 1.1, 'gpu_freq_scale': 1.1, 'memory_freq_scale': 1.0},
                emergency_actions=['reduce_clocks', 'increase_fan', 'limit_power']
            )
        
        # OLED Steam Deck thermal profile (better thermal characteristics)
        if self.model.value.startswith('oled'):
            profiles['oled_balanced'] = ThermalProfile(
                name="OLED Balanced",
                sensors=[
                    ThermalSensor(ThermalZone.APU_CORE, "/sys/class/hwmon/hwmon0/temp1_input", 
                                critical_temp=97.0, warning_temp=87.0, target_temp=77.0),
                    ThermalSensor(ThermalZone.GPU_CORE, "/sys/class/hwmon/hwmon1/temp1_input", 
                                critical_temp=92.0, warning_temp=82.0, target_temp=72.0),
                    ThermalSensor(ThermalZone.SKIN_TEMP, "/sys/class/hwmon/hwmon2/temp1_input", 
                                critical_temp=48.0, warning_temp=43.0, target_temp=38.0),
                    ThermalSensor(ThermalZone.BATTERY, "/sys/class/power_supply/BAT1/temp", 
                                scale_factor=10.0, critical_temp=55.0, warning_temp=45.0, target_temp=32.0)
                ],
                power_limits=PowerLimits(
                    cpu_watts_max=18.0,
                    gpu_watts_max=18.0,
                    total_watts_max=30.0,
                    battery_watts_max=25.0,
                    thermal_watts_max=25.0
                ),
                fan_curve={
                    30.0: 0.0,
                    40.0: 15.0,
                    50.0: 30.0,
                    60.0: 45.0,
                    70.0: 65.0,
                    80.0: 80.0,
                    90.0: 100.0
                },
                thermal_trip_points={'warning': 87.0, 'critical': 97.0, 'emergency': 102.0},
                performance_scaling={'cpu_freq_scale': 1.0, 'gpu_freq_scale': 1.0, 'memory_freq_scale': 1.0},
                emergency_actions=['reduce_clocks', 'increase_fan', 'limit_power']
            )
            
            profiles['oled_performance'] = ThermalProfile(
                name="OLED Performance",
                sensors=profiles['oled_balanced'].sensors,
                power_limits=PowerLimits(
                    cpu_watts_max=25.0,
                    gpu_watts_max=25.0,
                    total_watts_max=40.0,
                    battery_watts_max=30.0,
                    thermal_watts_max=35.0
                ),
                fan_curve={
                    30.0: 5.0,    # More aggressive cooling
                    40.0: 25.0,
                    50.0: 40.0,
                    60.0: 55.0,
                    70.0: 75.0,
                    80.0: 90.0,
                    90.0: 100.0
                },
                thermal_trip_points={'warning': 89.0, 'critical': 99.0, 'emergency': 104.0},
                performance_scaling={'cpu_freq_scale': 1.15, 'gpu_freq_scale': 1.15, 'memory_freq_scale': 1.05},
                emergency_actions=['reduce_clocks', 'increase_fan', 'limit_power']
            )
        
        # Battery-saving profiles for both models
        for model_prefix in ['lcd', 'oled']:
            if model_prefix in [p.split('_')[0] for p in profiles.keys()]:
                base_profile = profiles[f'{model_prefix}_balanced']
                
                profiles[f'{model_prefix}_power_save'] = ThermalProfile(
                    name=f"{model_prefix.upper()} Power Save",
                    sensors=base_profile.sensors,
                    power_limits=PowerLimits(
                        cpu_watts_max=8.0,
                        gpu_watts_max=8.0,
                        total_watts_max=12.0,
                        battery_watts_max=10.0,
                        thermal_watts_max=10.0
                    ),
                    fan_curve={temp: max(0, speed - 20) for temp, speed in base_profile.fan_curve.items()},
                    thermal_trip_points={k: v - 5.0 for k, v in base_profile.thermal_trip_points.items()},
                    performance_scaling={'cpu_freq_scale': 0.7, 'gpu_freq_scale': 0.7, 'memory_freq_scale': 0.9},
                    emergency_actions=base_profile.emergency_actions
                )
        
        return profiles
    
    def _select_initial_profile(self):
        """Select appropriate initial thermal profile"""
        if self.model.value.startswith('oled'):
            profile_name = 'oled_balanced'
        elif self.model.value.startswith('lcd'):
            profile_name = 'lcd_balanced'
        else:
            # Generic fallback
            profile_name = list(self.thermal_profiles.keys())[0] if self.thermal_profiles else None
        
        if profile_name and profile_name in self.thermal_profiles:
            self.current_profile = self.thermal_profiles[profile_name]
            logger.info(f"Selected thermal profile: {self.current_profile.name}")
        else:
            logger.error("No suitable thermal profile found")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system thermal and power metrics"""
        metrics = SystemMetrics()
        
        try:
            # Read temperatures from sensors
            if self.current_profile:
                for sensor in self.current_profile.sensors:
                    if os.path.exists(sensor.path):
                        try:
                            with open(sensor.path, 'r') as f:
                                raw_value = float(f.read().strip())
                                temp_celsius = (raw_value / sensor.scale_factor) + sensor.offset
                                metrics.temperatures[sensor.zone] = temp_celsius
                        except (ValueError, IOError) as e:
                            logger.debug(f"Failed to read sensor {sensor.path}: {e}")
            
            # Read power consumption
            power_paths = {
                'total': '/sys/class/power_supply/ADP1/power_now',
                'battery': '/sys/class/power_supply/BAT1/power_now'
            }
            
            for name, path in power_paths.items():
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            power_microwatts = float(f.read().strip())
                            metrics.power_consumption[name] = power_microwatts / 1000000.0  # Convert to watts
                    except (ValueError, IOError):
                        pass
            
            # Read fan speeds
            fan_paths = ['/sys/class/hwmon/hwmon0/fan1_input', '/sys/class/hwmon/hwmon1/fan1_input']
            for i, path in enumerate(fan_paths):
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            fan_rpm = float(f.read().strip())
                            metrics.fan_speeds[f'fan{i}'] = fan_rpm
                    except (ValueError, IOError):
                        pass
            
            # Read CPU frequency
            cpu_freq_paths = ['/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq']
            for path in cpu_freq_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            freq_khz = float(f.read().strip())
                            metrics.cpu_freq = freq_khz / 1000.0  # Convert to MHz
                            break
                    except (ValueError, IOError):
                        pass
            
            # Read GPU frequency
            gpu_freq_paths = ['/sys/class/drm/card0/device/pp_dpm_sclk', '/sys/class/drm/card1/device/pp_dpm_sclk']
            for path in gpu_freq_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            dpm_states = f.read().strip()
                            for line in dmp_states.split('\n'):
                                if '*' in line:  # Current active state
                                    freq_str = line.split(':')[1].strip().rstrip('Mhz')
                                    metrics.gpu_freq = float(freq_str)
                                    break
                            break
                    except (ValueError, IOError, IndexError):
                        pass
            
            # Read battery information
            battery_paths = {
                'capacity': '/sys/class/power_supply/BAT1/capacity',
                'status': '/sys/class/power_supply/BAT1/status'
            }
            
            for info_type, path in battery_paths.items():
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            value = f.read().strip()
                            if info_type == 'capacity':
                                metrics.battery_level = float(value)
                            elif info_type == 'status':
                                metrics.battery_charging = (value.lower() == 'charging')
                    except (ValueError, IOError):
                        pass
            
            # Check AC adapter
            ac_paths = ['/sys/class/power_supply/ADP1/online', '/sys/class/power_supply/AC/online']
            for path in ac_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            metrics.ac_connected = (f.read().strip() == '1')
                            break
                    except (ValueError, IOError):
                        pass
            
            # Check gaming mode (look for gamescope process)
            try:
                metrics.gaming_mode_active = any('gamescope' in p.name() for p in psutil.process_iter(['name']))
            except Exception:
                metrics.gaming_mode_active = False
            
            # Check if thermal throttling is active
            thermal_throttle_paths = ['/sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count']
            for path in thermal_throttle_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            throttle_count = int(f.read().strip())
                            # If throttle count increased since last check, throttling is active
                            metrics.thermal_throttling_active = (throttle_count > 0)
                    except (ValueError, IOError):
                        pass
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def start_monitoring(self, interval: float = 2.0, callback: Optional[Callable] = None):
        """Start continuous thermal monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        if callback:
            self.callbacks['metrics'] = callback
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started thermal monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop thermal monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        logger.info("Stopped thermal monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop with adaptive thermal management"""
        consecutive_high_temps = 0
        last_metrics = None
        
        while self.monitoring_active:
            try:
                current_metrics = self.get_system_metrics()
                
                # Store metrics history
                self.metrics_history.append({
                    'timestamp': time.time(),
                    'metrics': current_metrics
                })
                
                # Keep history limited (last 10 minutes)
                if len(self.metrics_history) > 300:
                    self.metrics_history.pop(0)
                
                # Thermal management logic
                if self.current_profile and current_metrics.temperatures:
                    max_temp = max(current_metrics.temperatures.values())
                    
                    # Check for emergency temperature
                    if max_temp >= self.emergency_temperature:
                        if not self.emergency_triggered:
                            self._trigger_emergency_protection(current_metrics)
                            self.emergency_triggered = True
                    elif self.emergency_triggered and max_temp < self.emergency_temperature - 5.0:
                        self.emergency_triggered = False
                        logger.info("Emergency thermal protection lifted")
                    
                    # Adaptive profile switching
                    warning_temp = self.current_profile.thermal_trip_points.get('warning', 85.0)
                    
                    if max_temp > warning_temp:
                        consecutive_high_temps += 1
                        if consecutive_high_temps >= 3:  # 3 consecutive readings above warning
                            self._apply_thermal_mitigation(current_metrics)
                            consecutive_high_temps = 0
                    else:
                        consecutive_high_temps = 0
                        
                        # Check if we can return to higher performance
                        if max_temp < warning_temp - 10.0:
                            self._restore_performance(current_metrics)
                
                # Battery-aware power management
                if current_metrics.battery_level > 0:
                    self._manage_power_profile(current_metrics)
                
                # Execute callback if registered
                if 'metrics' in self.callbacks:
                    try:
                        self.callbacks['metrics'](current_metrics)
                    except Exception as e:
                        logger.error(f"Metrics callback error: {e}")
                
                last_metrics = current_metrics
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring loop: {e}")
                time.sleep(interval)
    
    def _trigger_emergency_protection(self, metrics: SystemMetrics):
        """Trigger emergency thermal protection measures"""
        logger.critical(f"EMERGENCY: Thermal protection triggered! Max temp: {max(metrics.temperatures.values())}°C")
        
        # Execute emergency actions
        if self.current_profile:
            for action in self.current_profile.emergency_actions:
                try:
                    if action == 'reduce_clocks':
                        self._reduce_cpu_clocks(emergency=True)
                        self._reduce_gpu_clocks(emergency=True)
                    elif action == 'increase_fan':
                        self._set_fan_speed(100.0)
                    elif action == 'limit_power':
                        self._limit_power_consumption(emergency=True)
                    elif action == 'suspend_system':
                        # Last resort - suspend system to prevent damage
                        subprocess.run(['systemctl', 'suspend'], check=False)
                        
                except Exception as e:
                    logger.error(f"Failed to execute emergency action {action}: {e}")
        
        # Log thermal event
        self.thermal_events.append({
            'timestamp': time.time(),
            'type': 'emergency',
            'max_temperature': max(metrics.temperatures.values()),
            'actions_taken': self.current_profile.emergency_actions if self.current_profile else []
        })
    
    def _apply_thermal_mitigation(self, metrics: SystemMetrics):
        """Apply thermal mitigation measures"""
        max_temp = max(metrics.temperatures.values())
        logger.warning(f"Applying thermal mitigation for temperature: {max_temp}°C")
        
        # Reduce performance scaling
        self._reduce_cpu_clocks()
        self._reduce_gpu_clocks()
        
        # Increase fan speed based on temperature
        target_fan_speed = self._calculate_fan_speed(max_temp)
        self._set_fan_speed(target_fan_speed)
        
        # Reduce power limits
        self._limit_power_consumption()
    
    def _restore_performance(self, metrics: SystemMetrics):
        """Restore performance after thermal conditions improve"""
        logger.info("Thermal conditions improved, restoring performance")
        
        # Restore normal clock speeds
        self._restore_cpu_clocks()
        self._restore_gpu_clocks()
        
        # Restore normal power limits
        self._restore_power_limits()
    
    def _manage_power_profile(self, metrics: SystemMetrics):
        """Manage power profile based on battery state"""
        battery_level = metrics.battery_level
        ac_connected = metrics.ac_connected
        
        # Battery emergency mode
        if battery_level <= 10.0 and not ac_connected:
            if not self.current_profile.name.endswith('power_save'):
                profile_name = f"{self.model.value.split('_')[0]}_power_save"
                if profile_name in self.thermal_profiles:
                    self.switch_profile(profile_name)
                    logger.warning(f"Battery emergency: Switched to power save mode ({battery_level}%)")
        
        # Battery low mode
        elif battery_level <= 20.0 and not ac_connected:
            if 'performance' in self.current_profile.name.lower():
                profile_name = f"{self.model.value.split('_')[0]}_balanced"
                if profile_name in self.thermal_profiles:
                    self.switch_profile(profile_name)
                    logger.info(f"Low battery: Switched to balanced mode ({battery_level}%)")
    
    def _calculate_fan_speed(self, temperature: float) -> float:
        """Calculate target fan speed based on temperature"""
        if not self.current_profile:
            return 50.0  # Default 50% fan speed
        
        fan_curve = self.current_profile.fan_curve
        
        # Find the appropriate fan speed from the curve
        temp_points = sorted(fan_curve.keys())
        
        if temperature <= temp_points[0]:
            return fan_curve[temp_points[0]]
        elif temperature >= temp_points[-1]:
            return fan_curve[temp_points[-1]]
        else:
            # Linear interpolation between curve points
            for i in range(len(temp_points) - 1):
                if temp_points[i] <= temperature <= temp_points[i + 1]:
                    temp_range = temp_points[i + 1] - temp_points[i]
                    temp_offset = temperature - temp_points[i]
                    speed_range = fan_curve[temp_points[i + 1]] - fan_curve[temp_points[i]]
                    interpolated_speed = fan_curve[temp_points[i]] + (temp_offset / temp_range) * speed_range
                    return interpolated_speed
        
        return 50.0  # Fallback
    
    def _set_fan_speed(self, speed_percent: float):
        """Set fan speed (if controllable)"""
        try:
            # Steam Deck fan control paths (may require kernel module)
            fan_control_paths = [
                '/sys/class/hwmon/hwmon0/pwm1',
                '/sys/class/hwmon/hwmon1/pwm1'
            ]
            
            # Convert percentage to PWM value (0-255)
            pwm_value = int((speed_percent / 100.0) * 255)
            
            for path in fan_control_paths:
                if os.path.exists(path) and os.access(path, os.W_OK):
                    with open(path, 'w') as f:
                        f.write(str(pwm_value))
                    logger.debug(f"Set fan speed to {speed_percent}% (PWM: {pwm_value})")
                    break
            else:
                logger.debug("Fan speed control not available or not writable")
                
        except Exception as e:
            logger.error(f"Failed to set fan speed: {e}")
    
    def _reduce_cpu_clocks(self, emergency: bool = False):
        """Reduce CPU clock speeds for thermal management"""
        try:
            reduction_factor = 0.5 if emergency else 0.8
            
            # Find available CPU frequencies
            cpu_freq_paths = [f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_available_frequencies' 
                             for i in range(8)]  # Steam Deck has 8 cores
            
            for path in cpu_freq_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        available_freqs = [int(x) for x in f.read().strip().split()]
                        if available_freqs:
                            target_freq = int(min(available_freqs) * (1 + reduction_factor) / 2)
                            # Set maximum frequency
                            max_freq_path = path.replace('scaling_available_frequencies', 'scaling_max_freq')
                            if os.path.exists(max_freq_path) and os.access(max_freq_path, os.W_OK):
                                with open(max_freq_path, 'w') as f:
                                    f.write(str(target_freq))
                                logger.debug(f"Reduced CPU max frequency to {target_freq} kHz")
                            break
                
        except Exception as e:
            logger.error(f"Failed to reduce CPU clocks: {e}")
    
    def _reduce_gpu_clocks(self, emergency: bool = False):
        """Reduce GPU clock speeds for thermal management"""
        try:
            # Steam Deck GPU clock control (may require specific permissions)
            gpu_freq_paths = [
                '/sys/class/drm/card0/device/pp_od_clk_voltage',
                '/sys/class/drm/card1/device/pp_od_clk_voltage'
            ]
            
            reduction_factor = 0.6 if emergency else 0.8
            
            for path in gpu_freq_paths:
                if os.path.exists(path) and os.access(path, os.W_OK):
                    # This is a simplified approach - actual implementation would need
                    # to parse current settings and apply reduction
                    logger.debug(f"GPU clock reduction requested ({reduction_factor}x)")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to reduce GPU clocks: {e}")
    
    def _limit_power_consumption(self, emergency: bool = False):
        """Limit system power consumption"""
        try:
            if not self.current_profile:
                return
            
            power_limit = (self.current_profile.power_limits.thermal_watts_max * 0.7 
                          if emergency else self.current_profile.power_limits.thermal_watts_max)
            
            # AMD APU power control paths
            power_control_paths = [
                '/sys/class/drm/card0/device/power_dpm_force_performance_level',
                '/sys/class/drm/card1/device/power_dpm_force_performance_level'
            ]
            
            for path in power_control_paths:
                if os.path.exists(path) and os.access(path, os.W_OK):
                    with open(path, 'w') as f:
                        f.write('low')  # Force low power mode
                    logger.debug(f"Applied power limit: {power_limit}W")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to limit power consumption: {e}")
    
    def _restore_cpu_clocks(self):
        """Restore normal CPU clock speeds"""
        try:
            # Reset CPU frequency limits to maximum
            cpu_freq_paths = [f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq' 
                             for i in range(8)]
            
            for path in cpu_freq_paths:
                if os.path.exists(path) and os.access(path, os.W_OK):
                    # Read the maximum available frequency
                    cpuinfo_max_path = path.replace('scaling_max_freq', 'cpuinfo_max_freq')
                    if os.path.exists(cpuinfo_max_path):
                        with open(cpuinfo_max_path, 'r') as f:
                            max_freq = f.read().strip()
                        with open(path, 'w') as f:
                            f.write(max_freq)
                        logger.debug("Restored CPU clock speeds")
                        break
                        
        except Exception as e:
            logger.error(f"Failed to restore CPU clocks: {e}")
    
    def _restore_gpu_clocks(self):
        """Restore normal GPU clock speeds"""
        try:
            gpu_perf_paths = [
                '/sys/class/drm/card0/device/power_dpm_force_performance_level',
                '/sys/class/drm/card1/device/power_dpm_force_performance_level'
            ]
            
            for path in gpu_perf_paths:
                if os.path.exists(path) and os.access(path, os.W_OK):
                    with open(path, 'w') as f:
                        f.write('auto')  # Restore automatic performance scaling
                    logger.debug("Restored GPU clock speeds")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to restore GPU clocks: {e}")
    
    def _restore_power_limits(self):
        """Restore normal power limits"""
        try:
            # Restore automatic power management
            self._restore_gpu_clocks()
            logger.debug("Restored power limits")
        except Exception as e:
            logger.error(f"Failed to restore power limits: {e}")
    
    def switch_profile(self, profile_name: str) -> bool:
        """Switch to a different thermal profile"""
        if profile_name not in self.thermal_profiles:
            logger.error(f"Thermal profile '{profile_name}' not found")
            return False
        
        self.current_profile = self.thermal_profiles[profile_name]
        logger.info(f"Switched to thermal profile: {self.current_profile.name}")
        return True
    
    def get_thermal_report(self) -> Dict[str, Any]:
        """Generate comprehensive thermal report"""
        current_metrics = self.get_system_metrics()
        
        # Calculate averages from history
        temp_history = []
        power_history = []
        
        for record in self.metrics_history[-60:]:  # Last 2 minutes
            if record['metrics'].temperatures:
                temp_history.append(max(record['metrics'].temperatures.values()))
            if 'total' in record['metrics'].power_consumption:
                power_history.append(record['metrics'].power_consumption['total'])
        
        avg_temp = sum(temp_history) / len(temp_history) if temp_history else 0.0
        avg_power = sum(power_history) / len(power_history) if power_history else 0.0
        
        return {
            "hardware": {
                "model": self.model.value,
                "current_profile": self.current_profile.name if self.current_profile else "none"
            },
            "current_metrics": {
                "temperatures": {zone.value: temp for zone, temp in current_metrics.temperatures.items()},
                "power_consumption": current_metrics.power_consumption,
                "fan_speeds": current_metrics.fan_speeds,
                "cpu_frequency_mhz": current_metrics.cpu_freq,
                "gpu_frequency_mhz": current_metrics.gpu_freq,
                "battery_level_percent": current_metrics.battery_level,
                "battery_charging": current_metrics.battery_charging,
                "ac_connected": current_metrics.ac_connected,
                "gaming_mode_active": current_metrics.gaming_mode_active,
                "thermal_throttling_active": current_metrics.thermal_throttling_active
            },
            "averages_last_2_minutes": {
                "temperature_celsius": avg_temp,
                "power_consumption_watts": avg_power
            },
            "thermal_events": {
                "total_events": len(self.thermal_events),
                "recent_events": [e for e in self.thermal_events if time.time() - e['timestamp'] < 3600]  # Last hour
            },
            "emergency_status": {
                "triggered": self.emergency_triggered,
                "threshold_celsius": self.emergency_temperature
            }
        }
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for thermal events"""
        self.callbacks[event_type] = callback
        logger.info(f"Registered callback for event type: {event_type}")


def main():
    """Test the thermal management system"""
    print("🌡️  Steam Deck Thermal Manager Test")
    print("=" * 40)
    
    manager = ThermalManager()
    
    print(f"Detected model: {manager.model.value}")
    print(f"Current profile: {manager.current_profile.name if manager.current_profile else 'None'}")
    
    # Test metrics collection
    metrics = manager.get_system_metrics()
    print(f"\nCurrent Metrics:")
    print(f"Temperatures: {metrics.temperatures}")
    print(f"Power consumption: {metrics.power_consumption}")
    print(f"Battery: {metrics.battery_level}% ({'charging' if metrics.battery_charging else 'discharging'})")
    print(f"Gaming mode: {metrics.gaming_mode_active}")
    
    # Test profile switching
    available_profiles = list(manager.thermal_profiles.keys())
    print(f"\nAvailable profiles: {available_profiles}")
    
    # Start monitoring for a short time
    print("\nStarting thermal monitoring for 10 seconds...")
    manager.start_monitoring(interval=1.0)
    time.sleep(10)
    manager.stop_monitoring()
    
    # Generate report
    report = manager.get_thermal_report()
    print(f"\nGenerated thermal report with {len(report)} sections")
    
    print("\n🚀 Test completed!")


if __name__ == "__main__":
    main()