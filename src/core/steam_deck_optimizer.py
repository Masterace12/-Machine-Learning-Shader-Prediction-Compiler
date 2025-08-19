#!/usr/bin/env python3
"""
Steam Deck Optimizer for ML Shader Prediction Compiler

This module provides Steam Deck specific optimizations, thermal management,
power efficiency, and compatibility validation for the Enhanced ML Predictor.
Designed to work seamlessly with the Steam Deck's unique hardware and software
environment, including Gaming Mode, thermal constraints, and power management.

Features:
- Steam Deck hardware detection and profiling
- Thermal-aware performance scaling
- Gaming Mode integration and background optimization
- Power-efficient dependency selection
- APU-specific optimizations
- Memory constraint management for handheld gaming
- Battery life optimization
- Compatibility validation for SteamOS
"""

import os
import sys
import time
import json
import threading
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import logging

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# STEAM DECK HARDWARE DETECTION
# =============================================================================

@dataclass
class SteamDeckHardwareProfile:
    """Steam Deck hardware configuration profile"""
    model: str  # 'lcd', 'oled', 'unknown'
    cpu_cores: int
    cpu_max_freq_mhz: int
    memory_gb: float
    storage_type: str  # 'emmc', 'nvme', 'sd'
    display_resolution: Tuple[int, int]
    battery_capacity_wh: float
    tdp_watts: int
    thermal_limit_celsius: float
    boost_supported: bool
    
@dataclass
class SteamDeckState:
    """Current Steam Deck runtime state"""
    cpu_frequency_mhz: int
    cpu_temperature_celsius: float
    memory_usage_mb: float
    battery_percent: float
    battery_time_remaining_min: Optional[float]
    power_draw_watts: float
    thermal_throttling: bool
    gaming_mode_active: bool
    steam_running: bool
    performance_governor: str
    fan_speed_rpm: Optional[int]
    dock_connected: bool

# =============================================================================
# STEAM DECK OPTIMIZER
# =============================================================================

class SteamDeckOptimizer:
    """
    Comprehensive Steam Deck optimization system
    """
    
    def __init__(self):
        self.is_steam_deck = self._detect_steam_deck()
        self.hardware_profile: Optional[SteamDeckHardwareProfile] = None
        self.optimization_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.optimization_callbacks: List[Callable] = []
        self.performance_profiles: Dict[str, Dict[str, Any]] = {}
        self.thermal_history: List[Tuple[float, float]] = []  # (timestamp, temperature)
        self.power_history: List[Tuple[float, float]] = []    # (timestamp, power_draw)
        
        if self.is_steam_deck:
            self.hardware_profile = self._detect_hardware_profile()
            self._setup_steam_deck_profiles()
            logger.info(f"Steam Deck detected: {self.hardware_profile.model} model")
        else:
            logger.info("Not running on Steam Deck - optimizer will provide generic optimizations")
    
    def _detect_steam_deck(self) -> bool:
        """Comprehensive Steam Deck detection"""
        try:
            from .pure_python_fallbacks import PureSteamDeckDetector
            return PureSteamDeckDetector.is_steam_deck()
        except ImportError:
            # Fallback detection methods
            detection_methods = [
                self._check_dmi_info,
                self._check_user_directory,
                self._check_cpu_info,
                self._check_environment,
                self._check_steam_installation
            ]
            
            return any(method() for method in detection_methods)
    
    def _check_dmi_info(self) -> bool:
        """Check DMI information for Steam Deck identifiers"""
        try:
            dmi_files = [
                '/sys/devices/virtual/dmi/id/product_name',
                '/sys/devices/virtual/dmi/id/board_name',
                '/sys/devices/virtual/dmi/id/sys_vendor'
            ]
            
            for dmi_file in dmi_files:
                if os.path.exists(dmi_file):
                    with open(dmi_file, 'r') as f:
                        content = f.read().strip().lower()
                        if any(identifier in content for identifier in ['jupiter', 'galileo', 'valve']):
                            return True
        except Exception:
            pass
        return False
    
    def _check_user_directory(self) -> bool:
        """Check for Steam Deck user directory"""
        return os.path.exists('/home/deck')
    
    def _check_cpu_info(self) -> bool:
        """Check CPU info for Steam Deck APU"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read().lower()
                return 'amd custom apu' in content or 'van gogh' in content
        except Exception:
            return False
    
    def _check_environment(self) -> bool:
        """Check environment variables"""
        return os.environ.get('SteamDeck') is not None
    
    def _check_steam_installation(self) -> bool:
        """Check for Steam Deck specific Steam installation"""
        steam_paths = [
            '/home/deck/.steam',
            '/home/deck/.local/share/Steam'
        ]
        return any(os.path.exists(path) for path in steam_paths)
    
    def _detect_hardware_profile(self) -> SteamDeckHardwareProfile:
        """Detect Steam Deck hardware configuration"""
        
        # Detect model (LCD vs OLED)
        model = self._detect_steam_deck_model()
        
        # Get CPU information
        cpu_cores = self._get_cpu_core_count()
        cpu_max_freq = self._get_cpu_max_frequency()
        
        # Get memory information
        memory_gb = self._get_memory_size_gb()
        
        # Detect storage type
        storage_type = self._detect_storage_type()
        
        # Model-specific configurations with enhanced OLED optimizations
        if model == 'oled':
            battery_capacity = 50.0  # Wh (25% larger battery)
            display_resolution = (1280, 800)
            thermal_limit = 82.0  # Better cooling allows lower thermal target
        else:  # LCD or unknown
            battery_capacity = 40.0  # Wh
            display_resolution = (1280, 800)
            thermal_limit = 90.0
        
        return SteamDeckHardwareProfile(
            model=model,
            cpu_cores=cpu_cores,
            cpu_max_freq_mhz=cpu_max_freq,
            memory_gb=memory_gb,
            storage_type=storage_type,
            display_resolution=display_resolution,
            battery_capacity_wh=battery_capacity,
            tdp_watts=15,  # Steam Deck APU TDP
            thermal_limit_celsius=thermal_limit,
            boost_supported=True
        )
    
    def _detect_steam_deck_model(self) -> str:
        """Detect Steam Deck model (LCD vs OLED)"""
        try:
            # Check DMI for model information
            product_files = [
                '/sys/devices/virtual/dmi/id/product_name',
                '/sys/devices/virtual/dmi/id/board_name'
            ]
            
            for file_path in product_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read().strip().lower()
                        if 'galileo' in content:
                            return 'oled'
                        elif 'jupiter' in content:
                            return 'lcd'
            
            # Fallback: check for OLED-specific features
            oled_indicators = [
                '/sys/class/backlight/amdgpu_bl1',  # OLED backlight
                '/sys/devices/platform/jupiter'     # Platform device
            ]
            
            if any(os.path.exists(indicator) for indicator in oled_indicators):
                return 'oled'
            
        except Exception:
            pass
        
        return 'lcd'  # Default to LCD if detection fails
    
    def _get_cpu_core_count(self) -> int:
        """Get CPU core count"""
        try:
            return os.cpu_count() or 4
        except Exception:
            return 4  # Steam Deck has 4 cores
    
    def _get_cpu_max_frequency(self) -> int:
        """Get maximum CPU frequency in MHz"""
        try:
            # Try to read from cpufreq
            freq_files = [
                '/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq',
                '/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq'
            ]
            
            for freq_file in freq_files:
                if os.path.exists(freq_file):
                    with open(freq_file, 'r') as f:
                        freq_khz = int(f.read().strip())
                        return freq_khz // 1000  # Convert to MHz
            
            # Fallback: parse /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'cpu MHz' in line:
                        freq_mhz = float(line.split(':')[1].strip())
                        return int(freq_mhz)
        
        except Exception:
            pass
        
        return 3500  # Steam Deck APU typical max frequency
    
    def _get_memory_size_gb(self) -> float:
        """Get total memory size in GB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_kb = int(line.split()[1])
                        return mem_kb / (1024 * 1024)  # Convert to GB
        except Exception:
            pass
        
        return 16.0  # Steam Deck has 16GB LPDDR5
    
    def _detect_storage_type(self) -> str:
        """Detect primary storage type"""
        try:
            # Check for NVMe
            if os.path.exists('/sys/class/nvme'):
                nvme_devices = os.listdir('/sys/class/nvme')
                if nvme_devices:
                    return 'nvme'
            
            # Check for eMMC
            emmc_paths = [
                '/sys/class/mmc_host/mmc0',
                '/dev/mmcblk0'
            ]
            if any(os.path.exists(path) for path in emmc_paths):
                return 'emmc'
            
            # Check for SD card as primary (unlikely but possible)
            if os.path.exists('/dev/mmcblk1'):
                return 'sd'
                
        except Exception:
            pass
        
        return 'emmc'  # Base Steam Deck uses eMMC
    
    def _setup_steam_deck_profiles(self) -> None:
        """Setup Steam Deck specific performance profiles"""
        
        # Maximum performance profile (docked/AC power) with OLED optimizations
        oled_max_power = 22.0 if self.hardware_profile and self.hardware_profile.model == 'oled' else 20.0
        oled_gpu_limit = 18 if self.hardware_profile and self.hardware_profile.model == 'oled' else 15
        
        self.performance_profiles['maximum'] = {
            'description': 'Maximum performance for docked Steam Deck (OLED enhanced)',
            'cpu_governor': 'performance',
            'cpu_max_freq_pct': 100,
            'gpu_power_limit': oled_gpu_limit,
            'memory_aggressive': True,
            'ml_dependencies': ['numpy', 'scikit-learn', 'lightgbm', 'numba'],
            'thermal_limit': 82.0 if self.hardware_profile and self.hardware_profile.model == 'oled' else 85.0,
            'power_limit_watts': oled_max_power,
            'battery_usage': 'unlimited',
            'oled_optimized': self.hardware_profile and self.hardware_profile.model == 'oled'
        }
        
        # Balanced profile (default) with OLED efficiency improvements
        oled_balanced_power = 16.5 if self.hardware_profile and self.hardware_profile.model == 'oled' else 15.0
        oled_balanced_gpu = 14 if self.hardware_profile and self.hardware_profile.model == 'oled' else 12
        
        self.performance_profiles['balanced'] = {
            'description': 'Balanced performance and efficiency (OLED optimized)',
            'cpu_governor': 'powersave',
            'cpu_max_freq_pct': 85 if self.hardware_profile and self.hardware_profile.model == 'oled' else 80,
            'gpu_power_limit': oled_balanced_gpu,
            'memory_aggressive': False,
            'ml_dependencies': ['numpy', 'scikit-learn', 'psutil'],
            'thermal_limit': 78.0 if self.hardware_profile and self.hardware_profile.model == 'oled' else 80.0,
            'power_limit_watts': oled_balanced_power,
            'battery_usage': 'moderate',
            'oled_optimized': self.hardware_profile and self.hardware_profile.model == 'oled'
        }
        
        # Gaming mode profile (background operation)
        self.performance_profiles['gaming'] = {
            'description': 'Background operation during gaming',
            'cpu_governor': 'powersave',
            'cpu_max_freq_pct': 50,
            'gpu_power_limit': 8,
            'memory_aggressive': False,
            'ml_dependencies': ['psutil'],
            'thermal_limit': 90.0,
            'power_limit_watts': 8.0,
            'battery_usage': 'minimal'
        }
        
        # Battery saving profile
        self.performance_profiles['battery'] = {
            'description': 'Maximum battery life',
            'cpu_governor': 'powersave',
            'cpu_max_freq_pct': 40,
            'gpu_power_limit': 6,
            'memory_aggressive': False,
            'ml_dependencies': [],  # Pure Python only
            'thermal_limit': 70.0,
            'power_limit_watts': 6.0,
            'battery_usage': 'conservative'
        }
        
        # Thermal emergency profile
        self.performance_profiles['thermal_emergency'] = {
            'description': 'Emergency thermal protection',
            'cpu_governor': 'powersave',
            'cpu_max_freq_pct': 25,
            'gpu_power_limit': 4,
            'memory_aggressive': False,
            'ml_dependencies': [],  # Pure Python only
            'thermal_limit': 95.0,
            'power_limit_watts': 4.0,
            'battery_usage': 'emergency'
        }
        
        logger.info(f"Setup {len(self.performance_profiles)} Steam Deck performance profiles")
    
    def get_current_state(self) -> SteamDeckState:
        """Get current Steam Deck runtime state"""
        if not self.is_steam_deck:
            # Return dummy state for non-Steam Deck systems
            return SteamDeckState(
                cpu_frequency_mhz=2000,
                cpu_temperature_celsius=50.0,
                memory_usage_mb=1024.0,
                battery_percent=100.0,
                battery_time_remaining_min=None,
                power_draw_watts=10.0,
                thermal_throttling=False,
                gaming_mode_active=False,
                steam_running=False,
                performance_governor='balanced',
                fan_speed_rpm=None,
                dock_connected=False
            )
        
        # Get CPU frequency
        cpu_freq = self._get_current_cpu_frequency()
        
        # Get CPU temperature
        cpu_temp = self._get_cpu_temperature()
        
        # Get memory usage
        memory_usage = self._get_memory_usage_mb()
        
        # Get battery information
        battery_percent, battery_time, power_draw = self._get_battery_info()
        
        # Check for thermal throttling
        thermal_throttling = self._is_thermal_throttling()
        
        # Check gaming mode
        gaming_mode = self._is_gaming_mode_active()
        
        # Check Steam status
        steam_running = self._is_steam_running()
        
        # Get performance governor
        governor = self._get_performance_governor()
        
        # Get fan speed (if available)
        fan_speed = self._get_fan_speed()
        
        # Check dock connection
        dock_connected = self._is_dock_connected()
        
        return SteamDeckState(
            cpu_frequency_mhz=cpu_freq,
            cpu_temperature_celsius=cpu_temp,
            memory_usage_mb=memory_usage,
            battery_percent=battery_percent,
            battery_time_remaining_min=battery_time,
            power_draw_watts=power_draw,
            thermal_throttling=thermal_throttling,
            gaming_mode_active=gaming_mode,
            steam_running=steam_running,
            performance_governor=governor,
            fan_speed_rpm=fan_speed,
            dock_connected=dock_connected
        )
    
    def _get_current_cpu_frequency(self) -> int:
        """Get current CPU frequency in MHz"""
        try:
            freq_files = [
                '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq',
                '/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq'
            ]
            
            for freq_file in freq_files:
                if os.path.exists(freq_file):
                    with open(freq_file, 'r') as f:
                        freq_khz = int(f.read().strip())
                        return freq_khz // 1000
        except Exception:
            pass
        
        return 2000  # Fallback frequency
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature in Celsius"""
        try:
            from .pure_python_fallbacks import PureThermalMonitor
            thermal_monitor = PureThermalMonitor()
            return thermal_monitor.get_cpu_temperature()
        except ImportError:
            # Fallback thermal reading
            thermal_zones = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp'
            ]
            
            for zone in thermal_zones:
                try:
                    if os.path.exists(zone):
                        with open(zone, 'r') as f:
                            temp_millic = int(f.read().strip())
                            temp_celsius = temp_millic / 1000.0
                            if 20.0 <= temp_celsius <= 120.0:
                                return temp_celsius
                except Exception:
                    continue
            
            return 55.0  # Safe fallback temperature
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    key, value = line.split(':')
                    meminfo[key.strip()] = int(value.strip().split()[0])  # KB
                
                total_mb = meminfo['MemTotal'] / 1024
                available_mb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0)) / 1024
                used_mb = total_mb - available_mb
                
                return used_mb
        except Exception:
            return 4096.0  # Fallback: 4GB used
    
    def _get_battery_info(self) -> Tuple[float, Optional[float], float]:
        """Get battery percentage, time remaining, and power draw"""
        battery_percent = 100.0
        battery_time = None
        power_draw = 10.0
        
        try:
            # Check for battery information
            battery_path = '/sys/class/power_supply/BAT1'
            if os.path.exists(battery_path):
                # Read battery capacity
                capacity_file = os.path.join(battery_path, 'capacity')
                if os.path.exists(capacity_file):
                    with open(capacity_file, 'r') as f:
                        battery_percent = float(f.read().strip())
                
                # Read power draw (if available)
                power_file = os.path.join(battery_path, 'power_now')
                if os.path.exists(power_file):
                    with open(power_file, 'r') as f:
                        power_uw = int(f.read().strip())
                        power_draw = power_uw / 1000000.0  # Convert to watts
                
                # Estimate time remaining
                if battery_percent > 0 and power_draw > 0:
                    if self.hardware_profile:
                        battery_wh = self.hardware_profile.battery_capacity_wh
                        remaining_wh = battery_wh * (battery_percent / 100.0)
                        battery_time = (remaining_wh / power_draw) * 60  # Minutes
            
        except Exception:
            pass
        
        return battery_percent, battery_time, power_draw
    
    def _is_thermal_throttling(self) -> bool:
        """Check if CPU is thermal throttling"""
        try:
            # Check CPU frequency vs maximum
            current_freq = self._get_current_cpu_frequency()
            if self.hardware_profile:
                max_freq = self.hardware_profile.cpu_max_freq_mhz
                # Consider throttling if frequency is significantly below max
                if current_freq < max_freq * 0.8:
                    # Also check temperature
                    temp = self._get_cpu_temperature()
                    if temp > 75.0:  # Likely thermal throttling
                        return True
            
            # Check thermal zone throttling files
            throttle_files = [
                '/sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count',
                '/sys/class/thermal/thermal_zone0/trip_point_0_temp'
            ]
            
            for file_path in throttle_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        if content and int(content) > 0:
                            return True
        
        except Exception:
            pass
        
        return False
    
    def _is_gaming_mode_active(self) -> bool:
        """Check if Steam Gaming Mode is active"""
        try:
            # Check for gamescope process (Gaming Mode uses gamescope)
            result = subprocess.run(['pgrep', '-f', 'gamescope'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                return True
            
            # Check for Steam Big Picture mode
            result = subprocess.run(['pgrep', '-f', 'steamwebhelper.*gamepadui'], 
                                  capture_output=True, text=True, timeout=3)
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _is_steam_running(self) -> bool:
        """Check if Steam is running"""
        try:
            result = subprocess.run(['pgrep', '-f', 'steam'], 
                                  capture_output=True, text=True, timeout=3)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_performance_governor(self) -> str:
        """Get current CPU performance governor"""
        try:
            governor_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
            if os.path.exists(governor_file):
                with open(governor_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        
        return 'unknown'
    
    def _get_fan_speed(self) -> Optional[int]:
        """Get fan speed in RPM (if available)"""
        try:
            # Steam Deck fan control paths
            fan_paths = [
                '/sys/class/hwmon/hwmon0/fan1_input',
                '/sys/class/hwmon/hwmon1/fan1_input',
                '/sys/class/hwmon/hwmon2/fan1_input'
            ]
            
            for fan_path in fan_paths:
                if os.path.exists(fan_path):
                    with open(fan_path, 'r') as f:
                        return int(f.read().strip())
        except Exception:
            pass
        
        return None
    
    def _is_dock_connected(self) -> bool:
        """Check if Steam Deck is connected to dock"""
        try:
            # Check for external display connections
            display_paths = [
                '/sys/class/drm/card0-DP-1/status',
                '/sys/class/drm/card0-DP-2/status',
                '/sys/class/drm/card0-HDMI-A-1/status'
            ]
            
            for display_path in display_paths:
                if os.path.exists(display_path):
                    with open(display_path, 'r') as f:
                        status = f.read().strip()
                        if status == 'connected':
                            return True
            
            # Check for USB-C dock indicators
            usb_indicators = [
                '/sys/class/power_supply/ADP1',  # AC adapter
                '/sys/bus/usb/devices/*/product'  # USB devices
            ]
            
            # Simple heuristic: if on AC power and not battery, likely docked
            battery_path = '/sys/class/power_supply/BAT1/status'
            ac_path = '/sys/class/power_supply/ADP1'
            
            if os.path.exists(ac_path) and os.path.exists(battery_path):
                return True  # AC adapter present, likely docked
                
        except Exception:
            pass
        
        return False
    
    def select_optimal_profile(self, state: Optional[SteamDeckState] = None) -> str:
        """Select optimal performance profile based on current state"""
        if state is None:
            state = self.get_current_state()
        
        # Emergency thermal protection
        if state.cpu_temperature_celsius > 90.0 or state.thermal_throttling:
            return 'thermal_emergency'
        
        # Gaming mode optimization
        if state.gaming_mode_active:
            return 'gaming'
        
        # Battery conservation
        if state.battery_percent is not None:
            if state.battery_percent < 20.0:
                return 'battery'
            elif state.battery_percent < 50.0 and not state.dock_connected:
                return 'battery'
        
        # Docked mode optimization
        if state.dock_connected:
            # Use maximum performance when docked and cool
            if state.cpu_temperature_celsius < 70.0:
                return 'maximum'
            else:
                return 'balanced'
        
        # Default balanced profile
        return 'balanced'
    
    def apply_optimization_profile(self, profile_name: str) -> bool:
        """Apply a specific optimization profile"""
        if not self.is_steam_deck:
            logger.info(f"Not on Steam Deck - skipping profile application: {profile_name}")
            return True
        
        if profile_name not in self.performance_profiles:
            logger.error(f"Unknown profile: {profile_name}")
            return False
        
        profile = self.performance_profiles[profile_name]
        logger.info(f"Applying Steam Deck optimization profile: {profile_name}")
        
        success = True
        
        try:
            # Apply CPU governor if possible
            if self._can_modify_system():
                success &= self._set_cpu_governor(profile['cpu_governor'])
                success &= self._set_cpu_frequency_limit(profile['cpu_max_freq_pct'])
            
            # Set thermal limits (informational)
            if hasattr(self, '_thermal_limit'):
                self._thermal_limit = profile['thermal_limit']
            
            # Notify dependency manager about ML dependencies
            self._notify_dependency_changes(profile['ml_dependencies'])
            
            # Log profile application
            logger.info(f"Profile '{profile_name}' applied successfully: {profile['description']}")
            
            # Notify callbacks
            for callback in self.optimization_callbacks:
                try:
                    callback(profile_name, profile)
                except Exception as e:
                    logger.error(f"Error in optimization callback: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error applying profile '{profile_name}': {e}")
            return False
    
    def _can_modify_system(self) -> bool:
        """Check if we can modify system settings"""
        # Check if running as root or with appropriate permissions
        return os.geteuid() == 0 or os.path.exists('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
    
    def _set_cpu_governor(self, governor: str) -> bool:
        """Set CPU performance governor"""
        try:
            governor_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
            if os.path.exists(governor_file):
                # Check if governor is available
                available_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors'
                if os.path.exists(available_file):
                    with open(available_file, 'r') as f:
                        available_governors = f.read().strip().split()
                        if governor not in available_governors:
                            logger.warning(f"Governor '{governor}' not available, available: {available_governors}")
                            return False
                
                # Set the governor
                with open(governor_file, 'w') as f:
                    f.write(governor)
                logger.info(f"Set CPU governor to: {governor}")
                return True
        except Exception as e:
            logger.error(f"Failed to set CPU governor to '{governor}': {e}")
        
        return False
    
    def _set_cpu_frequency_limit(self, percentage: int) -> bool:
        """Set CPU frequency limit as percentage of maximum"""
        try:
            max_freq_file = '/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq'
            scaling_max_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq'
            
            if os.path.exists(max_freq_file) and os.path.exists(scaling_max_file):
                with open(max_freq_file, 'r') as f:
                    max_freq_khz = int(f.read().strip())
                
                target_freq_khz = int(max_freq_khz * percentage / 100)
                
                with open(scaling_max_file, 'w') as f:
                    f.write(str(target_freq_khz))
                
                logger.info(f"Set CPU frequency limit to {percentage}% ({target_freq_khz // 1000} MHz)")
                return True
        except Exception as e:
            logger.error(f"Failed to set CPU frequency limit: {e}")
        
        return False
    
    def _notify_dependency_changes(self, dependencies: List[str]) -> None:
        """Notify runtime dependency manager about profile changes"""
        try:
            from .runtime_dependency_manager import get_runtime_manager
            manager = get_runtime_manager()
            
            # Create a temporary profile for these dependencies
            steam_deck_profile_id = f'steam_deck_auto_{int(time.time())}'
            
            # This would be integrated with the runtime manager
            logger.debug(f"Notified dependency manager about profile with dependencies: {dependencies}")
            
        except ImportError:
            logger.debug("Runtime dependency manager not available")
    
    def start_adaptive_optimization(self, interval: float = 10.0) -> None:
        """Start adaptive optimization based on changing conditions"""
        if self.optimization_active:
            logger.warning("Adaptive optimization already active")
            return
        
        self.optimization_active = True
        self.monitoring_thread = threading.Thread(
            target=self._optimization_loop,
            args=(interval,),
            name="SteamDeckOptimizer",
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Steam Deck adaptive optimization started")
    
    def stop_adaptive_optimization(self) -> None:
        """Stop adaptive optimization"""
        self.optimization_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=30.0)
        logger.info("Steam Deck adaptive optimization stopped")
    
    def _optimization_loop(self, interval: float) -> None:
        """Main optimization monitoring loop"""
        logger.info("Steam Deck optimization loop started")
        last_profile = None
        
        while self.optimization_active:
            try:
                # Get current state
                state = self.get_current_state()
                
                # Record thermal and power history
                current_time = time.time()
                self.thermal_history.append((current_time, state.cpu_temperature_celsius))
                self.power_history.append((current_time, state.power_draw_watts))
                
                # Keep history manageable
                if len(self.thermal_history) > 100:
                    self.thermal_history.pop(0)
                if len(self.power_history) > 100:
                    self.power_history.pop(0)
                
                # Select optimal profile
                optimal_profile = self.select_optimal_profile(state)
                
                # Apply profile if it changed
                if optimal_profile != last_profile:
                    logger.info(f"Steam Deck conditions changed, switching to profile: {optimal_profile}")
                    if self.apply_optimization_profile(optimal_profile):
                        last_profile = optimal_profile
                    else:
                        logger.error(f"Failed to apply profile: {optimal_profile}")
                
                # Sleep until next check
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in Steam Deck optimization loop: {e}")
                time.sleep(interval)
        
        logger.info("Steam Deck optimization loop ended")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on current state"""
        state = self.get_current_state()
        recommendations = []
        
        # Thermal recommendations
        if state.cpu_temperature_celsius > 85.0:
            recommendations.append({
                'type': 'thermal',
                'priority': 'high',
                'title': 'High Temperature Warning',
                'description': f'CPU temperature is {state.cpu_temperature_celsius:.1f}°C',
                'action': 'Consider reducing computational load or improving ventilation',
                'profile_suggestion': 'thermal_emergency'
            })
        elif state.cpu_temperature_celsius > 75.0:
            recommendations.append({
                'type': 'thermal',
                'priority': 'medium',
                'title': 'Elevated Temperature',
                'description': f'CPU temperature is {state.cpu_temperature_celsius:.1f}°C',
                'action': 'Consider switching to a more conservative profile',
                'profile_suggestion': 'balanced'
            })
        
        # Battery recommendations
        if state.battery_percent is not None:
            if state.battery_percent < 20.0:
                recommendations.append({
                    'type': 'battery',
                    'priority': 'high',
                    'title': 'Low Battery',
                    'description': f'Battery at {state.battery_percent:.0f}%',
                    'action': 'Switch to power saving mode or connect charger',
                    'profile_suggestion': 'battery'
                })
            elif state.battery_percent < 50.0 and state.power_draw_watts > 12.0:
                recommendations.append({
                    'type': 'battery',
                    'priority': 'medium',
                    'title': 'High Power Consumption',
                    'description': f'Using {state.power_draw_watts:.1f}W with {state.battery_percent:.0f}% battery',
                    'action': 'Consider reducing performance to extend battery life',
                    'profile_suggestion': 'balanced'
                })
        
        # Gaming mode recommendations
        if state.gaming_mode_active:
            recommendations.append({
                'type': 'gaming',
                'priority': 'info',
                'title': 'Gaming Mode Active',
                'description': 'Steam Gaming Mode detected',
                'action': 'ML operations will run in background with minimal impact',
                'profile_suggestion': 'gaming'
            })
        
        # Memory recommendations
        if state.memory_usage_mb > 12000:  # > 12GB
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'title': 'High Memory Usage',
                'description': f'Using {state.memory_usage_mb:.0f}MB of memory',
                'action': 'Consider using lighter ML dependencies',
                'profile_suggestion': 'conservative'
            })
        
        # Performance recommendations
        if state.dock_connected and not state.gaming_mode_active:
            if state.cpu_temperature_celsius < 70.0:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'info',
                    'title': 'Docked Mode Optimization',
                    'description': 'Steam Deck is docked with good thermal conditions',
                    'action': 'Can use maximum performance profile',
                    'profile_suggestion': 'maximum'
                })
        
        return recommendations
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Get Steam Deck compatibility report"""
        state = self.get_current_state()
        
        # Check dependency compatibility
        dependency_compatibility = self._check_dependency_compatibility()
        
        # Check thermal safety
        thermal_safety = 'good' if state.cpu_temperature_celsius < 75.0 else 'warning' if state.cpu_temperature_celsius < 85.0 else 'critical'
        
        # Check power efficiency
        power_efficiency = 'excellent' if state.power_draw_watts < 8.0 else 'good' if state.power_draw_watts < 12.0 else 'poor'
        
        # Check gaming mode compatibility
        gaming_compatibility = 'optimized' if state.gaming_mode_active else 'compatible'
        
        # Overall compatibility score
        scores = []
        if thermal_safety == 'good':
            scores.append(1.0)
        elif thermal_safety == 'warning':
            scores.append(0.6)
        else:
            scores.append(0.2)
        
        if power_efficiency == 'excellent':
            scores.append(1.0)
        elif power_efficiency == 'good':
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        scores.append(dependency_compatibility['score'])
        
        overall_score = sum(scores) / len(scores)
        
        return {
            'overall_score': overall_score,
            'overall_rating': 'excellent' if overall_score > 0.8 else 'good' if overall_score > 0.6 else 'needs_improvement',
            'thermal_safety': thermal_safety,
            'power_efficiency': power_efficiency,
            'gaming_compatibility': gaming_compatibility,
            'dependency_compatibility': dependency_compatibility,
            'hardware_profile': {
                'model': self.hardware_profile.model if self.hardware_profile else 'unknown',
                'cpu_cores': self.hardware_profile.cpu_cores if self.hardware_profile else 0,
                'memory_gb': self.hardware_profile.memory_gb if self.hardware_profile else 0,
                'storage_type': self.hardware_profile.storage_type if self.hardware_profile else 'unknown'
            } if self.is_steam_deck else None,
            'current_state': {
                'cpu_temperature': state.cpu_temperature_celsius,
                'memory_usage_mb': state.memory_usage_mb,
                'battery_percent': state.battery_percent,
                'power_draw_watts': state.power_draw_watts,
                'gaming_mode': state.gaming_mode_active,
                'dock_connected': state.dock_connected
            },
            'recommendations': self.get_optimization_recommendations()
        }
    
    def _check_dependency_compatibility(self) -> Dict[str, Any]:
        """Check compatibility of ML dependencies with Steam Deck"""
        try:
            from .dependency_coordinator import get_coordinator
            coordinator = get_coordinator()
            
            steam_deck_friendly = []
            problematic = []
            
            for name, state in coordinator.dependency_states.items():
                if state.available:
                    if state.spec.steam_deck_compatible:
                        steam_deck_friendly.append(name)
                    else:
                        problematic.append(name)
            
            score = len(steam_deck_friendly) / max(len(steam_deck_friendly) + len(problematic), 1)
            
            return {
                'score': score,
                'steam_deck_friendly': steam_deck_friendly,
                'problematic': problematic,
                'total_dependencies': len(steam_deck_friendly) + len(problematic)
            }
            
        except ImportError:
            return {
                'score': 0.5,
                'steam_deck_friendly': [],
                'problematic': [],
                'total_dependencies': 0,
                'error': 'Dependency coordinator not available'
            }
    
    def add_optimization_callback(self, callback: Callable) -> None:
        """Add callback for optimization profile changes"""
        self.optimization_callbacks.append(callback)
    
    def remove_optimization_callback(self, callback: Callable) -> None:
        """Remove optimization callback"""
        if callback in self.optimization_callbacks:
            self.optimization_callbacks.remove(callback)
    
    @contextmanager
    def temporary_optimization(self, profile_name: str):
        """Context manager for temporary optimization profile"""
        if not self.is_steam_deck:
            yield
            return
        
        # Get current state
        original_state = self.get_current_state()
        original_profile = self.select_optimal_profile(original_state)
        
        try:
            # Apply temporary profile
            if self.apply_optimization_profile(profile_name):
                yield
            else:
                yield  # Still yield even if profile application failed
        finally:
            # Restore original profile
            self.apply_optimization_profile(original_profile)
    
    def export_optimization_report(self, path: Path) -> None:
        """Export comprehensive optimization report"""
        state = self.get_current_state()
        compatibility = self.get_compatibility_report()
        recommendations = self.get_optimization_recommendations()
        
        report = {
            'timestamp': time.time(),
            'is_steam_deck': self.is_steam_deck,
            'hardware_profile': {
                'model': self.hardware_profile.model,
                'cpu_cores': self.hardware_profile.cpu_cores,
                'cpu_max_freq_mhz': self.hardware_profile.cpu_max_freq_mhz,
                'memory_gb': self.hardware_profile.memory_gb,
                'storage_type': self.hardware_profile.storage_type,
                'thermal_limit_celsius': self.hardware_profile.thermal_limit_celsius
            } if self.hardware_profile else None,
            'current_state': {
                'cpu_frequency_mhz': state.cpu_frequency_mhz,
                'cpu_temperature_celsius': state.cpu_temperature_celsius,
                'memory_usage_mb': state.memory_usage_mb,
                'battery_percent': state.battery_percent,
                'power_draw_watts': state.power_draw_watts,
                'thermal_throttling': state.thermal_throttling,
                'gaming_mode_active': state.gaming_mode_active,
                'dock_connected': state.dock_connected,
                'performance_governor': state.performance_governor
            },
            'compatibility_report': compatibility,
            'optimization_recommendations': recommendations,
            'available_profiles': list(self.performance_profiles.keys()),
            'thermal_history': self.thermal_history[-20:],  # Last 20 readings
            'power_history': self.power_history[-20:],      # Last 20 readings
            'optimization_active': self.optimization_active
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Steam Deck optimization report exported to {path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global optimizer instance
_steam_deck_optimizer: Optional[SteamDeckOptimizer] = None

def get_steam_deck_optimizer() -> SteamDeckOptimizer:
    """Get or create the global Steam Deck optimizer"""
    global _steam_deck_optimizer
    if _steam_deck_optimizer is None:
        _steam_deck_optimizer = SteamDeckOptimizer()
    return _steam_deck_optimizer

def is_steam_deck() -> bool:
    """Quick check if running on Steam Deck"""
    optimizer = get_steam_deck_optimizer()
    return optimizer.is_steam_deck

def optimize_for_steam_deck() -> bool:
    """Apply optimal Steam Deck optimizations"""
    optimizer = get_steam_deck_optimizer()
    if not optimizer.is_steam_deck:
        return False
    
    state = optimizer.get_current_state()
    optimal_profile = optimizer.select_optimal_profile(state)
    return optimizer.apply_optimization_profile(optimal_profile)

def get_steam_deck_state() -> SteamDeckState:
    """Get current Steam Deck state"""
    optimizer = get_steam_deck_optimizer()
    return optimizer.get_current_state()

@contextmanager
def gaming_mode_optimization():
    """Context manager for gaming mode optimization"""
    optimizer = get_steam_deck_optimizer()
    with optimizer.temporary_optimization('gaming'):
        yield

@contextmanager
def maximum_performance():
    """Context manager for maximum performance"""
    optimizer = get_steam_deck_optimizer()
    with optimizer.temporary_optimization('maximum'):
        yield

@contextmanager
def battery_saving():
    """Context manager for battery saving"""
    optimizer = get_steam_deck_optimizer()
    with optimizer.temporary_optimization('battery'):
        yield


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n🎮 Steam Deck Optimizer Test Suite")
    print("=" * 45)
    
    # Create optimizer
    optimizer = SteamDeckOptimizer()
    
    # Show detection results
    print(f"\n📱 Steam Deck Detection:")
    print(f"  Is Steam Deck: {optimizer.is_steam_deck}")
    
    if optimizer.hardware_profile:
        print(f"  Model: {optimizer.hardware_profile.model}")
        print(f"  CPU Cores: {optimizer.hardware_profile.cpu_cores}")
        print(f"  Max CPU Freq: {optimizer.hardware_profile.cpu_max_freq_mhz} MHz")
        print(f"  Memory: {optimizer.hardware_profile.memory_gb:.1f} GB")
        print(f"  Storage: {optimizer.hardware_profile.storage_type}")
        print(f"  Battery: {optimizer.hardware_profile.battery_capacity_wh} Wh")
    
    # Get current state
    print(f"\n📊 Current State:")
    state = optimizer.get_current_state()
    print(f"  CPU Frequency: {state.cpu_frequency_mhz} MHz")
    print(f"  CPU Temperature: {state.cpu_temperature_celsius:.1f}°C")
    print(f"  Memory Usage: {state.memory_usage_mb:.0f} MB")
    print(f"  Battery: {state.battery_percent:.0f}%")
    print(f"  Power Draw: {state.power_draw_watts:.1f}W")
    print(f"  Gaming Mode: {state.gaming_mode_active}")
    print(f"  Dock Connected: {state.dock_connected}")
    print(f"  Thermal Throttling: {state.thermal_throttling}")
    
    # Show available profiles
    print(f"\n🎯 Available Profiles:")
    for profile_name, profile in optimizer.performance_profiles.items():
        print(f"  • {profile_name}: {profile['description']}")
    
    # Select optimal profile
    optimal_profile = optimizer.select_optimal_profile(state)
    print(f"\n🚀 Optimal Profile: {optimal_profile}")
    
    # Get optimization recommendations
    print(f"\n💡 Optimization Recommendations:")
    recommendations = optimizer.get_optimization_recommendations()
    if recommendations:
        for rec in recommendations:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢", "info": "ℹ️"}.get(rec['priority'], "ℹ️")
            print(f"  {priority_emoji} {rec['title']}: {rec['description']}")
            print(f"    Action: {rec['action']}")
    else:
        print("  ✅ No optimization recommendations - system is running optimally")
    
    # Get compatibility report
    print(f"\n🔍 Compatibility Report:")
    compatibility = optimizer.get_compatibility_report()
    print(f"  Overall Rating: {compatibility['overall_rating']} ({compatibility['overall_score']:.1%})")
    print(f"  Thermal Safety: {compatibility['thermal_safety']}")
    print(f"  Power Efficiency: {compatibility['power_efficiency']}")
    print(f"  Gaming Compatibility: {compatibility['gaming_compatibility']}")
    
    # Test profile application
    print(f"\n🔧 Testing Profile Application:")
    test_profiles = ['balanced', 'battery', 'gaming']
    for profile_name in test_profiles:
        if profile_name in optimizer.performance_profiles:
            success = optimizer.apply_optimization_profile(profile_name)
            print(f"  Apply {profile_name}: {'✅' if success else '❌'}")
    
    # Test context managers
    print(f"\n🎭 Testing Context Managers:")
    
    print(f"  Before context: CPU temp = {optimizer.get_current_state().cpu_temperature_celsius:.1f}°C")
    
    with optimizer.temporary_optimization('battery'):
        temp_in_context = optimizer.get_current_state().cpu_temperature_celsius
        print(f"  In battery context: CPU temp = {temp_in_context:.1f}°C")
    
    temp_after_context = optimizer.get_current_state().cpu_temperature_celsius
    print(f"  After context: CPU temp = {temp_after_context:.1f}°C")
    
    # Export report
    report_path = Path("/tmp/steam_deck_optimization_report.json")
    optimizer.export_optimization_report(report_path)
    print(f"\n💾 Optimization report exported to {report_path}")
    
    print(f"\n✅ Steam Deck Optimizer test completed successfully!")
    
    if optimizer.is_steam_deck:
        print(f"🎮 Running on Steam Deck {optimizer.hardware_profile.model} - optimizations active!")
    else:
        print(f"💻 Not running on Steam Deck - generic optimizations available")