#!/usr/bin/env python3
"""
Steam Deck Hardware Integration for ML Shader Prediction

This module provides comprehensive Steam Deck hardware integration including:
- LCD/OLED model detection and optimization
- Thermal management integration with shader prediction
- Battery optimization and power state monitoring
- Dock detection and performance scaling
- Gaming Mode hardware event handling
- APU-specific optimizations for RDNA2 and Zen2
- Hardware-aware shader prediction prioritization

Features:
- Real-time hardware monitoring with minimal overhead
- Adaptive ML prediction based on thermal conditions
- Battery-aware shader compilation scheduling
- Dock-aware performance optimization
- Gaming Mode priority management
- Hardware event-driven optimization switching
"""

import os
import sys
import time
import json
import asyncio
import threading
import subprocess
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import struct

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# STEAM DECK HARDWARE DETECTION
# =============================================================================

class SteamDeckModel(Enum):
    """Steam Deck model variants"""
    LCD_64GB = "lcd_64gb"
    LCD_256GB = "lcd_256gb"
    LCD_512GB = "lcd_512gb"
    OLED_512GB = "oled_512gb"
    OLED_1TB = "oled_1tb"
    UNKNOWN = "unknown"

class PowerState(Enum):
    """Power states"""
    AC_POWERED = "ac_powered"
    BATTERY_HIGH = "battery_high"      # >80%
    BATTERY_MEDIUM = "battery_medium"  # 40-80%
    BATTERY_LOW = "battery_low"        # 20-40%
    BATTERY_CRITICAL = "battery_critical"  # <20%

class ThermalState(Enum):
    """Thermal states"""
    COOL = "cool"          # <60°C
    WARM = "warm"          # 60-70°C
    HOT = "hot"            # 70-80°C
    CRITICAL = "critical"  # >80°C

class DockState(Enum):
    """Dock connection states"""
    UNDOCKED = "undocked"
    DOCKED_HDMI = "docked_hdmi"
    DOCKED_DP = "docked_dp"
    DOCKED_USB = "docked_usb"

@dataclass
class SteamDeckHardwareState:
    """Current Steam Deck hardware state"""
    model: SteamDeckModel
    power_state: PowerState
    thermal_state: ThermalState
    dock_state: DockState
    
    # Detailed metrics
    cpu_temperature: float
    cpu_frequency: int
    gpu_frequency: int
    memory_usage_mb: float
    battery_percentage: float
    battery_time_remaining_min: Optional[float]
    power_draw_watts: float
    fan_speed_rpm: Optional[int]
    
    # Performance indicators
    thermal_throttling: bool
    power_throttling: bool
    gaming_mode_active: bool
    performance_governor: str
    
    # Display and dock info
    internal_display_active: bool
    external_displays: List[str]
    dock_power_delivery: bool
    
    timestamp: float = field(default_factory=time.time)

@dataclass
class HardwareOptimizationProfile:
    """Hardware-specific optimization profile"""
    name: str
    description: str
    
    # ML prediction settings
    prediction_frequency_hz: float
    prediction_priority: str  # 'high', 'medium', 'low', 'minimal'
    prediction_batch_size: int
    
    # Compilation settings
    compilation_thread_limit: int
    compilation_priority: str
    async_compilation: bool
    
    # Cache settings
    cache_aggressiveness: str  # 'conservative', 'balanced', 'aggressive'
    cache_cleanup_frequency_min: int
    
    # Thermal management
    thermal_limit_celsius: float
    thermal_throttle_threshold: float
    
    # Power management
    power_limit_watts: Optional[float]
    cpu_governor: str
    gpu_power_state: str

# =============================================================================
# HARDWARE MONITORING
# =============================================================================

class SteamDeckHardwareMonitor:
    """
    Comprehensive Steam Deck hardware monitoring system
    """
    
    def __init__(self):
        self.is_steam_deck = self._detect_steam_deck()
        self.model = self._detect_model() if self.is_steam_deck else SteamDeckModel.UNKNOWN
        self.monitoring_active = False
        self.state_callbacks: List[Callable] = []
        self.hardware_state: Optional[SteamDeckHardwareState] = None
        self.optimization_profiles = self._create_optimization_profiles()
        
        # Integrate with existing Steam Deck optimizer
        self.deck_optimizer = None
        try:
            from .steam_deck_optimizer import get_steam_deck_optimizer
            self.deck_optimizer = get_steam_deck_optimizer()
            logger.info("Integrated with existing Steam Deck optimizer")
        except ImportError:
            logger.warning("Steam Deck optimizer not available")
        
        if self.is_steam_deck:
            logger.info(f"Steam Deck hardware monitor initialized for {self.model.value}")
        else:
            logger.info("Hardware monitor initialized for non-Steam Deck system")
    
    def _detect_steam_deck(self) -> bool:
        """Detect if running on Steam Deck"""
        detection_methods = [
            lambda: os.path.exists('/home/deck'),
            lambda: self._check_dmi_identifiers(),
            lambda: self._check_cpu_signature(),
            lambda: self._check_steam_deck_environment()
        ]
        
        return any(method() for method in detection_methods)
    
    def _check_dmi_identifiers(self) -> bool:
        """Check DMI system identifiers"""
        try:
            dmi_files = [
                '/sys/devices/virtual/dmi/id/product_name',
                '/sys/devices/virtual/dmi/id/board_name',
                '/sys/devices/virtual/dmi/id/sys_vendor'
            ]
            
            steam_deck_identifiers = ['jupiter', 'galileo', 'valve']
            
            for dmi_file in dmi_files:
                if os.path.exists(dmi_file):
                    with open(dmi_file, 'r') as f:
                        content = f.read().strip().lower()
                        if any(identifier in content for identifier in steam_deck_identifiers):
                            return True
        except Exception:
            pass
        
        return False
    
    def _check_cpu_signature(self) -> bool:
        """Check CPU signature for Steam Deck APU"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read().lower()
                # Steam Deck uses AMD Van Gogh APU
                return 'van gogh' in content or ('amd custom apu' in content and '0x90' in content)
        except Exception:
            return False
    
    def _check_steam_deck_environment(self) -> bool:
        """Check for Steam Deck-specific environment"""
        return os.environ.get('SteamDeck') is not None
    
    def _detect_model(self) -> SteamDeckModel:
        """Detect specific Steam Deck model"""
        if not self.is_steam_deck:
            return SteamDeckModel.UNKNOWN
        
        try:
            # Check DMI for model-specific identifiers
            product_files = [
                '/sys/devices/virtual/dmi/id/product_name',
                '/sys/devices/virtual/dmi/id/board_name'
            ]
            
            for file_path in product_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read().strip().lower()
                        
                        # OLED models use "Galileo" codename
                        if 'galileo' in content:
                            # Check storage to differentiate OLED models
                            storage_size = self._detect_storage_size()
                            if storage_size >= 900:  # 1TB model
                                return SteamDeckModel.OLED_1TB
                            else:
                                return SteamDeckModel.OLED_512GB
                        
                        # LCD models use "Jupiter" codename
                        elif 'jupiter' in content:
                            storage_size = self._detect_storage_size()
                            if storage_size >= 400:  # 512GB model
                                return SteamDeckModel.LCD_512GB
                            elif storage_size >= 200:  # 256GB model
                                return SteamDeckModel.LCD_256GB
                            else:  # 64GB model
                                return SteamDeckModel.LCD_64GB
            
            # Fallback detection based on OLED-specific hardware
            if self._has_oled_hardware():
                return SteamDeckModel.OLED_512GB
            else:
                return SteamDeckModel.LCD_256GB  # Most common LCD model
                
        except Exception as e:
            logger.error(f"Model detection error: {e}")
            return SteamDeckModel.UNKNOWN
    
    def _detect_storage_size(self) -> int:
        """Detect internal storage size in GB"""
        try:
            # Check for main storage device
            storage_devices = [
                '/dev/nvme0n1',  # NVMe SSD
                '/dev/mmcblk0'   # eMMC
            ]
            
            for device in storage_devices:
                if os.path.exists(device):
                    result = subprocess.run(['lsblk', '-bno', 'SIZE', device], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        size_bytes = int(result.stdout.strip())
                        size_gb = size_bytes // (1000**3)  # Using GB (not GiB)
                        return size_gb
        except Exception:
            pass
        
        return 0
    
    def _has_oled_hardware(self) -> bool:
        """Check for OLED-specific hardware indicators"""
        oled_indicators = [
            '/sys/class/backlight/amdgpu_bl1',  # OLED backlight control
            '/sys/devices/platform/jupiter'     # OLED platform device
        ]
        
        return any(os.path.exists(indicator) for indicator in oled_indicators)
    
    def _create_optimization_profiles(self) -> Dict[str, HardwareOptimizationProfile]:
        """Create hardware-specific optimization profiles"""
        profiles = {}
        
        # Maximum performance (docked, cool, AC power)
        profiles['maximum'] = HardwareOptimizationProfile(
            name='maximum',
            description='Maximum performance for optimal conditions',
            prediction_frequency_hz=30.0,
            prediction_priority='high',
            prediction_batch_size=8,
            compilation_thread_limit=8,
            compilation_priority='high',
            async_compilation=True,
            cache_aggressiveness='aggressive',
            cache_cleanup_frequency_min=60,
            thermal_limit_celsius=75.0,
            thermal_throttle_threshold=70.0,
            power_limit_watts=None,
            cpu_governor='performance',
            gpu_power_state='high'
        )
        
        # Balanced performance (normal handheld use)
        profiles['balanced'] = HardwareOptimizationProfile(
            name='balanced',
            description='Balanced performance and efficiency',
            prediction_frequency_hz=15.0,
            prediction_priority='medium',
            prediction_batch_size=4,
            compilation_thread_limit=4,
            compilation_priority='normal',
            async_compilation=True,
            cache_aggressiveness='balanced',
            cache_cleanup_frequency_min=120,
            thermal_limit_celsius=70.0,
            thermal_throttle_threshold=65.0,
            power_limit_watts=12.0,
            cpu_governor='schedutil',
            gpu_power_state='medium'
        )
        
        # Gaming mode (background operation)
        profiles['gaming'] = HardwareOptimizationProfile(
            name='gaming',
            description='Background operation during gaming',
            prediction_frequency_hz=5.0,
            prediction_priority='low',
            prediction_batch_size=2,
            compilation_thread_limit=2,
            compilation_priority='background',
            async_compilation=True,
            cache_aggressiveness='conservative',
            cache_cleanup_frequency_min=300,
            thermal_limit_celsius=80.0,
            thermal_throttle_threshold=75.0,
            power_limit_watts=8.0,
            cpu_governor='powersave',
            gpu_power_state='low'
        )
        
        # Battery optimization
        profiles['battery'] = HardwareOptimizationProfile(
            name='battery',
            description='Maximum battery life optimization',
            prediction_frequency_hz=2.0,
            prediction_priority='minimal',
            prediction_batch_size=1,
            compilation_thread_limit=1,
            compilation_priority='idle',
            async_compilation=False,
            cache_aggressiveness='conservative',
            cache_cleanup_frequency_min=600,
            thermal_limit_celsius=65.0,
            thermal_throttle_threshold=60.0,
            power_limit_watts=6.0,
            cpu_governor='powersave',
            gpu_power_state='minimal'
        )
        
        # Thermal emergency
        profiles['thermal_emergency'] = HardwareOptimizationProfile(
            name='thermal_emergency',
            description='Emergency thermal protection',
            prediction_frequency_hz=0.5,
            prediction_priority='minimal',
            prediction_batch_size=1,
            compilation_thread_limit=1,
            compilation_priority='idle',
            async_compilation=False,
            cache_aggressiveness='conservative',
            cache_cleanup_frequency_min=1200,
            thermal_limit_celsius=90.0,
            thermal_throttle_threshold=85.0,
            power_limit_watts=4.0,
            cpu_governor='powersave',
            gpu_power_state='minimal'
        )
        
        return profiles
    
    async def start_monitoring(self, update_interval: float = 5.0) -> None:
        """Start hardware monitoring"""
        if not self.is_steam_deck:
            logger.info("Not running on Steam Deck, skipping hardware monitoring")
            return
        
        self.monitoring_active = True
        logger.info("Starting Steam Deck hardware monitoring")
        
        while self.monitoring_active:
            try:
                # Update hardware state
                new_state = await self._get_current_hardware_state()
                
                # Check for significant state changes
                if self._state_changed_significantly(self.hardware_state, new_state):
                    logger.info(f"Hardware state change detected: {new_state.thermal_state.value}, {new_state.power_state.value}")
                    
                    # Notify callbacks
                    for callback in self.state_callbacks:
                        try:
                            callback(new_state)
                        except Exception as e:
                            logger.error(f"Hardware state callback error: {e}")
                
                self.hardware_state = new_state
                
                # Sleep until next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Hardware monitoring error: {e}")
                await asyncio.sleep(update_interval * 2)
    
    def stop_monitoring(self) -> None:
        """Stop hardware monitoring"""
        self.monitoring_active = False
        logger.info("Steam Deck hardware monitoring stopped")
    
    async def _get_current_hardware_state(self) -> SteamDeckHardwareState:
        """Get current hardware state"""
        
        # CPU metrics
        cpu_temp = self._get_cpu_temperature()
        cpu_freq = self._get_cpu_frequency()
        
        # GPU metrics
        gpu_freq = self._get_gpu_frequency()
        
        # Memory metrics
        memory_usage = self._get_memory_usage()
        
        # Power metrics
        battery_percent, battery_time, power_draw = self._get_power_metrics()
        
        # Fan metrics
        fan_speed = self._get_fan_speed()
        
        # System state
        thermal_throttling = self._is_thermal_throttling()
        power_throttling = self._is_power_throttling()
        gaming_mode = self._is_gaming_mode_active()
        governor = self._get_cpu_governor()
        
        # Display and dock
        displays = self._get_display_info()
        dock_state = self._get_dock_state()
        
        # Determine states
        power_state = self._determine_power_state(battery_percent, dock_state != DockState.UNDOCKED)
        thermal_state = self._determine_thermal_state(cpu_temp, thermal_throttling)
        
        return SteamDeckHardwareState(
            model=self.model,
            power_state=power_state,
            thermal_state=thermal_state,
            dock_state=dock_state,
            cpu_temperature=cpu_temp,
            cpu_frequency=cpu_freq,
            gpu_frequency=gpu_freq,
            memory_usage_mb=memory_usage,
            battery_percentage=battery_percent,
            battery_time_remaining_min=battery_time,
            power_draw_watts=power_draw,
            fan_speed_rpm=fan_speed,
            thermal_throttling=thermal_throttling,
            power_throttling=power_throttling,
            gaming_mode_active=gaming_mode,
            performance_governor=governor,
            internal_display_active=displays['internal_active'],
            external_displays=displays['external_displays'],
            dock_power_delivery=dock_state != DockState.UNDOCKED
        )
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature"""
        if self.deck_optimizer:
            return self.deck_optimizer._get_cpu_temperature()
        
        # Fallback implementation
        try:
            thermal_zones = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp'
            ]
            
            for zone in thermal_zones:
                if os.path.exists(zone):
                    with open(zone, 'r') as f:
                        temp_millic = int(f.read().strip())
                        temp_celsius = temp_millic / 1000.0
                        if 20.0 <= temp_celsius <= 120.0:
                            return temp_celsius
        except Exception:
            pass
        
        return 50.0  # Safe fallback
    
    def _get_cpu_frequency(self) -> int:
        """Get current CPU frequency in MHz"""
        try:
            freq_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'
            if os.path.exists(freq_file):
                with open(freq_file, 'r') as f:
                    freq_khz = int(f.read().strip())
                    return freq_khz // 1000
        except Exception:
            pass
        
        return 2000  # Fallback
    
    def _get_gpu_frequency(self) -> int:
        """Get current GPU frequency in MHz"""
        try:
            # AMD GPU frequency paths for Steam Deck
            gpu_freq_paths = [
                '/sys/class/drm/card0/device/pp_dpm_sclk',
                '/sys/class/drm/card1/device/pp_dpm_sclk'
            ]
            
            for freq_path in gpu_freq_paths:
                if os.path.exists(freq_path):
                    with open(freq_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if '*' in line:  # Current frequency marked with *
                                freq_str = line.split(':')[1].split('MHz')[0].strip()
                                return int(freq_str)
        except Exception:
            pass
        
        return 800  # Fallback GPU frequency
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.deck_optimizer:
            return self.deck_optimizer._get_memory_usage_mb()
        
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    key, value = line.split(':')
                    meminfo[key.strip()] = int(value.strip().split()[0])
                
                total_mb = meminfo['MemTotal'] / 1024
                available_mb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0)) / 1024
                used_mb = total_mb - available_mb
                return used_mb
        except Exception:
            return 4096.0  # Fallback
    
    def _get_power_metrics(self) -> Tuple[float, Optional[float], float]:
        """Get battery percentage, time remaining, and power draw"""
        if self.deck_optimizer:
            return self.deck_optimizer._get_battery_info()
        
        battery_percent = 100.0
        battery_time = None
        power_draw = 10.0
        
        try:
            battery_path = '/sys/class/power_supply/BAT1'
            if os.path.exists(battery_path):
                # Battery capacity
                capacity_file = os.path.join(battery_path, 'capacity')
                if os.path.exists(capacity_file):
                    with open(capacity_file, 'r') as f:
                        battery_percent = float(f.read().strip())
                
                # Power draw
                power_file = os.path.join(battery_path, 'power_now')
                if os.path.exists(power_file):
                    with open(power_file, 'r') as f:
                        power_uw = int(f.read().strip())
                        power_draw = power_uw / 1000000.0
                
                # Calculate time remaining (simplified)
                if battery_percent > 0 and power_draw > 0:
                    # Assume 40Wh for LCD, 50Wh for OLED
                    battery_wh = 50.0 if self.model in [SteamDeckModel.OLED_512GB, SteamDeckModel.OLED_1TB] else 40.0
                    remaining_wh = battery_wh * (battery_percent / 100.0)
                    battery_time = (remaining_wh / power_draw) * 60  # Minutes
        except Exception:
            pass
        
        return battery_percent, battery_time, power_draw
    
    def _get_fan_speed(self) -> Optional[int]:
        """Get fan speed in RPM"""
        try:
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
    
    def _is_thermal_throttling(self) -> bool:
        """Check if system is thermal throttling"""
        if self.deck_optimizer:
            return self.deck_optimizer._is_thermal_throttling()
        
        try:
            # Check CPU frequency vs maximum
            current_freq = self._get_cpu_frequency()
            temp = self._get_cpu_temperature()
            
            # Simple heuristic: if temp > 75°C and frequency is reduced
            if temp > 75.0 and current_freq < 3000:  # Below typical max frequency
                return True
        except Exception:
            pass
        
        return False
    
    def _is_power_throttling(self) -> bool:
        """Check if system is power throttling"""
        try:
            # Check if on battery and power draw is high
            battery_percent, _, power_draw = self._get_power_metrics()
            dock_state = self._get_dock_state()
            
            # Power throttling likely if on battery with high power draw
            if dock_state == DockState.UNDOCKED and power_draw > 15.0:
                return True
        except Exception:
            pass
        
        return False
    
    def _is_gaming_mode_active(self) -> bool:
        """Check if Gaming Mode is active"""
        try:
            result = subprocess.run(['pgrep', '-f', 'gamescope'], 
                                  capture_output=True, timeout=3)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_cpu_governor(self) -> str:
        """Get current CPU governor"""
        try:
            governor_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
            if os.path.exists(governor_file):
                with open(governor_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        
        return 'unknown'
    
    def _get_display_info(self) -> Dict[str, Any]:
        """Get display information"""
        display_info = {
            'internal_active': True,
            'external_displays': []
        }
        
        try:
            # Check for external displays
            display_paths = [
                '/sys/class/drm/card0-DP-1/status',
                '/sys/class/drm/card0-DP-2/status',
                '/sys/class/drm/card0-HDMI-A-1/status'
            ]
            
            for i, display_path in enumerate(display_paths):
                if os.path.exists(display_path):
                    with open(display_path, 'r') as f:
                        status = f.read().strip()
                        if status == 'connected':
                            display_type = 'DP' if 'DP' in display_path else 'HDMI'
                            display_info['external_displays'].append(f"{display_type}-{i+1}")
        except Exception:
            pass
        
        return display_info
    
    def _get_dock_state(self) -> DockState:
        """Get dock connection state"""
        try:
            display_info = self._get_display_info()
            
            # Check for external displays
            if display_info['external_displays']:
                for display in display_info['external_displays']:
                    if 'HDMI' in display:
                        return DockState.DOCKED_HDMI
                    elif 'DP' in display:
                        return DockState.DOCKED_DP
            
            # Check for AC adapter (simple dock detection)
            if os.path.exists('/sys/class/power_supply/ADP1'):
                return DockState.DOCKED_USB
                
        except Exception:
            pass
        
        return DockState.UNDOCKED
    
    def _determine_power_state(self, battery_percent: float, is_docked: bool) -> PowerState:
        """Determine power state from battery and dock status"""
        if is_docked:
            return PowerState.AC_POWERED
        elif battery_percent > 80:
            return PowerState.BATTERY_HIGH
        elif battery_percent > 40:
            return PowerState.BATTERY_MEDIUM
        elif battery_percent > 20:
            return PowerState.BATTERY_LOW
        else:
            return PowerState.BATTERY_CRITICAL
    
    def _determine_thermal_state(self, temperature: float, throttling: bool) -> ThermalState:
        """Determine thermal state from temperature and throttling"""
        if throttling or temperature > 80:
            return ThermalState.CRITICAL
        elif temperature > 70:
            return ThermalState.HOT
        elif temperature > 60:
            return ThermalState.WARM
        else:
            return ThermalState.COOL
    
    def _state_changed_significantly(self, old_state: Optional[SteamDeckHardwareState], 
                                   new_state: SteamDeckHardwareState) -> bool:
        """Check if hardware state changed significantly"""
        if old_state is None:
            return True
        
        # Check for significant state changes
        significant_changes = [
            old_state.power_state != new_state.power_state,
            old_state.thermal_state != new_state.thermal_state,
            old_state.dock_state != new_state.dock_state,
            old_state.gaming_mode_active != new_state.gaming_mode_active,
            abs(old_state.cpu_temperature - new_state.cpu_temperature) > 5.0,
            abs(old_state.battery_percentage - new_state.battery_percentage) > 10.0
        ]
        
        return any(significant_changes)
    
    def get_optimal_profile(self, state: Optional[SteamDeckHardwareState] = None) -> str:
        """Get optimal optimization profile for current state"""
        if state is None:
            state = self.hardware_state
        
        if state is None:
            return 'balanced'
        
        # Emergency thermal protection
        if state.thermal_state == ThermalState.CRITICAL:
            return 'thermal_emergency'
        
        # Gaming mode (background operation)
        if state.gaming_mode_active:
            return 'gaming'
        
        # Battery-based decisions
        if state.power_state in [PowerState.BATTERY_LOW, PowerState.BATTERY_CRITICAL]:
            return 'battery'
        
        # Docked with good conditions = maximum performance
        if state.dock_state != DockState.UNDOCKED and state.thermal_state == ThermalState.COOL:
            return 'maximum'
        
        # Default balanced profile
        return 'balanced'
    
    def get_prediction_parameters(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Get ML prediction parameters for current conditions"""
        if profile_name is None:
            profile_name = self.get_optimal_profile()
        
        profile = self.optimization_profiles.get(profile_name, self.optimization_profiles['balanced'])
        
        return {
            'frequency_hz': profile.prediction_frequency_hz,
            'priority': profile.prediction_priority,
            'batch_size': profile.prediction_batch_size,
            'thread_limit': profile.compilation_thread_limit,
            'async_compilation': profile.async_compilation,
            'cache_aggressiveness': profile.cache_aggressiveness,
            'thermal_limit': profile.thermal_limit_celsius,
            'power_limit': profile.power_limit_watts
        }
    
    def add_state_callback(self, callback: Callable) -> None:
        """Add hardware state change callback"""
        self.state_callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable) -> None:
        """Remove hardware state change callback"""
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)
    
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """Get Steam Deck hardware capabilities"""
        if not self.is_steam_deck:
            return {'is_steam_deck': False}
        
        capabilities = {
            'is_steam_deck': True,
            'model': self.model.value,
            'has_oled_display': self.model in [SteamDeckModel.OLED_512GB, SteamDeckModel.OLED_1TB],
            'apu_architecture': 'RDNA2/Zen2',
            'cpu_cores': 4,
            'cpu_threads': 8,
            'gpu_compute_units': 8,
            'memory_gb': 16,
            'memory_type': 'LPDDR5',
            'storage_type': 'NVMe' if self.model in [SteamDeckModel.LCD_512GB, SteamDeckModel.OLED_512GB, SteamDeckModel.OLED_1TB] else 'eMMC',
            'dock_support': True,
            'external_display_support': True,
            'vulkan_support': True,
            'directx_support': True,  # Via Proton
            'hardware_video_decode': True,
            'hardware_video_encode': True
        }
        
        return capabilities

# =============================================================================
# HARDWARE-AWARE ML INTEGRATION
# =============================================================================

class HardwareAwareMLScheduler:
    """
    Schedules ML shader predictions based on hardware conditions
    """
    
    def __init__(self, hardware_monitor: SteamDeckHardwareMonitor):
        self.hardware_monitor = hardware_monitor
        self.ml_predictor = None
        self.prediction_queue: List[Dict[str, Any]] = []
        self.active_predictions: Set[str] = set()
        self.scheduler_active = False
        
        # Add ourselves as a hardware state callback
        self.hardware_monitor.add_state_callback(self._on_hardware_state_change)
        
        logger.info("Hardware-aware ML scheduler initialized")
    
    def set_ml_predictor(self, predictor):
        """Set the ML predictor instance"""
        self.ml_predictor = predictor
        logger.info("ML predictor integrated with hardware scheduler")
    
    def _on_hardware_state_change(self, new_state: SteamDeckHardwareState) -> None:
        """Handle hardware state changes"""
        logger.info(f"Hardware state changed - adjusting ML scheduling")
        
        # Get new prediction parameters
        profile_name = self.hardware_monitor.get_optimal_profile(new_state)
        params = self.hardware_monitor.get_prediction_parameters(profile_name)
        
        # Adjust prediction scheduling based on new conditions
        if new_state.thermal_state == ThermalState.CRITICAL:
            logger.warning("Critical thermal state - suspending ML predictions")
            self._suspend_predictions()
        elif new_state.gaming_mode_active:
            logger.info("Gaming mode active - reducing ML prediction frequency")
            self._reduce_prediction_frequency(params['frequency_hz'])
        elif new_state.power_state == PowerState.BATTERY_CRITICAL:
            logger.warning("Critical battery state - minimizing ML predictions")
            self._minimize_predictions()
        else:
            logger.info(f"Normal operation - using {profile_name} profile")
            self._resume_normal_predictions(params)
    
    def _suspend_predictions(self) -> None:
        """Suspend ML predictions temporarily"""
        # Implementation would suspend active predictions
        logger.info("ML predictions suspended due to hardware conditions")
    
    def _reduce_prediction_frequency(self, new_frequency: float) -> None:
        """Reduce prediction frequency"""
        logger.info(f"Reducing ML prediction frequency to {new_frequency} Hz")
    
    def _minimize_predictions(self) -> None:
        """Minimize predictions for battery saving"""
        logger.info("Minimizing ML predictions for battery conservation")
    
    def _resume_normal_predictions(self, params: Dict[str, Any]) -> None:
        """Resume normal prediction operation"""
        logger.info(f"Resuming normal ML predictions with {params['priority']} priority")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global hardware monitor instance
_hardware_monitor: Optional[SteamDeckHardwareMonitor] = None

def get_hardware_monitor() -> SteamDeckHardwareMonitor:
    """Get or create the global hardware monitor"""
    global _hardware_monitor
    if _hardware_monitor is None:
        _hardware_monitor = SteamDeckHardwareMonitor()
    return _hardware_monitor

async def start_hardware_monitoring():
    """Start hardware monitoring"""
    monitor = get_hardware_monitor()
    await monitor.start_monitoring()

def get_current_hardware_state() -> Optional[SteamDeckHardwareState]:
    """Get current hardware state"""
    monitor = get_hardware_monitor()
    return monitor.hardware_state

def get_optimal_ml_parameters() -> Dict[str, Any]:
    """Get optimal ML parameters for current hardware conditions"""
    monitor = get_hardware_monitor()
    return monitor.get_prediction_parameters()

@contextmanager
def hardware_aware_operation(operation_type: str = 'prediction'):
    """Context manager for hardware-aware operations"""
    monitor = get_hardware_monitor()
    state = monitor.hardware_state
    
    if state and state.thermal_state == ThermalState.CRITICAL:
        logger.warning(f"Skipping {operation_type} due to critical thermal conditions")
        yield False
    else:
        yield True

# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

async def test_hardware_integration():
    """Test Steam Deck hardware integration"""
    print("\n⚙️ Steam Deck Hardware Integration Test")
    print("=" * 50)
    
    monitor = SteamDeckHardwareMonitor()
    
    # Test hardware detection
    print(f"\n🔍 Hardware Detection:")
    print(f"  Is Steam Deck: {monitor.is_steam_deck}")
    print(f"  Model: {monitor.model.value}")
    
    if monitor.is_steam_deck:
        capabilities = monitor.get_hardware_capabilities()
        print(f"\n📋 Hardware Capabilities:")
        for key, value in capabilities.items():
            print(f"  {key}: {value}")
        
        # Test state monitoring
        print(f"\n📊 Getting Hardware State:")
        state = await monitor._get_current_hardware_state()
        
        print(f"  Power State: {state.power_state.value}")
        print(f"  Thermal State: {state.thermal_state.value}")
        print(f"  Dock State: {state.dock_state.value}")
        print(f"  CPU Temperature: {state.cpu_temperature:.1f}°C")
        print(f"  Battery: {state.battery_percentage:.0f}%")
        print(f"  Power Draw: {state.power_draw_watts:.1f}W")
        print(f"  Gaming Mode: {state.gaming_mode_active}")
        
        # Test optimization profiles
        print(f"\n🎯 Optimization Profiles:")
        optimal_profile = monitor.get_optimal_profile(state)
        print(f"  Optimal Profile: {optimal_profile}")
        
        params = monitor.get_prediction_parameters(optimal_profile)
        print(f"  Prediction Parameters:")
        for key, value in params.items():
            print(f"    {key}: {value}")
        
        # Test hardware monitoring
        print(f"\n👀 Starting Hardware Monitoring (5 seconds):")
        
        def state_callback(new_state):
            print(f"  State Update: {new_state.thermal_state.value}, {new_state.power_state.value}")
        
        monitor.add_state_callback(state_callback)
        
        # Start monitoring briefly
        monitoring_task = asyncio.create_task(monitor.start_monitoring(update_interval=1.0))
        await asyncio.sleep(5)
        monitor.stop_monitoring()
        
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
    
    print(f"\n✅ Steam Deck hardware integration test completed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        asyncio.run(test_hardware_integration())
    else:
        print("Steam Deck Hardware Integration")
        print("Usage: --test to run test suite")