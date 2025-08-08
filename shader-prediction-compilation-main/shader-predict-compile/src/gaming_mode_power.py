#!/usr/bin/env python3

import os
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import subprocess
import logging

class PowerProfile(Enum):
    """Power profiles optimized for different scenarios"""
    BATTERY_SAVE = "battery_save"      # Maximum battery life
    BALANCED = "balanced"              # Balanced performance/battery  
    PERFORMANCE = "performance"        # Maximum performance
    HANDHELD_GAMING = "handheld_gaming"  # Optimized for 40-60 FPS gaming
    DOCKED_GAMING = "docked_gaming"    # Unlimited power, max performance

@dataclass
class PowerSettings:
    cpu_governor: str
    cpu_max_freq: Optional[int]  # MHz
    gpu_power_limit: Optional[int]  # Watts
    tdp_limit: Optional[int]  # Watts (4-15W range for Steam Deck)
    fan_curve: str
    memory_frequency: Optional[int]  # MHz
    shader_compile_threads: int
    background_priority: int  # Nice value

class GamingModePowerManager:
    """
    Gaming Mode specific power management implementing Steam Deck's
    configurable 4-15W TDP and adaptive resource management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('gaming_power_manager')
        self.current_profile = PowerProfile.BALANCED
        self.is_gaming_mode = False
        self.is_docked = False
        self.battery_level = 100
        
        # Steam Deck specific power paths
        self.power_paths = {
            'cpu_governor': '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor',
            'cpu_max_freq': '/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq',
            'gpu_power_cap': '/sys/class/drm/card0/device/power_dpm_force_performance_level',
            'tdp_control': '/sys/class/hwmon/hwmon0/power1_cap',
            'battery_capacity': '/sys/class/power_supply/BAT1/capacity',
            'ac_adapter': '/sys/class/power_supply/ADP1/online',
            'thermal_zone': '/sys/class/thermal/thermal_zone0/temp'
        }
        
        # Predefined power profiles optimized for Steam Deck
        self.power_profiles = {
            PowerProfile.BATTERY_SAVE: PowerSettings(
                cpu_governor='powersave',
                cpu_max_freq=1600,  # 1.6 GHz limit
                gpu_power_limit=8,   # 8W GPU limit
                tdp_limit=4,         # 4W TDP (minimum)
                fan_curve='quiet',
                memory_frequency=800,  # Lower memory clock
                shader_compile_threads=2,  # Minimal compilation
                background_priority=19  # Lowest priority
            ),
            PowerProfile.BALANCED: PowerSettings(
                cpu_governor='schedutil',
                cpu_max_freq=2400,  # 2.4 GHz balanced
                gpu_power_limit=12,  # 12W GPU
                tdp_limit=8,         # 8W TDP
                fan_curve='balanced',
                memory_frequency=1600,
                shader_compile_threads=3,
                background_priority=10
            ),
            PowerProfile.HANDHELD_GAMING: PowerSettings(
                cpu_governor='performance',
                cpu_max_freq=2800,  # 2.8 GHz for gaming
                gpu_power_limit=15,  # 15W GPU for 40-60 FPS target
                tdp_limit=12,        # 12W TDP for sustained gaming
                fan_curve='gaming',
                memory_frequency=1600,
                shader_compile_threads=4,
                background_priority=5
            ),
            PowerProfile.PERFORMANCE: PowerSettings(
                cpu_governor='performance',
                cpu_max_freq=3500,  # Full 3.5 GHz
                gpu_power_limit=18,  # Higher GPU power
                tdp_limit=15,        # 15W TDP (maximum)
                fan_curve='performance',
                memory_frequency=1600,
                shader_compile_threads=6,
                background_priority=0
            ),
            PowerProfile.DOCKED_GAMING: PowerSettings(
                cpu_governor='performance',
                cpu_max_freq=3500,  # Full performance when docked
                gpu_power_limit=22,  # Maximum GPU power
                tdp_limit=15,        # 15W TDP
                fan_curve='aggressive',
                memory_frequency=1600,
                shader_compile_threads=8,  # More threads when on AC
                background_priority=-5  # Higher than normal priority
            )
        }
        
        self.monitoring_active = False
        self.callbacks = []
        
    def detect_gaming_mode(self) -> bool:
        """Detect if Gaming Mode (gamescope) is active"""
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] == 'gamescope':
                    return True
        except:
            pass
        return False
    
    def detect_power_state(self) -> Dict[str, any]:
        """Detect current power state and conditions"""
        state = {
            'is_docked': False,
            'battery_level': 100,
            'thermal_state': 'normal',
            'cpu_temp': 0,
            'power_draw': 0,
            'gaming_mode': False
        }
        
        try:
            # Check AC adapter
            if Path(self.power_paths['ac_adapter']).exists():
                with open(self.power_paths['ac_adapter'], 'r') as f:
                    state['is_docked'] = f.read().strip() == '1'
            
            # Check battery level
            if Path(self.power_paths['battery_capacity']).exists():
                with open(self.power_paths['battery_capacity'], 'r') as f:
                    state['battery_level'] = int(f.read().strip())
            
            # Check thermal state
            if Path(self.power_paths['thermal_zone']).exists():
                with open(self.power_paths['thermal_zone'], 'r') as f:
                    temp_millicelsius = int(f.read().strip())
                    state['cpu_temp'] = temp_millicelsius / 1000
                    
                    if state['cpu_temp'] > 80:
                        state['thermal_state'] = 'hot'
                    elif state['cpu_temp'] > 70:
                        state['thermal_state'] = 'warm'
            
            # Detect gaming mode
            state['gaming_mode'] = self.detect_gaming_mode()
            
        except Exception as e:
            self.logger.warning(f"Could not detect power state: {e}")
            
        return state
    
    def select_optimal_profile(self, power_state: Dict) -> PowerProfile:
        """Select optimal power profile based on current state"""
        
        # Critical battery - force battery save mode
        if power_state['battery_level'] < 15 and not power_state['is_docked']:
            return PowerProfile.BATTERY_SAVE
        
        # Thermal throttling - reduce performance
        if power_state['thermal_state'] == 'hot':
            if power_state['is_docked']:
                return PowerProfile.BALANCED
            else:
                return PowerProfile.BATTERY_SAVE
        
        # Gaming mode active
        if power_state['gaming_mode']:
            if power_state['is_docked']:
                return PowerProfile.DOCKED_GAMING
            elif power_state['battery_level'] > 30:
                return PowerProfile.HANDHELD_GAMING
            else:
                return PowerProfile.BALANCED
        
        # Non-gaming scenarios
        if power_state['is_docked']:
            return PowerProfile.PERFORMANCE
        elif power_state['battery_level'] < 30:
            return PowerProfile.BATTERY_SAVE
        else:
            return PowerProfile.BALANCED
    
    def apply_power_profile(self, profile: PowerProfile) -> bool:
        """Apply power profile settings"""
        settings = self.power_profiles[profile]
        applied = []
        errors = []
        
        try:
            # Apply CPU governor
            if self._write_sysfs(self.power_paths['cpu_governor'], settings.cpu_governor):
                applied.append(f"CPU governor: {settings.cpu_governor}")
            else:
                errors.append("CPU governor")
            
            # Apply CPU frequency limit
            if settings.cpu_max_freq:
                freq_hz = settings.cpu_max_freq * 1000  # Convert MHz to Hz
                if self._write_sysfs(self.power_paths['cpu_max_freq'], str(freq_hz)):
                    applied.append(f"CPU max freq: {settings.cpu_max_freq} MHz")
                else:
                    errors.append("CPU frequency")
            
            # Apply GPU power settings via environment variables
            os.environ['RADV_FORCE_PERFORMANCE'] = str(settings.gpu_power_limit)
            applied.append(f"GPU power limit: {settings.gpu_power_limit}W")
            
            # Apply TDP limit (requires special handling)
            if self._apply_tdp_limit(settings.tdp_limit):
                applied.append(f"TDP limit: {settings.tdp_limit}W")
            else:
                errors.append("TDP limit")
            
            # Set process priority for background shader compilation
            try:
                os.nice(settings.background_priority)
                applied.append(f"Process priority: {settings.background_priority}")
            except:
                errors.append("Process priority")
            
            # Apply memory optimizations
            self._apply_memory_optimizations(settings)
            
            self.current_profile = profile
            self.logger.info(f"Applied power profile {profile.value}: {', '.join(applied)}")
            
            if errors:
                self.logger.warning(f"Failed to apply: {', '.join(errors)}")
            
            return len(errors) == 0
            
        except Exception as e:
            self.logger.error(f"Failed to apply power profile {profile.value}: {e}")
            return False
    
    def _write_sysfs(self, path: str, value: str) -> bool:
        """Write value to sysfs path with error handling"""
        try:
            if Path(path).exists():
                with open(path, 'w') as f:
                    f.write(value)
                return True
        except:
            pass
        return False
    
    def _apply_tdp_limit(self, tdp_watts: int) -> bool:
        """Apply TDP limit using Steam Deck specific methods"""
        try:
            # Try direct sysfs approach
            tdp_microwatts = tdp_watts * 1000000
            if self._write_sysfs(self.power_paths['tdp_control'], str(tdp_microwatts)):
                return True
            
            # Try PowerTools method if available
            try:
                subprocess.run(['powertools', 'set-tdp', str(tdp_watts)], 
                             check=True, capture_output=True)
                return True
            except:
                pass
            
            # Try manual ACPI control
            try:
                subprocess.run(['sudo', 'sh', '-c', 
                              f'echo {tdp_microwatts} > /sys/class/hwmon/hwmon0/power1_cap'],
                             check=True, capture_output=True)
                return True
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"Could not apply TDP limit: {e}")
            
        return False
    
    def _apply_memory_optimizations(self, settings: PowerSettings):
        """Apply memory-related optimizations"""
        try:
            # Set memory governor for power efficiency
            if settings.memory_frequency and settings.memory_frequency < 1600:
                os.environ['MESA_SHADER_CACHE_DISABLE'] = '0'  # Keep cache enabled
                
            # Adjust swap settings for battery life
            if settings.tdp_limit <= 8:  # Low power mode
                # Reduce swappiness to preserve battery
                subprocess.run(['sudo', 'sysctl', 'vm.swappiness=10'], 
                             check=False, capture_output=True)
            else:
                # Normal swappiness for performance
                subprocess.run(['sudo', 'sysctl', 'vm.swappiness=60'], 
                             check=False, capture_output=True)
                
        except Exception as e:
            self.logger.debug(f"Memory optimization warning: {e}")
    
    def start_adaptive_power_management(self, callback: Optional[Callable] = None):
        """Start adaptive power management based on system state"""
        self.monitoring_active = True
        
        if callback:
            self.callbacks.append(callback)
        
        def power_monitor_loop():
            last_profile = None
            
            while self.monitoring_active:
                try:
                    # Detect current state
                    power_state = self.detect_power_state()
                    
                    # Select optimal profile
                    optimal_profile = self.select_optimal_profile(power_state)
                    
                    # Apply profile if changed
                    if optimal_profile != last_profile:
                        if self.apply_power_profile(optimal_profile):
                            last_profile = optimal_profile
                            
                            # Notify callbacks
                            for cb in self.callbacks:
                                try:
                                    cb(optimal_profile, power_state)
                                except:
                                    pass
                    
                    # Update instance state
                    self.is_gaming_mode = power_state['gaming_mode']
                    self.is_docked = power_state['is_docked']
                    self.battery_level = power_state['battery_level']
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Power monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=power_monitor_loop, daemon=True)
        monitor_thread.start()
        
        self.logger.info("Started adaptive power management")
    
    def stop_adaptive_power_management(self):
        """Stop adaptive power management"""
        self.monitoring_active = False
        self.logger.info("Stopped adaptive power management")
    
    def get_shader_compilation_settings(self) -> Dict:
        """Get optimal shader compilation settings for current power profile"""
        settings = self.power_profiles[self.current_profile]
        
        return {
            'max_threads': settings.shader_compile_threads,
            'priority': settings.background_priority,
            'memory_limit_mb': 1024 if settings.tdp_limit <= 8 else 2048,
            'compile_in_background': settings.tdp_limit > 8,
            'aggressive_caching': settings.tdp_limit <= 8,  # Cache more aggressively on battery
            'thermal_throttling': True,
            'power_aware': True
        }
    
    def get_current_status(self) -> Dict:
        """Get current power management status"""
        power_state = self.detect_power_state()
        
        return {
            'current_profile': self.current_profile.value,
            'monitoring_active': self.monitoring_active,
            'power_state': power_state,
            'shader_settings': self.get_shader_compilation_settings(),
            'profile_details': {
                'tdp_limit': self.power_profiles[self.current_profile].tdp_limit,
                'cpu_governor': self.power_profiles[self.current_profile].cpu_governor,
                'gpu_power_limit': self.power_profiles[self.current_profile].gpu_power_limit
            }
        }
    
    def force_profile(self, profile: PowerProfile) -> bool:
        """Force a specific power profile (for testing/manual control)"""
        if self.apply_power_profile(profile):
            self.logger.info(f"Forced power profile to {profile.value}")
            return True
        return False

if __name__ == '__main__':
    # Example usage
    power_manager = GamingModePowerManager()
    
    def profile_change_callback(profile: PowerProfile, state: Dict):
        print(f"Power profile changed to {profile.value}")
        print(f"State: Gaming={state['gaming_mode']}, Docked={state['is_docked']}, "
              f"Battery={state['battery_level']}%")
    
    # Start adaptive management
    power_manager.start_adaptive_power_management(profile_change_callback)
    
    try:
        # Monitor for 60 seconds
        for i in range(12):
            status = power_manager.get_current_status()
            print(f"Status update {i+1}: {status['current_profile']}")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        power_manager.stop_adaptive_power_management()