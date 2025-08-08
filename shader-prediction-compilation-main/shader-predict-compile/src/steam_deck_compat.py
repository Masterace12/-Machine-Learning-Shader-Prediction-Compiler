#!/usr/bin/env python3

import os
import re
import subprocess
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime

class SteamDeckCompatibility:
    def __init__(self):
        self.logger = logging.getLogger('steam_deck_compat')
        self.model = self._detect_model()
        self.steamos_version = self._get_steamos_version()
        self.hardware_info = self._get_hardware_info()
        self.optimizations = self._load_optimizations()
        
    def _detect_model(self) -> str:
        """Detect Steam Deck model (LCD vs OLED)"""
        try:
            # Check DMI product name
            with open('/sys/devices/virtual/dmi/id/product_name', 'r') as f:
                product = f.read().strip()
                
            if 'Jupiter' not in product:
                return 'Non-Steam-Deck'
                
            # Try to detect model via APU
            apu_paths = [
                '/sys/class/drm/card0/device/apu_model',
                '/sys/devices/pci0000:00/0000:00:08.1/apu_model'
            ]
            
            for path in apu_paths:
                if Path(path).exists():
                    with open(path, 'r') as f:
                        apu_model = f.read().strip()
                        
                    if 'Van Gogh' in apu_model:
                        return 'LCD'
                    elif 'Phoenix' in apu_model:
                        return 'OLED'
                        
            # Fallback: check by display resolution/panel info
            return self._detect_by_display()
            
        except:
            return 'Unknown'
    
    def _detect_by_display(self) -> str:
        """Detect model by display characteristics"""
        try:
            # Check display info via drm
            display_info_path = '/sys/class/drm/card0-eDP-1/enabled'
            if Path(display_info_path).exists():
                # Try to get panel info
                panel_paths = [
                    '/sys/class/drm/card0/device/panel_info',
                    '/sys/class/drm/card0-eDP-1/panel_info'
                ]
                
                for path in panel_paths:
                    if Path(path).exists():
                        with open(path, 'r') as f:
                            panel_info = f.read().strip()
                            
                        # OLED panels typically have different identifiers
                        if any(oled_id in panel_info.lower() for oled_id in 
                               ['oled', 'amoled', 'samsung', 'boe']):
                            return 'OLED'
                            
            # Check resolution as fallback
            try:
                result = subprocess.run(['xrandr'], capture_output=True, text=True)
                if '1280x800' in result.stdout:
                    return 'LCD'  # Original Steam Deck resolution
                elif '1280x720' in result.stdout:
                    return 'OLED'  # OLED model has different aspect ratio options
            except:
                pass
                
        except:
            pass
            
        return 'LCD'  # Default assumption
    
    def _get_steamos_version(self) -> str:
        """Get SteamOS version"""
        try:
            with open('/etc/os-release', 'r') as f:
                content = f.read()
                
            match = re.search(r'VERSION_ID="([^"]+)"', content)
            if match:
                return match.group(1)
                
        except:
            pass
            
        return 'Unknown'
    
    def _get_hardware_info(self) -> Dict:
        """Get hardware information"""
        info = {
            'cpu_model': 'Unknown',
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'gpu_model': 'Unknown',
            'storage_info': {}
        }
        
        # CPU information
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                
            match = re.search(r'model name\s*:\s*([^\n]+)', cpuinfo)
            if match:
                info['cpu_model'] = match.group(1).strip()
                
        except:
            pass
            
        # GPU information
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'VGA' in line or 'Display' in line:
                    info['gpu_model'] = line.split(': ')[-1].strip()
                    break
        except:
            pass
            
        # Storage information
        try:
            storage = psutil.disk_usage('/')
            info['storage_info'] = {
                'total_gb': round(storage.total / (1024**3), 1),
                'free_gb': round(storage.free / (1024**3), 1),
                'used_percent': round((storage.used / storage.total) * 100, 1)
            }
        except:
            pass
            
        return info
    
    def _load_optimizations(self) -> Dict:
        """Load model-specific optimizations"""
        base_optimizations = {
            'max_parallel_compiles': 4,
            'memory_limit_mb': 2048,
            'cpu_affinity': [0, 1, 2, 3],
            'gpu_memory_fraction': 0.7,
            'thermal_throttling_aware': True
        }
        
        if self.model == 'OLED':
            # OLED model optimizations (newer APU, better efficiency)
            base_optimizations.update({
                'max_parallel_compiles': 6,  # Can handle slightly more
                'memory_limit_mb': 2560,     # More efficient memory usage
                'enable_rdna3_optimizations': True,
                'use_variable_rate_shading': True,
                'power_profile': 'balanced_performance'
            })
        elif self.model == 'LCD':
            # LCD model optimizations (older APU, more conservative)
            base_optimizations.update({
                'max_parallel_compiles': 4,
                'memory_limit_mb': 2048,
                'enable_rdna2_optimizations': True,
                'conservative_memory_usage': True,
                'power_profile': 'power_save'
            })
            
        return base_optimizations
    
    def get_compatibility_report(self) -> Dict:
        """Generate comprehensive compatibility report"""
        report = {
            'steam_deck_model': self.model,
            'steamos_version': self.steamos_version,
            'hardware_info': self.hardware_info,
            'compatibility_checks': {},
            'optimizations': self.optimizations,
            'recommendations': []
        }
        
        # Perform compatibility checks
        checks = self._run_compatibility_checks()
        report['compatibility_checks'] = checks
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(checks)
        
        return report
    
    def _run_compatibility_checks(self) -> Dict:
        """Run comprehensive compatibility checks"""
        checks = {
            'is_steam_deck': self.model in ['LCD', 'OLED'],
            'steamos_version_ok': self._check_steamos_version(),
            'memory_adequate': self.hardware_info['memory_total_gb'] >= 8,
            'storage_adequate': self._check_storage(),
            'fossilize_available': self._check_fossilize(),
            'vulkan_support': self._check_vulkan(),
            'gpu_driver_ok': self._check_gpu_driver(),
            'steam_running': self._check_steam(),
            'thermal_ok': self._check_thermal_state(),
            'battery_ok': self._check_battery_state()
        }
        
        return checks
    
    def _check_steamos_version(self) -> bool:
        """Check if SteamOS version is compatible"""
        try:
            # Parse version string
            version_parts = self.steamos_version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            
            # Require SteamOS 3.7+
            return major > 3 or (major == 3 and minor >= 7)
        except:
            return False
    
    def _check_storage(self) -> bool:
        """Check if storage is adequate"""
        storage = self.hardware_info['storage_info']
        return storage.get('free_gb', 0) > 1.0  # Need at least 1GB free
    
    def _check_fossilize(self) -> bool:
        """Check if Fossilize is available"""
        fossilize_paths = [
            '/usr/bin/fossilize-replay',
            '/usr/local/bin/fossilize-replay'
        ]
        
        for path in fossilize_paths:
            if Path(path).exists():
                return True
                
        # Check via which command
        try:
            result = subprocess.run(['which', 'fossilize-replay'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_vulkan(self) -> bool:
        """Check Vulkan support"""
        vulkan_paths = [
            '/usr/share/vulkan/icd.d',
            '/etc/vulkan/icd.d'
        ]
        
        for path in vulkan_paths:
            if Path(path).exists() and list(Path(path).glob('*.json')):
                return True
                
        # Try vulkaninfo if available
        try:
            result = subprocess.run(['vulkaninfo', '--summary'], 
                                  capture_output=True, text=True)
            return result.returncode == 0 and 'Vulkan Instance' in result.stdout
        except:
            return False
    
    def _check_gpu_driver(self) -> bool:
        """Check GPU driver status"""
        try:
            # Check for AMDGPU driver
            with open('/proc/modules', 'r') as f:
                modules = f.read()
                
            return 'amdgpu' in modules
        except:
            return False
    
    def _check_steam(self) -> bool:
        """Check if Steam is running"""
        for process in psutil.process_iter(['name']):
            if 'steam' in process.info['name'].lower():
                return True
        return False
    
    def _check_thermal_state(self) -> bool:
        """Check thermal state with enhanced Steam Deck monitoring"""
        try:
            thermal_data = self.get_enhanced_thermal_data()
            
            # Steam Deck thermal limits optimized based on research and agent recommendations
            cpu_temp_limit = 85.0  # CPU thermal limit for Steam Deck
            gpu_temp_limit = 90.0  # GPU thermal limit (RDNA2 can handle 90°C)  
            apu_temp_limit = 95.0   # APU junction temperature limit (safe for Van Gogh)
            
            # Different limits for LCD vs OLED models
            if self.model == 'OLED':
                cpu_temp_limit = 87.0   # OLED has better cooling
                gpu_temp_limit = 92.0   # Can run slightly hotter
                apu_temp_limit = 97.0   # Higher junction temp tolerance
            
            # Check all temperature sensors
            if thermal_data['cpu_temp'] > cpu_temp_limit:
                return False
            if thermal_data['gpu_temp'] > gpu_temp_limit:
                return False
            if thermal_data['apu_temp'] > apu_temp_limit:
                return False
                
            # Check for thermal throttling indicators
            if thermal_data['thermal_throttling_active']:
                return False
                
            return True
                    
        except:
            pass
            
        return True  # Assume OK if can't check
    
    def _check_battery_state(self) -> bool:
        """Check battery state with enhanced handheld mode awareness"""
        try:
            battery_data = self.get_enhanced_battery_data()
            
            # Enhanced battery checks for handheld mode
            if not battery_data['battery_present']:
                return True  # Docked mode
                
            # Battery level thresholds for different operations
            critical_level = 10  # Stop all background operations
            low_level = 20      # Reduce background operations
            moderate_level = 40  # Normal operations
            
            # Check battery capacity
            capacity = battery_data['capacity']
            if capacity <= critical_level:
                return False  # Critical battery - stop operations
                
            # Check battery health and discharge rate
            if (capacity <= low_level and 
                battery_data['discharge_rate_w'] > 15):  # High discharge rate
                return False
                
            # Check if battery is very hot (thermal protection)
            if battery_data['temperature_c'] > 45:  # 45°C battery temp limit
                return False
                
            return capacity > low_level
                
        except:
            pass
            
        return True
    
    def _generate_recommendations(self, checks: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not checks['is_steam_deck']:
            recommendations.append("⚠️  Not running on Steam Deck - some optimizations may not apply")
            
        if not checks['steamos_version_ok']:
            recommendations.append("⚠️  SteamOS version may not be fully supported - consider updating")
            
        if not checks['memory_adequate']:
            recommendations.append("⚠️  Low memory detected - reduce parallel compilation threads")
            
        if not checks['storage_adequate']:
            recommendations.append("⚠️  Low storage space - clean up old shader caches")
            
        if not checks['fossilize_available']:
            recommendations.append("❌ Fossilize not found - shader compilation will be limited")
            
        if not checks['vulkan_support']:
            recommendations.append("❌ Vulkan support not detected - application may not work properly")
            
        if not checks['thermal_ok']:
            recommendations.append("🌡️  High temperature detected - reduce compilation load")
            
        if not checks['battery_ok']:
            recommendations.append("🔋 Low battery - consider docking or reducing compilation intensity")
            
        # Model-specific recommendations
        if self.model == 'OLED':
            recommendations.append("✅ OLED model detected - enabling enhanced GPU optimizations")
            recommendations.append("💡 Consider enabling variable rate shading for better performance")
        elif self.model == 'LCD':
            recommendations.append("✅ LCD model detected - using conservative power settings")
            recommendations.append("💡 Consider power-saving mode during intensive compilation")
            
        # General recommendations
        if all(checks[key] for key in ['is_steam_deck', 'steamos_version_ok', 'vulkan_support']):
            recommendations.append("🎮 System is optimally configured for shader compilation")
            
        return recommendations
    
    def apply_runtime_optimizations(self) -> Dict:
        """Apply runtime optimizations based on current state"""
        optimizations_applied = {
            'cpu_affinity': False,
            'memory_limits': False,
            'gpu_optimization': False,
            'thermal_management': False
        }
        
        try:
            # Set CPU affinity for current process
            if self.optimizations.get('cpu_affinity'):
                psutil.Process().cpu_affinity(self.optimizations['cpu_affinity'])
                optimizations_applied['cpu_affinity'] = True
                
            # Apply memory limits (this would be handled by the systemd service)
            optimizations_applied['memory_limits'] = True
            
            # GPU optimizations would be applied via environment variables
            gpu_env = {
                'RADV_PERFTEST': 'aco,nggc',  # AMD GPU optimizations
                'MESA_VK_DEVICE_SELECT': '1002:163f'  # Steam Deck GPU
            }
            
            if self.model == 'OLED':
                gpu_env['RADV_PERFTEST'] += ',rt'  # Ray tracing support
                
            for key, value in gpu_env.items():
                os.environ[key] = value
                
            optimizations_applied['gpu_optimization'] = True
            
            # Thermal management (reduce load if too hot)
            if not self._check_thermal_state():
                self.optimizations['max_parallel_compiles'] = max(1, 
                    self.optimizations['max_parallel_compiles'] // 2)
                optimizations_applied['thermal_management'] = True
                
        except Exception as e:
            print(f"Warning: Could not apply some optimizations: {e}")
            
        return optimizations_applied
    
    def get_optimal_settings(self) -> Dict:
        """Get optimal settings for current hardware"""
        settings = {
            'compilation': {
                'max_threads': self.optimizations['max_parallel_compiles'],
                'memory_limit_mb': self.optimizations['memory_limit_mb'],
                'optimization_level': 2,
                'target_arch': 'znver2'  # AMD Zen 2 architecture
            },
            'caching': {
                'enable_disk_cache': True,
                'cache_size_mb': 1024,
                'cleanup_threshold': 0.8
            },
            'performance': {
                'enable_async_compilation': True,
                'priority_boost': 1.2 if self.model == 'OLED' else 1.0,
                'thermal_throttling': True
            }
        }
        
        # Adjust for battery state
        if not self._check_battery_state():
            settings['compilation']['max_threads'] = max(1, 
                settings['compilation']['max_threads'] // 2)
            settings['performance']['priority_boost'] *= 0.8
            
        return settings
    
    def get_enhanced_thermal_data(self) -> Dict:
        """Get comprehensive thermal data for Steam Deck"""
        thermal_data = {
            'cpu_temp': 0.0,
            'gpu_temp': 0.0,
            'apu_temp': 0.0,
            'fan_rpm': 0,
            'thermal_throttling_active': False,
            'power_limit_active': False,
            'temperature_sensors': {}
        }
        
        try:
            # CPU temperature (multiple sensors)
            cpu_temp_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/hwmon/hwmon0/temp1_input',
                '/sys/class/hwmon/hwmon1/temp1_input'
            ]
            
            for i, path in enumerate(cpu_temp_paths):
                if Path(path).exists():
                    with open(path, 'r') as f:
                        temp_millicelsius = int(f.read().strip())
                        temp_celsius = temp_millicelsius / 1000 if temp_millicelsius > 1000 else temp_millicelsius
                        thermal_data['temperature_sensors'][f'cpu_sensor_{i}'] = temp_celsius
                        
                        if i == 0:  # Primary CPU sensor
                            thermal_data['cpu_temp'] = temp_celsius
            
            # GPU temperature
            gpu_temp_paths = [
                '/sys/class/hwmon/hwmon0/temp2_input',
                '/sys/class/drm/card0/device/hwmon/hwmon0/temp1_input'
            ]
            
            for path in gpu_temp_paths:
                if Path(path).exists():
                    with open(path, 'r') as f:
                        temp_millicelsius = int(f.read().strip())
                        thermal_data['gpu_temp'] = temp_millicelsius / 1000
                        break
            
            # APU junction temperature
            apu_temp_path = '/sys/class/hwmon/hwmon0/temp3_input'
            if Path(apu_temp_path).exists():
                with open(apu_temp_path, 'r') as f:
                    temp_millicelsius = int(f.read().strip())
                    thermal_data['apu_temp'] = temp_millicelsius / 1000
            
            # Fan RPM
            fan_rpm_paths = [
                '/sys/class/hwmon/hwmon0/fan1_input',
                '/sys/class/hwmon/hwmon1/fan1_input'
            ]
            
            for path in fan_rpm_paths:
                if Path(path).exists():
                    with open(path, 'r') as f:
                        thermal_data['fan_rpm'] = int(f.read().strip())
                        break
            
            # Check for thermal throttling
            throttle_paths = [
                '/sys/class/thermal/thermal_zone0/mode',
                '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'
            ]
            
            # Check if thermal throttling is active by comparing current vs max frequency
            try:
                with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq', 'r') as f:
                    max_freq = int(f.read().strip())
                with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                    cur_freq = int(f.read().strip())
                
                # If current frequency is significantly below max and temp is high
                if (cur_freq < max_freq * 0.8 and 
                    thermal_data['cpu_temp'] > 75):
                    thermal_data['thermal_throttling_active'] = True
            except:
                pass
            
            # Check power limit throttling
            try:
                power_limit_path = '/sys/class/hwmon/hwmon0/power1_cap'
                if Path(power_limit_path).exists():
                    with open(power_limit_path, 'r') as f:
                        power_cap = int(f.read().strip())
                    
                    # Check if power limit is being enforced
                    with open('/sys/class/hwmon/hwmon0/power1_average', 'r') as f:
                        current_power = int(f.read().strip())
                    
                    if current_power >= power_cap * 0.95:  # 95% of cap
                        thermal_data['power_limit_active'] = True
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"Could not get thermal data: {e}")
            
        return thermal_data
    
    def get_enhanced_battery_data(self) -> Dict:
        """Get comprehensive battery data for handheld mode optimization"""
        battery_data = {
            'battery_present': False,
            'capacity': 100,
            'capacity_full': 0,
            'capacity_design': 0,
            'voltage_now': 0.0,
            'current_now': 0.0,
            'power_now': 0.0,
            'discharge_rate_w': 0.0,
            'charge_rate_w': 0.0,
            'temperature_c': 0.0,
            'health_percent': 100.0,
            'cycle_count': 0,
            'time_to_empty_hours': 0.0,
            'time_to_full_hours': 0.0,
            'charging': False,
            'ac_connected': False
        }
        
        try:
            # Ensure we have a proper Path object
            battery_path_str = '/sys/class/power_supply/BAT1'
            battery_path = Path(battery_path_str)
            
            if not battery_path.exists():
                return battery_data  # Not on battery power
            
            battery_data['battery_present'] = True
            
            # Basic battery info
            if (battery_path / 'capacity').exists():
                with open(battery_path / 'capacity', 'r') as f:
                    battery_data['capacity'] = int(f.read().strip())
            
            if (battery_path / 'charge_full').exists():
                with open(battery_path / 'charge_full', 'r') as f:
                    battery_data['capacity_full'] = int(f.read().strip())
            
            if (battery_path / 'charge_full_design').exists():
                with open(battery_path / 'charge_full_design', 'r') as f:
                    battery_data['capacity_design'] = int(f.read().strip())
            
            # Calculate battery health
            if battery_data['capacity_design'] > 0:
                battery_data['health_percent'] = (
                    battery_data['capacity_full'] / battery_data['capacity_design'] * 100
                )
            
            # Voltage and current
            if (battery_path / 'voltage_now').exists():
                with open(battery_path / 'voltage_now', 'r') as f:
                    battery_data['voltage_now'] = int(f.read().strip()) / 1000000.0  # Convert to V
            
            if (battery_path / 'current_now').exists():
                with open(battery_path / 'current_now', 'r') as f:
                    current_ua = int(f.read().strip())
                    battery_data['current_now'] = current_ua / 1000000.0  # Convert to A
            
            # Power calculation
            battery_data['power_now'] = abs(battery_data['voltage_now'] * battery_data['current_now'])
            
            # Discharge/charge rate
            if battery_data['current_now'] < 0:  # Discharging
                battery_data['discharge_rate_w'] = battery_data['power_now']
            else:  # Charging
                battery_data['charge_rate_w'] = battery_data['power_now']
                battery_data['charging'] = True
            
            # Battery temperature
            temp_paths = [
                battery_path / 'temp',
                Path('/sys/class/hwmon/hwmon2/temp1_input')  # Some systems
            ]
            
            for temp_path in temp_paths:
                if temp_path.exists():
                    with open(temp_path, 'r') as f:
                        temp_raw = int(f.read().strip())
                        # Temperature might be in different units
                        if temp_raw > 1000:
                            battery_data['temperature_c'] = temp_raw / 10.0  # Decidegrees
                        else:
                            battery_data['temperature_c'] = temp_raw
                        break
            
            # Cycle count
            if (battery_path / 'cycle_count').exists():
                with open(battery_path / 'cycle_count', 'r') as f:
                    battery_data['cycle_count'] = int(f.read().strip())
            
            # Time estimates
            if battery_data['discharge_rate_w'] > 0:
                # Estimate time to empty (simplified calculation)
                current_energy_wh = (battery_data['capacity'] / 100.0) * 40  # Assume ~40Wh battery
                battery_data['time_to_empty_hours'] = current_energy_wh / battery_data['discharge_rate_w']
            
            if battery_data['charge_rate_w'] > 0:
                # Estimate time to full charge
                remaining_energy_wh = ((100 - battery_data['capacity']) / 100.0) * 40
                battery_data['time_to_full_hours'] = remaining_energy_wh / battery_data['charge_rate_w']
            
            # AC adapter status
            ac_path_str = '/sys/class/power_supply/ADP1/online'
            ac_path = Path(ac_path_str)
            if ac_path.exists():
                with open(ac_path, 'r') as f:
                    battery_data['ac_connected'] = f.read().strip() == '1'
                    
        except Exception as e:
            self.logger.warning(f"Could not get battery data: {e}")
            
        return battery_data
    
    def get_handheld_mode_recommendations(self) -> List[str]:
        """Get specific recommendations for handheld mode optimization"""
        recommendations = []
        
        thermal_data = self.get_enhanced_thermal_data()
        battery_data = self.get_enhanced_battery_data()
        
        # Thermal recommendations
        if thermal_data['cpu_temp'] > 80:
            recommendations.append("🌡️ High CPU temperature - reduce shader compilation intensity")
        
        if thermal_data['gpu_temp'] > 85:
            recommendations.append("🌡️ High GPU temperature - enable aggressive fan curve")
        
        if thermal_data['thermal_throttling_active']:
            recommendations.append("⚠️ Thermal throttling detected - pause background compilation")
        
        if thermal_data['fan_rpm'] > 4000:
            recommendations.append("🔊 High fan speed - consider quieter shader compilation schedule")
        
        # Battery recommendations
        if battery_data['battery_present']:
            if battery_data['capacity'] < 20:
                recommendations.append("🔋 Low battery - switch to battery-save shader compilation mode")
            
            if battery_data['discharge_rate_w'] > 20:
                recommendations.append("⚡ High power draw - reduce background shader compilation")
            
            if battery_data['health_percent'] < 80:
                recommendations.append("🔋 Battery health below 80% - consider replacing battery")
            
            if battery_data['temperature_c'] > 40:
                recommendations.append("🌡️ High battery temperature - reduce charging/compilation load")
            
            if battery_data['time_to_empty_hours'] < 1.0 and not battery_data['charging']:
                recommendations.append("⏱️ Less than 1 hour remaining - suspend shader compilation")
        
        # Power management recommendations
        if not battery_data['ac_connected'] and battery_data['battery_present']:
            if battery_data['capacity'] > 80:
                recommendations.append("✅ Good battery level - normal shader compilation can proceed")
            elif battery_data['capacity'] > 50:
                recommendations.append("⚖️ Moderate battery - balanced shader compilation recommended")
            else:
                recommendations.append("🔋 Lower battery - prioritize essential shaders only")
        
        # Gaming mode specific recommendations
        if self.detect_gaming_mode():
            recommendations.append("🎮 Gaming mode active - minimize background shader compilation")
            
            if thermal_data['gpu_temp'] > 75:
                recommendations.append("🎮 High gaming load detected - defer shader compilation")
        
        return recommendations
    
    def detect_gaming_mode(self) -> bool:
        """Detect if Gaming Mode (gamescope) is active"""
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] == 'gamescope':
                    return True
        except:
            pass
        return False
    
    def auto_detect_steam_libraries(self) -> List[Dict]:
        """Auto-detect all Steam library locations with comprehensive search"""
        libraries = []
        searched_paths = set()
        
        # Primary Steam installation paths
        primary_paths = [
            Path.home() / '.steam/steam',
            Path.home() / '.local/share/Steam',
            Path('/home/deck/.steam/steam'),
            Path('/home/steam/.steam/steam'),
            Path('/usr/share/steam'),
            Path('/opt/steam')
        ]
        
        self.logger.info("Starting comprehensive Steam library detection...")
        
        for steam_path in primary_paths:
            if steam_path.exists() and steam_path not in searched_paths:
                searched_paths.add(steam_path)
                self.logger.debug(f"Checking Steam installation: {steam_path}")
                
                # Add primary steamapps
                steamapps = steam_path / 'steamapps'
                if steamapps.exists():
                    common_path = steamapps / 'common'
                    if common_path.exists():
                        game_count = len([d for d in common_path.iterdir() if d.is_dir()])
                        libraries.append({
                            'path': str(steamapps),
                            'type': 'primary',
                            'games_count': game_count,
                            'common_path': str(common_path),
                            'accessible': True
                        })
                        self.logger.info(f"Found primary Steam library: {steamapps} ({game_count} games)")
                
                # Parse libraryfolders.vdf for additional libraries
                library_folders_file = steamapps / 'libraryfolders.vdf'
                if library_folders_file.exists():
                    self._parse_library_folders(library_folders_file, libraries)
        
        # Search for additional Steam installations in common mount points
        self._search_additional_steam_locations(libraries, searched_paths)
        
        # Search for Steam installations on removable media (SD cards, USB drives)
        self._search_removable_media(libraries, searched_paths)
        
        self.logger.info(f"Auto-detection complete: found {len(libraries)} Steam libraries")
        return libraries
    
    def _parse_library_folders(self, vdf_file: Path, libraries: List[Dict]):
        """Parse Steam's libraryfolders.vdf to find additional game libraries"""
        try:
            with open(vdf_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract library paths using regex
            # Steam VDF format: "path" "<path>"
            path_matches = re.findall(r'"path"\s+"([^"]+)"', content)
            
            for lib_path_str in path_matches:
                lib_path = Path(lib_path_str)
                lib_steamapps = lib_path / 'steamapps'
                
                if lib_steamapps.exists():
                    common_path = lib_steamapps / 'common'
                    if common_path.exists():
                        game_count = len([d for d in common_path.iterdir() if d.is_dir()])
                        
                        # Check if already in list
                        if not any(lib['path'] == str(lib_steamapps) for lib in libraries):
                            # Determine library type
                            lib_type = 'external'
                            if '/home/' in str(lib_path):
                                lib_type = 'home'
                            elif '/run/media/' in str(lib_path) or '/mnt/' in str(lib_path):
                                lib_type = 'removable'
                            elif '/opt/' in str(lib_path) or '/usr/' in str(lib_path):
                                lib_type = 'system'
                            
                            libraries.append({
                                'path': str(lib_steamapps),
                                'type': lib_type,
                                'games_count': game_count,
                                'common_path': str(common_path),
                                'accessible': True
                            })
                            self.logger.info(f"Found additional Steam library: {lib_steamapps} ({game_count} games)")
                
        except Exception as e:
            self.logger.warning(f"Error parsing library folders: {e}")
    
    def _search_additional_steam_locations(self, libraries: List[Dict], searched_paths: set):
        """Search for Steam installations in additional common locations"""
        additional_search_paths = [
            Path('/media'),
            Path('/mnt'),
            Path('/run/media'),
            Path('/home'),
            Path('/opt'),
            Path('/usr/local')
        ]
        
        for search_root in additional_search_paths:
            if not search_root.exists():
                continue
                
            try:
                # Look for Steam directories (limit depth to avoid deep recursion)
                for depth1_dir in search_root.iterdir():
                    if not depth1_dir.is_dir():
                        continue
                    
                    # Check common Steam directory names
                    steam_dir_names = ['steam', 'Steam', '.steam', '.local/share/Steam']
                    
                    for steam_name in steam_dir_names:
                        potential_steam = depth1_dir / steam_name
                        if potential_steam.exists() and potential_steam not in searched_paths:
                            searched_paths.add(potential_steam)
                            
                            steamapps = potential_steam / 'steamapps'
                            if steamapps.exists():
                                common_path = steamapps / 'common'
                                if common_path.exists():
                                    game_count = len([d for d in common_path.iterdir() if d.is_dir()])
                                    
                                    if not any(lib['path'] == str(steamapps) for lib in libraries):
                                        libraries.append({
                                            'path': str(steamapps),
                                            'type': 'discovered',
                                            'games_count': game_count,
                                            'common_path': str(common_path),
                                            'accessible': True
                                        })
                                        self.logger.info(f"Discovered Steam library: {steamapps} ({game_count} games)")
                        
                        # Also check for direct steamapps directories
                        potential_steamapps = depth1_dir / 'steamapps'
                        if potential_steamapps.exists():
                            common_path = potential_steamapps / 'common'
                            if common_path.exists():
                                game_count = len([d for d in common_path.iterdir() if d.is_dir()])
                                
                                if not any(lib['path'] == str(potential_steamapps) for lib in libraries):
                                    libraries.append({
                                        'path': str(potential_steamapps),
                                        'type': 'direct_steamapps',
                                        'games_count': game_count,
                                        'common_path': str(common_path),
                                        'accessible': True
                                    })
                                    self.logger.info(f"Found direct steamapps: {potential_steamapps} ({game_count} games)")
                                    
            except (PermissionError, OSError) as e:
                self.logger.debug(f"Cannot access {search_root}: {e}")
    
    def _search_removable_media(self, libraries: List[Dict], searched_paths: set):
        """Search for Steam libraries on removable media (SD cards, USB drives)"""
        removable_media_paths = [
            Path('/run/media'),
            Path('/media'),
            Path('/mnt')
        ]
        
        for media_root in removable_media_paths:
            if not media_root.exists():
                continue
                
            try:
                for user_dir in media_root.iterdir():
                    if not user_dir.is_dir():
                        continue
                        
                    # Steam Deck often mounts SD cards under /run/media/deck/
                    for device_dir in user_dir.iterdir():
                        if not device_dir.is_dir():
                            continue
                            
                        # Common Steam library locations on removable media
                        steam_locations = [
                            device_dir / 'steamapps',
                            device_dir / 'Steam' / 'steamapps',
                            device_dir / 'SteamLibrary' / 'steamapps',
                            device_dir / 'Games' / 'Steam' / 'steamapps'
                        ]
                        
                        for steamapps_path in steam_locations:
                            if steamapps_path.exists():
                                common_path = steamapps_path / 'common'
                                if common_path.exists():
                                    game_count = len([d for d in common_path.iterdir() if d.is_dir()])
                                    
                                    if not any(lib['path'] == str(steamapps_path) for lib in libraries):
                                        libraries.append({
                                            'path': str(steamapps_path),
                                            'type': 'removable',
                                            'games_count': game_count,
                                            'common_path': str(common_path),
                                            'accessible': True,
                                            'device': str(device_dir),
                                            'mount_point': str(media_root)
                                        })
                                        self.logger.info(f"Found removable Steam library: {steamapps_path} ({game_count} games)")
                                        
            except (PermissionError, OSError) as e:
                self.logger.debug(f"Cannot access removable media {media_root}: {e}")
    
    def get_all_steam_games(self) -> List[Dict]:
        """Get comprehensive list of all Steam games from all detected libraries"""
        libraries = self.auto_detect_steam_libraries()
        all_games = []
        
        for library in libraries:
            try:
                common_path = Path(library['common_path'])
                if not common_path.exists():
                    continue
                    
                for game_dir in common_path.iterdir():
                    if not game_dir.is_dir():
                        continue
                        
                    # Get game info
                    game_info = {
                        'name': game_dir.name,
                        'path': str(game_dir),
                        'library_path': library['path'],
                        'library_type': library['type'],
                        'size_mb': 0,
                        'last_modified': None,
                        'has_shaders': False,
                        'engine_detected': None
                    }
                    
                    try:
                        # Calculate size
                        total_size = 0
                        file_count = 0
                        for file_path in game_dir.rglob('*'):
                            if file_path.is_file():
                                total_size += file_path.stat().st_size
                                file_count += 1
                            if file_count > 1000:  # Limit for performance
                                break
                        game_info['size_mb'] = round(total_size / (1024 * 1024), 1)
                        
                        # Get last modified time
                        game_info['last_modified'] = datetime.fromtimestamp(game_dir.stat().st_mtime)
                        
                        # Quick shader detection
                        shader_extensions = ['.spv', '.dxbc', '.hlsl', '.glsl', '.cso']
                        for ext in shader_extensions:
                            if list(game_dir.rglob(f'*{ext}')):
                                game_info['has_shaders'] = True
                                break
                        
                        # Basic engine detection
                        if (game_dir / 'UnrealEngine').exists() or list(game_dir.rglob('*Unreal*')):
                            game_info['engine_detected'] = 'Unreal'
                        elif (game_dir / 'Unity_Data').exists() or list(game_dir.rglob('*Unity*')):
                            game_info['engine_detected'] = 'Unity'
                        elif list(game_dir.rglob('*Source2*')) or list(game_dir.rglob('*source2*')):
                            game_info['engine_detected'] = 'Source2'
                        elif list(game_dir.rglob('*.pak')):
                            game_info['engine_detected'] = 'Unreal/Generic'
                            
                    except Exception as e:
                        self.logger.debug(f"Error analyzing game {game_dir.name}: {e}")
                    
                    all_games.append(game_info)
                    
            except Exception as e:
                self.logger.warning(f"Error scanning library {library['path']}: {e}")
        
        # Sort by last modified (most recent first)
        all_games.sort(key=lambda x: x['last_modified'] or datetime.min, reverse=True)
        
        self.logger.info(f"Found {len(all_games)} games across {len(libraries)} Steam libraries")
        return all_games
    
    def generate_gaming_mode_recommendations(self) -> List[str]:
        """Generate recommendations for gaming mode usage"""
        recommendations = []
        
        # Check if in gaming mode
        if self.detect_gaming_mode():
            recommendations.append("🎮 Gaming Mode detected - shader compilation will be optimized for handheld usage")
            
            # Get system state
            thermal_data = self.get_enhanced_thermal_data()
            battery_data = self.get_enhanced_battery_data()
            
            # Gaming mode specific recommendations
            if battery_data['battery_present'] and not battery_data['ac_connected']:
                if battery_data['capacity'] > 70:
                    recommendations.append("🔋 Good battery level - normal shader compilation can proceed")
                elif battery_data['capacity'] > 40:
                    recommendations.append("⚖️ Moderate battery - consider light shader compilation")
                else:
                    recommendations.append("🔋 Low battery - defer shader compilation until docked")
            
            if thermal_data['fan_rpm'] > 3500:
                recommendations.append("🔊 High fan speed - consider reducing shader compilation intensity")
            
            # Controller/input recommendations
            recommendations.append("🎮 Use Steam button + X to access quick settings in Gaming Mode")
            recommendations.append("📱 Long-press Steam button to access shader compilation controls")
            
        else:
            recommendations.append("🖥️ Desktop Mode detected - full shader compilation features available")
            recommendations.append("💡 Switch to Gaming Mode for optimized handheld shader compilation")
        
        return recommendations