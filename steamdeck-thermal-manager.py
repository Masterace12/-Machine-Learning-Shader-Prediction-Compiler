#!/usr/bin/env python3
"""
Steam Deck Thermal and Battery Management System
Advanced thermal throttling and battery awareness for ML Shader Prediction Compiler

Features:
- Real-time thermal monitoring with predictive throttling
- Battery level awareness with adaptive performance scaling  
- Gaming mode detection and automatic resource adjustment
- Predictive thermal modeling to prevent overheating
- Integration with systemd cgroups for resource enforcement
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import signal

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

@dataclass
class ThermalState:
    """Current thermal state of the system"""
    cpu_temp: float
    gpu_temp: float
    skin_temp: float  # Estimated external temperature
    fan_speed: int
    thermal_throttled: bool
    timestamp: float

@dataclass
class BatteryState:
    """Current battery state"""
    level_percent: int
    charging: bool
    power_draw_watts: float
    time_remaining_minutes: Optional[int]
    health_percent: int
    timestamp: float

@dataclass 
class PerformanceProfile:
    """Performance profile for different operating conditions"""
    name: str
    cpu_limit_percent: int
    memory_limit_mb: int
    io_weight: int
    nice_level: int
    enabled_features: List[str]
    thermal_threshold: float
    battery_threshold: int

class SteamDeckThermalManager:
    """Advanced thermal and battery management for Steam Deck"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".config/shader-predict-ml/thermal-config.json"
        self.data_dir = Path.home() / ".local/share/shader-predict-ml/data"
        self.cache_dir = Path.home() / ".cache/shader-predict-ml/thermal"
        
        # Create directories
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.config = self._load_config()
        
        # Thermal monitoring state
        self.thermal_history = deque(maxlen=300)  # 5 minutes at 1Hz
        self.battery_history = deque(maxlen=120)  # 2 minutes at 1Hz
        self.current_profile = "balanced"
        self.gaming_mode = False
        self.thermal_emergency = False
        
        # Performance profiles
        self.profiles = {
            "performance": PerformanceProfile(
                name="performance",
                cpu_limit_percent=15,
                memory_limit_mb=600,
                io_weight=50,
                nice_level=10,
                enabled_features=["ml_prediction", "p2p_sharing", "training"],
                thermal_threshold=85.0,
                battery_threshold=30
            ),
            "balanced": PerformanceProfile(
                name="balanced", 
                cpu_limit_percent=8,
                memory_limit_mb=400,
                io_weight=25,
                nice_level=15,
                enabled_features=["ml_prediction", "p2p_sharing"],
                thermal_threshold=80.0,
                battery_threshold=20
            ),
            "power_save": PerformanceProfile(
                name="power_save",
                cpu_limit_percent=5,
                memory_limit_mb=200,
                io_weight=10,
                nice_level=19,
                enabled_features=["ml_prediction"],
                thermal_threshold=75.0,
                battery_threshold=15
            ),
            "gaming": PerformanceProfile(
                name="gaming",
                cpu_limit_percent=3,
                memory_limit_mb=150,
                io_weight=5,
                nice_level=19,
                enabled_features=[],
                thermal_threshold=75.0,
                battery_threshold=10
            ),
            "thermal_emergency": PerformanceProfile(
                name="thermal_emergency",
                cpu_limit_percent=2,
                memory_limit_mb=100,
                io_weight=1,
                nice_level=19,
                enabled_features=[],
                thermal_threshold=70.0,
                battery_threshold=5
            )
        }
        
        # Steam Deck hardware detection
        self.steam_deck_model = self._detect_steam_deck_model()
        self.thermal_zones = self._find_thermal_zones()
        self.battery_path = self._find_battery_path()
        
        # Monitoring thread control
        self.monitoring_active = False
        self.monitor_thread = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for thermal manager"""
        logger = logging.getLogger('SteamDeckThermalManager')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.cache_dir / 'thermal_manager.log'
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1024*1024, backupCount=2
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler for debug
        if os.getenv('DEBUG') == '1':
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_config(self) -> Dict:
        """Load thermal management configuration"""
        default_config = {
            "version": "3.0.0",
            "thermal": {
                "monitoring_interval": 1.0,
                "cpu_temp_critical": 90.0,
                "cpu_temp_warning": 80.0,
                "gpu_temp_critical": 85.0,
                "gpu_temp_warning": 75.0,
                "throttle_hysteresis": 5.0,
                "emergency_shutdown": 95.0
            },
            "battery": {
                "monitoring_interval": 5.0,
                "critical_level": 5,
                "warning_level": 15,
                "power_save_level": 25,
                "charging_detection": True
            },
            "gaming": {
                "detection_interval": 2.0,
                "steam_processes": ["steam", "steamwebhelper", "steamos-session"],
                "game_processes": ["reaper", "wine", "proton"],
                "gamescope_detection": True
            },
            "profiles": {
                "auto_switch": True,
                "manual_override": False,
                "transition_delay": 10.0
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
        
        # Save default config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save default config: {e}")
        
        return default_config
    
    def _detect_steam_deck_model(self) -> str:
        """Detect Steam Deck model (LCD vs OLED)"""
        try:
            with open('/sys/devices/virtual/dmi/id/product_name') as f:
                product_name = f.read().strip()
                
            if 'Jupiter' in product_name:
                return 'LCD'
            elif 'Galileo' in product_name:
                return 'OLED'
            else:
                return 'unknown'
        except FileNotFoundError:
            return 'unknown'
    
    def _find_thermal_zones(self) -> Dict[str, str]:
        """Find available thermal zones"""
        zones = {}
        thermal_dir = Path('/sys/class/thermal')
        
        if not thermal_dir.exists():
            return zones
        
        try:
            for zone_dir in thermal_dir.glob('thermal_zone*'):
                zone_name = zone_dir.name
                temp_file = zone_dir / 'temp'
                type_file = zone_dir / 'type'
                
                if temp_file.exists():
                    zone_type = 'unknown'
                    if type_file.exists():
                        try:
                            zone_type = type_file.read_text().strip()
                        except:
                            pass
                    
                    zones[zone_name] = {
                        'path': str(temp_file),
                        'type': zone_type
                    }
            
            self.logger.info(f"Found thermal zones: {list(zones.keys())}")
            return zones
        except Exception as e:
            self.logger.error(f"Error finding thermal zones: {e}")
            return zones
    
    def _find_battery_path(self) -> Optional[str]:
        """Find battery information path"""
        battery_dir = Path('/sys/class/power_supply')
        
        if not battery_dir.exists():
            return None
        
        # Look for main battery (usually BAT1 on Steam Deck)
        for bat_dir in battery_dir.glob('BAT*'):
            capacity_file = bat_dir / 'capacity'
            if capacity_file.exists():
                self.logger.info(f"Found battery: {bat_dir}")
                return str(bat_dir)
        
        return None
    
    def get_thermal_state(self) -> ThermalState:
        """Get current thermal state"""
        cpu_temp = 0.0
        gpu_temp = 0.0
        
        # Read CPU temperature
        if 'thermal_zone0' in self.thermal_zones:
            try:
                with open(self.thermal_zones['thermal_zone0']['path']) as f:
                    cpu_temp = int(f.read().strip()) / 1000.0
            except Exception as e:
                self.logger.debug(f"Failed to read CPU temp: {e}")
        
        # Estimate GPU temperature (often thermal_zone1 or similar)
        if 'thermal_zone1' in self.thermal_zones:
            try:
                with open(self.thermal_zones['thermal_zone1']['path']) as f:
                    gpu_temp = int(f.read().strip()) / 1000.0
            except Exception as e:
                self.logger.debug(f"Failed to read GPU temp: {e}")
        
        # Estimate skin temperature (simplified)
        skin_temp = max(cpu_temp, gpu_temp) * 0.8  # Rough approximation
        
        # Get fan speed (if available)
        fan_speed = self._get_fan_speed()
        
        # Check if thermal throttling is active
        thermal_throttled = cpu_temp > self.config['thermal']['cpu_temp_warning']
        
        return ThermalState(
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            skin_temp=skin_temp,
            fan_speed=fan_speed,
            thermal_throttled=thermal_throttled,
            timestamp=time.time()
        )
    
    def _get_fan_speed(self) -> int:
        """Get current fan speed (RPM)"""
        fan_paths = [
            '/sys/class/hwmon/hwmon4/fan1_input',
            '/sys/class/hwmon/hwmon3/fan1_input',
            '/sys/class/hwmon/hwmon2/fan1_input'
        ]
        
        for path in fan_paths:
            try:
                with open(path) as f:
                    return int(f.read().strip())
            except:
                continue
        
        return 0
    
    def get_battery_state(self) -> BatteryState:
        """Get current battery state"""
        if not self.battery_path:
            return BatteryState(
                level_percent=100,
                charging=True,
                power_draw_watts=0.0,
                time_remaining_minutes=None,
                health_percent=100,
                timestamp=time.time()
            )
        
        try:
            battery_dir = Path(self.battery_path)
            
            # Battery level
            with open(battery_dir / 'capacity') as f:
                level = int(f.read().strip())
            
            # Charging status
            with open(battery_dir / 'status') as f:
                status = f.read().strip().lower()
                charging = status in ['charging', 'full']
            
            # Power draw (if available)
            power_draw = 0.0
            try:
                with open(battery_dir / 'power_now') as f:
                    power_draw = int(f.read().strip()) / 1000000.0  # Convert to watts
            except:
                pass
            
            # Battery health (if available)
            health = 100
            try:
                with open(battery_dir / 'health') as f:
                    health_str = f.read().strip()
                    if health_str.isdigit():
                        health = int(health_str)
            except:
                pass
            
            # Estimate time remaining
            time_remaining = None
            if not charging and power_draw > 0:
                # Very rough estimate
                time_remaining = int((level / 100.0) * (40.0 / power_draw) * 60)  # 40Wh typical capacity
            
            return BatteryState(
                level_percent=level,
                charging=charging,
                power_draw_watts=power_draw,
                time_remaining_minutes=time_remaining,
                health_percent=health,
                timestamp=time.time()
            )
        
        except Exception as e:
            self.logger.error(f"Failed to read battery state: {e}")
            return BatteryState(
                level_percent=50,
                charging=False,
                power_draw_watts=10.0,
                time_remaining_minutes=120,
                health_percent=100,
                timestamp=time.time()
            )
    
    def detect_gaming_mode(self) -> bool:
        """Detect if gaming mode is active"""
        try:
            # Check for gamescope (Steam Deck gaming mode)
            if subprocess.run(['pgrep', '-x', 'gamescope'], 
                            capture_output=True).returncode == 0:
                return True
            
            # Check for Steam processes
            if HAS_PSUTIL:
                for proc in psutil.process_iter(['name']):
                    try:
                        if proc.info['name'] in ['reaper', 'wine', 'proton']:
                            return True
                    except:
                        pass
            
            return False
        except Exception as e:
            self.logger.debug(f"Gaming mode detection failed: {e}")
            return False
    
    def select_performance_profile(self, thermal_state: ThermalState, 
                                 battery_state: BatteryState, 
                                 gaming_mode: bool) -> str:
        """Select appropriate performance profile based on system state"""
        
        # Emergency thermal protection
        if thermal_state.cpu_temp > self.config['thermal']['emergency_shutdown']:
            self.logger.critical(f"EMERGENCY: CPU temperature {thermal_state.cpu_temp}°C!")
            return "thermal_emergency"
        
        # Thermal throttling
        if (thermal_state.cpu_temp > self.config['thermal']['cpu_temp_critical'] or
            thermal_state.gpu_temp > self.config['thermal']['gpu_temp_critical']):
            return "thermal_emergency"
        
        # Gaming mode takes precedence
        if gaming_mode:
            return "gaming"
        
        # Battery-based selection
        if not battery_state.charging:
            if battery_state.level_percent <= self.config['battery']['critical_level']:
                return "thermal_emergency"  # Maximum power saving
            elif battery_state.level_percent <= self.config['battery']['warning_level']:
                return "power_save"
            elif battery_state.level_percent <= self.config['battery']['power_save_level']:
                return "power_save"
        
        # Thermal-based selection
        if thermal_state.cpu_temp > self.config['thermal']['cpu_temp_warning']:
            return "power_save"
        
        # Default balanced profile
        return "balanced"
    
    def apply_performance_profile(self, profile_name: str) -> bool:
        """Apply performance profile using systemd"""
        if profile_name not in self.profiles:
            self.logger.error(f"Unknown profile: {profile_name}")
            return False
        
        profile = self.profiles[profile_name]
        service_name = "shader-predict-ml.service"
        
        try:
            # Apply CPU quota
            subprocess.run([
                'systemctl', '--user', 'set-property', service_name,
                f'CPUQuota={profile.cpu_limit_percent}%'
            ], check=True, capture_output=True)
            
            # Apply memory limit  
            subprocess.run([
                'systemctl', '--user', 'set-property', service_name,
                f'MemoryMax={profile.memory_limit_mb}M'
            ], check=True, capture_output=True)
            
            # Apply I/O weight
            subprocess.run([
                'systemctl', '--user', 'set-property', service_name,
                f'IOWeight={profile.io_weight}'
            ], check=True, capture_output=True)
            
            self.logger.info(f"Applied profile '{profile_name}': "
                           f"CPU={profile.cpu_limit_percent}%, "
                           f"Memory={profile.memory_limit_mb}MB, "
                           f"IO={profile.io_weight}")
            
            return True
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to apply profile {profile_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error applying profile {profile_name}: {e}")
            return False
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("Starting thermal and battery monitoring")
        
        last_profile_change = 0
        profile_stable_time = self.config['profiles']['transition_delay']
        
        while self.monitoring_active:
            try:
                # Get current system state
                thermal_state = self.get_thermal_state()
                battery_state = self.get_battery_state()
                gaming_mode = self.detect_gaming_mode()
                
                # Store historical data
                self.thermal_history.append(thermal_state)
                self.battery_history.append(battery_state)
                
                # Select appropriate profile
                recommended_profile = self.select_performance_profile(
                    thermal_state, battery_state, gaming_mode
                )
                
                # Apply profile change if needed (with hysteresis)
                current_time = time.time()
                if (recommended_profile != self.current_profile and
                    current_time - last_profile_change > profile_stable_time):
                    
                    if self.apply_performance_profile(recommended_profile):
                        self.logger.info(f"Profile changed: {self.current_profile} -> {recommended_profile}")
                        self.current_profile = recommended_profile
                        last_profile_change = current_time
                
                # Update state
                self.gaming_mode = gaming_mode
                self.thermal_emergency = recommended_profile == "thermal_emergency"
                
                # Log periodic status
                if len(self.thermal_history) % 60 == 0:  # Every minute
                    self.logger.info(f"Status - Profile: {self.current_profile}, "
                                   f"CPU: {thermal_state.cpu_temp:.1f}°C, "
                                   f"Battery: {battery_state.level_percent}% "
                                   f"({'charging' if battery_state.charging else 'discharging'}), "
                                   f"Gaming: {gaming_mode}")
                
                # Save state periodically
                if len(self.thermal_history) % 300 == 0:  # Every 5 minutes
                    self._save_state()
                
                # Sleep until next monitoring cycle
                time.sleep(self.config['thermal']['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Longer delay on error
    
    def _save_state(self):
        """Save current state and history"""
        try:
            state_file = self.cache_dir / 'current_state.json'
            history_file = self.cache_dir / 'thermal_history.json'
            
            # Current state
            current_state = {
                'timestamp': time.time(),
                'current_profile': self.current_profile,
                'gaming_mode': self.gaming_mode,
                'thermal_emergency': self.thermal_emergency,
                'steam_deck_model': self.steam_deck_model
            }
            
            with open(state_file, 'w') as f:
                json.dump(current_state, f, indent=2)
            
            # Historical data (last hour)
            recent_thermal = list(self.thermal_history)[-60:]  # Last 60 samples
            recent_battery = list(self.battery_history)[-60:]
            
            history_data = {
                'thermal_history': [asdict(t) for t in recent_thermal],
                'battery_history': [asdict(b) for b in recent_battery]
            }
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Thermal monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self._save_state()
        self.logger.info("Thermal monitoring stopped")
    
    def get_status(self) -> Dict:
        """Get current status for external queries"""
        thermal_state = self.get_thermal_state()
        battery_state = self.get_battery_state()
        
        return {
            'thermal': asdict(thermal_state),
            'battery': asdict(battery_state),
            'current_profile': self.current_profile,
            'gaming_mode': self.gaming_mode,
            'thermal_emergency': self.thermal_emergency,
            'steam_deck_model': self.steam_deck_model,
            'monitoring_active': self.monitoring_active
        }

def main():
    """Main entry point for thermal manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Steam Deck Thermal and Battery Manager')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--profile', choices=['performance', 'balanced', 'power_save', 'gaming'], 
                       help='Set performance profile manually')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        os.environ['DEBUG'] = '1'
    
    manager = SteamDeckThermalManager()
    
    def signal_handler(signum, frame):
        manager.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.status:
        status = manager.get_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.profile:
        if manager.apply_performance_profile(args.profile):
            print(f"Applied profile: {args.profile}")
        else:
            print(f"Failed to apply profile: {args.profile}")
            sys.exit(1)
        return
    
    if args.daemon:
        manager.start_monitoring()
        try:
            while manager.monitoring_active:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            manager.stop_monitoring()
    else:
        # One-time monitoring
        status = manager.get_status()
        print(json.dumps(status, indent=2))

if __name__ == '__main__':
    main()