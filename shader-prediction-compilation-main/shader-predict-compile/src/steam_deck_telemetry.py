#!/usr/bin/env python3
"""
Steam Deck Hardware Telemetry and Training Data Collection System
Collects real-time hardware performance data, shader compilation metrics,
and gameplay patterns for ML model training.
"""

import os
import time
import json
import threading
import logging
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import hashlib
import subprocess
from datetime import datetime

try:
    import pyudev
    HAS_PYUDEV = True
except ImportError:
    HAS_PYUDEV = False

@dataclass
class HardwareTelemetry:
    """Steam Deck hardware telemetry snapshot"""
    timestamp: float
    
    # CPU metrics
    cpu_usage_percent: float
    cpu_freq_mhz: float
    cpu_temp_c: float
    cpu_power_w: float
    
    # GPU metrics
    gpu_usage_percent: float
    gpu_freq_mhz: float
    gpu_temp_c: float
    gpu_power_w: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    
    # APU/System metrics
    apu_temp_c: float
    fan_rpm: int
    battery_level_percent: float
    battery_voltage_v: float
    battery_current_a: float
    battery_power_w: float
    battery_temp_c: float
    
    # Memory metrics
    ram_usage_mb: float
    ram_total_mb: float
    swap_usage_mb: float
    
    # Performance state
    tdp_limit_w: float
    thermal_throttled: bool
    power_throttled: bool
    
    # Steam Deck model
    model_type: str  # 'LCD' or 'OLED'
    
    # Current game context
    active_game_id: Optional[str]
    active_game_process: Optional[str]
    
@dataclass
class ShaderCompilationEvent:
    """Shader compilation event with telemetry"""
    timestamp: float
    shader_hash: str
    shader_path: str
    game_id: str
    compilation_start_time: float
    compilation_end_time: float
    compilation_success: bool
    error_message: Optional[str]
    compiler_flags: List[str]
    telemetry_snapshot: HardwareTelemetry

class SteamDeckTelemetryCollector:
    """Collects comprehensive telemetry data from Steam Deck hardware"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.collection_thread = None
        
        # Data storage
        self.telemetry_history = deque(maxlen=3600)  # 1 hour at 1Hz
        self.shader_events = deque(maxlen=1000)
        self.gameplay_sessions = defaultdict(list)
        
        # Hardware paths and interfaces
        self.hwmon_paths = self._discover_hardware_monitoring_paths()
        self.steam_deck_model = self._detect_steam_deck_model()
        
        # Callbacks for real-time processing
        self.telemetry_callbacks = []
        self.shader_event_callbacks = []
        
        self.logger = self._setup_logging()
        
        # Gaming context tracking
        self.current_game_context = {
            'game_id': None,
            'process_name': None,
            'start_time': None,
            'shader_usage': []
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for telemetry collector"""
        logger = logging.getLogger('SteamDeckTelemetry')
        logger.setLevel(logging.INFO)
        
        log_dir = Path.home() / ".steam-deck-shader-ml"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / 'telemetry.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _discover_hardware_monitoring_paths(self) -> Dict[str, str]:
        """Discover Steam Deck hardware monitoring paths"""
        hwmon_paths = {}
        
        try:
            # Standard hwmon paths for Steam Deck
            hwmon_base = Path('/sys/class/hwmon')
            
            if hwmon_base.exists():
                for hwmon_dir in hwmon_base.iterdir():
                    name_file = hwmon_dir / 'name'
                    if name_file.exists():
                        name = name_file.read_text().strip()
                        
                        # Map Steam Deck specific sensors
                        if 'k10temp' in name.lower():  # CPU temperature
                            hwmon_paths['cpu_temp'] = str(hwmon_dir)
                        elif 'amdgpu' in name.lower():  # GPU sensors
                            hwmon_paths['gpu'] = str(hwmon_dir)
                        elif 'jupiter' in name.lower():  # Steam Deck APU
                            hwmon_paths['apu'] = str(hwmon_dir)
            
            # Additional Steam Deck specific paths
            steam_deck_paths = {
                'battery': '/sys/class/power_supply/BAT0',
                'cpu_freq': '/sys/devices/system/cpu/cpu0/cpufreq',
                'fan': '/sys/class/thermal/cooling_device0',
                'tdp': '/sys/class/drm/card0/device'
            }
            
            for key, path in steam_deck_paths.items():
                if Path(path).exists():
                    hwmon_paths[key] = path
                    
        except Exception as e:
            self.logger.warning(f"Error discovering hardware paths: {e}")
        
        return hwmon_paths
    
    def _detect_steam_deck_model(self) -> str:
        """Detect Steam Deck model (LCD vs OLED)"""
        try:
            # Check DMI information for model detection
            dmi_product = Path('/sys/devices/virtual/dmi/id/product_name')
            if dmi_product.exists():
                product = dmi_product.read_text().strip()
                if 'oled' in product.lower():
                    return 'OLED'
            
            # Check display resolution as fallback
            try:
                result = subprocess.run(['xrandr'], capture_output=True, text=True)
                if '1280x800' in result.stdout:
                    return 'LCD'
                elif '1280x720' in result.stdout or '1920x1200' in result.stdout:
                    return 'OLED'  # OLED can do multiple resolutions
            except:
                pass
                
            # Default fallback
            return 'LCD'
            
        except Exception as e:
            self.logger.warning(f"Error detecting Steam Deck model: {e}")
            return 'LCD'
    
    def start_collection(self):
        """Start telemetry collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info("Started telemetry collection")
    
    def stop_collection(self):
        """Stop telemetry collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        self.logger.info("Stopped telemetry collection")
    
    def _collection_loop(self):
        """Main telemetry collection loop"""
        while self.is_collecting:
            try:
                # Collect hardware telemetry
                telemetry = self._collect_hardware_telemetry()
                if telemetry:
                    self.telemetry_history.append(telemetry)
                    
                    # Update game context
                    self._update_game_context(telemetry)
                    
                    # Call registered callbacks
                    for callback in self.telemetry_callbacks:
                        try:
                            callback(telemetry)
                        except Exception as e:
                            self.logger.error(f"Error in telemetry callback: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in telemetry collection loop: {e}")
                time.sleep(1.0)
    
    def _collect_hardware_telemetry(self) -> Optional[HardwareTelemetry]:
        """Collect comprehensive hardware telemetry"""
        try:
            timestamp = time.time()
            
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            cpu_freq = self._read_cpu_frequency()
            cpu_temp = self._read_temperature('cpu_temp', 'temp1_input')
            cpu_power = self._read_power('cpu', 'power1_input')
            
            # GPU metrics
            gpu_usage = self._read_gpu_usage()
            gpu_freq = self._read_gpu_frequency()
            gpu_temp = self._read_temperature('gpu', 'temp1_input')
            gpu_power = self._read_power('gpu', 'power1_input')
            gpu_memory = self._read_gpu_memory()
            
            # APU and system
            apu_temp = self._read_temperature('apu', 'temp1_input')
            fan_rpm = self._read_fan_speed()
            
            # Battery metrics
            battery_info = self._read_battery_info()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Performance state
            tdp_limit = self._read_tdp_limit()
            thermal_throttled = self._check_thermal_throttling(cpu_temp, gpu_temp, apu_temp)
            power_throttled = self._check_power_throttling(battery_info)
            
            # Game context
            active_game = self._get_active_game()
            
            telemetry = HardwareTelemetry(
                timestamp=timestamp,
                cpu_usage_percent=cpu_usage,
                cpu_freq_mhz=cpu_freq,
                cpu_temp_c=cpu_temp,
                cpu_power_w=cpu_power,
                gpu_usage_percent=gpu_usage,
                gpu_freq_mhz=gpu_freq,
                gpu_temp_c=gpu_temp,
                gpu_power_w=gpu_power,
                gpu_memory_used_mb=gpu_memory['used'],
                gpu_memory_total_mb=gpu_memory['total'],
                apu_temp_c=apu_temp,
                fan_rpm=fan_rpm,
                battery_level_percent=battery_info['level'],
                battery_voltage_v=battery_info['voltage'],
                battery_current_a=battery_info['current'],
                battery_power_w=battery_info['power'],
                battery_temp_c=battery_info['temperature'],
                ram_usage_mb=memory.used / (1024*1024),
                ram_total_mb=memory.total / (1024*1024),
                swap_usage_mb=swap.used / (1024*1024),
                tdp_limit_w=tdp_limit,
                thermal_throttled=thermal_throttled,
                power_throttled=power_throttled,
                model_type=self.steam_deck_model,
                active_game_id=active_game['id'],
                active_game_process=active_game['process']
            )
            
            return telemetry
            
        except Exception as e:
            self.logger.error(f"Error collecting hardware telemetry: {e}")
            return None
    
    def _read_cpu_frequency(self) -> float:
        """Read current CPU frequency"""
        try:
            freq_path = Path(self.hwmon_paths.get('cpu_freq', '')) / 'scaling_cur_freq'
            if freq_path.exists():
                freq_khz = float(freq_path.read_text().strip())
                return freq_khz / 1000.0  # Convert to MHz
        except:
            pass
        
        # Fallback to psutil
        try:
            return psutil.cpu_freq().current
        except:
            return 0.0
    
    def _read_temperature(self, sensor_type: str, filename: str) -> float:
        """Read temperature from hwmon sensor"""
        try:
            sensor_path = self.hwmon_paths.get(sensor_type, '')
            if sensor_path:
                temp_file = Path(sensor_path) / filename
                if temp_file.exists():
                    temp_millicelsius = float(temp_file.read_text().strip())
                    return temp_millicelsius / 1000.0
        except:
            pass
        
        return 0.0
    
    def _read_power(self, sensor_type: str, filename: str) -> float:
        """Read power consumption from hwmon sensor"""
        try:
            sensor_path = self.hwmon_paths.get(sensor_type, '')
            if sensor_path:
                power_file = Path(sensor_path) / filename
                if power_file.exists():
                    power_microwatts = float(power_file.read_text().strip())
                    return power_microwatts / 1000000.0  # Convert to watts
        except:
            pass
        
        return 0.0
    
    def _read_gpu_usage(self) -> float:
        """Read GPU usage percentage"""
        try:
            # Try AMD GPU usage
            gpu_path = self.hwmon_paths.get('gpu', '')
            if gpu_path:
                usage_file = Path(gpu_path) / 'device' / 'gpu_busy_percent'
                if usage_file.exists():
                    return float(usage_file.read_text().strip())
        except:
            pass
        
        return 0.0
    
    def _read_gpu_frequency(self) -> float:
        """Read GPU frequency"""
        try:
            gpu_path = self.hwmon_paths.get('gpu', '')
            if gpu_path:
                freq_file = Path(gpu_path) / 'device' / 'pp_dpm_sclk'
                if freq_file.exists():
                    freq_data = freq_file.read_text().strip()
                    # Parse current frequency (marked with *)
                    for line in freq_data.split('\n'):
                        if '*' in line:
                            freq_str = line.split()[1].rstrip('Mhz')
                            return float(freq_str)
        except:
            pass
        
        return 0.0
    
    def _read_gpu_memory(self) -> Dict[str, float]:
        """Read GPU memory usage"""
        try:
            gpu_path = self.hwmon_paths.get('gpu', '')
            if gpu_path:
                device_path = Path(gpu_path) / 'device'
                
                # Read VRAM usage
                vram_used_file = device_path / 'mem_info_vram_used'
                vram_total_file = device_path / 'mem_info_vram_total'
                
                if vram_used_file.exists() and vram_total_file.exists():
                    used_bytes = float(vram_used_file.read_text().strip())
                    total_bytes = float(vram_total_file.read_text().strip())
                    
                    return {
                        'used': used_bytes / (1024*1024),  # Convert to MB
                        'total': total_bytes / (1024*1024)
                    }
        except:
            pass
        
        return {'used': 0.0, 'total': 0.0}
    
    def _read_fan_speed(self) -> int:
        """Read fan RPM"""
        try:
            fan_path = self.hwmon_paths.get('fan', '')
            if fan_path:
                fan_file = Path(fan_path) / 'cur_state'
                if fan_file.exists():
                    # This gives fan level, convert to approximate RPM
                    fan_level = int(fan_file.read_text().strip())
                    # Steam Deck fan levels roughly correspond to RPM
                    return fan_level * 500  # Rough approximation
        except:
            pass
        
        return 0
    
    def _read_battery_info(self) -> Dict[str, float]:
        """Read comprehensive battery information"""
        battery_info = {
            'level': 0.0,
            'voltage': 0.0,
            'current': 0.0,
            'power': 0.0,
            'temperature': 0.0
        }
        
        try:
            battery_path = Path(self.hwmon_paths.get('battery', '/sys/class/power_supply/BAT0'))
            
            if battery_path.exists():
                # Battery level
                capacity_file = battery_path / 'capacity'
                if capacity_file.exists():
                    battery_info['level'] = float(capacity_file.read_text().strip())
                
                # Voltage
                voltage_file = battery_path / 'voltage_now'
                if voltage_file.exists():
                    voltage_uv = float(voltage_file.read_text().strip())
                    battery_info['voltage'] = voltage_uv / 1000000.0
                
                # Current
                current_file = battery_path / 'current_now'
                if current_file.exists():
                    current_ua = float(current_file.read_text().strip())
                    battery_info['current'] = current_ua / 1000000.0
                
                # Power
                power_file = battery_path / 'power_now'
                if power_file.exists():
                    power_uw = float(power_file.read_text().strip())
                    battery_info['power'] = power_uw / 1000000.0
                else:
                    # Calculate from voltage and current
                    battery_info['power'] = battery_info['voltage'] * battery_info['current']
                
                # Temperature
                temp_file = battery_path / 'temp'
                if temp_file.exists():
                    temp_tenth_celsius = float(temp_file.read_text().strip())
                    battery_info['temperature'] = temp_tenth_celsius / 10.0
                    
        except Exception as e:
            self.logger.warning(f"Error reading battery info: {e}")
        
        return battery_info
    
    def _read_tdp_limit(self) -> float:
        """Read current TDP limit"""
        try:
            tdp_path = self.hwmon_paths.get('tdp', '')
            if tdp_path:
                power_limit_file = Path(tdp_path) / 'power1_cap'
                if power_limit_file.exists():
                    power_limit_uw = float(power_limit_file.read_text().strip())
                    return power_limit_uw / 1000000.0  # Convert to watts
        except:
            pass
        
        return 15.0  # Default Steam Deck TDP
    
    def _check_thermal_throttling(self, cpu_temp: float, gpu_temp: float, apu_temp: float) -> bool:
        """Check if system is thermally throttling"""
        # Steam Deck thermal limits
        thermal_limits = {
            'LCD': {'cpu': 80.0, 'gpu': 85.0, 'apu': 90.0},
            'OLED': {'cpu': 85.0, 'gpu': 90.0, 'apu': 95.0}
        }
        
        limits = thermal_limits[self.steam_deck_model]
        return (cpu_temp > limits['cpu'] or 
                gpu_temp > limits['gpu'] or 
                apu_temp > limits['apu'])
    
    def _check_power_throttling(self, battery_info: Dict[str, float]) -> bool:
        """Check if system is power throttling"""
        # Power throttling occurs at low battery or high discharge rate
        return (battery_info['level'] < 20.0 or 
                battery_info['power'] > 20.0)  # > 20W discharge
    
    def _get_active_game(self) -> Dict[str, Optional[str]]:
        """Get currently active game information"""
        try:
            # Look for Steam processes first
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'reaper' in proc.info['name'].lower():  # Steam game process
                        cmdline = ' '.join(proc.info['cmdline'])
                        # Extract game ID from Steam command line
                        if 'steam://rungameid/' in cmdline:
                            game_id = cmdline.split('steam://rungameid/')[1].split()[0]
                            return {
                                'id': game_id,
                                'process': proc.info['name']
                            }
                    elif any(game_hint in proc.info['name'].lower() 
                           for game_hint in ['game', 'unity', 'unreal', 'engine']):
                        return {
                            'id': proc.info['name'],
                            'process': proc.info['name']
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error detecting active game: {e}")
        
        return {'id': None, 'process': None}
    
    def _update_game_context(self, telemetry: HardwareTelemetry):
        """Update current game context"""
        if telemetry.active_game_id != self.current_game_context['game_id']:
            # Game changed
            if self.current_game_context['game_id']:
                # End previous session
                self._end_gameplay_session()
            
            # Start new session
            if telemetry.active_game_id:
                self._start_gameplay_session(telemetry.active_game_id, telemetry.active_game_process)
    
    def _start_gameplay_session(self, game_id: str, process_name: str):
        """Start tracking a new gameplay session"""
        self.current_game_context = {
            'game_id': game_id,
            'process_name': process_name,
            'start_time': time.time(),
            'shader_usage': []
        }
        
        self.logger.info(f"Started gameplay session for {game_id}")
    
    def _end_gameplay_session(self):
        """End current gameplay session"""
        if self.current_game_context['game_id']:
            session_data = {
                'game_id': self.current_game_context['game_id'],
                'duration': time.time() - self.current_game_context['start_time'],
                'shader_usage': self.current_game_context['shader_usage'],
                'telemetry_summary': self._summarize_session_telemetry()
            }
            
            self.gameplay_sessions[self.current_game_context['game_id']].append(session_data)
            self.logger.info(f"Ended gameplay session for {self.current_game_context['game_id']}")
        
        self.current_game_context = {
            'game_id': None,
            'process_name': None,
            'start_time': None,
            'shader_usage': []
        }
    
    def _summarize_session_telemetry(self) -> Dict:
        """Summarize telemetry for current session"""
        if not self.telemetry_history:
            return {}
        
        session_start = self.current_game_context['start_time']
        session_telemetry = [t for t in self.telemetry_history if t.timestamp >= session_start]
        
        if not session_telemetry:
            return {}
        
        return {
            'avg_cpu_temp': sum(t.cpu_temp_c for t in session_telemetry) / len(session_telemetry),
            'avg_gpu_temp': sum(t.gpu_temp_c for t in session_telemetry) / len(session_telemetry),
            'avg_battery_power': sum(t.battery_power_w for t in session_telemetry) / len(session_telemetry),
            'thermal_throttle_events': sum(1 for t in session_telemetry if t.thermal_throttled),
            'power_throttle_events': sum(1 for t in session_telemetry if t.power_throttled)
        }
    
    def record_shader_compilation(self, shader_hash: str, shader_path: str, 
                                 compilation_time: float, success: bool,
                                 error_message: str = None, compiler_flags: List[str] = None):
        """Record a shader compilation event with telemetry"""
        try:
            # Get current telemetry snapshot
            current_telemetry = self._collect_hardware_telemetry()
            if not current_telemetry:
                return
            
            # Create shader compilation event
            event = ShaderCompilationEvent(
                timestamp=time.time(),
                shader_hash=shader_hash,
                shader_path=shader_path,
                game_id=current_telemetry.active_game_id or 'unknown',
                compilation_start_time=time.time() - compilation_time,
                compilation_end_time=time.time(),
                compilation_success=success,
                error_message=error_message,
                compiler_flags=compiler_flags or [],
                telemetry_snapshot=current_telemetry
            )
            
            self.shader_events.append(event)
            
            # Update current game context
            if self.current_game_context['game_id']:
                self.current_game_context['shader_usage'].append({
                    'shader_hash': shader_hash,
                    'timestamp': time.time(),
                    'success': success
                })
            
            # Call shader event callbacks
            for callback in self.shader_event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in shader event callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error recording shader compilation: {e}")
    
    def add_telemetry_callback(self, callback: Callable[[HardwareTelemetry], None]):
        """Add callback for real-time telemetry processing"""
        self.telemetry_callbacks.append(callback)
    
    def add_shader_event_callback(self, callback: Callable[[ShaderCompilationEvent], None]):
        """Add callback for shader compilation events"""
        self.shader_event_callbacks.append(callback)
    
    def export_training_dataset(self, output_path: Path, 
                               include_telemetry: bool = True,
                               include_gameplay: bool = True):
        """Export collected data as ML training dataset"""
        try:
            dataset = {
                'version': '1.0',
                'collection_period': {
                    'start': min(t.timestamp for t in self.telemetry_history) if self.telemetry_history else 0,
                    'end': max(t.timestamp for t in self.telemetry_history) if self.telemetry_history else 0
                },
                'steam_deck_model': self.steam_deck_model,
                'hardware_config': self._get_hardware_config(),
                'shader_events': [],
                'telemetry_samples': [],
                'gameplay_sessions': dict(self.gameplay_sessions) if include_gameplay else {}
            }
            
            # Export shader compilation events
            for event in self.shader_events:
                dataset['shader_events'].append({
                    'shader_hash': event.shader_hash,
                    'shader_path': event.shader_path,
                    'game_id': event.game_id,
                    'compilation_time_ms': (event.compilation_end_time - event.compilation_start_time) * 1000,
                    'success': event.compilation_success,
                    'error_message': event.error_message,
                    'compiler_flags': event.compiler_flags,
                    'telemetry': asdict(event.telemetry_snapshot) if include_telemetry else None
                })
            
            # Export telemetry samples (subsample to reduce size)
            if include_telemetry:
                telemetry_sample_rate = max(1, len(self.telemetry_history) // 1000)  # Max 1000 samples
                for i, telemetry in enumerate(self.telemetry_history):
                    if i % telemetry_sample_rate == 0:
                        dataset['telemetry_samples'].append(asdict(telemetry))
            
            # Write dataset
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            self.logger.info(f"Exported training dataset to {output_path}")
            self.logger.info(f"Dataset contains {len(dataset['shader_events'])} shader events, "
                           f"{len(dataset['telemetry_samples'])} telemetry samples")
            
        except Exception as e:
            self.logger.error(f"Error exporting training dataset: {e}")
    
    def _get_hardware_config(self) -> Dict:
        """Get static hardware configuration"""
        try:
            return {
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'ram_gb': psutil.virtual_memory().total / (1024**3),
                'steam_deck_model': self.steam_deck_model,
                'hwmon_paths': self.hwmon_paths
            }
        except:
            return {}
    
    def get_recent_telemetry(self, seconds: int = 60) -> List[HardwareTelemetry]:
        """Get recent telemetry data"""
        cutoff_time = time.time() - seconds
        return [t for t in self.telemetry_history if t.timestamp > cutoff_time]
    
    def get_current_thermal_state(self) -> Dict:
        """Get current thermal state for ML predictions"""
        if not self.telemetry_history:
            return {}
        
        latest = self.telemetry_history[-1]
        return {
            'cpu_temp_c': latest.cpu_temp_c,
            'gpu_temp_c': latest.gpu_temp_c,
            'apu_temp_c': latest.apu_temp_c,
            'thermal_throttled': latest.thermal_throttled,
            'fan_rpm': latest.fan_rpm,
            'battery_level': latest.battery_level_percent,
            'power_throttled': latest.power_throttled
        }