"""
Steam Deck Integration Module
Provides hardware-specific optimizations and system integration
for shader prediction on Steam Deck's RDNA2 GPU
"""

import os
import time
import json
import subprocess
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import psutil
from collections import deque
import logging

# Import our shader prediction system
from shader_prediction_system import (
    SteamDeckShaderPredictor, 
    ThermalState, 
    ShaderMetrics,
    ShaderType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SteamDeckHardwareState:
    """Current Steam Deck hardware state"""
    gpu_temp: float
    cpu_temp: float
    gpu_power: float
    cpu_power: float
    memory_used_mb: float
    gpu_clock_mhz: int
    memory_clock_mhz: int
    fan_speed_rpm: int
    battery_level: float
    charging: bool
    tdp_limit: float
    timestamp: float


class SteamDeckHardwareMonitor:
    """
    Hardware monitoring specific to Steam Deck
    Uses AMD GPU drivers and Steam Deck specific sensors
    """
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.is_monitoring = False
        self.hardware_state = None
        self.state_history = deque(maxlen=300)  # 5 minutes at 1Hz
        self.callbacks = []
        
        # Steam Deck specific paths (these may vary based on kernel/drivers)
        self.sensor_paths = {
            'gpu_temp': '/sys/class/hwmon/hwmon0/temp1_input',
            'cpu_temp': '/sys/class/thermal/thermal_zone0/temp',
            'gpu_power': '/sys/class/hwmon/hwmon0/power1_input',
            'fan_speed': '/sys/class/hwmon/hwmon0/fan1_input'
        }
        
        # Check if we're running on Steam Deck
        self.is_steam_deck = self._detect_steam_deck()
        
    def _detect_steam_deck(self) -> bool:
        """Detect if running on actual Steam Deck hardware"""
        try:
            # Check for Steam Deck specific identifiers
            with open('/sys/class/dmi/id/product_name', 'r') as f:
                product_name = f.read().strip()
                
            if 'Jupiter' in product_name or 'Steam Deck' in product_name:
                return True
                
            # Check for AMD Van Gogh APU (Steam Deck's chip)
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            if 'AMD Custom APU' in result.stdout:
                return True
                
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
            
        logger.warning("Not running on Steam Deck hardware - using simulated values")
        return False
        
    def start_monitoring(self):
        """Start hardware monitoring thread"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Hardware monitoring started")
        
    def stop_monitoring(self):
        """Stop hardware monitoring"""
        self.is_monitoring = False
        logger.info("Hardware monitoring stopped")
        
    def add_callback(self, callback: Callable[[SteamDeckHardwareState], None]):
        """Add callback for hardware state updates"""
        self.callbacks.append(callback)
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                state = self._read_hardware_state()
                self.hardware_state = state
                self.state_history.append(state)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(state)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(self.update_interval)
                
    def _read_hardware_state(self) -> SteamDeckHardwareState:
        """Read current hardware state"""
        if self.is_steam_deck:
            return self._read_real_hardware()
        else:
            return self._simulate_hardware()
            
    def _read_real_hardware(self) -> SteamDeckHardwareState:
        """Read real Steam Deck hardware sensors"""
        try:
            # GPU temperature (millidegrees to Celsius)
            gpu_temp = self._read_sensor_file(self.sensor_paths['gpu_temp'], scale=0.001)
            
            # CPU temperature
            cpu_temp = self._read_sensor_file(self.sensor_paths['cpu_temp'], scale=0.001)
            
            # GPU power (microwatts to watts)
            gpu_power = self._read_sensor_file(self.sensor_paths['gpu_power'], scale=0.000001)
            
            # Fan speed
            fan_speed = self._read_sensor_file(self.sensor_paths['fan_speed'])
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            memory_used_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
            
            # Battery info
            battery = psutil.sensors_battery()
            battery_level = battery.percent if battery else 100.0
            charging = battery.power_plugged if battery else False
            
            # GPU clocks (try to read from AMD driver)
            gpu_clock_mhz = self._get_gpu_clock()
            memory_clock_mhz = self._get_memory_clock()
            
            # TDP limit (Steam Deck typically runs at 15W)
            tdp_limit = 15.0  # Can be adjusted via Steam settings
            
            # CPU power estimation (total - GPU)
            cpu_power = max(0, tdp_limit - gpu_power)
            
            return SteamDeckHardwareState(
                gpu_temp=gpu_temp,
                cpu_temp=cpu_temp,
                gpu_power=gpu_power,
                cpu_power=cpu_power,
                memory_used_mb=memory_used_mb,
                gpu_clock_mhz=gpu_clock_mhz,
                memory_clock_mhz=memory_clock_mhz,
                fan_speed_rpm=fan_speed,
                battery_level=battery_level,
                charging=charging,
                tdp_limit=tdp_limit,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error reading hardware: {e}")
            return self._simulate_hardware()
            
    def _simulate_hardware(self) -> SteamDeckHardwareState:
        """Simulate hardware state for testing"""
        # Simulate varying conditions
        base_time = time.time()
        
        # Temperature varies based on load simulation
        gpu_temp = 65.0 + 10.0 * (0.5 + 0.3 * (base_time % 60) / 60)
        cpu_temp = 60.0 + 15.0 * (0.4 + 0.4 * (base_time % 45) / 45)
        
        # Power varies with temperature
        gpu_power = 8.0 + 4.0 * (gpu_temp - 65.0) / 20.0
        cpu_power = 7.0 + 3.0 * (cpu_temp - 60.0) / 25.0
        
        # Memory simulation
        memory_used_mb = 8000 + 2000 * (0.3 + 0.5 * (base_time % 30) / 30)
        
        # Clock speeds
        gpu_clock_mhz = int(1600 + 200 * (gpu_power - 8.0) / 4.0)
        memory_clock_mhz = 1600
        
        # Fan speed based on temperature
        fan_speed_rpm = int(1000 + 2000 * max(0, (max(gpu_temp, cpu_temp) - 60) / 30))
        
        # Battery simulation
        battery_level = 75.0
        charging = False
        
        return SteamDeckHardwareState(
            gpu_temp=gpu_temp,
            cpu_temp=cpu_temp,
            gpu_power=gpu_power,
            cpu_power=cpu_power,
            memory_used_mb=memory_used_mb,
            gpu_clock_mhz=gpu_clock_mhz,
            memory_clock_mhz=memory_clock_mhz,
            fan_speed_rpm=fan_speed_rpm,
            battery_level=battery_level,
            charging=charging,
            tdp_limit=15.0,
            timestamp=time.time()
        )
        
    def _read_sensor_file(self, path: str, scale: float = 1.0, default: float = 0.0) -> float:
        """Read sensor value from file"""
        try:
            with open(path, 'r') as f:
                value = float(f.read().strip()) * scale
                return value
        except (FileNotFoundError, ValueError, PermissionError):
            return default
            
    def _get_gpu_clock(self) -> int:
        """Get current GPU clock speed"""
        try:
            # Try AMD driver interface
            with open('/sys/class/drm/card0/device/pp_dpm_sclk', 'r') as f:
                lines = f.readlines()
                
            # Find active clock (marked with *)
            for line in lines:
                if '*' in line:
                    clock_str = line.split(':')[1].strip().replace('*', '').replace('Mhz', '')
                    return int(clock_str)
                    
        except (FileNotFoundError, ValueError, IndexError):
            pass
            
        return 1600  # Default Steam Deck GPU clock
        
    def _get_memory_clock(self) -> int:
        """Get current memory clock speed"""
        try:
            with open('/sys/class/drm/card0/device/pp_dpm_mclk', 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if '*' in line:
                    clock_str = line.split(':')[1].strip().replace('*', '').replace('Mhz', '')
                    return int(clock_str)
                    
        except (FileNotFoundError, ValueError, IndexError):
            pass
            
        return 1600  # Default Steam Deck memory clock
        
    def get_thermal_headroom(self) -> float:
        """Calculate thermal headroom in degrees Celsius"""
        if not self.hardware_state:
            return 20.0
            
        max_temp = max(self.hardware_state.gpu_temp, self.hardware_state.cpu_temp)
        return max(0, 85.0 - max_temp)  # 85°C throttle point
        
    def get_power_headroom(self) -> float:
        """Calculate power headroom in watts"""
        if not self.hardware_state:
            return 5.0
            
        used_power = self.hardware_state.gpu_power + self.hardware_state.cpu_power
        return max(0, self.hardware_state.tdp_limit - used_power)


class SteamDeckGameIntegration:
    """
    Integration with Steam games and Proton for shader prediction
    """
    
    def __init__(self, predictor: SteamDeckShaderPredictor):
        self.predictor = predictor
        self.active_games = {}
        self.game_configs = {}
        self.shader_hooks = {}
        
        # Steam paths
        self.steam_path = self._find_steam_path()
        self.shader_cache_path = os.path.join(
            self.steam_path, 
            "steamapps", "shadercache"
        ) if self.steam_path else None
        
    def _find_steam_path(self) -> Optional[str]:
        """Find Steam installation path"""
        possible_paths = [
            os.path.expanduser("~/.steam/steam"),
            os.path.expanduser("~/.local/share/Steam"),
            "/home/deck/.steam/steam",  # Steam Deck default
            "/home/deck/.local/share/Steam"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        return None
        
    def register_game(self, app_id: str, game_name: str, config: Dict = None):
        """Register a game for shader prediction"""
        self.active_games[app_id] = {
            'name': game_name,
            'start_time': time.time(),
            'shader_count': 0,
            'total_compile_time': 0.0,
            'predictions_made': 0,
            'predictions_accurate': 0
        }
        
        self.game_configs[app_id] = config or {}
        logger.info(f"Registered game: {game_name} (ID: {app_id})")
        
    def process_shader_request(self, app_id: str, shader_data: Dict, 
                              hardware_state: SteamDeckHardwareState) -> Dict:
        """Process shader compilation request from game"""
        
        # Enhance shader data with game context
        enhanced_shader = shader_data.copy()
        enhanced_shader['game_id'] = app_id
        
        # Convert hardware state to GPU state format
        gpu_state = {
            'temperature': hardware_state.gpu_temp,
            'power': hardware_state.gpu_power,
            'memory_used': hardware_state.memory_used_mb,
            'clock_mhz': hardware_state.gpu_clock_mhz
        }
        
        # Get prediction from main system
        result = self.predictor.process_shader_compilation(
            enhanced_shader, 
            gpu_state
        )
        
        # Update game statistics
        if app_id in self.active_games:
            self.active_games[app_id]['predictions_made'] += 1
            
        # Add game-specific optimizations
        result = self._apply_game_optimizations(app_id, result, hardware_state)
        
        return result
        
    def _apply_game_optimizations(self, app_id: str, result: Dict, 
                                 hardware_state: SteamDeckHardwareState) -> Dict:
        """Apply game-specific optimizations"""
        
        config = self.game_configs.get(app_id, {})
        
        # Adjust predictions based on game profile
        if 'shader_complexity_modifier' in config:
            modifier = config['shader_complexity_modifier']
            result['predicted_compilation_time_ms'] *= modifier
            
        # Battery-aware adjustments
        if hardware_state.battery_level < 20 and not hardware_state.charging:
            # More aggressive scheduling when battery is low
            if result['predicted_compilation_time_ms'] > 100:
                result['schedule'] = "delayed_battery_save"
                result['can_compile_now'] = False
                
        # Temperature-aware game-specific adjustments
        if hardware_state.gpu_temp > 80:
            # Some games are more temperature sensitive
            if config.get('temperature_sensitive', False):
                result['can_compile_now'] = False
                result['schedule'] = "delayed_thermal"
                
        return result
        
    def monitor_shader_cache(self, app_id: str) -> Dict:
        """Monitor game's shader cache status"""
        if not self.shader_cache_path:
            return {'status': 'unavailable'}
            
        cache_path = os.path.join(self.shader_cache_path, app_id)
        if not os.path.exists(cache_path):
            return {'status': 'no_cache', 'path': cache_path}
            
        # Get cache statistics
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except OSError:
                    pass
                    
        return {
            'status': 'available',
            'path': cache_path,
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'last_modified': os.path.getmtime(cache_path)
        }
        
    def get_game_statistics(self, app_id: str) -> Dict:
        """Get statistics for a specific game"""
        if app_id not in self.active_games:
            return {}
            
        stats = self.active_games[app_id].copy()
        
        # Calculate accuracy
        if stats['predictions_made'] > 0:
            stats['prediction_accuracy'] = (
                stats['predictions_accurate'] / stats['predictions_made']
            )
        else:
            stats['prediction_accuracy'] = 0.0
            
        # Add cache info
        stats['cache_info'] = self.monitor_shader_cache(app_id)
        
        return stats


class SteamDeckOptimizedSystem:
    """
    Complete Steam Deck optimized shader prediction system
    """
    
    def __init__(self, config_path: str = "steamdeck_config.json"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.predictor = SteamDeckShaderPredictor(self.config.get('predictor', {}))
        self.hardware_monitor = SteamDeckHardwareMonitor(
            update_interval=self.config.get('monitor_interval', 1.0)
        )
        self.game_integration = SteamDeckGameIntegration(self.predictor)
        
        # Set up hardware monitoring callback
        self.hardware_monitor.add_callback(self._on_hardware_update)
        
        # Runtime state
        self.is_running = False
        self.current_app_id = None
        
        logger.info("Steam Deck optimized system initialized")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        default_config = {
            'predictor': {
                'model_type': 'ensemble',
                'cache_size': 2000,  # Larger cache for Steam Deck
                'max_temp': 83.0,    # Conservative for handheld
                'power_budget': 12.0, # Leave headroom
                'sequence_length': 75,
                'buffer_size': 15000,
                'auto_train_interval': 500,
                'min_training_samples': 200
            },
            'monitor_interval': 0.5,  # More frequent monitoring
            'thermal_protection': True,
            'battery_optimization': True,
            'performance_mode': 'balanced',  # balanced, performance, battery_save
            'game_profiles': {}
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Merge configurations
            config = default_config.copy()
            config.update(user_config)
            return config
            
        except FileNotFoundError:
            logger.info(f"Config file {config_path} not found, using defaults")
            return default_config
            
    def start(self):
        """Start the complete system"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start hardware monitoring
        self.hardware_monitor.start_monitoring()
        
        # Wait for initial hardware reading
        time.sleep(1.0)
        
        logger.info("Steam Deck shader prediction system started")
        
    def stop(self):
        """Stop the system"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop hardware monitoring
        self.hardware_monitor.stop_monitoring()
        
        # Save system state
        self.predictor.save_state("steamdeck_save")
        
        logger.info("Steam Deck shader prediction system stopped")
        
    def set_active_game(self, app_id: str, game_name: str):
        """Set currently active game"""
        self.current_app_id = app_id
        
        # Load game-specific profile if available
        game_config = self.config.get('game_profiles', {}).get(app_id, {})
        self.game_integration.register_game(app_id, game_name, game_config)
        
        # Adjust system settings based on game
        self._apply_game_profile(app_id, game_config)
        
        logger.info(f"Active game set: {game_name} (ID: {app_id})")
        
    def _apply_game_profile(self, app_id: str, config: Dict):
        """Apply game-specific system optimizations"""
        
        # Adjust thermal limits for specific games
        if 'thermal_limit' in config:
            self.predictor.scheduler.max_temp = config['thermal_limit']
            
        # Adjust power budget
        if 'power_budget' in config:
            self.predictor.scheduler.power_budget = config['power_budget']
            
        # Adjust cache size for shader-heavy games
        if 'cache_size' in config:
            self.predictor.predictor.cache_size = config['cache_size']
            
    def process_shader(self, shader_data: Dict) -> Dict:
        """Main shader processing entry point"""
        if not self.is_running or not self.hardware_monitor.hardware_state:
            return {'error': 'system_not_ready'}
            
        if not self.current_app_id:
            return {'error': 'no_active_game'}
            
        # Process through game integration
        result = self.game_integration.process_shader_request(
            self.current_app_id,
            shader_data,
            self.hardware_monitor.hardware_state
        )
        
        # Add system-specific information
        result['hardware_state'] = {
            'gpu_temp': self.hardware_monitor.hardware_state.gpu_temp,
            'thermal_headroom': self.hardware_monitor.get_thermal_headroom(),
            'power_headroom': self.hardware_monitor.get_power_headroom(),
            'battery_level': self.hardware_monitor.hardware_state.battery_level,
            'fan_speed': self.hardware_monitor.hardware_state.fan_speed_rpm
        }
        
        return result
        
    def record_compilation_result(self, shader_data: Dict, result: Dict):
        """Record compilation result for training"""
        enhanced_shader = shader_data.copy()
        enhanced_shader['game_id'] = self.current_app_id
        
        self.predictor.record_compilation_result(enhanced_shader, result)
        
        # Update game statistics
        if self.current_app_id in self.game_integration.active_games:
            game_stats = self.game_integration.active_games[self.current_app_id]
            game_stats['shader_count'] += 1
            game_stats['total_compile_time'] += result.get('time_ms', 0)
            
    def _on_hardware_update(self, hardware_state: SteamDeckHardwareState):
        """Callback for hardware state updates"""
        
        # Implement emergency thermal protection
        if (self.config.get('thermal_protection', True) and 
            hardware_state.gpu_temp > 88):
            
            logger.warning(f"Emergency thermal protection at {hardware_state.gpu_temp}°C")
            # Disable all shader compilation temporarily
            self.predictor.scheduler.compilation_slots[ThermalState.HOT] = 0
            
        # Battery optimization
        if (self.config.get('battery_optimization', True) and
            hardware_state.battery_level < 15 and not hardware_state.charging):
            
            logger.info("Battery optimization mode activated")
            # Reduce compilation frequency
            for state in ThermalState:
                if state != ThermalState.COOL:
                    slots = self.predictor.scheduler.compilation_slots[state]
                    self.predictor.scheduler.compilation_slots[state] = max(0, slots - 1)
                    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'running': self.is_running,
            'active_game': self.current_app_id,
            'hardware': {},
            'predictor': self.predictor.get_statistics(),
            'games': {}
        }
        
        # Hardware status
        if self.hardware_monitor.hardware_state:
            hw = self.hardware_monitor.hardware_state
            status['hardware'] = {
                'gpu_temp': hw.gpu_temp,
                'cpu_temp': hw.cpu_temp,
                'gpu_power': hw.gpu_power,
                'battery_level': hw.battery_level,
                'thermal_headroom': self.hardware_monitor.get_thermal_headroom(),
                'power_headroom': self.hardware_monitor.get_power_headroom(),
                'is_steam_deck': self.hardware_monitor.is_steam_deck
            }
            
        # Game statistics
        for app_id in self.game_integration.active_games:
            status['games'][app_id] = self.game_integration.get_game_statistics(app_id)
            
        return status
        
    def export_performance_data(self, filepath: str = "steamdeck_performance.json"):
        """Export comprehensive performance data"""
        data = {
            'system_status': self.get_system_status(),
            'hardware_history': list(self.hardware_monitor.state_history),
            'predictor_stats': self.predictor.get_statistics(),
            'config': self.config,
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Performance data exported to {filepath}")


def create_example_game_profiles() -> Dict:
    """Create example game profiles for common Steam Deck games"""
    return {
        # Cyberpunk 2077 - Heavy shader usage
        "1091500": {
            "name": "Cyberpunk 2077",
            "shader_complexity_modifier": 1.3,
            "thermal_limit": 80.0,
            "power_budget": 14.0,
            "cache_size": 3000,
            "temperature_sensitive": True,
            "priority_boost": 2
        },
        
        # Elden Ring - Medium shader usage
        "1245620": {
            "name": "ELDEN RING",
            "shader_complexity_modifier": 1.1,
            "thermal_limit": 83.0,
            "power_budget": 13.0,
            "cache_size": 2000,
            "temperature_sensitive": False,
            "priority_boost": 1
        },
        
        # Portal 2 - Light shader usage
        "620": {
            "name": "Portal 2",
            "shader_complexity_modifier": 0.8,
            "thermal_limit": 85.0,
            "power_budget": 10.0,
            "cache_size": 1000,
            "temperature_sensitive": False,
            "priority_boost": 0
        },
        
        # Baldur's Gate 3 - Heavy but optimized
        "1086940": {
            "name": "Baldur's Gate 3",
            "shader_complexity_modifier": 1.2,
            "thermal_limit": 82.0,
            "power_budget": 13.5,
            "cache_size": 2500,
            "temperature_sensitive": True,
            "priority_boost": 1
        }
    }


if __name__ == "__main__":
    # Example usage
    print("Steam Deck Integration System")
    
    # Create example configuration
    example_config = {
        'predictor': {
            'model_type': 'ensemble',
            'cache_size': 2000,
            'max_temp': 83.0,
            'power_budget': 12.0
        },
        'monitor_interval': 0.5,
        'thermal_protection': True,
        'battery_optimization': True,
        'performance_mode': 'balanced',
        'game_profiles': create_example_game_profiles()
    }
    
    # Save example config
    with open('steamdeck_config.json', 'w') as f:
        json.dump(example_config, f, indent=2)
        
    print("Example configuration created: steamdeck_config.json")
    
    # Create and start system
    system = SteamDeckOptimizedSystem()
    system.start()
    
    # Example game setup
    system.set_active_game("1091500", "Cyberpunk 2077")
    
    # Example shader processing
    example_shader = {
        'hash': 'cyberpunk_main_shader_001',
        'type': 'fragment',
        'bytecode_size': 4096,
        'instruction_count': 280,
        'register_pressure': 48,
        'texture_samples': 8,
        'branch_complexity': 5,
        'loop_depth': 3,
        'scene_id': 'night_city',
        'priority': 2,
        'variant_count': 5
    }
    
    # Process shader
    result = system.process_shader(example_shader)
    print(f"Shader processing result: {json.dumps(result, indent=2)}")
    
    # Get system status
    status = system.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")
    
    # Export data
    system.export_performance_data()
    
    # Stop system
    time.sleep(2)
    system.stop()