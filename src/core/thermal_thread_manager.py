#!/usr/bin/env python3
"""
Thermal-Aware Thread Management System for Steam Deck
===================================================

Provides real-time thermal monitoring and thread management that adapts to
Steam Deck's thermal conditions and hardware constraints.

Steam Deck Thermal Management:
- AMD Van Gogh APU thermal monitoring
- Dynamic thread scaling based on temperature
- Emergency thermal protection
- Battery and AC power awareness
- Gaming mode thermal optimization

Usage:
    thermal_manager = ThermalThreadManager()
    await thermal_manager.start_monitoring()
    thread_limit = thermal_manager.get_safe_thread_count()
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json

# Early threading setup MUST be done before any other imports
if 'setup_threading' not in sys.modules:
    import setup_threading
    setup_threading.configure_for_steam_deck()

logger = logging.getLogger(__name__)

class ThermalState(Enum):
    """Thermal states for thread management."""
    UNKNOWN = "unknown"
    COLD = "cold"        # < 50°C
    NORMAL = "normal"    # 50-65°C
    WARM = "warm"        # 65-75°C  
    HOT = "hot"          # 75-85°C
    CRITICAL = "critical" # > 85°C
    EMERGENCY = "emergency" # > 95°C

class PowerState(Enum):
    """Power states for thread management.""" 
    UNKNOWN = "unknown"
    BATTERY = "battery"
    AC_POWER = "ac_power"
    LOW_BATTERY = "low_battery"  # < 20%
    CRITICAL_BATTERY = "critical_battery"  # < 10%

@dataclass
class ThermalSensor:
    """Thermal sensor information."""
    name: str
    path: str
    temperature: float = 0.0
    max_temp: float = 100.0
    critical_temp: float = 105.0
    last_read: float = 0.0
    read_errors: int = 0

@dataclass 
class PowerInfo:
    """Power supply information."""
    state: PowerState = PowerState.UNKNOWN
    battery_level: float = 100.0
    is_charging: bool = False
    power_draw: float = 0.0  # Watts
    time_remaining: float = 0.0  # Minutes
    ac_connected: bool = False

@dataclass
class ThermalPolicy:
    """Thermal management policy."""
    max_threads: int
    cpu_limit_percent: Optional[int] = None
    memory_limit_mb: Optional[int] = None
    background_tasks: bool = True
    ml_inference: bool = True
    compilation: bool = True
    description: str = ""

class ThermalThreadManager:
    """Thermal-aware thread management for Steam Deck."""
    
    # Steam Deck thermal sensor paths
    THERMAL_ZONES = [
        '/sys/class/thermal/thermal_zone0/temp',  # CPU
        '/sys/class/thermal/thermal_zone1/temp',  # GPU  
        '/sys/class/thermal/thermal_zone2/temp',  # APU
        '/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0C0A:00/power_supply/BAT1/temp'  # Battery
    ]
    
    # Power supply paths
    POWER_SUPPLY_PATHS = {
        'battery': '/sys/class/power_supply/BAT1',
        'ac': '/sys/class/power_supply/ADP1'
    }
    
    # Thermal policies for different states
    THERMAL_POLICIES = {
        ThermalState.COLD: ThermalPolicy(
            max_threads=4,
            background_tasks=True,
            ml_inference=True,
            compilation=True,
            description="Cold - full performance"
        ),
        ThermalState.NORMAL: ThermalPolicy(
            max_threads=4,
            background_tasks=True,
            ml_inference=True, 
            compilation=True,
            description="Normal - full performance"
        ),
        ThermalState.WARM: ThermalPolicy(
            max_threads=2,
            background_tasks=True,
            ml_inference=True,
            compilation=True,
            description="Warm - moderate performance"
        ),
        ThermalState.HOT: ThermalPolicy(
            max_threads=1,
            background_tasks=False,
            ml_inference=True,
            compilation=False,
            description="Hot - reduced performance"
        ),
        ThermalState.CRITICAL: ThermalPolicy(
            max_threads=1,
            background_tasks=False,
            ml_inference=False,
            compilation=False,
            description="Critical - minimal activity"
        ),
        ThermalState.EMERGENCY: ThermalPolicy(
            max_threads=1,
            background_tasks=False,
            ml_inference=False,
            compilation=False,
            description="Emergency - shutdown imminent"
        )
    }
    
    # Power-based policies
    POWER_POLICIES = {
        PowerState.AC_POWER: {'thread_multiplier': 1.0, 'aggressive_cooling': False},
        PowerState.BATTERY: {'thread_multiplier': 0.75, 'aggressive_cooling': True},
        PowerState.LOW_BATTERY: {'thread_multiplier': 0.5, 'aggressive_cooling': True}, 
        PowerState.CRITICAL_BATTERY: {'thread_multiplier': 0.25, 'aggressive_cooling': True}
    }
    
    def __init__(self):
        self.is_steam_deck = setup_threading.is_steam_deck()
        self.sensors: List[ThermalSensor] = []
        self.power_info = PowerInfo()
        self.current_thermal_state = ThermalState.UNKNOWN
        self.current_power_state = PowerState.UNKNOWN
        
        # Monitoring state
        self.monitoring_active = False
        self.update_interval = 2.0  # seconds
        self.last_update = 0.0
        
        # Callbacks
        self.thermal_callbacks: List[Callable[[ThermalState], None]] = []
        self.power_callbacks: List[Callable[[PowerState], None]] = []
        self.emergency_callbacks: List[Callable[[], None]] = []
        
        # Thread management
        self.current_thread_limit = 4
        self.emergency_shutdown_requested = False
        
        # Statistics
        self.thermal_events = {state: 0 for state in ThermalState}
        self.throttle_events = 0
        self.emergency_shutdowns = 0
        
        logger.info(f"Thermal thread manager initialized - Steam Deck: {self.is_steam_deck}")
        
        # Initialize sensors
        self._discover_thermal_sensors()
        self._discover_power_supplies()
    
    def _discover_thermal_sensors(self) -> None:
        """Discover available thermal sensors."""
        self.sensors = []
        
        # Check predefined thermal zones
        for i, zone_path in enumerate(self.THERMAL_ZONES):
            zone_file = Path(zone_path)
            if zone_file.exists():
                try:
                    # Test read to ensure sensor works
                    temp_data = zone_file.read_text().strip()
                    temp_celsius = int(temp_data) / 1000.0
                    
                    if 0 <= temp_celsius <= 150:  # Reasonable range
                        sensor_name = f"thermal_zone_{i}"
                        if 'BAT1' in zone_path:
                            sensor_name = "battery"
                        elif i == 0:
                            sensor_name = "cpu"
                        elif i == 1:
                            sensor_name = "gpu"
                        elif i == 2:
                            sensor_name = "apu"
                        
                        sensor = ThermalSensor(
                            name=sensor_name,
                            path=zone_path,
                            temperature=temp_celsius
                        )
                        self.sensors.append(sensor)
                        logger.info(f"Discovered thermal sensor: {sensor_name} ({temp_celsius:.1f}°C)")
                        
                except (IOError, ValueError) as e:
                    logger.debug(f"Thermal sensor {zone_path} not usable: {e}")
        
        # Auto-discover additional thermal zones
        thermal_dir = Path('/sys/class/thermal')
        if thermal_dir.exists():
            for zone_dir in thermal_dir.glob('thermal_zone*/'):
                zone_temp_file = zone_dir / 'temp'
                zone_name_file = zone_dir / 'type'
                
                if zone_temp_file.exists() and str(zone_temp_file) not in [s.path for s in self.sensors]:
                    try:
                        temp_data = zone_temp_file.read_text().strip()
                        temp_celsius = int(temp_data) / 1000.0
                        
                        # Get sensor name from type file
                        sensor_name = zone_dir.name
                        if zone_name_file.exists():
                            try:
                                sensor_name = zone_name_file.read_text().strip()
                            except IOError:
                                pass
                        
                        if 0 <= temp_celsius <= 150:
                            sensor = ThermalSensor(
                                name=sensor_name,
                                path=str(zone_temp_file),
                                temperature=temp_celsius
                            )
                            self.sensors.append(sensor)
                            logger.debug(f"Auto-discovered thermal sensor: {sensor_name}")
                            
                    except (IOError, ValueError):
                        continue
        
        logger.info(f"Total thermal sensors discovered: {len(self.sensors)}")
    
    def _discover_power_supplies(self) -> None:
        """Discover power supply information."""
        try:
            battery_dir = Path(self.POWER_SUPPLY_PATHS['battery'])
            if battery_dir.exists():
                logger.info(f"Battery power supply found: {battery_dir}")
            
            ac_dir = Path(self.POWER_SUPPLY_PATHS.get('ac', '/sys/class/power_supply/ADP1'))
            if ac_dir.exists():
                logger.info(f"AC power supply found: {ac_dir}")
            else:
                # Search for AC adapter
                power_dir = Path('/sys/class/power_supply')
                for supply_dir in power_dir.glob('A*'):
                    type_file = supply_dir / 'type'
                    if type_file.exists():
                        try:
                            supply_type = type_file.read_text().strip()
                            if supply_type == 'Mains':
                                self.POWER_SUPPLY_PATHS['ac'] = str(supply_dir)
                                logger.info(f"AC adapter found: {supply_dir}")
                                break
                        except IOError:
                            continue
                            
        except Exception as e:
            logger.warning(f"Power supply discovery failed: {e}")
    
    def _read_thermal_sensors(self) -> Dict[str, float]:
        """Read current temperatures from all sensors."""
        temperatures = {}
        current_time = time.time()
        
        for sensor in self.sensors:
            try:
                sensor_path = Path(sensor.path)
                if not sensor_path.exists():
                    sensor.read_errors += 1
                    continue
                
                temp_data = sensor_path.read_text().strip()
                temp_celsius = int(temp_data) / 1000.0
                
                # Validate temperature reading
                if 0 <= temp_celsius <= 150:
                    sensor.temperature = temp_celsius
                    sensor.last_read = current_time
                    temperatures[sensor.name] = temp_celsius
                    sensor.read_errors = 0
                else:
                    sensor.read_errors += 1
                    logger.warning(f"Invalid temperature reading from {sensor.name}: {temp_celsius}°C")
                    
            except (IOError, ValueError) as e:
                sensor.read_errors += 1
                if sensor.read_errors <= 3:  # Only log first few errors
                    logger.warning(f"Failed to read thermal sensor {sensor.name}: {e}")
        
        return temperatures
    
    def _read_power_info(self) -> PowerInfo:
        """Read current power supply information."""
        power_info = PowerInfo()
        
        try:
            # Read battery information
            battery_dir = Path(self.POWER_SUPPLY_PATHS['battery'])
            if battery_dir.exists():
                try:
                    # Battery level
                    capacity_file = battery_dir / 'capacity'
                    if capacity_file.exists():
                        power_info.battery_level = float(capacity_file.read_text().strip())
                    
                    # Charging status
                    status_file = battery_dir / 'status'
                    if status_file.exists():
                        status = status_file.read_text().strip().lower()
                        power_info.is_charging = status in ['charging', 'full']
                    
                    # Power draw (if available)
                    power_file = battery_dir / 'power_now'
                    if power_file.exists():
                        power_microwatts = int(power_file.read_text().strip())
                        power_info.power_draw = power_microwatts / 1_000_000.0  # Convert to watts
                        
                except (IOError, ValueError) as e:
                    logger.debug(f"Battery info read error: {e}")
            
            # Read AC adapter information
            ac_path = self.POWER_SUPPLY_PATHS.get('ac')
            if ac_path:
                ac_dir = Path(ac_path)
                if ac_dir.exists():
                    try:
                        online_file = ac_dir / 'online'
                        if online_file.exists():
                            power_info.ac_connected = online_file.read_text().strip() == '1'
                    except (IOError, ValueError):
                        pass
            
            # Determine power state
            if power_info.ac_connected:
                power_info.state = PowerState.AC_POWER
            elif power_info.battery_level < 10:
                power_info.state = PowerState.CRITICAL_BATTERY  
            elif power_info.battery_level < 20:
                power_info.state = PowerState.LOW_BATTERY
            else:
                power_info.state = PowerState.BATTERY
                
        except Exception as e:
            logger.warning(f"Power info read failed: {e}")
            power_info.state = PowerState.UNKNOWN
        
        return power_info
    
    def _determine_thermal_state(self, temperatures: Dict[str, float]) -> ThermalState:
        """Determine thermal state from temperature readings."""
        if not temperatures:
            return ThermalState.UNKNOWN
        
        # Get maximum temperature across all sensors
        max_temp = max(temperatures.values())
        
        # Determine state based on maximum temperature
        if max_temp >= 95:
            return ThermalState.EMERGENCY
        elif max_temp >= 85:
            return ThermalState.CRITICAL
        elif max_temp >= 75:
            return ThermalState.HOT
        elif max_temp >= 65:
            return ThermalState.WARM
        elif max_temp >= 50:
            return ThermalState.NORMAL
        else:
            return ThermalState.COLD
    
    def _apply_thermal_policy(self, thermal_state: ThermalState, power_state: PowerState) -> None:
        """Apply thermal and power management policy."""
        try:
            # Get base thermal policy
            policy = self.THERMAL_POLICIES.get(thermal_state, self.THERMAL_POLICIES[ThermalState.NORMAL])
            
            # Apply power state modifications
            power_policy = self.POWER_POLICIES.get(power_state, self.POWER_POLICIES[PowerState.BATTERY])
            thread_multiplier = power_policy['thread_multiplier']
            
            # Calculate final thread limit
            base_threads = policy.max_threads
            adjusted_threads = max(1, int(base_threads * thread_multiplier))
            
            # Update thread limit
            old_limit = self.current_thread_limit
            self.current_thread_limit = adjusted_threads
            
            # Apply environment variables if thread limit changed
            if old_limit != adjusted_threads:
                os.environ['OMP_NUM_THREADS'] = str(adjusted_threads)
                os.environ['NUMEXPR_NUM_THREADS'] = str(adjusted_threads)
                logger.info(f"Thread limit adjusted: {old_limit} -> {adjusted_threads} (thermal: {thermal_state.value}, power: {power_state.value})")
            
            # Handle emergency states
            if thermal_state == ThermalState.EMERGENCY:
                logger.error("EMERGENCY THERMAL STATE - requesting shutdown")
                self.emergency_shutdown_requested = True
                self.emergency_shutdowns += 1
                
                # Notify emergency callbacks
                for callback in self.emergency_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Emergency callback failed: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to apply thermal policy: {e}")
    
    async def update_thermal_state(self) -> Tuple[ThermalState, PowerState]:
        """Update thermal and power states."""
        current_time = time.time()
        
        # Rate limit updates
        if current_time - self.last_update < self.update_interval:
            return self.current_thermal_state, self.current_power_state
        
        try:
            # Read sensors
            temperatures = self._read_thermal_sensors()
            power_info = self._read_power_info()
            
            # Determine states
            new_thermal_state = self._determine_thermal_state(temperatures)
            new_power_state = power_info.state
            
            # Check for state changes
            thermal_changed = new_thermal_state != self.current_thermal_state
            power_changed = new_power_state != self.current_power_state
            
            if thermal_changed:
                logger.info(f"Thermal state changed: {self.current_thermal_state.value} -> {new_thermal_state.value}")
                self.current_thermal_state = new_thermal_state
                self.thermal_events[new_thermal_state] += 1
                
                # Notify thermal callbacks
                for callback in self.thermal_callbacks:
                    try:
                        callback(new_thermal_state)
                    except Exception as e:
                        logger.error(f"Thermal callback failed: {e}")
            
            if power_changed:
                logger.info(f"Power state changed: {self.current_power_state.value} -> {new_power_state.value}")
                self.current_power_state = new_power_state
                
                # Notify power callbacks
                for callback in self.power_callbacks:
                    try:
                        callback(new_power_state)
                    except Exception as e:
                        logger.error(f"Power callback failed: {e}")
            
            # Update power info
            self.power_info = power_info
            
            # Apply policies if states changed
            if thermal_changed or power_changed:
                self._apply_thermal_policy(new_thermal_state, new_power_state)
            
            self.last_update = current_time
            
        except Exception as e:
            logger.error(f"Thermal state update failed: {e}")
        
        return self.current_thermal_state, self.current_power_state
    
    def get_safe_thread_count(self) -> int:
        """Get safe thread count for current thermal/power conditions."""
        return self.current_thread_limit
    
    def get_thermal_state(self) -> ThermalState:
        """Get current thermal state."""
        return self.current_thermal_state
    
    def get_power_state(self) -> PowerState:
        """Get current power state."""  
        return self.current_power_state
    
    def get_power_info(self) -> PowerInfo:
        """Get current power information."""
        return self.power_info
    
    def get_temperatures(self) -> Dict[str, float]:
        """Get current temperatures from all sensors."""
        return {sensor.name: sensor.temperature for sensor in self.sensors}
    
    def is_emergency_state(self) -> bool:
        """Check if in emergency thermal state."""
        return self.current_thermal_state == ThermalState.EMERGENCY or self.emergency_shutdown_requested
    
    def is_throttling_active(self) -> bool:
        """Check if thermal throttling is active."""
        return self.current_thermal_state in [ThermalState.HOT, ThermalState.CRITICAL, ThermalState.EMERGENCY]
    
    def can_run_background_tasks(self) -> bool:
        """Check if background tasks are allowed in current thermal state."""
        policy = self.THERMAL_POLICIES.get(self.current_thermal_state)
        return policy.background_tasks if policy else False
    
    def can_run_ml_inference(self) -> bool:
        """Check if ML inference is allowed in current thermal state."""
        policy = self.THERMAL_POLICIES.get(self.current_thermal_state)
        return policy.ml_inference if policy else False
    
    def can_run_compilation(self) -> bool:
        """Check if compilation is allowed in current thermal state."""
        policy = self.THERMAL_POLICIES.get(self.current_thermal_state)
        return policy.compilation if policy else False
    
    def add_thermal_callback(self, callback: Callable[[ThermalState], None]) -> None:
        """Add callback for thermal state changes."""
        self.thermal_callbacks.append(callback)
    
    def add_power_callback(self, callback: Callable[[PowerState], None]) -> None:
        """Add callback for power state changes."""
        self.power_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for emergency thermal events."""
        self.emergency_callbacks.append(callback)
    
    async def start_monitoring(self) -> bool:
        """Start thermal monitoring.""" 
        if self.monitoring_active:
            logger.warning("Thermal monitoring already active")
            return True
        
        try:
            # Initial state update
            await self.update_thermal_state()
            
            self.monitoring_active = True
            logger.info(f"Thermal monitoring started - {len(self.sensors)} sensors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start thermal monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> None:
        """Stop thermal monitoring."""
        self.monitoring_active = False
        logger.info("Thermal monitoring stopped")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive thermal status summary."""
        return {
            'thermal_state': self.current_thermal_state.value,
            'power_state': self.current_power_state.value,
            'thread_limit': self.current_thread_limit,
            'temperatures': self.get_temperatures(),
            'power_info': {
                'battery_level': self.power_info.battery_level,
                'is_charging': self.power_info.is_charging,
                'ac_connected': self.power_info.ac_connected,
                'power_draw': self.power_info.power_draw
            },
            'capabilities': {
                'background_tasks': self.can_run_background_tasks(),
                'ml_inference': self.can_run_ml_inference(),
                'compilation': self.can_run_compilation()
            },
            'statistics': {
                'thermal_events': dict(self.thermal_events),
                'throttle_events': self.throttle_events,
                'emergency_shutdowns': self.emergency_shutdowns
            },
            'sensors': len(self.sensors),
            'emergency_state': self.is_emergency_state()
        }

# Global thermal manager instance
_thermal_manager = None

async def get_thermal_manager() -> ThermalThreadManager:
    """Get the global thermal thread manager instance."""
    global _thermal_manager
    if _thermal_manager is None:
        _thermal_manager = ThermalThreadManager()
        await _thermal_manager.start_monitoring()
    return _thermal_manager

def get_thermal_manager_sync() -> ThermalThreadManager:
    """Get thermal manager instance (synchronous)."""
    global _thermal_manager
    if _thermal_manager is None:
        _thermal_manager = ThermalThreadManager()
    return _thermal_manager

# Convenience functions
async def get_safe_thread_count() -> int:
    """Get safe thread count for current conditions."""
    manager = await get_thermal_manager()
    await manager.update_thermal_state()
    return manager.get_safe_thread_count()

def get_safe_thread_count_sync() -> int:
    """Get safe thread count (synchronous)."""
    manager = get_thermal_manager_sync()
    # Quick thermal check
    temperatures = manager._read_thermal_sensors()
    if not temperatures:
        return 2  # Conservative default
    
    max_temp = max(temperatures.values())
    if max_temp >= 85:
        return 1  # Critical
    elif max_temp >= 75:
        return 1  # Hot
    elif max_temp >= 65:
        return 2  # Warm
    else:
        return 4  # Normal/Cold