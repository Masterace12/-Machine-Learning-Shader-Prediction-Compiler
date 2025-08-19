#!/usr/bin/env python3
"""
Enhanced D-Bus Interface for Steam Deck Gaming Mode Integration

This module provides comprehensive D-Bus communication with Steam and Gaming Mode
for seamless integration without interfering with gaming performance.

Features:
- Gaming Mode D-Bus signal monitoring
- Steam client state detection and tracking
- Gamescope session integration
- Game launch/termination detection
- Overlay state monitoring
- Steam Deck hardware event integration
- Background operation that doesn't impact gaming

Supports multiple D-Bus backends with automatic fallback:
- dbus-next (preferred pure Python implementation)
- jeepney (alternative pure Python)
- pydbus (if available)
- dbus-python (legacy support)
"""

import os
import sys
import time
import json
import asyncio
import threading
import subprocess
import logging
import signal
from typing import Dict, List, Any, Optional, Callable, Set, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager
from enum import Enum
import weakref

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# D-BUS BACKEND DETECTION AND LOADING
# =============================================================================

class DBusBackend(Enum):
    """Available D-Bus backends"""
    DBUS_NEXT = "dbus_next"
    JEEPNEY = "jeepney"
    PYDBUS = "pydbus"
    DBUS_PYTHON = "dbus_python"
    NONE = "none"

class DBusCapabilities:
    """D-Bus backend capabilities"""
    
    def __init__(self):
        self.backend = self._detect_backend()
        self.session_bus_available = self._check_session_bus()
        self.steam_dbus_available = self._check_steam_dbus()
        self.gamescope_dbus_available = self._check_gamescope_dbus()
        
    def _detect_backend(self) -> DBusBackend:
        """Detect best available D-Bus backend"""
        backends = [
            (DBusBackend.DBUS_NEXT, self._try_dbus_next),
            (DBusBackend.JEEPNEY, self._try_jeepney),
            (DBusBackend.PYDBUS, self._try_pydbus),
            (DBusBackend.DBUS_PYTHON, self._try_dbus_python)
        ]
        
        for backend, test_func in backends:
            if test_func():
                logger.info(f"D-Bus backend detected: {backend.value}")
                return backend
        
        logger.warning("No D-Bus backend available")
        return DBusBackend.NONE
    
    def _try_dbus_next(self) -> bool:
        try:
            import dbus_next
            return True
        except ImportError:
            return False
    
    def _try_jeepney(self) -> bool:
        try:
            import jeepney
            return True
        except ImportError:
            return False
    
    def _try_pydbus(self) -> bool:
        try:
            import pydbus
            return True
        except ImportError:
            return False
    
    def _try_dbus_python(self) -> bool:
        try:
            import dbus
            return True
        except ImportError:
            return False
    
    def _check_session_bus(self) -> bool:
        """Check if session bus is available"""
        try:
            # Check if D-Bus session is running
            dbus_session = os.environ.get('DBUS_SESSION_BUS_ADDRESS')
            if dbus_session:
                return True
            
            # Check for default session bus
            return os.path.exists('/run/user/{}/bus'.format(os.getuid()))
        except Exception:
            return False
    
    def _check_steam_dbus(self) -> bool:
        """Check if Steam D-Bus interface is available"""
        try:
            result = subprocess.run([
                'dbus-send', '--session', '--dest=org.freedesktop.DBus',
                '--type=method_call', '--print-reply',
                '/org/freedesktop/DBus', 'org.freedesktop.DBus.ListNames'
            ], capture_output=True, text=True, timeout=5)
            
            return 'com.steampowered' in result.stdout
        except Exception:
            return False
    
    def _check_gamescope_dbus(self) -> bool:
        """Check if Gamescope D-Bus interface is available"""
        try:
            result = subprocess.run([
                'dbus-send', '--session', '--dest=org.freedesktop.DBus',
                '--type=method_call', '--print-reply',
                '/org/freedesktop/DBus', 'org.freedesktop.DBus.ListNames'
            ], capture_output=True, text=True, timeout=5)
            
            return 'org.gamescope' in result.stdout
        except Exception:
            return False

# =============================================================================
# STEAM GAMING MODE EVENTS
# =============================================================================

@dataclass
class SteamGameEvent:
    """Steam game event data"""
    event_type: str  # 'launched', 'terminated', 'suspended', 'resumed'
    app_id: str
    process_id: Optional[int]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GamingModeEvent:
    """Gaming Mode state event"""
    event_type: str  # 'activated', 'deactivated', 'overlay_shown', 'overlay_hidden'
    state_data: Dict[str, Any]
    timestamp: float

@dataclass
class SteamDeckHardwareEvent:
    """Steam Deck hardware event"""
    event_type: str  # 'dock_connected', 'dock_disconnected', 'battery_low', 'thermal_warning'
    hardware_data: Dict[str, Any]
    timestamp: float

# =============================================================================
# D-BUS INTERFACE IMPLEMENTATIONS
# =============================================================================

class DBusNextInterface:
    """D-Bus interface using dbus-next library"""
    
    def __init__(self):
        self.bus = None
        self.steam_proxy = None
        self.gamescope_proxy = None
        self.signal_handlers = []
        
    async def initialize(self) -> bool:
        """Initialize dbus-next interface"""
        try:
            from dbus_next.aio import MessageBus
            from dbus_next import BusType, Message, MessageType
            
            self.bus = await MessageBus(bus_type=BusType.SESSION).connect()
            logger.info("dbus-next session bus connected")
            
            # Setup Steam interface
            await self._setup_steam_interface()
            
            # Setup Gamescope interface
            await self._setup_gamescope_interface()
            
            return True
            
        except Exception as e:
            logger.error(f"dbus-next initialization failed: {e}")
            return False
    
    async def _setup_steam_interface(self):
        """Setup Steam D-Bus interface"""
        try:
            # Steam main interface
            steam_introspection = await self.bus.introspect(
                'com.steampowered.Steam', '/steam'
            )
            self.steam_proxy = self.bus.get_proxy_object(
                'com.steampowered.Steam', '/steam', steam_introspection
            )
            
            # Setup signal handlers for game events
            await self._setup_steam_signals()
            
            logger.debug("Steam D-Bus interface setup complete")
            
        except Exception as e:
            logger.debug(f"Steam D-Bus interface setup failed: {e}")
    
    async def _setup_gamescope_interface(self):
        """Setup Gamescope D-Bus interface"""
        try:
            # Gamescope Gaming Mode interface
            gamescope_introspection = await self.bus.introspect(
                'org.gamescope.GameMode', '/org/gamescope/GameMode'
            )
            self.gamescope_proxy = self.bus.get_proxy_object(
                'org.gamescope.GameMode', '/org/gamescope/GameMode', gamescope_introspection
            )
            
            # Setup signal handlers for Gaming Mode
            await self._setup_gamescope_signals()
            
            logger.debug("Gamescope D-Bus interface setup complete")
            
        except Exception as e:
            logger.debug(f"Gamescope D-Bus interface setup failed: {e}")
    
    async def _setup_steam_signals(self):
        """Setup Steam signal handlers"""
        if not self.steam_proxy:
            return
        
        try:
            # Game launch signals
            self.steam_proxy.on_game_launched = self._on_steam_game_launched
            self.steam_proxy.on_game_terminated = self._on_steam_game_terminated
            self.steam_proxy.on_overlay_state_changed = self._on_steam_overlay_changed
            
        except Exception as e:
            logger.debug(f"Steam signal setup error: {e}")
    
    async def _setup_gamescope_signals(self):
        """Setup Gamescope signal handlers"""
        if not self.gamescope_proxy:
            return
        
        try:
            # Gaming Mode signals
            self.gamescope_proxy.on_gaming_mode_changed = self._on_gaming_mode_changed
            self.gamescope_proxy.on_window_focus_changed = self._on_window_focus_changed
            
        except Exception as e:
            logger.debug(f"Gamescope signal setup error: {e}")
    
    def _on_steam_game_launched(self, app_id: str, pid: int):
        """Handle Steam game launch signal"""
        event = SteamGameEvent(
            event_type='launched',
            app_id=str(app_id),
            process_id=pid,
            timestamp=time.time()
        )
        self._dispatch_event('steam_game', event)
    
    def _on_steam_game_terminated(self, app_id: str, exit_code: int):
        """Handle Steam game termination signal"""
        event = SteamGameEvent(
            event_type='terminated',
            app_id=str(app_id),
            process_id=None,
            timestamp=time.time(),
            metadata={'exit_code': exit_code}
        )
        self._dispatch_event('steam_game', event)
    
    def _on_steam_overlay_changed(self, visible: bool):
        """Handle Steam overlay state change"""
        event_type = 'overlay_shown' if visible else 'overlay_hidden'
        event = GamingModeEvent(
            event_type=event_type,
            state_data={'overlay_visible': visible},
            timestamp=time.time()
        )
        self._dispatch_event('gaming_mode', event)
    
    def _on_gaming_mode_changed(self, active: bool):
        """Handle Gaming Mode state change"""
        event_type = 'activated' if active else 'deactivated'
        event = GamingModeEvent(
            event_type=event_type,
            state_data={'gaming_mode_active': active},
            timestamp=time.time()
        )
        self._dispatch_event('gaming_mode', event)
    
    def _on_window_focus_changed(self, window_id: int, has_focus: bool):
        """Handle window focus change in Gaming Mode"""
        event = GamingModeEvent(
            event_type='window_focus_changed',
            state_data={'window_id': window_id, 'has_focus': has_focus},
            timestamp=time.time()
        )
        self._dispatch_event('gaming_mode', event)
    
    def _dispatch_event(self, event_category: str, event_data: Any):
        """Dispatch event to registered handlers"""
        # This would be implemented by the parent class
        pass
    
    async def disconnect(self):
        """Disconnect from D-Bus"""
        if self.bus:
            await self.bus.disconnect()

class JeepneyInterface:
    """D-Bus interface using jeepney library"""
    
    def __init__(self):
        self.connection = None
        
    async def initialize(self) -> bool:
        """Initialize jeepney interface"""
        try:
            import jeepney
            from jeepney.io.asyncio import open_dbus_connection
            
            self.connection = await open_dbus_connection(bus='SESSION')
            logger.info("Jeepney session bus connected")
            
            # Setup basic monitoring
            await self._setup_basic_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Jeepney initialization failed: {e}")
            return False
    
    async def _setup_basic_monitoring(self):
        """Setup basic D-Bus monitoring with jeepney"""
        # Simplified monitoring implementation
        # Jeepney requires more manual signal handling
        pass
    
    async def disconnect(self):
        """Disconnect from D-Bus"""
        if self.connection:
            self.connection.close()

# =============================================================================
# FALLBACK PROCESS MONITORING
# =============================================================================

class ProcessMonitorFallback:
    """Fallback process monitoring when D-Bus is unavailable"""
    
    def __init__(self):
        self.monitoring = False
        self.known_processes: Set[int] = set()
        self.game_processes: Dict[int, str] = {}
        
    async def initialize(self) -> bool:
        """Initialize process monitoring"""
        self.monitoring = True
        logger.info("Process monitoring fallback initialized")
        return True
    
    async def start_monitoring(self):
        """Start process monitoring loop"""
        while self.monitoring:
            try:
                await self._scan_processes()
                await asyncio.sleep(2)  # Check every 2 seconds
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _scan_processes(self):
        """Scan for Steam and game processes"""
        try:
            # Use psutil if available
            try:
                import psutil
                await self._scan_with_psutil()
            except ImportError:
                await self._scan_with_proc()
                
        except Exception as e:
            logger.error(f"Process scanning error: {e}")
    
    async def _scan_with_psutil(self):
        """Scan processes using psutil"""
        import psutil
        
        current_pids = set()
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                pid = proc.info['pid']
                name = proc.info['name'] or ""
                cmdline = proc.info['cmdline'] or []
                
                current_pids.add(pid)
                
                # Check for Steam processes
                if 'steam' in name.lower():
                    cmdline_str = ' '.join(cmdline)
                    
                    # Check for game launch
                    if '-applaunch' in cmdline_str:
                        app_id = self._extract_app_id(cmdline_str)
                        if app_id and pid not in self.known_processes:
                            self._on_game_process_detected(app_id, pid)
                        self.game_processes[pid] = app_id or "unknown"
                
                # Check for gamescope (Gaming Mode indicator)
                elif 'gamescope' in name.lower():
                    if pid not in self.known_processes:
                        self._on_gaming_mode_detected(True)
                
                self.known_processes.add(pid)
                
            except Exception:
                continue
        
        # Check for terminated processes
        terminated_pids = self.known_processes - current_pids
        for pid in terminated_pids:
            if pid in self.game_processes:
                app_id = self.game_processes[pid]
                self._on_game_process_terminated(app_id, pid)
                del self.game_processes[pid]
            
            self.known_processes.discard(pid)
    
    async def _scan_with_proc(self):
        """Scan processes using /proc filesystem"""
        try:
            proc_dir = Path('/proc')
            current_pids = set()
            
            for pid_dir in proc_dir.iterdir():
                if pid_dir.is_dir() and pid_dir.name.isdigit():
                    try:
                        pid = int(pid_dir.name)
                        current_pids.add(pid)
                        
                        if pid not in self.known_processes:
                            cmdline_file = pid_dir / 'cmdline'
                            if cmdline_file.exists():
                                with open(cmdline_file, 'rb') as f:
                                    cmdline_bytes = f.read()
                                    cmdline = cmdline_bytes.decode('utf-8', errors='ignore')
                                    
                                    if 'steam' in cmdline.lower() and '-applaunch' in cmdline:
                                        app_id = self._extract_app_id(cmdline)
                                        if app_id:
                                            self._on_game_process_detected(app_id, pid)
                                            self.game_processes[pid] = app_id
                        
                        self.known_processes.add(pid)
                        
                    except Exception:
                        continue
            
            # Handle terminated processes
            terminated_pids = self.known_processes - current_pids
            for pid in terminated_pids:
                if pid in self.game_processes:
                    app_id = self.game_processes[pid]
                    self._on_game_process_terminated(app_id, pid)
                    del self.game_processes[pid]
                
                self.known_processes.discard(pid)
                
        except Exception as e:
            logger.error(f"Proc scanning error: {e}")
    
    def _extract_app_id(self, cmdline: str) -> Optional[str]:
        """Extract Steam app ID from command line"""
        try:
            parts = cmdline.split('-applaunch')
            if len(parts) > 1:
                app_id = parts[1].strip().split()[0]
                if app_id.isdigit():
                    return app_id
        except Exception:
            pass
        return None
    
    def _on_game_process_detected(self, app_id: str, pid: int):
        """Handle game process detection"""
        event = SteamGameEvent(
            event_type='launched',
            app_id=app_id,
            process_id=pid,
            timestamp=time.time(),
            metadata={'detection_method': 'process_monitor'}
        )
        self._dispatch_event('steam_game', event)
    
    def _on_game_process_terminated(self, app_id: str, pid: int):
        """Handle game process termination"""
        event = SteamGameEvent(
            event_type='terminated',
            app_id=app_id,
            process_id=pid,
            timestamp=time.time(),
            metadata={'detection_method': 'process_monitor'}
        )
        self._dispatch_event('steam_game', event)
    
    def _on_gaming_mode_detected(self, active: bool):
        """Handle Gaming Mode detection"""
        event = GamingModeEvent(
            event_type='activated' if active else 'deactivated',
            state_data={'gaming_mode_active': active},
            timestamp=time.time()
        )
        self._dispatch_event('gaming_mode', event)
    
    def _dispatch_event(self, event_category: str, event_data: Any):
        """Dispatch event to registered handlers"""
        # This would be implemented by the parent class
        pass
    
    async def disconnect(self):
        """Stop monitoring"""
        self.monitoring = False

# =============================================================================
# MAIN STEAM D-BUS INTERFACE
# =============================================================================

class SteamDBusInterface:
    """
    Main Steam D-Bus interface with automatic backend selection and fallback
    """
    
    def __init__(self):
        self.capabilities = DBusCapabilities()
        self.interface = None
        self.event_handlers: Dict[str, List[Callable]] = {
            'steam_game': [],
            'gaming_mode': [],
            'hardware': []
        }
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Weak references to prevent circular references
        self._callback_refs: List[weakref.ref] = []
        
    async def initialize(self) -> bool:
        """Initialize the D-Bus interface"""
        logger.info(f"Initializing Steam D-Bus interface with backend: {self.capabilities.backend.value}")
        
        if self.capabilities.backend == DBusBackend.DBUS_NEXT:
            self.interface = DBusNextInterface()
        elif self.capabilities.backend == DBusBackend.JEEPNEY:
            self.interface = JeepneyInterface()
        else:
            # Fallback to process monitoring
            self.interface = ProcessMonitorFallback()
        
        # Connect interface events to our handlers
        if hasattr(self.interface, '_dispatch_event'):
            self.interface._dispatch_event = self._handle_interface_event
        
        success = await self.interface.initialize()
        if success:
            logger.info("Steam D-Bus interface initialized successfully")
        else:
            logger.warning("Steam D-Bus interface initialization failed")
        
        return success
    
    async def start_monitoring(self) -> None:
        """Start event monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start interface-specific monitoring
        if hasattr(self.interface, 'start_monitoring'):
            self.monitoring_task = asyncio.create_task(
                self.interface.start_monitoring()
            )
        
        # Start hardware monitoring for Steam Deck
        hardware_task = asyncio.create_task(self._monitor_hardware_events())
        
        logger.info("Steam D-Bus monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop event monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.interface:
            await self.interface.disconnect()
        
        logger.info("Steam D-Bus monitoring stopped")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            # Store weak reference to prevent memory leaks
            self._callback_refs.append(weakref.ref(handler))
            logger.debug(f"Added {event_type} event handler")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Remove event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                logger.debug(f"Removed {event_type} event handler")
            except ValueError:
                pass
    
    def _handle_interface_event(self, event_category: str, event_data: Any) -> None:
        """Handle events from the D-Bus interface"""
        if event_category in self.event_handlers:
            handlers = self.event_handlers[event_category]
            for handler in handlers.copy():  # Copy to avoid modification during iteration
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
                    # Remove failed handlers
                    try:
                        handlers.remove(handler)
                    except ValueError:
                        pass
    
    async def _monitor_hardware_events(self) -> None:
        """Monitor Steam Deck hardware events"""
        if not self._is_steam_deck():
            return
        
        logger.debug("Starting Steam Deck hardware monitoring")
        
        last_dock_state = None
        last_battery_level = None
        
        while self.monitoring_active:
            try:
                # Monitor dock connection
                dock_connected = self._check_dock_connection()
                if dock_connected != last_dock_state:
                    event_type = 'dock_connected' if dock_connected else 'dock_disconnected'
                    event = SteamDeckHardwareEvent(
                        event_type=event_type,
                        hardware_data={'dock_connected': dock_connected},
                        timestamp=time.time()
                    )
                    self._handle_interface_event('hardware', event)
                    last_dock_state = dock_connected
                
                # Monitor battery level
                battery_level = self._get_battery_level()
                if battery_level is not None:
                    if last_battery_level is None:
                        last_battery_level = battery_level
                    elif abs(battery_level - last_battery_level) > 5:  # 5% change
                        if battery_level < 20 and last_battery_level >= 20:
                            event = SteamDeckHardwareEvent(
                                event_type='battery_low',
                                hardware_data={'battery_level': battery_level},
                                timestamp=time.time()
                            )
                            self._handle_interface_event('hardware', event)
                        last_battery_level = battery_level
                
                # Monitor thermal state
                thermal_state = self._get_thermal_state()
                if thermal_state.get('throttling', False):
                    event = SteamDeckHardwareEvent(
                        event_type='thermal_warning',
                        hardware_data=thermal_state,
                        timestamp=time.time()
                    )
                    self._handle_interface_event('hardware', event)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Hardware monitoring error: {e}")
                await asyncio.sleep(30)
    
    def _is_steam_deck(self) -> bool:
        """Check if running on Steam Deck"""
        return os.path.exists('/home/deck')
    
    def _check_dock_connection(self) -> bool:
        """Check Steam Deck dock connection"""
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
            
            return False
            
        except Exception:
            return False
    
    def _get_battery_level(self) -> Optional[float]:
        """Get battery level percentage"""
        try:
            battery_path = '/sys/class/power_supply/BAT1/capacity'
            if os.path.exists(battery_path):
                with open(battery_path, 'r') as f:
                    return float(f.read().strip())
        except Exception:
            pass
        return None
    
    def _get_thermal_state(self) -> Dict[str, Any]:
        """Get thermal state information"""
        thermal_state = {
            'temperature': 0.0,
            'throttling': False,
            'fan_speed': None
        }
        
        try:
            # Get CPU temperature
            thermal_zones = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp'
            ]
            
            for zone in thermal_zones:
                if os.path.exists(zone):
                    with open(zone, 'r') as f:
                        temp_millic = int(f.read().strip())
                        thermal_state['temperature'] = temp_millic / 1000.0
                        break
            
            # Check for throttling
            if thermal_state['temperature'] > 85.0:
                thermal_state['throttling'] = True
            
            # Get fan speed if available
            fan_paths = [
                '/sys/class/hwmon/hwmon0/fan1_input',
                '/sys/class/hwmon/hwmon1/fan1_input'
            ]
            
            for fan_path in fan_paths:
                if os.path.exists(fan_path):
                    with open(fan_path, 'r') as f:
                        thermal_state['fan_speed'] = int(f.read().strip())
                        break
                        
        except Exception as e:
            logger.debug(f"Thermal state error: {e}")
        
        return thermal_state
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get D-Bus capabilities information"""
        return {
            'backend': self.capabilities.backend.value,
            'session_bus_available': self.capabilities.session_bus_available,
            'steam_dbus_available': self.capabilities.steam_dbus_available,
            'gamescope_dbus_available': self.capabilities.gamescope_dbus_available,
            'monitoring_active': self.monitoring_active,
            'steam_deck_hardware': self._is_steam_deck()
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global D-Bus interface instance
_steam_dbus: Optional[SteamDBusInterface] = None

def get_steam_dbus() -> SteamDBusInterface:
    """Get or create the global Steam D-Bus interface"""
    global _steam_dbus
    if _steam_dbus is None:
        _steam_dbus = SteamDBusInterface()
    return _steam_dbus

async def start_steam_dbus_monitoring():
    """Start Steam D-Bus monitoring"""
    dbus_interface = get_steam_dbus()
    await dbus_interface.initialize()
    await dbus_interface.start_monitoring()

async def stop_steam_dbus_monitoring():
    """Stop Steam D-Bus monitoring"""
    dbus_interface = get_steam_dbus()
    await dbus_interface.stop_monitoring()

@asynccontextmanager
async def steam_dbus_context():
    """Context manager for Steam D-Bus monitoring"""
    dbus_interface = get_steam_dbus()
    try:
        await dbus_interface.initialize()
        await dbus_interface.start_monitoring()
        yield dbus_interface
    finally:
        await dbus_interface.stop_monitoring()

# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

async def test_dbus_interface():
    """Test the D-Bus interface"""
    print("\n🔌 Steam D-Bus Interface Test")
    print("=" * 40)
    
    dbus_interface = SteamDBusInterface()
    
    # Test capabilities
    capabilities = dbus_interface.get_capabilities()
    print(f"\n📋 Capabilities:")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    
    # Test event handlers
    def game_event_handler(event: SteamGameEvent):
        print(f"🎮 Game Event: {event.event_type} - App ID: {event.app_id}")
    
    def gaming_mode_handler(event: GamingModeEvent):
        print(f"🕹️ Gaming Mode Event: {event.event_type}")
    
    def hardware_handler(event: SteamDeckHardwareEvent):
        print(f"⚙️ Hardware Event: {event.event_type}")
    
    dbus_interface.add_event_handler('steam_game', game_event_handler)
    dbus_interface.add_event_handler('gaming_mode', gaming_mode_handler)
    dbus_interface.add_event_handler('hardware', hardware_handler)
    
    # Initialize and start monitoring
    print(f"\n🚀 Initializing D-Bus interface...")
    success = await dbus_interface.initialize()
    print(f"  Initialization: {'✅' if success else '❌'}")
    
    if success:
        print(f"\n👀 Starting monitoring (will run for 30 seconds)...")
        await dbus_interface.start_monitoring()
        
        # Wait for events
        await asyncio.sleep(30)
        
        print(f"\n🛑 Stopping monitoring...")
        await dbus_interface.stop_monitoring()
    
    print(f"\n✅ D-Bus interface test completed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        asyncio.run(test_dbus_interface())
    else:
        print("Steam D-Bus Interface for Gaming Mode Integration")
        print("Usage: --test to run test suite")