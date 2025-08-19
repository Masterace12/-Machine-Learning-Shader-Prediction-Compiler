#!/usr/bin/env python3
"""
Steam D-Bus Integration Layer for Real-Time Gaming Mode Detection
================================================================

Provides real-time Steam client integration for the ML Shader Prediction Compiler.
Handles gaming mode detection, process coordination, and resource management.

Steam Deck Integration Features:
- Real-time gaming mode detection via D-Bus
- Steam client process monitoring  
- Gaming session lifecycle tracking
- Thread resource coordination with Steam
- Steam overlay and input system awareness

Usage:
    integration = SteamIntegration()
    if integration.is_gaming_mode_active():
        # Apply gaming mode optimizations
        pass
"""

import os
import sys
import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from enum import Enum

# Early threading setup MUST be done before any other imports
if 'setup_threading' not in sys.modules:
    import setup_threading
    setup_threading.configure_for_steam_deck()

# Safe imports after threading configuration
try:
    import dbus
    import dbus.mainloop.glib
    from gi.repository import GLib
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

try:
    from dbus_next.aio import MessageBus
    from dbus_next import BusType
    DBUS_NEXT_AVAILABLE = True
except ImportError:
    DBUS_NEXT_AVAILABLE = False

logger = logging.getLogger(__name__)

class SteamState(Enum):
    """Steam client states."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    GAMING = "gaming"
    OVERLAY_ACTIVE = "overlay_active"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"

@dataclass
class GameSession:
    """Information about active gaming session."""
    app_id: Optional[str] = None
    app_name: Optional[str] = None
    start_time: float = 0.0
    process_id: Optional[int] = None
    is_proton: bool = False
    is_native: bool = False
    overlay_active: bool = False

@dataclass
class SteamResourceUsage:
    """Steam client resource usage information."""
    process_count: int = 0
    thread_count: int = 0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    reserved_threads: int = 0
    overlay_threads: int = 0

class SteamIntegration:
    """Steam D-Bus integration and gaming mode detection."""
    
    # Steam D-Bus service identifiers
    STEAM_DBUS_SERVICES = [
        'com.steampowered.Steam',
        'org.freedesktop.steam',
        'com.valve.steam'
    ]
    
    # Steam process names to monitor
    STEAM_PROCESSES = [
        'steam',
        'steamwebhelper', 
        'steamos-session',
        'gamescope',
        'steam.exe'
    ]
    
    # Gaming mode detection patterns
    GAMING_INDICATORS = [
        'gamescope',
        'steam_app_',
        'proton',
        'wine'
    ]
    
    def __init__(self):
        self.is_steam_deck = setup_threading.is_steam_deck()
        self.current_state = SteamState.UNKNOWN
        self.game_session = GameSession()
        self.resource_usage = SteamResourceUsage()
        self.state_callbacks: List[Callable[[SteamState], None]] = []
        self.game_callbacks: List[Callable[[GameSession], None]] = []
        
        # D-Bus connections
        self.dbus_session = None
        self.dbus_system = None
        self.dbus_available = DBUS_AVAILABLE or DBUS_NEXT_AVAILABLE
        
        # Monitoring state
        self.monitoring_active = False
        self.last_state_check = 0.0
        self.state_cache_duration = 1.0  # Cache state for 1 second
        
        logger.info(f"Steam integration initialized - Steam Deck: {self.is_steam_deck}, D-Bus: {self.dbus_available}")
    
    async def initialize_dbus(self) -> bool:
        """Initialize D-Bus connections for Steam monitoring."""
        if not self.dbus_available:
            logger.warning("D-Bus not available - using process-based detection only")
            return False
        
        try:
            if DBUS_NEXT_AVAILABLE:
                # Prefer dbus-next for async support
                self.dbus_session = await MessageBus(bus_type=BusType.SESSION).connect()
                logger.info("Connected to D-Bus session bus (dbus-next)")
                return True
            elif DBUS_AVAILABLE:
                # Fallback to python-dbus
                dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
                self.dbus_session = dbus.SessionBus()
                logger.info("Connected to D-Bus session bus (python-dbus)")
                return True
        except Exception as e:
            logger.warning(f"D-Bus initialization failed: {e}")
            self.dbus_available = False
        
        return False
    
    def _get_steam_processes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get information about running Steam processes."""
        steam_processes = {'steam': [], 'gaming': [], 'other': []}
        
        try:
            # Get detailed process information
            result = subprocess.run(
                ['ps', '-eo', 'pid,ppid,comm,args,%cpu,%mem,nlwp'], 
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return steam_processes
            
            for line in result.stdout.splitlines()[1:]:  # Skip header
                try:
                    parts = line.split(None, 6)
                    if len(parts) < 7:
                        continue
                    
                    pid, ppid, comm, args, cpu, mem, threads = parts
                    process_info = {
                        'pid': int(pid),
                        'ppid': int(ppid), 
                        'comm': comm,
                        'args': args,
                        'cpu_percent': float(cpu),
                        'memory_percent': float(mem),
                        'thread_count': int(threads)
                    }
                    
                    # Categorize processes
                    if any(steam_proc in comm.lower() for steam_proc in self.STEAM_PROCESSES):
                        steam_processes['steam'].append(process_info)
                    elif any(gaming_proc in args.lower() for gaming_proc in self.GAMING_INDICATORS):
                        steam_processes['gaming'].append(process_info)
                    
                except (ValueError, IndexError):
                    continue
                    
        except Exception as e:
            logger.warning(f"Process enumeration failed: {e}")
        
        return steam_processes
    
    def _detect_gaming_mode_process(self) -> bool:
        """Detect gaming mode via process analysis."""
        processes = self._get_steam_processes()
        
        # Update resource usage
        total_threads = sum(p['thread_count'] for p in processes['steam'])
        total_memory = sum(p['memory_percent'] for p in processes['steam'])
        total_cpu = sum(p['cpu_percent'] for p in processes['steam'])
        
        self.resource_usage.process_count = len(processes['steam'])
        self.resource_usage.thread_count = total_threads
        self.resource_usage.memory_mb = total_memory * 160  # Rough estimate (16GB * percentage)
        self.resource_usage.cpu_percent = min(total_cpu, 100.0)
        self.resource_usage.reserved_threads = max(8, total_threads // 2)  # Conservative estimate
        
        # Gaming mode indicators
        gaming_active = len(processes['gaming']) > 0
        
        # Check for gamescope (Steam Deck gaming mode)
        if self.is_steam_deck:
            gamescope_active = any('gamescope' in p['comm'].lower() for p in processes['steam'])
            gaming_active = gaming_active or gamescope_active
        
        # Check for Steam overlay processes
        overlay_processes = [p for p in processes['steam'] if 'overlay' in p['comm'].lower()]
        self.resource_usage.overlay_threads = sum(p['thread_count'] for p in overlay_processes)
        
        return gaming_active
    
    async def _detect_gaming_mode_dbus(self) -> bool:
        """Detect gaming mode via D-Bus (when available).""" 
        if not self.dbus_available or not self.dbus_session:
            return False
        
        try:
            # Try to detect Steam client state via D-Bus
            for service_name in self.STEAM_DBUS_SERVICES:
                try:
                    if DBUS_NEXT_AVAILABLE:
                        # Use dbus-next async interface
                        introspection = await self.dbus_session.introspect(service_name, '/')
                        # Steam D-Bus interface detection would go here
                        # This is a placeholder - actual Steam D-Bus API may vary
                        return False  # Not implemented yet
                    elif DBUS_AVAILABLE:
                        # Use python-dbus interface
                        proxy = self.dbus_session.get_object(service_name, '/')
                        # Steam D-Bus interface detection would go here
                        return False  # Not implemented yet
                        
                except Exception:
                    continue  # Try next service
        except Exception as e:
            logger.debug(f"D-Bus gaming mode detection failed: {e}")
        
        return False
    
    def _determine_steam_state(self, processes: Dict[str, List[Dict[str, Any]]]) -> SteamState:
        """Determine current Steam state from process information."""
        steam_running = len(processes['steam']) > 0
        gaming_active = len(processes['gaming']) > 0
        
        if not steam_running:
            return SteamState.STOPPED
        elif gaming_active:
            return SteamState.GAMING
        elif steam_running:
            # Check if overlay is active (high thread count in Steam processes)
            high_thread_processes = [p for p in processes['steam'] if p['thread_count'] > 10]
            if high_thread_processes:
                return SteamState.OVERLAY_ACTIVE
            return SteamState.RUNNING
        else:
            return SteamState.UNKNOWN
    
    def _update_game_session(self, processes: Dict[str, List[Dict[str, Any]]]) -> None:
        """Update game session information from process data.""" 
        gaming_processes = processes['gaming']
        
        if not gaming_processes:
            # No gaming processes - clear session
            if self.game_session.app_id:
                logger.info("Gaming session ended")
                self.game_session = GameSession()
            return
        
        # Find the main game process (largest PID usually indicates most recent)
        main_game_process = max(gaming_processes, key=lambda p: p['pid'])
        
        # Extract game information
        args = main_game_process['args'].lower()
        
        # Check if this is a new game session
        new_pid = main_game_process['pid'] 
        if new_pid != self.game_session.process_id:
            logger.info(f"New gaming session detected: PID {new_pid}")
            
            self.game_session.process_id = new_pid
            self.game_session.start_time = time.time()
            self.game_session.is_proton = 'proton' in args or 'wine' in args
            self.game_session.is_native = not self.game_session.is_proton
            
            # Try to extract app ID from Steam process args
            try:
                if 'steam_app_' in args:
                    app_start = args.find('steam_app_') + len('steam_app_')
                    app_end = args.find(' ', app_start)
                    if app_end == -1:
                        app_end = len(args)
                    self.game_session.app_id = args[app_start:app_end]
                    logger.info(f"Detected Steam app ID: {self.game_session.app_id}")
            except Exception as e:
                logger.debug(f"App ID extraction failed: {e}")
        
        # Check overlay state
        self.game_session.overlay_active = self.resource_usage.overlay_threads > 5
    
    async def update_state(self) -> SteamState:
        """Update Steam state and return current state.""" 
        current_time = time.time()
        
        # Use cached state if recent
        if current_time - self.last_state_check < self.state_cache_duration:
            return self.current_state
        
        try:
            # Get process information
            processes = self._get_steam_processes()
            
            # Try D-Bus detection first (if available)
            dbus_gaming = await self._detect_gaming_mode_dbus()
            
            # Fallback to process-based detection
            process_gaming = self._detect_gaming_mode_process()
            
            # Combine detection methods
            gaming_active = dbus_gaming or process_gaming
            
            # Determine new state
            new_state = self._determine_steam_state(processes)
            
            # Update game session
            self._update_game_session(processes)
            
            # Check for state changes
            if new_state != self.current_state:
                logger.info(f"Steam state changed: {self.current_state} -> {new_state}")
                old_state = self.current_state
                self.current_state = new_state
                
                # Notify callbacks
                for callback in self.state_callbacks:
                    try:
                        callback(new_state)
                    except Exception as e:
                        logger.error(f"State callback failed: {e}")
            
            # Notify game callbacks if session changed  
            if self.game_session.process_id:
                for callback in self.game_callbacks:
                    try:
                        callback(self.game_session)
                    except Exception as e:
                        logger.error(f"Game callback failed: {e}")
            
            self.last_state_check = current_time
            
        except Exception as e:
            logger.error(f"State update failed: {e}")
        
        return self.current_state
    
    def is_gaming_mode_active(self) -> bool:
        """Check if gaming mode is currently active."""
        return self.current_state in [SteamState.GAMING, SteamState.OVERLAY_ACTIVE]
    
    def is_steam_running(self) -> bool:
        """Check if Steam client is running."""
        return self.current_state not in [SteamState.STOPPED, SteamState.UNKNOWN]
    
    def get_resource_usage(self) -> SteamResourceUsage:
        """Get current Steam resource usage."""
        return self.resource_usage
    
    def get_recommended_thread_limit(self) -> int:
        """Get recommended thread limit based on Steam state."""
        if not self.is_steam_running():
            return 4  # More threads when Steam is not running
        elif self.is_gaming_mode_active():
            return 1  # Minimal threads during gaming
        elif self.current_state == SteamState.OVERLAY_ACTIVE:
            return 1  # Conservative when overlay is active
        else:
            return 2  # Moderate when Steam is running but not gaming
    
    def get_game_session(self) -> Optional[GameSession]:
        """Get current game session information.""" 
        if self.game_session.process_id:
            return self.game_session
        return None
    
    def add_state_callback(self, callback: Callable[[SteamState], None]) -> None:
        """Add callback for Steam state changes."""
        self.state_callbacks.append(callback)
    
    def add_game_callback(self, callback: Callable[[GameSession], None]) -> None:
        """Add callback for game session changes."""
        self.game_callbacks.append(callback)
    
    async def start_monitoring(self) -> bool:
        """Start continuous Steam state monitoring."""
        if self.monitoring_active:
            logger.warning("Steam monitoring already active")
            return True
        
        try:
            # Initialize D-Bus if available
            await self.initialize_dbus()
            
            # Initial state update
            await self.update_state()
            
            self.monitoring_active = True
            logger.info("Steam monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Steam monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> None:
        """Stop Steam state monitoring."""
        self.monitoring_active = False
        
        # Close D-Bus connections
        if self.dbus_session:
            try:
                self.dbus_session.close()
            except Exception:
                pass
            self.dbus_session = None
        
        logger.info("Steam monitoring stopped")
    
    async def wait_for_gaming_mode(self, timeout: float = 30.0) -> bool:
        """Wait for gaming mode to become active."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            await self.update_state()
            if self.is_gaming_mode_active():
                return True
            await asyncio.sleep(0.5)
        
        return False
    
    async def wait_for_gaming_mode_end(self, timeout: float = 300.0) -> bool:
        """Wait for gaming mode to end."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            await self.update_state()
            if not self.is_gaming_mode_active():
                return True
            await asyncio.sleep(1.0)
        
        return False

# Global Steam integration instance  
_steam_integration = None

async def get_steam_integration() -> SteamIntegration:
    """Get the global Steam integration instance."""
    global _steam_integration
    if _steam_integration is None:
        _steam_integration = SteamIntegration()
        await _steam_integration.start_monitoring()
    return _steam_integration

def get_steam_integration_sync() -> SteamIntegration:
    """Get Steam integration instance (synchronous)."""
    global _steam_integration
    if _steam_integration is None:
        _steam_integration = SteamIntegration()
    return _steam_integration

# Convenience functions
async def is_gaming_active() -> bool:
    """Check if gaming mode is active."""
    integration = await get_steam_integration()
    await integration.update_state()
    return integration.is_gaming_mode_active()

def is_gaming_active_sync() -> bool:
    """Check if gaming mode is active (synchronous)."""
    integration = get_steam_integration_sync()
    # Run a quick synchronous check
    processes = integration._get_steam_processes()
    return len(processes['gaming']) > 0

async def get_recommended_threads() -> int:
    """Get recommended thread count based on Steam state."""
    integration = await get_steam_integration()
    await integration.update_state()
    return integration.get_recommended_thread_limit()

def get_recommended_threads_sync() -> int:
    """Get recommended thread count (synchronous)."""
    integration = get_steam_integration_sync()
    # Quick check without D-Bus
    processes = integration._get_steam_processes()
    gaming_active = len(processes['gaming']) > 0
    steam_running = len(processes['steam']) > 0
    
    if gaming_active:
        return 1  # Minimal during gaming
    elif steam_running:
        return 2  # Moderate when Steam running  
    else:
        return 4  # More when Steam not running