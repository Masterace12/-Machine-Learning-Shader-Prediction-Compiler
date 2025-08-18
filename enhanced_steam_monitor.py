#!/usr/bin/env python3
"""
Enhanced Steam integration monitor for shader prediction
Uses pure Python alternatives to avoid D-Bus compilation issues
"""

import os
import sys
import time
import json
import asyncio
import threading
import subprocess
import logging
from pathlib import Path
from typing import Set, Optional, Dict, Any

# Multi-tier fallback system for Steam monitoring
HAS_DBUS_NEXT = False
HAS_JEEPNEY = False
HAS_WATCHDOG = False
HAS_PSUTIL = False

# Try dbus-next (preferred pure Python D-Bus)
try:
    import dbus_next
    from dbus_next.aio import MessageBus
    from dbus_next import BusType
    HAS_DBUS_NEXT = True
except ImportError:
    pass

# Try jeepney (alternative pure Python D-Bus)
try:
    import jeepney
    HAS_JEEPNEY = True
except ImportError:
    pass

# Try watchdog for file system monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    # Create dummy base class if watchdog is not available
    class FileSystemEventHandler:
        def on_modified(self, event):
            pass
    HAS_WATCHDOG = False

# Try psutil for process monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DBusNextSteamMonitor:
    """Steam monitor using dbus-next (pure Python)"""
    
    def __init__(self):
        self.bus = None
        self.known_games: Set[str] = set()
        
    async def start_monitoring(self):
        """Start D-Bus monitoring using dbus-next"""
        try:
            self.bus = await MessageBus(bus_type=BusType.SESSION).connect()
            logger.info("D-Bus monitoring started with dbus-next")
            
            # Monitor for Steam process changes
            await self._monitor_steam_dbus()
        except Exception as e:
            logger.error(f"dbus-next monitoring failed: {e}")
            raise
            
    async def _monitor_steam_dbus(self):
        """Monitor Steam via D-Bus signals"""
        # This would implement Steam D-Bus signal monitoring
        # For now, fall back to process monitoring
        await self._monitor_steam_processes()
        
    async def _monitor_steam_processes(self):
        """Monitor Steam processes as fallback"""
        while True:
            try:
                if HAS_PSUTIL:
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        if proc.info['name'] and 'steam' in proc.info['name'].lower():
                            cmdline = ' '.join(proc.info['cmdline'] or [])
                            if '-applaunch' in cmdline:
                                app_id = self._extract_app_id(cmdline)
                                if app_id and app_id not in self.known_games:
                                    self.known_games.add(app_id)
                                    await self._handle_game_launch(app_id)
            except Exception as e:
                logger.warning(f"Process monitoring error: {e}")
                
            await asyncio.sleep(2)
    
    def _extract_app_id(self, cmdline: str) -> Optional[str]:
        """Extract Steam app ID from command line"""
        parts = cmdline.split('-applaunch')
        if len(parts) > 1:
            app_id = parts[1].strip().split()[0]
            return app_id
        return None
    
    async def _handle_game_launch(self, app_id: str):
        """Handle Steam game launch"""
        logger.info(f"Game launched: Steam App ID {app_id}")
        await self._trigger_shader_prediction(app_id)
    
    async def _trigger_shader_prediction(self, app_id: str):
        """Trigger shader prediction for the launched game"""
        try:
            main_script = Path(__file__).parent / "main.py"
            subprocess.Popen([
                sys.executable,
                str(main_script),
                '--predict-for-app', app_id
            ])
            logger.info(f"Shader prediction triggered for app {app_id}")
        except Exception as e:
            logger.error(f"Failed to trigger prediction for app {app_id}: {e}")


class WatchdogSteamMonitor(FileSystemEventHandler):
    """Steam monitor using file system watching"""
    
    def __init__(self):
        self.observer = Observer()
        self.known_games: Set[str] = set()
        self.steam_log_paths = [
            Path.home() / ".steam" / "logs",
            Path.home() / ".local" / "share" / "Steam" / "logs",
            Path("/tmp").glob("proton_*"),
        ]
        
    def start_monitoring(self):
        """Start file system monitoring"""
        logger.info("Starting file system monitoring for Steam")
        
        # Watch Steam log directories
        for log_path in self.steam_log_paths:
            if isinstance(log_path, Path) and log_path.exists():
                self.observer.schedule(self, str(log_path), recursive=True)
                logger.info(f"Monitoring Steam logs at: {log_path}")
            elif hasattr(log_path, '__iter__'):  # glob result
                for path in log_path:
                    if path.exists():
                        self.observer.schedule(self, str(path), recursive=True)
        
        self.observer.start()
        
        # Also start process monitoring as backup
        threading.Thread(target=self._monitor_processes, daemon=True).start()
    
    def on_modified(self, event):
        """Handle file system events"""
        if event.is_directory:
            return
            
        # Check for Steam game launch indicators
        if "steam" in event.src_path.lower() and any(ext in event.src_path for ext in [".log", ".txt"]):
            self._check_log_for_game_launch(event.src_path)
    
    def _check_log_for_game_launch(self, log_path: str):
        """Check log file for game launch events"""
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read last few lines for recent events
                lines = f.readlines()[-10:]
                for line in lines:
                    if "AppID" in line and "launching" in line.lower():
                        app_id = self._extract_app_id_from_log(line)
                        if app_id and app_id not in self.known_games:
                            self.known_games.add(app_id)
                            self._handle_game_launch(app_id)
        except Exception as e:
            logger.debug(f"Log parsing error for {log_path}: {e}")
    
    def _extract_app_id_from_log(self, line: str) -> Optional[str]:
        """Extract app ID from Steam log line"""
        import re
        match = re.search(r'AppID[:\s]+(\d+)', line)
        return match.group(1) if match else None
    
    def _monitor_processes(self):
        """Backup process monitoring"""
        while True:
            try:
                if HAS_PSUTIL:
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        if proc.info['name'] and 'steam' in proc.info['name'].lower():
                            cmdline = ' '.join(proc.info['cmdline'] or [])
                            if '-applaunch' in cmdline:
                                app_id = self._extract_app_id(cmdline)
                                if app_id and app_id not in self.known_games:
                                    self.known_games.add(app_id)
                                    self._handle_game_launch(app_id)
            except Exception as e:
                logger.warning(f"Process monitoring error: {e}")
                
            time.sleep(5)
    
    def _extract_app_id(self, cmdline: str) -> Optional[str]:
        """Extract Steam app ID from command line"""
        parts = cmdline.split('-applaunch')
        if len(parts) > 1:
            app_id = parts[1].strip().split()[0]
            return app_id
        return None
    
    def _handle_game_launch(self, app_id: str):
        """Handle Steam game launch"""
        logger.info(f"Game launched: Steam App ID {app_id}")
        self._trigger_shader_prediction(app_id)
    
    def _trigger_shader_prediction(self, app_id: str):
        """Trigger shader prediction for the launched game"""
        try:
            main_script = Path(__file__).parent / "main.py"
            subprocess.Popen([
                sys.executable,
                str(main_script),
                '--predict-for-app', app_id
            ])
            logger.info(f"Shader prediction triggered for app {app_id}")
        except Exception as e:
            logger.error(f"Failed to trigger prediction for app {app_id}: {e}")
    
    def stop_monitoring(self):
        """Stop file system monitoring"""
        self.observer.stop()
        self.observer.join()


class FallbackSteamMonitor:
    """Fallback Steam monitor using only psutil (no external deps)"""
    
    def __init__(self):
        self.known_games: Set[str] = set()
        self.monitoring = False
        
    def start_monitoring(self):
        """Start basic process monitoring"""
        logger.info("Starting fallback Steam monitoring (psutil only)")
        self.monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                if HAS_PSUTIL:
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        if proc.info['name'] and 'steam' in proc.info['name'].lower():
                            cmdline = ' '.join(proc.info['cmdline'] or [])
                            if '-applaunch' in cmdline:
                                app_id = self._extract_app_id(cmdline)
                                if app_id and app_id not in self.known_games:
                                    self.known_games.add(app_id)
                                    self._handle_game_launch(app_id)
                else:
                    logger.warning("psutil not available, cannot monitor Steam")
                    break
            except Exception as e:
                logger.warning(f"Steam monitoring error: {e}")
                
            time.sleep(3)
    
    def _extract_app_id(self, cmdline: str) -> Optional[str]:
        """Extract Steam app ID from command line"""
        parts = cmdline.split('-applaunch')
        if len(parts) > 1:
            app_id = parts[1].strip().split()[0]
            return app_id
        return None
    
    def _handle_game_launch(self, app_id: str):
        """Handle Steam game launch"""
        logger.info(f"Game launched: Steam App ID {app_id}")
        self._trigger_shader_prediction(app_id)
    
    def _trigger_shader_prediction(self, app_id: str):
        """Trigger shader prediction for the launched game"""
        try:
            main_script = Path(__file__).parent / "main.py"
            subprocess.Popen([
                sys.executable,
                str(main_script),
                '--predict-for-app', app_id
            ])
            logger.info(f"Shader prediction triggered for app {app_id}")
        except Exception as e:
            logger.error(f"Failed to trigger prediction for app {app_id}: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False


class EnhancedSteamMonitor:
    """Enhanced Steam monitor with multiple fallback strategies"""
    
    def __init__(self):
        self.monitor = None
        self.monitor_type = None
        
    async def start_monitoring(self):
        """Start the best available monitoring method"""
        # Try dbus-next first (best option)
        if HAS_DBUS_NEXT:
            try:
                self.monitor = DBusNextSteamMonitor()
                await self.monitor.start_monitoring()
                self.monitor_type = "dbus-next"
                logger.info("Using dbus-next for Steam monitoring")
                return
            except Exception as e:
                logger.warning(f"dbus-next failed: {e}")
        
        # Try watchdog file monitoring
        if HAS_WATCHDOG:
            try:
                self.monitor = WatchdogSteamMonitor()
                self.monitor.start_monitoring()
                self.monitor_type = "watchdog"
                logger.info("Using file system monitoring for Steam")
                return
            except Exception as e:
                logger.warning(f"Watchdog failed: {e}")
        
        # Fallback to basic process monitoring
        try:
            self.monitor = FallbackSteamMonitor()
            self.monitor.start_monitoring()
            self.monitor_type = "fallback"
            logger.info("Using fallback process monitoring for Steam")
        except Exception as e:
            logger.error(f"All monitoring methods failed: {e}")
            raise
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitor and hasattr(self.monitor, 'stop_monitoring'):
            self.monitor.stop_monitoring()


async def main():
    """Main entry point"""
    logger.info("Starting Enhanced Steam Monitor")
    logger.info(f"Available methods: dbus-next={HAS_DBUS_NEXT}, watchdog={HAS_WATCHDOG}, psutil={HAS_PSUTIL}")
    
    monitor = EnhancedSteamMonitor()
    
    try:
        await monitor.start_monitoring()
        
        # Keep running
        while True:
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Shutting down Steam monitor")
    except Exception as e:
        logger.error(f"Steam monitor error: {e}")
    finally:
        monitor.stop_monitoring()


if __name__ == '__main__':
    asyncio.run(main())