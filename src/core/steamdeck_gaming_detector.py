#!/usr/bin/env python3
"""
Steam Deck Gaming Mode Detector
Detects when games are running and adapts system resources accordingly
"""

import os
import re
import time
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Set, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import psutil

class GamingState(Enum):
    """Gaming activity state"""
    IDLE = "idle"                    # No games running
    LAUNCHING = "launching"          # Game starting up
    ACTIVE_2D = "active_2d"         # 2D/light game
    ACTIVE_3D = "active_3d"         # 3D/demanding game
    STEAM_OVERLAY = "steam_overlay"  # Steam overlay active
    SUSPENDED = "suspended"          # Game suspended/minimized

@dataclass
class GameProcess:
    """Information about a detected game process"""
    pid: int
    name: str
    exe_path: str
    cpu_percent: float
    memory_mb: float
    gpu_usage: int
    is_steam_game: bool
    launch_time: float

@dataclass
class ResourceUsage:
    """Current system resource usage"""
    cpu_percent: float
    memory_percent: float
    gpu_percent: int
    thermal_temp: float
    battery_percent: int
    is_charging: bool

class SteamDeckGamingDetector:
    """
    Steam Deck gaming mode detector
    
    Features:
    - Game process detection
    - Steam integration via D-Bus
    - GPU utilization monitoring
    - Adaptive resource management
    - Gaming-specific optimizations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Detection state
        self.current_state = GamingState.IDLE
        self.detected_games: Dict[int, GameProcess] = {}
        self.last_state_change = time.time()
        
        # Configuration
        self.gpu_threshold_2d = 20      # GPU usage % for 2D games
        self.gpu_threshold_3d = 50      # GPU usage % for 3D games
        self.cpu_threshold_game = 25    # CPU usage % for game detection
        self.memory_threshold_game = 200  # Memory MB for game detection
        
        # Known game patterns
        self.game_executable_patterns = [
            r'.*\.exe$',      # Windows games via Proton
            r'.*\.x86_64$',   # Linux native games
            r'gamepadui',     # Steam Big Picture
            r'steam',         # Steam client
            r'.*game.*',      # Generic game pattern
        ]
        
        # Process blacklist (not games)
        self.process_blacklist = {
            'python3', 'python', 'bash', 'sh', 'systemd', 'dbus', 'pipewire',
            'pulseaudio', 'NetworkManager', 'systemd-', 'kworker', 'ksoftirqd',
            'migration', 'chrome', 'firefox', 'code', 'kate', 'dolphin',
            'plasmashell', 'kwin_x11', 'Xorg', 'sddm'
        }
        
        # Steam process names
        self.steam_processes = {
            'steam', 'steamwebhelper', 'steamos-session', 'gamepadui',
            'fossilize_replay', 'steamtours'
        }
        
        # Threading
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        self._state_lock = threading.Lock()
        
        # Performance tracking
        self.state_history: List[tuple] = []  # (timestamp, state, resource_usage)
        self.max_history_size = 1000
        
        self.logger.info("Steam Deck gaming detector initialized")
    
    def _is_game_process(self, proc_info: Dict) -> bool:
        """Determine if a process is likely a game"""
        name = proc_info.get('name', '').lower()
        exe = proc_info.get('exe', '').lower()
        
        # Check blacklist first
        if any(blacklisted in name for blacklisted in self.process_blacklist):
            return False
        
        # Check for game executable patterns
        for pattern in self.game_executable_patterns:
            if re.match(pattern, name) or re.match(pattern, exe):
                # Additional checks for resource usage
                cpu_percent = proc_info.get('cpu_percent', 0)
                memory_mb = proc_info.get('memory_info', psutil.pmem(0, 0)).rss / (1024 * 1024)
                
                if cpu_percent > self.cpu_threshold_game or memory_mb > self.memory_threshold_game:
                    return True
        
        return False
    
    def _is_steam_process(self, proc_name: str) -> bool:
        """Check if process is Steam-related"""
        return any(steam_proc in proc_name.lower() for steam_proc in self.steam_processes)
    
    def _get_gpu_usage(self) -> int:
        """Get current GPU usage percentage"""
        try:
            gpu_busy_path = "/sys/class/drm/card0/device/gpu_busy_percent"
            if Path(gpu_busy_path).exists():
                with open(gpu_busy_path, 'r') as f:
                    return int(f.read().strip())
        except Exception:
            pass
        
        # Fallback: estimate from process GPU usage
        try:
            # Use nvidia-smi style command if available
            result = subprocess.run(['radeontop', '-d', '-', '-l', '1'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # Parse radeontop output
                for line in result.stdout.split('\n'):
                    if 'Graphics pipe' in line:
                        match = re.search(r'(\d+)\.?\d*%', line)
                        if match:
                            return int(float(match.group(1)))
        except Exception:
            pass
        
        return 0
    
    def _get_resource_usage(self) -> ResourceUsage:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        gpu_percent = self._get_gpu_usage()
        
        # Get temperature
        thermal_temp = 70.0  # Default
        try:
            temp_path = "/sys/class/thermal/thermal_zone0/temp"
            if Path(temp_path).exists():
                with open(temp_path, 'r') as f:
                    temp_raw = int(f.read().strip())
                    thermal_temp = temp_raw / 1000.0 if temp_raw > 1000 else temp_raw
        except Exception:
            pass
        
        # Get battery info
        battery_percent = 100
        is_charging = True
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_percent = int(battery.percent)
                is_charging = battery.power_plugged
        except Exception:
            pass
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_percent=gpu_percent,
            thermal_temp=thermal_temp,
            battery_percent=battery_percent,
            is_charging=is_charging
        )
    
    def _detect_steam_state(self) -> Optional[GamingState]:
        """Detect Steam-specific gaming state via D-Bus"""
        try:
            # Try to query Steam via D-Bus
            result = subprocess.run([
                'dbus-send', '--session', '--print-reply',
                '--dest=org.freedesktop.portal.Desktop',
                '/org/freedesktop/portal/desktop',
                'org.freedesktop.portal.GameMode.QueryStatus'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0 and 'true' in result.stdout.lower():
                return GamingState.ACTIVE_3D
                
        except Exception:
            pass
        
        # Fallback: check for Steam overlay
        for proc in psutil.process_iter(['pid', 'name']):
            name = proc.info['name'].lower()
            if 'gameoverlayui' in name or 'steamoverlay' in name:
                return GamingState.STEAM_OVERLAY
        
        return None
    
    def _scan_processes(self) -> List[GameProcess]:
        """Scan for game processes"""
        detected_games = []
        current_time = time.time()
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'cpu_percent', 'memory_info']):
                proc_info = proc.info
                
                if self._is_game_process(proc_info):
                    try:
                        memory_mb = proc_info['memory_info'].rss / (1024 * 1024)
                        
                        game_proc = GameProcess(
                            pid=proc_info['pid'],
                            name=proc_info['name'],
                            exe_path=proc_info.get('exe', ''),
                            cpu_percent=proc_info.get('cpu_percent', 0),
                            memory_mb=memory_mb,
                            gpu_usage=0,  # Per-process GPU usage not easily available
                            is_steam_game=self._is_steam_process(proc_info['name']),
                            launch_time=current_time
                        )
                        
                        detected_games.append(game_proc)
                        
                    except Exception as e:
                        self.logger.debug(f"Error processing game process {proc_info['pid']}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Process scanning error: {e}")
        
        return detected_games
    
    def _determine_gaming_state(self, games: List[GameProcess], resource_usage: ResourceUsage) -> GamingState:
        """Determine current gaming state based on detected games and resource usage"""
        
        # Check Steam-specific state first
        steam_state = self._detect_steam_state()
        if steam_state:
            return steam_state
        
        # No games detected
        if not games:
            return GamingState.IDLE
        
        # Check if any high-resource games are running
        high_resource_games = [
            game for game in games 
            if game.cpu_percent > 30 or game.memory_mb > 500
        ]
        
        if high_resource_games:
            # Determine 2D vs 3D based on GPU usage
            if resource_usage.gpu_percent > self.gpu_threshold_3d:
                return GamingState.ACTIVE_3D
            elif resource_usage.gpu_percent > self.gpu_threshold_2d:
                return GamingState.ACTIVE_2D
            else:
                # High CPU/memory but low GPU - could be launching
                return GamingState.LAUNCHING
        
        # Low resource games or background Steam processes
        if any(game.is_steam_game for game in games):
            return GamingState.ACTIVE_2D
        
        return GamingState.IDLE
    
    def start_monitoring(self, interval: float = 2.0):
        """Start gaming state monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._shutdown_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            name="gaming_detector",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Gaming state monitoring started")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Scan for games and get resource usage
                detected_games = self._scan_processes()
                resource_usage = self._get_resource_usage()
                
                # Determine gaming state
                new_state = self._determine_gaming_state(detected_games, resource_usage)
                
                # Update state if changed
                with self._state_lock:
                    if new_state != self.current_state:
                        self.logger.info(f"Gaming state changed: {self.current_state.value} -> {new_state.value}")
                        self.current_state = new_state
                        self.last_state_change = time.time()
                    
                    # Update detected games
                    self.detected_games = {game.pid: game for game in detected_games}
                    
                    # Update history
                    self.state_history.append((start_time, new_state.value, resource_usage))
                    if len(self.state_history) > self.max_history_size:
                        self.state_history.pop(0)
                
                # Sleep for the remaining interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Gaming detection error: {e}")
                time.sleep(interval)
    
    def stop_monitoring(self):
        """Stop gaming state monitoring"""
        if self._monitoring_thread:
            self._shutdown_event.set()
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("Gaming state monitoring stopped")
    
    def get_gaming_state(self) -> GamingState:
        """Get current gaming state"""
        with self._state_lock:
            return self.current_state
    
    def is_gaming_active(self) -> bool:
        """Check if gaming is currently active"""
        return self.get_gaming_state() in [
            GamingState.ACTIVE_2D, 
            GamingState.ACTIVE_3D, 
            GamingState.LAUNCHING
        ]
    
    def get_current_games(self) -> List[GameProcess]:
        """Get currently detected game processes"""
        with self._state_lock:
            return list(self.detected_games.values())
    
    def get_resource_recommendations(self) -> Dict[str, Any]:
        """Get resource allocation recommendations based on gaming state"""
        state = self.get_gaming_state()
        
        recommendations = {
            "background_threads": 6,
            "ml_threads": 2,
            "compilation_threads": 2,
            "cache_size_mb": 64,
            "thermal_monitoring_interval": 5.0,
            "enable_prediction_cache": True,
            "priority_gaming": False
        }
        
        if state == GamingState.ACTIVE_3D:
            # Demanding 3D game - minimal background work
            recommendations.update({
                "background_threads": 2,
                "ml_threads": 1,
                "compilation_threads": 0,  # Pause compilation
                "cache_size_mb": 32,
                "thermal_monitoring_interval": 2.0,
                "priority_gaming": True
            })
        
        elif state == GamingState.ACTIVE_2D:
            # Light 2D game - reduced background work
            recommendations.update({
                "background_threads": 3,
                "ml_threads": 1,
                "compilation_threads": 1,
                "cache_size_mb": 48,
                "thermal_monitoring_interval": 3.0,
                "priority_gaming": True
            })
        
        elif state == GamingState.LAUNCHING:
            # Game launching - minimal interference
            recommendations.update({
                "background_threads": 2,
                "ml_threads": 1,
                "compilation_threads": 0,
                "cache_size_mb": 32,
                "thermal_monitoring_interval": 1.0,
                "priority_gaming": True
            })
        
        elif state == GamingState.STEAM_OVERLAY:
            # Steam overlay active - medium restrictions
            recommendations.update({
                "background_threads": 4,
                "ml_threads": 2,
                "compilation_threads": 1,
                "cache_size_mb": 48,
                "thermal_monitoring_interval": 2.0,
                "priority_gaming": True
            })
        
        # Idle state uses default recommendations
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive gaming detection status"""
        with self._state_lock:
            games_info = [
                {
                    "pid": game.pid,
                    "name": game.name,
                    "cpu_percent": game.cpu_percent,
                    "memory_mb": game.memory_mb,
                    "is_steam_game": game.is_steam_game
                }
                for game in self.detected_games.values()
            ]
            
            resource_usage = self._get_resource_usage()
            
            return {
                "gaming_state": self.current_state.value,
                "is_gaming_active": self.is_gaming_active(),
                "detected_games_count": len(self.detected_games),
                "detected_games": games_info,
                "resource_usage": {
                    "cpu_percent": resource_usage.cpu_percent,
                    "memory_percent": resource_usage.memory_percent,
                    "gpu_percent": resource_usage.gpu_percent,
                    "thermal_temp": resource_usage.thermal_temp,
                    "battery_percent": resource_usage.battery_percent,
                    "is_charging": resource_usage.is_charging
                },
                "recommendations": self.get_resource_recommendations(),
                "last_state_change": self.last_state_change,
                "monitoring_active": self._monitoring_thread is not None and self._monitoring_thread.is_alive()
            }


# Global gaming detector instance
_gaming_detector = None

def get_gaming_detector() -> SteamDeckGamingDetector:
    """Get or create global gaming detector"""
    global _gaming_detector
    if _gaming_detector is None:
        _gaming_detector = SteamDeckGamingDetector()
    return _gaming_detector


if __name__ == "__main__":
    # Test gaming detector
    logging.basicConfig(level=logging.INFO)
    
    print("🎮 Steam Deck Gaming Detector Test")
    print("=" * 40)
    
    detector = get_gaming_detector()
    
    # Start monitoring
    detector.start_monitoring()
    
    try:
        # Monitor for a few seconds
        for i in range(5):
            time.sleep(2)
            status = detector.get_status()
            
            print(f"\nScan {i+1}:")
            print(f"  Gaming State: {status['gaming_state']}")
            print(f"  Detected Games: {status['detected_games_count']}")
            print(f"  GPU Usage: {status['resource_usage']['gpu_percent']}%")
            print(f"  Gaming Active: {status['is_gaming_active']}")
            
            if status['detected_games']:
                print("  Games:")
                for game in status['detected_games']:
                    print(f"    {game['name']} (PID: {game['pid']}, CPU: {game['cpu_percent']:.1f}%)")
    
    finally:
        detector.stop_monitoring()
        print("\n✅ Gaming detector test completed")