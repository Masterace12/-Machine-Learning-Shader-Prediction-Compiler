#!/usr/bin/env python3
"""
Steam Deck Test Fixtures and Mock Hardware
Comprehensive mocking system for Steam Deck hardware components
"""

import os
import time
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager

import pytest


class MockSteamDeckModel(Enum):
    """Mock Steam Deck models for testing"""
    LCD_64GB = "lcd_64gb"
    LCD_256GB = "lcd_256gb" 
    LCD_512GB = "lcd_512gb"
    OLED_512GB = "oled_512gb"
    OLED_1TB = "oled_1tb"


@dataclass
class MockHardwareState:
    """Mock hardware state for testing"""
    model: MockSteamDeckModel = MockSteamDeckModel.LCD_256GB
    cpu_temperature: float = 65.0
    gpu_temperature: float = 70.0
    battery_capacity: int = 75
    battery_status: str = "Discharging"
    power_draw: int = 12000000  # microWatts
    fan_speed: int = 2500
    cpu_frequency: int = 2800000  # kHz
    gpu_frequency: int = 1600
    thermal_throttling: bool = False
    gaming_mode_active: bool = False
    docked: bool = False
    external_displays: List[str] = field(default_factory=list)
    
    # Thermal zones
    thermal_zones: Dict[str, float] = field(default_factory=lambda: {
        'thermal_zone0': 65000,  # CPU in millicelsius
        'thermal_zone1': 70000,  # GPU in millicelsius
        'thermal_zone2': 45000,  # Battery
    })
    
    # Power supply info
    power_supply_info: Dict[str, Any] = field(default_factory=lambda: {
        'BAT1': {
            'capacity': 75,
            'status': 'Discharging',
            'power_now': 12000000,
            'voltage_now': 7400000,
            'current_now': 1620000
        }
    })
    
    # DRM display info
    drm_displays: Dict[str, str] = field(default_factory=lambda: {
        'card0-eDP-1': 'connected',
        'card0-DP-1': 'disconnected',
        'card0-DP-2': 'disconnected', 
        'card0-HDMI-A-1': 'disconnected'
    })


class MockFilesystemManager:
    """Manages mock filesystem for Steam Deck hardware simulation"""
    
    def __init__(self, temp_dir: Path, hardware_state: MockHardwareState):
        self.temp_dir = temp_dir
        self.hardware_state = hardware_state
        self._setup_mock_filesystem()
    
    def _setup_mock_filesystem(self):
        """Create mock filesystem structure"""
        # Thermal zones
        thermal_dir = self.temp_dir / "sys/class/thermal"
        thermal_dir.mkdir(parents=True)
        
        for i, (zone, temp) in enumerate(self.hardware_state.thermal_zones.items()):
            zone_dir = thermal_dir / f"thermal_zone{i}"
            zone_dir.mkdir()
            (zone_dir / "temp").write_text(str(temp))
            (zone_dir / "type").write_text(zone)
        
        # Power supply
        power_dir = self.temp_dir / "sys/class/power_supply"
        power_dir.mkdir(parents=True)
        
        for name, info in self.hardware_state.power_supply_info.items():
            supply_dir = power_dir / name
            supply_dir.mkdir()
            for key, value in info.items():
                (supply_dir / key).write_text(str(value))
        
        # DMI info
        dmi_dir = self.temp_dir / "sys/devices/virtual/dmi/id"
        dmi_dir.mkdir(parents=True)
        
        model_map = {
            MockSteamDeckModel.LCD_64GB: "Jupiter",
            MockSteamDeckModel.LCD_256GB: "Jupiter", 
            MockSteamDeckModel.LCD_512GB: "Jupiter",
            MockSteamDeckModel.OLED_512GB: "Galileo",
            MockSteamDeckModel.OLED_1TB: "Galileo"
        }
        
        product_name = model_map[self.hardware_state.model]
        (dmi_dir / "product_name").write_text(product_name)
        (dmi_dir / "board_name").write_text(product_name)
        (dmi_dir / "sys_vendor").write_text("Valve")
        
        # DRM displays
        drm_dir = self.temp_dir / "sys/class/drm"
        drm_dir.mkdir(parents=True)
        
        for display, status in self.hardware_state.drm_displays.items():
            display_dir = drm_dir / display
            display_dir.mkdir()
            (display_dir / "status").write_text(status)
        
        # CPU frequency
        cpu_dir = self.temp_dir / "sys/devices/system/cpu/cpu0/cpufreq"
        cpu_dir.mkdir(parents=True)
        (cpu_dir / "scaling_cur_freq").write_text(str(self.hardware_state.cpu_frequency))
        (cpu_dir / "scaling_governor").write_text("schedutil")
        
        # GPU frequency (AMD specific)
        gpu_dir = self.temp_dir / "sys/class/drm/card0/device"
        gpu_dir.mkdir(parents=True)
        gpu_freq_content = f"0: 200Mhz\n1: 400Mhz\n2: 800Mhz\n3: {self.hardware_state.gpu_frequency}Mhz *\n"
        (gpu_dir / "pp_dpm_sclk").write_text(gpu_freq_content)
        
        # Fan speed
        hwmon_dir = self.temp_dir / "sys/class/hwmon/hwmon0"
        hwmon_dir.mkdir(parents=True)
        (hwmon_dir / "fan1_input").write_text(str(self.hardware_state.fan_speed))
        
        # Memory info
        proc_dir = self.temp_dir / "proc"
        proc_dir.mkdir(parents=True)
        meminfo_content = """MemTotal:       16384000 kB
MemFree:         8192000 kB
MemAvailable:   12288000 kB
Buffers:          512000 kB
Cached:          2048000 kB
SwapCached:            0 kB
"""
        (proc_dir / "meminfo").write_text(meminfo_content)
        
        # CPU info
        cpuinfo_content = """processor       : 0
vendor_id       : AuthenticAMD
cpu family      : 23
model           : 144
model name      : AMD Custom APU 0405
stepping        : 1
microcode       : 0x0
cpu MHz         : 2800.000
cache size      : 512 KB
"""
        (proc_dir / "cpuinfo").write_text(cpuinfo_content)
    
    def update_hardware_state(self, new_state: MockHardwareState):
        """Update mock hardware state and filesystem"""
        self.hardware_state = new_state
        self._setup_mock_filesystem()


class MockDBusInterface:
    """Mock D-Bus interface for Steam integration testing"""
    
    def __init__(self):
        self.gaming_mode_active = False
        self.steam_processes = []
        self.steam_apps = {}
        self.callbacks = []
    
    def is_gaming_mode_active(self) -> bool:
        return self.gaming_mode_active
    
    def get_running_apps(self) -> List[Dict[str, Any]]:
        return list(self.steam_apps.values())
    
    def set_gaming_mode(self, active: bool):
        self.gaming_mode_active = active
        self._notify_callbacks('gaming_mode_changed', {'active': active})
    
    def add_steam_app(self, app_id: int, name: str, is_running: bool = True):
        self.steam_apps[app_id] = {
            'app_id': app_id,
            'name': name,
            'is_running': is_running,
            'cpu_usage': 25.0,
            'memory_usage': 512.0
        }
        self._notify_callbacks('app_started', {'app_id': app_id, 'name': name})
    
    def remove_steam_app(self, app_id: int):
        if app_id in self.steam_apps:
            app_info = self.steam_apps.pop(app_id)
            self._notify_callbacks('app_stopped', {'app_id': app_id})
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception:
                pass


@pytest.fixture
def mock_hardware_state():
    """Provide a default mock hardware state"""
    return MockHardwareState()


@pytest.fixture
def mock_steamdeck_lcd():
    """Mock LCD Steam Deck"""
    return MockHardwareState(
        model=MockSteamDeckModel.LCD_256GB,
        cpu_temperature=65.0,
        battery_capacity=80
    )


@pytest.fixture
def mock_steamdeck_oled():
    """Mock OLED Steam Deck"""
    return MockHardwareState(
        model=MockSteamDeckModel.OLED_512GB,
        cpu_temperature=62.0,
        battery_capacity=85,
        thermal_zones={
            'thermal_zone0': 62000,
            'thermal_zone1': 67000, 
            'thermal_zone2': 40000
        }
    )


@pytest.fixture
def mock_steamdeck_gaming():
    """Mock Steam Deck in gaming mode"""
    return MockHardwareState(
        model=MockSteamDeckModel.LCD_512GB,
        cpu_temperature=75.0,
        gaming_mode_active=True,
        power_draw=18000000,
        fan_speed=3500
    )


@pytest.fixture
def mock_steamdeck_docked():
    """Mock docked Steam Deck"""
    return MockHardwareState(
        model=MockSteamDeckModel.OLED_1TB,
        cpu_temperature=70.0,
        docked=True,
        external_displays=['DP-1'],
        power_supply_info={
            'BAT1': {
                'capacity': 100,
                'status': 'Full',
                'power_now': 0,
                'voltage_now': 8400000,
                'current_now': 0
            }
        },
        drm_displays={
            'card0-eDP-1': 'connected',
            'card0-DP-1': 'connected',
            'card0-DP-2': 'disconnected',
            'card0-HDMI-A-1': 'disconnected'
        }
    )


@pytest.fixture
def mock_steamdeck_thermal_throttling():
    """Mock Steam Deck under thermal throttling"""
    return MockHardwareState(
        model=MockSteamDeckModel.LCD_256GB,
        cpu_temperature=88.0,
        thermal_throttling=True,
        cpu_frequency=2200000,  # Reduced frequency
        fan_speed=4500,  # High fan speed
        thermal_zones={
            'thermal_zone0': 88000,
            'thermal_zone1': 92000,
            'thermal_zone2': 55000
        }
    )


@pytest.fixture
def mock_steamdeck_low_battery():
    """Mock Steam Deck with low battery"""
    return MockHardwareState(
        model=MockSteamDeckModel.LCD_64GB,
        battery_capacity=15,
        power_draw=8000000,  # Low power draw
        cpu_temperature=58.0,  # Cooler due to throttling
        power_supply_info={
            'BAT1': {
                'capacity': 15,
                'status': 'Discharging',
                'power_now': 8000000,
                'voltage_now': 6800000,
                'current_now': 1176000
            }
        }
    )


@pytest.fixture
def mock_filesystem(tmp_path, mock_hardware_state):
    """Create mock Steam Deck filesystem"""
    fs_manager = MockFilesystemManager(tmp_path, mock_hardware_state)
    return fs_manager


@pytest.fixture
def mock_dbus_interface():
    """Mock D-Bus interface for Steam integration"""
    return MockDBusInterface()


@contextmanager
def mock_steamdeck_environment(hardware_state: MockHardwareState, temp_dir: Optional[Path] = None):
    """Context manager for complete Steam Deck environment mocking"""
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
    
    fs_manager = MockFilesystemManager(temp_dir, hardware_state)
    dbus_mock = MockDBusInterface()
    
    # Patch system paths
    patches = [
        patch.dict(os.environ, {'SteamDeck': '1'}),
        patch('os.path.exists', side_effect=lambda path: _mock_path_exists(path, temp_dir)),
        patch('pathlib.Path.exists', side_effect=lambda self: _mock_path_exists(str(self), temp_dir)),
        patch('pathlib.Path.read_text', side_effect=lambda self: _mock_read_text(str(self), temp_dir)),
        patch('builtins.open', side_effect=lambda path, *args, **kwargs: _mock_open(path, temp_dir, *args, **kwargs)),
        patch('subprocess.run', side_effect=_mock_subprocess_run),
        patch('psutil.process_iter', side_effect=lambda: _mock_process_iter(hardware_state)),
    ]
    
    try:
        # Apply all patches
        for p in patches:
            p.start()
        
        # Set gaming mode if specified
        if hardware_state.gaming_mode_active:
            dbus_mock.set_gaming_mode(True)
            dbus_mock.add_steam_app(12345, "Test Game")
        
        yield {
            'filesystem': fs_manager,
            'dbus': dbus_mock,
            'temp_dir': temp_dir
        }
    
    finally:
        # Clean up patches
        for p in reversed(patches):
            try:
                p.stop()
            except Exception:
                pass


def _mock_path_exists(path: str, temp_dir: Path) -> bool:
    """Mock path.exists() calls"""
    if path.startswith('/sys') or path.startswith('/proc'):
        mock_path = temp_dir / path.lstrip('/')
        return mock_path.exists()
    elif path == '/home/deck':
        return True
    elif 'thermal' in path or 'power_supply' in path or 'dmi' in path:
        mock_path = temp_dir / path.lstrip('/')
        return mock_path.exists()
    else:
        return os.path.exists(path)


def _mock_read_text(path: str, temp_dir: Path) -> str:
    """Mock file reading for hardware info"""
    if path.startswith('/sys') or path.startswith('/proc'):
        mock_path = temp_dir / path.lstrip('/')
        if mock_path.exists():
            return mock_path.read_text()
    raise FileNotFoundError(f"Mock file not found: {path}")


def _mock_open(path: str, temp_dir: Path, *args, **kwargs):
    """Mock file opening"""
    if path.startswith('/sys') or path.startswith('/proc'):
        mock_path = temp_dir / path.lstrip('/')
        if mock_path.exists():
            return open(mock_path, *args, **kwargs)
    return open(path, *args, **kwargs)


def _mock_subprocess_run(args, **kwargs):
    """Mock subprocess.run calls"""
    if args[0] == 'pgrep':
        if 'gamescope' in args:
            # Mock gaming mode detection
            return Mock(returncode=0, stdout="1234\n")
        elif 'steam' in args:
            return Mock(returncode=0, stdout="5678\n9012\n")
    elif args[0] == 'lsblk':
        # Mock storage size detection
        return Mock(returncode=0, stdout="256060514304\n")  # 256GB
    
    return Mock(returncode=1, stdout="", stderr="")


def _mock_process_iter(hardware_state: MockHardwareState):
    """Mock psutil.process_iter for game detection"""
    processes = []
    
    # Always include some system processes
    processes.append(Mock(
        pid=1,
        name="systemd",
        exe="/usr/lib/systemd/systemd",
        cpu_percent=lambda: 0.1,
        memory_info=lambda: Mock(rss=1024*1024)
    ))
    
    # Add Steam processes if gaming
    if hardware_state.gaming_mode_active:
        processes.append(Mock(
            pid=1234,
            name="gamescope",
            exe="/usr/bin/gamescope",
            cpu_percent=lambda: 15.0,
            memory_info=lambda: Mock(rss=128*1024*1024)
        ))
        
        processes.append(Mock(
            pid=5678,
            name="TestGame.exe",
            exe="/home/deck/.steam/steamapps/common/TestGame/TestGame.exe",
            cpu_percent=lambda: 45.0,
            memory_info=lambda: Mock(rss=512*1024*1024)
        ))
    
    return processes


# Utility functions for test scenarios

def create_thermal_stress_scenario(base_state: MockHardwareState) -> MockHardwareState:
    """Create a thermal stress test scenario"""
    base_state.cpu_temperature = 92.0
    base_state.thermal_throttling = True
    base_state.fan_speed = 4800
    base_state.cpu_frequency = 2000000  # Throttled
    base_state.thermal_zones = {
        'thermal_zone0': 92000,
        'thermal_zone1': 95000,
        'thermal_zone2': 58000
    }
    return base_state


def create_battery_critical_scenario(base_state: MockHardwareState) -> MockHardwareState:
    """Create a critical battery scenario"""
    base_state.battery_capacity = 8
    base_state.power_draw = 6000000  # Very low power
    base_state.cpu_temperature = 55.0  # Cool due to aggressive throttling
    base_state.power_supply_info['BAT1']['capacity'] = 8
    base_state.power_supply_info['BAT1']['power_now'] = 6000000
    return base_state


def create_intensive_gaming_scenario(base_state: MockHardwareState) -> MockHardwareState:
    """Create an intensive gaming scenario"""
    base_state.gaming_mode_active = True
    base_state.cpu_temperature = 82.0
    base_state.power_draw = 16000000
    base_state.fan_speed = 4000
    base_state.cpu_frequency = 3200000  # Boosted
    base_state.gpu_frequency = 1700  # High GPU frequency
    return base_state


def create_dock_scenario(base_state: MockHardwareState) -> MockHardwareState:
    """Create a docked scenario with external display"""
    base_state.docked = True
    base_state.external_displays = ['HDMI-A-1']
    base_state.power_supply_info['BAT1']['status'] = 'Charging'
    base_state.power_supply_info['BAT1']['capacity'] = 95
    base_state.drm_displays['card0-HDMI-A-1'] = 'connected'
    return base_state


# Performance test utilities

class PerformanceTimer:
    """Simple performance timer for benchmarks"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


def benchmark_test(iterations: int = 100):
    """Decorator for benchmark tests"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(iterations):
                with PerformanceTimer() as timer:
                    result = func(*args, **kwargs)
                times.append(timer.elapsed)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\nBenchmark Results for {func.__name__}:")
            print(f"  Iterations: {iterations}")
            print(f"  Average: {avg_time*1000:.2f}ms")
            print(f"  Min: {min_time*1000:.2f}ms")
            print(f"  Max: {max_time*1000:.2f}ms")
            
            return result
        return wrapper
    return decorator


# Steam Deck specific test markers

steamdeck_hardware_required = pytest.mark.skipif(
    not os.path.exists('/home/deck'), 
    reason="Steam Deck hardware required"
)

steamdeck_lcd_only = pytest.mark.skipif(
    not (os.path.exists('/sys/devices/virtual/dmi/id/product_name') and 
         'jupiter' in open('/sys/devices/virtual/dmi/id/product_name').read().lower()),
    reason="LCD Steam Deck required"
)

steamdeck_oled_only = pytest.mark.skipif(
    not (os.path.exists('/sys/devices/virtual/dmi/id/product_name') and 
         'galileo' in open('/sys/devices/virtual/dmi/id/product_name').read().lower()),
    reason="OLED Steam Deck required"
)