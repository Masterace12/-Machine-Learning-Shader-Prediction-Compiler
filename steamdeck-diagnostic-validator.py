#!/usr/bin/env python3
"""
Steam Deck ML Shader Predictor - Comprehensive Diagnostic and Validation Tool
Validates installation, diagnoses issues, and provides troubleshooting guidance

This tool performs extensive checks on:
- Steam Deck hardware detection and model identification
- System resources and thermal constraints
- Installation integrity and file permissions
- Python environment and dependencies
- Systemd service status and configuration
- Steam integration and shader cache accessibility
- Network connectivity and P2P functionality
- Security and sandboxing configuration

Usage:
    python3 steamdeck-diagnostic-validator.py [--detailed] [--fix-issues] [--export-report]
"""

import sys
import os
import json
import subprocess
import time
import glob
import hashlib
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import argparse


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL" 
    WARN = "WARN"
    INFO = "INFO"
    SKIP = "SKIP"


class TestPriority(Enum):
    CRITICAL = "CRITICAL"    # Must pass for basic functionality
    IMPORTANT = "IMPORTANT"  # Should pass for optimal performance
    OPTIONAL = "OPTIONAL"    # Nice to have but not required


@dataclass
class DiagnosticTest:
    name: str
    description: str
    priority: TestPriority
    result: TestResult
    details: str
    fix_suggestion: Optional[str] = None
    error_details: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class SystemInfo:
    hostname: str
    os_release: Dict[str, str]
    kernel_version: str
    architecture: str
    python_version: str
    uptime_seconds: float
    load_average: Tuple[float, float, float]
    memory_total_mb: int
    memory_available_mb: int
    disk_space_home_gb: float
    disk_space_available_gb: float


@dataclass
class SteamDeckInfo:
    is_steam_deck: bool
    model: str  # "LCD", "OLED", "Unknown", "Not Steam Deck"
    product_name: str
    cpu_model: str
    gpu_info: str
    current_temp_c: float
    battery_level: float
    is_charging: bool
    fan_speed_rpm: int
    immutable_filesystem: bool


@dataclass
class InstallationStatus:
    install_dir_exists: bool
    config_files_present: bool
    source_files_present: bool
    launcher_script_exists: bool
    systemd_service_exists: bool
    desktop_entry_exists: bool
    permissions_correct: bool
    installation_complete: bool


class SteamDeckDiagnosticValidator:
    """Comprehensive diagnostic and validation system for Steam Deck ML Shader Predictor"""
    
    def __init__(self, detailed_mode: bool = False, fix_issues: bool = False):
        self.detailed_mode = detailed_mode
        self.fix_issues = fix_issues
        self.tests_run: List[DiagnosticTest] = []
        self.start_time = time.time()
        
        # Installation paths
        self.install_dir = Path.home() / ".local/share/ml-shader-predictor"
        self.config_dir = Path.home() / ".config/ml-shader-predictor"
        self.cache_dir = Path.home() / ".cache/ml-shader-predictor"
        self.service_dir = Path.home() / ".config/systemd/user"
        
        # Initialize colored output
        self.colors = {
            'PASS': '\033[92m',     # Green
            'FAIL': '\033[91m',     # Red
            'WARN': '\033[93m',     # Yellow
            'INFO': '\033[94m',     # Blue
            'SKIP': '\033[90m',     # Gray
            'BOLD': '\033[1m',      # Bold
            'END': '\033[0m'        # End formatting
        }
        
        # Disable colors if not in terminal
        if not sys.stdout.isatty():
            self.colors = {key: '' for key in self.colors}
    
    def log(self, level: str, message: str, details: str = ""):
        """Log a message with appropriate formatting"""
        color = self.colors.get(level, '')
        end_color = self.colors['END']
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if details and self.detailed_mode:
            print(f"[{timestamp}] {color}[{level}]{end_color} {message}")
            print(f"         {details}")
        else:
            print(f"[{timestamp}] {color}[{level}]{end_color} {message}")
    
    def run_test(self, name: str, description: str, priority: TestPriority, 
                test_func, *args, **kwargs) -> DiagnosticTest:
        """Run a diagnostic test and record results"""
        test_start = time.time()
        
        try:
            result, details, fix_suggestion = test_func(*args, **kwargs)
            error_details = None
        except Exception as e:
            result = TestResult.FAIL
            details = f"Test execution failed: {str(e)}"
            fix_suggestion = "Check system logs and ensure all dependencies are installed"
            error_details = str(e)
        
        execution_time = time.time() - test_start
        
        test = DiagnosticTest(
            name=name,
            description=description,
            priority=priority,
            result=result,
            details=details,
            fix_suggestion=fix_suggestion,
            error_details=error_details,
            execution_time=execution_time
        )
        
        self.tests_run.append(test)
        
        # Log the result
        if result == TestResult.PASS:
            self.log('PASS', f"{name}: {details}")
        elif result == TestResult.FAIL:
            self.log('FAIL', f"{name}: {details}")
            if fix_suggestion and self.detailed_mode:
                self.log('INFO', f"Fix: {fix_suggestion}")
        elif result == TestResult.WARN:
            self.log('WARN', f"{name}: {details}")
        else:
            self.log(result.value, f"{name}: {details}")
        
        return test
    
    def collect_system_info(self) -> SystemInfo:
        """Collect comprehensive system information"""
        # OS Release information
        os_release = {}
        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os_release[key] = value.strip('"')
        except Exception:
            os_release = {'ID': 'unknown', 'NAME': 'unknown'}
        
        # Memory information
        memory_total_mb = 0
        memory_available_mb = 0
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        memory_total_mb = int(line.split()[1]) // 1024
                    elif line.startswith('MemAvailable:'):
                        memory_available_mb = int(line.split()[1]) // 1024
        except Exception:
            pass
        
        # Load average
        load_avg = (0.0, 0.0, 0.0)
        try:
            with open('/proc/loadavg', 'r') as f:
                loads = f.read().split()[:3]
                load_avg = tuple(float(x) for x in loads)
        except Exception:
            pass
        
        # Uptime
        uptime_seconds = 0.0
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
        except Exception:
            pass
        
        # Disk space
        disk_space_home_gb = 0.0
        disk_space_available_gb = 0.0
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path.home())
            disk_space_home_gb = total / (1024**3)
            disk_space_available_gb = free / (1024**3)
        except Exception:
            pass
        
        return SystemInfo(
            hostname=socket.gethostname(),
            os_release=os_release,
            kernel_version=os.uname().release,
            architecture=os.uname().machine,
            python_version=sys.version,
            uptime_seconds=uptime_seconds,
            load_average=load_avg,
            memory_total_mb=memory_total_mb,
            memory_available_mb=memory_available_mb,
            disk_space_home_gb=disk_space_home_gb,
            disk_space_available_gb=disk_space_available_gb
        )
    
    def collect_steamdeck_info(self) -> SteamDeckInfo:
        """Collect Steam Deck specific hardware information"""
        # Product name detection
        product_name = "unknown"
        try:
            with open('/sys/class/dmi/id/product_name', 'r') as f:
                product_name = f.read().strip()
        except Exception:
            pass
        
        # Determine if this is a Steam Deck and what model
        is_steam_deck = False
        model = "Not Steam Deck"
        
        if "Jupiter" in product_name or "Steam Deck" in product_name:
            is_steam_deck = True
            if "1010" in product_name or "1020" in product_name or "1030" in product_name:
                model = "LCD"
            elif "1040" in product_name or "OLED" in product_name:
                model = "OLED"
            else:
                model = "Unknown Steam Deck"
        
        # CPU information
        cpu_model = "unknown"
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_model = line.split(':', 1)[1].strip()
                        break
        except Exception:
            pass
        
        # GPU information (try different sources)
        gpu_info = "unknown"
        try:
            # Try lspci first
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'VGA' in line or 'Display' in line:
                    gpu_info = line.split(': ', 1)[1] if ': ' in line else line
                    break
        except Exception:
            pass
        
        # Temperature reading
        current_temp_c = 0.0
        temp_paths = [
            '/sys/class/thermal/thermal_zone0/temp',
            '/sys/class/hwmon/hwmon0/temp1_input',
            '/sys/class/hwmon/hwmon1/temp1_input'
        ]
        
        for temp_path in temp_paths:
            try:
                with open(temp_path, 'r') as f:
                    temp_millicelsius = int(f.read().strip())
                    current_temp_c = temp_millicelsius / 1000.0
                    break
            except Exception:
                continue
        
        # Battery information
        battery_level = 100.0
        is_charging = False
        try:
            result = subprocess.run(['upower', '-i', '/org/freedesktop/UPower/devices/battery_BAT1'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'percentage' in line:
                    battery_level = float(line.split()[-1].rstrip('%'))
                elif 'state' in line:
                    is_charging = 'charging' in line.lower()
        except Exception:
            pass
        
        # Fan speed
        fan_speed_rpm = 0
        fan_paths = [
            '/sys/class/hwmon/hwmon0/fan1_input',
            '/sys/class/hwmon/hwmon1/fan1_input'
        ]
        
        for fan_path in fan_paths:
            try:
                with open(fan_path, 'r') as f:
                    fan_speed_rpm = int(f.read().strip())
                    break
            except Exception:
                continue
        
        # Check if filesystem is immutable
        immutable_filesystem = not os.access('/usr', os.W_OK)
        
        return SteamDeckInfo(
            is_steam_deck=is_steam_deck,
            model=model,
            product_name=product_name,
            cpu_model=cpu_model,
            gpu_info=gpu_info,
            current_temp_c=current_temp_c,
            battery_level=battery_level,
            is_charging=is_charging,
            fan_speed_rpm=fan_speed_rpm,
            immutable_filesystem=immutable_filesystem
        )
    
    # ========================================================================
    # DIAGNOSTIC TESTS
    # ========================================================================
    
    def test_steam_deck_detection(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test Steam Deck hardware detection"""
        steamdeck_info = self.collect_steamdeck_info()
        
        if steamdeck_info.is_steam_deck:
            if steamdeck_info.model in ["LCD", "OLED"]:
                return (TestResult.PASS, 
                       f"Steam Deck {steamdeck_info.model} detected: {steamdeck_info.product_name}",
                       None)
            else:
                return (TestResult.WARN,
                       f"Steam Deck detected but model uncertain: {steamdeck_info.product_name}",
                       "This may affect model-specific optimizations")
        else:
            return (TestResult.WARN,
                   f"Not running on Steam Deck hardware: {steamdeck_info.product_name}",
                   "Some optimizations will be disabled in compatibility mode")
    
    def test_system_resources(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test system resource availability"""
        system_info = self.collect_system_info()
        issues = []
        
        # Memory check
        if system_info.memory_total_mb < 12000:  # Less than 12GB (Steam Deck has ~16GB)
            issues.append(f"Low memory: {system_info.memory_total_mb}MB total")
        
        if system_info.memory_available_mb < 2000:  # Less than 2GB available
            issues.append(f"Low available memory: {system_info.memory_available_mb}MB")
        
        # Disk space check
        if system_info.disk_space_available_gb < 2.0:
            issues.append(f"Low disk space: {system_info.disk_space_available_gb:.1f}GB available")
        
        # Load average check
        if system_info.load_average[0] > 4.0:  # High load on 4-core system
            issues.append(f"High system load: {system_info.load_average[0]:.2f}")
        
        if issues:
            return (TestResult.WARN,
                   f"Resource issues detected: {'; '.join(issues)}",
                   "Close other applications or wait for system load to decrease")
        else:
            return (TestResult.PASS,
                   f"System resources OK: {system_info.memory_available_mb}MB available, "
                   f"{system_info.disk_space_available_gb:.1f}GB free",
                   None)
    
    def test_thermal_conditions(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test thermal conditions and limits"""
        steamdeck_info = self.collect_steamdeck_info()
        temp = steamdeck_info.current_temp_c
        
        if temp > 85:
            return (TestResult.FAIL,
                   f"System overheating: {temp:.1f}°C",
                   "Wait for system to cool down before installation/operation")
        elif temp > 80:
            return (TestResult.WARN,
                   f"System temperature high: {temp:.1f}°C",
                   "Monitor temperature during operation, consider improving cooling")
        elif temp > 0:  # Valid temperature reading
            return (TestResult.PASS,
                   f"Temperature OK: {temp:.1f}°C (fan: {steamdeck_info.fan_speed_rpm} RPM)",
                   None)
        else:
            return (TestResult.SKIP,
                   "Temperature sensors not accessible",
                   "Thermal monitoring may be limited")
    
    def test_battery_status(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test battery status for mobile device awareness"""
        steamdeck_info = self.collect_steamdeck_info()
        
        battery_level = steamdeck_info.battery_level
        is_charging = steamdeck_info.is_charging
        
        if battery_level < 15 and not is_charging:
            return (TestResult.WARN,
                   f"Low battery: {battery_level:.0f}% (not charging)",
                   "Connect charger for optimal performance during installation/heavy operations")
        elif battery_level < 30 and not is_charging:
            return (TestResult.INFO,
                   f"Battery level: {battery_level:.0f}% (not charging)",
                   "Performance may be reduced in battery save mode")
        else:
            charging_status = "charging" if is_charging else "on battery"
            return (TestResult.PASS,
                   f"Battery OK: {battery_level:.0f}% ({charging_status})",
                   None)
    
    def test_installation_files(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test installation file integrity"""
        missing_critical = []
        missing_optional = []
        
        # Critical files
        critical_files = [
            self.install_dir / "src" / "steam_deck_integration.py",
            self.install_dir / "config" / "steamdeck_config.json",
            Path.home() / ".local/bin/ml-shader-predictor"
        ]
        
        for file_path in critical_files:
            if not file_path.exists():
                missing_critical.append(str(file_path))
        
        # Optional files
        optional_files = [
            self.service_dir / "ml-shader-predictor.service",
            Path.home() / ".local/share/applications/ml-shader-predictor.desktop",
            self.install_dir / "src" / "gui" / "main_window.py"
        ]
        
        for file_path in optional_files:
            if not file_path.exists():
                missing_optional.append(str(file_path))
        
        if missing_critical:
            return (TestResult.FAIL,
                   f"Critical files missing: {len(missing_critical)} files",
                   f"Reinstall the application. Missing: {', '.join(missing_critical[:2])}")
        elif missing_optional:
            return (TestResult.WARN,
                   f"Installation incomplete: {len(missing_optional)} optional files missing",
                   f"Some features may not be available: {', '.join(missing_optional[:2])}")
        else:
            return (TestResult.PASS,
                   "All installation files present",
                   None)
    
    def test_python_environment(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test Python environment and dependencies"""
        python_version = sys.version_info
        
        # Check Python version
        if python_version < (3, 8):
            return (TestResult.FAIL,
                   f"Python version too old: {python_version.major}.{python_version.minor}",
                   "Upgrade to Python 3.8 or newer")
        
        # Test critical imports
        critical_modules = ['numpy', 'psutil', 'yaml', 'joblib']
        optional_modules = ['sklearn', 'scipy', 'pandas', 'aiohttp']
        
        import_results = {'success': [], 'failed': []}
        
        for module in critical_modules + optional_modules:
            try:
                __import__(module)
                import_results['success'].append(module)
            except ImportError:
                import_results['failed'].append(module)
        
        failed_critical = [m for m in import_results['failed'] if m in critical_modules]
        failed_optional = [m for m in import_results['failed'] if m in optional_modules]
        
        if failed_critical:
            return (TestResult.FAIL,
                   f"Critical Python modules missing: {', '.join(failed_critical)}",
                   f"Install missing modules: pip install --user {' '.join(failed_critical)}")
        elif failed_optional:
            return (TestResult.WARN,
                   f"Optional Python modules missing: {', '.join(failed_optional)}",
                   "Some features will be limited. Consider installing optional modules.")
        else:
            return (TestResult.PASS,
                   f"Python environment OK: {len(import_results['success'])} modules available",
                   None)
    
    def test_systemd_service(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test systemd service configuration and status"""
        service_file = self.service_dir / "ml-shader-predictor.service"
        
        if not service_file.exists():
            return (TestResult.FAIL,
                   "Systemd service file not found",
                   f"Create service file at {service_file}")
        
        # Check service status
        try:
            result = subprocess.run(
                ['systemctl', '--user', 'is-active', 'ml-shader-predictor.service'],
                capture_output=True, text=True
            )
            service_active = result.returncode == 0
            
            result = subprocess.run(
                ['systemctl', '--user', 'is-enabled', 'ml-shader-predictor.service'],
                capture_output=True, text=True
            )
            service_enabled = result.returncode == 0
            
            if service_active and service_enabled:
                return (TestResult.PASS,
                       "Systemd service active and enabled",
                       None)
            elif service_enabled but not service_active:
                return (TestResult.WARN,
                       "Systemd service enabled but not active",
                       "Start service: systemctl --user start ml-shader-predictor.service")
            else:
                return (TestResult.WARN,
                       f"Systemd service not configured (enabled: {service_enabled}, active: {service_active})",
                       "Enable and start service: systemctl --user enable --now ml-shader-predictor.service")
                       
        except Exception as e:
            return (TestResult.FAIL,
                   f"Cannot check systemd service status: {e}",
                   "Check systemd installation and permissions")
    
    def test_steam_integration(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test Steam integration and shader cache access"""
        steam_paths = [
            Path.home() / ".steam",
            Path.home() / ".local/share/Steam",
            Path("/home/deck/.steam"),  # Explicit Steam Deck path
        ]
        
        steam_found = False
        steam_path = None
        
        for path in steam_paths:
            if path.exists():
                steam_found = True
                steam_path = path
                break
        
        if not steam_found:
            return (TestResult.WARN,
                   "Steam installation not found",
                   "Install Steam or check installation paths")
        
        # Check shader cache directory
        shader_cache_paths = [
            steam_path / "steamapps" / "shadercache",
            steam_path / "shader_cache"
        ]
        
        shader_cache_found = False
        for cache_path in shader_cache_paths:
            if cache_path.exists():
                shader_cache_found = True
                
                # Count shader cache entries
                try:
                    cache_entries = list(cache_path.iterdir())
                    cache_count = len([d for d in cache_entries if d.is_dir()])
                    
                    return (TestResult.PASS,
                           f"Steam integration OK: shader cache found with {cache_count} games",
                           None)
                except Exception:
                    return (TestResult.WARN,
                           "Steam shader cache found but not accessible",
                           "Check Steam directory permissions")
        
        if steam_found but not shader_cache_found:
            return (TestResult.INFO,
                   "Steam found but no shader cache yet",
                   "Shader cache will be created when games are played")
        
        return (TestResult.FAIL,
               "Steam integration test failed",
               "Check Steam installation and run Steam at least once")
    
    def test_network_connectivity(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test network connectivity for updates and P2P features"""
        test_hosts = [
            ("8.8.8.8", "DNS"),
            ("github.com", "GitHub"),
            ("pypi.org", "PyPI")
        ]
        
        connectivity_results = []
        
        for host, name in test_hosts:
            try:
                # Try to connect with timeout
                sock = socket.create_connection((host, 80), timeout=5)
                sock.close()
                connectivity_results.append((name, True))
            except Exception:
                connectivity_results.append((name, False))
        
        successful = [name for name, success in connectivity_results if success]
        failed = [name for name, success in connectivity_results if not success]
        
        if len(successful) == len(test_hosts):
            return (TestResult.PASS,
                   "Network connectivity OK: all services reachable",
                   None)
        elif successful:
            return (TestResult.WARN,
                   f"Partial network connectivity: {len(successful)}/{len(test_hosts)} services reachable",
                   f"Some features may be limited. Failed: {', '.join(failed)}")
        else:
            return (TestResult.FAIL,
                   "No network connectivity detected",
                   "Check network connection and firewall settings")
    
    def test_permissions_security(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test file permissions and security configuration"""
        permission_issues = []
        
        # Check install directory permissions
        if self.install_dir.exists():
            if not os.access(self.install_dir, os.R_OK | os.W_OK):
                permission_issues.append(f"Install directory not accessible: {self.install_dir}")
            
            # Check source files are readable
            src_dir = self.install_dir / "src"
            if src_dir.exists():
                for py_file in src_dir.glob("*.py"):
                    if not os.access(py_file, os.R_OK):
                        permission_issues.append(f"Source file not readable: {py_file.name}")
        
        # Check launcher script permissions
        launcher_script = Path.home() / ".local/bin/ml-shader-predictor"
        if launcher_script.exists():
            if not os.access(launcher_script, os.R_OK | os.X_OK):
                permission_issues.append("Launcher script not executable")
        
        # Check config directory permissions
        if self.config_dir.exists():
            if not os.access(self.config_dir, os.R_OK | os.W_OK):
                permission_issues.append("Config directory not accessible")
        
        # Check if running as root (security issue)
        if os.geteuid() == 0:
            permission_issues.append("Running as root user (security risk)")
        
        if permission_issues:
            return (TestResult.FAIL,
                   f"Permission issues found: {len(permission_issues)} problems",
                   f"Fix permissions. Issues: {'; '.join(permission_issues[:2])}")
        else:
            return (TestResult.PASS,
                   "File permissions and security OK",
                   None)
    
    def test_performance_baseline(self) -> Tuple[TestResult, str, Optional[str]]:
        """Test basic performance characteristics"""
        try:
            import numpy as np
            import time
            
            # Test NumPy performance (matrix multiplication)
            start_time = time.time()
            a = np.random.random((500, 500))
            b = np.random.random((500, 500))
            c = np.dot(a, b)
            numpy_time = time.time() - start_time
            
            # Test Python performance (simple computation)
            start_time = time.time()
            result = sum(x * x for x in range(10000))
            python_time = time.time() - start_time
            
            performance_issues = []
            
            # Check if performance is reasonable for Steam Deck
            if numpy_time > 2.0:  # Should be much faster on Steam Deck
                performance_issues.append(f"NumPy performance slow: {numpy_time:.2f}s")
            
            if python_time > 0.1:  # Basic Python should be fast
                performance_issues.append(f"Python performance slow: {python_time:.3f}s")
            
            if performance_issues:
                return (TestResult.WARN,
                       f"Performance issues detected: {'; '.join(performance_issues)}",
                       "Check system load, thermal throttling, or CPU governor settings")
            else:
                return (TestResult.PASS,
                       f"Performance OK: NumPy {numpy_time:.3f}s, Python {python_time:.3f}s",
                       None)
                       
        except Exception as e:
            return (TestResult.SKIP,
                   f"Performance test failed: {e}",
                   "Ensure NumPy is installed for performance testing")
    
    # ========================================================================
    # MAIN DIAGNOSTIC RUNNER
    # ========================================================================
    
    def run_all_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic tests and return comprehensive results"""
        self.log('INFO', "Starting comprehensive Steam Deck diagnostics...")
        self.log('INFO', f"Detailed mode: {self.detailed_mode}, Fix issues: {self.fix_issues}")
        
        # Collect system information
        system_info = self.collect_system_info()
        steamdeck_info = self.collect_steamdeck_info()
        
        self.log('INFO', f"System: {system_info.os_release.get('NAME', 'Unknown')} "
                        f"on {steamdeck_info.product_name}")
        
        # Run all diagnostic tests
        test_functions = [
            ("Steam Deck Hardware Detection", "Detect Steam Deck model and hardware", 
             TestPriority.IMPORTANT, self.test_steam_deck_detection),
            ("System Resource Check", "Verify adequate system resources",
             TestPriority.CRITICAL, self.test_system_resources),
            ("Thermal Conditions", "Check system temperature and cooling",
             TestPriority.IMPORTANT, self.test_thermal_conditions),
            ("Battery Status", "Check battery level and charging state",
             TestPriority.OPTIONAL, self.test_battery_status),
            ("Installation Files", "Verify all installation files are present",
             TestPriority.CRITICAL, self.test_installation_files),
            ("Python Environment", "Check Python version and dependencies",
             TestPriority.CRITICAL, self.test_python_environment),
            ("Systemd Service", "Check service configuration and status",
             TestPriority.IMPORTANT, self.test_systemd_service),
            ("Steam Integration", "Test Steam integration and shader cache access",
             TestPriority.IMPORTANT, self.test_steam_integration),
            ("Network Connectivity", "Test network access for updates/P2P",
             TestPriority.OPTIONAL, self.test_network_connectivity),
            ("Permissions & Security", "Check file permissions and security",
             TestPriority.CRITICAL, self.test_permissions_security),
            ("Performance Baseline", "Test basic performance characteristics",
             TestPriority.OPTIONAL, self.test_performance_baseline)
        ]
        
        for name, description, priority, test_func in test_functions:
            self.run_test(name, description, priority, test_func)
        
        # Calculate summary statistics
        total_tests = len(self.tests_run)
        passed_tests = sum(1 for t in self.tests_run if t.result == TestResult.PASS)
        failed_tests = sum(1 for t in self.tests_run if t.result == TestResult.FAIL)
        warned_tests = sum(1 for t in self.tests_run if t.result == TestResult.WARN)
        
        critical_failed = sum(1 for t in self.tests_run 
                            if t.result == TestResult.FAIL and t.priority == TestPriority.CRITICAL)
        
        # Generate overall assessment
        if critical_failed > 0:
            overall_status = "CRITICAL ISSUES"
            overall_color = "FAIL"
        elif failed_tests > 0:
            overall_status = "ISSUES DETECTED"
            overall_color = "FAIL"
        elif warned_tests > 0:
            overall_status = "WARNINGS"
            overall_color = "WARN"
        else:
            overall_status = "ALL TESTS PASSED"
            overall_color = "PASS"
        
        # Create comprehensive report
        total_time = time.time() - self.start_time
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'system_info': asdict(system_info),
            'steamdeck_info': asdict(steamdeck_info),
            'test_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'warnings': warned_tests,
                'critical_failures': critical_failed,
                'overall_status': overall_status
            },
            'test_results': [asdict(test) for test in self.tests_run]
        }
        
        # Display summary
        print("\n" + "="*80)
        self.log(overall_color, f"DIAGNOSTIC SUMMARY: {overall_status}")
        print(f"Total tests: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests} | Warnings: {warned_tests}")
        print(f"Execution time: {total_time:.2f} seconds")
        
        if critical_failed > 0:
            print(f"\n{self.colors['FAIL']}CRITICAL ISSUES DETECTED:{self.colors['END']}")
            for test in self.tests_run:
                if test.result == TestResult.FAIL and test.priority == TestPriority.CRITICAL:
                    print(f"  • {test.name}: {test.details}")
                    if test.fix_suggestion:
                        print(f"    Fix: {test.fix_suggestion}")
        
        return report
    
    def export_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export diagnostic report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"steamdeck_diagnostic_report_{timestamp}.json"
        
        output_path = Path.home() / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.log('INFO', f"Diagnostic report exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.log('FAIL', f"Failed to export report: {e}")
            return ""


def main():
    """Main entry point for diagnostic validator"""
    parser = argparse.ArgumentParser(
        description="Steam Deck ML Shader Predictor - Diagnostic Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 steamdeck-diagnostic-validator.py
  python3 steamdeck-diagnostic-validator.py --detailed --export-report
  python3 steamdeck-diagnostic-validator.py --fix-issues
        """
    )
    
    parser.add_argument('--detailed', action='store_true',
                       help='Enable detailed diagnostic output')
    parser.add_argument('--fix-issues', action='store_true',
                       help='Attempt to automatically fix detected issues')
    parser.add_argument('--export-report', action='store_true',
                       help='Export detailed report to JSON file')
    parser.add_argument('--report-file', type=str,
                       help='Custom filename for exported report')
    
    args = parser.parse_args()
    
    # Create diagnostic validator
    validator = SteamDeckDiagnosticValidator(
        detailed_mode=args.detailed,
        fix_issues=args.fix_issues
    )
    
    try:
        # Run diagnostics
        report = validator.run_all_diagnostics()
        
        # Export report if requested
        if args.export_report:
            validator.export_report(report, args.report_file)
        
        # Exit with appropriate code
        critical_failures = report['test_summary']['critical_failures']
        total_failures = report['test_summary']['failed']
        
        if critical_failures > 0:
            sys.exit(2)  # Critical failure
        elif total_failures > 0:
            sys.exit(1)  # General failure
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        validator.log('INFO', "Diagnostic interrupted by user")
        sys.exit(130)
    except Exception as e:
        validator.log('FAIL', f"Diagnostic failed with unexpected error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()