#!/usr/bin/env python3
"""
Enhanced RADV (AMD Radeon Vulkan) Optimizer for Steam Deck

This module provides comprehensive RADV driver optimizations specifically
tailored for Steam Deck's AMD APU, including Van Gogh and Phoenix architectures.

Key Features:
- Hardware-specific RADV configuration
- Thermal-aware shader compilation
- Power-efficient GPU scheduling
- Steam integration optimizations
- Dynamic performance tuning
"""

import os
import sys
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteamDeckAPU(Enum):
    """Steam Deck APU types with specific optimizations"""
    VAN_GOGH = "van_gogh"      # LCD models - Device ID 1002:163f
    PHOENIX = "phoenix"        # OLED models - Device ID 1002:15bf
    UNKNOWN = "unknown"


class RadVOptimizationLevel(Enum):
    """RADV optimization levels"""
    POWER_SAVE = "power_save"     # Battery optimization
    BALANCED = "balanced"         # Default performance
    PERFORMANCE = "performance"   # Maximum performance
    THERMAL_LIMIT = "thermal"     # Thermal-constrained


@dataclass
class RadVConfiguration:
    """RADV configuration parameters"""
    perftest_flags: List[str]
    debug_flags: List[str]
    environment_vars: Dict[str, str]
    shader_cache_size: int
    compile_threads: int
    thermal_limit: float
    power_limit_watts: float
    memory_budget_mb: int
    
    
@dataclass
class GPUMetrics:
    """Real-time GPU metrics"""
    temperature_celsius: float
    power_usage_watts: float
    gpu_utilization: float
    memory_usage_mb: int
    clock_speed_mhz: int
    voltage: float
    fan_rpm: Optional[int] = None


class RadVOptimizer:
    """Enhanced RADV optimizer for Steam Deck"""
    
    def __init__(self):
        self.apu_type = self._detect_apu_type()
        self.current_config = None
        self.metrics_thread = None
        self.monitoring_active = False
        self.performance_history = []
        self.thermal_history = []
        self.optimization_level = RadVOptimizationLevel.BALANCED
        
        # Configuration database
        self.optimization_configs = self._initialize_configurations()
        
        # Initialize with hardware-appropriate settings
        self.apply_optimization_level(self.optimization_level)
        
    def _detect_apu_type(self) -> SteamDeckAPU:
        """Detect Steam Deck APU type for specific optimizations"""
        try:
            # Check PCI device ID
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
            if result.returncode == 0:
                pci_output = result.stdout
                
                # Van Gogh APU (LCD Steam Deck)
                if "1002:163f" in pci_output:
                    logger.info("Detected Van Gogh APU (LCD Steam Deck)")
                    return SteamDeckAPU.VAN_GOGH
                # Phoenix APU (OLED Steam Deck)  
                elif "1002:15bf" in pci_output:
                    logger.info("Detected Phoenix APU (OLED Steam Deck)")
                    return SteamDeckAPU.PHOENIX
            
            # Fallback: Check DMI information
            dmi_files = [
                "/sys/devices/virtual/dmi/id/board_name",
                "/sys/class/dmi/id/board_name"
            ]
            
            for dmi_file in dmi_files:
                if os.path.exists(dmi_file):
                    with open(dmi_file, 'r') as f:
                        board_name = f.read().strip()
                        if "Galileo" in board_name:
                            logger.info("Detected Phoenix APU via DMI (OLED Steam Deck)")
                            return SteamDeckAPU.PHOENIX
                        elif "Jupiter" in board_name:
                            logger.info("Detected Van Gogh APU via DMI (LCD Steam Deck)")
                            return SteamDeckAPU.VAN_GOGH
            
            logger.warning("Could not detect specific Steam Deck APU type")
            return SteamDeckAPU.UNKNOWN
            
        except Exception as e:
            logger.error(f"APU detection failed: {e}")
            return SteamDeckAPU.UNKNOWN
    
    def _initialize_configurations(self) -> Dict[RadVOptimizationLevel, RadVConfiguration]:
        """Initialize RADV configurations for different optimization levels"""
        configs = {}
        
        # Base configuration for Van Gogh (LCD)
        if self.apu_type == SteamDeckAPU.VAN_GOGH:
            configs[RadVOptimizationLevel.POWER_SAVE] = RadVConfiguration(
                perftest_flags=["aco"],  # Minimal flags for power saving
                debug_flags=["noshaderdb", "nocompute"],
                environment_vars={
                    "RADV_PERFTEST": "aco",
                    "RADV_DEBUG": "noshaderdb,nocompute",
                    "MESA_VK_DEVICE_SELECT": "1002:163f",
                    "RADV_LOWER_DISCARD_TO_DEMOTE": "1",
                    "MESA_GLSL_CACHE_DISABLE": "0",
                    "MESA_GLSL_CACHE_MAX_SIZE": "268435456",  # 256MB
                    "__GL_SHADER_DISK_CACHE": "1",
                    "__GL_SHADER_DISK_CACHE_SIZE": "268435456",
                    "DXVK_ASYNC": "1",
                    "RADV_FORCE_FAMILY": "GFX103"  # Van Gogh family
                },
                shader_cache_size=256,
                compile_threads=2,
                thermal_limit=83.0,
                power_limit_watts=12.0,
                memory_budget_mb=1024
            )
            
            configs[RadVOptimizationLevel.BALANCED] = RadVConfiguration(
                perftest_flags=["aco", "nggc", "sam"],
                debug_flags=["noshaderdb"],
                environment_vars={
                    "RADV_PERFTEST": "aco,nggc,sam",
                    "RADV_DEBUG": "noshaderdb",
                    "MESA_VK_DEVICE_SELECT": "1002:163f",
                    "RADV_LOWER_DISCARD_TO_DEMOTE": "1",
                    "MESA_GLSL_CACHE_DISABLE": "0", 
                    "MESA_GLSL_CACHE_MAX_SIZE": "536870912",  # 512MB
                    "__GL_SHADER_DISK_CACHE": "1",
                    "__GL_SHADER_DISK_CACHE_SIZE": "536870912",
                    "DXVK_ASYNC": "1",
                    "RADV_FORCE_FAMILY": "GFX103",
                    "RADV_THREAD_TRACE": "0"  # Disable for performance
                },
                shader_cache_size=512,
                compile_threads=4,
                thermal_limit=85.0,
                power_limit_watts=15.0,
                memory_budget_mb=1536
            )
            
            configs[RadVOptimizationLevel.PERFORMANCE] = RadVConfiguration(
                perftest_flags=["aco", "nggc", "sam", "rt", "ngg_streamout"],
                debug_flags=[],  # No debug flags for max performance
                environment_vars={
                    "RADV_PERFTEST": "aco,nggc,sam,rt,ngg_streamout",
                    "RADV_DEBUG": "",
                    "MESA_VK_DEVICE_SELECT": "1002:163f",
                    "RADV_LOWER_DISCARD_TO_DEMOTE": "1",
                    "MESA_GLSL_CACHE_DISABLE": "0",
                    "MESA_GLSL_CACHE_MAX_SIZE": "1073741824",  # 1GB
                    "__GL_SHADER_DISK_CACHE": "1",
                    "__GL_SHADER_DISK_CACHE_SIZE": "1073741824",
                    "DXVK_ASYNC": "1",
                    "RADV_FORCE_FAMILY": "GFX103",
                    "RADV_FORCE_VRS": "2x2",  # Variable Rate Shading
                    "RADV_TEX_ANISO": "-1"  # Max anisotropic filtering
                },
                shader_cache_size=1024,
                compile_threads=6,
                thermal_limit=87.0,
                power_limit_watts=20.0,
                memory_budget_mb=2048
            )
            
        # Configuration for Phoenix (OLED)
        elif self.apu_type == SteamDeckAPU.PHOENIX:
            configs[RadVOptimizationLevel.BALANCED] = RadVConfiguration(
                perftest_flags=["aco", "nggc", "sam", "rt"],
                debug_flags=["noshaderdb"],
                environment_vars={
                    "RADV_PERFTEST": "aco,nggc,sam,rt",
                    "RADV_DEBUG": "noshaderdb",
                    "MESA_VK_DEVICE_SELECT": "1002:15bf",
                    "RADV_LOWER_DISCARD_TO_DEMOTE": "1",
                    "MESA_GLSL_CACHE_DISABLE": "0",
                    "MESA_GLSL_CACHE_MAX_SIZE": "1073741824",  # 1GB - OLED has more memory headroom
                    "__GL_SHADER_DISK_CACHE": "1",
                    "__GL_SHADER_DISK_CACHE_SIZE": "1073741824",
                    "DXVK_ASYNC": "1",
                    "RADV_FORCE_FAMILY": "GFX1103",  # Phoenix family
                    "RADV_THREAD_TRACE": "0"
                },
                shader_cache_size=1024,
                compile_threads=6,
                thermal_limit=87.0,
                power_limit_watts=18.0,
                memory_budget_mb=2560
            )
            
            configs[RadVOptimizationLevel.PERFORMANCE] = RadVConfiguration(
                perftest_flags=["aco", "nggc", "sam", "rt", "ngg_streamout", "shader_ballot"],
                debug_flags=[],
                environment_vars={
                    "RADV_PERFTEST": "aco,nggc,sam,rt,ngg_streamout,shader_ballot",
                    "RADV_DEBUG": "",
                    "MESA_VK_DEVICE_SELECT": "1002:15bf",
                    "RADV_LOWER_DISCARD_TO_DEMOTE": "1",
                    "MESA_GLSL_CACHE_DISABLE": "0",
                    "MESA_GLSL_CACHE_MAX_SIZE": "1610612736",  # 1.5GB
                    "__GL_SHADER_DISK_CACHE": "1", 
                    "__GL_SHADER_DISK_CACHE_SIZE": "1610612736",
                    "DXVK_ASYNC": "1",
                    "RADV_FORCE_FAMILY": "GFX1103",
                    "RADV_FORCE_VRS": "2x2",
                    "RADV_TEX_ANISO": "-1",
                    "RADV_ENABLE_MRT_OUTPUT_NAN_FIXUP": "0"  # Phoenix optimization
                },
                shader_cache_size=1536,
                compile_threads=8,
                thermal_limit=89.0,
                power_limit_watts=25.0,
                memory_budget_mb=3072
            )
        
        # Add thermal-limited configuration for both APUs
        for apu_type in [SteamDeckAPU.VAN_GOGH, SteamDeckAPU.PHOENIX]:
            if apu_type in [self.apu_type]:
                # Thermal-limited is based on balanced but with reduced limits
                thermal_config = configs.get(RadVOptimizationLevel.BALANCED)
                if thermal_config:
                    configs[RadVOptimizationLevel.THERMAL_LIMIT] = RadVConfiguration(
                        perftest_flags=["aco", "nggc"],  # Reduced flags
                        debug_flags=["noshaderdb", "nocompute"],
                        environment_vars=thermal_config.environment_vars.copy(),
                        shader_cache_size=thermal_config.shader_cache_size // 2,
                        compile_threads=max(1, thermal_config.compile_threads // 2),
                        thermal_limit=thermal_config.thermal_limit - 5.0,
                        power_limit_watts=thermal_config.power_limit_watts * 0.7,
                        memory_budget_mb=thermal_config.memory_budget_mb // 2
                    )
        
        return configs
    
    def apply_optimization_level(self, level: RadVOptimizationLevel) -> bool:
        """Apply specific optimization level"""
        try:
            if level not in self.optimization_configs:
                logger.error(f"Optimization level {level} not available for APU {self.apu_type}")
                return False
            
            config = self.optimization_configs[level]
            self.current_config = config
            self.optimization_level = level
            
            # Apply environment variables
            for var, value in config.environment_vars.items():
                os.environ[var] = str(value)
                logger.debug(f"Set {var}={value}")
            
            logger.info(f"Applied RADV optimization level: {level.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimization level {level}: {e}")
            return False
    
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get real-time GPU metrics from Steam Deck"""
        try:
            metrics = GPUMetrics(
                temperature_celsius=0.0,
                power_usage_watts=0.0,
                gpu_utilization=0.0,
                memory_usage_mb=0,
                clock_speed_mhz=0,
                voltage=0.0
            )
            
            # Read GPU temperature from hwmon
            hwmon_paths = [
                "/sys/class/hwmon/hwmon0/temp1_input",
                "/sys/class/hwmon/hwmon1/temp1_input",
                "/sys/class/hwmon/hwmon2/temp1_input"
            ]
            
            for path in hwmon_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            temp_millicelsius = int(f.read().strip())
                            metrics.temperature_celsius = temp_millicelsius / 1000.0
                            break
                    except (ValueError, IOError):
                        continue
            
            # Read GPU utilization from DRM
            drm_paths = [
                "/sys/class/drm/card0/device/gpu_busy_percent",
                "/sys/class/drm/card1/device/gpu_busy_percent"
            ]
            
            for path in drm_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            metrics.gpu_utilization = float(f.read().strip())
                            break
                    except (ValueError, IOError):
                        continue
            
            # Read power consumption
            power_paths = [
                "/sys/class/power_supply/ADP1/power_now",
                "/sys/class/power_supply/BAT1/power_now"
            ]
            
            for path in power_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            power_microwatts = int(f.read().strip())
                            metrics.power_usage_watts = power_microwatts / 1000000.0
                            break
                    except (ValueError, IOError):
                        continue
            
            # Read GPU clock speed
            clock_paths = [
                "/sys/class/drm/card0/device/pp_dpm_sclk",
                "/sys/class/drm/card1/device/pp_dpm_sclk"
            ]
            
            for path in clock_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            clock_info = f.read().strip()
                            # Parse current clock from DPM state
                            for line in clock_info.split('\n'):
                                if '*' in line:  # Current state marked with *
                                    clock_mhz = int(line.split(':')[1].strip().rstrip('Mhz'))
                                    metrics.clock_speed_mhz = clock_mhz
                                    break
                    except (ValueError, IOError, IndexError):
                        continue
            
            # Read GPU memory usage (approximation)
            try:
                gpu_memory = psutil.virtual_memory()
                # Steam Deck APU shares system memory
                metrics.memory_usage_mb = int((gpu_memory.total - gpu_memory.available) / (1024 * 1024))
            except Exception:
                pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring GPU metrics and adaptive optimization"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.metrics_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self.metrics_thread.start()
        logger.info("Started GPU monitoring and adaptive optimization")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5.0)
        logger.info("Stopped GPU monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop with adaptive optimization"""
        consecutive_thermal_events = 0
        
        while self.monitoring_active:
            try:
                metrics = self.get_gpu_metrics()
                if not metrics:
                    time.sleep(interval)
                    continue
                
                # Store metrics history
                self.performance_history.append({
                    'timestamp': time.time(),
                    'gpu_utilization': metrics.gpu_utilization,
                    'clock_speed': metrics.clock_speed_mhz,
                    'power_usage': metrics.power_usage_watts
                })
                
                self.thermal_history.append({
                    'timestamp': time.time(),
                    'temperature': metrics.temperature_celsius
                })
                
                # Keep history limited
                if len(self.performance_history) > 300:  # 5 minutes at 1Hz
                    self.performance_history.pop(0)
                if len(self.thermal_history) > 300:
                    self.thermal_history.pop(0)
                
                # Adaptive optimization logic
                current_temp = metrics.temperature_celsius
                current_config = self.current_config
                
                if current_config:
                    # Check for thermal throttling
                    if current_temp > current_config.thermal_limit:
                        consecutive_thermal_events += 1
                        if consecutive_thermal_events >= 3:  # 3 consecutive high temps
                            if self.optimization_level != RadVOptimizationLevel.THERMAL_LIMIT:
                                logger.warning(f"High temperature ({current_temp}°C), switching to thermal limit mode")
                                self.apply_optimization_level(RadVOptimizationLevel.THERMAL_LIMIT)
                            consecutive_thermal_events = 0
                    else:
                        consecutive_thermal_events = 0
                        
                        # Check if we can increase performance
                        if (self.optimization_level == RadVOptimizationLevel.THERMAL_LIMIT and 
                            current_temp < current_config.thermal_limit - 5.0):
                            logger.info(f"Temperature normalized ({current_temp}°C), returning to balanced mode")
                            self.apply_optimization_level(RadVOptimizationLevel.BALANCED)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def optimize_for_game(self, game_id: Optional[str] = None, game_name: Optional[str] = None) -> bool:
        """Apply game-specific RADV optimizations"""
        try:
            # Game-specific optimization database
            game_optimizations = {
                # Steam AppIDs for common games
                "1091500": {  # Cyberpunk 2077
                    "level": RadVOptimizationLevel.PERFORMANCE,
                    "additional_vars": {
                        "RADV_FORCE_VRS": "2x2",  # VRS helps with performance
                        "DXVK_ASYNC": "1"
                    }
                },
                "70": {  # Half-Life
                    "level": RadVOptimizationLevel.BALANCED,
                    "additional_vars": {}
                },
                "570": {  # Dota 2
                    "level": RadVOptimizationLevel.PERFORMANCE,
                    "additional_vars": {
                        "RADV_TEX_ANISO": "16"
                    }
                },
                "default": {
                    "level": RadVOptimizationLevel.BALANCED,
                    "additional_vars": {}
                }
            }
            
            # Select optimization
            optimization = game_optimizations.get(game_id, game_optimizations["default"])
            
            # Apply optimization level
            if self.apply_optimization_level(optimization["level"]):
                # Apply additional game-specific variables
                for var, value in optimization["additional_vars"].items():
                    os.environ[var] = str(value)
                    logger.debug(f"Game-specific: Set {var}={value}")
                
                logger.info(f"Applied game-specific optimization for {game_name or game_id or 'unknown'}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply game-specific optimization: {e}")
            return False
    
    def create_steam_compatibility_tool(self, install_dir: str) -> bool:
        """Create Steam compatibility tool integration"""
        try:
            compat_dir = Path(install_dir) / "steam_compat"
            compat_dir.mkdir(exist_ok=True)
            
            # Create compatibility tool script
            compat_script = compat_dir / "ml_shader_optimizer"
            with open(compat_script, 'w') as f:
                f.write(f'''#!/bin/bash
# ML Shader Predictor Steam Compatibility Tool

# Set up RADV optimizations
source "{install_dir}/radv_optimizations.sh"

# Start shader predictor service if not running
if ! pgrep -f "shader_prediction_system" > /dev/null; then
    "{install_dir}/venv/bin/python" "{install_dir}/src/shader_prediction_system.py" --service &
fi

# Execute game with optimizations
exec "$@"
''')
            
            compat_script.chmod(0o755)
            
            # Create compatibility tool manifest
            tool_manifest = compat_dir / "compatibilitytool.vdf"
            with open(tool_manifest, 'w') as f:
                f.write('''
"compatibilitytools"
{
    "compat_tools"
    {
        "MLShaderOptimizer"
        {
            "install_path" "."
            "display_name" "ML Shader Optimizer"
            "from_oslist" "windows"
            "to_oslist" "linux"
            "tool" "./ml_shader_optimizer"
        }
    }
}
''')
            
            logger.info(f"Created Steam compatibility tool at: {compat_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Steam compatibility tool: {e}")
            return False
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        try:
            metrics = self.get_gpu_metrics()
            
            # Calculate performance metrics
            avg_gpu_util = 0.0
            avg_clock_speed = 0.0
            if self.performance_history:
                avg_gpu_util = sum(p['gpu_utilization'] for p in self.performance_history[-60:]) / min(60, len(self.performance_history))
                avg_clock_speed = sum(p['clock_speed'] for p in self.performance_history[-60:]) / min(60, len(self.performance_history))
            
            avg_temperature = 0.0
            if self.thermal_history:
                avg_temperature = sum(t['temperature'] for t in self.thermal_history[-60:]) / min(60, len(self.thermal_history))
            
            report = {
                "hardware": {
                    "apu_type": self.apu_type.value,
                    "device_id": "1002:163f" if self.apu_type == SteamDeckAPU.VAN_GOGH else "1002:15bf"
                },
                "current_optimization": {
                    "level": self.optimization_level.value,
                    "perftest_flags": self.current_config.perftest_flags if self.current_config else [],
                    "debug_flags": self.current_config.debug_flags if self.current_config else [],
                    "shader_cache_size_mb": self.current_config.shader_cache_size if self.current_config else 0,
                    "compile_threads": self.current_config.compile_threads if self.current_config else 0
                },
                "current_metrics": {
                    "temperature_celsius": metrics.temperature_celsius if metrics else 0.0,
                    "gpu_utilization_percent": metrics.gpu_utilization if metrics else 0.0,
                    "clock_speed_mhz": metrics.clock_speed_mhz if metrics else 0,
                    "power_usage_watts": metrics.power_usage_watts if metrics else 0.0
                },
                "averages_last_minute": {
                    "gpu_utilization_percent": avg_gpu_util,
                    "clock_speed_mhz": avg_clock_speed,
                    "temperature_celsius": avg_temperature
                },
                "environment_variables": dict(os.environ) if self.current_config else {}
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate optimization report: {e}")
            return {"error": str(e)}


def main():
    """Test the RADV optimizer"""
    print("🎮 Steam Deck RADV Optimizer Test")
    print("=" * 40)
    
    optimizer = RadVOptimizer()
    
    print(f"Detected APU: {optimizer.apu_type.value}")
    print(f"Current optimization: {optimizer.optimization_level.value}")
    
    # Test metrics
    metrics = optimizer.get_gpu_metrics()
    if metrics:
        print(f"Temperature: {metrics.temperature_celsius}°C")
        print(f"GPU Utilization: {metrics.gpu_utilization}%")
        print(f"Clock Speed: {metrics.clock_speed_mhz} MHz")
        print(f"Power Usage: {metrics.power_usage_watts}W")
    
    # Test different optimization levels
    for level in RadVOptimizationLevel:
        if level in optimizer.optimization_configs:
            print(f"\nTesting optimization level: {level.value}")
            if optimizer.apply_optimization_level(level):
                print("✓ Applied successfully")
            else:
                print("✗ Failed to apply")
    
    # Generate report
    report = optimizer.generate_optimization_report()
    print(f"\nOptimization report generated with {len(report)} sections")
    
    print("\n🚀 Test completed!")


if __name__ == "__main__":
    main()