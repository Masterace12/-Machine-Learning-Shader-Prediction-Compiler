#!/usr/bin/env python3
"""
RDNA 2 GPU Optimizer for OLED Steam Deck
Hardware-specific optimizations for AMD RDNA 2 architecture with OLED model enhancements
"""

import os
import time
import threading
import logging
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import psutil

logger = logging.getLogger(__name__)


class RDNA2WaveMode(Enum):
    """RDNA 2 wave execution modes"""
    WAVE32 = "wave32"  # Better for smaller workloads
    WAVE64 = "wave64"  # Better for larger workloads
    AUTO = "auto"      # Let driver decide


class GPUMemoryPool(Enum):
    """GPU memory pool types"""
    VRAM = "vram"              # Dedicated GPU memory
    GTT = "gtt"                # Graphics Translation Table (system memory)
    SHARED = "shared"          # APU shared memory pool
    INFINITY_CACHE = "l3"      # RDNA 2 Infinity Cache


@dataclass
class GPUMetrics:
    """RDNA 2 GPU performance metrics"""
    gpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    clock_speed_mhz: int = 0
    memory_clock_mhz: int = 0
    temperature_celsius: float = 0.0
    power_draw_watts: float = 0.0
    shader_engines_active: int = 0
    compute_units_active: int = 0
    wave_occupancy_percent: float = 0.0
    infinity_cache_hit_rate: float = 0.0
    thermal_throttling: bool = False
    memory_pressure: bool = False


@dataclass
class ShaderOptimization:
    """Shader compilation optimization settings for RDNA 2"""
    wave_mode: RDNA2WaveMode = RDNA2WaveMode.AUTO
    compiler_backend: str = "aco"  # ACO is better for RDNA 2
    optimization_level: int = 2
    enable_wave32: bool = True
    enable_ngg: bool = True  # Next-Gen Geometry
    enable_mesh_shaders: bool = True
    memory_pool_preference: GPUMemoryPool = GPUMemoryPool.SHARED
    cache_locality_optimization: bool = True
    thread_group_size_hint: int = 64


class RDNA2GPUOptimizer:
    """RDNA 2 GPU optimizer with OLED Steam Deck specific enhancements"""
    
    def __init__(self, oled_model: bool = True):
        """
        Initialize RDNA 2 GPU optimizer
        
        Args:
            oled_model: True if running on OLED Steam Deck for enhanced optimizations
        """
        self.oled_model = oled_model
        self.gpu_device_path = self._find_amdgpu_device()
        self.sysfs_gpu_path = Path("/sys/class/drm/card0/device")
        
        # GPU configuration
        self.compute_units = 8  # Steam Deck has 8 CUs
        self.shader_engines = 2  # RDNA 2 on Steam Deck
        self.max_clock_mhz = 1600  # Steam Deck GPU max frequency
        
        # OLED model has enhanced cooling and can sustain higher clocks
        if self.oled_model:
            self.sustained_clock_target = 1400  # Higher sustained performance
            self.boost_duration_ms = 15000      # Longer boost periods
            self.thermal_headroom_factor = 1.2  # Better cooling headroom
        else:
            self.sustained_clock_target = 1200
            self.boost_duration_ms = 10000
            self.thermal_headroom_factor = 1.0
        
        # Performance monitoring
        self.current_metrics = GPUMetrics()
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._metrics_lock = threading.Lock()
        
        # Shader optimization cache
        self.shader_optimizations: Dict[str, ShaderOptimization] = {}
        self.optimization_profiles = self._create_optimization_profiles()
        
        # Memory management
        self.unified_memory_size_mb = 16 * 1024  # 16GB shared with CPU
        self.gpu_reserved_memory_mb = 1024  # Reserve for GPU-intensive tasks
        self.memory_bandwidth_gbps = 88.0  # LPDDR5 bandwidth
        
        # Driver configuration
        self.radv_perftest = []  # Radeon Vulkan performance test options
        self.mesa_config = {}   # Mesa driver configuration
        
        self._initialize_gpu_state()
        
        logger.info(f"RDNA 2 GPU optimizer initialized (OLED: {self.oled_model})")
    
    def _find_amdgpu_device(self) -> Optional[Path]:
        """Find AMDGPU device path"""
        try:
            # Look for AMDGPU device
            for gpu_path in Path("/dev/dri").glob("card*"):
                if gpu_path.is_char_device():
                    return gpu_path
            
            # Fallback
            return Path("/dev/dri/card0")
        except Exception:
            return None
    
    def _initialize_gpu_state(self):
        """Initialize GPU state and capabilities"""
        try:
            # Set up RADV performance optimizations for OLED
            if self.oled_model:
                self.radv_perftest = [
                    "aco",          # ACO compiler backend
                    "nggc",         # NGG culling
                    "tccompat",     # Texture compression compatibility
                    "sam",          # Smart Access Memory
                    "ngg"           # Next-Gen Geometry
                ]
            else:
                self.radv_perftest = [
                    "aco",          # ACO compiler backend
                    "nggc"          # NGG culling (conservative for LCD)
                ]
            
            # Mesa driver configuration for OLED optimizations
            self.mesa_config = {
                "RADV_PERFTEST": ",".join(self.radv_perftest),
                "RADV_DEBUG": "" if self.oled_model else "nohiz",  # OLED can use HiZ
                "MESA_VK_WSI_PRESENT_MODE": "mailbox" if self.oled_model else "fifo",
                "RADV_INVARIANT_GEOM": "true",  # Geometry invariance
                "AMD_VULKAN_ICD": "RADV"  # Use RADV driver
            }
            
            # Apply environment variables
            for key, value in self.mesa_config.items():
                os.environ[key] = value
            
            logger.info("GPU driver configuration applied")
            
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
    
    def _create_optimization_profiles(self) -> Dict[str, ShaderOptimization]:
        """Create shader optimization profiles"""
        profiles = {}
        
        # High-performance profile for OLED
        profiles["oled_performance"] = ShaderOptimization(
            wave_mode=RDNA2WaveMode.WAVE32,
            compiler_backend="aco",
            optimization_level=3,
            enable_wave32=True,
            enable_ngg=True,
            enable_mesh_shaders=True,
            memory_pool_preference=GPUMemoryPool.SHARED,
            cache_locality_optimization=True,
            thread_group_size_hint=128  # Larger workgroups for OLED
        )
        
        # Balanced profile
        profiles["balanced"] = ShaderOptimization(
            wave_mode=RDNA2WaveMode.AUTO,
            compiler_backend="aco",
            optimization_level=2,
            enable_wave32=True,
            enable_ngg=self.oled_model,
            enable_mesh_shaders=self.oled_model,
            memory_pool_preference=GPUMemoryPool.SHARED,
            cache_locality_optimization=True,
            thread_group_size_hint=64
        )
        
        # Power-efficient profile
        profiles["power_efficient"] = ShaderOptimization(
            wave_mode=RDNA2WaveMode.WAVE64,
            compiler_backend="aco",
            optimization_level=1,
            enable_wave32=False,
            enable_ngg=False,
            enable_mesh_shaders=False,
            memory_pool_preference=GPUMemoryPool.GTT,
            cache_locality_optimization=False,
            thread_group_size_hint=32
        )
        
        return profiles
    
    def _read_gpu_sysfs(self, filename: str) -> Optional[str]:
        """Read GPU sysfs file"""
        try:
            file_path = self.sysfs_gpu_path / filename
            if file_path.exists():
                return file_path.read_text().strip()
        except Exception as e:
            logger.debug(f"Failed to read GPU sysfs {filename}: {e}")
        return None
    
    def _write_gpu_sysfs(self, filename: str, value: str) -> bool:
        """Write to GPU sysfs file (requires permissions)"""
        try:
            file_path = self.sysfs_gpu_path / filename
            if file_path.exists() and os.access(file_path, os.W_OK):
                file_path.write_text(value)
                return True
        except Exception as e:
            logger.debug(f"Failed to write GPU sysfs {filename}: {e}")
        return False
    
    def get_gpu_metrics(self) -> GPUMetrics:
        """Get current GPU performance metrics"""
        with self._metrics_lock:
            try:
                # GPU utilization
                gpu_busy = self._read_gpu_sysfs("gpu_busy_percent")
                if gpu_busy:
                    self.current_metrics.gpu_utilization_percent = float(gpu_busy)
                
                # Memory utilization
                mem_info = self._read_gpu_sysfs("mem_info_vram_used")
                mem_total = self._read_gpu_sysfs("mem_info_vram_total")
                if mem_info and mem_total:
                    used_mb = int(mem_info) // (1024 * 1024)
                    total_mb = int(mem_total) // (1024 * 1024)
                    self.current_metrics.memory_utilization_percent = (used_mb / total_mb) * 100
                
                # Clock speeds
                gpu_clock = self._read_gpu_sysfs("pp_dpm_sclk")
                if gpu_clock:
                    # Parse current clock from DPM state
                    lines = gpu_clock.split('\n')
                    for line in lines:
                        if '*' in line:  # Current active state
                            clock_str = line.split(':')[1].strip().replace('Mhz', '').replace('*', '')
                            self.current_metrics.clock_speed_mhz = int(clock_str)
                            break
                
                # Memory clock
                mem_clock = self._read_gpu_sysfs("pp_dpm_mclk")
                if mem_clock:
                    lines = mem_clock.split('\n')
                    for line in lines:
                        if '*' in line:
                            clock_str = line.split(':')[1].strip().replace('Mhz', '').replace('*', '')
                            self.current_metrics.memory_clock_mhz = int(clock_str)
                            break
                
                # Temperature
                temp = self._read_gpu_sysfs("hwmon/hwmon*/temp1_input")
                if not temp:
                    # Try alternative paths
                    for hwmon_dir in Path("/sys/class/hwmon").glob("hwmon*"):
                        temp_file = hwmon_dir / "temp1_input"
                        if temp_file.exists():
                            temp = temp_file.read_text().strip()
                            break
                
                if temp:
                    self.current_metrics.temperature_celsius = int(temp) / 1000.0
                
                # Power draw (estimate from GPU utilization)
                gpu_util = self.current_metrics.gpu_utilization_percent
                estimated_power = (gpu_util / 100.0) * 15.0  # Max ~15W for GPU
                self.current_metrics.power_draw_watts = estimated_power
                
                # Thermal throttling check
                self.current_metrics.thermal_throttling = (
                    self.current_metrics.temperature_celsius > 85.0 or
                    self.current_metrics.clock_speed_mhz < self.sustained_clock_target * 0.8
                )
                
                # Memory pressure check (simplified)
                mem_util = self.current_metrics.memory_utilization_percent
                self.current_metrics.memory_pressure = mem_util > 85.0
                
            except Exception as e:
                logger.error(f"GPU metrics collection failed: {e}")
            
            return self.current_metrics
    
    def optimize_for_shader_compilation(self, profile_name: str = "balanced") -> Dict[str, Any]:
        """Optimize GPU for shader compilation workload"""
        if profile_name not in self.optimization_profiles:
            logger.error(f"Unknown optimization profile: {profile_name}")
            return {}
        
        optimization = self.optimization_profiles[profile_name]
        applied_settings = {}
        
        try:
            # Set power profile for sustained performance (OLED benefit)
            if self.oled_model:
                power_profile = "high"  # OLED can sustain higher power
            else:
                power_profile = "auto"  # Conservative for LCD
            
            if self._write_gpu_sysfs("power_dpm_force_performance_level", power_profile):
                applied_settings["power_profile"] = power_profile
            
            # Configure wave mode via environment variable
            if optimization.wave_mode == RDNA2WaveMode.WAVE32:
                os.environ["RADV_FORCE_WAVE32"] = "1"
                applied_settings["wave_mode"] = "wave32"
            elif optimization.wave_mode == RDNA2WaveMode.WAVE64:
                os.environ.pop("RADV_FORCE_WAVE32", None)
                applied_settings["wave_mode"] = "wave64"
            
            # Memory pool configuration
            if optimization.memory_pool_preference == GPUMemoryPool.SHARED:
                # Optimize for unified memory architecture
                os.environ["RADV_HEAP_BUDGET_TOLERANCE"] = "90"
                applied_settings["memory_optimization"] = "unified"
            
            # Cache optimization for OLED
            if optimization.cache_locality_optimization and self.oled_model:
                os.environ["RADV_ENABLE_MRT_OUTPUT_RELOCATIONS"] = "1"
                applied_settings["cache_optimization"] = "enabled"
            
            # Thread group size hints
            applied_settings["thread_group_size"] = optimization.thread_group_size_hint
            applied_settings["compiler_backend"] = optimization.compiler_backend
            applied_settings["optimization_level"] = optimization.optimization_level
            
            logger.info(f"Applied GPU optimization profile '{profile_name}': {applied_settings}")
            
        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
        
        return applied_settings
    
    def set_memory_budget(self, shader_compilation_mb: int) -> bool:
        """Set memory budget for shader compilation"""
        try:
            # Reserve memory for shader compilation
            available_memory = self.unified_memory_size_mb - self.gpu_reserved_memory_mb
            compilation_budget = min(shader_compilation_mb, available_memory // 2)
            
            # Configure memory pressure thresholds
            if self.oled_model:
                # OLED can handle higher memory pressure due to better cooling
                memory_threshold = 0.90
            else:
                memory_threshold = 0.80
            
            # Set environment variables for memory management
            os.environ["RADV_MAX_ALLOC_SIZE"] = str(compilation_budget * 1024 * 1024)
            os.environ["RADV_HEAP_BUDGET_TOLERANCE"] = str(int(memory_threshold * 100))
            
            logger.info(f"Set memory budget: {compilation_budget}MB for shader compilation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set memory budget: {e}")
            return False
    
    def monitor_gaming_workload(self) -> Dict[str, Any]:
        """Monitor GPU workload to detect gaming activity"""
        metrics = self.get_gpu_metrics()
        
        # Gaming detection heuristics
        gaming_indicators = {
            "high_gpu_utilization": metrics.gpu_utilization_percent > 40,
            "sustained_clocks": metrics.clock_speed_mhz > self.max_clock_mhz * 0.6,
            "memory_active": metrics.memory_utilization_percent > 30,
            "consistent_workload": True  # Would need historical data
        }
        
        gaming_score = sum(gaming_indicators.values()) / len(gaming_indicators)
        gaming_active = gaming_score > 0.5
        
        # Recommend compilation throttling during gaming
        compilation_recommendation = {
            "gaming_detected": gaming_active,
            "recommended_threads": 1 if gaming_active else (4 if self.oled_model else 2),
            "compilation_priority": "low" if gaming_active else "normal",
            "background_only": gaming_active,
            "thermal_aware": metrics.thermal_throttling
        }
        
        return compilation_recommendation
    
    def start_monitoring(self, interval: float = 2.0):
        """Start GPU monitoring thread"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitoring_loop():
            while self._monitoring_active:
                try:
                    self.get_gpu_metrics()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"GPU monitoring error: {e}")
                    time.sleep(interval * 2)
        
        self._monitoring_thread = threading.Thread(
            target=monitoring_loop,
            name="RDNA2_Monitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("RDNA 2 GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self._monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        logger.info("RDNA 2 GPU monitoring stopped")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get GPU optimization recommendations"""
        metrics = self.get_gpu_metrics()
        recommendations = []
        
        # Thermal recommendations
        if metrics.thermal_throttling:
            recommendations.append({
                "type": "thermal",
                "priority": "high",
                "message": f"GPU thermal throttling detected ({metrics.temperature_celsius:.1f}°C)",
                "action": "Reduce shader compilation load or improve cooling",
                "oled_benefit": "OLED model provides better thermal headroom"
            })
        
        # Memory recommendations
        if metrics.memory_pressure:
            recommendations.append({
                "type": "memory",
                "priority": "medium", 
                "message": f"High GPU memory utilization ({metrics.memory_utilization_percent:.1f}%)",
                "action": "Reduce concurrent shader compilations",
                "oled_benefit": "OLED allows more aggressive memory usage"
            })
        
        # Performance recommendations
        if metrics.gpu_utilization_percent > 80:
            recommendations.append({
                "type": "performance",
                "priority": "info",
                "message": "High GPU utilization detected - likely gaming",
                "action": "Throttle background shader compilation",
                "oled_benefit": "OLED can maintain higher sustained performance"
            })
        
        # OLED-specific recommendations
        if self.oled_model and not metrics.thermal_throttling:
            recommendations.append({
                "type": "oled_optimization",
                "priority": "info",
                "message": "OLED model detected with good thermals",
                "action": "Can use aggressive optimization profiles",
                "oled_benefit": "Enhanced cooling enables higher performance"
            })
        
        return recommendations
    
    def cleanup(self):
        """Clean up GPU optimizer resources"""
        self.stop_monitoring()
        
        # Reset environment variables
        env_vars_to_reset = [
            "RADV_FORCE_WAVE32",
            "RADV_HEAP_BUDGET_TOLERANCE", 
            "RADV_MAX_ALLOC_SIZE",
            "RADV_ENABLE_MRT_OUTPUT_RELOCATIONS"
        ]
        
        for env_var in env_vars_to_reset:
            os.environ.pop(env_var, None)
        
        logger.info("RDNA 2 GPU optimizer cleanup completed")


# Global GPU optimizer instance
_rdna2_optimizer: Optional[RDNA2GPUOptimizer] = None


def get_rdna2_optimizer(oled_model: bool = True) -> RDNA2GPUOptimizer:
    """Get global RDNA 2 GPU optimizer"""
    global _rdna2_optimizer
    
    if _rdna2_optimizer is None:
        _rdna2_optimizer = RDNA2GPUOptimizer(oled_model=oled_model)
    
    return _rdna2_optimizer


if __name__ == "__main__":
    # Test RDNA 2 GPU optimizer
    logging.basicConfig(level=logging.INFO)
    
    print("🎮 RDNA 2 GPU Optimizer Test (OLED Steam Deck)")
    print("=" * 50)
    
    optimizer = get_rdna2_optimizer(oled_model=True)
    
    # Get GPU metrics
    print("GPU Metrics:")
    metrics = optimizer.get_gpu_metrics()
    print(f"  Utilization: {metrics.gpu_utilization_percent:.1f}%")
    print(f"  Clock Speed: {metrics.clock_speed_mhz} MHz")
    print(f"  Temperature: {metrics.temperature_celsius:.1f}°C")
    print(f"  Memory Usage: {metrics.memory_utilization_percent:.1f}%")
    print(f"  Thermal Throttling: {'Yes' if metrics.thermal_throttling else 'No'}")
    
    # Test optimization profiles
    print(f"\nApplying OLED performance profile...")
    settings = optimizer.optimize_for_shader_compilation("oled_performance")
    print(f"Applied settings: {settings}")
    
    # Gaming workload detection
    print(f"\nGaming workload analysis:")
    gaming_info = optimizer.monitor_gaming_workload()
    print(f"  Gaming detected: {gaming_info['gaming_detected']}")
    print(f"  Recommended threads: {gaming_info['recommended_threads']}")
    print(f"  Background only: {gaming_info['background_only']}")
    
    # Get recommendations
    print(f"\nOptimization recommendations:")
    recommendations = optimizer.get_optimization_recommendations()
    for rec in recommendations:
        priority_symbol = {"high": "🔴", "medium": "🟡", "info": "ℹ️"}.get(rec["priority"], "ℹ️")
        print(f"  {priority_symbol} {rec['message']}")
        print(f"    Action: {rec['action']}")
        if "oled_benefit" in rec:
            print(f"    OLED Benefit: {rec['oled_benefit']}")
    
    # Cleanup
    optimizer.cleanup()
    print(f"\n✅ RDNA 2 GPU optimizer test completed")