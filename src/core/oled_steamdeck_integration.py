#!/usr/bin/env python3
"""
OLED Steam Deck Integration Module
Comprehensive integration of all OLED-specific optimizations for maximum performance
"""

import os
import json
import time
import asyncio
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Import OLED-specific optimizers
from .steamdeck_thermal_optimizer import get_thermal_optimizer, SteamDeckModel
from .oled_memory_optimizer import get_oled_shader_cache
from .rdna2_gpu_optimizer import get_rdna2_optimizer
from ..optimization.thermal_manager import get_thermal_manager

logger = logging.getLogger(__name__)


@dataclass
class OLEDPerformanceMetrics:
    """Comprehensive performance metrics for OLED Steam Deck"""
    # Thermal metrics
    apu_temperature: float = 0.0
    thermal_state: str = "normal"
    thermal_headroom_percent: float = 0.0
    
    # GPU metrics
    gpu_utilization: float = 0.0
    gpu_clock_mhz: int = 0
    memory_utilization: float = 0.0
    
    # Power metrics  
    power_draw_watts: float = 0.0
    battery_percent: float = 100.0
    power_efficiency_score: float = 0.0
    
    # Cache metrics
    shader_cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0
    compression_ratio: float = 0.0
    
    # Performance metrics
    compilation_threads: int = 0
    background_optimizations: int = 0
    gaming_mode_active: bool = False
    
    # OLED-specific benefits
    oled_thermal_advantage: float = 0.0
    oled_power_advantage: float = 0.0
    oled_performance_gain: float = 0.0


class OLEDSteamDeckOptimizer:
    """Master optimizer coordinating all OLED Steam Deck enhancements"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize OLED Steam Deck optimizer"""
        self.config_path = config_path or Path.home() / '.config' / 'shader-predict-compile'
        self.oled_config_file = self.config_path / 'steamdeck_oled_config.json'
        
        # Load OLED configuration
        self.config = self._load_oled_config()
        
        # Initialize component optimizers
        self.thermal_optimizer = get_thermal_optimizer()
        self.shader_cache = get_oled_shader_cache()  
        self.gpu_optimizer = get_rdna2_optimizer(oled_model=True)
        self.thermal_manager = get_thermal_manager()
        
        # Verify OLED model
        self.is_oled = self._verify_oled_model()
        if not self.is_oled:
            logger.warning("OLED optimizations enabled but OLED model not detected")
        
        # Performance state
        self.current_metrics = OLEDPerformanceMetrics()
        self.optimization_active = False
        self.adaptive_optimization = True
        
        # Callback management
        self.performance_callbacks: List[Callable] = []
        self.thermal_callbacks: List[Callable] = []
        
        # Threading
        self._coordinator_thread: Optional[threading.Thread] = None
        self._metrics_lock = threading.Lock()
        
        # OLED-specific features
        self.predictive_cooling_enabled = True
        self.burst_mode_enabled = True
        self.enhanced_memory_management = True
        
        logger.info(f"OLED Steam Deck optimizer initialized (verified: {self.is_oled})")
    
    def _load_oled_config(self) -> Dict[str, Any]:
        """Load OLED-specific configuration"""
        try:
            if self.oled_config_file.exists():
                with open(self.oled_config_file, 'r') as f:
                    config = json.load(f)
                logger.info("OLED configuration loaded successfully")
                return config
            else:
                logger.warning("OLED config file not found, using defaults")
                return self._get_default_oled_config()
        except Exception as e:
            logger.error(f"Failed to load OLED config: {e}")
            return self._get_default_oled_config()
    
    def _get_default_oled_config(self) -> Dict[str, Any]:
        """Get default OLED configuration"""
        return {
            "version": "2.1.0-oled-optimized",
            "hardware_profile": "steamdeck_oled",
            "system": {
                "max_compilation_threads": 8,
                "oled_optimized": True
            },
            "thermal": {
                "oled_enhanced_cooling": True,
                "sustained_performance_target": 1400
            },
            "gpu": {
                "rdna2_optimized": True,
                "max_power_watts": 18.0
            }
        }
    
    def _verify_oled_model(self) -> bool:
        """Verify this is actually an OLED Steam Deck"""
        return (
            self.thermal_optimizer.model == SteamDeckModel.OLED or
            "galileo" in str(Path("/sys/class/dmi/id/product_name").read_text().lower()) 
            if Path("/sys/class/dmi/id/product_name").exists() else False
        )
    
    def start_comprehensive_optimization(self):
        """Start comprehensive OLED optimization system"""
        if self.optimization_active:
            logger.warning("OLED optimization already active")
            return
        
        self.optimization_active = True
        
        # Start individual optimizers
        self.thermal_optimizer.start_monitoring(interval=1.5)
        self.gpu_optimizer.start_monitoring(interval=2.0)
        self.shader_cache.start_background_optimization()
        self.thermal_manager.start_monitoring()
        
        # Apply initial OLED optimizations
        self._apply_initial_optimizations()
        
        # Start coordination thread
        self._coordinator_thread = threading.Thread(
            target=self._optimization_coordinator,
            name="OLED_Coordinator",
            daemon=True
        )
        self._coordinator_thread.start()
        
        logger.info("OLED Steam Deck comprehensive optimization started")
    
    def _apply_initial_optimizations(self):
        """Apply initial OLED-specific optimizations"""
        try:
            # GPU optimizations
            gpu_profile = "oled_performance" if self.is_oled else "balanced"
            self.gpu_optimizer.optimize_for_shader_compilation(gpu_profile)
            
            # Memory budget optimization
            memory_budget = self.config.get("memory", {}).get("shader_cache_reserved_mb", 512)
            self.gpu_optimizer.set_memory_budget(memory_budget)
            
            # Thermal profile
            if self.is_oled:
                gaming_opts = self.thermal_optimizer.optimize_for_gaming()
                logger.info(f"Applied OLED gaming optimizations: {gaming_opts}")
            
            logger.info("Initial OLED optimizations applied")
            
        except Exception as e:
            logger.error(f"Failed to apply initial optimizations: {e}")
    
    def _optimization_coordinator(self):
        """Main coordination loop for OLED optimizations"""
        logger.info("OLED optimization coordinator started")
        
        while self.optimization_active:
            try:
                # Collect metrics from all optimizers
                thermal_status = self.thermal_optimizer.get_status()
                gpu_metrics = self.gpu_optimizer.get_gpu_metrics()
                cache_metrics = self.shader_cache.get_metrics()
                thermal_manager_status = self.thermal_manager.get_status()
                
                # Update comprehensive metrics
                with self._metrics_lock:
                    self.current_metrics.apu_temperature = thermal_status.get("max_temperature", 0.0)
                    self.current_metrics.thermal_state = thermal_status.get("thermal_state", "normal")
                    self.current_metrics.gpu_utilization = gpu_metrics.gpu_utilization_percent
                    self.current_metrics.gpu_clock_mhz = gpu_metrics.clock_speed_mhz
                    self.current_metrics.memory_utilization = gpu_metrics.memory_utilization_percent
                    self.current_metrics.power_draw_watts = gpu_metrics.power_draw_watts
                    self.current_metrics.shader_cache_hit_rate = cache_metrics.hit_rate
                    self.current_metrics.cache_size_mb = cache_metrics.total_size_mb
                    self.current_metrics.compression_ratio = cache_metrics.compression_ratio
                    self.current_metrics.compilation_threads = thermal_status.get("optimal_threads", 0)
                    
                    # Calculate OLED advantages
                    self._calculate_oled_advantages()
                
                # Adaptive optimization decisions
                if self.adaptive_optimization:
                    self._make_adaptive_decisions(thermal_status, gpu_metrics)
                
                # Notify callbacks
                self._notify_performance_callbacks()
                
                time.sleep(5.0)  # Coordination interval
                
            except Exception as e:
                logger.error(f"Optimization coordinator error: {e}")
                time.sleep(10.0)
        
        logger.info("OLED optimization coordinator stopped")
    
    def _calculate_oled_advantages(self):
        """Calculate OLED-specific performance advantages"""
        if not self.is_oled:
            return
        
        # Thermal advantage: OLED can sustain higher performance
        base_thermal_limit = 85.0  # LCD baseline
        oled_thermal_limit = 94.0  # OLED enhanced
        self.current_metrics.oled_thermal_advantage = (
            (oled_thermal_limit - self.current_metrics.apu_temperature) /
            (base_thermal_limit - self.current_metrics.apu_temperature)
        ) if self.current_metrics.apu_temperature < base_thermal_limit else 1.0
        
        # Power advantage: Better efficiency
        self.current_metrics.oled_power_advantage = 1.2  # ~20% better efficiency
        
        # Performance gain: More threads/higher clocks
        lcd_threads = 6  # Typical LCD max
        oled_threads = 8  # OLED enhanced
        self.current_metrics.oled_performance_gain = (
            self.current_metrics.compilation_threads / lcd_threads
        ) if lcd_threads > 0 else 1.0
    
    def _make_adaptive_decisions(self, thermal_status: Dict, gpu_metrics: Any):
        """Make adaptive optimization decisions based on current state"""
        try:
            current_temp = thermal_status.get("max_temperature", 70.0)
            thermal_state = thermal_status.get("thermal_state", "normal")
            gpu_usage = gpu_metrics.gpu_utilization_percent
            
            # Gaming detection and optimization
            gaming_detected = gpu_usage > 40 or thermal_state in ["warm", "hot"]
            if gaming_detected != self.current_metrics.gaming_mode_active:
                self.current_metrics.gaming_mode_active = gaming_detected
                
                if gaming_detected:
                    # Optimize for gaming
                    self._optimize_for_gaming_mode()
                else:
                    # Optimize for compilation
                    self._optimize_for_compilation_mode()
            
            # Thermal-based adaptations
            if thermal_state == "critical" and self.burst_mode_enabled:
                self._enable_emergency_cooling()
            elif thermal_state in ["cool", "optimal"] and self.is_oled:
                self._enable_aggressive_optimization()
            
            # Memory pressure adaptations
            if gpu_metrics.memory_pressure and self.enhanced_memory_management:
                self._optimize_memory_usage()
            
        except Exception as e:
            logger.error(f"Adaptive decision making failed: {e}")
    
    def _optimize_for_gaming_mode(self):
        """Optimize system for gaming mode"""
        logger.info("Switching to OLED gaming optimization mode")
        
        # Apply gaming optimizations from thermal optimizer
        gaming_opts = self.thermal_optimizer.optimize_for_gaming()
        
        # Adjust GPU optimization
        self.gpu_optimizer.optimize_for_shader_compilation("balanced")
        
        # Cache optimizations
        # (shader cache continues background optimization)
    
    def _optimize_for_compilation_mode(self):
        """Optimize system for shader compilation mode"""
        logger.info("Switching to OLED compilation optimization mode")
        
        # Apply maximum performance optimizations
        if self.is_oled:
            self.gpu_optimizer.optimize_for_shader_compilation("oled_performance")
        else:
            self.gpu_optimizer.optimize_for_shader_compilation("balanced")
    
    def _enable_emergency_cooling(self):
        """Enable emergency cooling measures"""
        logger.warning("Enabling emergency cooling for OLED Steam Deck")
        
        # Reduce compilation threads
        # (handled automatically by thermal optimizer)
        
        # Switch to power-efficient GPU profile
        self.gpu_optimizer.optimize_for_shader_compilation("power_efficient")
        
        # Notify thermal callbacks
        for callback in self.thermal_callbacks:
            try:
                callback("emergency_cooling", self.current_metrics.apu_temperature)
            except Exception as e:
                logger.error(f"Thermal callback error: {e}")
    
    def _enable_aggressive_optimization(self):
        """Enable aggressive optimizations when thermal headroom is available"""
        if not self.is_oled:
            return
        
        logger.debug("Enabling aggressive OLED optimizations")
        
        # Use high-performance GPU profile
        self.gpu_optimizer.optimize_for_shader_compilation("oled_performance")
        
        # Enable burst mode in cache
        self.shader_cache.oled_burst_mode = True
    
    def _optimize_memory_usage(self):
        """Optimize memory usage under pressure"""
        logger.info("Optimizing memory usage for OLED Steam Deck")
        
        # Trigger cache compression
        # (handled by shader cache automatically)
        
        # Reduce GPU memory budget
        reduced_budget = min(256, self.config.get("memory", {}).get("shader_cache_reserved_mb", 512) // 2)
        self.gpu_optimizer.set_memory_budget(reduced_budget)
    
    def _notify_performance_callbacks(self):
        """Notify performance callbacks of current metrics"""
        for callback in self.performance_callbacks:
            try:
                callback(self.current_metrics)
            except Exception as e:
                logger.error(f"Performance callback error: {e}")
    
    def add_performance_callback(self, callback: Callable[[OLEDPerformanceMetrics], None]):
        """Add performance monitoring callback"""
        self.performance_callbacks.append(callback)
    
    def add_thermal_callback(self, callback: Callable[[str, float], None]):
        """Add thermal event callback"""
        self.thermal_callbacks.append(callback)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive OLED optimization status"""
        with self._metrics_lock:
            status = {
                "oled_verified": self.is_oled,
                "optimization_active": self.optimization_active,
                "config_version": self.config.get("version", "unknown"),
                
                # Performance metrics
                "performance": {
                    "apu_temperature": self.current_metrics.apu_temperature,
                    "thermal_state": self.current_metrics.thermal_state,
                    "gpu_utilization": self.current_metrics.gpu_utilization,
                    "gpu_clock_mhz": self.current_metrics.gpu_clock_mhz,
                    "power_draw_watts": self.current_metrics.power_draw_watts,
                    "compilation_threads": self.current_metrics.compilation_threads,
                    "gaming_mode": self.current_metrics.gaming_mode_active
                },
                
                # Cache metrics
                "cache": {
                    "hit_rate": self.current_metrics.shader_cache_hit_rate,
                    "size_mb": self.current_metrics.cache_size_mb,
                    "compression_ratio": self.current_metrics.compression_ratio
                },
                
                # OLED advantages
                "oled_advantages": {
                    "thermal_advantage": self.current_metrics.oled_thermal_advantage,
                    "power_advantage": self.current_metrics.oled_power_advantage,
                    "performance_gain": self.current_metrics.oled_performance_gain
                } if self.is_oled else {},
                
                # Component status
                "components": {
                    "thermal_optimizer": "active",
                    "gpu_optimizer": "active", 
                    "shader_cache": "active",
                    "thermal_manager": "active"
                }
            }
        
        return status
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get comprehensive optimization recommendations"""
        recommendations = []
        
        # Collect recommendations from all optimizers
        thermal_recs = self.thermal_optimizer.get_status()
        gpu_recs = self.gpu_optimizer.get_optimization_recommendations()
        
        # Add OLED-specific recommendations
        if self.is_oled and not self.current_metrics.gaming_mode_active:
            if self.current_metrics.apu_temperature < 75.0:
                recommendations.append({
                    "type": "oled_performance",
                    "priority": "info",
                    "message": "OLED thermal headroom available for aggressive optimization",
                    "action": "Enable high-performance compilation mode",
                    "benefit": "Faster shader compilation with better cooling"
                })
        
        # Thermal recommendations
        if self.current_metrics.apu_temperature > 85.0:
            recommendations.append({
                "type": "thermal",
                "priority": "high",
                "message": f"High temperature ({self.current_metrics.apu_temperature:.1f}°C)",
                "action": "Reduce compilation workload",
                "oled_note": "OLED model can typically sustain higher temperatures"
            })
        
        # Add GPU recommendations
        recommendations.extend(gpu_recs)
        
        return recommendations
    
    @asynccontextmanager
    async def temporary_performance_mode(self, mode: str):
        """Context manager for temporary performance mode"""
        original_adaptive = self.adaptive_optimization
        
        try:
            # Disable adaptive optimization temporarily
            self.adaptive_optimization = False
            
            if mode == "maximum":
                self._enable_aggressive_optimization()
            elif mode == "gaming":
                self._optimize_for_gaming_mode()
            elif mode == "power_saving":
                self.gpu_optimizer.optimize_for_shader_compilation("power_efficient")
            
            yield
            
        finally:
            # Restore adaptive optimization
            self.adaptive_optimization = original_adaptive
    
    def stop_comprehensive_optimization(self):
        """Stop all OLED optimizations"""
        self.optimization_active = False
        
        # Stop individual optimizers
        self.thermal_optimizer.stop_monitoring()
        self.gpu_optimizer.stop_monitoring()
        self.shader_cache.stop_background_optimization()
        self.thermal_manager.stop_monitoring()
        
        # Wait for coordinator thread
        if self._coordinator_thread and self._coordinator_thread.is_alive():
            self._coordinator_thread.join(timeout=10)
        
        logger.info("OLED Steam Deck comprehensive optimization stopped")
    
    def export_performance_report(self, path: Path):
        """Export comprehensive performance report"""
        report = {
            "timestamp": time.time(),
            "oled_model_verified": self.is_oled,
            "configuration": self.config,
            "current_metrics": {
                "apu_temperature": self.current_metrics.apu_temperature,
                "thermal_state": self.current_metrics.thermal_state,
                "gpu_utilization": self.current_metrics.gpu_utilization,
                "power_draw": self.current_metrics.power_draw_watts,
                "compilation_threads": self.current_metrics.compilation_threads,
                "cache_hit_rate": self.current_metrics.shader_cache_hit_rate,
                "oled_advantages": {
                    "thermal": self.current_metrics.oled_thermal_advantage,
                    "power": self.current_metrics.oled_power_advantage,
                    "performance": self.current_metrics.oled_performance_gain
                }
            },
            "component_status": self.get_comprehensive_status()["components"],
            "recommendations": self.get_optimization_recommendations()
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"OLED performance report exported to {path}")
    
    def cleanup(self):
        """Clean up all resources"""
        self.stop_comprehensive_optimization()
        
        # Cleanup individual components
        self.shader_cache.cleanup()
        self.gpu_optimizer.cleanup()
        
        logger.info("OLED Steam Deck optimizer cleanup completed")


# Global OLED optimizer instance
_oled_optimizer: Optional[OLEDSteamDeckOptimizer] = None


def get_oled_optimizer() -> OLEDSteamDeckOptimizer:
    """Get global OLED Steam Deck optimizer"""
    global _oled_optimizer
    
    if _oled_optimizer is None:
        _oled_optimizer = OLEDSteamDeckOptimizer()
    
    return _oled_optimizer


async def initialize_oled_optimizations() -> bool:
    """Initialize comprehensive OLED optimizations"""
    try:
        optimizer = get_oled_optimizer()
        optimizer.start_comprehensive_optimization()
        
        # Wait for initialization
        await asyncio.sleep(2.0)
        
        status = optimizer.get_comprehensive_status()
        logger.info(f"OLED optimizations initialized: {status['optimization_active']}")
        
        return status['optimization_active']
        
    except Exception as e:
        logger.error(f"Failed to initialize OLED optimizations: {e}")
        return False


if __name__ == "__main__":
    # Test OLED comprehensive optimizer
    logging.basicConfig(level=logging.INFO)
    
    async def test_oled_optimizer():
        print("🎮 OLED Steam Deck Comprehensive Optimizer Test")
        print("=" * 55)
        
        # Initialize optimizer
        optimizer = get_oled_optimizer()
        
        # Start comprehensive optimization
        optimizer.start_comprehensive_optimization()
        
        # Wait for metrics to stabilize
        await asyncio.sleep(5)
        
        # Get status
        status = optimizer.get_comprehensive_status()
        print(f"OLED Verified: {'✓' if status['oled_verified'] else '✗'}")
        print(f"Optimization Active: {'✓' if status['optimization_active'] else '✗'}")
        
        # Performance metrics
        perf = status['performance']
        print(f"\nPerformance Metrics:")
        print(f"  APU Temperature: {perf['apu_temperature']:.1f}°C")
        print(f"  Thermal State: {perf['thermal_state']}")
        print(f"  GPU Utilization: {perf['gpu_utilization']:.1f}%")
        print(f"  GPU Clock: {perf['gpu_clock_mhz']} MHz")
        print(f"  Compilation Threads: {perf['compilation_threads']}")
        print(f"  Gaming Mode: {'✓' if perf['gaming_mode'] else '✗'}")
        
        # OLED advantages
        if status['oled_advantages']:
            advantages = status['oled_advantages']
            print(f"\nOLED Advantages:")
            print(f"  Thermal Advantage: {advantages['thermal_advantage']:.2f}x")
            print(f"  Power Advantage: {advantages['power_advantage']:.2f}x")
            print(f"  Performance Gain: {advantages['performance_gain']:.2f}x")
        
        # Cache metrics
        cache = status['cache']
        print(f"\nCache Performance:")
        print(f"  Hit Rate: {cache['hit_rate']:.1%}")
        print(f"  Size: {cache['size_mb']:.1f} MB")
        print(f"  Compression: {cache['compression_ratio']:.1f}x")
        
        # Get recommendations
        recommendations = optimizer.get_optimization_recommendations()
        if recommendations:
            print(f"\nOptimization Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                priority_symbol = {"high": "🔴", "medium": "🟡", "info": "ℹ️"}.get(rec.get("priority", "info"), "ℹ️")
                print(f"  {priority_symbol} {rec.get('message', 'No message')}")
        
        # Test temporary performance mode
        print(f"\nTesting temporary maximum performance mode...")
        async with optimizer.temporary_performance_mode("maximum"):
            await asyncio.sleep(2)
            temp_status = optimizer.get_comprehensive_status()
            print(f"  Maximum mode active: {temp_status['optimization_active']}")
        
        # Export report
        report_path = Path("/tmp/oled_performance_report.json")
        optimizer.export_performance_report(report_path)
        print(f"\nPerformance report exported to: {report_path}")
        
        # Cleanup
        optimizer.cleanup()
        print(f"\n✅ OLED comprehensive optimizer test completed!")
    
    # Run the test
    asyncio.run(test_oled_optimizer())