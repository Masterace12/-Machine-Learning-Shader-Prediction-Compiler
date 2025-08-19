#!/usr/bin/env python3
"""
Optimized Steam Deck ML Shader Prediction Compiler
Master integration script with all performance optimizations
"""

import sys
import os
import time
import asyncio
import logging
import signal
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# High-Performance ML-Only Imports
try:
    from src.core.ml_only_predictor import get_ml_predictor, HighPerformanceMLPredictor
    from src.core import get_optimized_cache_safe, HAS_SHADER_CACHE as CORE_HAS_CACHE
    from src.rust_integration import get_system_info, is_steam_deck
    HAS_ML_PREDICTOR = True  # ML is now required, not optional
    print("✅ High-Performance ML predictor loaded")
except ImportError as e:
    print(f"❌ CRITICAL: ML libraries required for operation: {e}")
    print("   Install with: pip install numpy scikit-learn lightgbm")
    print("   Or run: ./install.sh")
    sys.exit(1)  # Fail fast - ML is mandatory

try:
    from src.core import HybridVulkanCache, OptimizedShaderCache
    HAS_CACHE_MANAGER = CORE_HAS_CACHE if 'CORE_HAS_CACHE' in locals() else True
except ImportError as e:
    print(f"⚠️  Cache manager not available: {e}")
    HAS_CACHE_MANAGER = False
    if 'get_optimized_cache_safe' not in locals():
        get_optimized_cache_safe = lambda: None

try:
    from src.optimization import (
        OptimizedThermalManager, ThermalManager, get_thermal_manager,
        HAS_OPTIMIZED_THERMAL, HAS_THERMAL_MANAGER as OPT_HAS_THERMAL
    )
    HAS_THERMAL_MANAGER = HAS_OPTIMIZED_THERMAL or OPT_HAS_THERMAL
except ImportError as e:
    print(f"⚠️  Thermal manager not available: {e}")
    HAS_THERMAL_MANAGER = False
    get_thermal_manager = lambda: None

try:
    from src.monitoring import (
        PerformanceMonitor, get_performance_monitor,
        HAS_PERFORMANCE_MONITOR as MON_HAS_PERF_MON
    )
    HAS_PERFORMANCE_MONITOR = MON_HAS_PERF_MON
except ImportError as e:
    print(f"⚠️  Performance monitor not available: {e}")
    HAS_PERFORMANCE_MONITOR = False
    get_performance_monitor = lambda: None

# Fallback imports for system info
try:
    from src.rust_integration import get_system_info, is_steam_deck
except ImportError:
    def get_system_info():
        import platform
        import os
        return {
            "is_steam_deck": os.path.exists("/home/deck"),
            "cpu_count": os.cpu_count() or 4,
            "platform": platform.system(),
            "backend": "python_fallback"
        }
    
    def is_steam_deck():
        import os
        return os.path.exists("/home/deck")

HAS_OPTIMIZED_COMPONENTS = any([HAS_ML_PREDICTOR, HAS_CACHE_MANAGER, HAS_THERMAL_MANAGER, HAS_PERFORMANCE_MONITOR])


@dataclass
class SystemConfig:
    """System configuration"""
    enable_ml_prediction: bool = True
    enable_cache: bool = True
    enable_thermal_management: bool = True
    enable_performance_monitoring: bool = True
    enable_async: bool = True
    
    # Performance limits
    max_memory_mb: int = 200
    max_compilation_threads: int = 4
    
    # Monitoring intervals
    thermal_check_interval: float = 1.0
    performance_check_interval: float = 2.0
    cache_cleanup_interval: float = 300.0  # 5 minutes
    
    # Steam Deck optimizations
    steam_deck_optimized: bool = True
    detect_gaming_mode: bool = True
    respect_battery_level: bool = True


class OptimizedShaderSystem:
    """Integrated optimized shader prediction system"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize optimized system"""
        self.config = config or SystemConfig()
        
        # Components (initialized lazily)
        self._ml_predictor = None
        self._cache_manager = None
        self._thermal_manager = None
        self._performance_monitor = None
        
        # State
        self.running = False
        self.shutdown_requested = False
        
        # Statistics
        self.stats = {
            "predictions_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "thermal_throttle_events": 0,
            "uptime_seconds": 0,
            "start_time": 0
        }
        
        # Logger
        self.logger = self._setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Optimized shader system initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    Path.home() / '.cache' / 'shader-predict-compile' / 'system.log'
                )
            ]
        )
        
        # Create cache directory
        (Path.home() / '.cache' / 'shader-predict-compile').mkdir(parents=True, exist_ok=True)
        
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True
    
    @property
    def ml_predictor(self):
        """High-Performance ML-only predictor"""
        if self._ml_predictor is None:
            if self.config.enable_ml_prediction:
                try:
                    # Initialize high-performance ML predictor with maximum optimization
                    self._ml_predictor = get_ml_predictor(force_reload=False)
                    self.logger.info("High-Performance ML predictor initialized")
                    
                    # Log performance capabilities
                    metrics = self._ml_predictor.get_performance_metrics()
                    self.logger.info(f"ML Model: {metrics['primary_model']}, Throughput: {metrics['predictions_per_second']:.0f} pred/sec")
                    
                except Exception as e:
                    self.logger.error(f"CRITICAL: Failed to initialize ML predictor: {e}")
                    raise RuntimeError("ML predictor is required for operation")
        return self._ml_predictor
    
    @property
    def cache_manager(self):
        """Lazy-loaded cache manager"""
        if self._cache_manager is None and HAS_CACHE_MANAGER:
            if self.config.enable_cache:
                try:
                    self._cache_manager = get_optimized_cache_safe()
                    if self._cache_manager is not None:
                        self.logger.info("Cache manager initialized")
                    else:
                        self.logger.warning("Cache manager could not be initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize cache manager: {e}")
        return self._cache_manager
    
    @property
    def thermal_manager(self):
        """Lazy-loaded thermal manager"""
        if self._thermal_manager is None and HAS_THERMAL_MANAGER:
            if self.config.enable_thermal_management:
                try:
                    self._thermal_manager = get_thermal_manager()
                    if self._thermal_manager is not None:
                        self.logger.info("Thermal manager initialized")
                    else:
                        self.logger.warning("Thermal manager could not be initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize thermal manager: {e}")
        return self._thermal_manager
    
    @property
    def performance_monitor(self):
        """Lazy-loaded performance monitor"""
        if self._performance_monitor is None and HAS_PERFORMANCE_MONITOR:
            if self.config.enable_performance_monitoring:
                try:
                    self._performance_monitor = get_performance_monitor()
                    if self._performance_monitor is not None:
                        self.logger.info("Performance monitor initialized")
                    else:
                        self.logger.warning("Performance monitor could not be initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize performance monitor: {e}")
        return self._performance_monitor
    
    def _detect_steam_deck(self) -> bool:
        """Detect if running on Steam Deck"""
        try:
            # Check DMI product name
            dmi_path = Path("/sys/class/dmi/id/product_name")
            if dmi_path.exists():
                product_name = dmi_path.read_text().strip().lower()
                return "jupiter" in product_name or "steamdeck" in product_name
        except Exception:
            pass
        
        # Check for Steam Deck specific directories
        steamdeck_indicators = [
            Path("/dev/hwmon0"),  # Hardware monitoring
            Path("/sys/class/power_supply/BAT0"),  # Battery
        ]
        
        return any(path.exists() for path in steamdeck_indicators)
    
    def _optimize_for_steam_deck(self):
        """Apply Steam Deck specific optimizations"""
        if not self.config.steam_deck_optimized:
            return
        
        is_steam_deck = self._detect_steam_deck()
        
        if is_steam_deck:
            self.logger.info("Steam Deck detected - applying optimizations")
            
            # Reduce memory limits for Steam Deck
            self.config.max_memory_mb = min(self.config.max_memory_mb, 150)
            
            # Adjust thread counts for Steam Deck's 4-core CPU
            self.config.max_compilation_threads = min(self.config.max_compilation_threads, 4)
            
            # Enable battery awareness
            self.config.respect_battery_level = True
            
            # Faster thermal checks for better responsiveness
            self.config.thermal_check_interval = 0.5
            
        else:
            self.logger.info("Generic Linux system detected")
    
    async def _monitor_system_health(self):
        """Monitor system health and adjust performance"""
        while self.running and not self.shutdown_requested:
            try:
                # Get system status
                thermal_status = None
                if self.thermal_manager:
                    thermal_status = self.thermal_manager.get_status()
                
                performance_report = None
                if self.performance_monitor:
                    performance_report = self.performance_monitor.get_performance_report()
                
                # Adaptive performance adjustment
                if thermal_status and thermal_status.get("thermal_state") in ["hot", "throttling", "critical"]:
                    self.stats["thermal_throttle_events"] += 1
                    
                    # Reduce compilation threads under thermal stress
                    if thermal_status.get("compilation_threads", 0) > 1:
                        self.logger.warning("Thermal throttling detected - reducing compilation threads")
                
                # Memory pressure handling
                if performance_report:
                    memory_usage = performance_report.get("recent_metrics", {}).get("memory_usage_percent", 0)
                    if memory_usage > 85:
                        self.logger.warning(f"High memory usage: {memory_usage:.1f}%")
                        
                        # Trigger cache cleanup if available
                        if self.cache_manager and hasattr(self.cache_manager, 'get_stats'):
                            try:
                                await self.cache_manager.get_stats()  # This triggers internal cleanup
                            except Exception as e:
                                self.logger.warning(f"Cache cleanup failed: {e}")
                
                await asyncio.sleep(self.config.performance_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.config.performance_check_interval * 2)
    
    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        while self.running and not self.shutdown_requested:
            try:
                # Update statistics
                self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
                
                # Collect performance metrics
                if self.ml_predictor and hasattr(self.ml_predictor, 'get_performance_stats'):
                    try:
                        ml_stats = self.ml_predictor.get_performance_stats()
                        # Update global stats from ML predictor
                    except Exception as e:
                        self.logger.warning(f"Failed to get ML stats: {e}")
                
                if self.cache_manager and hasattr(self.cache_manager, 'get_stats'):
                    try:
                        cache_stats = self.cache_manager.get_stats()
                        self.stats["cache_hits"] = cache_stats.get("hits", {}).get("hot", 0) + \
                                                 cache_stats.get("hits", {}).get("warm", 0) + \
                                                 cache_stats.get("hits", {}).get("cold", 0)
                        self.stats["cache_misses"] = cache_stats.get("misses", 0)
                    except Exception as e:
                        self.logger.warning(f"Failed to get cache stats: {e}")
                
                # Log periodic status
                if self.stats["uptime_seconds"] % 300 < self.config.cache_cleanup_interval:  # Every 5 minutes
                    self.logger.info(f"System status: {self.get_system_status()}")
                
                await asyncio.sleep(self.config.cache_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Maintenance error: {e}")
                await asyncio.sleep(self.config.cache_cleanup_interval)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "running": self.running,
            "uptime_seconds": self.stats["uptime_seconds"],
            "statistics": self.stats.copy(),
            "components": {
                "ml_predictor": self._ml_predictor is not None,
                "cache_manager": self._cache_manager is not None,
                "thermal_manager": self._thermal_manager is not None,
                "performance_monitor": self._performance_monitor is not None
            },
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "max_compilation_threads": self.config.max_compilation_threads,
                "steam_deck_optimized": self.config.steam_deck_optimized
            }
        }
        
        # Add component-specific status
        if self.thermal_manager and hasattr(self.thermal_manager, 'get_status'):
            try:
                status["thermal"] = self.thermal_manager.get_status()
            except Exception as e:
                status["thermal"] = {"error": str(e)}
        
        if self.performance_monitor and hasattr(self.performance_monitor, 'get_performance_report'):
            try:
                perf_report = self.performance_monitor.get_performance_report()
                status["performance"] = {
                    "health_score": perf_report.get("health_score", 0),
                    "health_description": perf_report.get("health_description", "Unknown")
                }
            except Exception as e:
                status["performance"] = {"error": str(e)}
        
        if self.cache_manager and hasattr(self.cache_manager, 'get_stats'):
            try:
                status["cache"] = self.cache_manager.get_stats()
            except Exception as e:
                status["cache"] = {"error": str(e)}
        
        if self.ml_predictor and hasattr(self.ml_predictor, 'get_performance_metrics'):
            try:
                ml_metrics = self.ml_predictor.get_performance_metrics()
                status["ml"] = {
                    "model_type": "High-Performance ML",
                    "primary_model": ml_metrics.get("primary_model", "Unknown"),
                    "predictions_per_second": ml_metrics.get("predictions_per_second", 0),
                    "average_prediction_time_ms": ml_metrics.get("average_prediction_time_ms", 0),
                    "prediction_count": ml_metrics.get("prediction_count", 0),
                    "optimization_level": ml_metrics.get("optimization_level", 0),
                    "performance_features_active": sum(ml_metrics.get("performance_features", {}).values()),
                    "backend": "ML-Only (No Fallbacks)"
                }
            except Exception as e:
                status["ml"] = {"error": str(e)}
        
        return status
    
    async def start_async(self):
        """Start the system in async mode"""
        if self.running:
            return
        
        self.running = True
        self.stats["start_time"] = time.time()
        
        self.logger.info("🚀 Starting optimized shader system (async mode)")
        
        # Apply Steam Deck optimizations
        self._optimize_for_steam_deck()
        
        # Start components
        if self.thermal_manager and hasattr(self.thermal_manager, 'start_monitoring'):
            try:
                self.thermal_manager.start_monitoring()
            except Exception as e:
                self.logger.warning(f"Failed to start thermal monitoring: {e}")
        
        if self.performance_monitor and hasattr(self.performance_monitor, 'start_monitoring'):
            try:
                self.performance_monitor.start_monitoring()
            except Exception as e:
                self.logger.warning(f"Failed to start performance monitoring: {e}")
        
        # Start background tasks
        tasks = []
        
        if self.config.enable_performance_monitoring:
            tasks.append(asyncio.create_task(self._monitor_system_health()))
        
        tasks.append(asyncio.create_task(self._periodic_maintenance()))
        
        try:
            # Wait for shutdown signal
            while not self.shutdown_requested:
                await asyncio.sleep(1)
        
        except Exception as e:
            self.logger.error(f"System error: {e}")
        
        finally:
            # Cancel background tasks
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            await self.shutdown_async()
    
    def start(self):
        """Start the system (sync interface)"""
        if self.config.enable_async:
            try:
                asyncio.run(self.start_async())
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested via keyboard interrupt")
        else:
            self.start_sync()
    
    def start_sync(self):
        """Start the system in sync mode"""
        if self.running:
            return
        
        self.running = True
        self.stats["start_time"] = time.time()
        
        self.logger.info("🚀 Starting optimized shader system (sync mode)")
        
        # Apply Steam Deck optimizations
        self._optimize_for_steam_deck()
        
        # Start components
        if self.thermal_manager and hasattr(self.thermal_manager, 'start_monitoring'):
            try:
                self.thermal_manager.start_monitoring()
            except Exception as e:
                self.logger.warning(f"Failed to start thermal monitoring: {e}")
        
        if self.performance_monitor and hasattr(self.performance_monitor, 'start_monitoring'):
            try:
                self.performance_monitor.start_monitoring()
            except Exception as e:
                self.logger.warning(f"Failed to start performance monitoring: {e}")
        
        try:
            # Main loop
            while not self.shutdown_requested:
                time.sleep(1)
                
                # Update uptime
                self.stats["uptime_seconds"] = time.time() - self.stats["start_time"]
                
                # Periodic logging
                if int(self.stats["uptime_seconds"]) % 300 == 0:  # Every 5 minutes
                    self.logger.info(f"System status: uptime {self.stats['uptime_seconds']:.0f}s")
        
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested via keyboard interrupt")
        
        finally:
            self.shutdown_sync()
    
    async def shutdown_async(self):
        """Shutdown system (async)"""
        if not self.running:
            return
        
        self.logger.info("🛑 Shutting down optimized shader system...")
        self.running = False
        
        # Stop components
        if self.thermal_manager and hasattr(self.thermal_manager, 'stop_monitoring'):
            try:
                self.thermal_manager.stop_monitoring()
            except Exception as e:
                self.logger.warning(f"Failed to stop thermal monitoring: {e}")
        
        if self.performance_monitor and hasattr(self.performance_monitor, 'stop_monitoring'):
            try:
                self.performance_monitor.stop_monitoring()
            except Exception as e:
                self.logger.warning(f"Failed to stop performance monitoring: {e}")
        
        if self.cache_manager and hasattr(self.cache_manager, 'close'):
            try:
                self.cache_manager.close()
            except Exception as e:
                self.logger.warning(f"Failed to close cache manager: {e}")
        
        if self.ml_predictor and hasattr(self.ml_predictor, 'cleanup'):
            try:
                self.ml_predictor.cleanup()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup ML predictor: {e}")
        
        # Log final statistics
        final_stats = self.get_system_status()
        self.logger.info(f"Final system statistics: {final_stats['statistics']}")
        
        self.logger.info("✓ System shutdown completed")
    
    def shutdown_sync(self):
        """Shutdown system (sync)"""
        asyncio.run(self.shutdown_async())


def main():
    """Main entry point"""
    print("🎮 Steam Deck ML Shader Prediction Compiler (Optimized)")
    print("="*60)
    
    if not HAS_OPTIMIZED_COMPONENTS:
        print("❌ Optimized components not available!")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements-optimized.txt")
        return 1
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Shader Prediction System")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--no-async", action="store_true", help="Disable async mode")
    parser.add_argument("--no-thermal", action="store_true", help="Disable thermal management")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable performance monitoring")
    parser.add_argument("--memory-limit", type=int, default=200, help="Memory limit in MB")
    parser.add_argument("--threads", type=int, default=4, help="Max compilation threads")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--test", action="store_true", help="Run system test")
    
    args = parser.parse_args()
    
    # Load configuration
    config = SystemConfig(
        enable_async=not args.no_async,
        enable_thermal_management=not args.no_thermal,
        enable_performance_monitoring=not args.no_monitoring,
        max_memory_mb=args.memory_limit,
        max_compilation_threads=args.threads
    )
    
    if args.config and args.config.exists():
        try:
            config_data = json.loads(args.config.read_text())
            # Apply config overrides
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except Exception as e:
            print(f"⚠️  Config file error: {e}")
    
    # Create system
    system = OptimizedShaderSystem(config)
    
    if args.status:
        # Show status
        status = system.get_system_status()
        print(json.dumps(status, indent=2))
        return 0
    
    if args.test:
        # Run system test
        print("🧪 Running system test...")
        
        # Quick initialization test
        try:
            if system.ml_predictor:
                print("✓ ML predictor initialized")
            if system.cache_manager:
                print("✓ Cache manager initialized")
            if system.thermal_manager:
                print("✓ Thermal manager initialized")
            if system.performance_monitor:
                print("✓ Performance monitor initialized")
            
            print("✅ System test passed")
            return 0
            
        except Exception as e:
            print(f"❌ System test failed: {e}")
            return 1
    
    # Start system
    try:
        system.start()
        return 0
    except Exception as e:
        print(f"❌ System startup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())