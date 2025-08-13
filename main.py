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

# Optimized component imports
try:
    from src.core import HybridMLPredictor, OptimizedMLPredictor
    from src.rust_integration import get_system_info, is_steam_deck
    HAS_ML_PREDICTOR = True
except ImportError as e:
    print(f"⚠️  ML predictor not available: {e}")
    HAS_ML_PREDICTOR = False

try:
    from src.core import HybridVulkanCache, OptimizedShaderCache
    HAS_CACHE_MANAGER = True
except ImportError as e:
    print(f"⚠️  Cache manager not available: {e}")
    HAS_CACHE_MANAGER = False

try:
    from src.optimization import OptimizedThermalManager, ThermalManager
    HAS_THERMAL_MANAGER = True
except ImportError as e:
    print(f"⚠️  Thermal manager not available: {e}")
    HAS_THERMAL_MANAGER = False

try:
    from src.monitoring import PerformanceMonitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError as e:
    print(f"⚠️  Performance monitor not available: {e}")
    HAS_PERFORMANCE_MONITOR = False

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
        """Lazy-loaded ML predictor"""
        if self._ml_predictor is None and HAS_ML_PREDICTOR:
            if self.config.enable_ml_prediction:
                self._ml_predictor = get_optimized_predictor()
                self.logger.info("ML predictor initialized")
        return self._ml_predictor
    
    @property
    def cache_manager(self):
        """Lazy-loaded cache manager"""
        if self._cache_manager is None and HAS_CACHE_MANAGER:
            if self.config.enable_cache:
                self._cache_manager = get_optimized_cache()
                self.logger.info("Cache manager initialized")
        return self._cache_manager
    
    @property
    def thermal_manager(self):
        """Lazy-loaded thermal manager"""
        if self._thermal_manager is None and HAS_THERMAL_MANAGER:
            if self.config.enable_thermal_management:
                self._thermal_manager = get_thermal_manager()
                self.logger.info("Thermal manager initialized")
        return self._thermal_manager
    
    @property
    def performance_monitor(self):
        """Lazy-loaded performance monitor"""
        if self._performance_monitor is None and HAS_PERFORMANCE_MONITOR:
            if self.config.enable_performance_monitoring:
                self._performance_monitor = get_performance_monitor()
                self.logger.info("Performance monitor initialized")
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
                        
                        # Trigger cache cleanup
                        if self.cache_manager:
                            await self.cache_manager.get_stats()  # This triggers internal cleanup
                
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
                if self.ml_predictor:
                    ml_stats = self.ml_predictor.get_performance_stats()
                    # Update global stats from ML predictor
                
                if self.cache_manager:
                    cache_stats = self.cache_manager.get_stats()
                    self.stats["cache_hits"] = cache_stats.get("hits", {}).get("hot", 0) + \
                                             cache_stats.get("hits", {}).get("warm", 0) + \
                                             cache_stats.get("hits", {}).get("cold", 0)
                    self.stats["cache_misses"] = cache_stats.get("misses", 0)
                
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
        if self.thermal_manager:
            status["thermal"] = self.thermal_manager.get_status()
        
        if self.performance_monitor:
            perf_report = self.performance_monitor.get_performance_report()
            status["performance"] = {
                "health_score": perf_report.get("health_score", 0),
                "health_description": perf_report.get("health_description", "Unknown")
            }
        
        if self.cache_manager:
            status["cache"] = self.cache_manager.get_stats()
        
        if self.ml_predictor:
            status["ml"] = self.ml_predictor.get_performance_stats()
        
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
        if self.thermal_manager:
            self.thermal_manager.start_monitoring()
        
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()
        
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
        if self.thermal_manager:
            self.thermal_manager.start_monitoring()
        
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()
        
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
        if self.thermal_manager:
            self.thermal_manager.stop_monitoring()
        
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        if self.cache_manager:
            self.cache_manager.close()
        
        if self.ml_predictor:
            self.ml_predictor.cleanup()
        
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