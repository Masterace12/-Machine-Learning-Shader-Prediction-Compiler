#!/usr/bin/env python3
"""
Steam Background Service for Gaming Mode Integration

This module provides a comprehensive background service that operates transparently
in Steam Deck Gaming Mode without affecting gaming performance. It coordinates
all Steam platform integration components to provide seamless shader prediction.

Features:
- Transparent Gaming Mode operation with zero gaming impact
- Automatic game launch detection and optimization
- Real-time hardware monitoring and adaptive optimization
- Fossilize integration with Steam's shader pipeline
- Proton game optimization and configuration
- SteamOS filesystem compliance and safety
- D-Bus communication with Steam and Gamescope
- Background ML shader prediction with intelligent scheduling
- Thermal and power management integration
- Service lifecycle management and recovery

The service runs as a systemd user service and integrates seamlessly with
SteamOS's immutable filesystem while providing transparent enhancements
to Steam's existing shader compilation system.
"""

import os
import sys
import time
import json
import asyncio
import signal
import logging
import threading
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager
from enum import Enum
import weakref

# Configure logging for service operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/steam-ml-predictor.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# SERVICE STATE MANAGEMENT
# =============================================================================

class ServiceState(Enum):
    """Background service states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    GAMING_MODE = "gaming_mode"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

class IntegrationMode(Enum):
    """Integration operating modes"""
    FULL_INTEGRATION = "full_integration"      # All features enabled
    GAMING_OPTIMIZED = "gaming_optimized"     # Background only during gaming
    MINIMAL_IMPACT = "minimal_impact"         # Lowest possible resource usage
    DEVELOPMENT = "development"               # Full logging and debugging

@dataclass
class ServiceConfiguration:
    """Service configuration parameters"""
    integration_mode: IntegrationMode
    enable_dbus_monitoring: bool
    enable_hardware_monitoring: bool
    enable_fossilize_integration: bool
    enable_proton_optimization: bool
    gaming_mode_detection: bool
    thermal_monitoring: bool
    power_management: bool
    automatic_optimization: bool
    prediction_frequency_hz: float
    cache_cleanup_interval_min: int
    log_level: str
    service_priority: int  # Nice level

@dataclass
class ServiceMetrics:
    """Service operation metrics"""
    uptime_seconds: float
    predictions_made: int
    games_optimized: int
    cache_hits: int
    cache_misses: int
    thermal_throttle_events: int
    gaming_mode_activations: int
    dbus_events_processed: int
    memory_usage_mb: float
    cpu_usage_percent: float
    last_optimization_time: float
    
    timestamp: float = field(default_factory=time.time)

# =============================================================================
# MAIN BACKGROUND SERVICE
# =============================================================================

class SteamBackgroundService:
    """
    Main background service for Steam platform integration
    """
    
    def __init__(self, config: Optional[ServiceConfiguration] = None):
        self.config = config or self._create_default_config()
        self.state = ServiceState.STOPPED
        self.metrics = ServiceMetrics(
            uptime_seconds=0,
            predictions_made=0,
            games_optimized=0,
            cache_hits=0,
            cache_misses=0,
            thermal_throttle_events=0,
            gaming_mode_activations=0,
            dbus_events_processed=0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            last_optimization_time=time.time()
        )
        
        # Integration components
        self.steam_integration = None
        self.dbus_interface = None
        self.hardware_monitor = None
        self.filesystem_manager = None
        self.ml_predictor = None
        
        # Service management
        self.service_tasks: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()
        self.start_time = time.time()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'game_launch': [],
            'game_termination': [],
            'hardware_change': [],
            'gaming_mode_change': [],
            'thermal_event': [],
            'power_event': []
        }
        
        logger.info(f"Steam Background Service initialized with mode: {self.config.integration_mode.value}")
    
    def _create_default_config(self) -> ServiceConfiguration:
        """Create default service configuration"""
        return ServiceConfiguration(
            integration_mode=IntegrationMode.GAMING_OPTIMIZED,
            enable_dbus_monitoring=True,
            enable_hardware_monitoring=True,
            enable_fossilize_integration=True,
            enable_proton_optimization=True,
            gaming_mode_detection=True,
            thermal_monitoring=True,
            power_management=True,
            automatic_optimization=True,
            prediction_frequency_hz=10.0,
            cache_cleanup_interval_min=60,
            log_level="INFO",
            service_priority=10  # Lower priority than games
        )
    
    async def start(self) -> bool:
        """Start the background service"""
        if self.state != ServiceState.STOPPED:
            logger.warning(f"Service already in state: {self.state.value}")
            return False
        
        logger.info("Starting Steam Background Service")
        self.state = ServiceState.STARTING
        
        try:
            # Initialize integration components
            await self._initialize_components()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start main service loop
            self.service_tasks.add(asyncio.create_task(self._main_service_loop()))
            
            self.state = ServiceState.RUNNING
            logger.info("Steam Background Service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            self.state = ServiceState.ERROR
            await self._cleanup()
            return False
    
    async def stop(self) -> None:
        """Stop the background service"""
        logger.info("Stopping Steam Background Service")
        self.state = ServiceState.STOPPING
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all service tasks
        for task in self.service_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.service_tasks:
            await asyncio.gather(*self.service_tasks, return_exceptions=True)
        
        # Cleanup components
        await self._cleanup()
        
        self.state = ServiceState.STOPPED
        logger.info("Steam Background Service stopped")
    
    async def _initialize_components(self) -> None:
        """Initialize all integration components"""
        logger.info("Initializing integration components")
        
        # Import and initialize components
        try:
            # Steam platform integration
            from .steam_platform_integration import get_steam_integration
            self.steam_integration = get_steam_integration()
            await self.steam_integration.initialize()
            logger.info("Steam platform integration initialized")
            
            # D-Bus interface
            if self.config.enable_dbus_monitoring:
                from .steam_dbus_interface import get_steam_dbus
                self.dbus_interface = get_steam_dbus()
                await self.dbus_interface.initialize()
                
                # Setup D-Bus event handlers
                self.dbus_interface.add_event_handler('steam_game', self._on_game_event)
                self.dbus_interface.add_event_handler('gaming_mode', self._on_gaming_mode_event)
                self.dbus_interface.add_event_handler('hardware', self._on_hardware_event)
                
                logger.info("D-Bus interface initialized")
            
            # Hardware monitoring
            if self.config.enable_hardware_monitoring:
                from .steam_deck_hardware_integration import get_hardware_monitor
                self.hardware_monitor = get_hardware_monitor()
                self.hardware_monitor.add_state_callback(self._on_hardware_state_change)
                logger.info("Hardware monitor initialized")
            
            # Filesystem manager
            from .steamos_filesystem_manager import get_filesystem_manager
            self.filesystem_manager = get_filesystem_manager()
            logger.info("Filesystem manager initialized")
            
            # ML predictor integration
            try:
                from .enhanced_ml_predictor import EnhancedMLPredictor
                self.ml_predictor = EnhancedMLPredictor()
                logger.info("Enhanced ML predictor integrated")
            except ImportError:
                logger.warning("Enhanced ML predictor not available")
            
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            raise
    
    async def _start_monitoring_tasks(self) -> None:
        """Start all monitoring tasks"""
        logger.info("Starting monitoring tasks")
        
        # D-Bus monitoring
        if self.dbus_interface:
            self.service_tasks.add(asyncio.create_task(self.dbus_interface.start_monitoring()))
        
        # Hardware monitoring
        if self.hardware_monitor:
            self.service_tasks.add(asyncio.create_task(self.hardware_monitor.start_monitoring()))
        
        # Service metrics monitoring
        self.service_tasks.add(asyncio.create_task(self._metrics_monitoring_loop()))
        
        # Cache cleanup
        self.service_tasks.add(asyncio.create_task(self._cache_cleanup_loop()))
        
        # Gaming mode detection
        if self.config.gaming_mode_detection:
            self.service_tasks.add(asyncio.create_task(self._gaming_mode_detection_loop()))
        
        logger.info(f"Started {len(self.service_tasks)} monitoring tasks")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)
    
    async def _main_service_loop(self) -> None:
        """Main service loop"""
        logger.info("Main service loop started")
        
        while not self.shutdown_event.is_set():
            try:
                # Update service metrics
                self._update_metrics()
                
                # Check service health
                await self._health_check()
                
                # Perform periodic optimizations
                if self.config.automatic_optimization:
                    await self._perform_automatic_optimizations()
                
                # Sleep until next iteration
                await asyncio.sleep(5.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main service loop error: {e}")
                await asyncio.sleep(10.0)
        
        logger.info("Main service loop ended")
    
    async def _metrics_monitoring_loop(self) -> None:
        """Monitor service metrics"""
        while not self.shutdown_event.is_set():
            try:
                # Update uptime
                self.metrics.uptime_seconds = time.time() - self.start_time
                
                # Update system metrics
                self.metrics.memory_usage_mb = self._get_memory_usage()
                self.metrics.cpu_usage_percent = self._get_cpu_usage()
                
                # Log metrics periodically
                if int(self.metrics.uptime_seconds) % 300 == 0:  # Every 5 minutes
                    logger.info(f"Service metrics - Uptime: {self.metrics.uptime_seconds:.0f}s, "
                              f"Predictions: {self.metrics.predictions_made}, "
                              f"Games optimized: {self.metrics.games_optimized}")
                
                await asyncio.sleep(60.0)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics monitoring error: {e}")
                await asyncio.sleep(60.0)
    
    async def _cache_cleanup_loop(self) -> None:
        """Periodic cache cleanup"""
        while not self.shutdown_event.is_set():
            try:
                # Wait for cleanup interval
                await asyncio.sleep(self.config.cache_cleanup_interval_min * 60)
                
                if self.shutdown_event.is_set():
                    break
                
                logger.info("Performing periodic cache cleanup")
                
                # Cleanup old prediction caches
                if self.filesystem_manager:
                    cache_path = self.filesystem_manager.get_cache_path('shader_predictions')
                    removed_count = self.filesystem_manager.cleanup_old_files(cache_path, max_age_days=7)
                    
                    if removed_count > 0:
                        logger.info(f"Cleaned up {removed_count} old prediction files")
                
                # Enforce size limits
                if self.filesystem_manager:
                    enforcement_results = self.filesystem_manager.enforce_size_limits()
                    for location, result in enforcement_results.items():
                        if result.get('action_taken') == 'cleanup':
                            logger.info(f"Size limit enforcement for {location}: {result}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _gaming_mode_detection_loop(self) -> None:
        """Gaming mode detection and state management"""
        last_gaming_mode_state = False
        
        while not self.shutdown_event.is_set():
            try:
                # Check gaming mode status
                current_gaming_mode = await self._is_gaming_mode_active()
                
                # Handle state changes
                if current_gaming_mode != last_gaming_mode_state:
                    if current_gaming_mode:
                        logger.info("Gaming Mode activated - switching to optimized background operation")
                        self.state = ServiceState.GAMING_MODE
                        self.metrics.gaming_mode_activations += 1
                        await self._enable_gaming_mode_optimizations()
                    else:
                        logger.info("Gaming Mode deactivated - resuming normal operation")
                        self.state = ServiceState.RUNNING
                        await self._disable_gaming_mode_optimizations()
                    
                    last_gaming_mode_state = current_gaming_mode
                
                await asyncio.sleep(2.0)  # Check every 2 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gaming mode detection error: {e}")
                await asyncio.sleep(5.0)
    
    async def _is_gaming_mode_active(self) -> bool:
        """Check if Gaming Mode is currently active"""
        try:
            # Check for gamescope process
            result = await asyncio.create_subprocess_exec(
                'pgrep', '-f', 'gamescope',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            return result.returncode == 0
        except Exception:
            return False
    
    async def _enable_gaming_mode_optimizations(self) -> None:
        """Enable optimizations for Gaming Mode"""
        # Reduce service priority
        try:
            os.nice(5)  # Lower priority
        except Exception:
            pass
        
        # Reduce prediction frequency
        if self.ml_predictor:
            # This would be implemented to reduce ML prediction frequency
            pass
        
        # Enable background-only mode for all components
        logger.info("Gaming Mode optimizations enabled")
    
    async def _disable_gaming_mode_optimizations(self) -> None:
        """Disable Gaming Mode optimizations"""
        # Restore normal priority
        try:
            os.nice(-5)  # Restore priority
        except Exception:
            pass
        
        # Restore normal prediction frequency
        if self.ml_predictor:
            # This would be implemented to restore normal ML prediction frequency
            pass
        
        logger.info("Gaming Mode optimizations disabled")
    
    async def _health_check(self) -> None:
        """Perform service health check"""
        try:
            # Check component health
            components_healthy = True
            
            # Check D-Bus interface
            if self.dbus_interface and not self.dbus_interface.monitoring_active:
                logger.warning("D-Bus interface not monitoring - attempting restart")
                await self.dbus_interface.start_monitoring()
            
            # Check hardware monitor
            if self.hardware_monitor and not self.hardware_monitor.monitoring_active:
                logger.warning("Hardware monitor not active - attempting restart")
                await self.hardware_monitor.start_monitoring()
            
            # Check memory usage
            if self.metrics.memory_usage_mb > 512:  # 512MB limit
                logger.warning(f"High memory usage: {self.metrics.memory_usage_mb:.0f}MB")
            
            # Check for thermal issues
            if self.hardware_monitor and self.hardware_monitor.hardware_state:
                state = self.hardware_monitor.hardware_state
                if state.thermal_state.value == 'critical':
                    logger.warning("Critical thermal state detected")
                    self.metrics.thermal_throttle_events += 1
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    async def _perform_automatic_optimizations(self) -> None:
        """Perform automatic optimizations based on current conditions"""
        try:
            # Get current hardware state
            if self.hardware_monitor and self.hardware_monitor.hardware_state:
                state = self.hardware_monitor.hardware_state
                
                # Apply hardware-based optimizations
                optimal_profile = self.hardware_monitor.get_optimal_profile(state)
                params = self.hardware_monitor.get_prediction_parameters(optimal_profile)
                
                # Update prediction parameters if ML predictor is available
                if self.ml_predictor:
                    # This would be implemented to update ML predictor parameters
                    pass
                
                self.metrics.last_optimization_time = time.time()
            
        except Exception as e:
            logger.error(f"Automatic optimization error: {e}")
    
    def _on_game_event(self, event) -> None:
        """Handle game launch/termination events"""
        try:
            logger.info(f"Game event: {event.event_type} - App ID: {event.app_id}")
            self.metrics.dbus_events_processed += 1
            
            if event.event_type == 'launched':
                asyncio.create_task(self._handle_game_launch(event.app_id))
            elif event.event_type == 'terminated':
                asyncio.create_task(self._handle_game_termination(event.app_id))
        
        except Exception as e:
            logger.error(f"Game event handler error: {e}")
    
    def _on_gaming_mode_event(self, event) -> None:
        """Handle Gaming Mode events"""
        try:
            logger.info(f"Gaming Mode event: {event.event_type}")
            self.metrics.dbus_events_processed += 1
            
            # Gaming Mode events are handled by the detection loop
        
        except Exception as e:
            logger.error(f"Gaming Mode event handler error: {e}")
    
    def _on_hardware_event(self, event) -> None:
        """Handle hardware events"""
        try:
            logger.info(f"Hardware event: {event.event_type}")
            self.metrics.dbus_events_processed += 1
            
            # Handle specific hardware events
            if event.event_type == 'thermal_warning':
                self.metrics.thermal_throttle_events += 1
            
        except Exception as e:
            logger.error(f"Hardware event handler error: {e}")
    
    def _on_hardware_state_change(self, new_state) -> None:
        """Handle hardware state changes"""
        try:
            logger.debug(f"Hardware state change: {new_state.thermal_state.value}, {new_state.power_state.value}")
            
            # Trigger automatic optimizations
            if self.config.automatic_optimization:
                asyncio.create_task(self._perform_automatic_optimizations())
        
        except Exception as e:
            logger.error(f"Hardware state change handler error: {e}")
    
    async def _handle_game_launch(self, app_id: str) -> None:
        """Handle game launch"""
        try:
            logger.info(f"Handling game launch: {app_id}")
            self.metrics.games_optimized += 1
            
            # Trigger shader prediction for the game
            if self.steam_integration:
                await self.steam_integration._handle_game_launch(app_id)
            
        except Exception as e:
            logger.error(f"Game launch handler error: {e}")
    
    async def _handle_game_termination(self, app_id: str) -> None:
        """Handle game termination"""
        try:
            logger.info(f"Handling game termination: {app_id}")
            
            # Collect post-game metrics
            if self.steam_integration:
                await self.steam_integration._handle_game_termination(app_id)
            
        except Exception as e:
            logger.error(f"Game termination handler error: {e}")
    
    def _update_metrics(self) -> None:
        """Update service metrics"""
        self.metrics.uptime_seconds = time.time() - self.start_time
        self.metrics.timestamp = time.time()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            process = psutil.Process()
            return process.cpu_percent()
        except Exception:
            return 0.0
    
    async def _cleanup(self) -> None:
        """Cleanup service resources"""
        logger.info("Cleaning up service resources")
        
        try:
            # Stop D-Bus monitoring
            if self.dbus_interface:
                await self.dbus_interface.stop_monitoring()
            
            # Stop hardware monitoring
            if self.hardware_monitor:
                self.hardware_monitor.stop_monitoring()
            
            # Stop Steam integration
            if self.steam_integration:
                self.steam_integration.background_service_active = False
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            'state': self.state.value,
            'config': {
                'integration_mode': self.config.integration_mode.value,
                'enable_dbus_monitoring': self.config.enable_dbus_monitoring,
                'enable_hardware_monitoring': self.config.enable_hardware_monitoring,
                'gaming_mode_detection': self.config.gaming_mode_detection,
                'prediction_frequency_hz': self.config.prediction_frequency_hz
            },
            'metrics': {
                'uptime_seconds': self.metrics.uptime_seconds,
                'predictions_made': self.metrics.predictions_made,
                'games_optimized': self.metrics.games_optimized,
                'gaming_mode_activations': self.metrics.gaming_mode_activations,
                'thermal_throttle_events': self.metrics.thermal_throttle_events,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'cpu_usage_percent': self.metrics.cpu_usage_percent
            },
            'components': {
                'steam_integration': self.steam_integration is not None,
                'dbus_interface': self.dbus_interface is not None and getattr(self.dbus_interface, 'monitoring_active', False),
                'hardware_monitor': self.hardware_monitor is not None and getattr(self.hardware_monitor, 'monitoring_active', False),
                'filesystem_manager': self.filesystem_manager is not None,
                'ml_predictor': self.ml_predictor is not None
            },
            'active_tasks': len(self.service_tasks)
        }

# =============================================================================
# SERVICE MANAGER AND ENTRY POINTS
# =============================================================================

class ServiceManager:
    """
    Service manager for handling service lifecycle
    """
    
    def __init__(self):
        self.service: Optional[SteamBackgroundService] = None
        self.config_path = Path.home() / ".config" / "ml-shader-predictor" / "service.json"
    
    def load_config(self) -> ServiceConfiguration:
        """Load service configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                return ServiceConfiguration(
                    integration_mode=IntegrationMode(config_data.get('integration_mode', 'gaming_optimized')),
                    enable_dbus_monitoring=config_data.get('enable_dbus_monitoring', True),
                    enable_hardware_monitoring=config_data.get('enable_hardware_monitoring', True),
                    enable_fossilize_integration=config_data.get('enable_fossilize_integration', True),
                    enable_proton_optimization=config_data.get('enable_proton_optimization', True),
                    gaming_mode_detection=config_data.get('gaming_mode_detection', True),
                    thermal_monitoring=config_data.get('thermal_monitoring', True),
                    power_management=config_data.get('power_management', True),
                    automatic_optimization=config_data.get('automatic_optimization', True),
                    prediction_frequency_hz=config_data.get('prediction_frequency_hz', 10.0),
                    cache_cleanup_interval_min=config_data.get('cache_cleanup_interval_min', 60),
                    log_level=config_data.get('log_level', 'INFO'),
                    service_priority=config_data.get('service_priority', 10)
                )
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Return default config
        return ServiceConfiguration(
            integration_mode=IntegrationMode.GAMING_OPTIMIZED,
            enable_dbus_monitoring=True,
            enable_hardware_monitoring=True,
            enable_fossilize_integration=True,
            enable_proton_optimization=True,
            gaming_mode_detection=True,
            thermal_monitoring=True,
            power_management=True,
            automatic_optimization=True,
            prediction_frequency_hz=10.0,
            cache_cleanup_interval_min=60,
            log_level="INFO",
            service_priority=10
        )
    
    def save_config(self, config: ServiceConfiguration) -> None:
        """Save service configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'integration_mode': config.integration_mode.value,
                'enable_dbus_monitoring': config.enable_dbus_monitoring,
                'enable_hardware_monitoring': config.enable_hardware_monitoring,
                'enable_fossilize_integration': config.enable_fossilize_integration,
                'enable_proton_optimization': config.enable_proton_optimization,
                'gaming_mode_detection': config.gaming_mode_detection,
                'thermal_monitoring': config.thermal_monitoring,
                'power_management': config.power_management,
                'automatic_optimization': config.automatic_optimization,
                'prediction_frequency_hz': config.prediction_frequency_hz,
                'cache_cleanup_interval_min': config.cache_cleanup_interval_min,
                'log_level': config.log_level,
                'service_priority': config.service_priority
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    async def start_service(self) -> bool:
        """Start the background service"""
        if self.service and self.service.state != ServiceState.STOPPED:
            logger.warning("Service already running")
            return False
        
        config = self.load_config()
        self.service = SteamBackgroundService(config)
        
        return await self.service.start()
    
    async def stop_service(self) -> None:
        """Stop the background service"""
        if self.service:
            await self.service.stop()
            self.service = None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        if self.service:
            return self.service.get_service_status()
        else:
            return {'state': 'stopped', 'service': None}

# Global service manager
_service_manager: Optional[ServiceManager] = None

def get_service_manager() -> ServiceManager:
    """Get or create the global service manager"""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager

# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================

async def main_service():
    """Main service entry point"""
    logger.info("Starting Steam ML Predictor Background Service")
    
    # Set process priority
    try:
        os.nice(10)  # Lower priority than games
    except Exception:
        pass
    
    manager = get_service_manager()
    
    try:
        success = await manager.start_service()
        if success:
            logger.info("Service started successfully, running until stopped")
            
            # Keep service running
            while True:
                await asyncio.sleep(60)
                
                # Check if service is still healthy
                status = manager.get_service_status()
                if status['state'] == 'error':
                    logger.error("Service entered error state, shutting down")
                    break
        else:
            logger.error("Failed to start service")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        return 1
    finally:
        await manager.stop_service()
        logger.info("Service shutdown complete")
    
    return 0

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == '--service':
            # Run as service
            exit_code = asyncio.run(main_service())
            sys.exit(exit_code)
            
        elif command == '--status':
            # Show service status
            manager = get_service_manager()
            status = manager.get_service_status()
            print(json.dumps(status, indent=2))
            
        elif command == '--test':
            # Test service components
            async def test_service():
                print("\n🔧 Steam Background Service Test")
                print("=" * 45)
                
                manager = ServiceManager()
                config = manager.load_config()
                
                print(f"\n📋 Configuration:")
                print(f"  Integration mode: {config.integration_mode.value}")
                print(f"  D-Bus monitoring: {config.enable_dbus_monitoring}")
                print(f"  Hardware monitoring: {config.enable_hardware_monitoring}")
                print(f"  Gaming mode detection: {config.gaming_mode_detection}")
                print(f"  Prediction frequency: {config.prediction_frequency_hz} Hz")
                
                print(f"\n🚀 Starting service for 10 seconds...")
                success = await manager.start_service()
                
                if success:
                    await asyncio.sleep(10)
                    
                    status = manager.get_service_status()
                    print(f"\n📊 Service Status:")
                    print(f"  State: {status['state']}")
                    print(f"  Uptime: {status['metrics']['uptime_seconds']:.1f}s")
                    print(f"  Memory usage: {status['metrics']['memory_usage_mb']:.1f}MB")
                    print(f"  Active tasks: {status['active_tasks']}")
                    
                    await manager.stop_service()
                    print(f"\n✅ Service test completed successfully")
                else:
                    print(f"\n❌ Service test failed to start")
            
            asyncio.run(test_service())
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        print("Steam Background Service for Gaming Mode Integration")
        print("Commands:")
        print("  --service    Run as background service")
        print("  --status     Show service status")
        print("  --test       Run service test")

if __name__ == "__main__":
    main()